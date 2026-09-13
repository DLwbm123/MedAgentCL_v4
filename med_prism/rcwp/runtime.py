"""Training-only hooks and strictly factor-bound sidecars. No inference changes."""
import json
from pathlib import Path
import torch
from .core import bank_binding, factors, signed_score, net_penalty

SCHEMA = "MedPRISM_RCWP_summary_v1"


def wrappers(model):
    from med_prism.adapters.shared_private import SharedPrivateLinear
    return {name: m for name, m in model.named_modules() if isinstance(m, SharedPrivateLinear)}


def save_summary(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def validate_summary(value, modules):
    if value.get("schema") != SCHEMA or value.get("method_version") != "2.1":
        raise ValueError("Invalid RCWP schema/version")
    task = value["task_id"]
    if set(value["fields"]) != set(modules):
        raise ValueError("RCWP wrapper names mismatch")
    if not isinstance(task, int) or task < 1:
        raise ValueError("Invalid task id")
    if not isinstance(value["nu"], float) or not 1 <= value["nu"] < float("inf"):
        raise ValueError("Invalid nu")
    for name, m in modules.items():
        f = value["fields"][name]
        if f["binding"] != bank_binding(m, task):
            raise ValueError(f"RCWP A/B/scaling/dtype hash mismatch: {task}:{name}")
        A, _ = factors(m, task)
        r = len(A)
        shapes = {"M":(r,r+1), "mu":(r,), "scale":(r,)}
        for k, shape in shapes.items():
            t = f[k]
            if t.dtype != torch.float32 or tuple(t.shape) != shape or not torch.isfinite(t).all():
                raise ValueError(f"Invalid RCWP field {name}:{k}")
        if (f["scale"] < 1e-6).any():
            raise ValueError("Invalid coordinate scale")
    return value


def load_summary(path, modules):
    return validate_summary(torch.load(path, map_location="cpu", weights_only=True), modules)


class WriteProtection:
    """One original forward only. Hooks are removed before checkpoint recomputation.

    Auxiliary operations explicitly enable grad even during reentrant checkpoint's
    first no-grad forward. They depend solely on detached inputs and current A/B.
    Locked batch-one/nonpacking recipe, no changed model outputs or RNG calls.
    """
    def __init__(self, model, task, summaries):
        self.modules = wrappers(model)
        self.task = task
        self.summaries = [validate_summary(s,self.modules) for s in summaries]
        if [s["task_id"] for s in summaries] != list(range(1,task)):
            raise ValueError("Exactly one frozen RCWP summary per previous task is required")
        for m in self.modules.values():
            if list(m._active_tasks) != list(range(1,task+1)):
                raise ValueError("Cumulative active banks mismatch")
        self.cache = {}
        for name,m in self.modules.items():
            self.cache[name] = []
            for s in summaries:
                A,B = factors(m,s["task_id"])
                f = s["fields"][name]
                self.cache[name].append((A,B,*[f[k].to(A.device).detach() for k in ("M","mu","scale")]))
        self.handles = []

    def begin(self, mask):
        if self.handles:
            raise RuntimeError("Nested RCWP capture")
        if mask.ndim != 2 or not ((mask == 0)|(mask == 1)).all():
            raise ValueError("RCWP needs an explicit nonpacking binary token mask")
        self.mask, self.seen, self.terms = mask.bool(), set(), []
        if self.task == 1:
            return
        for name,m in self.modules.items():
            def capture(module, args, name=name):
                if name in self.seen:
                    raise RuntimeError("Duplicate wrapper in original RCWP forward")
                self.seen.add(name)
                h = args[0]
                if h.shape[:2] != self.mask.shape:
                    raise ValueError("RCWP token alignment mismatch")
                with torch.enable_grad(), torch.autocast(h.device.type, enabled=False):
                    A,B = factors(module,self.task,detach=False)
                    scores = [signed_score(h,A,B,*old).masked_fill(~self.mask,0).sum(-1)
                              for old in self.cache[name]]
                    self.terms.append(torch.stack(scores))
            self.handles.append(m.register_forward_pre_hook(capture))

    def cancel(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def finish(self):
        self.cancel()
        if self.task == 1:
            return torch.zeros((),device=self.mask.device), {"rcwp_loss":0.,
                "rcwp_signed_score_mean":0.,"rcwp_negative_fraction":0.,
                "rcwp_negative_score_mean":0.,"rcwp_per_history_task":{}}
        if self.seen != set(self.modules):
            raise RuntimeError("Missing RCWP wrappers")
        scores = torch.stack(self.terms).sum(0)
        self.terms = []
        nu = scores.new_tensor([s["nu"] for s in self.summaries])
        loss = net_penalty(scores,nu)
        detached = scores.detach()
        negative = detached[detached<0]
        return loss,{"rcwp_loss":float(loss.detach()),
            "rcwp_signed_score_mean":float(detached.mean()),
            "rcwp_negative_fraction":float((detached<0).float().mean()),
            "rcwp_negative_score_mean":float(negative.mean()) if negative.numel() else 0.,
            "rcwp_per_history_task":{str(s["task_id"]):float(detached[i].mean())
                                     for i,s in enumerate(self.summaries)}}
