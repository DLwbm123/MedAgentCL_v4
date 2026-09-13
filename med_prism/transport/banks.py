"""Mapping: concatenate expert A rows, concatenate scaled expert B columns.

The repair budget applies to ONE complete historical task bank (rank 16),
not separately to each rank-1 expert. No new nn.Module is installed.
"""
from dataclasses import dataclass
import hashlib
import torch


@dataclass
class Bank:
    module_name: str
    task_id: int
    names: list[str]
    factors: list

    def matrices(self, device=None, dtype=torch.float32):
        A = torch.cat([f.A.detach().to(device=device, dtype=dtype) for f in self.factors], dim=0)
        B = torch.cat([f.B.detach().to(device=device, dtype=dtype) *
                       f.scaling.detach().to(device=device, dtype=dtype) for f in self.factors], dim=1)
        return A, B

    @torch.no_grad()
    def commit(self, A):
        offset = 0
        for factor in self.factors:
            rows = factor.A.shape[0]
            factor.A.copy_(A[offset:offset + rows].to(factor.A))
            offset += rows
        if offset != A.shape[0]:
            raise ValueError("Bank shape mismatch")


def banks_for_wrapper(name, wrapper):
    if wrapper.private_adapter_type != "rank1_expert_bank":
        raise ValueError("v2 pilot requires the verified no-geo rank-1 bank architecture")
    result = []
    for task in wrapper.task_ids:
        factors = wrapper.task_experts(task)
        if not factors:
            raise ValueError("Empty task bank")
        names = [f"{name}.experts.{wrapper.expert_key(task, f.expert_id)}.A" for f in factors]
        result.append(Bank(name, task, names, factors))
    return result


def tensor_hash(tensor):
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def fingerprint(model):
    """Hash every persistent tensor, including base weights, in bounded chunks.

No full base copy; CPU staging is at most the largest individual tensor.
"""
    return {name: {"shape": list(t.shape), "dtype": str(t.dtype), "numel": t.numel(),
                   "sha256": tensor_hash(t)} for name, t in model.state_dict().items()}


def check_invariants(before, after, allowed):
    if before.keys() != after.keys():
        raise RuntimeError("TPM changed tensor names/count")
    changed = []
    for name in before:
        if any(before[name][k] != after[name][k] for k in ("shape", "dtype", "numel")):
            raise RuntimeError(f"TPM changed schema: {name}")
        if before[name]["sha256"] != after[name]["sha256"]:
            if name not in allowed:
                raise RuntimeError(f"TPM invariant violated: non-approved tensor changed: {name}")
            changed.append(name)
    return {"status": "PASS", "changed_tensors": changed, "allowed_tensors": sorted(allowed),
            "persistent_tensor_count": len(before),
            "persistent_scalar_count": sum(v["numel"] for v in before.values()),
            "base_shared_current_historical_B_unchanged": True}


def geometry(banks, current_task):
    from med_prism.projection.orth_losses import rank1_key_cosines
    current = next(b for b in banks if b.task_id == current_task).matrices(device="cpu")[0]
    old = [b.matrices(device="cpu")[0] for b in banks if b.task_id < current_task]
    def stats(values):
        if not values:
            return {"pair_count": 0, "mean_abs": None, "rms": None, "max_abs": None}
        x = torch.cat([v.flatten() for v in values]).abs()
        return {"pair_count": x.numel(), "mean_abs": float(x.mean()),
                "rms": float(x.square().mean().sqrt()), "max_abs": float(x.max())}
    return {"history_vs_current": stats([rank1_key_cosines(current, x) for x in old]),
            "history_between_banks": stats([rank1_key_cosines(x, y) for i, x in enumerate(old) for y in old[i+1:]])}
