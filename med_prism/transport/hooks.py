"""Audited Qwen3-VL native-block bridge (transformers 4.57.6).

Capture native first-layer kwargs and visual side inputs from a real full pass.
Reuse those kwargs in the native decoder block, followed by its parent model's
native DeepStack operation. There is no handwritten attention/RoPE implementation.
"""
from contextlib import contextmanager
import inspect
import hashlib
import torch
from .banks import banks_for_wrapper
from .calibration import select_tokens


def tree_to(value, device, clone=False):
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device)
        return value.clone() if clone else value
    if isinstance(value, dict):
        return {k: tree_to(v, device, clone) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree_to(v, device, clone) for v in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported native block metadata type: {type(value).__name__}")


@contextmanager
def cleanup_hooks():
    handles = []
    try:
        yield handles
    finally:
        for handle in reversed(handles):
            handle.remove()


class QwenBlockBridge:
    def __init__(self, model):
        from med_prism.adapters.injection import iter_shared_private_wrappers
        self.model = model
        self.backbone = model.model
        self.text = self.backbone.language_model
        if type(self.text).__name__ != "Qwen3VLTextModel":
            raise TypeError("Only audited dense Qwen3VLTextModel is supported; no silent architecture fallback")
        self.blocks = self.text.layers
        self.device = next(self.text.parameters()).device
        if len({p.device for p in model.parameters()}) != 1:
            raise ValueError("TPM bridge requires one-device base; CPU caches are managed explicitly")
        self.targets = {}
        self.block_targets = []
        for index, block in enumerate(self.blocks):
            entries = []
            for projection in ("q_proj", "v_proj"):
                wrapper = getattr(block.self_attn, projection)
                name = next((name for name, module in model.named_modules() if module is wrapper), None)
                if name is None or not hasattr(wrapper, "task_experts"):
                    raise ValueError(f"Missing shared/private {projection} at block {index}")
                self.targets[name] = wrapper
                entries.append(name)
            self.block_targets.append(entries)
        actual = {name for name, _ in iter_shared_private_wrappers(model)}
        if actual != set(self.targets):
            raise ValueError("Adapters outside q/v decoder blocks would invalidate shared upstream inputs")
        self.banks = {name: banks_for_wrapper(name, module) for name, module in self.targets.items()}
        self.audit = {"text_class": type(self.text).__name__, "block_count": len(self.blocks),
                      "projection_count": len(self.targets), "device": str(self.device),
                      "native_text_forward_sha256": hashlib.sha256(inspect.getsource(type(self.text).forward).encode()).hexdigest(),
                      "native_block_forward_sha256": hashlib.sha256(inspect.getsource(type(self.blocks[0]).forward).encode()).hexdigest(),
                      "order": "sequential", "student_block_passes": 2,
                      "deepstack": "native _deepstack_process after committed block"}

    def model_inputs(self, batch):
        # Mirror existing Seq2SeqTrainer cleanup. Packed-sequence metadata is kept.
        removed = {"labels", "loss_scale", "text_position_ids", "channel", "compute_loss_func"}
        return {k: tree_to(v, self.device) for k, v in batch.items() if k not in removed}

    @torch.no_grad()
    def collect_teacher(self, record, task, config):
        batch = record["batch"]
        record["teacher"] = {}
        metadata = {}
        def text_pre(module, args, kwargs):
            for key in ("visual_pos_masks", "deepstack_visual_embeds"):
                metadata[key] = tree_to(kwargs.get(key), "cpu", clone=True)
            visual = metadata["visual_pos_masks"]
            positions, stats = select_tokens(batch["attention_mask"][0], batch["labels"][0],
                                             visual[0] if visual is not None else None,
                                             count=config.tokens_per_sample,
                                             seed=config.seed + int(hashlib.sha256(record["id"].encode()).hexdigest()[:8], 16))
            record["positions"], record["selection"] = positions, stats
        def first_pre(module, args, kwargs):
            if not args or args[0].shape[:2] != batch["input_ids"].shape:
                raise ValueError("Native first block input/token shape mismatch")
            if kwargs.get("past_key_values") is not None:
                raise ValueError("Calibration must not use a KV cache")
            record["hidden"] = tree_to(args[0], "cpu", clone=True)
            record["kwargs"] = tree_to(kwargs, "cpu", clone=True)
        def capture(name):
            def hook(module, args):
                x = args[0]
                if x.shape[:2] != batch["input_ids"].shape:
                    raise ValueError("Projection sequence is not aligned to collator input")
                selected = x[0, record["positions"].to(x.device)].float()
                record["teacher"][name] = {
                    "sumsq": float(selected.double().square().sum()), "numel": selected.numel(),
                    "targets": {b.task_id: (selected @ b.matrices(device=x.device)[0].T).cpu()
                                for b in self.banks[name] if b.task_id < task}}
            return hook
        with cleanup_hooks() as handles:
            handles.append(self.text.register_forward_pre_hook(text_pre, with_kwargs=True))
            handles.append(self.blocks[0].register_forward_pre_hook(first_pre, with_kwargs=True))
            for name, wrapper in self.targets.items():
                handles.append(wrapper.register_forward_pre_hook(capture(name)))
            output = self.backbone(**self.model_inputs(batch), use_cache=False, return_dict=True)
            if not torch.isfinite(output.last_hidden_state).all():
                raise RuntimeError("Nonfinite teacher hidden state")
        record["visual"] = metadata
        if set(record["teacher"]) != set(self.targets):
            raise RuntimeError("Not all teacher projections were observed")

    @torch.no_grad()
    def run_block(self, index, record, *, capture=False):
        hidden = tree_to(record["hidden"], self.device, clone=True)
        kwargs = tree_to(record["kwargs"], self.device)
        captured = {}
        def make_hook(name):
            def hook(module, args):
                captured[name] = args[0][0, record["positions"].to(self.device)].detach().float().cpu()
            return hook
        with cleanup_hooks() as handles:
            if capture:
                for name in self.block_targets[index]:
                    handles.append(self.targets[name].register_forward_pre_hook(make_hook(name)))
            output = self.blocks[index](hidden, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise TypeError("Native block output contract changed")
        embeds = record["visual"]["deepstack_visual_embeds"]
        if embeds is not None and index < len(embeds):
            output = self.text._deepstack_process(output,
                tree_to(record["visual"]["visual_pos_masks"], self.device), tree_to(embeds[index], self.device))
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError(f"Nonfinite student block {index}")
        return output.detach().cpu(), captured

    @torch.no_grad()
    def check_native_equivalence(self, record):
        """One tiny teacher-forced batch, unchanged factors; no generation."""
        original = record["hidden"]
        try:
            direct = self.backbone(**self.model_inputs(record["batch"]), use_cache=False,
                                   return_dict=True).last_hidden_state.detach().cpu()
            for i in range(len(self.blocks)):
                record["hidden"], _ = self.run_block(i, record)
            streamed = self.text.norm(record["hidden"].to(self.device)).cpu()
            error = float((direct.float() - streamed.float()).abs().max())
            exact = torch.equal(direct, streamed)
            # Both use identical native operations. Require exact equality for this audited path.
            if not exact:
                raise RuntimeError(f"Native/blockwise hidden mismatch: max_abs={error}")
            return {"status": "PASS", "bitwise_equal": exact, "max_abs": error,
                    "shape": list(direct.shape), "includes_deepstack": record["visual"]["deepstack_visual_embeds"] is not None}
        finally:
            record["hidden"] = original
