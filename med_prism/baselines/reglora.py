"""RegLoRA (SEFE component only) primitives.

Ported from official SEFE commit 2f61efe3a7da0407850399db6bc51f5ddc00f001.
ASD and every other SEFE component are intentionally absent.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from med_prism.baselines.octopus import canonical_base_weight_name, lora_pairs
from scripts.medicalskill_v1_2_baselines.baseline_harness import (
    inspect_adapter, sha256_file,
)

OFFICIAL_REPOSITORY = "https://github.com/jinpeng0528/SEFE"
OFFICIAL_COMMIT = "2f61efe3a7da0407850399db6bc51f5ddc00f001"
TOP_FRACTION = 0.02
DEFAULT_INDEX_CHUNK_SIZE = 262144


class _SelectedEntryAbsMean(torch.autograd.Function):
    """Memory-bounded exact mean(abs((B @ A)[row, col])).

    The cumulative SEFE mask becomes very large by Task 5. A normal indexed
    expression retains two ``[num_indices, rank]`` gathers for backward. This
    custom autograd function keeps indices on CPU and recomputes bounded chunks
    in backward, preserving the same value and parameter gradients.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, index: torch.Tensor,
                chunk_size: int) -> torch.Tensor:
        if index.ndim != 2 or index.shape[1] != 2 or index.dtype not in {torch.int32, torch.int64}:
            raise RuntimeError("RegLoRA indices must be int32/int64 [N,2]")
        if chunk_size < 1:
            raise ValueError("RegLoRA chunk size must be positive")
        ctx.save_for_backward(a, b, index)
        ctx.chunk_size = int(chunk_size)
        count = int(index.shape[0])
        if count < 1:
            raise RuntimeError("RegLoRA protected index set is empty")
        total = torch.zeros((), device=a.device, dtype=torch.float32)
        a_t = a.transpose(0, 1)
        for start in range(0, count, ctx.chunk_size):
            current = index[start:start + ctx.chunk_size].to(a.device)
            selected = (b[current[:, 0]] * a_t[current[:, 1]]).sum(-1)
            total = total + selected.abs().float().sum()
        return total / count

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b, index = ctx.saved_tensors
        count = int(index.shape[0])
        grad_a_t = torch.zeros((a.shape[1], a.shape[0]), device=a.device, dtype=a.dtype)
        grad_b = torch.zeros_like(b)
        a_t = a.transpose(0, 1)
        scale = (grad_output.float() / count).to(a.dtype)
        for start in range(0, count, ctx.chunk_size):
            current = index[start:start + ctx.chunk_size].to(a.device)
            rows, cols = current[:, 0], current[:, 1]
            b_rows = b[rows]
            a_cols = a_t[cols]
            coefficient = (b_rows * a_cols).sum(-1).sign() * scale
            grad_b.index_add_(0, rows, coefficient.unsqueeze(1) * a_cols)
            grad_a_t.index_add_(0, cols, coefficient.unsqueeze(1) * b_rows)
        return grad_a_t.transpose(0, 1).contiguous(), grad_b, None, None


def selected_entry_abs_mean(a: torch.Tensor, b: torch.Tensor, index: torch.Tensor,
                            chunk_size: int = DEFAULT_INDEX_CHUNK_SIZE) -> torch.Tensor:
    return _SelectedEntryAbsMean.apply(a, b, index, chunk_size)



def _checkpoint_pairs(checkpoint: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    info = inspect_adapter(checkpoint)
    state = load_file(info["adapter_weights"], device="cpu")
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        if ".lora_A." in key:
            grouped.setdefault(key.split(".lora_A.", 1)[0], {})["A"] = value
        elif ".lora_B." in key:
            grouped.setdefault(key.split(".lora_B.", 1)[0], {})["B"] = value
    pairs = {canonical_base_weight_name(name + ".lora_A.default.weight"): (parts["A"], parts["B"])
             for name, parts in grouped.items() if set(parts) == {"A", "B"}}
    if not pairs:
        raise RuntimeError(f"No LoRA pairs in {checkpoint}")
    return pairs, info


def build_importance_mask(checkpoint: Path, output: Path,
                          previous: Path | None = None,
                          top_fraction: float = TOP_FRACTION) -> dict[str, Any]:
    """Build official |BA| top-2% indices and concatenate prior indices.

    Deliberately does not deduplicate: official ``key_elements_identification.py``
    uses ``torch.cat`` and the loss indexes the resulting rows directly.
    """
    if top_fraction != TOP_FRACTION:
        raise ValueError("Formal RegLoRA uses the official top fraction 0.02")
    started = time.perf_counter()
    pairs, info = _checkpoint_pairs(checkpoint)
    legacy = load_importance_mask(previous)["masks"] if previous else {}
    masks: dict[str, torch.Tensor] = {}
    layer_shapes: dict[str, list[int]] = {}
    current_counts: dict[str, int] = {}
    for name, (a, b) in sorted(pairs.items()):
        metric = torch.abs((b @ a).float())
        k = int(top_fraction * metric.numel())
        if k < 1:
            raise RuntimeError(f"Official int(0.02*numel) is zero for {name}")
        flat = torch.topk(metric.reshape(-1), k).indices
        indices = torch.stack((torch.div(flat, metric.shape[1], rounding_mode="floor"),
                               flat % metric.shape[1]), dim=1).cpu()
        if previous:
            if name not in legacy:
                raise RuntimeError(f"Previous mask missing layer {name}")
            indices = torch.cat((indices, legacy[name]), dim=0)
        masks[name] = indices
        layer_shapes[name] = list(metric.shape)
        current_counts[name] = k
    if previous and set(legacy) != set(masks):
        raise RuntimeError("Previous/current RegLoRA mask layer sets differ")
    payload = {
        "format_version": "medicalskill_cl_reglora_mask_v1",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "importance_metric": "abs(lora_B @ lora_A)",
        "top_fraction_per_task": top_fraction,
        "accumulation": "direct_concatenation_without_deduplication",
        "source_checkpoint": info["checkpoint"],
        "source_adapter_weights_sha256": info["adapter_weights_sha256"],
        "previous_mask": str(previous.resolve()) if previous else None,
        "previous_mask_sha256": sha256_file(previous) if previous else None,
        "layer_shapes": layer_shapes,
        "current_task_selected_counts": current_counts,
        "masks": masks,
    }
    output = output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    meta = inspect_importance_mask(output)
    meta["importance_mask_build_seconds"] = time.perf_counter() - started
    return meta


def load_importance_mask(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"masks": {}}
    value = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    required = {"format_version", "importance_metric", "top_fraction_per_task", "masks", "layer_shapes"}
    if not required.issubset(value) or value["format_version"] != "medicalskill_cl_reglora_mask_v1":
        raise RuntimeError(f"Invalid RegLoRA mask: {path}")
    return value


def inspect_importance_mask(path: Path) -> dict[str, Any]:
    path = path.resolve(); value = load_importance_mask(path)
    stored = sum(int(x.shape[0]) for x in value["masks"].values())
    unique = sum(int(torch.unique(x, dim=0).shape[0]) for x in value["masks"].values())
    total = sum(int(s[0]) * int(s[1]) for s in value["layer_shapes"].values())
    return {"status": "PASS", "mask": str(path), "mask_sha256": sha256_file(path),
            "mask_storage_bytes": path.stat().st_size, "layer_count": len(value["masks"]),
            "stored_index_count": stored, "protected_position_count": unique,
            "protected_position_fraction": unique / total if total else 0.0,
            "top_fraction_per_task": value["top_fraction_per_task"],
            "accumulation": value.get("accumulation")}


def reglora_regularizer(model: torch.nn.Module, mask_source: Path | dict[str, Any] | None) -> tuple[torch.Tensor, dict[str, Any]]:
    pairs = lora_pairs(model)
    if not pairs:
        raise RuntimeError("No current RegLoRA pairs")
    loss = pairs[0][1].new_zeros(())
    if mask_source is None:
        return loss, {"protected_position_count": 0, "protected_position_fraction": 0.0}
    value = mask_source if isinstance(mask_source, dict) else load_importance_mask(mask_source)
    masks = value["masks"]
    seen = set()
    for name, a, b in pairs:
        if name not in masks:
            raise RuntimeError(f"Mask missing current layer {name}")
        index = masks[name]
        chunk_size = int(os.environ.get("REGLORA_INDEX_CHUNK_SIZE", DEFAULT_INDEX_CHUNK_SIZE))
        loss = loss + selected_entry_abs_mean(a, b, index, chunk_size)
        seen.add(name)
    if seen != set(masks):
        raise RuntimeError("Mask/current RegLoRA layer sets differ")
    # Official SEFE divides by 224 = 32 layers x 7 targeted linears.
    # The locked q/v-only protocol therefore divides by its 72 active linears.
    loss = loss / len(pairs)
    meta = dict(value.get("_runtime_meta") or {})
    if not meta:
        meta = inspect_importance_mask(mask_source)
    meta["loss_normalization_divisor"] = len(pairs)
    meta["loss_implementation"] = "memory_bounded_custom_autograd_selected_entry_dot_without_dense_BA"
    meta["index_chunk_size"] = int(os.environ.get("REGLORA_INDEX_CHUNK_SIZE", DEFAULT_INDEX_CHUNK_SIZE))
    meta["runtime_index_device"] = next(iter(masks.values())).device.type
    return loss, meta
