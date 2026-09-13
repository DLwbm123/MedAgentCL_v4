from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from med_prism.adapters.injection import iter_rank1_wrappers


@dataclass(frozen=True)
class OrthLossResult:
    loss: torch.Tensor
    squared_loss: torch.Tensor
    layer_count: int
    pair_count: int
    mean_overlap: float
    max_overlap: float
    mean_abs_overlap: float
    rms_overlap: float
    max_abs_overlap: float
    current_task_id: int
    old_task_count: int
    loss_type: str
    eps: float


@dataclass(frozen=True)
class KeyIsolationResult:
    loss: torch.Tensor
    squared_loss: torch.Tensor
    layer_count: int
    pair_count: int
    mean_abs_cosine: float
    rms_cosine: float
    max_abs_cosine: float
    current_task_id: int
    old_task_count: int
    eps: float


def rank1_overlap_squared(
    current_a: torch.Tensor,
    current_b: torch.Tensor,
    old_a: torch.Tensor,
    old_b: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    if current_a.ndim != 2 or old_a.ndim != 2:
        raise ValueError("A tensors must have shape [experts, in_features]")
    if current_b.ndim != 2 or old_b.ndim != 2:
        raise ValueError("B tensors must have shape [experts, out_features]")
    current_a = F.normalize(current_a.float(), dim=-1, eps=eps)
    current_b = F.normalize(current_b.float(), dim=-1, eps=eps)
    old_a = F.normalize(old_a.detach().float(), dim=-1, eps=eps)
    old_b = F.normalize(old_b.detach().float(), dim=-1, eps=eps)
    return ((current_b @ old_b.T) * (current_a @ old_a.T)).pow(2)


def rank1_key_cosines(
    current_a: torch.Tensor,
    old_a: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    if current_a.ndim != 2 or old_a.ndim != 2:
        raise ValueError("A tensors must have shape [experts, in_features]")
    current = F.normalize(current_a.float(), dim=-1, eps=eps)
    historical = F.normalize(old_a.detach().float(), dim=-1, eps=eps)
    return current @ historical.T


def _current_and_old(wrapper, current_task_id: int):
    current = wrapper.task_experts(current_task_id)
    old = [
        expert
        for task_id in wrapper.task_ids
        if task_id != current_task_id
        for expert in wrapper.task_experts(task_id)
    ]
    old_task_ids = {
        task_id for task_id in wrapper.task_ids if task_id != current_task_id
    }
    return current, old, old_task_ids


def compute_rank1_orth_loss(
    model,
    *,
    current_task_id: int,
    loss_type: str,
    eps: float = 1e-8,
) -> OrthLossResult:
    if loss_type not in {"squared", "rms"}:
        raise ValueError(f"Unknown orth loss type: {loss_type}")
    weighted = []
    overlaps = []
    layer_count = 0
    pair_count = 0
    fallback = None
    old_task_ids: set[int] = set()
    for _, wrapper in iter_rank1_wrappers(model):
        current, old, layer_old_task_ids = _current_and_old(
            wrapper, current_task_id
        )
        old_task_ids.update(layer_old_task_ids)
        if fallback is None and current:
            fallback = sum(
                expert.A.sum() * 0.0 + expert.B.sum() * 0.0
                for expert in current
            )
        if not current or not old:
            continue
        current_a = torch.cat([expert.A for expert in current], dim=0)
        current_b = torch.cat([expert.B.T for expert in current], dim=0)
        old_a = torch.cat([expert.A for expert in old], dim=0)
        old_b = torch.cat([expert.B.T for expert in old], dim=0)
        overlap = rank1_overlap_squared(
            current_a, current_b, old_a, old_b, eps=eps
        )
        weighted.append(overlap.sum())
        overlaps.append(overlap.detach())
        pair_count += overlap.numel()
        layer_count += 1
    if pair_count:
        squared = torch.stack(weighted).sum() / pair_count
    elif fallback is not None:
        squared = fallback
    else:
        squared = torch.tensor(0.0)
    loss = (
        squared
        if loss_type == "squared" or pair_count == 0
        else torch.sqrt(squared + eps)
    )
    if overlaps:
        flat = torch.cat(
            [overlap.reshape(-1).float().cpu() for overlap in overlaps]
        )
        mean_overlap = float(flat.mean())
        max_overlap = float(flat.max())
        absolute = flat.clamp_min(0.0).sqrt()
        mean_abs_overlap = float(absolute.mean())
        rms_overlap = float(flat.mean().sqrt())
        max_abs_overlap = float(absolute.max())
    else:
        mean_overlap = max_overlap = 0.0
        mean_abs_overlap = rms_overlap = max_abs_overlap = 0.0
    return OrthLossResult(
        loss=loss,
        squared_loss=squared,
        layer_count=layer_count,
        pair_count=pair_count,
        mean_overlap=mean_overlap,
        max_overlap=max_overlap,
        mean_abs_overlap=mean_abs_overlap,
        rms_overlap=rms_overlap,
        max_abs_overlap=max_abs_overlap,
        current_task_id=current_task_id,
        old_task_count=len(old_task_ids),
        loss_type=loss_type,
        eps=eps,
    )


def compute_rank1_key_isolation_loss(
    model,
    *,
    current_task_id: int,
    eps: float = 1e-8,
) -> KeyIsolationResult:
    squared_terms = []
    detached_cosines = []
    layer_count = 0
    pair_count = 0
    fallback_terms = []
    old_task_ids: set[int] = set()
    for _, wrapper in iter_rank1_wrappers(model):
        current, old, layer_old_task_ids = _current_and_old(
            wrapper, current_task_id
        )
        old_task_ids.update(layer_old_task_ids)
        if current:
            fallback_terms.extend(expert.A.sum() * 0.0 for expert in current)
        if not current or not old:
            continue
        current_a = torch.cat([expert.A for expert in current], dim=0)
        old_a = torch.cat([expert.A for expert in old], dim=0)
        cosine = rank1_key_cosines(current_a, old_a, eps=eps)
        squared_terms.append(cosine.pow(2).sum())
        detached_cosines.append(cosine.detach())
        pair_count += cosine.numel()
        layer_count += 1
    if pair_count:
        squared = torch.stack(squared_terms).sum() / pair_count
        loss = torch.sqrt(squared + eps)
    elif fallback_terms:
        squared = torch.stack(fallback_terms).sum()
        loss = squared
    else:
        squared = loss = torch.tensor(0.0)
    if detached_cosines:
        flat = torch.cat(
            [cosine.reshape(-1).float().cpu() for cosine in detached_cosines]
        ).abs()
        mean_abs = float(flat.mean())
        rms = float(flat.pow(2).mean().sqrt())
        maximum = float(flat.max())
    else:
        mean_abs = rms = maximum = 0.0
    return KeyIsolationResult(
        loss=loss,
        squared_loss=squared,
        layer_count=layer_count,
        pair_count=pair_count,
        mean_abs_cosine=mean_abs,
        rms_cosine=rms,
        max_abs_cosine=maximum,
        current_task_id=current_task_id,
        old_task_count=len(old_task_ids),
        eps=eps,
    )
