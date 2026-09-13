from __future__ import annotations

from dataclasses import dataclass

import torch

from med_prism.adapters.injection import iter_shared_private_wrappers


@dataclass(frozen=True)
class SharedDriftResult:
    loss: torch.Tensor
    mode: str
    factor_loss: torch.Tensor | None
    effective_numerator: float
    effective_reference_squared_norm: float
    effective_reference_norm: float
    effective_delta_norm: float
    effective_relative_shift: float
    layer_count: int
    task1_zero: bool


def _zero_from_shared(model) -> tuple[torch.Tensor, int]:
    terms = []
    layer_count = 0
    for _, wrapper in iter_shared_private_wrappers(model):
        layer_count += 1
        shared = wrapper.shared
        terms.append(shared.A.sum() * 0.0 + shared.B.sum() * 0.0)
    zero = torch.tensor(0.0) if not terms else torch.stack(terms).sum()
    return zero, layer_count


def compute_shared_drift_loss(model, *, eps: float = 1e-8) -> torch.Tensor:
    """Legacy v1.0 factor-space shared drift."""
    numerator = []
    denominator = []
    fallback = None
    for _, wrapper in iter_shared_private_wrappers(model):
        shared = wrapper.shared
        if fallback is None:
            fallback = shared.A.sum() * 0.0 + shared.B.sum() * 0.0
        numerator.extend(
            [
                (shared.A - shared.anchor_A.detach()).float().pow(2).sum(),
                (shared.B - shared.anchor_B.detach()).float().pow(2).sum(),
            ]
        )
        denominator.extend(
            [
                shared.anchor_A.detach().float().pow(2).sum(),
                shared.anchor_B.detach().float().pow(2).sum(),
            ]
        )
    if not numerator:
        return torch.tensor(0.0) if fallback is None else fallback
    return torch.stack(numerator).sum() / torch.stack(denominator).sum().clamp_min(eps)


def _effective_terms(shared) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact ||BA||^2, ||B0A0||^2, and <BA,B0A0> without materializing BA."""
    a = shared.A.float()
    b = shared.B.float()
    a0 = shared.anchor_A.detach().float()
    b0 = shared.anchor_B.detach().float()
    scale = shared.scaling.detach().float().to(device=a.device)
    current = torch.trace((b.T @ b) @ (a @ a.T)) * scale.pow(2)
    reference = torch.trace((b0.T @ b0) @ (a0 @ a0.T)) * scale.pow(2)
    cross = torch.trace((b.T @ b0) @ (a0 @ a.T)) * scale.pow(2)
    return current, reference, cross


def compute_effective_shared_drift(
    model,
    *,
    current_task_id: int,
    eps: float = 1e-8,
    include_factor_diagnostic: bool = True,
) -> SharedDriftResult:
    zero, layer_count = _zero_from_shared(model)
    factor = compute_shared_drift_loss(model, eps=eps) if include_factor_diagnostic else None
    if current_task_id == 1:
        return SharedDriftResult(
            loss=zero,
            mode="effective_BA",
            factor_loss=factor,
            effective_numerator=0.0,
            effective_reference_squared_norm=0.0,
            effective_reference_norm=0.0,
            effective_delta_norm=0.0,
            effective_relative_shift=0.0,
            layer_count=layer_count,
            task1_zero=True,
        )

    numerators = []
    references = []
    for _, wrapper in iter_shared_private_wrappers(model):
        current, reference, cross = _effective_terms(wrapper.shared)
        numerators.append((current + reference - 2.0 * cross).clamp_min(0.0))
        references.append(reference.clamp_min(0.0))
    if not numerators:
        numerator = zero
        reference_squared_norm = zero.detach()
    else:
        numerator = torch.stack(numerators).sum()
        reference_squared_norm = torch.stack(references).sum()
    denominator = reference_squared_norm.clamp_min(eps)
    loss = numerator / denominator
    numerator_value = float(numerator.detach().cpu())
    reference_value = float(reference_squared_norm.detach().cpu())
    delta_norm = max(numerator_value, 0.0) ** 0.5
    reference_norm = max(reference_value, 0.0) ** 0.5
    return SharedDriftResult(
        loss=loss,
        mode="effective_BA",
        factor_loss=factor,
        effective_numerator=numerator_value,
        effective_reference_squared_norm=reference_value,
        effective_reference_norm=reference_norm,
        effective_delta_norm=delta_norm,
        effective_relative_shift=delta_norm / max(reference_norm, eps**0.5),
        layer_count=layer_count,
        task1_zero=False,
    )


def compute_shared_drift(
    model,
    *,
    mode: str,
    current_task_id: int,
    eps: float = 1e-8,
    include_factor_diagnostic: bool = True,
) -> SharedDriftResult:
    if mode == "effective_BA":
        return compute_effective_shared_drift(
            model,
            current_task_id=current_task_id,
            eps=eps,
            include_factor_diagnostic=include_factor_diagnostic,
        )
    if mode != "factor":
        raise ValueError(f"Unknown shared drift mode: {mode}")
    factor = compute_shared_drift_loss(model, eps=eps)
    return SharedDriftResult(
        loss=factor,
        mode="factor",
        factor_loss=factor,
        effective_numerator=0.0,
        effective_reference_squared_norm=0.0,
        effective_reference_norm=0.0,
        effective_delta_norm=0.0,
        effective_relative_shift=0.0,
        layer_count=sum(1 for _ in iter_shared_private_wrappers(model)),
        task1_zero=False,
    )
