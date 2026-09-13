from __future__ import annotations


from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OptimizerGroupAudit:
    shared_parameter_names: tuple[str, ...]
    current_private_parameter_names: tuple[str, ...]
    historical_private_parameter_names: tuple[str, ...]
    group_names: tuple[str, ...]
    shared_lr: float
    private_lr: float

    @property
    def shared_private_lr_ratio(self) -> float:
        return self.shared_lr / self.private_lr


def build_shared_private_optimizer_groups(
    model: torch.nn.Module,
    *,
    current_task_id: int,
    decay_parameter_names: set[str],
    weight_decay: float,
    shared_lr: float,
    private_lr: float,
) -> tuple[list[dict], OptimizerGroupAudit]:
    """Build deterministic, disjoint shared/current-private AdamW groups."""
    current_expert_marker = f"task_{current_task_id:04d}__"
    current_adapter_marker = f"task_{current_task_id:04d}."
    shared: list[tuple[str, torch.nn.Parameter]] = []
    current: list[tuple[str, torch.nn.Parameter]] = []
    historical: list[tuple[str, torch.nn.Parameter]] = []
    unexpected_trainable: list[str] = []
    for name, parameter in model.named_parameters():
        if ".shared." in name:
            if parameter.requires_grad:
                shared.append((name, parameter))
        elif ".experts." in name or ".task_adapters." in name:
            is_current = (
                ".experts." in name and current_expert_marker in name
            ) or (
                ".task_adapters." in name and current_adapter_marker in name
            )
            if is_current:
                if parameter.requires_grad:
                    current.append((name, parameter))
            else:
                historical.append((name, parameter))
                if parameter.requires_grad:
                    unexpected_trainable.append(name)
        elif parameter.requires_grad:
            unexpected_trainable.append(name)
    if unexpected_trainable:
        raise RuntimeError(
            "Only shared and current private parameters may be trainable: "
            f"{unexpected_trainable[:5]}"
        )
    if not shared or not current:
        raise RuntimeError("Shared-Fix requires non-empty shared and private groups")

    grouped: list[dict] = []
    for component, parameters, lr in (
        ("shared", shared, shared_lr),
        ("private", current, private_lr),
    ):
        for decay, decay_value in ((True, weight_decay), (False, 0.0)):
            selected = [
                parameter
                for name, parameter in parameters
                if (name in decay_parameter_names) == decay
            ]
            if selected:
                grouped.append(
                    {
                        "params": selected,
                        "lr": lr,
                        "weight_decay": decay_value,
                        "med_prism_group": component,
                        "med_prism_decay": decay,
                    }
                )

    parameter_ids = [id(p) for group in grouped for p in group["params"]]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("A parameter appears in multiple optimizer groups")
    expected_ids = {id(parameter) for _, parameter in shared + current}
    if set(parameter_ids) != expected_ids:
        raise RuntimeError("Optimizer groups do not cover the exact trainable set")
    audit = OptimizerGroupAudit(
        shared_parameter_names=tuple(sorted(name for name, _ in shared)),
        current_private_parameter_names=tuple(sorted(name for name, _ in current)),
        historical_private_parameter_names=tuple(
            sorted(name for name, _ in historical)
        ),
        group_names=tuple(group["med_prism_group"] for group in grouped),
        shared_lr=float(shared_lr),
        private_lr=float(private_lr),
    )
    return grouped, audit


def optimizer_component_lrs(optimizer) -> tuple[float | None, float | None]:
    shared = {
        float(group["lr"])
        for group in optimizer.param_groups
        if group.get("med_prism_group") == "shared"
    }
    private = {
        float(group["lr"])
        for group in optimizer.param_groups
        if group.get("med_prism_group") == "private"
    }
    if len(shared) > 1 or len(private) > 1:
        raise RuntimeError("Scheduler broke component LR consistency")
    return (next(iter(shared), None), next(iter(private), None))


def optimizer_component_initial_lrs(
    optimizer,
) -> tuple[float | None, float | None]:
    """Return scheduler base LRs while preserving component group identity."""
    shared = {
        float(group.get("initial_lr", group["lr"]))
        for group in optimizer.param_groups
        if group.get("med_prism_group") == "shared"
    }
    private = {
        float(group.get("initial_lr", group["lr"]))
        for group in optimizer.param_groups
        if group.get("med_prism_group") == "private"
    }
    if len(shared) > 1 or len(private) > 1:
        raise RuntimeError("Scheduler base LRs disagree within a component")
    return (next(iter(shared), None), next(iter(private), None))
