from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch.nn as nn

from .rank1_lora import Rank1ExpertLinear
from .shared_private import SharedPrivateLinear


@dataclass(frozen=True)
class TargetInventory:
    names: tuple[str, ...]
    q_proj: int
    v_proj: int
    language: int
    vision: int
    merger: int
    sha256: str


def classify_target_names(names: list[str] | tuple[str, ...]) -> TargetInventory:
    selected = sorted(
        name
        for name in names
        if (
            ".language_model." in name
            and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
            and not any(
                token in name
                for token in ("merger", "deepstack_merger", "aligner", "projector")
            )
        )
    )
    digest = hashlib.sha256(("\n".join(selected) + "\n").encode()).hexdigest()
    return TargetInventory(
        names=tuple(selected),
        q_proj=sum(name.endswith(".q_proj") for name in selected),
        v_proj=sum(name.endswith(".v_proj") for name in selected),
        language=len(selected),
        vision=sum(".visual." in name for name in selected),
        merger=sum(
            any(
                token in name
                for token in ("merger", "deepstack_merger", "aligner", "projector")
            )
            for name in selected
        ),
        sha256=digest,
    )


def _find_language_module(model: nn.Module) -> tuple[str, nn.Module]:
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "Qwen3VLTextModel"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Qwen3VLTextModel, found {len(matches)}")
    return matches[0]


def collect_language_targets(model: nn.Module) -> TargetInventory:
    language_prefix, language = _find_language_module(model)
    names = []
    for relative_name, module in language.named_modules():
        if not relative_name:
            continue
        leaf = relative_name.rsplit(".", 1)[-1]
        if leaf not in {"q_proj", "v_proj"}:
            continue
        if not isinstance(module, nn.Linear):
            if isinstance(module, Rank1ExpertLinear):
                names.append(module.module_identity)
                continue
            raise TypeError(
                f"Target {language_prefix}.{relative_name} is {type(module).__name__}, expected Linear"
            )
        names.append(f"{language_prefix}.{relative_name}")
    inventory = classify_target_names(names)
    if (
        inventory.language != 72
        or inventory.q_proj != 36
        or inventory.v_proj != 36
        or inventory.vision
        or inventory.merger
    ):
        raise RuntimeError(f"Qwen3-VL target audit failed: {inventory}")
    return inventory


def inject_rank1_wrappers(model: nn.Module, *, dropout: float) -> TargetInventory:
    inventory = collect_language_targets(model)
    candidates = []
    for module_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            full_name = child_name if not module_name else f"{module_name}.{child_name}"
            if full_name not in inventory.names:
                continue
            if isinstance(child, Rank1ExpertLinear):
                continue
            if not isinstance(child, nn.Linear):
                raise TypeError(f"Target {full_name} is not nn.Linear")
            candidates.append((full_name, parent, child_name, child))
    for full_name, parent, child_name, child in candidates:
        setattr(
            parent,
            child_name,
            Rank1ExpertLinear(
                child,
                module_identity=full_name,
                dropout=dropout,
            ),
        )
    if len(candidates) != 72:
        raise RuntimeError(f"Expected 72 wrapper replacements, got {len(candidates)}")
    return inventory


def iter_rank1_wrappers(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, Rank1ExpertLinear):
            yield name, module


def add_task_bank(
    model: nn.Module,
    *,
    task_id: int,
    experts_per_task: int,
    alpha: float,
    trainable: bool,
    private_rank: int | None = None,
) -> None:
    wrappers = list(iter_rank1_wrappers(model))
    if len(wrappers) != 72:
        raise RuntimeError(
            f"Expected 72 Rank1ExpertLinear wrappers, got {len(wrappers)}"
        )
    for _, wrapper in wrappers:
        if isinstance(wrapper, SharedPrivateLinear):
            wrapper.add_private_task(
                task_id,
                private_rank=(
                    experts_per_task if private_rank is None else private_rank
                ),
                experts_per_task=experts_per_task,
                alpha=alpha,
                trainable=trainable,
            )
        else:
            wrapper.add_task(
                task_id,
                experts_per_task=experts_per_task,
                alpha=alpha,
                trainable=trainable,
            )


def freeze_except_task(model: nn.Module, current_task_id: int) -> None:
    for _, wrapper in iter_rank1_wrappers(model):
        wrapper.freeze_except(current_task_id)


def inject_shared_private_wrappers(
    model: nn.Module,
    *,
    shared_rank: int,
    shared_alpha: float,
    shared_lr_scale: float,
    dropout: float,
    shared_trainable: bool,
    private_adapter_type: str = "rank1_expert_bank",
) -> TargetInventory:
    inventory = collect_language_targets(model)
    candidates = []
    for module_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            full_name = child_name if not module_name else f"{module_name}.{child_name}"
            if full_name not in inventory.names:
                continue
            if isinstance(child, SharedPrivateLinear):
                raise ValueError(f"Duplicate shared component at {full_name}")
            if not isinstance(child, nn.Linear):
                raise TypeError(f"Target {full_name} is not nn.Linear")
            candidates.append((full_name, parent, child_name, child))
    for full_name, parent, child_name, child in candidates:
        setattr(
            parent,
            child_name,
            SharedPrivateLinear(
                child,
                module_identity=full_name,
                shared_rank=shared_rank,
                shared_alpha=shared_alpha,
                shared_lr_scale=shared_lr_scale,
                dropout=dropout,
                shared_trainable=shared_trainable,
                private_adapter_type=private_adapter_type,
            ),
        )
    if len(candidates) != 72:
        raise RuntimeError(
            f"Expected 72 shared/private replacements, got {len(candidates)}"
        )
    return inventory


def iter_shared_private_wrappers(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, SharedPrivateLinear):
            yield name, module


def freeze_shared_private_except(model: nn.Module, current_task_id: int) -> None:
    wrappers = list(iter_shared_private_wrappers(model))
    if len(wrappers) != 72:
        raise RuntimeError(f"Expected 72 shared/private wrappers, got {len(wrappers)}")
    for _, wrapper in wrappers:
        wrapper.shared.set_trainable(True)
        wrapper.freeze_except(current_task_id)
