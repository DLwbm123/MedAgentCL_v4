#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from med_prism.adapters.shared_private import SharedPrivateLinear
from med_prism.config import MODEL_ID, MODEL_REVISION, SharedPrivateConfig
from med_prism.optimization.shared_private import (
    build_shared_private_optimizer_groups,
)
from med_prism.projection.orth_losses import (
    compute_rank1_key_isolation_loss,
    compute_rank1_orth_loss,
)
from med_prism.projection.shared_private import compute_shared_drift


class Tiny(nn.Module):
    def __init__(self, private_adapter_type: str):
        super().__init__()
        self.proj = SharedPrivateLinear(
            nn.Linear(8, 6, bias=False),
            module_identity="model.language_model.layers.0.q_proj",
            shared_rank=4,
            shared_alpha=4,
            shared_lr_scale=1,
            dropout=0,
            shared_trainable=True,
            private_adapter_type=private_adapter_type,
        )


def config(ablation: str) -> SharedPrivateConfig:
    rank16 = ablation == "rank16_private"
    return SharedPrivateConfig(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        current_task_id=2,
        shared_rank=32,
        shared_alpha=32,
        shared_lr_scale=0.1,
        private_experts_per_task=1 if rank16 else 16,
        private_rank=16,
        private_adapter_type=(
            "standard_rank16_lora" if rank16 else "rank1_expert_bank"
        ),
        private_alpha=16,
        orth_lambda=0 if ablation in {"no_geo_orth", "rank16_private"} else 0.1,
        shared_drift_lambda=0 if ablation == "no_shared_drift" else 0.01,
        method_version="1.2",
        method_variant=ablation,
        ablation_name=ablation,
        shared_optimizer_mode="separate_lr_param_groups",
        shared_lr=1e-5,
        private_lr=1e-4,
        shared_gradient_hook_enabled=False,
        shared_drift_mode="effective_BA",
        key_isolation_enabled=not rank16,
        key_loss_mode="disabled" if rank16 else "rms_cosine_A",
        key_lambda=0 if rank16 else 0.1,
        key_history_scope="none" if rank16 else "all_previous_tasks_same_layer",
        shared_source_manifest="previous_shared.json",
        private_source_manifests=("previous_private.json",),
        shared_output_dir="shared",
        private_output_dir="private",
    )


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_configs() -> None:
    geo = config("no_geo_orth")
    assert geo.orth_lambda == 0
    assert geo.key_isolation_enabled and geo.key_lambda == 0.1
    assert geo.shared_drift_lambda == 0.01

    drift = config("no_shared_drift")
    assert drift.shared_drift_lambda == 0
    assert drift.shared_lr == 1e-5 and drift.private_lr == 1e-4
    assert drift.orth_lambda == 0.1 and drift.key_isolation_enabled

    rank16 = config("rank16_private")
    assert rank16.private_adapter_type == "standard_rank16_lora"
    assert rank16.private_experts_per_task == 1 and rank16.private_rank == 16
    assert rank16.orth_lambda == 0 and not rank16.key_isolation_enabled


def test_capacity_and_ownership() -> None:
    rank1 = Tiny("rank1_expert_bank")
    rank1.proj.add_private_task(
        1, private_rank=16, experts_per_task=16, alpha=16, trainable=False
    )
    rank1.proj.add_private_task(
        2, private_rank=16, experts_per_task=16, alpha=16, trainable=True
    )
    standard = Tiny("standard_rank16_lora")
    standard.proj.add_private_task(
        1, private_rank=16, experts_per_task=1, alpha=16, trainable=False
    )
    standard.proj.add_private_task(
        2, private_rank=16, experts_per_task=1, alpha=16, trainable=True
    )
    rank1_private = parameter_count(rank1.proj.experts)
    standard_private = parameter_count(standard.proj.task_adapters)
    assert rank1_private == standard_private == 2 * 16 * (8 + 6)
    assert len(standard.proj.task_adapters) == 2
    assert len(standard.proj.experts) == 0
    standard.proj.freeze_except(2)
    assert not any(p.requires_grad for p in standard.proj.task_adapter(1).parameters())
    assert all(p.requires_grad for p in standard.proj.task_adapter(2).parameters())
    standard.proj.set_active_tasks([1, 2])
    assert standard.proj(torch.randn(3, 8)).shape == (3, 6)


def test_rank16_optimizer_ownership() -> None:
    model = Tiny("standard_rank16_lora")
    model.proj.add_private_task(
        1, private_rank=16, experts_per_task=1, alpha=16, trainable=False
    )
    model.proj.add_private_task(
        2, private_rank=16, experts_per_task=1, alpha=16, trainable=True
    )
    model.proj.freeze_except(2)
    groups, audit = build_shared_private_optimizer_groups(
        model,
        current_task_id=2,
        decay_parameter_names={name for name, _ in model.named_parameters()},
        weight_decay=0.1,
        shared_lr=1e-5,
        private_lr=1e-4,
    )
    assert groups
    assert audit.current_private_parameter_names
    assert all(
        ".task_adapters.task_0002." in name
        for name in audit.current_private_parameter_names
    )
    assert audit.historical_private_parameter_names
    assert all(
        ".task_adapters.task_0001." in name
        for name in audit.historical_private_parameter_names
    )


def test_objectives() -> None:
    model = Tiny("rank1_expert_bank")
    for task, trainable in ((1, False), (2, True)):
        model.proj.add_private_task(
            task,
            private_rank=2,
            experts_per_task=2,
            alpha=2,
            trainable=trainable,
        )
    with torch.no_grad():
        for expert in model.proj.task_experts(1) + model.proj.task_experts(2):
            expert.B.normal_()
    geo = compute_rank1_orth_loss(
        model, current_task_id=2, loss_type="rms", eps=1e-8
    )
    key = compute_rank1_key_isolation_loss(model, current_task_id=2, eps=1e-8)
    assert geo.pair_count > 0 and geo.loss.item() > 0
    assert key.pair_count > 0 and key.loss.item() > 0
    assert (geo.loss * config("no_geo_orth").orth_lambda).item() == 0

    with torch.no_grad():
        model.proj.shared.B.normal_()
        model.proj.shared.reset_anchor()
        model.proj.shared.B.add_(0.1)
    drift = compute_shared_drift(
        model,
        mode="effective_BA",
        current_task_id=2,
        eps=1e-8,
    )
    assert drift.loss.item() > 0
    assert (drift.loss * config("no_shared_drift").shared_drift_lambda).item() == 0


if __name__ == "__main__":
    test_configs()
    test_capacity_and_ownership()
    test_rank16_optimizer_ownership()
    test_objectives()
    print("MED-PRISM V1.2 ABLATION CPU TESTS PASS")
