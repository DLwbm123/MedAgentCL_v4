from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rank1_lora import Rank1ExpertLinear


class SharedAdapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        lr_scale: float,
        device: torch.device,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("Shared rank must be positive")
        self.rank = int(rank)
        self.register_buffer(
            "alpha", torch.tensor(float(alpha), dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "scaling",
            torch.tensor(float(alpha) / rank, dtype=torch.float32),
            persistent=True,
        )
        self.A = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=dtype),
            requires_grad=trainable,
        )
        self.B = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=dtype),
            requires_grad=trainable,
        )
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.register_buffer("anchor_A", self.A.detach().clone(), persistent=False)
        self.register_buffer("anchor_B", self.B.detach().clone(), persistent=False)
        self.gradient_hook_enabled = lr_scale != 1.0
        if lr_scale != 1.0:
            self.A.register_hook(lambda gradient: gradient * lr_scale)
            self.B.register_hook(lambda gradient: gradient * lr_scale)

    def set_trainable(self, trainable: bool) -> None:
        self.A.requires_grad_(trainable)
        self.B.requires_grad_(trainable)

    def reset_anchor(self) -> None:
        self.anchor_A.copy_(self.A.detach())
        self.anchor_B.copy_(self.B.detach())

    def forward(self, x: torch.Tensor, dropout: nn.Module) -> torch.Tensor:
        scale = self.scaling.to(device=x.device, dtype=x.dtype)
        return F.linear(F.linear(dropout(x), self.A), self.B) * scale

    def tensor_hash(self) -> str:
        digest = hashlib.sha256()
        for tensor in (self.A, self.B):
            digest.update(
                tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
        return digest.hexdigest()


class TaskPrivateLoRA(nn.Module):
    """One conventional task-private LoRA; its rank dimensions are not experts."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        task_id: int,
        rank: int,
        alpha: float,
        device: torch.device,
        dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.task_id = int(task_id)
        self.rank = int(rank)
        self.register_buffer(
            "alpha", torch.tensor(float(alpha), dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "scaling",
            torch.tensor(float(alpha) / rank, dtype=torch.float32),
            persistent=True,
        )
        self.A = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=dtype),
            requires_grad=trainable,
        )
        self.B = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=dtype),
            requires_grad=trainable,
        )
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def set_trainable(self, trainable: bool) -> None:
        self.A.requires_grad_(trainable)
        self.B.requires_grad_(trainable)

    def forward(self, x: torch.Tensor, dropout: nn.Module) -> torch.Tensor:
        scale = self.scaling.to(device=x.device, dtype=x.dtype)
        return F.linear(F.linear(dropout(x), self.A), self.B) * scale


class SharedPrivateLinear(Rank1ExpertLinear):
    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        module_identity: str,
        shared_rank: int,
        shared_alpha: float,
        shared_lr_scale: float,
        dropout: float,
        shared_trainable: bool,
        private_adapter_type: str = "rank1_expert_bank",
    ) -> None:
        super().__init__(
            base_layer,
            module_identity=module_identity,
            dropout=dropout,
        )
        self.shared = SharedAdapter(
            self.in_features,
            self.out_features,
            rank=shared_rank,
            alpha=shared_alpha,
            lr_scale=shared_lr_scale,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
            trainable=shared_trainable,
        )
        if private_adapter_type not in {
            "rank1_expert_bank",
            "standard_rank16_lora",
        }:
            raise ValueError(f"Unknown private adapter type: {private_adapter_type}")
        self.private_adapter_type = private_adapter_type
        self.task_adapters = nn.ModuleDict()

    @staticmethod
    def task_adapter_key(task_id: int) -> str:
        return f"task_{task_id:04d}"

    @property
    def task_ids(self) -> tuple[int, ...]:
        if self.private_adapter_type == "rank1_expert_bank":
            return super().task_ids
        return tuple(
            sorted(int(key.removeprefix("task_")) for key in self.task_adapters)
        )

    def add_private_task(
        self,
        task_id: int,
        *,
        private_rank: int,
        experts_per_task: int,
        alpha: float,
        trainable: bool,
    ) -> None:
        if self.private_adapter_type == "rank1_expert_bank":
            if private_rank != experts_per_task:
                raise ValueError("Rank-1 bank rank/expert count mismatch")
            super().add_task(
                task_id,
                experts_per_task=experts_per_task,
                alpha=alpha,
                trainable=trainable,
            )
            return
        if private_rank != 16 or experts_per_task != 1:
            raise ValueError("Standard private adapter must be one rank-16 LoRA")
        key = self.task_adapter_key(task_id)
        if key in self.task_adapters:
            raise ValueError(f"Duplicate task-private LoRA: {task_id}")
        self.task_adapters[key] = TaskPrivateLoRA(
            self.in_features,
            self.out_features,
            task_id=task_id,
            rank=private_rank,
            alpha=alpha,
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype,
            trainable=trainable,
        )
        self._active_tasks = sorted(set(self._active_tasks + [task_id]))

    def task_adapter(self, task_id: int) -> TaskPrivateLoRA | None:
        if self.private_adapter_type != "standard_rank16_lora":
            return None
        key = self.task_adapter_key(task_id)
        return self.task_adapters[key] if key in self.task_adapters else None

    def set_active_tasks(self, task_ids) -> None:
        if self.private_adapter_type == "rank1_expert_bank":
            super().set_active_tasks(task_ids)
            return
        task_ids = sorted(set(int(task_id) for task_id in task_ids))
        missing = [task_id for task_id in task_ids if task_id not in self.task_ids]
        if missing:
            raise ValueError(f"Cannot activate missing task-private LoRAs: {missing}")
        self._active_tasks = task_ids

    def freeze_except(self, current_task_id: int) -> None:
        if self.private_adapter_type == "rank1_expert_bank":
            super().freeze_except(current_task_id)
            return
        if current_task_id not in self.task_ids:
            raise ValueError(f"Missing current task-private LoRA: {current_task_id}")
        for adapter in self.task_adapters.values():
            adapter.set_trainable(adapter.task_id == current_task_id)

    def shared_contribution(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(x, self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x) + self.shared_contribution(x)
        for task_id in self.active_tasks:
            if self.private_adapter_type == "rank1_expert_bank":
                result = result + self.task_contribution(x, task_id)
            else:
                adapter = self.task_adapter(task_id)
                if adapter is None:
                    raise RuntimeError(f"Missing active task-private LoRA: {task_id}")
                result = result + adapter(x, self.dropout)
        return result
