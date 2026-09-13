from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ExpertIdentity:
    task_id: int
    expert_id: int
    alpha: float
    per_task_rank: int
    scaling: float
    trainable: bool
    active: bool


class Rank1Expert(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        task_id: int,
        expert_id: int,
        alpha: float,
        per_task_rank: int,
        device: torch.device,
        dtype: torch.dtype,
        trainable: bool,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if per_task_rank <= 0:
            raise ValueError("per_task_rank must be positive")
        self.task_id = int(task_id)
        self.expert_id = int(expert_id)
        self.per_task_rank = int(per_task_rank)
        self.register_buffer(
            "alpha", torch.tensor(float(alpha), dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "scaling",
            torch.tensor(float(alpha) / per_task_rank, dtype=torch.float32),
            persistent=True,
        )
        self.A = nn.Parameter(
            torch.empty(1, in_features, device=device, dtype=dtype),
            requires_grad=trainable,
        )
        self.B = nn.Parameter(
            torch.zeros(out_features, 1, device=device, dtype=dtype),
            requires_grad=trainable,
        )
        nn.init.normal_(self.A, mean=0.0, std=init_std)

    def set_trainable(self, trainable: bool) -> None:
        self.A.requires_grad_(trainable)
        self.B.requires_grad_(trainable)

    def forward(self, x: torch.Tensor, dropout: nn.Module) -> torch.Tensor:
        scale = self.scaling.to(device=x.device, dtype=x.dtype)
        return F.linear(F.linear(dropout(x), self.A), self.B) * scale

    def tensor_hash(self) -> str:
        digest = hashlib.sha256()
        for tensor in (self.A, self.B):
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


class Rank1ExpertLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        module_identity: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("Rank1ExpertLinear requires nn.Linear")
        self.base_layer = base_layer
        self.module_identity = module_identity
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.dropout = nn.Dropout(dropout)
        self.experts = nn.ModuleDict()
        self._task_keys: dict[int, list[str]] = {}
        self._active_tasks: list[int] = []
        self.base_layer.requires_grad_(False)

    @staticmethod
    def expert_key(task_id: int, expert_id: int) -> str:
        return f"task_{task_id:04d}__expert_{expert_id:04d}"

    @property
    def task_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._task_keys))

    @property
    def active_tasks(self) -> tuple[int, ...]:
        return tuple(self._active_tasks)

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def add_task(
        self,
        task_id: int,
        *,
        experts_per_task: int,
        alpha: float,
        trainable: bool,
    ) -> None:
        if task_id in self._task_keys:
            raise ValueError(f"Duplicate task bank: {task_id}")
        if experts_per_task <= 0:
            raise ValueError("experts_per_task must be positive")
        keys = []
        for expert_id in range(experts_per_task):
            key = self.expert_key(task_id, expert_id)
            self.experts[key] = Rank1Expert(
                self.in_features,
                self.out_features,
                task_id=task_id,
                expert_id=expert_id,
                alpha=alpha,
                per_task_rank=experts_per_task,
                device=self.base_layer.weight.device,
                dtype=self.base_layer.weight.dtype,
                trainable=trainable,
            )
            keys.append(key)
        self._task_keys[task_id] = keys
        self._active_tasks = sorted(set(self._active_tasks + [task_id]))

    def set_active_tasks(self, task_ids: Iterable[int]) -> None:
        task_ids = sorted(set(int(task_id) for task_id in task_ids))
        missing = [task_id for task_id in task_ids if task_id not in self._task_keys]
        if missing:
            raise ValueError(f"Cannot activate missing task banks: {missing}")
        self._active_tasks = task_ids

    def freeze_except(self, current_task_id: int) -> None:
        if current_task_id not in self._task_keys:
            raise ValueError(f"Missing current task bank: {current_task_id}")
        for task_id, keys in self._task_keys.items():
            for key in keys:
                self.experts[key].set_trainable(task_id == current_task_id)

    def task_experts(self, task_id: int) -> list[Rank1Expert]:
        if task_id not in self._task_keys:
            return []
        return [self.experts[key] for key in self._task_keys[task_id]]

    def expert_identities(self) -> list[ExpertIdentity]:
        identities = []
        for task_id in self.task_ids:
            for expert in self.task_experts(task_id):
                identities.append(
                    ExpertIdentity(
                        task_id=task_id,
                        expert_id=expert.expert_id,
                        alpha=float(expert.alpha.item()),
                        per_task_rank=expert.per_task_rank,
                        scaling=float(expert.scaling.item()),
                        trainable=expert.A.requires_grad or expert.B.requires_grad,
                        active=task_id in self._active_tasks,
                    )
                )
        return identities

    def task_contribution(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        contribution = torch.zeros(
            *x.shape[:-1],
            self.out_features,
            device=x.device,
            dtype=x.dtype,
        )
        for expert in self.task_experts(task_id):
            contribution = contribution + expert(x, self.dropout)
        return contribution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        for task_id in self._active_tasks:
            result = result + self.task_contribution(x, task_id)
        return result
