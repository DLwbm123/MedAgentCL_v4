from __future__ import annotations

import torch
import torch.nn as nn

from med_prism.adapters.injection import add_task_bank, inject_rank1_wrappers


class ToyAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)


class ToyLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_attn = ToyAttention(width)


class Qwen3VLTextModel(nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(width) for _ in range(36)])

    def forward(self, x):
        for layer in self.layers:
            x = layer.self_attn.q_proj(x) + layer.self_attn.v_proj(x)
        return x


class ToyContainer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.language_model = Qwen3VLTextModel(width)


class ToyVision(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.merger = nn.Linear(width, width, bias=False)


class ToyQwen3VL(nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        torch.manual_seed(7)
        self.model = ToyContainer(width)
        self.visual = ToyVision(width)

    def forward(self, x):
        return self.model.language_model(x)


def make_bank(task_specs=((1, 4, 4.0, True),), width: int = 4):
    model = ToyQwen3VL(width)
    model.requires_grad_(False)
    inject_rank1_wrappers(model, dropout=0.0)
    for task_id, experts, alpha, trainable in task_specs:
        add_task_bank(
            model,
            task_id=task_id,
            experts_per_task=experts,
            alpha=alpha,
            trainable=trainable,
        )
    return model


def set_expert(expert, a, b):
    with torch.no_grad():
        expert.A.copy_(torch.as_tensor(a, dtype=expert.A.dtype).reshape_as(expert.A))
        expert.B.copy_(torch.as_tensor(b, dtype=expert.B.dtype).reshape_as(expert.B))
