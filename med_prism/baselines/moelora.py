"""Dense token-routed CoIN MoELoRA on PEFT LoRA tensors.

Official source: zackschen/CoIN commit 41411ab9ad77a520bc61f07a09fe9eba2f14cf6a.
The official rank is the total concatenated expert rank; every token is routed
through every expert with softmax weights.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from peft.tuners.lora.layer import Linear as PeftLoraLinear

OFFICIAL_REPOSITORY = "https://github.com/zackschen/CoIN"
OFFICIAL_COMMIT = "41411ab9ad77a520bc61f07a09fe9eba2f14cf6a"
NUM_EXPERTS = 4
TOTAL_RANK = 48
RANK_PER_EXPERT = 12
_PATCHED = False
_ORIGINAL_UPDATE = None
_ORIGINAL_FORWARD = None


def _dense_moe_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    if not hasattr(self, "lora_router"):
        return _ORIGINAL_FORWARD(self, x, *args, **kwargs)
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)
    if adapter_names is not None:
        raise RuntimeError("MoELoRA does not support sample-selected adapters")
    if self.disable_adapters:
        if self.merged: self.unmerge()
        return self.base_layer(x, *args, **kwargs)
    if self.merged:
        raise RuntimeError("Input-conditioned MoELoRA cannot be statically merged")
    result = self.base_layer(x, *args, **kwargs)
    result_dtype = result.dtype
    for adapter in self.active_adapters:
        if adapter not in self.lora_A:
            continue
        if self.use_dora[adapter]:
            raise RuntimeError("MoELoRA does not support DoRA")
        a = self.lora_A[adapter].weight
        b = self.lora_B[adapter].weight
        rank = int(a.shape[0])
        experts = int(self.moelora_num_experts)
        if rank % experts:
            raise RuntimeError("Total LoRA rank must divide evenly across experts")
        per = rank // experts
        x_cast = self._cast_input_dtype(x, a.dtype)
        dropped = self.lora_dropout[adapter](x_cast)
        probabilities = torch.softmax(self.lora_router[adapter](x_cast), dim=-1)
        routed = torch.zeros_like(result, dtype=a.dtype)
        for index in range(experts):
            sl = slice(index * per, (index + 1) * per)
            expert = torch.nn.functional.linear(
                torch.nn.functional.linear(dropped, a[sl]), b[:, sl]
            )
            routed = routed + expert * probabilities[..., index].unsqueeze(-1)
        result = result + routed * self.scaling[adapter]
        self.moelora_last_router_probabilities = probabilities.detach()
        self.moelora_forward_calls += 1
    return result.to(result_dtype)


def enable_moelora_patch(num_experts: int = NUM_EXPERTS) -> None:
    """Patch PEFT construction before model/adapters are loaded."""
    global _PATCHED, _ORIGINAL_UPDATE, _ORIGINAL_FORWARD
    if num_experts != NUM_EXPERTS:
        raise ValueError("Formal CoIN configuration uses four experts")
    if _PATCHED:
        return
    _ORIGINAL_UPDATE = PeftLoraLinear.update_layer
    _ORIGINAL_FORWARD = PeftLoraLinear.forward

    def update(self, adapter_name, r, lora_alpha, lora_dropout,
               init_lora_weights, use_rslora, use_dora=False, lora_bias=False):
        _ORIGINAL_UPDATE(self, adapter_name, r, lora_alpha, lora_dropout,
                         init_lora_weights, use_rslora, use_dora, lora_bias)
        if int(r) != TOTAL_RANK:
            raise RuntimeError(f"MoELoRA total rank must be {TOTAL_RANK}, found {r}")
        if int(r) % num_experts:
            raise RuntimeError("Total rank is not divisible by expert count")
        if not hasattr(self, "lora_router"):
            self.lora_router = nn.ModuleDict()
        router = nn.Linear(self.in_features, num_experts, bias=False,
                           device=self.lora_A[adapter_name].weight.device,
                           dtype=self.lora_A[adapter_name].weight.dtype)
        nn.init.kaiming_uniform_(router.weight, a=math.sqrt(5))
        self.lora_router[adapter_name] = router
        self.moelora_num_experts = num_experts
        self.moelora_forward_calls = 0
        self.moelora_last_router_probabilities = None

    PeftLoraLinear.update_layer = update
    PeftLoraLinear.forward = _dense_moe_forward
    _PATCHED = True


def moelora_runtime_stats(model: torch.nn.Module) -> dict[str, Any]:
    modules = [m for m in model.modules() if isinstance(m, PeftLoraLinear) and hasattr(m, "lora_router")]
    if not modules:
        raise RuntimeError("No patched MoELoRA layers")
    probs = [m.moelora_last_router_probabilities.float().reshape(-1, NUM_EXPERTS)
             for m in modules if m.moelora_last_router_probabilities is not None]
    if probs:
        all_probs = torch.cat(probs)
        entropy = (-(all_probs * all_probs.clamp_min(1e-12).log()).sum(-1).mean()).item()
        utilization = all_probs.mean(0).cpu().tolist()
        normalized = float((all_probs.sum(-1) - 1).abs().max().item())
    else:
        entropy, utilization, normalized = None, None, None
    router_parameters = sum(p.numel() for m in modules for p in m.lora_router.parameters())
    return {"status": "PASS", "lora_layer_count": len(modules), "num_experts": NUM_EXPERTS,
            "rank_per_expert": RANK_PER_EXPERT, "total_expert_rank": TOTAL_RANK,
            "routing_type": "token_level_dense_softmax_all_experts",
            "router_output_dimensions": [NUM_EXPERTS], "router_parameters": router_parameters,
            "min_forward_calls": min(m.moelora_forward_calls for m in modules),
            "routing_entropy": entropy, "mean_expert_utilization": utilization,
            "max_probability_sum_error": normalized}
