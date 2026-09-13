"""O-LoRA primitives ported from official commit 07117e1fc4a5f5ad9308a815a42cee8f46502dc8."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from med_prism.baselines.octopus import (
    OctopusHistoricalLinear,
    canonical_base_weight_name,
    inspect_octopus_stage2_serialization,
    lora_pairs,
)
from scripts.medicalskill_v1_2_baselines.baseline_harness import inspect_adapter

OFFICIAL_REPOSITORY = "https://github.com/cmnfriend/O-LoRA"
OFFICIAL_COMMIT = "07117e1fc4a5f5ad9308a815a42cee8f46502dc8"


def olora_regularizers(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return official sum|A_old A_new^T| and sum current ||A||_2+||B||_2.

    Official code concatenates prior A rows into ``lora_A`` and uses the fresh
    task A as ``loranew_A``.  Wave-1 stores each prior adapter independently;
    summing each frozen block is algebraically identical to concatenation.
    """
    pairs = lora_pairs(model)
    if not pairs:
        raise RuntimeError("No trainable current LoRA pairs")
    reference = pairs[0][1]
    orthogonal = reference.new_zeros(())
    secondary = reference.new_zeros(())
    historical_blocks = 0
    modules = {canonical_base_weight_name(module_name + ".weight"): module
               for module_name, module in model.named_modules()
               if isinstance(module, OctopusHistoricalLinear)}
    for name, current_a, current_b in pairs:
        secondary = secondary + torch.norm(current_a, p=2) + torch.norm(current_b, p=2)
        module = modules.get(name)
        if isinstance(module, OctopusHistoricalLinear):
            for old_a in module.extras_A["default"].values():
                orthogonal = orthogonal + torch.abs(old_a.linear.weight @ current_a.T).sum()
                historical_blocks += 1
    return orthogonal, secondary, {
        "lora_layer_count": len(pairs),
        "historical_a_block_count": historical_blocks,
        "official_loss": "sum(abs(A_old @ A_current.T)) + lambda_2*sum(L2(current_A,current_B))",
    }


def inspect_olora_serialization(checkpoint: Path) -> dict[str, Any]:
    value = inspect_octopus_stage2_serialization(checkpoint)
    value["semantics"] = "fresh current-task LoRA only; frozen O-LoRA history excluded"
    return value


def historical_a_hashes(checkpoints: list[Path]) -> list[dict[str, Any]]:
    result = []
    for task, checkpoint in enumerate(checkpoints, 1):
        info = inspect_adapter(checkpoint)
        state = load_file(info["adapter_weights"], device="cpu")
        count = sum(".lora_A." in key for key in state)
        if not count:
            raise RuntimeError(f"No historical LoRA-A tensors: {checkpoint}")
        result.append({"task": task, "checkpoint": info["checkpoint"],
                       "adapter_weights_sha256": info["adapter_weights_sha256"],
                       "lora_a_tensor_count": count})
    return result
