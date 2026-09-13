"""Faithful Octopus gradient-space regularization primitives."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from peft.tuners.lora.layer import Linear as PeftLoraLinear
from safetensors.torch import load_file
from torch import nn

from scripts.medicalskill_v1_2_baselines.baseline_harness import (
    inspect_adapter,
    load_json,
    sha256_file,
)


class FrozenLoRAExtra(nn.Module):
    """Frozen historical LoRA projection used by the official Octopus forward."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


class OctopusHistoricalLinear(PeftLoraLinear):
    """PEFT Linear with the official frozen ``extras_A/extras_B`` branches."""

    def init_octopus_extras(self, module_name: str) -> None:
        adapter_names = list(self.r.keys())
        if adapter_names != ["default"]:
            raise RuntimeError(f"Octopus requires only the default adapter: {adapter_names}")
        if self.use_dora["default"]:
            raise RuntimeError("Official Octopus frozen extras do not support DoRA")
        self.adapter_name = "default"
        self.module_name = module_name
        self.extras_A = nn.ModuleDict({"default": nn.ModuleDict()})
        self.extras_B = nn.ModuleDict({"default": nn.ModuleDict()})
        self.octopus_extra_forward_calls = 0

    def add_lora_extra(
        self, lora_a_extra: torch.Tensor, lora_b_extra: torch.Tensor, index: int
    ) -> None:
        current_a = self.lora_A["default"]
        current_b = self.lora_B["default"]
        if tuple(lora_a_extra.shape) != tuple(current_a.weight.shape):
            raise RuntimeError(f"Historical LoRA-A shape mismatch in {self.module_name}")
        if tuple(lora_b_extra.shape) != tuple(current_b.weight.shape):
            raise RuntimeError(f"Historical LoRA-B shape mismatch in {self.module_name}")
        linear_a = nn.Linear(
            current_a.in_features,
            current_a.out_features,
            bias=False,
            device=current_a.weight.device,
            dtype=current_a.weight.dtype,
        )
        linear_b = nn.Linear(
            current_b.in_features,
            current_b.out_features,
            bias=False,
            device=current_b.weight.device,
            dtype=current_b.weight.dtype,
        )
        linear_a.weight = nn.Parameter(
            lora_a_extra.detach()
            .to(device=current_a.weight.device, dtype=current_a.weight.dtype)
            .clone(),
            requires_grad=False,
        )
        linear_b.weight = nn.Parameter(
            lora_b_extra.detach()
            .to(device=current_b.weight.device, dtype=current_b.weight.dtype)
            .clone(),
            requires_grad=False,
        )
        key = str(index)
        self.extras_A["default"][key] = FrozenLoRAExtra(linear_a)
        self.extras_B["default"][key] = FrozenLoRAExtra(linear_b)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # This intentionally follows official swift/tuners/base.py:71-105.
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs
            )
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            result_dtype = result.dtype
            if list(self.active_adapters) != ["default"]:
                raise RuntimeError(
                    f"Octopus requires the default active adapter: {self.active_adapters}"
                )
            lora_a = self.lora_A["default"]
            lora_b = self.lora_B["default"]
            dropout = self.lora_dropout["default"]
            scaling = self.scaling["default"]
            x = self._cast_input_dtype(x, lora_a.weight.dtype)
            x_dropout = dropout(x)
            result = result + lora_b(lora_a(x_dropout)) * scaling
            for index in range(len(self.extras_B["default"])):
                key = str(index)
                result = result + self.extras_B["default"][key](
                    self.extras_A["default"][key](x_dropout)
                ) * scaling
            self.octopus_extra_forward_calls += 1
            result = result.to(result_dtype)
        return result


def _adapter_lora_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    info = inspect_adapter(path.resolve())
    state = {
        key: value
        for key, value in load_file(info["adapter_weights"], device="cpu").items()
        if ".lora_A." in key or ".lora_B." in key
    }
    if not state:
        raise RuntimeError(f"No LoRA tensors in historical adapter: {path}")
    return state, info


def attach_historical_stage2_extras(
    model: torch.nn.Module, adapter_paths: Iterable[Path]
) -> dict[str, Any]:
    """Attach every prior Stage-2 adapter as an official frozen forward extra."""
    paths = [Path(path).resolve() for path in adapter_paths]
    if not paths:
        raise ValueError("Stage-2 after Task 1 requires historical Stage-2 extras")
    historical = [_adapter_lora_state(path) for path in paths]
    expected_by_history = []
    for state, _ in historical:
        expected_by_history.append(
            {key.split(".lora_A.", 1)[0] for key in state if ".lora_A." in key}
        )
    attached_names = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, PeftLoraLinear):
            continue
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue
        for expected in expected_by_history:
            if name not in expected:
                raise RuntimeError(f"Historical adapter missing LoRA layer: {name}")
        module.__class__ = OctopusHistoricalLinear
        module.init_octopus_extras(name)
        for index, (state, _) in enumerate(historical):
            module.add_lora_extra(
                state[f"{name}.lora_A.weight"],
                state[f"{name}.lora_B.weight"],
                index,
            )
        attached_names.append(name)
    if not attached_names:
        raise RuntimeError("No PEFT LoRA Linear modules received historical extras")
    for index, expected in enumerate(expected_by_history):
        if set(attached_names) != expected:
            missing = sorted(expected - set(attached_names))
            unused = sorted(set(attached_names) - expected)
            raise RuntimeError(
                f"Historical adapter {index + 1} layer mismatch: "
                f"missing={missing[:5]}, unused={unused[:5]}"
            )
    extra_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if ".extras_A." in name or ".extras_B." in name
    ]
    if len(extra_parameters) != 2 * len(attached_names) * len(paths):
        raise RuntimeError("Unexpected historical frozen-extra tensor count")
    if any(parameter.requires_grad for parameter in extra_parameters):
        raise RuntimeError("Historical Stage-2 extras must be frozen")
    return {
        "semantics": "official_frozen_extras_A_extras_B",
        "historical_adapter_count": len(paths),
        "lora_layer_count": len(attached_names),
        "frozen_extra_tensor_count": len(extra_parameters),
        "trainable_extra_tensor_count": 0,
        "historical_adapters": [
            {
                "task": index + 1,
                "role": "stage2_frozen_forward_extra",
                "checkpoint": info["checkpoint"],
                "adapter_weights_sha256": info["adapter_weights_sha256"],
            }
            for index, (_, info) in enumerate(historical)
        ],
    }


def historical_extra_runtime_stats(model: torch.nn.Module) -> dict[str, Any]:
    modules = [
        module
        for module in model.modules()
        if isinstance(module, OctopusHistoricalLinear)
    ]
    if not modules:
        raise RuntimeError("No Octopus historical-extra modules are active")
    extra_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if ".extras_A." in name or ".extras_B." in name
    ]
    calls = [int(module.octopus_extra_forward_calls) for module in modules]
    history_counts = [len(module.extras_B["default"]) for module in modules]
    return {
        "status": "PASS",
        "lora_layer_count": len(modules),
        "historical_adapter_count": min(history_counts),
        "history_count_consistent": len(set(history_counts)) == 1,
        "min_forward_call_count": min(calls),
        "all_historical_extras_active_in_forward": min(calls) > 0,
        "frozen_extra_tensor_count": len(extra_parameters),
        "trainable_extra_tensor_count": sum(
            parameter.requires_grad for parameter in extra_parameters
        ),
    }


def inspect_octopus_stage2_serialization(checkpoint: Path) -> dict[str, Any]:
    info = inspect_adapter(checkpoint.resolve())
    state = load_file(info["adapter_weights"], device="cpu")
    keys = sorted(state)
    current = [key for key in keys if ".lora_A." in key or ".lora_B." in key]
    extras = [
        key
        for key in keys
        if ".extras_A." in key or ".extras_B." in key or "octopus_extra" in key
    ]
    if extras:
        raise RuntimeError(f"Historical frozen extras leaked into checkpoint: {extras[:5]}")
    if len(current) != len(keys) or not current:
        raise RuntimeError("Stage-2 checkpoint is not a current-LoRA-only adapter")
    return {
        "status": "PASS",
        "checkpoint": info["checkpoint"],
        "adapter_weights_sha256": info["adapter_weights_sha256"],
        "serialized_current_lora_tensor_count": len(current),
        "serialized_historical_extra_tensor_count": 0,
        "contains_only_current_default_lora": True,
        "semantics": "current_trainable_default_lora_only; frozen forward extras excluded",
    }


def canonical_base_weight_name(name: str) -> str:
    value = name
    while value.startswith("module."):
        value = value[len("module.") :]
    while value.startswith("base_model.model."):
        value = value[len("base_model.model.") :]
    for marker in (".lora_B.", ".lora_A."):
        if marker in value:
            value = value.split(marker, 1)[0] + ".weight"
    return value


def lora_pairs(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Parameter, torch.nn.Parameter]]:
    named = dict(model.named_parameters())
    result = []
    for name, parameter_b in named.items():
        if ".lora_B." not in name and not name.endswith(".lora_B.weight"):
            continue
        name_a = name.replace(".lora_B.", ".lora_A.")
        if name.endswith(".lora_B.weight"):
            name_a = name[: -len(".lora_B.weight")] + ".lora_A.weight"
        parameter_a = named.get(name_a)
        if parameter_a is None:
            raise RuntimeError(f"Missing matching LoRA A parameter for {name}")
        result.append((canonical_base_weight_name(name), parameter_a, parameter_b))
    if not result:
        raise RuntimeError("No LoRA A/B pairs found")
    return result


def octopus_layer_penalty(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    base_gradient: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Original |trace(normalize(B) normalize(A G^T))| without a dense trace.

    The official implementation detaches ``A @ G.T`` so this term updates B only.
    """
    if lora_a.ndim != 2 or lora_b.ndim != 2 or base_gradient.ndim != 2:
        raise ValueError("Octopus expects matrix-valued A, B, and base gradients")
    if (
        lora_a.shape[1] != base_gradient.shape[1]
        or lora_b.shape[0] != base_gradient.shape[0]
        or lora_b.shape[1] != lora_a.shape[0]
    ):
        raise ValueError(
            f"Incompatible shapes A={tuple(lora_a.shape)}, "
            f"B={tuple(lora_b.shape)}, G={tuple(base_gradient.shape)}"
        )
    normalized_b = lora_b / (torch.linalg.vector_norm(lora_b) + eps)
    gradients_a = (lora_a @ base_gradient.transpose(0, 1)).detach()
    normalized_gradients_a = gradients_a / (
        torch.linalg.vector_norm(gradients_a) + eps
    )
    return torch.sum(normalized_b * normalized_gradients_a.transpose(0, 1)).abs()


def octopus_regularizers(
    model: torch.nn.Module,
    merged_gradients: dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    pairs = lora_pairs(model)
    device = pairs[0][2].device
    orthogonal = torch.zeros((), device=device, dtype=torch.float32)
    norm = torch.zeros((), device=device, dtype=torch.float32)
    missing = []
    used = []
    for key, parameter_a, parameter_b in pairs:
        gradient = merged_gradients.get(key)
        if gradient is None:
            missing.append(key)
            continue
        gradient = gradient.to(
            device=parameter_b.device, dtype=parameter_b.dtype, non_blocking=True
        )
        orthogonal = orthogonal + octopus_layer_penalty(
            parameter_a, parameter_b, gradient, eps=eps
        ).float()
        norm = norm + torch.linalg.vector_norm(parameter_b).float()
        used.append(key)
    if missing:
        raise RuntimeError(
            "Historical gradients do not cover all LoRA B layers: "
            + json.dumps(missing[:10])
        )
    return orthogonal, norm, {
        "lora_layer_count": len(pairs),
        "gradient_layer_count": len(merged_gradients),
        "used_gradient_keys": used,
        "unused_gradient_keys": sorted(set(merged_gradients) - set(used)),
    }


def validate_gradient_manifest(
    path: Path,
    *,
    current_dataset_sha256: str | None = None,
    target_modules: Iterable[str] | None = None,
) -> dict[str, Any]:
    value = load_json(path)
    errors = []
    if value.get("status") != "PASS":
        errors.append("status")
    artifact = Path(value.get("gradient_artifact", ""))
    if not artifact.is_file():
        errors.append("artifact_missing")
    elif sha256_file(artifact) != value.get("gradient_artifact_sha256"):
        errors.append("artifact_hash")
    if (
        current_dataset_sha256 is not None
        and value.get("source_current_task_dataset_sha256")
        != current_dataset_sha256
    ):
        errors.append("current_dataset_hash")
    if target_modules is not None and sorted(value.get("target_modules") or []) != sorted(
        target_modules
    ):
        errors.append("target_modules")
    prefix = value.get("historical_stage2_prefix") or []
    if len(prefix) != int(value.get("historical_prefix_length", -1)):
        errors.append("prefix_length")
    for component in prefix:
        info = inspect_adapter(Path(component["checkpoint"]))
        if info["adapter_weights_sha256"] != component.get(
            "adapter_weights_sha256"
        ):
            errors.append(f"historical_adapter_hash_task_{component.get('task')}")
        if component.get("role") != "stage2":
            errors.append(f"historical_role_task_{component.get('task')}")
    if errors:
        raise RuntimeError(f"Invalid gradient manifest {path}: {errors}")
    return value


def load_merged_gradients(
    manifests: Iterable[Path],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    paths = list(manifests)
    if not paths:
        raise ValueError("Stage-2 training after Task 1 requires gradient manifests")
    merged: dict[str, torch.Tensor] = {}
    metadata = []
    expected_keys = None
    for path in paths:
        value = validate_gradient_manifest(path)
        state = {
            canonical_base_weight_name(key): tensor
            for key, tensor in load_file(value["gradient_artifact"], device="cpu").items()
        }
        keys = set(state)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise RuntimeError("Gradient artifacts contain different target layers")
        for key, tensor in state.items():
            tensor = tensor.float()
            merged[key] = tensor if key not in merged else merged[key] + tensor
        metadata.append(
            {
                "manifest": str(path.resolve()),
                "manifest_sha256": sha256_file(path),
                "gradient_artifact": value["gradient_artifact"],
                "gradient_artifact_sha256": value["gradient_artifact_sha256"],
                "historical_prefix_length": value["historical_prefix_length"],
            }
        )
    return merged, {
        "aggregation": "sum_of_historical_prefix_gradient_responses",
        "gradient_manifest_count": len(paths),
        "gradient_tensor_count": len(merged),
        "manifests": metadata,
    }


def stage2_inference_stack(completions: Iterable[Path]) -> list[dict[str, Any]]:
    stack = []
    for expected_task, path in enumerate(completions, 1):
        value = load_json(path)
        if value.get("status") != "PASS" or value.get("stage") != expected_task:
            raise RuntimeError(f"Invalid completion ordering: {path}")
        info = inspect_adapter(Path(value["stage2_checkpoint"]))
        stack.append(
            {
                "task": expected_task,
                "role": "stage2",
                "checkpoint": info["checkpoint"],
                "adapter_weights_sha256": info["adapter_weights_sha256"],
                "parameters": info["adapter_parameters"],
            }
        )
    return stack
