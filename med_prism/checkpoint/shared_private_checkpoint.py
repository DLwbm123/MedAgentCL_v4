from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from med_prism.adapters.injection import (
    add_task_bank,
    collect_language_targets,
    inject_shared_private_wrappers,
    iter_shared_private_wrappers,
)
from med_prism.config import MODEL_ID, MODEL_REVISION, SharedPrivateConfig

from .rank1_checkpoint import sha256_file


FORMAT_VERSION = "med_prism_shared_private_v1"
SHARED_MANIFEST = "shared_manifest.json"
PRIVATE_MANIFEST = "private_manifest.json"
SHARED_WEIGHTS = "shared_adapter.safetensors"
PRIVATE_WEIGHTS = "private_adapter.safetensors"


def _unwrap(model):
    return getattr(model, "module", model)


def _wrappers(model):
    result = list(iter_shared_private_wrappers(_unwrap(model)))
    if len(result) != 72:
        raise RuntimeError(f"Expected 72 shared/private wrappers, got {len(result)}")
    return result


def _schema(state: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
        for key, tensor in sorted(state.items())
    ]


def _write_component(
    directory: Path,
    *,
    component_type: str,
    weights_name: str,
    state: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not state:
        raise RuntimeError(f"Cannot save empty {component_type} component")
    directory.mkdir(parents=True, exist_ok=True)
    weights_path = directory / weights_name
    save_file(
        {key: tensor.detach().cpu().contiguous() for key, tensor in state.items()},
        str(weights_path),
        metadata={"format": "pt", "method": "med_prism_shared_private"},
    )
    manifest = {
        "format_version": FORMAT_VERSION,
        "method": "med_prism_shared_private",
        "component_type": component_type,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "target_modules": metadata.pop("target_modules"),
        "target_module_hash": metadata.pop("target_module_hash"),
        "tensor_schema": _schema(state),
        "tensor_count": len(state),
        "dtype": sorted(
            {str(tensor.dtype).replace("torch.", "") for tensor in state.values()}
        ),
        "weights_file": weights_name,
        "weights_sha256": sha256_file(weights_path),
        "merged": False,
        **metadata,
    }
    manifest_name = SHARED_MANIFEST if component_type == "shared" else PRIVATE_MANIFEST
    (directory / manifest_name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def save_shared_component(
    model,
    directory: str | Path,
    *,
    config: SharedPrivateConfig,
) -> dict[str, Any]:
    model = _unwrap(model)
    wrappers = _wrappers(model)
    inventory = collect_language_targets(model)
    state = {}
    for name, wrapper in wrappers:
        state[f"{name}.shared.A"] = wrapper.shared.A
        state[f"{name}.shared.B"] = wrapper.shared.B
    return _write_component(
        Path(directory),
        component_type="shared",
        weights_name=SHARED_WEIGHTS,
        state=state,
        metadata={
            "task_id": config.current_task_id,
            "stage": f"after_task_{config.current_task_id}",
            "rank": config.shared_rank,
            "experts_per_task": None,
            "alpha": config.shared_alpha,
            "scaling": config.shared_scaling,
            "private_adapter_type": config.private_adapter_type,
            "source_checkpoint": config.shared_source_manifest,
            "trainable": True,
            "frozen": False,
            "target_modules": list(inventory.names),
            "target_module_hash": inventory.sha256,
        },
    )


def save_private_component(
    model,
    directory: str | Path,
    *,
    config: SharedPrivateConfig,
) -> dict[str, Any]:
    model = _unwrap(model)
    wrappers = _wrappers(model)
    inventory = collect_language_targets(model)
    state = {}
    for name, wrapper in wrappers:
        if config.private_adapter_type == "rank1_expert_bank":
            for expert in wrapper.task_experts(config.current_task_id):
                key = wrapper.expert_key(config.current_task_id, expert.expert_id)
                state[f"{name}.experts.{key}.A"] = expert.A
                state[f"{name}.experts.{key}.B"] = expert.B
        else:
            adapter = wrapper.task_adapter(config.current_task_id)
            if adapter is None:
                raise RuntimeError(
                    f"Missing rank-16 private adapter for task {config.current_task_id}"
                )
            key = wrapper.task_adapter_key(config.current_task_id)
            state[f"{name}.task_adapters.{key}.A"] = adapter.A
            state[f"{name}.task_adapters.{key}.B"] = adapter.B
    return _write_component(
        Path(directory),
        component_type="private",
        weights_name=PRIVATE_WEIGHTS,
        state=state,
        metadata={
            "task_id": config.current_task_id,
            "rank": config.effective_private_rank,
            "experts_per_task": config.private_experts_per_task,
            "private_adapter_type": config.private_adapter_type,
            "explicit_rank1_expert_decomposition": (
                config.private_adapter_type == "rank1_expert_bank"
            ),
            "alpha": config.private_alpha,
            "scaling": config.private_scaling,
            "source_checkpoint": None,
            "orthogonal_reference_manifests": list(config.private_source_manifests),
            "trainable": True,
            "frozen": False,
            "target_modules": list(inventory.names),
            "target_module_hash": inventory.sha256,
        },
    )


def save_shared_private_components(model, config: SharedPrivateConfig) -> dict:
    return {
        "shared": save_shared_component(model, config.shared_output_dir, config=config),
        "private": save_private_component(
            model, config.private_output_dir, config=config
        ),
    }


def _read_manifest(path: str | Path, expected_name: str) -> tuple[Path, dict]:
    path = Path(path)
    if path.name != expected_name or not path.is_file():
        raise FileNotFoundError(f"Explicit {expected_name} path is required: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _validate(
    model,
    manifest_path: Path,
    manifest: dict,
    *,
    component_type: str,
) -> tuple[Path, dict[str, torch.Tensor]]:
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported shared/private format_version")
    if manifest.get("method") != "med_prism_shared_private":
        raise ValueError("Component method mismatch")
    if manifest.get("component_type") != component_type:
        raise ValueError("Component type mismatch")
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError("Component model ID mismatch")
    if manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("Component model revision mismatch")
    if manifest.get("merged") is not False:
        raise ValueError("Only unmerged components are accepted")
    inventory = collect_language_targets(model)
    if manifest.get("target_module_hash") != inventory.sha256:
        raise ValueError("Component target module hash mismatch")
    if manifest.get("target_modules") != list(inventory.names):
        raise ValueError("Component target module list mismatch")
    weights = manifest_path.parent / manifest.get("weights_file", "")
    if not weights.is_file():
        raise FileNotFoundError(f"Missing component weights: {weights}")
    if sha256_file(weights) != manifest.get("weights_sha256"):
        raise ValueError("Component weights SHA256 mismatch")
    state = load_file(str(weights))
    schema = {item["key"]: item for item in manifest.get("tensor_schema", [])}
    if len(schema) != manifest.get("tensor_count") or set(schema) != set(state):
        raise ValueError("Component tensor schema key mismatch")
    for key, tensor in state.items():
        if list(tensor.shape) != schema[key].get("shape"):
            raise ValueError(f"Component tensor schema shape mismatch: {key}")
        if str(tensor.dtype).replace("torch.", "") != schema[key].get("dtype"):
            raise ValueError(f"Component tensor schema dtype mismatch: {key}")
    return weights, state


def _copy_state(model, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(_unwrap(model).named_parameters())
    for key, tensor in state.items():
        if key not in parameters:
            raise ValueError(f"Component parameter is absent from model: {key}")
        if tuple(parameters[key].shape) != tuple(tensor.shape):
            raise ValueError(f"Component parameter shape mismatch: {key}")
        parameters[key].data.copy_(
            tensor.to(device=parameters[key].device, dtype=parameters[key].dtype)
        )


def load_shared_component(model, manifest_path: str | Path) -> dict:
    model = _unwrap(model)
    if getattr(model, "_med_prism_shared_loaded", False):
        raise ValueError("Duplicate shared component")
    path, manifest = _read_manifest(manifest_path, SHARED_MANIFEST)
    _, state = _validate(model, path, manifest, component_type="shared")
    _copy_state(model, state)
    for _, wrapper in _wrappers(model):
        wrapper.shared.reset_anchor()
    model._med_prism_shared_loaded = True
    model.med_prism_shared_manifest = str(path)
    return manifest


def load_private_component(
    model,
    manifest_path: str | Path,
    *,
    trainable: bool,
) -> dict:
    model = _unwrap(model)
    path, manifest = _read_manifest(manifest_path, PRIVATE_MANIFEST)
    task_id = int(manifest.get("task_id", 0))
    if task_id <= 0:
        raise ValueError("Private component requires positive task_id")
    existing = set(_wrappers(model)[0][1].task_ids)
    if task_id in existing:
        raise ValueError(f"Duplicate private task component: {task_id}")
    add_task_bank(
        model,
        task_id=task_id,
        experts_per_task=int(manifest["experts_per_task"]),
        private_rank=int(manifest["rank"]),
        alpha=float(manifest["alpha"]),
        trainable=trainable,
    )
    _, state = _validate(model, path, manifest, component_type="private")
    _copy_state(model, state)
    model.med_prism_private_manifests = {
        **getattr(model, "med_prism_private_manifests", {}),
        task_id: str(path),
    }
    return manifest


def compose_shared_private(
    model,
    *,
    shared_manifest: str | Path,
    private_manifests: list[str | Path],
) -> dict:
    shared_path, shared = _read_manifest(shared_manifest, SHARED_MANIFEST)
    private_headers = [
        _read_manifest(path, PRIVATE_MANIFEST)[1] for path in private_manifests
    ]
    private_types = {
        item.get("private_adapter_type", "rank1_expert_bank")
        for item in private_headers
    }
    if len(private_types) != 1:
        raise ValueError(f"Mixed private adapter types: {sorted(private_types)}")
    private_adapter_type = next(iter(private_types))
    inject_shared_private_wrappers(
        model,
        shared_rank=int(shared["rank"]),
        shared_alpha=float(shared["alpha"]),
        shared_lr_scale=1.0,
        dropout=0.0,
        shared_trainable=False,
        private_adapter_type=private_adapter_type,
    )
    load_shared_component(model, shared_path)
    private = [
        load_private_component(model, path, trainable=False)
        for path in private_manifests
    ]
    task_ids = sorted(int(item["task_id"]) for item in private)
    for _, wrapper in _wrappers(model):
        wrapper.shared.set_trainable(False)
        wrapper.set_active_tasks(task_ids)
    manifest = {
        "method": "med_prism_shared_private",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "shared_manifest": str(shared_path),
        "shared_load_count": 1,
        "private_manifests": [str(Path(path)) for path in private_manifests],
        "private_task_ids": task_ids,
        "private_adapter_type": private_adapter_type,
        "merged": False,
    }
    model.med_prism_active_component_manifest = manifest
    return manifest
