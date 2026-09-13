from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from med_prism.adapters.injection import (
    add_task_bank,
    collect_language_targets,
    inject_rank1_wrappers,
    iter_rank1_wrappers,
)
from med_prism.config import MODEL_ID, MODEL_REVISION, Rank1BankConfig
from med_prism.experts.expert_bank import summarize_expert_bank


SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = "med_prism_rank1_v1"
WEIGHTS_NAME = "rank1_adapter.safetensors"
MANIFEST_NAME = "rank1_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap(model):
    return getattr(model, "module", model)


def _adapter_state(model) -> dict[str, torch.Tensor]:
    state = {}
    for name, wrapper in iter_rank1_wrappers(_unwrap(model)):
        for task_id in wrapper.task_ids:
            for expert in wrapper.task_experts(task_id):
                key = wrapper.expert_key(task_id, expert.expert_id)
                state[f"{name}.experts.{key}.A"] = expert.A.detach().cpu().contiguous()
                state[f"{name}.experts.{key}.B"] = expert.B.detach().cpu().contiguous()
    if not state:
        raise RuntimeError("Cannot save an empty rank-1 expert bank")
    return state


def save_rank1_checkpoint(
    model,
    save_directory: str | Path,
    *,
    config: Rank1BankConfig | None = None,
) -> dict[str, Any]:
    model = _unwrap(model)
    config = config or getattr(model, "med_prism_config", None)
    if config is None:
        raise RuntimeError("Model is missing explicit med_prism_config")
    config.validate()
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    inventory = collect_language_targets(model)
    wrappers = list(iter_rank1_wrappers(model))
    if len(wrappers) != 72:
        raise RuntimeError(f"Expected 72 wrappers, got {len(wrappers)}")

    state = _adapter_state(model)
    weights_path = save_directory / WEIGHTS_NAME
    save_file(state, str(weights_path), metadata={"format": "pt", "method": "med_prism_rank1"})
    records = summarize_expert_bank(model)
    tasks = sorted({record.task_id for record in records})
    task_banks = []
    for task_id in tasks:
        task_records = [record for record in records if record.task_id == task_id]
        ranks = {record.per_task_rank for record in task_records}
        alphas = {record.alpha for record in task_records}
        scalings = {record.scaling for record in task_records}
        if len(ranks) != 1 or len(alphas) != 1 or len(scalings) != 1:
            raise RuntimeError(f"Inconsistent task metadata for task {task_id}")
        task_banks.append({
            "task_id": task_id,
            "experts_per_task": ranks.pop(),
            "alpha": alphas.pop(),
            "scaling": scalings.pop(),
            "frozen": all(record.frozen for record in task_records),
            "active": all(record.active for record in task_records),
        })

    parameter_names = dict(model.named_parameters())
    tensor_schema = []
    for key, tensor in sorted(state.items()):
        if key not in parameter_names:
            raise RuntimeError(f"Adapter state key is not a model parameter: {key}")
        parts = key.split(".experts.", 1)[1].split(".")
        expert_key, side = parts[0], parts[1]
        task_part, expert_part = expert_key.split("__")
        tensor_schema.append({
            "key": key,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "task_id": int(task_part.split("_")[1]),
            "expert_id": int(expert_part.split("_")[1]),
            "side": side,
        })

    dtype_counts: dict[str, int] = {}
    for tensor in state.values():
        dtype = str(tensor.dtype).replace("torch.", "")
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method": "med_prism_rank1",
        "backbone": MODEL_ID,
        "immutable_revision": MODEL_REVISION,
        "ms_swift_version": "4.4.1",
        "ms_swift_commit": subprocess.check_output(
            ["git", "rev-parse", "v4.4.1^{commit}"], text=True
        ).strip(),
        "current_task_id": config.current_task_id,
        "task_ids": tasks,
        "included_task_banks": tasks,
        "task_banks": task_banks,
        "experts_per_task": config.experts_per_task,
        "alpha": config.alpha,
        "per_task_rank": config.experts_per_task,
        "task_scaling": {
            str(item["task_id"]): item["scaling"] for item in task_banks
        },
        "orth_loss_type": config.orth_loss_type,
        "orth_lambda": config.orth_lambda,
        "orth_eps": config.orth_eps,
        "target_modules": list(inventory.names),
        "target_module_hash": inventory.sha256,
        "wrapper_count": len(wrappers),
        "expert_count": len(records),
        "expert_tensor_count": len(state),
        "old_expert_count": sum(record.task_id != config.current_task_id for record in records),
        "current_expert_count": sum(record.task_id == config.current_task_id for record in records),
        "old_experts_frozen": all(
            record.frozen for record in records if record.task_id != config.current_task_id
        ),
        "current_experts_trainable": all(
            not record.frozen for record in records if record.task_id == config.current_task_id
        ),
        "active_tasks": list(wrappers[0][1].active_tasks),
        "merge_state": "unmerged",
        "source_checkpoint": config.source_manifest,
        "dataset_manifest": config.dataset_manifest,
        "dataset_sha256": config.dataset_sha256,
        "seed": config.seed,
        "model_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "adapter_tensor_dtypes": dtype_counts,
        "weights_file": WEIGHTS_NAME,
        "safetensors_sha256": sha256_file(weights_path),
        "tensor_schema": tensor_schema,
        "config": config.to_dict(),
    }
    manifest_path = save_directory / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_manifest(
    manifest: dict[str, Any],
    *,
    expected_revision: str,
    expected_target_hash: str,
) -> None:
    if manifest.get("method") != "med_prism_rank1":
        raise ValueError("Checkpoint is not a Med-PRISM rank-1 bank")
    if manifest.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema")
    if manifest.get("immutable_revision") != expected_revision:
        raise ValueError("Backbone revision mismatch")
    if manifest.get("target_module_hash") != expected_target_hash:
        raise ValueError("Target module hash mismatch")
    task_ids = manifest.get("task_ids", [])
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate task IDs")
    banks = manifest.get("task_banks", [])
    if {item["task_id"] for item in banks} != set(task_ids):
        raise ValueError("Task bank metadata does not match task_ids")
    if manifest.get("included_task_banks") != task_ids:
        raise ValueError("Included task banks do not match task_ids")
    wrapper_count = int(manifest.get("wrapper_count", 0))
    if wrapper_count != 72 or len(manifest.get("target_modules", [])) != wrapper_count:
        raise ValueError("Checkpoint must describe exactly 72 language wrappers")
    expected_experts = wrapper_count * sum(
        int(bank["experts_per_task"]) for bank in banks
    )
    if manifest.get("expert_count") != expected_experts:
        raise ValueError("Expert count does not match task bank metadata")
    schema = manifest.get("tensor_schema", [])
    if manifest.get("expert_tensor_count") != expected_experts * 2:
        raise ValueError("Expert tensor count does not match task bank metadata")
    if len(schema) != expected_experts * 2:
        raise ValueError("Tensor schema is incomplete")
    identities = []
    for item in schema:
        key = item["key"]
        if ".experts." not in key or item["side"] not in {"A", "B"}:
            raise ValueError("Invalid rank-1 tensor schema entry")
        module_name = key.split(".experts.", 1)[0]
        identities.append(
            (module_name, item["task_id"], item["expert_id"], item["side"])
        )
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate task/expert tensor identity")
    side_pairs: dict[tuple[str, int, int], set[str]] = {}
    for module_name, task_id, expert_id, side in identities:
        side_pairs.setdefault((module_name, task_id, expert_id), set()).add(side)
    if any(sides != {"A", "B"} for sides in side_pairs.values()):
        raise ValueError("Every explicit expert must contain one A and one B tensor")
    for bank in banks:
        expected = bank["alpha"] / bank["experts_per_task"]
        if abs(float(bank["scaling"]) - expected) > 1e-12:
            raise ValueError("Manifest scaling is inconsistent with alpha/per-task rank")
    if manifest.get("merge_state") != "unmerged":
        raise ValueError("Merged checkpoints are not accepted")


def load_rank1_checkpoint(
    model,
    manifest_path: str | Path,
    *,
    expected_revision: str = MODEL_REVISION,
    trainable_task_id: int | None = None,
) -> dict[str, Any]:
    model = _unwrap(model)
    manifest_path = Path(manifest_path)
    if manifest_path.name != MANIFEST_NAME or not manifest_path.is_file():
        raise FileNotFoundError(
            f"An explicit {MANIFEST_NAME} path is required: {manifest_path}"
        )
    inventory = collect_language_targets(model)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(
        manifest,
        expected_revision=expected_revision,
        expected_target_hash=inventory.sha256,
    )
    weights_path = manifest_path.parent / manifest["weights_file"]
    if sha256_file(weights_path) != manifest["safetensors_sha256"]:
        raise ValueError("Safetensors hash mismatch")

    config = Rank1BankConfig.from_dict(manifest["config"])
    inject_rank1_wrappers(model, dropout=config.dropout)
    for bank in manifest["task_banks"]:
        add_task_bank(
            model,
            task_id=int(bank["task_id"]),
            experts_per_task=int(bank["experts_per_task"]),
            alpha=float(bank["alpha"]),
            trainable=int(bank["task_id"]) == trainable_task_id,
        )
    for _, wrapper in iter_rank1_wrappers(model):
        wrapper.set_active_tasks(manifest["active_tasks"])
        if trainable_task_id is not None:
            wrapper.freeze_except(trainable_task_id)
        else:
            for task_id in wrapper.task_ids:
                for expert in wrapper.task_experts(task_id):
                    expert.set_trainable(False)

    state = load_file(str(weights_path))
    schema = {item["key"]: item for item in manifest["tensor_schema"]}
    if set(state) != set(schema):
        missing = sorted(set(schema) - set(state))
        extra = sorted(set(state) - set(schema))
        raise ValueError(f"Tensor schema mismatch: missing={missing[:3]}, extra={extra[:3]}")
    parameters = dict(model.named_parameters())
    for key, tensor in state.items():
        if key not in parameters:
            raise ValueError(f"Checkpoint parameter is absent from model: {key}")
        expected_shape = tuple(schema[key]["shape"])
        if tuple(tensor.shape) != expected_shape or tuple(parameters[key].shape) != expected_shape:
            raise ValueError(f"Shape mismatch for {key}")
        parameters[key].data.copy_(
            tensor.to(device=parameters[key].device, dtype=parameters[key].dtype)
        )
    model.med_prism_source_manifest = str(manifest_path)
    model.med_prism_loaded_manifest = manifest
    return manifest
