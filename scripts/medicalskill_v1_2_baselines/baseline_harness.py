#!/usr/bin/env python3
"""Shared contracts for formal MedicalSkill-CL-v1.2 baseline runs.

This module is intentionally independent of any continual-learning algorithm.
It validates immutable inputs and records actual checkpoint/runtime properties.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DATASET_VERSION = "MedicalSkill-CL-v1.2-lite-10k1k"
SEED = 42
EVALUATION_PROTOCOL = "lower_triangular_seen_tasks_v1"
TASK_ORDER = [
    "task_01_vqa",
    "task_02_diagnosis_classification",
    "task_03_concept_recognition",
    "task_04_visual_grounding",
    "task_05_reasoning_vqa",
]
SNAPSHOT = (
    Path("/remote-home/wangbomin/huggingface_cache/hub")
    / "models--Qwen--Qwen3-VL-8B-Instruct"
    / "snapshots"
    / MODEL_REVISION
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def nonempty_line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


@dataclass(frozen=True)
class HistoricalMemory:
    raw_examples: bool = False
    features_or_prototypes: bool = False
    statistics_or_masks: bool = False
    historical_parameters: bool = False
    notes: str = ""

    def manifest(self) -> dict[str, Any]:
        value = asdict(self)
        value["privacy_claim"] = (
            "Retains no historical raw examples."
            if not self.raw_examples
            else "Retains historical raw examples."
        )
        value["formally_privacy_preserving"] = False
        return value


def validate_dataset_lock(data_root: Path) -> dict[str, Any]:
    data_root = data_root.resolve()
    acceptance_path = data_root / "audits" / "acceptance_matrix.json"
    lock_path = data_root / "manifests" / "dataset_lock.json"
    if not acceptance_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError("Formal dataset acceptance/lock files are missing")
    acceptance = load_json(acceptance_path)
    lock = load_json(lock_path)
    if acceptance.get("status") != "PASS":
        raise RuntimeError("Dataset acceptance matrix is not PASS")
    if lock.get("dataset_version") != DATASET_VERSION:
        raise RuntimeError(f"Unexpected dataset version: {lock.get('dataset_version')}")
    if int(lock.get("seed", -1)) != SEED:
        raise RuntimeError("Dataset seed differs from the formal seed")
    names = [item.get("task_name") for item in lock.get("tasks", [])]
    if names != TASK_ORDER:
        raise RuntimeError(f"Task order mismatch: {names}")
    selected = data_root / lock["selected_ids_file"]
    if sha256_file(selected) != lock["selected_ids_sha256"]:
        raise RuntimeError("Selected-ID hash mismatch")
    tasks: dict[str, Any] = {}
    total = 0
    for item in lock["tasks"]:
        task_name = item["task_name"]
        tasks[task_name] = {}
        for split in ("train", "test"):
            expected = item["splits"][split]
            path = data_root / expected["file"]
            actual_count = nonempty_line_count(path)
            actual_hash = sha256_file(path)
            if actual_count != int(expected["records"]):
                raise RuntimeError(f"Record-count mismatch: {path}")
            if actual_hash != expected["sha256"]:
                raise RuntimeError(f"Dataset hash mismatch: {path}")
            tasks[task_name][split] = {
                "file": str(path),
                "records": actual_count,
                "sha256": actual_hash,
            }
            total += actual_count
    if int(lock["selected_id_count"]) != total:
        raise RuntimeError(
            f"Selected-ID count mismatch: lock={lock['selected_id_count']} files={total}"
        )
    return {
        "status": "PASS",
        "dataset_version": DATASET_VERSION,
        "data_root": str(data_root),
        "dataset_lock": str(lock_path),
        "dataset_lock_sha256": sha256_file(lock_path),
        "selected_ids_sha256": lock["selected_ids_sha256"],
        "selected_id_count": total,
        "task_order": names,
        "tasks": tasks,
    }


def validate_backbone(revision: str = MODEL_REVISION) -> dict[str, Any]:
    if revision != MODEL_REVISION:
        raise RuntimeError(
            f"Backbone revision must be {MODEL_REVISION}, received {revision}"
        )
    config = SNAPSHOT / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Locked backbone snapshot is incomplete: {SNAPSHOT}")
    return {
        "status": "PASS",
        "model_id": MODEL_ID,
        "model_revision": revision,
        "snapshot": str(SNAPSHOT),
        "config_sha256": sha256_file(config),
    }


def adapter_weight_file(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = checkpoint / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No adapter weights under {checkpoint}")


def _weight_parameter_count(path: Path) -> tuple[int, int]:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        parameters = 0
        tensors = 0
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shape = handle.get_slice(key).get_shape()
                count = 1
                for dimension in shape:
                    count *= int(dimension)
                parameters += count
                tensors += 1
        return parameters, tensors
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    tensors = [value for value in state.values() if hasattr(value, "numel")]
    return sum(int(value.numel()) for value in tensors), len(tensors)


def inspect_adapter(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    config_path = checkpoint / "adapter_config.json"
    weights = adapter_weight_file(checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter config: {checkpoint}")
    config = load_json(config_path)
    parameters, tensors = _weight_parameter_count(weights)
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "adapter_config": str(config_path),
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_weights": str(weights),
        "adapter_weights_sha256": sha256_file(weights),
        "adapter_parameters": parameters,
        "adapter_tensor_count": tensors,
        "checkpoint_bytes": directory_bytes(checkpoint),
        "r": int(config.get("r", -1)),
        "lora_alpha": int(config.get("lora_alpha", -1)),
        "target_modules": sorted(config.get("target_modules") or []),
    }


def require_adapter_contract(
    checkpoint: Path, *, rank: int, target_modules: Iterable[str]
) -> dict[str, Any]:
    value = inspect_adapter(checkpoint)
    expected_targets = sorted(target_modules)
    if value["r"] != rank:
        raise RuntimeError(
            f"Adapter rank mismatch at {checkpoint}: {value['r']} != {rank}"
        )
    if value["target_modules"] != expected_targets:
        raise RuntimeError(
            f"Adapter targets mismatch at {checkpoint}: "
            f"{value['target_modules']} != {expected_targets}"
        )
    return value


def build_run_manifest(
    *,
    method: str,
    method_family: str,
    data_root: Path,
    output_root: Path,
    capacity_policy: dict[str, Any],
    historical_memory: HistoricalMemory,
    method_config: dict[str, Any],
) -> dict[str, Any]:
    dataset = validate_dataset_lock(data_root)
    backbone = validate_backbone()
    immutable = {
        "method": method,
        "method_family": method_family,
        "seed": SEED,
        "dataset": dataset,
        "backbone": backbone,
        "task_order": TASK_ORDER,
        "training_epochs_per_task": 1,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "generation_and_parsers": (
            "scripts.medicalskill_v1_2_med_prism.evaluate_formal_v1_2"
        ),
        "capacity_policy": capacity_policy,
        "historical_memory": historical_memory.manifest(),
        "method_config": method_config,
        "output_root": str(output_root.resolve()),
    }
    return {
        "status": "PASS",
        "format_version": "medicalskill_cl_baseline_run_v1",
        "created_at": utc_now(),
        "immutable_config_sha256": sha256_json(immutable),
        **immutable,
    }


def write_or_validate_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        previous = load_json(path)
        old = dict(previous)
        new = dict(payload)
        old.pop("created_at", None)
        new.pop("created_at", None)
        if old != new:
            raise RuntimeError(f"Existing run manifest differs: {path}")
        return
    atomic_json(path, payload)


def validate_runtime_record(path: Path, expected_category: str | None = None) -> dict[str, Any]:
    value = load_json(path)
    if value.get("status") not in {"PASS", "FAIL"}:
        raise RuntimeError(f"Invalid runtime status: {path}")
    if expected_category and value.get("category") != expected_category:
        raise RuntimeError(f"Runtime category mismatch: {path}")
    required = (
        "start_timestamp",
        "end_timestamp",
        "wall_clock_seconds",
        "gpu_count",
        "gpu_models",
        "command_exit_code",
    )
    if any(key not in value for key in required):
        raise RuntimeError(f"Incomplete runtime record: {path}")
    return value


def trainer_state(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "trainer_state.json"
    if not path.is_file():
        return {"training_steps": None, "trainer_state": None}
    value = load_json(path)
    return {
        "training_steps": value.get("global_step"),
        "trainer_state": str(path),
        "trainer_state_sha256": sha256_file(path),
    }


def validate_completion(
    path: Path, *, method: str, stage: int, immutable_config_sha256: str
) -> dict[str, Any]:
    value = load_json(path)
    errors = []
    if value.get("status") != "PASS":
        errors.append("status")
    if value.get("method") != method:
        errors.append("method")
    if int(value.get("stage", -1)) != stage:
        errors.append("stage")
    if value.get("immutable_config_sha256") != immutable_config_sha256:
        errors.append("immutable_config")
    for checkpoint in value.get("inference_stage2_checkpoints", []):
        info = inspect_adapter(Path(checkpoint["checkpoint"]))
        if info["adapter_weights_sha256"] != checkpoint["adapter_weights_sha256"]:
            errors.append(f"adapter_hash_task_{checkpoint.get('task')}")
    if errors:
        raise RuntimeError(f"Invalid completion {path}: {errors}")
    return value
