#!/usr/bin/env python3
"""Prepare and validate the immutable MedicalSkill-CL-v1.2 Med-PRISM run contract."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
REPO = Path("/root/MedAgentCL_v4")
DEFAULT_DATA = Path("/remote-home/wangbomin/MedicalSkill-CL-v1.2")
DEFAULT_OUTPUT = Path("/remote-home/wangbomin/med_prism_medicalskill_v1_2_seed42")
SNAPSHOT = Path(
    "/remote-home/wangbomin/huggingface_cache/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
) / MODEL_REVISION
TASKS = {
    1: "task_01_vqa",
    2: "task_02_diagnosis_classification",
    3: "task_03_concept_recognition",
    4: "task_04_visual_grounding",
    5: "task_05_reasoning_vqa",
}
EXPECTED = {
    1: (66492, 17796),
    2: (47968, 6804),
    3: (31452, 3548),
    4: (49480, 9623),
    5: (7347, 720),
}
LITE_VERSION = "MedicalSkill-CL-v1.2-lite-10k1k"
LITE_MANIFEST = "medicalskill_cl_v1_2_lite_manifest.json"
DEFAULT_PRIVATE_EXPERTS_PER_TASK = 16
DEFAULT_PRIVATE_ALPHA = 16.0
DEFAULT_ORTH_LAMBDA = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def dataset_profile(data: Path) -> dict[str, Any]:
    """Resolve an immutable full or lite v1.2 contract from dataset artifacts."""
    lock_path = data / "manifests/dataset_lock.json"
    if not lock_path.is_file():
        manifest = data / "manifests/medicalskill_cl_v1_2_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing full v1.2 manifest: {manifest}")
        return {
            "dataset_version": "MedicalSkill-CL-v1.2",
            "expected": EXPECTED,
            "manifest": manifest,
            "lock": None,
            "locked_splits": {},
        }

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("dataset_version") != LITE_VERSION:
        raise RuntimeError(
            f"Unsupported locked dataset version: {lock.get('dataset_version')!r}"
        )
    manifest = data / "manifests" / LITE_MANIFEST
    selected_ids = data / str(lock.get("selected_ids_file", ""))
    if not manifest.is_file() or not selected_ids.is_file():
        raise FileNotFoundError("Lite manifest or selected ID manifest is missing")
    if sha256(selected_ids) != lock.get("selected_ids_sha256"):
        raise RuntimeError("Lite selected ID manifest differs from dataset lock")
    expected: dict[int, tuple[int, int]] = {}
    locked_splits: dict[int, dict[str, dict[str, Any]]] = {}
    for task in lock.get("tasks", []):
        task_id = int(task["task_id"])
        splits = task["splits"]
        expected[task_id] = (
            int(splits["train"]["records"]),
            int(splits["test"]["records"]),
        )
        locked_splits[task_id] = splits
    if set(expected) != set(TASKS):
        raise RuntimeError(f"Lite lock task IDs differ: {sorted(expected)}")
    return {
        "dataset_version": LITE_VERSION,
        "expected": expected,
        "manifest": manifest,
        "lock": {
            "path": str(lock_path),
            "sha256": sha256(lock_path),
            "selected_ids_path": str(selected_ids),
            "selected_ids_sha256": lock["selected_ids_sha256"],
            "selected_id_count": lock["selected_id_count"],
        },
        "locked_splits": locked_splits,
    }


def assert_contract(data: Path) -> dict[str, Any]:
    acceptance_path = data / "audits/acceptance_matrix.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "PASS":
        raise RuntimeError(f"Dataset acceptance is not PASS: {acceptance_path}")
    if any(data.glob("task_06*")):
        raise RuntimeError("Unexpected sixth task in the five-stage release")
    profile = dataset_profile(data)
    results = {}
    for task_id, name in TASKS.items():
        train = data / name / "train.jsonl"
        test = data / name / "test.jsonl"
        if not train.is_file() or not test.is_file():
            raise FileNotFoundError(f"Missing train/test JSONL for {name}")
        train_count, test_count = count_jsonl(train), count_jsonl(test)
        expected = profile["expected"][task_id]
        if (train_count, test_count) != expected:
            raise RuntimeError(
                f"Unexpected {name} counts: {(train_count, test_count)} != {expected}"
            )
        train_sha, test_sha = sha256(train), sha256(test)
        locked = profile["locked_splits"].get(task_id)
        if locked and train_sha != locked["train"]["sha256"]:
            raise RuntimeError(f"Locked train hash mismatch: {train}")
        if locked and test_sha != locked["test"]["sha256"]:
            raise RuntimeError(f"Locked test hash mismatch: {test}")
        if (data / name / "val.jsonl").exists() or (data / name / "validation.jsonl").exists():
            raise RuntimeError(f"Validation split is forbidden: {name}")
        results[str(task_id)] = {
            "task_name": name,
            "train": str(train),
            "test": str(test),
            "train_count": train_count,
            "test_count": test_count,
            "train_sha256": train_sha,
            "test_sha256": test_sha,
        }
    return {
        "dataset_root": str(data),
        "dataset_version": profile["dataset_version"],
        "dataset_manifest": str(profile["manifest"]),
        "dataset_manifest_sha256": sha256(profile["manifest"]),
        "dataset_lock": profile["lock"],
        "acceptance_matrix": str(acceptance_path),
        "acceptance_sha256": sha256(acceptance_path),
        "tasks": results,
    }


def build_contract(
    data: Path,
    output: Path,
    seed: int,
    private_experts_per_task: int = DEFAULT_PRIVATE_EXPERTS_PER_TASK,
    private_alpha: float = DEFAULT_PRIVATE_ALPHA,
    orth_lambda: float = DEFAULT_ORTH_LAMBDA,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if private_experts_per_task <= 0:
        raise ValueError("private_experts_per_task must be positive")
    if private_alpha <= 0:
        raise ValueError("private_alpha must be positive")
    if orth_lambda < 0:
        raise ValueError("orth_lambda cannot be negative")
    if not SNAPSHOT.is_dir() or not (SNAPSHOT / "config.json").is_file():
        raise FileNotFoundError(f"Frozen model snapshot is missing: {SNAPSHOT}")
    for required in (
        REPO / "med_prism/swift_plugins/med_prism_rank1_plugin.py",
        REPO / "med_prism/swift_plugins/med_prism_shared_private_plugin.py",
        REPO / "scripts/med_prism_real_5skill/reload_probe_plugin.py",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    dataset = assert_contract(data)
    configs = {}
    for stage in range(1, 6):
        root = output / "med_prism" / f"stage_{stage:02d}"
        configs[stage] = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "current_task_id": stage,
            "shared_rank": 32,
            "shared_alpha": 32.0,
            "shared_lr_scale": 0.1,
            "private_experts_per_task": private_experts_per_task,
            "private_alpha": private_alpha,
            "dropout": 0.05,
            "orth_loss_type": "rms",
            "orth_lambda": orth_lambda,
            "orth_eps": 1e-8,
            "shared_drift_lambda": 0.01,
            "seed": seed,
            "shared_source_manifest": None if stage == 1 else str(output / "med_prism" / f"stage_{stage - 1:02d}" / "shared/shared_manifest.json"),
            "private_source_manifests": [str(output / "med_prism" / f"stage_{old:02d}" / "private/private_manifest.json") for old in range(1, stage)],
            "shared_output_dir": str(root / "shared"),
            "private_output_dir": str(root / "private"),
            "dataset_manifest": dataset["dataset_manifest"],
            "dataset_sha256": dataset["tasks"][str(stage)]["train_sha256"],
            "output_dir": str(root),
            "artifact_root": str(root / "runtime_audits"),
        }
    try:
        swift_version = importlib.metadata.version("ms-swift")
    except importlib.metadata.PackageNotFoundError:
        swift_version = importlib.metadata.version("swift")
    manifest = {
        "status": "PASS",
        "method": "med_prism_shared_private",
        "ordinary_lora": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot": str(SNAPSHOT),
        "seed": seed,
        "ms_swift_version": swift_version,
        "data": dataset,
        "output_root": str(output),
        "stage_order": list(TASKS.values()),
        "one_epoch": True,
        "shared_rank": 32,
        "private_rank1_experts_per_task": private_experts_per_task,
        "private_alpha": private_alpha,
        "orth_lambda": orth_lambda,
        "shared_drift_lambda": 0.01,
    }
    return manifest, configs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--private-experts-per-task",
        type=int,
        default=DEFAULT_PRIVATE_EXPERTS_PER_TASK,
    )
    parser.add_argument(
        "--private-alpha", type=float, default=DEFAULT_PRIVATE_ALPHA
    )
    parser.add_argument(
        "--orth-lambda", type=float, default=DEFAULT_ORTH_LAMBDA
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest, configs = build_contract(
        args.data_root.resolve(),
        args.output_root.resolve(),
        args.seed,
        private_experts_per_task=args.private_experts_per_task,
        private_alpha=args.private_alpha,
        orth_lambda=args.orth_lambda,
    )
    if not args.check_only:
        args.output_root.mkdir(parents=True, exist_ok=True)
        run_manifest = args.output_root / "run_manifest.json"
        if run_manifest.is_file():
            previous = json.loads(run_manifest.read_text(encoding="utf-8"))
            if previous != manifest:
                raise RuntimeError(f"Existing run contract differs: {run_manifest}")
        else:
            atomic_json(run_manifest, manifest)
        for stage, config in configs.items():
            path = args.output_root / "configs" / f"stage_{stage:02d}.json"
            if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != config:
                raise RuntimeError(f"Existing stage config differs: {path}")
            if not path.is_file():
                atomic_json(path, config)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())