#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.phase6a.prepare_phase6a_data import SOURCES, normalize_record, record_id


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/phase6a"
DATA_ROOT = Path("/remote-home/wangbomin/medagentcl_v4_phase6b/data")
OUTPUT_ROOT = Path("/remote-home/wangbomin/medagentcl_v4_phase6b")
CONFIG_ROOT = ROOT / "configs/phase6b"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_split(
    task_id: int,
    split: str,
) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    source = SOURCES[task_id][split]
    output = DATA_ROOT / f"task_{task_id}_{split}.jsonl"
    seen: set[tuple[str, str]] = set()
    rejected: dict[str, int] = {}
    source_count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as reader, output.open(
        "w",
        encoding="utf-8",
    ) as writer:
        for line in reader:
            if not line.strip():
                continue
            source_count += 1
            try:
                row = normalize_record(json.loads(line), task_id=task_id)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                key = str(error) or type(error).__name__
                rejected[key] = rejected.get(key, 0) + 1
                continue
            identity = (record_id(row), row["images"][0])
            if identity in seen:
                rejected["duplicate_identity"] = (
                    rejected.get("duplicate_identity", 0) + 1
                )
                continue
            seen.add(identity)
            writer.write(json.dumps(row, ensure_ascii=True) + "\n")
    if not seen:
        raise RuntimeError(f"No valid unique rows for Task {task_id} {split}")
    manifest = {
        "status": "PASS",
        "task_id": task_id,
        "dataset": SOURCES[task_id]["dataset"],
        "task_type": SOURCES[task_id]["task_type"],
        "split": split,
        "source_path": str(source),
        "source_sha256": sha256(source),
        "source_count": source_count,
        "unique_count": len(seen),
        "sampling_with_replacement": False,
        "oversampling": False,
        "rejected": rejected,
        "dataset_path": str(output),
        "dataset_sha256": sha256(output),
    }
    path = ARTIFACT / f"phase6b_{split}_manifest_task{task_id}.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest, seen


def rank1_config(
    task_id: int,
    manifest: dict[str, Any],
    task1_step_count: int,
) -> dict[str, Any]:
    method = OUTPUT_ROOT / "pure_rank1_e16"
    return {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_revision": REVISION,
        "current_task_id": task_id,
        "experts_per_task": 16,
        "alpha": 16.0,
        "dropout": 0.05,
        "orth_loss_type": "rms",
        "orth_lambda": 0.1,
        "orth_eps": 1e-8,
        "seed": 42,
        "source_manifest": (
            None
            if task_id == 1
            else str(
                method
                / "task_1/train"
                / f"checkpoint-{task1_step_count}"
                / "rank1_manifest.json"
            )
        ),
        "dataset_manifest": str(
            ARTIFACT / f"phase6b_train_manifest_task{task_id}.json"
        ),
        "dataset_sha256": manifest["dataset_sha256"],
        "output_dir": str(method / f"task_{task_id}"),
        "artifact_root": str(method / "runtime_audits"),
    }


def shared_config(task_id: int, manifest: dict[str, Any]) -> dict[str, Any]:
    method = OUTPUT_ROOT / "shared32_private16"
    return {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_revision": REVISION,
        "current_task_id": task_id,
        "shared_rank": 32,
        "shared_alpha": 32.0,
        "shared_lr_scale": 0.1,
        "private_experts_per_task": 16,
        "private_alpha": 16.0,
        "dropout": 0.05,
        "orth_loss_type": "rms",
        "orth_lambda": 0.1,
        "orth_eps": 1e-8,
        "shared_drift_lambda": 0.01,
        "seed": 42,
        "shared_source_manifest": (
            None
            if task_id == 1
            else str(method / "shared/after_task_1/shared_manifest.json")
        ),
        "private_source_manifests": (
            []
            if task_id == 1
            else [str(method / "private/task_1/private_manifest.json")]
        ),
        "shared_output_dir": str(method / f"shared/after_task_{task_id}"),
        "private_output_dir": str(method / f"private/task_{task_id}"),
        "dataset_manifest": str(
            ARTIFACT / f"phase6b_train_manifest_task{task_id}.json"
        ),
        "dataset_sha256": manifest["dataset_sha256"],
        "output_dir": str(method / f"task_{task_id}"),
        "artifact_root": str(method / "runtime_audits"),
    }


def main() -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifests: dict[tuple[int, str], dict[str, Any]] = {}
    identities: dict[tuple[int, str], set[tuple[str, str]]] = {}
    for task_id in (1, 2):
        for split in ("train", "test"):
            manifests[(task_id, split)], identities[(task_id, split)] = (
                prepare_split(task_id, split)
            )
        overlap = identities[(task_id, "train")] & identities[(task_id, "test")]
        if overlap:
            raise RuntimeError(f"Task {task_id} train/test overlap: {len(overlap)}")
    for task_id in (1, 2):
        configs = {
            "rank1": rank1_config(
                task_id,
                manifests[(task_id, "train")],
                manifests[(1, "train")]["unique_count"],
            ),
            "shared_private": shared_config(
                task_id,
                manifests[(task_id, "train")],
            ),
        }
        for name, payload in configs.items():
            path = CONFIG_ROOT / f"{name}_task_{task_id}.json"
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
    summary = {
        "status": "PASS",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_revision": REVISION,
        "seed": 42,
        "sampling_with_replacement": False,
        "oversampling": False,
        "train_test_overlap": {"1": 0, "2": 0},
        "counts": {
            str(task_id): {
                split: manifests[(task_id, split)]["unique_count"]
                for split in ("train", "test")
            }
            for task_id in (1, 2)
        },
        "manifests": {
            f"task_{task_id}_{split}": str(
                ARTIFACT / f"phase6b_{split}_manifest_task{task_id}.json"
            )
            for task_id in (1, 2)
            for split in ("train", "test")
        },
    }
    (ARTIFACT / "phase6b_data_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
