#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from med_prism.evaluation.cl import evaluate_prediction, extract_options


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/phase6a"
DATA = ARTIFACT / "data"
REMOTE = Path("/remote-home/wangbomin/medagentcl_v4_phase6a")
SEED = 42
TRAIN_COUNT = 128
TEST_COUNT = 64
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SOURCES = {
    1: {
        "train": Path(
            "/root/MedAgentCL/data/MedSkill_CL_4Skill/task_02_vqa_train.jsonl"
        ),
        "test": Path(
            "/root/MedAgentCL/data/MedSkill_CL_4Skill/task_02_vqa_test.jsonl"
        ),
        "task_type": "vqa",
        "dataset": "OmniMedVQA",
    },
    2: {
        "train": Path(
            "/root/MedAgentCL/data/MedIMeta_CL_Diag8/"
            "task_01_pneumonia_disease_class/train.jsonl"
        ),
        "test": Path(
            "/root/MedAgentCL/data/MedIMeta_CL_Diag8/"
            "task_01_pneumonia_disease_class/test.jsonl"
        ),
        "task_type": "classification",
        "dataset": "MedIMeta",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_id(row: dict[str, Any]) -> str:
    return str(
        row.get("source_sample_id")
        or row.get("question_id")
        or row.get("cl_sample_id")
        or row.get("id")
    )


def normalize_record(row: dict[str, Any], *, task_id: int) -> dict[str, Any]:
    row = dict(row)
    images = row.get("images") or [row.get("image")]
    if not images or not images[0]:
        raise ValueError("missing image")
    image = Path(images[0])
    if not image.is_file() or image.stat().st_size <= 0:
        raise ValueError("unreadable image")
    letters, options = extract_options(row)
    target_text = str(
        row.get("answer")
        or row.get("response")
        or row["messages"][-1]["content"]
    )
    parsed_target = evaluate_prediction(row, target_text)
    if not parsed_target["valid"] or not parsed_target["correct"]:
        raise ValueError("target is absent from options")
    row["images"] = [str(image)]
    row["image"] = str(image)
    row["option_letters"] = letters
    row["options"] = options
    row["answer_option"] = parsed_target["target_letter"]
    row["answer"] = parsed_target["target"]
    row["phase6a_task_id"] = task_id
    return row


def select(path: Path, *, task_id: int, split: str, limit: int) -> list[dict[str, Any]]:
    candidates = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = normalize_record(json.loads(line), task_id=task_id)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        identity = (record_id(row), row["images"][0])
        if identity in seen:
            continue
        seen.add(identity)
        score = hashlib.sha256(
            f"{SEED}\0{task_id}\0{split}\0{identity[0]}\0{identity[1]}".encode()
        ).hexdigest()
        candidates.append((score, line_number, row))
    candidates.sort(key=lambda item: item[0])
    selected = candidates[:limit]
    if not selected:
        raise RuntimeError(f"No valid unique Task {task_id} {split} samples")
    return [
        {
            **row,
            "phase6a_source_line": line_number,
            "phase6a_selection_sha256": score,
        }
        for score, line_number, row in selected
    ]


def write_split(task_id: int, split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_path = DATA / f"pilot_{split}_task{task_id}.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    source = SOURCES[task_id]
    samples = [
        {
            "sample_id": record_id(row),
            "image_path": row["images"][0],
            "image_exists": Path(row["images"][0]).is_file(),
            "image_size_bytes": Path(row["images"][0]).stat().st_size,
            "target": row["answer"],
            "target_letter": row["answer_option"],
            "option_count": len(row["options"]),
            "selection_sha256": row["phase6a_selection_sha256"],
            "source_line": row["phase6a_source_line"],
        }
        for row in rows
    ]
    manifest = {
        "status": "PASS",
        "task_id": task_id,
        "dataset": source["dataset"],
        "task_type": source["task_type"],
        "split": split,
        "seed": SEED,
        "requested_count": TRAIN_COUNT if split == "train" else TEST_COUNT,
        "sample_count": len(rows),
        "sampling_with_replacement": False,
        "all_unique": len(
            {(item["sample_id"], item["image_path"]) for item in samples}
        )
        == len(samples),
        "all_images_exist": all(item["image_exists"] for item in samples),
        "source_path": str(source[split]),
        "source_sha256": sha256(source[split]),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "samples": samples,
    }
    manifest_path = ARTIFACT / f"pilot_{split}_manifest_task{task_id}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def config(
    *,
    scope: str,
    task_id: int,
    dataset_manifest: dict[str, Any],
) -> dict[str, Any]:
    output = REMOTE / scope
    artifact_root = ARTIFACT / f"{scope}_runtime"
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
        "seed": SEED,
        "shared_source_manifest": (
            None
            if task_id == 1
            else str(output / "shared/after_task_1/shared_manifest.json")
        ),
        "private_source_manifests": (
            []
            if task_id == 1
            else [str(output / "private/task_1/private_manifest.json")]
        ),
        "shared_output_dir": str(output / f"shared/after_task_{task_id}"),
        "private_output_dir": str(output / f"private/task_{task_id}"),
        "dataset_manifest": str(
            ARTIFACT / f"pilot_train_manifest_task{task_id}.json"
        ),
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "output_dir": str(output / f"runs/task_{task_id}"),
        "artifact_root": str(artifact_root),
    }


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (ROOT / "configs/phase6a").mkdir(parents=True, exist_ok=True)
    REMOTE.mkdir(parents=True, exist_ok=True)
    manifests: dict[tuple[int, str], dict[str, Any]] = {}
    for task_id in (1, 2):
        for split, limit in (("train", TRAIN_COUNT), ("test", TEST_COUNT)):
            rows = select(
                SOURCES[task_id][split],
                task_id=task_id,
                split=split,
                limit=limit,
            )
            manifests[(task_id, split)] = write_split(task_id, split, rows)
    for scope in ("closure", "pilot"):
        for task_id in (1, 2):
            payload = config(
                scope=scope,
                task_id=task_id,
                dataset_manifest=manifests[(task_id, "train")],
            )
            path = ROOT / f"configs/phase6a/{scope}_task_{task_id}.json"
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
    summary = {
        "status": "PASS",
        "seed": SEED,
        "train_count_per_task": {
            str(task): manifests[(task, "train")]["sample_count"]
            for task in (1, 2)
        },
        "test_count_per_task": {
            str(task): manifests[(task, "test")]["sample_count"]
            for task in (1, 2)
        },
        "no_sampling_with_replacement": True,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
