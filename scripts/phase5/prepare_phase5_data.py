#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/phase5_shared_private/data"
OUTPUT = ROOT / "output/phase5_shared_private"
SOURCES = {
    1: Path("/root/MedAgentCL/data/MedSkill_CL_4Skill/task_02_vqa_train.jsonl"),
    2: Path(
        "/root/MedAgentCL/data/MedIMeta_CL_Diag8/"
        "task_01_pneumonia_disease_class/train.jsonl"
    ),
}
SEED = 42
SAMPLE_COUNT = 24


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_valid(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        images = row.get("images") or [row.get("image")]
        sample_id = str(
            row.get("source_sample_id") or row.get("question_id") or row.get("id")
        )
        image = Path(images[0]) if images and images[0] else None
        identity = (sample_id, str(image))
        if (
            identity in seen
            or image is None
            or not image.is_file()
            or image.stat().st_size == 0
        ):
            continue
        seen.add(identity)
        rows.append(row)
    return rows


def write_task(task_id: int, rows: list[dict]) -> tuple[Path, dict]:
    task_type = "vqa" if task_id == 1 else "classification"
    dataset_path = ARTIFACT / f"task{task_id}_{task_type}_train.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    samples = []
    for row in rows:
        image = Path((row.get("images") or [row["image"]])[0])
        answer = (
            row.get("answer") or row.get("response") or row["messages"][-1]["content"]
        )
        samples.append(
            {
                "sample_id": str(
                    row.get("source_sample_id")
                    or row.get("question_id")
                    or row.get("id")
                ),
                "record_id": str(row.get("id")),
                "image_path": str(image),
                "image_exists": image.is_file(),
                "image_size_bytes": image.stat().st_size,
                "image_sha256": sha256(image),
                "task_type": task_type,
                "label_or_answer": answer,
                "prompt_format": "swift messages + images; user contains <image>",
                "split": row.get("split") or row.get("source_split") or "train",
            }
        )
    manifest = {
        "status": "PASS",
        "task_id": task_id,
        "task_type": task_type,
        "source_path": str(SOURCES[task_id]),
        "source_sha256": sha256(SOURCES[task_id]),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "sample_count": len(rows),
        "seed": SEED,
        "selection": "random.Random(seed).shuffle(valid unique source/image rows)",
        "all_images_exist": all(item["image_exists"] for item in samples),
        "samples": samples,
    }
    manifest_path = ARTIFACT / (
        "task1_vqa_manifest.json"
        if task_id == 1
        else "task2_classification_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, manifest


def write_config(task_id: int, manifest_path: Path, manifest: dict) -> None:
    config = {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
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
        "shared_drift_lambda": 0.0,
        "seed": SEED,
        "shared_source_manifest": (
            None
            if task_id == 1
            else str(OUTPUT / "shared/after_task_1/shared_manifest.json")
        ),
        "private_source_manifests": (
            []
            if task_id == 1
            else [str(OUTPUT / "private/task_1/private_manifest.json")]
        ),
        "shared_output_dir": str(OUTPUT / f"shared/after_task_{task_id}"),
        "private_output_dir": str(OUTPUT / f"private/task_{task_id}"),
        "dataset_manifest": str(manifest_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "output_dir": str(OUTPUT / f"runs/task_{task_id}"),
    }
    path = ROOT / f"configs/phase5/task_{task_id}_shared_private.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for task_id in (1, 2):
        valid = read_valid(SOURCES[task_id])
        rng.shuffle(valid)
        selected = valid[:SAMPLE_COUNT]
        if len(selected) != SAMPLE_COUNT:
            raise RuntimeError(f"Task {task_id} has only {len(selected)} valid rows")
        manifest_path, manifest = write_task(task_id, selected)
        write_config(task_id, manifest_path, manifest)
        print(
            json.dumps({"task_id": task_id, "samples": len(selected), "status": "PASS"})
        )


if __name__ == "__main__":
    main()
