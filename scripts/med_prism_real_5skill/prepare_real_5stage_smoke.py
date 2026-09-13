#!/usr/bin/env python3
"""Freeze real MedicalSkill-CL-v1.1 smoke data and five Med-PRISM configs."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path("/root/MedAgentCL_v4")
DATA = Path("/remote-home/wangbomin/MedicalSkill-CL-v1.1")
OUT = ROOT / "artifacts/medicalskill_cl_v1_1_and_medprism_smoke"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TASKS = (
    "task_01_vqa", "task_02_diagnosis_classification", "task_03_concept_recognition",
    "task_04_visual_grounding", "task_05_reasoning_vqa",
)


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_select(values: list[dict[str, Any]], count: int, seed: int, *, grounding: bool = False) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    values = list(values)
    rng.shuffle(values)
    if not grounding:
        values.sort(key=lambda row: (len(row.get("images") or []), str(row.get("id"))))
        pool = values[: max(count * 20, count)]
        rng.shuffle(pool)
        return pool[:count]
    buckets: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in values:
        buckets[len((row.get("metadata") or {}).get("boxes_0_1000") or [])].append(row)
    selected = []
    for key in sorted(buckets, reverse=True):
        if key > 1 and buckets[key]:
            selected.append(buckets[key][0])
            break
    for row in values:
        if len(selected) == count:
            break
        if row not in selected:
            selected.append(row)
    return selected


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=3)
    args = parser.parse_args()
    smoke = OUT / "smoke_data"
    manifest = {"status": "PASS", "seed": 42, "source": str(DATA), "train_samples": args.train_samples, "eval_samples": args.eval_samples, "tasks": {}}
    for task_id, task_name in enumerate(TASKS, 1):
        train_source = DATA / task_name / "train.jsonl"
        test_source = DATA / task_name / "test.jsonl"
        train = stable_select(rows(train_source), args.train_samples, 4200 + task_id)
        test = stable_select(rows(test_source), args.eval_samples, 4300 + task_id, grounding=task_id == 4)
        train_path = smoke / f"task_{task_id:02d}_train_100.jsonl"
        test_path = smoke / f"task_{task_id:02d}_test_fixed.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(test_path, test)
        for row in train + test:
            if not row.get("messages") or not row.get("images") or any(not Path(path).is_file() for path in row["images"]):
                raise RuntimeError(f"Invalid smoke row {row.get('id')}")
        manifest["tasks"][str(task_id)] = {
            "name": task_name, "train_source": str(train_source), "test_source": str(test_source),
            "train_path": str(train_path), "test_path": str(test_path),
            "train_count": len(train), "test_count": len(test),
            "train_sha256": sha256(train_path), "test_sha256": sha256(test_path),
            "test_ids": [str(row["id"]) for row in test],
            "test_image_counts": [len(row["images"]) for row in test],
            "test_box_counts": [len((row.get("metadata") or {}).get("boxes_0_1000") or []) for row in test] if task_id == 4 else None,
        }
    (OUT / "smoke_data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    configs = OUT / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    for task_id in range(1, 6):
        stage = OUT / "med_prism" / f"stage_{task_id:02d}"
        payload = {
            "model_id": "Qwen/Qwen3-VL-8B-Instruct", "model_revision": REVISION,
            "current_task_id": task_id, "shared_rank": 32, "shared_alpha": 32.0,
            "shared_lr_scale": 0.1, "private_experts_per_task": 16,
            "private_alpha": 16.0, "dropout": 0.05, "orth_loss_type": "rms",
            "orth_lambda": 0.1, "orth_eps": 1e-8, "shared_drift_lambda": 0.01,
            "seed": 42,
            "shared_source_manifest": None if task_id == 1 else str(OUT / "med_prism" / f"stage_{task_id - 1:02d}" / "shared" / "shared_manifest.json"),
            "private_source_manifests": [str(OUT / "med_prism" / f"stage_{old:02d}" / "private" / "private_manifest.json") for old in range(1, task_id)],
            "shared_output_dir": str(stage / "shared"), "private_output_dir": str(stage / "private"),
            "dataset_manifest": str(OUT / "smoke_data_manifest.json"),
            "dataset_sha256": manifest["tasks"][str(task_id)]["train_sha256"],
            "output_dir": str(stage), "artifact_root": str(stage / "runtime_audits"),
        }
        (configs / f"stage_{task_id:02d}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
