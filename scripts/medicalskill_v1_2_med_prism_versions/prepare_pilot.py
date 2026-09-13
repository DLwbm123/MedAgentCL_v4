#!/usr/bin/env python3
"""Create deterministic train-derived pilot splits without touching the dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

TASKS = {
    1: "task_01_vqa",
    2: "task_02_diagnosis_classification",
    3: "task_03_concept_recognition",
    4: "task_04_visual_grounding",
    5: "task_05_reasoning_vqa",
}
DEFAULT_SOURCE = Path("/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(row: dict[str, Any]) -> str:
    value = row.get("id") or row.get("stable_id") or row.get("sample_id")
    if not value:
        raise ValueError("Every pilot candidate requires a stable ID")
    return str(value)


def hash_key(seed: int, task_id: int, domain: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}:{domain}:{value}".encode()).hexdigest()


def stratum(row: dict[str, Any]) -> tuple[str, ...]:
    source = row.get("source_dataset") or row.get("dataset") or row.get("source") or "unknown"
    modality = row.get("modality") or row.get("image_modality") or "unknown"
    label = (
        row.get("answer_option")
        or row.get("label")
        or row.get("class_label")
        or row.get("category")
        or "unknown"
    )
    return str(source), str(modality), str(label)


def stratified_hash_select(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    task_id: int,
    domain: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    queues = {
        key: deque(
            sorted(
                values,
                key=lambda row: hash_key(seed, task_id, domain, stable_id(row)),
            )
        )
        for key, values in groups.items()
    }
    ordered_keys = sorted(
        queues,
        key=lambda key: hash_key(seed, task_id, f"{domain}:stratum", repr(key)),
    )
    selected = []
    while len(selected) < min(limit, len(rows)):
        progressed = False
        for key in ordered_keys:
            if queues[key] and len(selected) < limit:
                selected.append(queues[key].popleft())
                progressed = True
        if not progressed:
            break
    return selected


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_selection(source: Path, *, train_limit: int, diagnostic_limit: int, seed: int):
    tasks = {}
    selected_ids = {}
    selected_rows = {}
    for task_id, task_name in TASKS.items():
        path = source / task_name / "train.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [stable_id(row) for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate stable IDs in {path}")
        pilot_train = stratified_hash_select(
            rows, limit=train_limit, seed=seed, task_id=task_id, domain="pilot_train"
        )
        train_ids = {stable_id(row) for row in pilot_train}
        remaining = [row for row in rows if stable_id(row) not in train_ids]
        pilot_diagnostic = stratified_hash_select(
            remaining,
            limit=diagnostic_limit,
            seed=seed,
            task_id=task_id,
            domain="pilot_diagnostic",
        )
        diagnostic_ids = {stable_id(row) for row in pilot_diagnostic}
        if train_ids & diagnostic_ids:
            raise RuntimeError(f"Pilot overlap for task {task_id}")
        selected_rows[task_id] = (pilot_train, pilot_diagnostic)
        selected_ids[str(task_id)] = {
            "task_name": task_name,
            "pilot_train": sorted(train_ids),
            "pilot_diagnostic": sorted(diagnostic_ids),
        }
        tasks[str(task_id)] = {
            "task_name": task_name,
            "source_train": str(path),
            "source_train_sha256": sha256(path),
            "source_train_count": len(rows),
            "pilot_train_count": len(pilot_train),
            "pilot_diagnostic_count": len(pilot_diagnostic),
            "overlap_count": 0,
            "stratum_fields": ["source_dataset_or_dataset", "modality", "label_or_answer_option"],
        }
    return tasks, selected_ids, selected_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-data-root", type=Path, required=True)
    parser.add_argument("--pilot-train-limit", type=int, default=1000)
    parser.add_argument("--pilot-diagnostic-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.seed != 42 or not 1 <= args.pilot_train_limit <= 1000 or not 1 <= args.pilot_diagnostic_limit <= 200:
        parser.error("Pilot locks seed=42, train<=1000, diagnostic<=200")
    source = args.source_data_root.resolve()
    output = args.output_data_root.resolve()
    tasks, ids, rows = build_selection(
        source,
        train_limit=args.pilot_train_limit,
        diagnostic_limit=args.pilot_diagnostic_limit,
        seed=args.seed,
    )
    preview = {
        "status": "PASS",
        "seed": args.seed,
        "selection_algorithm": "metadata_stratified_round_robin_then_sha256_id",
        "source_data_root": str(source),
        "output_data_root": str(output),
        "tasks": tasks,
    }
    if args.check_only:
        print(json.dumps(preview, indent=2, ensure_ascii=True))
        return 0
    for task_id, task_name in TASKS.items():
        train, diagnostic = rows[task_id]
        atomic_jsonl(output / task_name / "train.jsonl", train)
        atomic_jsonl(output / task_name / "test.jsonl", diagnostic)
        tasks[str(task_id)]["pilot_train_sha256"] = sha256(output / task_name / "train.jsonl")
        tasks[str(task_id)]["pilot_diagnostic_sha256"] = sha256(output / task_name / "test.jsonl")
    ids_path = output / "pilot_selected_ids.json"
    atomic_json(ids_path, ids)
    manifest = {
        **preview,
        "selected_ids_file": str(ids_path),
        "selected_ids_sha256": sha256(ids_path),
        "tasks": tasks,
    }
    manifest_path = output / "pilot_selection_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError(f"Existing pilot selection differs: {manifest_path}")
    else:
        atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
