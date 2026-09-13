#!/usr/bin/env python3
"""Auditable, deterministic replay buffers shared by MR-LoRA and Replay+LoRA."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from scripts.medicalskill_v1_2_baselines.baseline_harness import (
    SEED,
    TASK_ORDER,
    atomic_json,
    sha256_file,
    sha256_json,
    validate_dataset_lock,
)

ROUTER_ONLY = "router_training_only"
TRAINING_REPLAY = "model_training_replay"
PURPOSES = {ROUTER_ONLY, TRAINING_REPLAY}
POLICY = "balanced_per_task_stratified_stable_sha256_v1"
EXPERT_DESCRIPTIONS = {
    1: "Medical VQA: answer questions about medical images.",
    2: "Diagnosis classification: choose a diagnosis or disease class.",
    3: "Concept recognition: identify medical concepts supported by an image.",
    4: "Visual grounding: localize findings and return image coordinates.",
    5: "Reasoning VQA: solve clinical image questions with medical reasoning.",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                value["__line_number"] = line_number
                result.append(value)
    return result


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                clean = {key: value for key, value in row.items() if key != "__line_number"}
                handle.write(json.dumps(clean, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def question(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    return str(messages[0].get("content", "")) if messages else ""


def answer(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    return str(messages[-1].get("content", "")) if len(messages) > 1 else ""


def stratum(row: dict[str, Any], task_id: int) -> str:
    metadata = row.get("metadata") or {}
    keys = {
        1: ("source_dataset", "modality", "question_type"),
        2: ("dataset_id", "canonical_label"),
        3: ("source_dataset", "modality", "concept_structure"),
        4: ("medsg_task", "medsg_task_name", "source_dataset"),
        5: ("source_dataset", "reasoning_bucket", "modality"),
    }[task_id]
    values = [str(metadata.get(key, "")) for key in keys if metadata.get(key, "") not in (None, "")]
    return "|".join(values) if values else "unstratified"


@dataclass(frozen=True)
class ReplayRecord:
    task_id: int
    task_name: str
    sample_id: str
    source_dataset: str
    original_locked_train_file: str
    original_locked_train_sha256: str
    original_line_number: int
    sample_hash: str
    image_identifiers: tuple[str, ...]
    question_hash: str
    answer_hash: str | None
    answer_retained: bool
    replay_purpose: str
    selection_policy: str
    selection_seed: int
    stratum: str


def _record(
    row: dict[str, Any], task_id: int, train_file: Path, train_sha256: str,
    purpose: str, seed: int,
) -> ReplayRecord:
    metadata = row.get("metadata") or {}
    clean = {key: value for key, value in row.items() if key != "__line_number"}
    retain_answer = purpose == TRAINING_REPLAY
    return ReplayRecord(
        task_id=task_id,
        task_name=TASK_ORDER[task_id - 1],
        sample_id=str(row.get("id")),
        source_dataset=str(metadata.get("source_dataset") or metadata.get("dataset_id") or "unknown"),
        original_locked_train_file=str(train_file.resolve()),
        original_locked_train_sha256=train_sha256,
        original_line_number=int(row["__line_number"]),
        sample_hash=hashlib.sha256(canonical_bytes(clean)).hexdigest(),
        image_identifiers=tuple(str(value) for value in (row.get("images") or [])),
        question_hash=hash_text(question(row)),
        answer_hash=hash_text(answer(row)) if retain_answer else None,
        answer_retained=retain_answer,
        replay_purpose=purpose,
        selection_policy=POLICY,
        selection_seed=seed,
        stratum=stratum(row, task_id),
    )


def _stable_key(record: ReplayRecord) -> str:
    return hash_text(f"{record.selection_seed}|{record.task_id}|{record.stratum}|{record.sample_id}|{record.sample_hash}")


def task_quotas(total_budget: int, task_ids: list[int]) -> dict[int, int]:
    if total_budget < 0 or not task_ids:
        if total_budget == 0 and not task_ids:
            return {}
        raise ValueError("A non-negative budget and at least one task are required")
    ordered = sorted(set(task_ids))
    base, leftover = divmod(total_budget, len(ordered))
    return {task_id: base + (index < leftover) for index, task_id in enumerate(ordered)}


def _select_stratified(records: list[ReplayRecord], quota: int) -> list[ReplayRecord]:
    if quota >= len(records):
        return sorted(records, key=_stable_key)
    groups: dict[str, list[ReplayRecord]] = defaultdict(list)
    for record in records:
        groups[record.stratum].append(record)
    for values in groups.values():
        values.sort(key=_stable_key)
    chosen: list[ReplayRecord] = []
    strata = sorted(groups)
    while len(chosen) < quota:
        progressed = False
        for name in strata:
            if groups[name] and len(chosen) < quota:
                chosen.append(groups[name].pop(0))
                progressed = True
        if not progressed:
            break
    return sorted(chosen, key=lambda item: (item.task_id, _stable_key(item)))


def router_prompt(row: dict[str, Any], stage: int) -> str:
    pool = "\n".join(f"- EXPERT_{task_id:02d}: {EXPERT_DESCRIPTIONS[task_id]}" for task_id in range(1, stage + 1))
    return (
        "You are a generative router. Select exactly one expert for the medical image-question.\n"
        "Return only its token (for example EXPERT_01); do not answer the medical question.\n"
        f"Available experts:\n{pool}\n\nQuestion:\n{question(row)}"
    )


def router_row(row: dict[str, Any], task_id: int, stage: int) -> dict[str, Any]:
    return {
        "id": f"router_s{stage:02d}_t{task_id:02d}_{row['id']}",
        "task_id": task_id,
        "skill": "mrlora_generative_routing",
        "messages": [
            {"role": "user", "content": router_prompt(row, stage)},
            {"role": "assistant", "content": f"EXPERT_{task_id:02d}"},
        ],
        "images": list(row.get("images") or []),
        "metadata": {
            "source_sample_id": str(row["id"]),
            "source_task_id": task_id,
            "router_stage": stage,
            "replay_usage": ROUTER_ONLY,
            "target_derived_from_task_ownership": True,
            "original_target_retained": False,
        },
    }


def _assert_no_leakage(
    records: list[ReplayRecord], *, allowed_task_ids: set[int], data_root: Path, purpose: str
) -> dict[str, Any]:
    if any(record.task_id not in allowed_task_ids for record in records):
        raise RuntimeError("Replay contains current/future task information")
    identifiers = [(record.task_id, record.sample_id) for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Duplicate retained task/sample IDs")
    test_hashes: set[str] = set()
    test_ids: set[tuple[int, str]] = set()
    for task_id in range(1, 6):
        test_file = data_root / TASK_ORDER[task_id - 1] / "test.jsonl"
        for row in read_jsonl(test_file):
            clean = {key: value for key, value in row.items() if key != "__line_number"}
            test_hashes.add(hashlib.sha256(canonical_bytes(clean)).hexdigest())
            test_ids.add((task_id, str(row.get("id"))))
    if any(record.sample_hash in test_hashes for record in records):
        raise RuntimeError("Train/test canonical-row hash overlap entered replay")
    if any((record.task_id, record.sample_id) in test_ids for record in records):
        raise RuntimeError("A test sample ID entered replay")
    if purpose == ROUTER_ONLY and any(record.answer_retained or record.answer_hash for record in records):
        raise RuntimeError("Router-only replay retained target answers")
    return {"status": "PASS", "future_task_leakage": 0, "test_overlap": 0, "duplicates": 0}


def build_memory(
    *, data_root: Path, stage: int, purpose: str, output_dir: Path,
    total_budget: int | None = None, examples_per_task: int | None = None,
    include_current: bool = False, seed: int = SEED, previous_manifest: Path | None = None,
) -> dict[str, Any]:
    if purpose not in PURPOSES:
        raise ValueError(f"Unknown replay purpose: {purpose}")
    if not 1 <= stage <= 5:
        raise ValueError("stage must be in 1..5")
    dataset = validate_dataset_lock(data_root)
    last_task = stage if include_current else stage - 1
    allowed = list(range(1, last_task + 1))
    if examples_per_task is not None:
        quotas = {task_id: examples_per_task for task_id in allowed}
        memory_budget = examples_per_task * len(allowed)
    elif total_budget is not None:
        quotas = task_quotas(total_budget, allowed) if allowed else {}
        memory_budget = total_budget
    else:
        raise ValueError("Specify total_budget or examples_per_task")
    selected: list[ReplayRecord] = []
    raw_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for task_id in allowed:
        train_file = data_root / TASK_ORDER[task_id - 1] / "train.jsonl"
        train_sha256 = sha256_file(train_file)
        source_hashes[str(train_file.resolve())] = train_sha256
        records = []
        for row in read_jsonl(train_file):
            record = _record(row, task_id, train_file, train_sha256, purpose, seed)
            records.append(record)
            raw_by_key[(task_id, record.sample_id, record.sample_hash)] = row
        selected.extend(_select_stratified(records, min(quotas[task_id], len(records))))
    leakage = _assert_no_leakage(selected, allowed_task_ids=set(allowed), data_root=data_root, purpose=purpose)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    dataset_path = output_dir / ("router_train.jsonl" if purpose == ROUTER_ONLY else "replay_train.jsonl")
    atomic_jsonl(records_path, (asdict(record) for record in selected))
    materialized = []
    for record in selected:
        row = raw_by_key[(record.task_id, record.sample_id, record.sample_hash)]
        materialized.append(router_row(row, record.task_id, stage) if purpose == ROUTER_ONLY else row)
    atomic_jsonl(dataset_path, materialized)
    previous_hash = None
    if previous_manifest:
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        if previous.get("dataset_lock_sha256") != dataset["dataset_lock_sha256"]:
            raise RuntimeError("Stale replay buffer belongs to another dataset lock")
        previous_hash = previous.get("resulting_buffer_hash")
    count_per_task = {str(task_id): sum(record.task_id == task_id for record in selected) for task_id in allowed}
    record_payload = [asdict(record) for record in selected]
    resulting_hash = sha256_json(record_payload)
    manifest_core = {
        "format_version": "medicalskill_replay_manifest_v1",
        "status": "PASS",
        "stage": stage,
        "memory_policy": POLICY,
        "memory_budget": memory_budget,
        "memory_count": len(selected),
        "count_per_task": count_per_task,
        "replay_purpose": purpose,
        "replay_usage": purpose,
        "selection_seed": seed,
        "source_hashes": source_hashes,
        "dataset_lock_sha256": dataset["dataset_lock_sha256"],
        "selected_ids_sha256": dataset["selected_ids_sha256"],
        "retained_sample_ids": [f"{record.task_id}:{record.sample_id}" for record in selected],
        "logical_replay_bytes": sum(len(canonical_bytes(row)) + 1 for row in materialized),
        "physically_stored_metadata_bytes": records_path.stat().st_size,
        "copied_data_bytes": 0,
        "image_files_copied": 0,
        "previous_buffer_hash": previous_hash,
        "resulting_buffer_hash": resulting_hash,
        "records_file": str(records_path.resolve()),
        "records_sha256": sha256_file(records_path),
        "materialized_dataset": str(dataset_path.resolve()),
        "materialized_dataset_sha256": sha256_file(dataset_path),
        "leakage_audit": leakage,
    }
    manifest_core["buffer_manifest_sha256"] = sha256_json(manifest_core)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest_core:
            raise RuntimeError(f"Resume replay manifest mismatch: {manifest_path}")
    else:
        atomic_json(manifest_path, manifest_core)
    return manifest_core


def concatenate_current_and_replay(
    current_file: Path, replay_file: Path, output: Path, *, current_limit: int = 0
) -> dict[str, Any]:
    current = read_jsonl(current_file)
    if current_limit > 0:
        current = current[:current_limit]
    replay = read_jsonl(replay_file) if replay_file.is_file() else []
    current_ids = {(int(row.get("task_id", 0)), str(row.get("id"))) for row in current}
    replay_ids = {(int(row.get("task_id", 0)), str(row.get("id"))) for row in replay}
    if current_ids & replay_ids:
        raise RuntimeError("Current/replay mixture contains duplicate task/sample IDs")
    # HuggingFace/Arrow requires one stable schema. Task-native metadata contains
    # intentionally heterogeneous nested types, but Swift SFT only consumes these
    # three fields. Full metadata remains in the immutable sources and records.jsonl.
    training_rows = [
        {"id": row.get("id"), "messages": row.get("messages"), "images": row.get("images") or []}
        for row in [*current, *replay]
    ]
    atomic_jsonl(output, training_rows)
    return {
        "current_sample_count": len(current),
        "replay_sample_count": len(replay),
        "replay_current_ratio": len(replay) / len(current) if current else 0.0,
        "total_effective_training_examples": len(current) + len(replay),
        "duplicate_count": 0,
        "mixture_sha256": sha256_file(output),
        "one_epoch_semantics": "one pass through current plus replay JSONL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--purpose", choices=sorted(PURPOSES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-budget", type=int)
    parser.add_argument("--examples-per-task", type=int)
    parser.add_argument("--include-current", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--previous-manifest", type=Path)
    args = parser.parse_args()
    manifest = build_memory(**vars(args))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
