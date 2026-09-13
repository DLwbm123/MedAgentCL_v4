#!/usr/bin/env python3
"""Build MedicalSkill-CL-v1.2 from v1.1 and the frozen Kimi 35K closure."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO = Path("/root/MedAgentCL_v4")
BASE = Path("/remote-home/wangbomin")
V11 = BASE / "MedicalSkill-CL-v1.1"
OUTPUT = BASE / "MedicalSkill-CL-v1.2"
STAGING = BASE / ".MedicalSkill-CL-v1.2.building"
UPSTREAM = (
    REPO
    / "artifacts/medicalskill_cl_kimi_k3_v3_1_k2_6_routing"
)
CLOSURE = UPSTREAM / "cycle_02_35k_closure"
BUILD_ARTIFACT = REPO / "artifacts/medicalskill_cl_v1_2_build"
EXPECTED_FINAL_SHA = (
    "db9b39ed443a84570c8b9c83482cbe81ff3b34d947e6ebe68bd752947d278152"
)
EXPECTED_RESULTS_SHA = (
    "025ca724f7a47e170e352ca920f8dad2284b5a321891a86f477b463ac423ec4c"
)
TASKS = {
    1: ("task_01_vqa", "vqa"),
    2: ("task_02_diagnosis_classification", "diagnosis_classification"),
    3: ("task_03_concept_recognition", "concept_recognition"),
    4: ("task_04_visual_grounding", "visual_grounding"),
    5: ("task_05_reasoning_vqa", "reasoning_vqa"),
}
INHERITED_TASKS = (1, 2, 4, 5)
PROMPT = (
    "<image>\nIdentify the caption-supported medical concepts. "
    "Return a concise semicolon-separated list."
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object JSONL row: {path}:{line_number}")
            yield value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
    )


def normalized(value: str) -> str:
    return " ".join(str(value).casefold().split())


def file_inventory(root: Path, include_hash: bool) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        row = {
            "path": str(path),
            "relative_path": str(path.relative_to(root)),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if include_hash:
            row["sha256"] = sha256(path)
        rows.append(row)
    return rows


def inspect_version(root: Path) -> dict[str, Any]:
    tasks = []
    for directory in sorted(root.glob("task_*")):
        counts = {}
        schemas = {}
        prompts = {}
        for split in ("train", "test"):
            path = directory / f"{split}.jsonl"
            count = 0
            keys: set[str] = set()
            first = None
            if path.exists():
                for row in jsonl(path):
                    count += 1
                    keys.update(row)
                    if first is None:
                        first = row
            counts[split] = count
            schemas[split] = sorted(keys)
            if first:
                prompts[split] = [
                    item.get("content") for item in first.get("messages", [])
                ]
        tasks.append(
            {
                "task_name": directory.name,
                "counts": counts,
                "schema_keys": schemas,
                "first_prompt_and_target": prompts,
                "files": sorted(
                    path.name for path in directory.iterdir() if path.is_file()
                ),
            }
        )
    return {
        "absolute_path": str(root),
        "exists": root.is_dir(),
        "version_name": root.name,
        "task_count": len(tasks),
        "tasks": tasks,
        "has_validation_split": any(
            path.name.startswith(("val.", "validation."))
            for path in root.rglob("*")
            if path.is_file()
        ),
        "top_level_directories": sorted(
            path.name for path in root.iterdir() if path.is_dir()
        )
        if root.is_dir()
        else [],
    }


def upstream_contract() -> dict[str, Any]:
    final_path = CLOSURE / "final_35k_active_manifest.jsonl"
    results_path = UPSTREAM / "v3_1_results.jsonl"
    summary = json.loads(
        (CLOSURE / "cycle_02_completion_summary.json").read_text()
    )
    final_sha = sha256(final_path)
    results_sha = sha256(results_path)
    final_rows = list(jsonl(final_path))
    ids = [row["stable_id"] for row in final_rows]
    checks = {
        "final_manifest_sha256": final_sha == EXPECTED_FINAL_SHA,
        "results_sha256": results_sha == EXPECTED_RESULTS_SHA,
        "completion_summary_pass": summary.get("status") == "PASS",
        "final_rows_35000": len(final_rows) == 35_000,
        "stable_ids_unique_35000": len(set(ids)) == 35_000,
        "concept_lists_valid": all(
            isinstance(row.get("final_concepts"), list)
            and 1 <= len(row["final_concepts"]) <= 8
            and all(
                isinstance(item, str) and item.strip()
                for item in row["final_concepts"]
            )
            for row in final_rows
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"upstream contract failed: {checks}")
    return {
        "checks": checks,
        "final_manifest_path": str(final_path),
        "final_manifest_sha256": final_sha,
        "results_path": str(results_path),
        "results_sha256": results_sha,
        "completion_summary_path": str(
            CLOSURE / "cycle_02_completion_summary.json"
        ),
        "completion_summary_sha256": sha256(
            CLOSURE / "cycle_02_completion_summary.json"
        ),
        "rows": len(final_rows),
    }


def load_mapping() -> dict[str, Any]:
    final_rows = list(jsonl(CLOSURE / "final_35k_active_manifest.jsonl"))
    input_rows = {
        row["stable_id"]: row
        for row in jsonl(UPSTREAM / "concept_35k_v3_1_input_manifest.jsonl")
    }
    results_by_key = {}
    for row in jsonl(UPSTREAM / "v3_1_results.jsonl"):
        key = (row.get("stable_id"), row.get("run_id"))
        results_by_key[key] = row
    v11_rows = {}
    v11_split = {}
    for split in ("train", "test"):
        for row in jsonl(V11 / f"task_03_concept_recognition/{split}.jsonl"):
            v11_rows[row["id"]] = row
            v11_split[row["id"]] = split
    mapped = []
    errors = collections.Counter()
    for final in final_rows:
        stable_id = final["stable_id"]
        source = v11_rows.get(stable_id)
        selection = input_rows.get(stable_id)
        result = results_by_key.get((stable_id, final.get("result_run_id")))
        if source is None:
            errors["missing_v1_1"] += 1
            continue
        if selection is None:
            errors["missing_selection_manifest"] += 1
            continue
        if result is None:
            errors["missing_exact_result_run"] += 1
            continue
        if v11_split[stable_id] != selection.get("split"):
            errors["split_mismatch"] += 1
        if (
            selection.get("source_caption_sha256")
            != final.get("source_caption_sha256")
            or result.get("source_caption_sha256")
            != final.get("source_caption_sha256")
        ):
            errors["caption_hash_mismatch"] += 1
        if hashlib.sha256(
            str(result.get("source_caption") or "").encode()
        ).hexdigest() != final.get("source_caption_sha256"):
            errors["caption_recompute_mismatch"] += 1
        if (
            (source.get("metadata") or {}).get("lineage_group_id")
            != selection.get("current_lineage_group_id")
        ):
            errors["lineage_mismatch"] += 1
        if result.get("final_concepts") != final.get("final_concepts"):
            errors["result_concept_mismatch"] += 1
        mapped.append(
            {
                "stable_id": stable_id,
                "split": selection["split"],
                "final": final,
                "selection": selection,
                "result": result,
                "source": source,
            }
        )
    ids = [row["stable_id"] for row in mapped]
    if len(mapped) != 35_000 or len(set(ids)) != 35_000 or errors:
        raise RuntimeError(
            f"35K mapping failed: mapped={len(mapped)}, errors={dict(errors)}"
        )
    return {
        "rows": mapped,
        "split_counts": dict(
            collections.Counter(row["split"] for row in mapped)
        ),
        "errors": dict(errors),
        "v1_1_task3_total": len(v11_rows),
    }


def task3_record(item: dict[str, Any]) -> dict[str, Any]:
    source = item["source"]
    final = item["final"]
    result = item["result"]
    concepts = list(final["final_concepts"])
    metadata = dict(source.get("metadata") or {})
    for key in (
        "concepts",
        "all_target_concepts",
        "core_target_concepts",
        "model_id",
        "revision",
        "prompt_hash",
        "prompt_version",
        "concept_parser_version",
        "concept_repair_reasons_v2",
        "concept_repair_reasons_v3",
    ):
        metadata.pop(key, None)
    for forbidden in (
        "original_caption",
        "cleaned_caption",
        "source_caption",
        "raw_response",
    ):
        metadata.pop(forbidden, None)
    metadata.update(
        {
            "source_caption_sha256": final["source_caption_sha256"],
            "concepts": [{"canonical": concept} for concept in concepts],
            "all_target_concepts": concepts,
            "core_target_concepts": [normalized(value) for value in concepts],
            "concept_generation_model": "kimi-k2.6",
            "concept_request_model": "k3",
            "concept_reasoning_effort": "none",
            "concept_prompt_version": result.get("prompt_version"),
            "concept_prompt_hash": result.get("prompt_hash"),
            "concept_schema_hash": result.get("schema_hash"),
            "concept_run_config_hash": result.get("run_config_hash"),
            "concept_result_run_id": final.get("result_run_id"),
            "concept_max_items": 8,
            "manual_qc_status": "pending_partial_review"
            if final.get("audit_flags")
            else "not_flagged",
        }
    )
    return {
        "id": item["stable_id"],
        "task_id": 3,
        "skill": "concept_recognition",
        "messages": [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": "; ".join(concepts)},
        ],
        "images": [str(path) for path in source.get("images") or []],
        "metadata": metadata,
    }


def inventory_reports(root: Path, old_hashes: list[dict[str, Any]]) -> None:
    versions = [
        path
        for path in sorted(BASE.glob("MedicalSkill-CL*"))
        if path.is_dir() and path != OUTPUT
    ]
    inventory = {
        "created_at": now(),
        "read_only_inventory": True,
        "versions": [inspect_version(path) for path in versions],
    }
    write_json(root / "audits/old_version_inventory.json", inventory)
    write_json(
        root / "audits/v1_1_file_hashes_before.json",
        {
            "status": "PASS",
            "root": str(V11),
            "file_count": len(old_hashes),
            "files": old_hashes,
        },
    )
    atomic_text(
        root / "audits/v1_1_structure_report.md",
        "# MedicalSkill-CL-v1.1 structure\n\n"
        "v1.1 is the authoritative five-stage baseline:\n\n"
        "1. VQA\n2. Diagnosis Classification\n3. Concept Recognition\n"
        "4. Visual Grounding\n5. Reasoning VQA\n\n"
        "It has train/test only and no validation split. Caption Generation is "
        "absent as an independent task. Task 3 uses Qwen3-8B concept targets; "
        "Task 4 contains the closed approximately 50K balanced MedSG release.\n",
    )
    atomic_text(
        root / "audits/v1_1_to_v1_2_planned_diff.md",
        "# Planned v1.1 to v1.2 diff\n\n"
        "- Byte-for-byte inherit Tasks 1, 2, 4, and 5 JSONL data.\n"
        "- Replace Task 3 with the frozen 35K Kimi K2.6 routing selection.\n"
        "- Preserve frozen train/test membership; do not add validation.\n"
        "- Keep all 3,107 QC-flagged rows pending human review.\n"
        "- Do not call an external LLM or modify v1.1.\n",
    )


def copy_inherited_tasks(destination: Path) -> dict[str, Any]:
    report = {}
    for task_id in INHERITED_TASKS:
        name, skill = TASKS[task_id]
        source = V11 / name
        target = destination / name
        target.mkdir(parents=True, exist_ok=True)
        for source_file in sorted(source.iterdir()):
            if source_file.is_file():
                shutil.copyfile(source_file, target / source_file.name)
        split_rows = {}
        for split in ("train", "test"):
            source_file = source / f"{split}.jsonl"
            target_file = target / f"{split}.jsonl"
            count = sum(1 for _ in jsonl(target_file))
            split_rows[split] = {
                "records": count,
                "source_sha256": sha256(source_file),
                "target_sha256": sha256(target_file),
                "byte_identical": sha256(source_file) == sha256(target_file),
            }
            write_json(
                target / f"{split}_manifest.json",
                {
                    "version": "v1.2",
                    "task_id": task_id,
                    "skill": skill,
                    "split": split,
                    "records": count,
                    "file": str(OUTPUT / name / f"{split}.jsonl"),
                    "inherited_from": str(source_file),
                    "content_sha256": sha256(target_file),
                },
            )
        report[name] = split_rows
    return report


def build_task3(
    destination: Path, mapping: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_dir = destination / TASKS[3][0]
    task_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {"train": [], "test": []}
    provenance = []
    frequency: collections.Counter[str] = collections.Counter()
    histogram: collections.Counter[int] = collections.Counter()
    modalities: collections.Counter[tuple[str, str]] = collections.Counter()
    for item in mapping["rows"]:
        record = task3_record(item)
        split = item["split"]
        rows_by_split[split].append(record)
        concepts = item["final"]["final_concepts"]
        histogram[len(concepts)] += 1
        modalities[(split, item["selection"].get("modality") or "unknown")] += 1
        if split == "train":
            frequency.update(normalized(value) for value in concepts)
        provenance.append(
            {
                "stable_id": item["stable_id"],
                "split": split,
                "modality": item["selection"].get("modality"),
                "source_caption": item["result"].get("source_caption"),
                "source_caption_sha256": item["final"]["source_caption_sha256"],
                "concepts": concepts,
                "image_paths": record["images"],
                "lineage_group_id": record["metadata"].get(
                    "lineage_group_id"
                ),
                "canonical_lineage_tokens": record["metadata"].get(
                    "canonical_lineage_tokens"
                ),
                "audit_flags": item["final"].get("audit_flags") or [],
                "quality_status": item["final"].get("quality_status"),
                "result_run_id": item["final"].get("result_run_id"),
                "prompt_hash": item["result"].get("prompt_hash"),
                "schema_hash": item["result"].get("schema_hash"),
                "run_config_hash": item["result"].get("run_config_hash"),
                "request_model": "k3",
                "routed_model": "kimi-k2.6",
                "reasoning_effort": "none",
            }
        )
    for split, values in rows_by_split.items():
        write_jsonl(task_dir / f"{split}.jsonl", values)
        write_json(
            task_dir / f"{split}_manifest.json",
            {
                "version": "v1.2",
                "task_id": 3,
                "skill": "concept_recognition",
                "split": split,
                "records": len(values),
                "file": str(OUTPUT / TASKS[3][0] / f"{split}.jsonl"),
                "selection": "frozen Kimi 35K group-aware split",
                "content_sha256": sha256(task_dir / f"{split}.jsonl"),
            },
        )
    statistics = {
        "task_id": 3,
        "task_name": TASKS[3][0],
        "skill": TASKS[3][1],
        "train": len(rows_by_split["train"]),
        "test": len(rows_by_split["test"]),
        "total": sum(map(len, rows_by_split.values())),
        "replacement_sampling_used": False,
    }
    write_json(task_dir / "statistics.json", statistics)
    vocabulary = {
        "train_only": True,
        "normalization": "Unicode casefold plus whitespace collapse",
        "frequency": dict(frequency.most_common()),
        "core_frequency_ge_10": sorted(
            key for key, value in frequency.items() if value >= 10
        ),
    }
    write_json(task_dir / "concept_vocabulary.json", vocabulary)
    shutil.copyfile(
        V11 / TASKS[3][0] / "concept_synonyms.json",
        task_dir / "concept_synonyms.json",
    )
    write_jsonl(destination / "manifests/task3_provenance.jsonl", provenance)
    write_json(
        destination / "metrics/concept_train_vocabulary.json", vocabulary
    )
    write_json(
        destination / "audits/concept_count_histogram.json",
        {
            "status": "PASS",
            "total": len(provenance),
            "histogram": dict(sorted(histogram.items())),
            "cap8_count": histogram[8],
            "cap8_ratio": histogram[8] / len(provenance),
        },
    )
    modality_path = destination / "audits/concept_modality_distribution.csv"
    atomic_text(
        modality_path,
        "split,modality,count\n"
        + "".join(
            f"{split},{modality},{count}\n"
            for (split, modality), count in sorted(modalities.items())
        ),
    )
    train_vocab = set(frequency)
    test_concepts = collections.Counter(
        normalized(value)
        for item in mapping["rows"]
        if item["split"] == "test"
        for value in item["final"]["final_concepts"]
    )
    oov = {
        key: value for key, value in test_concepts.items() if key not in train_vocab
    }
    write_json(
        destination / "metrics/concept_test_oov_audit.json",
        {
            "status": "PASS",
            "test_vocabulary_built_only_after_train_vocabulary_freeze": True,
            "train_vocabulary_size": len(train_vocab),
            "test_vocabulary_size": len(test_concepts),
            "oov_vocabulary_size": len(oov),
            "oov_occurrences": sum(oov.values()),
            "oov_frequency": dict(sorted(oov.items())),
        },
    )
    return statistics, provenance


def qc_outputs(destination: Path, provenance: list[dict[str, Any]]) -> None:
    source_queue = CLOSURE / "manual_audit_queue.jsonl"
    queue = list(jsonl(source_queue))
    by_id = {row["stable_id"]: row for row in provenance}
    if len(queue) != 3_107 or len({row["stable_id"] for row in queue}) != 3_107:
        raise RuntimeError("manual QC queue contract failed")
    shutil.copyfile(source_queue, destination / "audits/manual_qc_queue.jsonl")
    write_json(
        destination / "audits/manual_qc_index.json",
        {
            "status": "PASS",
            "queue_size": len(queue),
            "automatic_exclusion_count": 0,
            "stable_id_to_qc_flags": {
                row["stable_id"]: row.get("audit_flags") or []
                for row in queue
            },
        },
    )
    path = destination / "audits/manual_qc_template.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "stable_id",
                    "split",
                    "modality",
                    "source_caption",
                    "concepts",
                    "qc_flags",
                    "human_decision",
                    "human_notes",
                ],
            )
            writer.writeheader()
            for queue_row in queue:
                item = by_id[queue_row["stable_id"]]
                writer.writerow(
                    {
                        "stable_id": item["stable_id"],
                        "split": item["split"],
                        "modality": item["modality"],
                        "source_caption": item["source_caption"],
                        "concepts": json.dumps(
                            item["concepts"], ensure_ascii=False
                        ),
                        "qc_flags": json.dumps(
                            queue_row.get("audit_flags") or [],
                            ensure_ascii=False,
                        ),
                        "human_decision": "",
                        "human_notes": "",
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def readme() -> str:
    return """# MedicalSkill-CL-v1.2

MedicalSkill-CL-v1.2 is the formal five-stage training release:

1. `task_01_vqa`
2. `task_02_diagnosis_classification`
3. `task_03_concept_recognition`
4. `task_04_visual_grounding`
5. `task_05_reasoning_vqa`

Relative to v1.1, Tasks 1, 2, 4, and 5 are byte-identical. Task 3 is
replaced by the frozen 35,000-record Kimi K2.6 routing result. Its frozen
group-aware split contains 31,452 train and 3,548 test records. No validation
split exists.

Task 3 model input contains only the image and the Concept Recognition
instruction. Source captions are retained only in audit provenance and are
never exposed as model input. Concept order and phrases are preserved from the
frozen final manifest. The request model was `k3`, routed to `kimi-k2.6`, with
`reasoning_effort=none`; prompt/schema/run-config hashes are recorded in
`manifests/task3_provenance.jsonl`.

The 3,107 QC-flagged records remain included. They are pending partial human
review and are available in `audits/manual_qc_template.csv`. No automatic QC
exclusion was applied. Future reviewed changes must create v1.2.1 rather than
modify this release in place.

Images are referenced by their existing absolute server paths; no image assets
are duplicated. Training and evaluation operate directly on each task's
`train.jsonl` and `test.jsonl`. Formal experiments should train each task for
one epoch and evaluate on test only.

Training entry:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/medicalskill_v1_2_train_one_epoch.sh TASK_ID
```

Evaluation entry:

```bash
bash scripts/medicalskill_v1_2_evaluate_task.sh TASK_ID PREDICTIONS_JSONL OUTPUT_JSON
```
"""


def build(destination: Path, mapping: dict[str, Any]) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("manifests", "audits", "metrics", "smoke", "scripts"):
        (destination / name).mkdir(exist_ok=True)
    old_hashes = file_inventory(V11, include_hash=True)
    inventory_reports(destination, old_hashes)
    inherited = copy_inherited_tasks(destination)
    task3_stats, provenance = build_task3(destination, mapping)
    qc_outputs(destination, provenance)
    tasks = []
    for task_id, (name, skill) in TASKS.items():
        if task_id == 3:
            stats = task3_stats
        else:
            stats = json.loads((destination / name / "statistics.json").read_text())
        tasks.append(
            {
                "task_id": task_id,
                "task_name": name,
                "skill": skill,
                "train": stats["train"],
                "test": stats["test"],
            }
        )
    contract = upstream_contract()
    manifest = {
        "version": "v1.2",
        "dataset_status": "train_ready",
        "concept_generation_status": "complete",
        "manual_qc_status": "pending_partial_review",
        "manual_qc_queue_size": 3_107,
        "automatic_qc_exclusion_count": 0,
        "created_at": now(),
        "source_v1_1_read_only": str(V11),
        "task_order": tasks,
        "validation_split": None,
        "evaluate_on": "test",
        "recommended_epochs_per_task": 1,
        "upstream_kimi_closure": contract,
        "task3_source_count": 35_000,
        "task3_mapped_count": 35_000,
        "task3_missing_count": 0,
        "task3_duplicate_count": 0,
        "task3_unexplained_exclusion_count": 0,
        "image_storage": {
            "mode": "shared_absolute_paths",
            "images_copied": False,
            "source_roots": [
                "/remote-home/wangbomin/OmniMedVQA",
                "/remote-home/wangbomin/MedIMeta",
                "/remote-home/wangbomin/MedTrinity-25M",
                "/remote-home/wangbomin/MedSG",
                "/remote-home/wangbomin/MedThinkVQA",
            ],
        },
    }
    write_json(destination / "manifests/medicalskill_cl_v1_2_manifest.json", manifest)
    write_json(
        destination / "manifests/medicalskill_cl_v1_2_statistics.json",
        {
            "status": "PASS",
            "tasks": tasks,
            "total_train": sum(item["train"] for item in tasks),
            "total_test": sum(item["test"] for item in tasks),
            "total": sum(item["train"] + item["test"] for item in tasks),
        },
    )
    write_json(
        destination / "audits/v1_1_to_v1_2_diff.json",
        {
            "status": "PASS",
            "inherited_tasks": inherited,
            "task3": {
                "v1_1": {"train": 62_694, "test": 7_038},
                "v1_2": {
                    "train": task3_stats["train"],
                    "test": task3_stats["test"],
                },
                "target_source": contract,
                "automatic_exclusions": 0,
            },
            "caption_generation_restored": False,
            "validation_split_added": False,
        },
    )
    atomic_text(
        destination / "audits/v1_1_to_v1_2_diff.md",
        "# v1.1 to v1.2 diff\n\n"
        "Tasks 1, 2, 4, and 5 retain byte-identical train/test JSONL files. "
        "Task 3 changes from 62,694/7,038 Qwen targets to "
        f"{task3_stats['train']:,}/{task3_stats['test']:,} frozen Kimi targets. "
        "No validation or Caption Generation task was added.\n",
    )
    write_json(
        destination / "audits/task3_kimi_integration_report.json",
        {
            "status": "PASS",
            "source": 35_000,
            "mapped": 35_000,
            "retained": 35_000,
            "missing": 0,
            "duplicate": 0,
            "deleted": 0,
            "manual_qc_queue_size": 3_107,
            "assistant_target": "semicolon-separated final concepts in frozen order",
            "caption_used_as_model_input": False,
            "upstream": contract,
        },
    )
    atomic_text(
        destination / "audits/task3_kimi_integration_report.md",
        "# Task 3 Kimi integration\n\n"
        "Status: **PASS**\n\n"
        "- Source/mapped/retained: 35,000/35,000/35,000\n"
        "- Missing/duplicate/deleted: 0/0/0\n"
        "- Train/test: 31,452/3,548\n"
        "- Source caption exposed to model: no\n"
        "- QC queue retained: 3,107\n",
    )
    effective = json.loads((V11 / "configs/effective_config.json").read_text())
    effective.update(
        {
            "version": "v1.2",
            "data_root": str(OUTPUT),
            "concept_model_id": "kimi-k2.6",
            "concept_request_model_id": "k3",
            "concept_reasoning_effort": "none",
            "concept_prompt_hash": (
                "fb63d4b1963d5b9c93188ed65b7ec5677eeaf7ecb8e38b86bb73e640f516ae68"
            ),
            "concept_schema_hash": (
                "14830fcd9b334ac7ce24c17cd0704a373ccec247da03bf1435dad139a8d38aea"
            ),
            "concept_run_config_hash": (
                "382703cd787c9c49cce0ccdc2cda71615ccbb16f35820cc4a9ad290243d230a1"
            ),
        }
    )
    write_json(destination / "configs/effective_config.json", effective)
    metrics = json.loads((V11 / "configs/metric_contracts.json").read_text())
    metrics["task_03_concept_recognition"].update(
        {
            "target_parser": (
                "semicolon-separated Kimi final concepts; case/whitespace normalized"
            ),
            "prediction_parser": (
                "v1.1-compatible semicolon concept parser; status prefixes optional"
            ),
        }
    )
    write_json(destination / "configs/metric_contracts.json", metrics)
    atomic_text(destination / "README.md", readme())
    for script in (
        "medicalskill_v1_2_build.py",
        "medicalskill_v1_2_validate.py",
        "medicalskill_v1_2_model_smoke.py",
        "medicalskill_v1_2_train_one_epoch.sh",
        "medicalskill_v1_2_evaluate_task.sh",
    ):
        source = REPO / "scripts/skill_incremental_v1_2" / script
        if source.exists():
            shutil.copyfile(source, destination / "scripts" / script)
    atomic_text(
        destination / "manifests/build_commands.txt",
        "python scripts/skill_incremental_v1_2/medicalskill_v1_2_build.py --dry-run\n"
        "python scripts/skill_incremental_v1_2/medicalskill_v1_2_build.py --build\n"
        "python scripts/skill_incremental_v1_2/medicalskill_v1_2_validate.py --static\n"
        "CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        "python scripts/skill_incremental_v1_2/medicalskill_v1_2_model_smoke.py\n"
        "python scripts/skill_incremental_v1_2/medicalskill_v1_2_validate.py --finalize\n",
    )
    try:
        git_state = subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO, text=True
        )
    except subprocess.CalledProcessError as error:
        git_state = f"git status failed: {error}\n"
    atomic_text(destination / "manifests/git_state.txt", git_state)
    return manifest


def dry_run(mapping: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "status": "DRY_RUN_PASS",
        "created_at": now(),
        "output_root": str(OUTPUT),
        "output_exists": OUTPUT.exists(),
        "planned_task3": mapping["split_counts"],
        "planned_task3_total": len(mapping["rows"]),
        "planned_deletions": 0,
        "planned_qc_exclusions": 0,
        "planned_inherited_tasks": list(INHERITED_TASKS),
        "caption_generation_task": False,
        "validation_split": False,
        "upstream": upstream_contract(),
    }
    write_json(BUILD_ARTIFACT / "dry_run_summary.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not V11.is_dir():
        raise RuntimeError(f"missing v1.1 baseline: {V11}")
    upstream_contract()
    mapping = load_mapping()
    if args.dry_run:
        print(json.dumps(dry_run(mapping), indent=2, ensure_ascii=False))
        return 0
    if OUTPUT.exists():
        manifest_path = OUTPUT / "manifests/medicalskill_cl_v1_2_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("version") == "v1.2"
                and manifest.get("upstream_kimi_closure", {}).get(
                    "final_manifest_sha256"
                )
                == EXPECTED_FINAL_SHA
            ):
                print(
                    json.dumps(
                        {"status": "ALREADY_BUILT", "output": str(OUTPUT)}
                    )
                )
                return 0
        raise RuntimeError(f"refusing to overwrite existing output: {OUTPUT}")
    if STAGING.exists():
        marker = STAGING / ".medicalskill_v1_2_staging"
        if not marker.exists():
            raise RuntimeError(f"unrecognized staging directory: {STAGING}")
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    atomic_text(STAGING / ".medicalskill_v1_2_staging", "v1.2\n")
    manifest = build(STAGING, mapping)
    (STAGING / ".medicalskill_v1_2_staging").unlink()
    os.replace(STAGING, OUTPUT)
    print(
        json.dumps(
            {
                "status": "BUILT",
                "output": str(OUTPUT),
                "tasks": manifest["task_order"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
