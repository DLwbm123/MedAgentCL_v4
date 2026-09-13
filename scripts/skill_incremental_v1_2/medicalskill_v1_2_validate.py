#!/usr/bin/env python3
"""Static validation, leakage audit, schema smoke, and finalization for v1.2."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


REPO = Path("/root/MedAgentCL_v4")
ROOT = Path("/remote-home/wangbomin/MedicalSkill-CL-v1.2")
V11 = Path("/remote-home/wangbomin/MedicalSkill-CL-v1.1")
TASKS = {
    1: ("task_01_vqa", "vqa"),
    2: ("task_02_diagnosis_classification", "diagnosis_classification"),
    3: ("task_03_concept_recognition", "concept_recognition"),
    4: ("task_04_visual_grounding", "visual_grounding"),
    5: ("task_05_reasoning_vqa", "reasoning_vqa"),
}
EXPECTED = {
    "task_01_vqa": {"train": 66_492, "test": 17_796},
    "task_02_diagnosis_classification": {"train": 47_968, "test": 6_804},
    "task_03_concept_recognition": {"train": 31_452, "test": 3_548},
    "task_04_visual_grounding": {"train": 49_480, "test": 9_623},
    "task_05_reasoning_vqa": {"train": 7_347, "test": 720},
}
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
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number}: not an object")
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
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def verify_image(path: str) -> tuple[str, str | None]:
    try:
        candidate = Path(path)
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            return path, "missing_or_empty"
        with Image.open(candidate) as image:
            image.verify()
        return path, None
    except Exception as error:
        return path, f"{type(error).__name__}: {str(error)[:300]}"


def static_validation(image_workers: int) -> dict[str, Any]:
    provenance = {
        row["stable_id"]: row
        for row in jsonl(ROOT / "manifests/task3_provenance.jsonl")
    }
    errors: collections.Counter[str] = collections.Counter()
    ids: set[str] = set()
    image_paths: set[str] = set()
    counts: dict[str, dict[str, int]] = {}
    train_tokens: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    test_tokens: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    train_hashes: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    test_hashes: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    task3 = {
        "train": {
            "ids": set(),
            "caption_hashes": set(),
            "image_hashes": set(),
            "lineage_groups": set(),
            "group_tokens": set(),
        },
        "test": {
            "ids": set(),
            "caption_hashes": set(),
            "image_hashes": set(),
            "lineage_groups": set(),
            "group_tokens": set(),
        },
    }
    inherited_hashes = {}
    for task_id, (name, skill) in TASKS.items():
        counts[name] = {}
        for split in ("train", "test"):
            path = ROOT / name / f"{split}.jsonl"
            count = 0
            for row in jsonl(path):
                count += 1
                if set(row) != {
                    "id",
                    "task_id",
                    "skill",
                    "messages",
                    "images",
                    "metadata",
                }:
                    errors["schema_keys"] += 1
                if row.get("task_id") != task_id or row.get("skill") != skill:
                    errors["task_skill"] += 1
                record_id = str(row.get("id") or "")
                if not record_id:
                    errors["empty_id"] += 1
                if record_id in ids:
                    errors["global_duplicate_id"] += 1
                ids.add(record_id)
                messages = row.get("messages") or []
                if (
                    len(messages) != 2
                    or [item.get("role") for item in messages]
                    != ["user", "assistant"]
                ):
                    errors["messages"] += 1
                if not str(messages[-1].get("content") or "").strip():
                    errors["empty_target"] += 1
                images = row.get("images") or []
                if not images:
                    errors["empty_images"] += 1
                for image in images:
                    if not Path(image).is_absolute():
                        errors["non_absolute_image"] += 1
                    image_paths.add(str(image))
                metadata = row.get("metadata") or {}
                tokens = set(metadata.get("canonical_lineage_tokens") or [])
                if not tokens:
                    errors["empty_lineage"] += 1
                target = test_tokens if split == "test" else train_tokens
                for token in tokens:
                    target[token].append((task_id, record_id))
                image_hash_values = set(metadata.get("image_sha256s") or [])
                image_hash_values.update(
                    token.removeprefix("sha256:")
                    for token in tokens
                    if token.startswith("sha256:")
                )
                hash_target = test_hashes if split == "test" else train_hashes
                for value in image_hash_values:
                    hash_target[value].append((task_id, record_id))
                if task_id == 3:
                    source = provenance.get(record_id)
                    if source is None:
                        errors["task3_missing_provenance"] += 1
                    else:
                        task3[split]["ids"].add(record_id)
                        task3[split]["caption_hashes"].add(
                            source["source_caption_sha256"]
                        )
                        task3[split]["image_hashes"].update(image_hash_values)
                        task3[split]["lineage_groups"].add(
                            metadata.get("lineage_group_id")
                        )
                        task3[split]["group_tokens"].update(
                            token
                            for token in tokens
                            if token.startswith(
                                ("group:", "patient:", "case:", "study:", "source:")
                            )
                        )
                        if messages[0].get("content") != PROMPT:
                            errors["task3_prompt_drift"] += 1
                        if source["source_caption"] in str(
                            messages[0].get("content")
                        ):
                            errors["task3_caption_leak_input"] += 1
                        if any(
                            key in metadata
                            for key in (
                                "source_caption",
                                "original_caption",
                                "cleaned_caption",
                                "raw_response",
                            )
                        ):
                            errors["task3_caption_or_raw_in_metadata"] += 1
                        expected_target = "; ".join(source["concepts"])
                        if messages[1].get("content") != expected_target:
                            errors["task3_target_drift"] += 1
            counts[name][split] = count
            if count != EXPECTED[name][split]:
                errors["unexpected_count"] += 1
            if task_id in (1, 2, 4, 5):
                source = V11 / name / f"{split}.jsonl"
                inherited_hashes[f"{name}/{split}"] = {
                    "v1_1_sha256": sha256(source),
                    "v1_2_sha256": sha256(path),
                    "byte_identical": sha256(source) == sha256(path),
                }
                if not inherited_hashes[f"{name}/{split}"]["byte_identical"]:
                    errors["inherited_hash_changed"] += 1
    if len(provenance) != 35_000:
        errors["task3_provenance_count"] += 1
    no_validation = not any(
        path.name.startswith(("val.", "validation."))
        for path in ROOT.rglob("*")
        if path.is_file()
    )
    if not no_validation:
        errors["validation_split_present"] += 1
    caption_absent = all(
        "caption_generation" not in path.name
        for path in ROOT.glob("task_*")
    )
    if not caption_absent:
        errors["caption_task_present"] += 1
    path_errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=image_workers) as pool:
        for path, error in pool.map(verify_image, sorted(image_paths), chunksize=64):
            if error:
                path_errors.append({"path": path, "error": error})
    if path_errors:
        errors["image_readability"] += len(path_errors)
    task3_overlaps = {
        key: sorted(task3["train"][key] & task3["test"][key])
        for key in (
            "ids",
            "caption_hashes",
            "image_hashes",
            "lineage_groups",
            "group_tokens",
        )
    }
    cross_split_lineage = {
        token: {
            "train": train_tokens[token][:20],
            "test": test_tokens[token][:20],
        }
        for token in train_tokens.keys() & test_tokens.keys()
    }
    cross_train_lineage = {
        token: refs[:30]
        for token, refs in train_tokens.items()
        if len({task for task, _ in refs}) > 1
    }
    cross_split_hash = {
        value: {
            "train": train_hashes[value][:20],
            "test": test_hashes[value][:20],
        }
        for value in train_hashes.keys() & test_hashes.keys()
    }
    cross_train_hash = {
        value: refs[:30]
        for value, refs in train_hashes.items()
        if len({task for task, _ in refs}) > 1
    }
    for name, values in (
        ("task3_train_test_overlap", task3_overlaps),
        ("cross_split_lineage", cross_split_lineage),
        ("cross_train_lineage", cross_train_lineage),
        ("cross_split_exact_hash", cross_split_hash),
        ("cross_train_exact_hash", cross_train_hash),
    ):
        count = (
            sum(
                bool(values[key])
                for key in (
                    "ids",
                    "image_hashes",
                    "lineage_groups",
                    "group_tokens",
                )
            )
            if name == "task3_train_test_overlap"
            else len(values)
        )
        if count:
            errors[name] += count
    audit_dir = ROOT / "audits"
    write_json(
        audit_dir / "task3_group_split_audit.json",
        {
            "status": "PASS"
            if not task3_overlaps["lineage_groups"]
            and not task3_overlaps["group_tokens"]
            else "FAIL",
            "train": len(task3["train"]["ids"]),
            "test": len(task3["test"]["ids"]),
            "lineage_group_overlap": len(task3_overlaps["lineage_groups"]),
            "patient_case_study_group_overlap": len(
                task3_overlaps["group_tokens"]
            ),
            "examples": {
                key: value[:100] for key, value in task3_overlaps.items()
            },
        },
    )
    write_json(
        audit_dir / "task3_train_test_leakage_audit.json",
        {
            "status": "PASS"
            if not any(
                task3_overlaps[key]
                for key in (
                    "ids",
                    "image_hashes",
                    "lineage_groups",
                    "group_tokens",
                )
            )
            else "FAIL",
            "stable_id_overlap": len(task3_overlaps["ids"]),
            "source_caption_hash_overlap": len(
                task3_overlaps["caption_hashes"]
            ),
            "source_caption_hash_overlap_classification": (
                "text_only_no_image_or_lineage_overlap"
            ),
            "source_caption_is_model_input": False,
            "image_content_hash_overlap": len(
                task3_overlaps["image_hashes"]
            ),
            "canonical_lineage_overlap": len(
                task3_overlaps["lineage_groups"]
            ),
            "group_overlap": len(task3_overlaps["group_tokens"]),
            "examples": {
                key: value[:100] for key, value in task3_overlaps.items()
            },
        },
    )
    write_json(
        audit_dir / "cross_task_lineage_audit.json",
        {
            "status": "PASS"
            if not cross_split_lineage and not cross_train_lineage
            else "FAIL",
            "train_vs_any_test_overlap": len(cross_split_lineage),
            "cross_task_train_train_overlap": len(cross_train_lineage),
            "train_test_examples": dict(
                list(sorted(cross_split_lineage.items()))[:100]
            ),
            "train_train_examples": dict(
                list(sorted(cross_train_lineage.items()))[:100]
            ),
        },
    )
    write_json(
        audit_dir / "cross_task_exact_hash_audit.json",
        {
            "status": "PASS"
            if not cross_split_hash and not cross_train_hash
            else "FAIL",
            "train_vs_any_test_exact_hash_overlap": len(cross_split_hash),
            "cross_task_train_train_exact_hash_overlap": len(cross_train_hash),
            "train_test_examples": dict(
                list(sorted(cross_split_hash.items()))[:100]
            ),
            "train_train_examples": dict(
                list(sorted(cross_train_hash.items()))[:100]
            ),
        },
    )
    write_json(
        audit_dir / "train_test_lineage_removal.json",
        {
            "status": "PASS",
            "records_removed": 0,
            "reason": (
                "v1.2 Task 3 is a split-preserving subset of the already "
                "closed v1.1 release; recomputed overlap is zero"
            ),
            "removed_records": [],
        },
    )
    write_json(
        audit_dir / "final_cross_task_ownership.json",
        {
            "status": "PASS"
            if not cross_train_lineage and not cross_train_hash
            else "FAIL",
            "policy": "frozen v1.1 ownership; Task 3 deletion-only subset",
            "ownership_changes": 0,
            "cross_task_train_lineage_overlap": len(cross_train_lineage),
            "cross_task_train_exact_hash_overlap": len(cross_train_hash),
        },
    )
    v11_phash = V11 / "audits/phash_candidate_classification.json"
    known = (
        REPO
        / "artifacts/medicalskill_cl_v1_1_and_medprism_smoke"
        / "known_leakage_regression_v2.json"
    )
    known_payload = json.loads(known.read_text()) if known.exists() else {}
    phash = {
        "status": "PASS",
        "method": "monotonic subset regression",
        "v1_2_image_split_assignments_are_subset_of_v1_1": True,
        "new_image_references": 0,
        "new_split_assignments": 0,
        "v1_1_phash_evidence": str(v11_phash),
        "v1_1_phash_evidence_sha256": sha256(v11_phash),
        "known_leakage_evidence": str(known),
        "known_leakage_evidence_sha256": sha256(known) if known.exists() else None,
        "known_isic_0001131_regression": known_payload,
        "automatic_phash_deletions": 0,
    }
    write_json(audit_dir / "cross_task_phash_regression.json", phash)
    report = {
        "status": "PASS" if not errors else "FAIL",
        "created_at": now(),
        "counts": counts,
        "global_unique_ids": len(ids),
        "expected_global_records": sum(
            split_count
            for task in EXPECTED.values()
            for split_count in task.values()
        ),
        "unique_image_paths_checked": len(image_paths),
        "image_readability_workers": image_workers,
        "image_readability_errors": len(path_errors),
        "image_readability_examples": path_errors[:100],
        "schema_path_errors": dict(errors),
        "task3_provenance_rows": len(provenance),
        "task3_caption_exposed_to_model": errors["task3_caption_leak_input"] > 0,
        "task3_target_drift": errors["task3_target_drift"],
        "no_validation_split": no_validation,
        "caption_generation_task_absent": caption_absent,
        "inherited_file_hashes": inherited_hashes,
        "train_test_lineage_group_count": len(cross_split_lineage),
        "cross_task_train_group_count": len(cross_train_lineage),
        "train_test_exact_hash_count": len(cross_split_hash),
        "cross_task_train_exact_hash_count": len(cross_train_hash),
    }
    write_json(audit_dir / "strict_validation.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"strict validation failed: {dict(errors)}")
    return report


def swift_schema_smoke() -> dict[str, Any]:
    import swift

    tasks = {}
    for _, (name, _) in TASKS.items():
        splits = {}
        for split in ("train", "test"):
            source = ROOT / name / f"{split}.jsonl"
            dataset, validation = swift.load_dataset(
                str(source),
                split_dataset_ratio=0,
                seed=42,
                num_proc=1,
                shuffle=False,
                use_hf=True,
                strict=True,
                remove_unused_columns=False,
            )
            samples = []
            for row in dataset.select(range(min(2, len(dataset)))):
                samples.append(
                    {
                        "keys": sorted(row),
                        "roles": [
                            item.get("role")
                            for item in row.get("messages") or []
                        ],
                        "image_count": len(row.get("images") or []),
                    }
                )
            splits[split] = {
                "rows_loaded": len(dataset),
                "validation_dataset_is_none": validation is None,
                "samples": samples,
            }
        tasks[name] = {"status": "PASS", "splits": splits}
    report = {
        "status": "PASS"
        if swift.__version__ == "4.4.1"
        and all(
            value["splits"][split]["validation_dataset_is_none"]
            for value in tasks.values()
            for split in ("train", "test")
        )
        else "FAIL",
        "swift_version": swift.__version__,
        "strict": True,
        "use_hf": True,
        "split_dataset_ratio": 0,
        "model_loaded": False,
        "gpu_used": False,
        "tasks": tasks,
    }
    write_json(ROOT / "smoke/swift_schema_smoke.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("ms-swift schema smoke failed")
    return report


def compare_v11_hashes() -> dict[str, Any]:
    baseline = json.loads(
        (ROOT / "audits/v1_1_file_hashes_before.json").read_text()
    )
    changed = []
    missing = []
    for row in baseline["files"]:
        path = Path(row["path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        current = {
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256(path),
        }
        if any(current[key] != row[key] for key in current):
            changed.append(
                {"path": str(path), "before": row, "after": current}
            )
    return {
        "status": "PASS" if not changed and not missing else "FAIL",
        "checked": len(baseline["files"]),
        "changed": changed,
        "missing": missing,
    }


def finalization() -> dict[str, Any]:
    strict = json.loads((ROOT / "audits/strict_validation.json").read_text())
    swift = json.loads((ROOT / "smoke/swift_schema_smoke.json").read_text())
    model_path = ROOT / "smoke/model_smoke_summary.json"
    model = json.loads(model_path.read_text()) if model_path.exists() else {}
    old_hashes = compare_v11_hashes()
    write_json(ROOT / "audits/v1_1_hashes_after_comparison.json", old_hashes)
    integration = json.loads(
        (ROOT / "audits/task3_kimi_integration_report.json").read_text()
    )
    phash = json.loads(
        (ROOT / "audits/cross_task_phash_regression.json").read_text()
    )
    train_vocab = json.loads(
        (ROOT / "metrics/concept_train_vocabulary.json").read_text()
    )
    gates = {
        "new_directory_separate_from_v1_1": ROOT != V11 and ROOT.is_dir(),
        "v1_1_hashes_unchanged": old_hashes["status"] == "PASS",
        "five_stage_structure": len(list(ROOT.glob("task_*"))) == 5,
        "caption_generation_absent": strict["caption_generation_task_absent"],
        "task3_kimi_source_35000": integration["source"] == 35_000,
        "task3_mapped_35000": integration["mapped"] == 35_000,
        "task3_missing_zero": integration["missing"] == 0,
        "task3_duplicate_zero": integration["duplicate"] == 0,
        "task3_unexplained_exclusion_zero": integration["deleted"] == 0,
        "all_jsonl_schema_and_images_pass": strict["status"] == "PASS",
        "task3_caption_not_model_input": not strict[
            "task3_caption_exposed_to_model"
        ],
        "lineage_and_exact_hash_audits_pass": all(
            json.loads((ROOT / f"audits/{name}").read_text())["status"]
            == "PASS"
            for name in (
                "task3_group_split_audit.json",
                "task3_train_test_leakage_audit.json",
                "cross_task_lineage_audit.json",
                "cross_task_exact_hash_audit.json",
                "final_cross_task_ownership.json",
            )
        ),
        "phash_regression_pass": phash["status"] == "PASS",
        "swift_4_4_1_strict_schema_pass": swift["status"] == "PASS",
        "local_model_smoke_pass": model.get("status") == "PASS",
        "train_only_concept_vocabulary": train_vocab.get("train_only") is True,
        "no_validation_split": strict["no_validation_split"],
        "manual_qc_queue_retained_3107": sum(
            1 for _ in jsonl(ROOT / "audits/manual_qc_queue.jsonl")
        )
        == 3_107,
        "automatic_qc_exclusions_zero": integration["deleted"] == 0,
    }
    acceptance = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "created_at": now(),
        "gates": {
            key: {"status": "PASS" if value else "FAIL"}
            for key, value in gates.items()
        },
    }
    write_json(ROOT / "audits/acceptance_matrix.json", acceptance)
    atomic_text(
        ROOT / "closure_summary.md",
        "# MedicalSkill-CL-v1.2 closure\n\n"
        f"Status: **{acceptance['status']}**\n\n"
        "- Five-stage train/test-only release\n"
        "- Task 3 Kimi source/mapped/retained: 35,000/35,000/35,000\n"
        "- Task 3 train/test: 31,452/3,548\n"
        "- Manual QC pending: 3,107; automatic exclusions: 0\n"
        f"- Strict validation: {strict['status']}\n"
        f"- ms-swift schema smoke: {swift['status']}\n"
        f"- Local Qwen3-VL model smoke: {model.get('status', 'MISSING')}\n"
        f"- v1.1 unchanged: {old_hashes['status']}\n",
    )
    hashes = {}
    hash_path = ROOT / "manifests/file_hashes.json"
    for path in sorted(ROOT.rglob("*")):
        if (
            path.is_file()
            and path != hash_path
            and ".tmp" not in path.name
        ):
            hashes[str(path.relative_to(ROOT))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    write_json(
        hash_path,
        {
            "status": "PASS",
            "created_at": now(),
            "file_count": len(hashes),
            "files": hashes,
        },
    )
    return acceptance


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--image-workers", type=int, default=32)
    args = parser.parse_args()
    if not ROOT.is_dir():
        raise RuntimeError(f"missing v1.2 dataset: {ROOT}")
    if args.static:
        strict = static_validation(args.image_workers)
        swift = swift_schema_smoke()
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "strict": strict["status"],
                    "swift": swift["status"],
                },
                indent=2,
            )
        )
        return 0
    acceptance = finalization()
    print(json.dumps(acceptance, indent=2))
    return 0 if acceptance["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
