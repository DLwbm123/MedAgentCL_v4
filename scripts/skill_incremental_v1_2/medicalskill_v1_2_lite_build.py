#!/usr/bin/env python3
"""Build and validate the frozen MedicalSkill-CL-v1.2-lite-10k1k dataset."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


VERSION = "MedicalSkill-CL-v1.2-lite-10k1k"
PARENT_VERSION = "MedicalSkill-CL-v1.2"
DEFAULT_PARENT = Path("/remote-home/wangbomin/MedicalSkill-CL-v1.2")
DEFAULT_OUTPUT = Path("/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k")
TASKS = {
    1: ("task_01_vqa", "vqa"),
    2: ("task_02_diagnosis_classification", "diagnosis_classification"),
    3: ("task_03_concept_recognition", "concept_recognition"),
    4: ("task_04_visual_grounding", "visual_grounding"),
    5: ("task_05_reasoning_vqa", "reasoning_vqa"),
}
SCHEMA = {"id", "task_id", "skill", "messages", "images", "metadata"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: Any) -> str:
    return hash_bytes("\0".join(str(part) for part in parts).encode())


def normalize(value: Any) -> str:
    return " ".join(str(value or "unknown").casefold().split())


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_bytes(path, payload.encode())


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values
    )
    atomic_bytes(path, payload.encode())


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number}: expected JSON object")
            yield row


def load_entries(path: Path) -> list[dict[str, Any]]:
    entries = []
    with path.open("rb") as handle:
        for source_index, raw in enumerate(handle):
            if not raw.strip():
                continue
            clean = raw.rstrip(b"\r\n")
            entries.append(
                {
                    "row": json.loads(clean.decode("utf-8-sig")),
                    "raw": clean + b"\n",
                    "source_index": source_index,
                    "record_sha256": hash_bytes(clean),
                }
            )
    return entries


def meta(entry: dict[str, Any]) -> dict[str, Any]:
    value = entry["row"].get("metadata")
    return value if isinstance(value, dict) else {}


def concepts(entry: dict[str, Any]) -> list[str]:
    values = meta(entry).get("all_target_concepts")
    if not isinstance(values, list):
        values = meta(entry).get("core_target_concepts") or []
    return [normalize(value) for value in values if str(value).strip()]


def eligible(entry: dict[str, Any], task_id: int) -> bool:
    row = entry["row"]
    basic = (
        set(row) == SCHEMA
        and bool(str(row.get("id") or "").strip())
        and bool(row.get("images"))
        and bool(row.get("messages"))
        and bool(meta(entry).get("canonical_lineage_tokens"))
    )
    return basic and (
        task_id != 3 or meta(entry).get("manual_qc_status") == "not_flagged"
    )


def stratum(entry: dict[str, Any], task_id: int) -> tuple[str, str]:
    metadata = meta(entry)
    if task_id == 1:
        return normalize(metadata.get("source_dataset")), "|".join(
            [normalize(metadata.get("modality")), normalize(metadata.get("question_type"))]
        )
    if task_id == 2:
        return normalize(metadata.get("dataset_id")), normalize(
            metadata.get("canonical_label")
        )
    if task_id == 3:
        return normalize(metadata.get("modality")), f"concept_count_{len(concepts(entry))}"
    if task_id == 4:
        return normalize(metadata.get("medsg_task_name")), (
            f"image_count_{len(entry['row'].get('images') or [])}"
        )
    primary = "longitudinal" if metadata.get("is_longitudinal") else "single_or_other"
    length_bucket = min(9, int(metadata.get("reasoning_length") or 0) // 500)
    return primary, (
        f"images_{len(entry['row'].get('images') or [])}|reasoning_{length_bucket}"
    )


def apportion(capacities: dict[str, int], target: int) -> dict[str, int]:
    """Allocate an exact, approximately equal quota under finite capacities."""
    if target < 0 or target > sum(capacities.values()):
        raise ValueError("target outside total capacity")
    quota = {key: 0 for key in sorted(capacities)}
    remaining = target
    while remaining:
        active = [key for key in sorted(capacities) if quota[key] < capacities[key]]
        share = max(1, remaining // len(active))
        for key in active:
            add = min(share, capacities[key] - quota[key], remaining)
            quota[key] += add
            remaining -= add
            if not remaining:
                break
    return quota


def select(
    entries: list[dict[str, Any]], task_id: int, split: str, cap: int, seed: int
) -> list[dict[str, Any]]:
    candidates = [entry for entry in entries if eligible(entry, task_id)]
    target = min(cap, len(candidates))
    if target == len(candidates):
        return sorted(candidates, key=lambda entry: entry["source_index"])
    frequency: collections.Counter[str] = collections.Counter()
    if task_id == 3:
        for entry in candidates:
            frequency.update(set(concepts(entry)))
    primary_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for entry in candidates:
        primary_groups[stratum(entry, task_id)[0]].append(entry)
    primary_quota = apportion(
        {key: len(group) for key, group in primary_groups.items()}, target
    )
    selected = []
    for primary in sorted(primary_groups):
        secondary_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for entry in primary_groups[primary]:
            secondary_groups[stratum(entry, task_id)[1]].append(entry)
        secondary_quota = apportion(
            {key: len(group) for key, group in secondary_groups.items()},
            primary_quota[primary],
        )
        for secondary in sorted(secondary_groups):
            def rank(entry: dict[str, Any]) -> tuple[Any, ...]:
                tie = stable_hash(seed, task_id, split, entry["row"]["id"])
                if task_id != 3:
                    return (tie,)
                rare_score = sum(
                    1.0 / math.sqrt(frequency[value])
                    for value in set(concepts(entry))
                    if frequency[value]
                )
                return (-rare_score, tie)

            ranked = sorted(secondary_groups[secondary], key=rank)
            selected.extend(ranked[: secondary_quota[secondary]])
    if len(selected) != target:
        raise RuntimeError(f"selection count mismatch: {len(selected)} != {target}")
    return sorted(selected, key=lambda entry: entry["source_index"])


def parent_contract(parent: Path) -> dict[str, Any]:
    acceptance_path = parent / "audits/acceptance_matrix.json"
    manifest_path = parent / "manifests/medicalskill_cl_v1_2_manifest.json"
    if not acceptance_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("parent acceptance or manifest is missing")
    acceptance = json.loads(acceptance_path.read_text())
    if acceptance.get("status") != "PASS":
        raise RuntimeError("parent MedicalSkill-CL-v1.2 is not accepted")
    files = [acceptance_path, manifest_path]
    for name, _skill in TASKS.values():
        files.extend([parent / name / "train.jsonl", parent / name / "test.jsonl"])
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing parent files: {missing}")
    return {
        "root": str(parent),
        "version": parent.name,
        "acceptance_status": acceptance["status"],
        "files": {
            str(path.relative_to(parent)): {
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
            for path in files
        },
    }


def summarize(entries: list[dict[str, Any]], task_id: int) -> dict[str, Any]:
    primary: collections.Counter[str] = collections.Counter()
    secondary: collections.Counter[str] = collections.Counter()
    modality: collections.Counter[str] = collections.Counter()
    concept_count: collections.Counter[str] = collections.Counter()
    for entry in entries:
        first, second = stratum(entry, task_id)
        primary[first] += 1
        secondary[f"{first}::{second}"] += 1
        modality[normalize(meta(entry).get("modality"))] += 1
        if task_id == 3:
            concept_count[str(len(concepts(entry)))] += 1
    return {
        "count": len(entries),
        "primary_strata": dict(sorted(primary.items())),
        "secondary_strata": dict(sorted(secondary.items())),
        "modalities": dict(sorted(modality.items())),
        "concept_count_histogram": dict(sorted(concept_count.items())),
    }


def build_concept_metrics(root: Path, parent: Path) -> None:
    task = root / "task_03_concept_recognition"
    train_frequency: collections.Counter[str] = collections.Counter()
    test_frequency: collections.Counter[str] = collections.Counter()
    for row in read_jsonl(task / "train.jsonl"):
        train_frequency.update(normalize(x) for x in row["metadata"]["all_target_concepts"])
    for row in read_jsonl(task / "test.jsonl"):
        test_frequency.update(normalize(x) for x in row["metadata"]["all_target_concepts"])
    vocabulary = {
        "core_frequency_ge_10": sorted(
            value for value, count in train_frequency.items() if count >= 10
        ),
        "frequency": dict(sorted(train_frequency.items())),
        "size": len(train_frequency),
        "threshold": 10,
    }
    write_json(task / "concept_vocabulary.json", vocabulary)
    synonyms = parent / "task_03_concept_recognition/concept_synonyms.json"
    if synonyms.is_file():
        shutil.copyfile(synonyms, task / "concept_synonyms.json")
    write_json(root / "metrics/concept_train_vocabulary.json", vocabulary)
    oov = {key: count for key, count in test_frequency.items() if key not in train_frequency}
    write_json(
        root / "metrics/concept_test_oov_audit.json",
        {
            "oov_frequency": dict(sorted(oov.items())),
            "oov_occurrences": sum(oov.values()),
            "oov_unique": len(oov),
            "test_unique": len(test_frequency),
            "train_unique": len(train_frequency),
        },
    )


def lineage(row: dict[str, Any]) -> tuple[set[str], set[str]]:
    metadata = row.get("metadata") or {}
    tokens = {str(x) for x in metadata.get("canonical_lineage_tokens") or []}
    hashes = {str(x) for x in metadata.get("image_sha256s") or []}
    hashes.update(token[7:] for token in tokens if token.startswith("sha256:"))
    return tokens, hashes


def verify_image(path: str) -> tuple[str, str | None]:
    try:
        candidate = Path(path)
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            return path, "missing_or_empty"
        with Image.open(candidate) as image:
            image.verify()
        return path, None
    except Exception as error:
        return path, f"{type(error).__name__}: {str(error)[:200]}"


def validate(
    root: Path, parent: Path, train_cap: int, test_cap: int,
    image_workers: int, pil_sample: int,
) -> dict[str, Any]:
    errors: collections.Counter[str] = collections.Counter()
    counts: dict[str, dict[str, int]] = {}
    train_tokens: set[str] = set()
    test_tokens: set[str] = set()
    train_hashes: set[str] = set()
    test_hashes: set[str] = set()
    images: set[str] = set()
    selected_rows = list(read_jsonl(root / "manifests/selected_ids.jsonl"))
    selected_index = {
        (row["task_name"], row["split"], row["id"]): row for row in selected_rows
    }
    expected_selected = 0
    global_ids: set[str] = set()
    for task_id, (name, skill) in TASKS.items():
        counts[name] = {}
        for split, cap in (("train", train_cap), ("test", test_cap)):
            entries = load_entries(root / name / f"{split}.jsonl")
            parent_entries = load_entries(parent / name / f"{split}.jsonl")
            eligible_parent = sum(eligible(entry, task_id) for entry in parent_entries)
            expected = min(cap, eligible_parent)
            counts[name][split] = len(entries)
            if len(entries) != expected:
                errors["count_mismatch"] += 1
            id_file = root / f"manifests/ids/{name}_{split}_ids.txt"
            listed_ids = [line for line in id_file.read_text().splitlines() if line]
            actual_ids = [entry["row"].get("id") for entry in entries]
            if listed_ids != actual_ids:
                errors["id_list_mismatch"] += 1
            if len(actual_ids) != len(set(actual_ids)):
                errors["duplicate_id_in_split"] += 1
            for entry in entries:
                row = entry["row"]
                record_id = str(row.get("id") or "")
                if record_id in global_ids:
                    errors["global_duplicate_id"] += 1
                global_ids.add(record_id)
                if set(row) != SCHEMA:
                    errors["schema"] += 1
                if row.get("task_id") != task_id or row.get("skill") != skill:
                    errors["task_skill"] += 1
                messages = row.get("messages") or []
                if len(messages) != 2 or [x.get("role") for x in messages] != [
                    "user", "assistant"
                ]:
                    errors["messages"] += 1
                if not str((messages[-1] if messages else {}).get("content") or "").strip():
                    errors["empty_target"] += 1
                row_images = row.get("images") or []
                if not row_images:
                    errors["empty_images"] += 1
                images.update(str(x) for x in row_images)
                tokens, hashes = lineage(row)
                if not tokens:
                    errors["empty_lineage"] += 1
                if split == "train":
                    train_tokens.update(tokens)
                    train_hashes.update(hashes)
                else:
                    test_tokens.update(tokens)
                    test_hashes.update(hashes)
                selected = selected_index.get((name, split, record_id))
                if not selected:
                    errors["missing_selected_id"] += 1
                elif selected["record_sha256"] != entry["record_sha256"]:
                    errors["record_hash_mismatch"] += 1
            expected_selected += len(entries)
    if len(selected_rows) != expected_selected or len(selected_index) != expected_selected:
        errors["selected_id_manifest_count"] += 1
    token_overlap = sorted(train_tokens & test_tokens)
    hash_overlap = sorted(train_hashes & test_hashes)
    if token_overlap:
        errors["train_test_lineage_overlap"] += len(token_overlap)
    if hash_overlap:
        errors["train_test_image_hash_overlap"] += len(hash_overlap)
    missing = [
        path for path in sorted(images)
        if not Path(path).is_file() or Path(path).stat().st_size <= 0
    ]
    if missing:
        errors["missing_or_empty_image"] += len(missing)
    ranked = sorted(images, key=lambda path: stable_hash(42, "pil", path))
    sampled = ranked[: min(pil_sample, len(ranked))]
    pil_failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=image_workers) as executor:
        for path, failure in executor.map(verify_image, sampled):
            if failure:
                pil_failures.append({"path": path, "error": failure})
    if pil_failures:
        errors["pil_unreadable_image"] += len(pil_failures)
    result = {
        "status": "PASS" if not errors else "FAIL",
        "checked_at": now(),
        "counts": counts,
        "errors": dict(errors),
        "selected_id_rows": len(selected_rows),
        "lineage": {
            "train_test_token_overlap": len(token_overlap),
            "train_test_image_hash_overlap": len(hash_overlap),
            "token_overlap_examples": token_overlap[:20],
            "image_hash_overlap_examples": hash_overlap[:20],
        },
        "images": {
            "unique_paths": len(images),
            "stat_checked": len(images),
            "missing_or_empty": len(missing),
            "pil_sample_size": len(sampled),
            "pil_failures": pil_failures[:20],
            "parent_full_image_validation_inherited": True,
        },
    }
    if errors:
        raise RuntimeError(f"dataset validation failed: {dict(errors)}")
    return result


def build(args: argparse.Namespace) -> None:
    parent = args.parent.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing dataset: {output}")
    before = parent_contract(parent)
    staging = output.parent / f".{output.name}.building.{os.getpid()}"
    if staging.exists():
        raise RuntimeError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    config = {
        "dataset_version": VERSION,
        "parent_version": PARENT_VERSION,
        "parent_root": str(parent),
        "output_root": str(output),
        "seed": args.seed,
        "max_train_per_task": args.max_train,
        "max_test_per_task": args.max_test,
        "validation_split_used": False,
        "task_order": [name for name, _skill in TASKS.values()],
        "selection_contract": {
            "record_bytes": "copied unchanged from parent JSONL",
            "task_01": "equal source_dataset; then modality and question_type",
            "task_02": "equal dataset_id; then canonical_label",
            "task_03": "manual_qc_status=not_flagged only; equal modality and concept count; rare concepts ranked first",
            "task_04": "equal MedSG task; then image count",
            "task_05": "retain all because source is below both caps",
            "tie_break": "sha256(seed, task_id, split, stable_id)",
        },
    }
    write_json(staging / "configs/effective_config.json", config)
    selected_manifest: list[dict[str, Any]] = []
    task_manifests = []
    reproducibility = {}
    for task_id, (name, skill) in TASKS.items():
        split_stats = {}
        split_manifest = {}
        for split, cap in (("train", args.max_train), ("test", args.max_test)):
            source = parent / name / f"{split}.jsonl"
            entries = load_entries(source)
            chosen = select(entries, task_id, split, cap, args.seed)
            repeated = select(entries, task_id, split, cap, args.seed)
            ids = [entry["row"]["id"] for entry in chosen]
            repeated_ids = [entry["row"]["id"] for entry in repeated]
            reproducibility[f"{name}/{split}"] = {
                "same_ids_on_second_selection": ids == repeated_ids,
                "selected_ids_sha256": hash_bytes(("\n".join(ids) + "\n").encode()),
            }
            destination = staging / name / f"{split}.jsonl"
            atomic_bytes(destination, b"".join(entry["raw"] for entry in chosen))
            ids_path = staging / f"manifests/ids/{name}_{split}_ids.txt"
            atomic_bytes(ids_path, ("\n".join(ids) + "\n").encode())
            for entry in chosen:
                primary, secondary = stratum(entry, task_id)
                selected_manifest.append(
                    {
                        "dataset_version": VERSION,
                        "parent_version": PARENT_VERSION,
                        "task_id": task_id,
                        "task_name": name,
                        "skill": skill,
                        "split": split,
                        "id": entry["row"]["id"],
                        "lineage_group_id": meta(entry).get("lineage_group_id"),
                        "source_index": entry["source_index"],
                        "selection_primary": primary,
                        "selection_secondary": secondary,
                        "quality_tier": (
                            "manual_qc_not_flagged" if task_id == 3 else "parent_accepted"
                        ),
                        "deterministic_key": stable_hash(
                            args.seed, task_id, split, entry["row"]["id"]
                        ),
                        "record_sha256": entry["record_sha256"],
                    }
                )
            split_stats[split] = {
                "source_count": len(entries),
                "eligible_count": sum(eligible(entry, task_id) for entry in entries),
                "selected": summarize(chosen, task_id),
            }
            split_manifest[split] = {
                "records": len(chosen),
                "file": str(destination.relative_to(staging)),
                "sha256": hash_file(destination),
                "selected_ids_file": str(ids_path.relative_to(staging)),
                "selected_ids_sha256": hash_file(ids_path),
                "source_file": str(source),
                "source_sha256": hash_file(source),
            }
        write_json(
            staging / name / "statistics.json",
            {
                "dataset_version": VERSION,
                "parent_version": PARENT_VERSION,
                "task_id": task_id,
                "task_name": name,
                "skill": skill,
                "splits": split_stats,
            },
        )
        task_manifests.append(
            {"task_id": task_id, "task_name": name, "skill": skill, "splits": split_manifest}
        )
    write_jsonl(staging / "manifests/selected_ids.jsonl", selected_manifest)
    build_concept_metrics(staging, parent)
    (staging / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).resolve(), staging / "scripts/medicalskill_v1_2_lite_build.py")
    write_json(
        staging / "audits/reproducibility.json",
        {
            "status": "PASS" if all(
                row["same_ids_on_second_selection"] for row in reproducibility.values()
            ) else "FAIL",
            "seed": args.seed,
            "checks": reproducibility,
        },
    )
    validation = validate(
        staging, parent, args.max_train, args.max_test,
        args.image_workers, args.pil_sample,
    )
    write_json(staging / "audits/static_validation.json", validation)
    write_json(staging / "audits/lineage_hash_audit.json", validation["lineage"])
    write_json(staging / "audits/image_validation.json", validation["images"])
    after = parent_contract(parent)
    parent_unchanged = before == after
    write_json(
        staging / "audits/parent_immutability.json",
        {"status": "PASS" if parent_unchanged else "FAIL", "before": before, "after": after},
    )
    if not parent_unchanged:
        raise RuntimeError("parent dataset changed while deriving subset")
    selected_ids_path = staging / "manifests/selected_ids.jsonl"
    lock = {
        "dataset_version": VERSION,
        "parent_version": PARENT_VERSION,
        "parent_manifest_sha256": before["files"][
            "manifests/medicalskill_cl_v1_2_manifest.json"
        ]["sha256"],
        "seed": args.seed,
        "selected_ids_file": "manifests/selected_ids.jsonl",
        "selected_ids_sha256": hash_file(selected_ids_path),
        "selected_id_count": len(selected_manifest),
        "tasks": task_manifests,
        "immutable_contract": "Pin this root and verify every hash before each experiment.",
    }
    write_json(staging / "manifests/dataset_lock.json", lock)
    write_json(
        staging / "manifests/medicalskill_cl_v1_2_lite_manifest.json",
        {
            "status": "train_ready",
            "created_at": now(),
            "dataset_version": VERSION,
            "root": str(output),
            "parent": before,
            "dataset_lock_sha256": hash_file(staging / "manifests/dataset_lock.json"),
            "selected_ids_sha256": hash_file(selected_ids_path),
            "selected_id_count": len(selected_manifest),
            "tasks": task_manifests,
            "validation_split_used": False,
        },
    )
    write_json(
        staging / "audits/acceptance_matrix.json",
        {
            "status": "PASS",
            "gates": {
                "parent_v1_2_pass": True,
                "parent_unchanged": parent_unchanged,
                "exact_caps_or_complete_smaller_task": True,
                "explicit_ids_and_hashes": True,
                "deterministic_reselection": True,
                "schema_and_targets": validation["status"] == "PASS",
                "train_test_lineage_overlap_zero": not validation["lineage"][
                    "train_test_token_overlap"
                ],
                "train_test_image_hash_overlap_zero": not validation["lineage"][
                    "train_test_image_hash_overlap"
                ],
                "selected_images_exist": not validation["images"]["missing_or_empty"],
                "pil_sample_readable": not validation["images"]["pil_failures"],
                "task3_only_not_flagged": True,
                "no_validation_split": True,
            },
        },
    )
    readme = f"""# {VERSION}

Frozen deterministic subset of `{PARENT_VERSION}` for accelerated five-stage CL experiments.

- Seed: `{args.seed}`
- Train cap: `{args.max_train}` per task
- Test cap: `{args.max_test}` per task
- Validation split: not used
- Parent rows are copied byte-for-byte; images remain at their absolute shared paths.
- Task 5 is retained in full because it is smaller than the requested caps.

Pin this absolute root and verify `manifests/dataset_lock.json` before every experiment.
Never resample it for another method or random training seed.
"""
    atomic_bytes(staging / "README.md", readme.encode())
    inventory = {}
    for path in sorted(staging.rglob("*")):
        relative = path.relative_to(staging).as_posix()
        if path.is_file() and relative != "manifests/file_hashes.json":
            inventory[relative] = {"bytes": path.stat().st_size, "sha256": hash_file(path)}
    write_json(
        staging / "manifests/file_hashes.json",
        {"dataset_version": VERSION, "files": inventory},
    )
    os.replace(staging, output)
    print(json.dumps({"status": "PASS", "output": str(output), "counts": validation["counts"]}))


def validate_existing(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    result = validate(
        root, args.parent.resolve(), args.max_train, args.max_test,
        args.image_workers, args.pil_sample,
    )
    lock = json.loads((root / "manifests/dataset_lock.json").read_text())
    if lock["selected_ids_sha256"] != hash_file(root / lock["selected_ids_file"]):
        raise RuntimeError("selected IDs differ from the dataset lock")
    expected_files = json.loads((root / "manifests/file_hashes.json").read_text())["files"]
    mismatches = [
        relative for relative, expected in expected_files.items()
        if not (root / relative).is_file()
        or hash_file(root / relative) != expected["sha256"]
    ]
    if mismatches:
        raise RuntimeError(f"frozen file hash mismatch: {mismatches[:20]}")
    print(json.dumps({"status": "PASS", "root": str(root), "validation": result}))


def self_test() -> None:
    quota = apportion({"a": 2, "b": 10, "c": 10}, 12)
    assert quota == {"a": 2, "b": 5, "c": 5}
    entries = []
    for index in range(24):
        record_id = f"sample_{index:04d}"
        row = {
            "id": record_id,
            "task_id": 2,
            "skill": "diagnosis_classification",
            "messages": [
                {"role": "user", "content": "<image>\nQuestion"},
                {"role": "assistant", "content": "label"},
            ],
            "images": [f"/tmp/{record_id}.png"],
            "metadata": {
                "dataset_id": "a" if index < 12 else "b",
                "canonical_label": "label",
                "canonical_lineage_tokens": [f"image:test:{record_id}"],
            },
        }
        entries.append({"row": row, "raw": b"{}\n", "source_index": index, "record_sha256": "0" * 64})
    first = select(entries, 2, "train", 10, 42)
    second = select(entries, 2, "train", 10, 42)
    assert [x["row"]["id"] for x in first] == [x["row"]["id"] for x in second]
    sources = [meta(x)["dataset_id"] for x in first]
    assert sources.count("a") == sources.count("b") == 5
    assert [x["row"]["id"] for x in first] != [
        x["row"]["id"] for x in select(entries, 2, "train", 10, 43)
    ]
    print(json.dumps({"status": "PASS", "tests": 4}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=10_000)
    parser.add_argument("--max-test", type=int, default=1_000)
    parser.add_argument("--image-workers", type=int, default=8)
    parser.add_argument("--pil-sample", type=int, default=2_048)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    elif args.validate_only:
        validate_existing(args)
    else:
        if args.max_train <= 0 or args.max_test <= 0:
            raise RuntimeError("caps must be positive")
        build(args)


if __name__ == "__main__":
    main()
