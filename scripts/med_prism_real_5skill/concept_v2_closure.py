#!/usr/bin/env python3
"""Deterministic MedicalSkill-CL-v1 concept-v2 repair and closure reports."""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import importlib.util
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/root/MedAgentCL_v4")
DATA = Path("/remote-home/wangbomin/MedicalSkill-CL-v1")
ARTIFACT = ROOT / "artifacts/med_prism_real_5skill_smoke"
PARSER_PATH = ROOT / "scripts/skill_incremental_v1/extract_medtrinity_concepts_v1.py"
RAW_PARTS = (
    DATA / "_concept/full_output_part_00.jsonl",
    DATA / "_concept/full_output_part_01.jsonl",
)
TASK_DIR = DATA / "task_03_concept_recognition"
PARSER_VERSION = "medicalskill_concept_parser_v2_3_artifact_uncertainty"

UNCERTAINTY = re.compile(
    r"\b(?:possible|possibly|potential|potentially|may|might|could|likely|"
    r"suggest|suggests|suggested|suggesting|suggestive|indicative\s+of)\b",
    re.I,
)
AFFIRMATIVE = re.compile(
    r"\b(?:confirmed|definite|definitively|demonstrates?|shows?|reveals?|"
    r"identified|there\s+is|evidence\s+of|diagnostic\s+of)\b",
    re.I,
)
EXACT_ARTIFACTS = {
    "central position": "drop_position_template",
    "cross sectional view": "drop_nonclinical_image_concept",
    "anatomical association": "drop_nonclinical_image_concept",
    "region of interest": "drop_image_annotation_concept",
    "roi": "drop_image_annotation_concept",
    "bounding box": "drop_image_annotation_concept",
    "area ratio": "drop_image_annotation_concept",
    "highlighted region": "drop_image_annotation_concept",
    "disease process": "drop_generic_pathology",
    "pathological process": "drop_generic_pathology",
    "pathology": "drop_generic_pathology",
    "generic pathology": "drop_generic_pathology",
    "abnormality": "drop_generic_pathology",
    "generic abnormality": "drop_generic_pathology",
    "proximity": "drop_nonclinical_image_concept",
    "spatial relationship": "drop_nonclinical_image_concept",
    "relative position": "drop_position_template",
    "position": "drop_position_template",
    "location": "drop_position_template",
    "center": "drop_position_template",
    "central": "drop_position_template",
    "upper": "drop_position_template",
    "lower": "drop_position_template",
    "left": "drop_position_template",
    "right": "drop_position_template",
}
ANNOTATION_PATTERN = re.compile(
    r"\b(?:bounding\s+box|highlighted\s+(?:area|region)|area\s+ratio|"
    r"region\s+of\s+interest|\broi\b|colored\s+overlay|image\s+annotation)\b",
    re.I,
)
POSITION_WORDS = {
    "upper", "lower", "left", "right", "center", "central", "middle",
    "position", "location", "region", "area", "side", "portion",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Non-object JSON at {path}:{line_number}")
            yield value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def load_base_parser():
    spec = importlib.util.spec_from_file_location("concept_parser_v1", PARSER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def artifact_reason(concept: dict[str, Any]) -> str | None:
    canonical = normalized(concept.get("canonical"))
    mention = normalized(concept.get("mention"))
    if canonical in EXACT_ARTIFACTS:
        return EXACT_ARTIFACTS[canonical]
    if ANNOTATION_PATTERN.search(canonical):
        return "drop_image_annotation_concept"
    tokens = set(canonical.split())
    if tokens and tokens <= POSITION_WORDS:
        return "drop_position_template"
    if canonical in {"disease", "process", "finding", "generic finding"}:
        return "drop_generic_pathology"
    if canonical in {"cross sectional", "cross section", "image view", "imaging view"}:
        return "drop_nonclinical_image_concept"
    return None


@functools.lru_cache(maxsize=100_000)
def caption_sentence_data(caption: str):
    result = []
    cue_tokens = {
        "possible", "possibly", "potential", "potentially", "may", "might",
        "could", "likely", "suggest", "suggests", "suggested", "suggesting",
        "suggestive",
    }
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", caption):
        normal = normalized(sentence)
        tokens = tuple(normal.split())
        cues = [(index, index + 1) for index, token in enumerate(tokens) if token in cue_tokens]
        cues.extend(
            (index, index + 2)
            for index in range(len(tokens) - 1)
            if tokens[index:index + 2] == ("indicative", "of")
        )
        result.append((sentence, normal, tokens, tuple(cues)))
    return tuple(result)


def should_be_uncertain(caption: str, concept: dict[str, Any]) -> bool:
    if concept.get("type") not in {"finding", "pathology"}:
        return False
    if concept.get("status") == "negated":
        return False
    mention = str(concept.get("mention") or concept.get("canonical") or "")
    needle = normalized(mention)
    mention_tokens = tuple(needle.split())
    matching = [item for item in caption_sentence_data(caption) if needle and needle in item[1]]
    if not matching:
        return False

    uncertain_sentences = []
    width = len(mention_tokens)
    for sentence, _, tokens, cues in matching:
        mentions = [
            (index, index + width)
            for index in range(len(tokens) - width + 1)
            if tokens[index:index + width] == mention_tokens
        ]
        if any(
            (cue_end <= mention_start and mention_start - cue_end <= 6)
            or (mention_end <= cue_start and cue_start - mention_end <= 5)
            for mention_start, mention_end in mentions
            for cue_start, cue_end in cues
        ):
            uncertain_sentences.append(sentence)
    if not uncertain_sentences:
        return False
    affirmative = any(
        AFFIRMATIVE.search(sentence) and not UNCERTAINTY.search(sentence)
        for sentence, *_ in matching
    )
    return not affirmative


def repair_row(row: dict[str, Any], parser) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before = [dict(item) for item in row.get("concepts") or []]
    parsed = parser.parse(row["raw_response"], row["source_caption"])
    repairs = list(parser.LAST_REPAIRS)
    after = []
    dropped = []
    for concept in parsed:
        reason = artifact_reason(concept)
        if reason:
            repairs.append(reason)
            dropped.append({"reason": reason, "concept": dict(concept)})
            continue
        concept = dict(concept)
        if should_be_uncertain(row["source_caption"], concept) and concept["status"] == "present":
            concept["status"] = "uncertain"
            repairs.append("repair_uncertainty_v2")
        after.append(concept)
        if len(after) == 8:
            break
    row = dict(row)
    row["concepts_before_v2"] = before
    row["concepts"] = after
    row["target"] = parser.target(after)
    row["repair_reasons_v2"] = dict(collections.Counter(repairs))
    row["parser_version"] = PARSER_VERSION
    row["error"] = "" if after else "empty_after_concept_v2_repair"
    return row, dropped


def audit_failures(row: dict[str, Any], concepts: list[dict[str, Any]]) -> collections.Counter:
    failures: collections.Counter[str] = collections.Counter()
    caption = str(row.get("source_caption") or "")
    seen = set()
    if not concepts:
        failures["empty_target"] += 1
    for concept in concepts:
        if artifact_reason(concept):
            failures["nonclinical_or_artifact_concept"] += 1
        mention = str(concept.get("mention") or "")
        if normalized(mention) not in normalized(caption):
            failures["unsupported_mention"] += 1
        key = (
            normalized(concept.get("canonical")),
            concept.get("type"),
            concept.get("status"),
            concept.get("laterality"),
        )
        if key in seen:
            failures["duplicate"] += 1
        seen.add(key)
        if should_be_uncertain(caption, concept) and concept.get("status") == "present":
            failures["uncertainty_status_error"] += 1
    return failures


def select_stratified(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    selected_ids = set()
    repair_buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    strata: dict[tuple[str, str, str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        row_id = str(row["id"])
        for repair in (row.get("repair_reasons_v2") or {}):
            if repair.startswith("drop_") or repair == "repair_uncertainty_v2":
                repair_buckets[repair].append(row)
        strata[(
            str(row.get("split") or "unknown"),
            str(row.get("source_kind") or "unknown"),
            str(row.get("modality") or "unknown"),
            len(row.get("concepts") or []),
        )].append(row)
    for key in sorted(repair_buckets):
        candidates = repair_buckets[key]
        rng.shuffle(candidates)
        for row in candidates[: min(10, len(candidates))]:
            if row["id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["id"])
    buckets = [values for _, values in sorted(strata.items())]
    for values in buckets:
        rng.shuffle(values)
    while len(selected) < count and any(buckets):
        next_buckets = []
        for values in buckets:
            while values and values[-1]["id"] in selected_ids:
                values.pop()
            if values and len(selected) < count:
                row = values.pop()
                selected.append(row)
                selected_ids.add(row["id"])
            if values:
                next_buckets.append(values)
        buckets = next_buckets
    return selected[:count]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_task_rows(path: Path, repaired: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in read_jsonl(path):
        source = repaired.get(str(row["id"]))
        if source is None or not source.get("target"):
            raise RuntimeError(f"Missing repaired concept target for {row['id']}")
        row["messages"][-1]["content"] = source["target"]
        metadata = row.setdefault("metadata", {})
        metadata["concepts"] = source["concepts"]
        metadata["all_target_concepts"] = [item["canonical"] for item in source["concepts"]]
        metadata["core_target_concepts"] = [normalized(item["canonical"]) for item in source["concepts"]]
        metadata["concept_parser_version"] = PARSER_VERSION
        metadata["concept_repair_reasons_v2"] = [
            {"reason": key, "count": value}
            for key, value in sorted(source["repair_reasons_v2"].items())
        ]
        output.append(row)
    return output


def build_reports(
    repaired_rows: list[dict[str, Any]],
    dropped_by_row: dict[str, list[dict[str, Any]]],
    task_rows: dict[str, list[dict[str, Any]]],
    artifact: Path,
) -> None:
    histogram = collections.Counter(len(row["concepts"]) for row in repaired_rows)
    total = len(repaired_rows)
    histogram_report = {
        "status": "PASS" if total and not histogram[0] else "FAIL",
        "records": total,
        "counts": {str(value): histogram[value] for value in range(1, 9)},
        "percentages": {str(value): 100.0 * histogram[value] / total for value in range(1, 9)},
        "empty": histogram[0],
    }
    saturation = {
        "status": "PASS" if total and not histogram[0] else "FAIL",
        "records": total,
        "exactly_eight": histogram[8],
        "exactly_eight_ratio": histogram[8] / total,
        "exactly_eight_percentage": 100.0 * histogram[8] / total,
        "cap": 8,
    }
    repair_rows: dict[str, set[str]] = collections.defaultdict(set)
    repair_events = collections.Counter()
    for row in repaired_rows:
        for repair, count in (row.get("repair_reasons_v2") or {}).items():
            repair_rows[repair].add(str(row["id"]))
            repair_events[repair] += int(count)
    repair_report = {
        "status": "PASS",
        "records": total,
        "unique_row_counts": {key: len(value) for key, value in sorted(repair_rows.items())},
        "event_counts": dict(sorted(repair_events.items())),
    }
    pre_vocab = collections.Counter()
    post_vocab = collections.Counter()
    examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in repaired_rows:
        for item in row.get("concepts_before_v2") or []:
            reason = artifact_reason(item)
            if reason:
                pre_vocab[normalized(item.get("canonical"))] += 1
                if len(examples[reason]) < 20:
                    examples[reason].append({"id": row["id"], "concept": item})
        for item in row["concepts"]:
            reason = artifact_reason(item)
            if reason:
                post_vocab[normalized(item.get("canonical"))] += 1
    artifact_report = {
        "status": "PASS" if pre_vocab and not post_vocab else "FAIL",
        "audit_rule": "central position, pure position templates, cross-sectional view, anatomical association, ROI/box/highlight concepts, generic disease/pathology and other nonclinical image concepts are failures",
        "pre_repair_occurrences": dict(pre_vocab.most_common()),
        "pre_repair_unique_rows": len(dropped_by_row),
        "post_repair_occurrences": dict(post_vocab.most_common()),
        "post_repair_failures": sum(post_vocab.values()),
        "examples_by_repair": examples,
    }
    modality_source: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str, str], list[int]] = collections.defaultdict(list)
    for row in repaired_rows:
        grouped[(
            str(row.get("split") or "unknown"),
            str(row.get("source_kind") or "unknown"),
            str(row.get("modality") or "unknown"),
        )].append(len(row["concepts"]))
    for key, values in sorted(grouped.items()):
        modality_source["|".join(key)] = {
            "records": len(values),
            "mean_concepts": sum(values) / len(values),
            "min_concepts": min(values),
            "max_concepts": max(values),
        }
    modality_report = {"status": "PASS", "strata": modality_source}
    sample = select_stratified(repaired_rows, 500, 42)
    pre_failures: collections.Counter[str] = collections.Counter()
    post_failures: collections.Counter[str] = collections.Counter()
    sample_rows = []
    for row in sample:
        before_failure = audit_failures(row, row.get("concepts_before_v2") or [])
        after_failure = audit_failures(row, row["concepts"])
        pre_failures.update(before_failure)
        post_failures.update(after_failure)
        sample_rows.append({
            "id": row["id"],
            "split": row.get("split"),
            "source_kind": row.get("source_kind"),
            "modality": row.get("modality"),
            "before": [item.get("canonical") for item in row.get("concepts_before_v2") or []],
            "after": [item.get("canonical") for item in row["concepts"]],
            "repairs": row.get("repair_reasons_v2") or {},
            "pre_failures": dict(before_failure),
            "post_failures": dict(after_failure),
        })
    semantic = {
        "status": "PASS" if len(sample) == 500 and sum(post_failures.values()) == 0 and pre_failures["nonclinical_or_artifact_concept"] > 0 else "FAIL",
        "seed": 42,
        "reviewed_sample_count": len(sample),
        "stratification": ["split", "source_kind", "modality", "concept_count", "repair_type"],
        "failure_standard": artifact_report["audit_rule"],
        "pre_repair_failure_counts": dict(pre_failures),
        "post_repair_failure_counts": dict(post_failures),
        "audit_rows": sample_rows,
    }
    for name, report in (
        ("concept_count_histogram.json", histogram_report),
        ("concept_cap8_saturation.json", saturation),
        ("concept_artifact_vocabulary_audit.json", artifact_report),
        ("concept_repair_unique_row_counts.json", repair_report),
        ("concept_modality_source_audit.json", modality_report),
        ("concept_semantic_audit_500_v2.json", semantic),
    ):
        atomic_json(artifact / name, report)
    report_statuses = [
        histogram_report["status"], saturation["status"], artifact_report["status"],
        semantic["status"],
    ]
    lines = [
        "# Concept v2 closure",
        "",
        f"Status: **{'PASS' if set(report_statuses) == {'PASS'} else 'FAIL'}**",
        "",
        f"Raw responses reparsed: {total}",
        f"Task-03 train/test retained: {len(task_rows['train'])}/{len(task_rows['test'])}",
        f"Exactly 8 concepts: {histogram[8]} ({100.0 * histogram[8] / total:.4f}%)",
        f"Pre-repair artifact failures: {sum(pre_vocab.values())}",
        f"Post-repair artifact failures: {sum(post_vocab.values())}",
        f"Semantic audit rows: {len(sample)}; post-repair failures: {sum(post_failures.values())}",
        "",
        "The audit treats central position, pure directional templates, cross-sectional view, anatomical association, annotation geometry, and unsupported generic pathology as failures.",
    ]
    (artifact / "concept_closure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if set(report_statuses) != {"PASS"}:
        raise RuntimeError(f"Concept v2 closure failed: {report_statuses}")


def main() -> int:
    global DATA, ARTIFACT, RAW_PARTS, TASK_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    DATA = args.data_root
    ARTIFACT = args.artifact_root
    RAW_PARTS = (
        DATA / "_concept/full_output_part_00.jsonl",
        DATA / "_concept/full_output_part_01.jsonl",
    )
    TASK_DIR = DATA / "task_03_concept_recognition"
    base_parser = load_base_parser()
    repaired_rows = []
    repaired_by_part: dict[Path, list[dict[str, Any]]] = {}
    dropped_by_row: dict[str, list[dict[str, Any]]] = {}
    for part in RAW_PARTS:
        part_rows = []
        for row in read_jsonl(part):
            repaired, dropped = repair_row(row, base_parser)
            part_rows.append(repaired)
            repaired_rows.append(repaired)
            if dropped:
                dropped_by_row[str(row["id"])] = dropped
        repaired_by_part[part] = part_rows
    empty = [row["id"] for row in repaired_rows if not row.get("target")]
    if empty:
        atomic_json(ARTIFACT / "concept_v2_empty_rows.json", {"count": len(empty), "ids": empty[:1000]})
        raise RuntimeError(f"Concept v2 would create {len(empty)} empty targets")
    repaired = {str(row["id"]): row for row in repaired_rows}
    task_rows = {
        split: update_task_rows(TASK_DIR / f"{split}.jsonl", repaired)
        for split in ("train", "test")
    }
    build_reports(repaired_rows, dropped_by_row, task_rows, ARTIFACT)
    dry_report = {
        "status": "PASS",
        "dry_run": args.dry_run,
        "raw_records": len(repaired_rows),
        "task_03_train": len(task_rows["train"]),
        "task_03_test": len(task_rows["test"]),
        "empty_targets": 0,
    }
    atomic_json(ARTIFACT / "concept_v2_execution.json", dry_report)
    if args.dry_run:
        print(json.dumps(dry_report, indent=2))
        return 0
    for part, values in repaired_by_part.items():
        atomic_jsonl(part, values)
    for split, values in task_rows.items():
        atomic_jsonl(TASK_DIR / f"{split}.jsonl", values)
    frequency = collections.Counter(
        normalized(concept["canonical"])
        for row in task_rows["train"]
        for concept in row["metadata"]["concepts"]
    )
    atomic_json(
        TASK_DIR / "concept_vocabulary.json",
        {
            "train_only": True,
            "parser_version": PARSER_VERSION,
            "frequency": dict(frequency.most_common()),
            "core_frequency_ge_10": sorted(key for key, value in frequency.items() if value >= 10),
        },
    )
    hashes = {
        str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in (*RAW_PARTS, TASK_DIR / "train.jsonl", TASK_DIR / "test.jsonl", TASK_DIR / "concept_vocabulary.json")
    }
    atomic_json(ARTIFACT / "concept_v2_file_hashes.json", {"status": "PASS", "files": hashes})
    print(json.dumps(dry_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
