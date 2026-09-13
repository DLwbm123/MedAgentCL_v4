#!/usr/bin/env python3
"""Build MedicalSkill-CL-v1.1 Concept v3 from preserved v2/raw responses."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/root/MedAgentCL_v4")
DATA = Path("/remote-home/wangbomin/MedicalSkill-CL-v1.1")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_v1_1_and_medprism_smoke"
AUDITOR_PATH = ROOT / "scripts/med_prism_real_5skill/concept_v3_auditor.py"
RAW_PARTS = (
    DATA / "_concept/full_output_part_00.jsonl",
    DATA / "_concept/full_output_part_01.jsonl",
)
TASK_DIR = DATA / "task_03_concept_recognition"
PARSER_VERSION = "medicalskill_concept_parser_v3_0_dedup_position_generic"

UNCERTAINTY = re.compile(
    r"\b(?:possible|possibly|potential|potentially|may|might|could|likely|"
    r"suggest(?:s|ed|ing|ive)?|indicative\s+of)\b", re.I,
)
NEGATION = re.compile(
    r"\b(?:no|not|without|absence\s+of|absent|negative\s+for|free\s+of|"
    r"does\s+not\s+show|did\s+not\s+show)\b", re.I,
)
GENERIC_EXACT = {
    "diagnosis", "disease", "disease process", "pathological process",
    "pathological condition", "pathological change", "generic pathology",
    "generic disease", "generic structure", "generic tissue", "generic process",
    "pathology", "abnormality", "generic abnormality", "finding",
    "generic finding", "process", "structure", "tissue",
}
NONCLINICAL_EXACT = {
    "cross sectional view", "cross section", "image view", "imaging view",
    "anatomical association", "spatial relationship", "relative position",
    "proximity", "central position", "peripheral position",
}
POSITION_ONLY = re.compile(
    r"^(?:(?:upper|lower|left|right|central|center|middle|peripheral|anterior|"
    r"posterior|medial|lateral)\s+)*(?:quadrant|portion|region|area|position|"
    r"location|side|view)$|^(?:upper|lower|left|right|central|center|middle|"
    r"peripheral|quadrant|position|location)$"
)
ROI = re.compile(
    r"\b(?:region\s+of\s+interest|roi|bounding\s+box|highlighted\s+(?:region|area)|"
    r"area\s+ratio|image\s+annotation|colored\s+overlay)\b", re.I,
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"Non-object JSON at {path}:{number}")
                yield value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def sentence_for(caption: str, mention: str) -> str:
    needle = norm(mention)
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", caption):
        if needle and needle in norm(sentence):
            return sentence
    return ""


def cue_near_mention(sentence: str, mention: str, cue: re.Pattern[str], before: int, after: int) -> bool:
    sentence_norm = norm(sentence)
    mention_tokens = norm(mention).split()
    tokens = sentence_norm.split()
    if not mention_tokens:
        return False
    spans = [(index, index + len(mention_tokens)) for index in range(len(tokens) - len(mention_tokens) + 1) if tokens[index:index + len(mention_tokens)] == mention_tokens]
    for match in cue.finditer(sentence_norm):
        cue_start = len(sentence_norm[:match.start()].split())
        cue_end = cue_start + len(match.group(0).split())
        for mention_start, mention_end in spans:
            if cue_end <= mention_start and mention_start - cue_end <= before:
                return True
            if mention_end <= cue_start and cue_start - mention_end <= after:
                return True
    return False

def drop_reason(concept: dict[str, Any]) -> str | None:
    canonical = norm(concept.get("canonical"))
    mention = norm(concept.get("mention"))
    if canonical in GENERIC_EXACT:
        return "drop_generic_exact_v3"
    if POSITION_ONLY.fullmatch(canonical):
        return "drop_position_template_v3"
    if canonical in NONCLINICAL_EXACT:
        return "drop_nonclinical_exact_v3"
    if ROI.search(canonical) or ROI.search(mention):
        return "drop_roi_annotation_v3"
    return None


def normalize_semantics(concept: dict[str, Any], caption: str, repairs: list[str]) -> dict[str, Any]:
    item = dict(concept)
    item["canonical"] = re.sub(r"\s+", " ", str(item.get("canonical") or "")).strip()
    item["mention"] = re.sub(r"\s+", " ", str(item.get("mention") or item["canonical"])).strip()
    laterality = str(item.get("laterality") or "unspecified").casefold()
    mention_norm = norm(item["mention"])
    if laterality not in {"left", "right", "bilateral", "midline", "unspecified"}:
        laterality = "unspecified"
        repairs.append("normalize_laterality_v3")
    if laterality == "unspecified":
        if "bilateral" in mention_norm or ("left" in mention_norm and "right" in mention_norm):
            laterality = "bilateral"
            repairs.append("infer_laterality_v3")
        elif re.search(r"\bleft\b", mention_norm):
            laterality = "left"
            repairs.append("infer_laterality_v3")
        elif re.search(r"\bright\b", mention_norm):
            laterality = "right"
            repairs.append("infer_laterality_v3")
        else:
            sentence_norm = norm(sentence_for(caption, item["mention"]))
            needle = norm(item["mention"])
            before = sentence_norm.split(needle, 1)[0].split()[-4:] if needle in sentence_norm else []
            if "bilateral" in before or ("left" in before and "right" in before):
                laterality = "bilateral"
                repairs.append("infer_laterality_v3")
            elif "left" in before:
                laterality = "left"
                repairs.append("infer_laterality_v3")
            elif "right" in before:
                laterality = "right"
                repairs.append("infer_laterality_v3")
    item["laterality"] = laterality

    sentence = sentence_for(caption, item["mention"])
    old_status = str(item.get("status") or "present").casefold()
    status = old_status if old_status in {"present", "uncertain", "negated"} else "present"
    if sentence and cue_near_mention(sentence, item["mention"], NEGATION, 4, 2):
        status = "negated"
    elif sentence and cue_near_mention(sentence, item["mention"], UNCERTAINTY, 6, 5):
        status = "uncertain"
    if status != old_status:
        repairs.append(f"normalize_{status}_v3")
    item["status"] = status
    return item


def target(concepts: list[dict[str, Any]]) -> str:
    values = []
    for item in concepts:
        text = str(item["canonical"])
        if item["laterality"] != "unspecified" and not re.search(rf"\b{re.escape(item['laterality'])}\b", text, re.I):
            text = f"{item['laterality']} {text}"
        if item["status"] != "present":
            text = f"{item['status']}: {text}"
        values.append(text)
    return "; ".join(values)


def repair_row(row: dict[str, Any]) -> dict[str, Any]:
    before = [dict(item) for item in row.get("concepts") or []]
    repairs: list[str] = []
    normalized = [normalize_semantics(item, str(row.get("source_caption") or ""), repairs) for item in before]
    filtered = []
    for item in normalized:
        reason = drop_reason(item)
        if reason:
            repairs.append(reason)
        else:
            filtered.append(item)
    deduplicated = []
    seen = set()
    for item in filtered:
        key = norm(item["canonical"])
        if not key or key in seen:
            repairs.append("exact_normalized_dedup_v3")
            continue
        seen.add(key)
        deduplicated.append(item)
        if len(deduplicated) == 8:
            break
    result = dict(row)
    result["concepts_before_v3"] = before
    result["concepts"] = deduplicated
    result["target"] = target(deduplicated)
    result["repair_reasons_v3"] = dict(collections.Counter(repairs))
    result["parser_version"] = PARSER_VERSION
    result["error"] = "" if deduplicated else "empty_after_concept_v3_repair"
    return result


def load_auditor():
    spec = importlib.util.spec_from_file_location("concept_v3_independent_auditor", AUDITOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def stratified_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        source = str((row.get("source_metadata") or {}).get("source_dataset") or "unknown")
        strata[(str(row.get("split")), str(row.get("modality")), str(row.get("source_kind")), source)].append(row)
    buckets = []
    for key in sorted(strata):
        values = strata[key]
        rng.shuffle(values)
        buckets.append(values)
    selected = []
    while len(selected) < count and any(buckets):
        next_buckets = []
        for values in buckets:
            if values and len(selected) < count:
                selected.append(values.pop())
            if values:
                next_buckets.append(values)
        buckets = next_buckets
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_task(path: Path, by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in read_jsonl(path):
        source = by_id.get(str(row["id"]))
        if source is None or not source["concepts"]:
            raise RuntimeError(f"Missing Concept v3 result for {row['id']}")
        row["messages"][-1]["content"] = source["target"]
        metadata = row.setdefault("metadata", {})
        metadata["concepts"] = source["concepts"]
        metadata["all_target_concepts"] = [item["canonical"] for item in source["concepts"]]
        metadata["core_target_concepts"] = [norm(item["canonical"]) for item in source["concepts"]]
        metadata["concept_parser_version"] = PARSER_VERSION
        metadata["concept_repair_reasons_v3"] = [
            {"reason": key, "count": value}
            for key, value in sorted(source["repair_reasons_v3"].items())
        ]
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data = args.data_root
    artifact = args.artifact_root
    raw_parts = (data / "_concept/full_output_part_00.jsonl", data / "_concept/full_output_part_01.jsonl")
    task_dir = data / "task_03_concept_recognition"
    repaired_by_part = {}
    rows = []
    for part in raw_parts:
        values = [repair_row(row) for row in read_jsonl(part)]
        repaired_by_part[part] = values
        rows.extend(values)
    empty = [row["id"] for row in rows if not row["concepts"]]
    if empty:
        atomic_json(artifact / "concept_v3_empty_rows.json", {"status": "FAIL", "count": len(empty), "ids": empty[:1000]})
        raise RuntimeError(f"Concept v3 creates {len(empty)} empty rows")
    by_id = {str(row["id"]): row for row in rows}
    task_rows = {split: update_task(task_dir / f"{split}.jsonl", by_id) for split in ("train", "test")}

    auditor = load_auditor()
    sample = stratified_sample(rows, 500, 42)
    audit_rows = []
    failures = collections.Counter()
    for row in sample:
        audit = auditor.audit_row(row, row["concepts"])
        failures.update(audit["reason_counts"])
        audit_rows.append({
            "id": row["id"], "split": row.get("split"), "modality": row.get("modality"),
            "source_kind": row.get("source_kind"), "source_caption": row.get("source_caption"),
            "raw_response": row.get("raw_response"),
            "structured_concepts_before": row.get("concepts_before_v3"),
            "structured_concepts_after": row.get("concepts"),
            "canonical": [item.get("canonical") for item in row["concepts"]],
            "type": [item.get("type") for item in row["concepts"]],
            "status_values": [item.get("status") for item in row["concepts"]],
            "laterality": [item.get("laterality") for item in row["concepts"]],
            "mention": [item.get("mention") for item in row["concepts"]],
            "repair_reasons": row.get("repair_reasons_v3"),
            "independent_audit_reasons": audit["reason_counts"],
            "audit_status": audit["status"],
        })
    audit_report = {
        "status": "PASS" if len(sample) == 500 and not failures else "FAIL",
        "seed": 42, "sample_count": len(sample),
        "stratification": ["split", "modality", "source_kind", "source_dataset"],
        "independent_module": str(AUDITOR_PATH),
        "checks": ["exact duplicate", "synonym duplicate", "pure positional concept", "annotation/ROI concept", "generic nonclinical concept", "caption support", "uncertainty cue consistency", "negation consistency", "modality/anatomy contradiction"],
        "failure_counts": dict(sorted(failures.items())), "rows": audit_rows,
    }
    atomic_json(artifact / "concept_semantic_audit_500_v3.json", audit_report)

    histogram = collections.Counter(len(row["concepts"]) for row in rows)
    repairs = collections.Counter()
    unique_repairs: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        for reason, count in row["repair_reasons_v3"].items():
            repairs[reason] += count
            unique_repairs[reason].add(str(row["id"]))
    exact_duplicates = sum(
        len(values) - len({norm(item["canonical"]) for item in values})
        for values in (row["concepts"] for row in rows)
    )
    summary = {
        "status": "PASS" if not exact_duplicates and audit_report["status"] == "PASS" else "FAIL",
        "raw_records": len(rows), "task_train": len(task_rows["train"]), "task_test": len(task_rows["test"]),
        "histogram": {str(i): histogram[i] for i in range(1, 9)},
        "exactly_8": histogram[8], "exactly_8_ratio": histogram[8] / len(rows),
        "final_exact_normalized_duplicates": exact_duplicates,
        "repair_event_counts": dict(sorted(repairs.items())),
        "repair_unique_row_counts": {key: len(value) for key, value in sorted(unique_repairs.items())},
        "independent_audit": audit_report["status"], "dry_run": args.dry_run,
    }
    atomic_json(artifact / "concept_v3_summary.json", summary)
    (artifact / "concept_v3_closure_report.md").write_text(
        "# Concept v3 closure\n\n"
        f"Status: **{summary['status']}**\n\n"
        f"Records: {len(rows)}; task train/test: {len(task_rows['train'])}/{len(task_rows['test'])}.\n\n"
        f"Final exact normalized duplicates: {exact_duplicates}.\n\n"
        f"Exactly 8 concepts: {histogram[8]} ({100 * histogram[8] / len(rows):.4f}%).\n\n"
        f"Independent 500-row audit: {audit_report['status']}; failures: {dict(failures)}.\n",
        encoding="utf-8",
    )
    if summary["status"] != "PASS":
        raise RuntimeError(f"Concept v3 closure failed: {summary}")
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
    for path, values in repaired_by_part.items():
        atomic_jsonl(path, values)
    for split, values in task_rows.items():
        atomic_jsonl(task_dir / f"{split}.jsonl", values)
    frequency = collections.Counter(norm(item["canonical"]) for row in task_rows["train"] for item in row["metadata"]["concepts"])
    atomic_json(task_dir / "concept_vocabulary.json", {
        "train_only": True, "parser_version": PARSER_VERSION,
        "frequency": dict(frequency.most_common()),
        "core_frequency_ge_10": sorted(key for key, count in frequency.items() if count >= 10),
    })
    files = [*raw_parts, task_dir / "train.jsonl", task_dir / "test.jsonl", task_dir / "concept_vocabulary.json"]
    atomic_json(artifact / "concept_v3_file_hashes.json", {
        "status": "PASS", "files": {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in files},
    })
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
