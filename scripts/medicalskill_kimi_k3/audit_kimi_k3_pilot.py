#!/usr/bin/env python3
"""Create automatic and human-review artifacts for the fixed Kimi K3 pilot."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_kimi_k3_pilot"
PILOT_SIZE = 2_000
HUMAN_AUDIT_SIZE = 500
SEED = 42

ARTIFACT_PATTERN = re.compile(
    r"\b(?:roi|region of interest|bounding box|bbox|area ratio|highlighted region|colored box|central position|cross-sectional view)\b",
    re.I,
)
GENERIC_PATTERN = re.compile(r"^(?:disease|disease process|pathology|abnormality|finding|medical image|anatomical association)$", re.I)
UNCERTAINTY_PATTERN = re.compile(r"\b(?:possible|possibly|potential|potentially|may|might|could|suggest|suggesting|suggestive|likely|indicative of)\b", re.I)
NEGATION_PATTERN = re.compile(r"\b(?:no|not|without|absent|negative for|free of|unremarkable)\b", re.I)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_results(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        result[str(row["stable_record_id"])] = row
    return result


def normalized_set(values: list[str]) -> set[str]:
    return {" ".join(str(value).casefold().split()).strip(" ;,.") for value in values if str(value).strip()}


def qwen_values(row: dict[str, Any]) -> list[str]:
    return list(row.get("qwen_v3_concepts") or [])


def proportion(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def proportional_sample(rows: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["pilot_stratum"], row["modality"], row["source_dataset"])].append(row)
    available = len(rows)
    raw = {key: total * len(group) / available for key, group in groups.items()}
    quotas = {key: min(len(groups[key]), math.floor(raw[key])) for key in groups}
    remaining = total - sum(quotas.values())
    for key in sorted(groups, key=lambda value: (-(raw[value] - quotas[value]), value)):
        if not remaining:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1
    selected = []
    for key in sorted(groups):
        ranked = sorted(
            groups[key],
            key=lambda row: hashlib.sha256(f"{SEED}|human|{row['stable_record_id']}".encode()).hexdigest(),
        )
        selected.extend(ranked[: quotas[key]])
    if len(selected) != total:
        raise RuntimeError(f"Human audit sample {len(selected)} != {total}")
    return selected


def git_state() -> str:
    commands = [
        ["git", "branch", "--show-current"],
        ["git", "rev-parse", "HEAD"],
        ["git", "-c", "core.pager=cat", "status", "--short"],
    ]
    labels = ["branch", "head", "status_short"]
    parts = []
    for label, command in zip(labels, commands):
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        parts.append(f"[{label}]\n{result.stdout.rstrip()}\n")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default=str(ARTIFACT))
    args = parser.parse_args()
    artifact = Path(args.artifact_root)
    selection = json.loads((artifact / "pilot_selection_manifest.json").read_text(encoding="utf-8"))
    selected = {row["stable_record_id"]: row for row in selection["records"]}
    frozen = {row["stable_record_id"]: row for row in read_jsonl(artifact / "input_freeze_manifest.jsonl") if row["stable_record_id"] in selected}
    results = latest_results(artifact / "pilot_results.jsonl")
    if len(selected) != PILOT_SIZE or len(frozen) != PILOT_SIZE:
        raise RuntimeError("Pilot selection/frozen input count contract failed")
    missing = sorted(set(selected) - set(results))
    if missing:
        raise RuntimeError(f"Pilot incomplete: {len(missing)} selected IDs have no result")

    status_counts = collections.Counter(results[row_id].get("status") for row_id in selected)
    successes = [results[row_id] for row_id in selected if results[row_id].get("status") == "success"]
    parse_success = 0
    count_compliant = 0
    empty = 0
    saturation = 0
    artifact_hits = []
    generic_hits = []
    duplicate_hits = []
    uncertainty_rows = []
    negation_rows = []
    histogram = collections.Counter()
    modality_source: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    jaccards = []
    qwen_counts = collections.Counter()
    kimi_counts = collections.Counter()
    detail_rows = []

    for row_id in selection["stable_record_ids"]:
        result, source, selected_row = results[row_id], frozen[row_id], selected[row_id]
        concepts = list(result.get("parsed_concepts") or [])
        qwen = qwen_values(selected_row)
        qwen_counts[len(qwen)] += 1
        risk_flags = []
        if result.get("status") == "success":
            parse_success += 1
            if 1 <= len(concepts) <= 8:
                count_compliant += 1
            if not concepts:
                empty += 1
            histogram[len(concepts)] += 1
            kimi_counts[len(concepts)] += 1
            if len(concepts) == 8:
                saturation += 1
            if any(ARTIFACT_PATTERN.search(value) for value in concepts):
                artifact_hits.append(row_id); risk_flags.append("artifact_term")
            if any(GENERIC_PATTERN.fullmatch(value.strip()) for value in concepts):
                generic_hits.append(row_id); risk_flags.append("generic_term")
            normalized = [" ".join(value.casefold().split()).strip(" ;,.") for value in concepts]
            if len(normalized) != len(set(normalized)):
                duplicate_hits.append(row_id); risk_flags.append("normalized_duplicate")
            left, right = normalized_set(concepts), normalized_set(qwen)
            jaccards.append(len(left & right) / len(left | right) if left | right else 1.0)
        caption = source["source_caption"]
        if UNCERTAINTY_PATTERN.search(caption):
            uncertainty_rows.append(row_id); risk_flags.append("caption_uncertainty_requires_human_review")
        if NEGATION_PATTERN.search(caption):
            negation_rows.append(row_id); risk_flags.append("caption_negation_requires_human_review")
        key = f"{source['modality']}|{source['source_dataset']}"
        modality_source[key][result.get("status", "missing")] += 1
        detail_rows.append(
            {
                "stable_record_id": row_id,
                "pilot_stratum": selected_row["pilot_stratum"],
                "modality": source["modality"],
                "source_dataset": source["source_dataset"],
                "automatic_risk_flags": risk_flags,
            }
        )

    success_count = len(successes)
    retries = sum(max(0, int(row.get("attempt_count", 0)) - 1) for row in results.values())
    token_totals = collections.Counter()
    for row in results.values():
        token_totals.update({key: int(value or 0) for key, value in (row.get("token_usage") or {}).items()})
    latencies = sorted(float(row.get("latency_seconds", 0)) for row in results.values())
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0

    machine_gates = {
        "json_parse_rate_100pct": parse_success == PILOT_SIZE,
        "concept_count_1_to_8_rate_100pct": count_compliant == PILOT_SIZE,
        "api_success_rate_ge_99_5pct": proportion(success_count, PILOT_SIZE) >= 0.995,
        "empty_rate_lt_0_5pct": proportion(empty, PILOT_SIZE) < 0.005,
        "generic_or_artifact_rate_le_1pct": proportion(len(set(generic_hits + artifact_hits)), PILOT_SIZE) <= 0.01,
        "normalized_duplicate_rate_le_0_5pct": proportion(len(duplicate_hits), PILOT_SIZE) <= 0.005,
    }
    manual_gates = {
        "caption_supported_precision_ge_98pct": "PENDING_HUMAN_REVIEW",
        "negation_error_le_1pct": "PENDING_HUMAN_REVIEW",
        "uncertainty_error_le_1pct": "PENDING_HUMAN_REVIEW",
        "semantic_duplicate_le_0_5pct": "PENDING_HUMAN_REVIEW",
    }
    auto_audit = {
        "status": "PASS" if all(machine_gates.values()) else "FAIL",
        "program_verified": {
            "selected_total": PILOT_SIZE,
            "result_total": len(results),
            "status_counts": dict(status_counts),
            "api_success_rate": proportion(success_count, PILOT_SIZE),
            "json_parse_rate": proportion(parse_success, PILOT_SIZE),
            "concept_count_1_to_8_rate": proportion(count_compliant, PILOT_SIZE),
            "empty_rate": proportion(empty, PILOT_SIZE),
            "concept_count_histogram": {str(key): value for key, value in sorted(histogram.items())},
            "eight_concept_saturation": proportion(saturation, PILOT_SIZE),
            "artifact_hit_rate": proportion(len(artifact_hits), PILOT_SIZE),
            "generic_hit_rate": proportion(len(generic_hits), PILOT_SIZE),
            "normalized_duplicate_rate": proportion(len(duplicate_hits), PILOT_SIZE),
            "uncertainty_caption_risk_count": len(uncertainty_rows),
            "negation_caption_risk_count": len(negation_rows),
            "qwen_count_histogram": {str(key): value for key, value in sorted(qwen_counts.items())},
            "kimi_count_histogram": {str(key): value for key, value in sorted(kimi_counts.items())},
            "mean_normalized_set_jaccard_with_qwen_v3": sum(jaccards) / len(jaccards) if jaccards else 0.0,
            "modality_source_status": {key: dict(value) for key, value in sorted(modality_source.items())},
            "token_usage": dict(token_totals),
            "latency_seconds_p50": p50,
            "latency_seconds_p95": p95,
            "retry_count": retries,
            "cache_success_count": success_count,
        },
        "machine_gates": machine_gates,
        "requires_human_confirmation": manual_gates,
        "automatic_risk_examples": {
            "artifact": artifact_hits[:100],
            "generic": generic_hits[:100],
            "duplicate": duplicate_hits[:100],
            "uncertainty": uncertainty_rows[:100],
            "negation": negation_rows[:100],
        },
        "record_risk_flags": detail_rows,
    }
    write_json(artifact / "pilot_auto_audit.json", auto_audit)

    human_rows = proportional_sample(detail_rows, HUMAN_AUDIT_SIZE)
    csv_path = artifact / "pilot_human_audit_500.csv"
    fields = [
        "stable_record_id", "modality", "source_dataset", "source_caption", "qwen_v3_concepts",
        "kimi_k3_concepts", "pilot_stratum", "automatic_risk_flags", "caption_supported",
        "negation_correct", "uncertainty_correct", "generic_or_artifact", "semantic_duplicate",
        "preferred_output", "reviewer_notes",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for audit_row in human_rows:
            row_id = audit_row["stable_record_id"]
            writer.writerow(
                {
                    "stable_record_id": row_id,
                    "modality": frozen[row_id]["modality"],
                    "source_dataset": frozen[row_id]["source_dataset"],
                    "source_caption": frozen[row_id]["source_caption"],
                    "qwen_v3_concepts": json.dumps(selected[row_id]["qwen_v3_concepts"], ensure_ascii=False),
                    "kimi_k3_concepts": json.dumps(results[row_id].get("parsed_concepts") or [], ensure_ascii=False),
                    "pilot_stratum": selected[row_id]["pilot_stratum"],
                    "automatic_risk_flags": ";".join(audit_row["automatic_risk_flags"]),
                    "caption_supported": "",
                    "negation_correct": "",
                    "uncertainty_correct": "",
                    "generic_or_artifact": "",
                    "semantic_duplicate": "",
                    "preferred_output": "",
                    "reviewer_notes": "",
                }
            )

    errors = collections.Counter()
    retry_reasons = collections.Counter()
    for row in results.values():
        if row.get("status") != "success":
            errors[row.get("status") or "unknown"] += 1
        retry_reasons.update(row.get("retry_reasons") or [])
    api_summary = {
        "status": "PASS" if proportion(success_count, PILOT_SIZE) >= 0.995 else "FAIL",
        "selected_unique_records": PILOT_SIZE,
        "result_unique_records": len(results),
        "success": success_count,
        "cache_success": success_count,
        "api_attempts": sum(int(row.get("attempt_count", 0)) for row in results.values()),
        "error_distribution": dict(errors),
        "retry_reason_distribution": dict(retry_reasons),
        "model_requested": sorted({row.get("model_requested") for row in results.values()}),
        "model_returned": sorted({value for row in results.values() if (value := row.get("model_returned")) is not None}),
        "reasoning_effort": sorted({row.get("reasoning_effort") for row in results.values()}),
        "token_usage": dict(token_totals),
        "credential_or_headers_recorded": False,
    }
    write_json(artifact / "pilot_api_summary.json", api_summary)
    write_json(
        artifact / "pilot_cache_index.json",
        {
            "status": "PASS",
            "count": len(results),
            "success_count": success_count,
            "records": {
                row_id: {
                    "status": results[row_id].get("status"),
                    "source_caption_sha256": results[row_id].get("source_caption_sha256"),
                    "prompt_hash": results[row_id].get("prompt_hash"),
                    "created_timestamp": results[row_id].get("created_timestamp"),
                }
                for row_id in selection["stable_record_ids"]
            },
        },
    )
    cost_report = {
        "status": "TOKEN_ACCOUNTING_COMPLETE_PRICING_PENDING",
        "token_usage": dict(token_totals),
        "api_attempts": api_summary["api_attempts"],
        "successful_unique_records": success_count,
        "estimated_cost": None,
        "currency": None,
        "reason": "No fixed kimi-k3 price was supplied in the project or returned by the API. Token totals are preserved for billing reconciliation; no unverified price was invented.",
    }
    write_json(artifact / "pilot_cost_report.json", cost_report)
    (artifact / "git_state_after.txt").write_text(git_state(), encoding="utf-8")

    machine_status = "PASS" if all(machine_gates.values()) else "FAIL"
    report_status = "PENDING_HUMAN_REVIEW" if machine_status == "PASS" else "BLOCKED_AUTOMATIC_GATES"
    report = f"""# Kimi K3 2,000-record pilot quality report

Status: **{report_status}**

## Program-verified results

- Unique selected/result records: {PILOT_SIZE}/{len(results)}
- API success: {success_count}/{PILOT_SIZE} ({proportion(success_count, PILOT_SIZE):.4%})
- JSON parse rate: {proportion(parse_success, PILOT_SIZE):.4%}
- 1-8 concept compliance: {proportion(count_compliant, PILOT_SIZE):.4%}
- Empty rate: {proportion(empty, PILOT_SIZE):.4%}
- Eight-concept saturation: {proportion(saturation, PILOT_SIZE):.4%}
- Generic/ROI/artifact automatic hit rate: {proportion(len(set(generic_hits + artifact_hits)), PILOT_SIZE):.4%}
- Normalized exact duplicate rate: {proportion(len(duplicate_hits), PILOT_SIZE):.4%}
- Automatic gates: {machine_status}

## Human confirmation still required

Caption-supported precision, negation correctness, uncertainty correctness, and semantic-duplicate rate are intentionally not inferred from regexes or another LLM. Complete `pilot_human_audit_500.csv` before approving the remaining 68,356 records.

## Decision

The full 70,356-record generation is **not started**. Entry to the full run remains blocked until the 500-row human audit is completed and all semantic acceptance thresholds pass.
"""
    (artifact / "pilot_quality_report.md").write_text(report, encoding="utf-8")
    write_json(
        artifact / "resume_state.json",
        {
            "status": "PILOT_COMPLETE_STOPPED" if len(results) == PILOT_SIZE else "PILOT_INCOMPLETE",
            "selected_unique_limit": PILOT_SIZE,
            "result_unique_count": len(results),
            "success_count": success_count,
            "full_generation_started": False,
            "next_action": "Complete the blank 500-row human semantic audit and obtain user approval.",
        },
    )

    hash_targets = sorted(
        path for path in artifact.iterdir()
        if path.is_file() and path.name not in {"artifact_hashes.json"}
    )
    write_json(artifact / "artifact_hashes.json", {path.name: sha_file(path) for path in hash_targets})
    print(json.dumps({"status": report_status, "success": success_count, "results": len(results), "human_audit": str(csv_path)}))


if __name__ == "__main__":
    main()
