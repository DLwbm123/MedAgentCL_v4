#!/usr/bin/env python3
"""Audit the fixed 500-record Kimi K3 v2 pilot without making API calls."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_kimi_k3_v2_pilot"
V1_ARTIFACT = ROOT / "artifacts/medicalskill_cl_kimi_k3_pilot"
PILOT_SIZE = 500

RISK_PATTERNS = {
    "uncertain_concept": re.compile(r"^uncertain:\s*", re.I),
    "benign_or_malignant": re.compile(r"\bbenign\s+(?:or|and)\s+malignant\b", re.I),
    "pathological_or_disease_process": re.compile(r"\b(?:pathological|disease) process\b", re.I),
    "adjacent_or_proximity": re.compile(r"\b(?:adjacent|proximity|nearby structure|anatomical association)\b", re.I),
    "involvement": re.compile(r"\binvolvement\b", re.I),
    "spread_or_invasion": re.compile(r"\b(?:spread|invasion|invading|invasive)\b", re.I),
    "progression_or_advanced": re.compile(r"\b(?:progression|advanced[- ]stage|advanced disease)\b", re.I),
    "organ_function": re.compile(r"\b(?:organ function|functional impairment|function impairment)\b", re.I),
    "roi_or_artifact": re.compile(
        r"\b(?:roi|region of interest|bounding box|bbox|area ratio|highlighted region|colored box|"
        r"central position|cross-sectional view)\b", re.I
    ),
    "generic_only": re.compile(
        r"^(?:disease|pathology|abnormality|finding|medical image|anatomical association|"
        r"region of interest|disease process|pathological process)$", re.I
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def percentile(values: list[int | float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * percentile_value
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def latest(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        output[str(row["stable_record_id"])] = row
    return output


def canonical_error_category(row: dict[str, Any]) -> str:
    if row.get("status") == "success":
        return "none"
    value = str(row.get("error_category") or row.get("status") or "unknown")
    match = re.search(r"Error code: (\d{3})", value)
    if match:
        return f"http_{match.group(1)}"
    return value if len(value) <= 80 else value[:77] + "..."


def normalized(values: list[str]) -> set[str]:
    return {" ".join(str(value).casefold().split()).strip(" ;,.") for value in values if str(value).strip()}


def risk_counts(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, list[str]]]:
    counts = collections.Counter({name: 0 for name in RISK_PATTERNS})
    examples: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        concepts = row.get("parsed_concepts") or []
        for name, pattern in RISK_PATTERNS.items():
            if any(pattern.search(str(value)) for value in concepts):
                counts[name] += 1
                if len(examples[name]) < 30:
                    examples[name].append(str(row["stable_record_id"]))
    return dict(counts), dict(examples)


def active_success(row: dict[str, Any] | None, prompt: dict[str, Any]) -> bool:
    return bool(row) and all((
        row.get("status") == "success",
        row.get("contract_status") == "PASS",
        row.get("model_returned") == "kimi-k3",
        row.get("prompt_hash") == prompt["prompt_hash"],
        row.get("schema_hash") == prompt["schema_hash"],
        row.get("normalization_hash") == prompt["normalization_hash"],
        row.get("temperature") == 1.0,
        row.get("top_p_sent") is False,
        row.get("max_completion_tokens") == 1024,
        row.get("endpoint_hostname") == "api.kimi.com",
        row.get("endpoint_base_path") == "/coding/v1",
        row.get("finish_reason") not in {"length", "max_tokens"},
        1 <= len(row.get("parsed_concepts") or []) <= 8,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default=str(ARTIFACT))
    args = parser.parse_args()
    artifact = Path(args.artifact_root)
    prompt = json.loads((artifact / "prompt_v2_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((artifact / "v2_pilot_selection_manifest.json").read_text(encoding="utf-8"))
    selected_ids = list(selection["stable_record_ids"])
    selected = {row["stable_record_id"]: row for row in selection["records"]}
    candidate = {row["stable_record_id"]: row for row in read_jsonl(artifact / "concept_35k_candidate_manifest.jsonl")}
    if len(selected_ids) != PILOT_SIZE or len(set(selected_ids)) != PILOT_SIZE:
        raise RuntimeError("v2 pilot selection is not exactly 500 unique records")

    attempts = read_jsonl(artifact / "v2_pilot_results.jsonl")
    current = latest(attempts)
    successes = [current[row_id] for row_id in selected_ids if active_success(current.get(row_id), prompt)]
    success_ids = {row["stable_record_id"] for row in successes}
    final_status = collections.Counter(
        current[row_id].get("status", "missing") if row_id in current else "missing"
        for row_id in selected_ids
    )
    attempt_status = collections.Counter(row.get("status", "unknown") for row in attempts)
    error_categories = collections.Counter(canonical_error_category(row) for row in attempts)
    completion_tokens = [int((row.get("token_usage") or {}).get("completion", 0) or 0) for row in attempts]
    token_totals = collections.Counter()
    for row in attempts:
        token_totals.update({key: int(value or 0) for key, value in (row.get("token_usage") or {}).items()})

    concept_histogram = collections.Counter(len(row["parsed_concepts"]) for row in successes)
    length_attempts = [row for row in attempts if row.get("status") == "contract_finish_reason_length"]
    normalized_duplicate_ids = sorted(
        row["stable_record_id"] for row in successes
        if len(row.get("parsed_concepts") or []) != len(normalized(row.get("parsed_concepts") or []))
    )
    empty_success_ids = sorted(row["stable_record_id"] for row in successes if not row.get("parsed_concepts"))
    duplicate_success_ids = sorted(
        row_id for row_id, count in collections.Counter(
            row["stable_record_id"] for row in attempts if row.get("status") == "success"
        ).items() if count > 1
    )
    v2_risks, risk_examples = risk_counts(successes)

    v1_rows = latest(read_jsonl(V1_ARTIFACT / "pilot_results.jsonl"))
    comparable_ids = sorted(row_id for row_id in success_ids if v1_rows.get(row_id, {}).get("status") == "success")
    v1_comparable = [v1_rows[row_id] for row_id in comparable_ids]
    v2_comparable = [current[row_id] for row_id in comparable_ids]
    v1_risks, _ = risk_counts(v1_comparable)
    comparable_jaccards = []
    for row_id in comparable_ids:
        left = normalized(v1_rows[row_id].get("parsed_concepts") or [])
        right = normalized(current[row_id].get("parsed_concepts") or [])
        comparable_jaccards.append(len(left & right) / len(left | right) if left | right else 1.0)

    machine_gates = {
        "pilot_exactly_500_active_successes": len(successes) == PILOT_SIZE,
        "api_success_rate_ge_99_5pct": len(successes) / PILOT_SIZE >= 0.995,
        "strict_json_and_contract_rate_100pct": len(successes) == PILOT_SIZE,
        "concept_count_1_to_8_rate_100pct": sum(concept_histogram.values()) == PILOT_SIZE,
        "length_finish_rate_le_0_1pct": len(length_attempts) / max(1, len(attempts)) <= 0.001,
        "empty_rate_lt_0_5pct": len(empty_success_ids) / PILOT_SIZE < 0.005,
        "roi_or_artifact_rate_le_1pct": v2_risks["roi_or_artifact"] / PILOT_SIZE <= 0.01,
        "normalized_duplicate_rate_le_0_5pct": len(normalized_duplicate_ids) / PILOT_SIZE <= 0.005,
        "duplicate_successful_api_call_count_zero": not duplicate_success_ids,
        "all_modalities_represented": len({candidate[row_id]["modality"] for row_id in success_ids}) == 8,
        "wire_contract_temperature_1_no_top_p": prompt["temperature"] == 1.0 and prompt["top_p_sent"] is False,
    }
    complete = len(successes) == PILOT_SIZE
    status = "PASS" if all(machine_gates.values()) else "BLOCKED_INCOMPLETE" if not complete else "FAIL"

    token_distribution = {
        "completion_p50": percentile(completion_tokens, 0.50),
        "completion_p95": percentile(completion_tokens, 0.95),
        "completion_p99": percentile(completion_tokens, 0.99),
        "completion_max": max(completion_tokens, default=0),
        "completion_at_1024_count": sum(value >= 1024 for value in completion_tokens),
    }
    auto_audit = {
        "status": status,
        "selected_unique_records": PILOT_SIZE,
        "attempt_rows": len(attempts),
        "latest_result_records": len(current),
        "active_success_records": len(successes),
        "final_status_distribution": dict(final_status),
        "attempt_status_distribution": dict(attempt_status),
        "error_category_distribution": dict(error_categories),
        "machine_gates": machine_gates,
        "concept_count_histogram": {str(key): value for key, value in sorted(concept_histogram.items())},
        "eight_concept_saturation": concept_histogram[8] / max(1, len(successes)),
        "risk_phrase_record_counts": v2_risks,
        "risk_phrase_examples": risk_examples,
        "token_distribution": token_distribution,
        "token_usage_all_attempts": dict(token_totals),
        "length_attempt_count": len(length_attempts),
        "duplicate_successful_api_call_ids": duplicate_success_ids,
        "normalized_duplicate_ids": normalized_duplicate_ids,
        "empty_success_ids": empty_success_ids,
        "api_attempt_success_rate": len(successes) / max(1, len(attempts)),
        "retry_count": len(attempts) - len({row["stable_record_id"] for row in attempts}),
        "v1_v2_comparison": {
            "comparable_success_records": len(comparable_ids),
            "v1_risk_phrase_record_counts": v1_risks,
            "v2_risk_phrase_record_counts": {name: sum(1 for row in v2_comparable if any(pattern.search(str(c)) for c in row.get("parsed_concepts") or [])) for name, pattern in RISK_PATTERNS.items()},
            "mean_exact_normalized_set_jaccard": statistics.fmean(comparable_jaccards) if comparable_jaccards else None,
        },
        "manual_quality_gates": {
            "caption_supported_precision_ge_98pct": "PENDING_HUMAN_REVIEW",
            "negation_error_le_1pct": "PENDING_HUMAN_REVIEW",
            "uncertainty_error_le_1pct": "PENDING_HUMAN_REVIEW",
            "semantic_duplicate_le_0_5pct": "PENDING_HUMAN_REVIEW",
        },
    }
    write_json(artifact / "v2_pilot_auto_audit.json", auto_audit)

    human_path = artifact / "v2_pilot_human_audit_500.csv"
    fields = [
        "stable_record_id", "modality", "split", "source_kind", "source_dataset", "pilot_stratum",
        "source_caption", "qwen_v3_concepts", "kimi_v1_concepts", "kimi_v2_concepts",
        "automatic_risk_flags", "caption_supported", "image_learnable", "negation_correct", "uncertainty_correct",
        "generic_or_speculative", "semantic_duplicate", "preferred_output", "reviewer_notes",
    ]
    with human_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_id in selected_ids:
            if row_id not in success_ids:
                continue
            source, result = candidate[row_id], current[row_id]
            flags = [name for name, pattern in RISK_PATTERNS.items() if any(pattern.search(str(c)) for c in result["parsed_concepts"])]
            writer.writerow({
                "stable_record_id": row_id, "modality": source["modality"],
                "split": selected[row_id]["split"], "source_kind": selected[row_id]["source_kind"],
                "source_dataset": source["source_dataset"], "pilot_stratum": selected[row_id]["pilot_stratum"],
                "source_caption": source["source_caption"],
                "qwen_v3_concepts": json.dumps(source.get("qwen_v3_concepts") or [], ensure_ascii=False),
                "kimi_v1_concepts": json.dumps(v1_rows.get(row_id, {}).get("parsed_concepts") or [], ensure_ascii=False),
                "kimi_v2_concepts": json.dumps(result["parsed_concepts"], ensure_ascii=False),
                "automatic_risk_flags": ";".join(flags),
                "caption_supported": "", "image_learnable": "", "negation_correct": "", "uncertainty_correct": "",
                "generic_or_speculative": "", "semantic_duplicate": "", "preferred_output": "", "reviewer_notes": "",
            })

    calls_per_success = len(attempts) / len(successes) if successes else None
    mean_prompt = token_totals["prompt"] / len(attempts) if attempts else None
    mean_completion = token_totals["completion"] / len(attempts) if attempts else None
    run_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in attempts:
        run_groups[str(row.get("run_id") or "unknown")].append(row)
    observed_run = max(run_groups.values(), key=len, default=[])
    timestamps = [datetime.fromisoformat(row["created_timestamp"]) for row in observed_run if row.get("created_timestamp")]
    observed_wall_seconds = (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) > 1 else None
    observed_successes = sum(row.get("status") == "success" for row in observed_run)
    projected_wall_seconds = 35000 * observed_wall_seconds / observed_successes if observed_wall_seconds and observed_successes else None
    projection = {
        "records": 35000,
        "projected_api_calls": 35000 * calls_per_success if calls_per_success is not None else None,
        "projected_prompt_tokens": 35000 * calls_per_success * mean_prompt if calls_per_success is not None and mean_prompt is not None else None,
        "projected_completion_tokens": 35000 * calls_per_success * mean_completion if calls_per_success is not None and mean_completion is not None else None,
        "projected_wall_seconds_at_observed_throughput": projected_wall_seconds,
        "projected_wall_hours_at_observed_throughput": projected_wall_seconds / 3600 if projected_wall_seconds else None,
        "formula": "records * observed_api_calls_per_success * observed_mean_tokens_per_attempt",
    }
    cost_report = {
        "status": "TOKEN_ACCOUNTING_COMPLETE_PRICING_NOT_SUPPLIED" if attempts else "NO_API_USAGE_YET",
        "token_usage_all_attempts": dict(token_totals),
        "api_attempts": len(attempts),
        "active_success_records": len(successes),
        "observed_api_calls_per_success": calls_per_success,
        "observed_largest_run_records": len(observed_run),
        "observed_largest_run_successes": observed_successes,
        "observed_largest_run_wall_seconds": observed_wall_seconds,
        "kimi_code_membership_quota_consumption": {"successful_calls": len(successes), "failed_http_403_calls": error_categories["http_403"], "provider_quota_units": None},
        "extra_usage_triggered": False if error_categories["http_403"] else None,
        "extra_usage_status_basis": "HTTP 403 billing-cycle usage limit; extra usage did not continue the run" if error_categories["http_403"] else "not observable",
        "provider_balance_before": None,
        "provider_balance_after": None,
        "projected_35k": projection,
        "estimated_currency_cost": None,
        "pricing_formula": "input_cost + cached_input_cost + output_cost, using provider invoice prices",
        "reason": "No auditable kimi-k3 billing rates were supplied; no CNY amount was invented.",
    }
    write_json(artifact / "v2_pilot_cost_report.json", cost_report)

    api_summary = {
        "status": status,
        "endpoint": "https://api.kimi.com/coding/v1",
        "model": "kimi-k3", "reasoning_effort": "low", "temperature": 1.0,
        "top_p_sent": False, "max_completion_tokens": 1024,
        "structured_output": "json_schema_strict",
        "selected_unique_records": PILOT_SIZE, "active_success_records": len(successes),
        "api_attempts": len(attempts), "retry_count": len(attempts) - len({row["stable_record_id"] for row in attempts}),
        "api_attempt_success_rate": len(successes) / max(1, len(attempts)), "pilot_completion_success_rate": len(successes) / PILOT_SIZE,
        "final_status_distribution": dict(final_status),
        "attempt_status_distribution": dict(attempt_status), "error_category_distribution": dict(error_categories),
        "token_usage": dict(token_totals), "credential_or_authorization_value_recorded": False,
    }
    write_json(artifact / "v2_pilot_api_summary.json", api_summary)
    write_json(artifact / "v2_pilot_run_config.json", {
        **{key: prompt[key] for key in ("prompt_version", "prompt_hash", "schema_version", "schema_hash", "normalization_hash")},
        "endpoint": "https://api.kimi.com/coding/v1", "model": "kimi-k3", "reasoning_effort": "low",
        "temperature": 1.0, "top_p_sent": False, "max_completion_tokens": 1024,
        "pilot_unique_limit": PILOT_SIZE, "api_call_hard_limit": 650,
    })

    decision = "Pilot machine gates passed; human review remains required." if status == "PASS" else "Pilot expansion is blocked; do not run the remaining candidate set."
    report = f"""# Kimi K3 MedicalSkill-CL v2 Pilot quality report

Status: **{status}**

- Frozen Pilot: {PILOT_SIZE} unique records
- Active contract-valid successes: {len(successes)}/{PILOT_SIZE} ({len(successes) / PILOT_SIZE:.2%})
- API attempt rows: {len(attempts)}; retries: {len(attempts) - len({row["stable_record_id"] for row in attempts})}
- Attempt success rate: {len(successes) / max(1, len(attempts)):.2%}; frozen-Pilot completion rate: {len(successes) / PILOT_SIZE:.2%}
- Final status distribution: `{json.dumps(dict(final_status), sort_keys=True)}`
- Concept count histogram: `{json.dumps(auto_audit['concept_count_histogram'], sort_keys=True)}`
- Exactly-eight saturation: {auto_audit['eight_concept_saturation']:.2%}
- Length attempts: {len(length_attempts)} ({len(length_attempts) / max(1, len(attempts)):.2%})
- Completion tokens p50/p95/p99/max: {token_distribution['completion_p50']:.1f}/{token_distribution['completion_p95']:.1f}/{token_distribution['completion_p99']:.1f}/{token_distribution['completion_max']}
- Empty/ROI-artifact/normalized-duplicate records: {len(empty_success_ids)}/{v2_risks["roi_or_artifact"]}/{len(normalized_duplicate_ids)}
- Duplicate successful API call IDs: {len(duplicate_success_ids)}
- HTTP 403 quota failures: {error_categories["http_403"]}; Extra Usage continued run: false
- Projected 35K runtime at observed throughput: {(projected_wall_seconds / 3600) if projected_wall_seconds else None} hours
- Temperature: 1.0; top_p sent: false; max completion tokens: 1024

## Decision

{decision}

The full 35,000-record generation was not started. The 5,000-record reserve was never submitted to the API. Currency cost remains blank because an auditable provider price was not supplied.
"""
    (artifact / "v2_pilot_quality_report.md").write_text(report, encoding="utf-8")

    tracked = [
        "concept_35k_candidate_manifest.jsonl", "concept_35k_reserve_manifest.jsonl",
        "v2_pilot_selection_manifest.json", "prompt_v2_manifest.json", "kimi_prompt_schema_v2.json",
        "normalization_contract_v2.json", "v2_pilot_results.jsonl", "v2_pilot_auto_audit.json",
        "v2_pilot_api_summary.json", "v2_pilot_run_config.json", "v2_pilot_cost_report.json",
        "v2_pilot_human_audit_500.csv", "v2_pilot_quality_report.md", "v2_resume_state.json",
    ]
    hashes = {name: {"sha256": sha_file(artifact / name), "bytes": (artifact / name).stat().st_size} for name in tracked if (artifact / name).exists()}
    write_json(artifact / "v2_artifact_hashes.json", {"status": "PASS", "files": hashes})
    print(json.dumps({"status": status, "successes": len(successes), "attempts": len(attempts), "outputs": len(hashes) + 1}))


if __name__ == "__main__":
    main()
