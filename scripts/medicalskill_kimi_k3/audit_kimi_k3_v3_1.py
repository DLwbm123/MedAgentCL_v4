#!/usr/bin/env python3
"""Audit the isolated MedicalSkill-CL Kimi v3.1 targeted 50-row Pilot."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_kimi_k3_v3_1 as runner  # noqa: E402

ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_kimi_k3_v3_1_k2_6_routing"
MANUAL_FIELDS = (
    "caption_supported",
    "negative_removed",
    "contradiction_resolved",
    "background_enumeration_absent",
    "important_positive_omitted",
    "reviewer_notes",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return runner.read_results(path)


def existing_manual(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["stable_id"]: row for row in csv.DictReader(handle)}


def write_human_csv(path: Path, records: list[dict[str, Any]],
                    latest: dict[str, dict[str, Any]],
                    sources: dict[str, dict[str, Any]]) -> None:
    old = existing_manual(path)
    fields = [
        "stable_id", "modality", "selection_reasons", "source_caption",
        "v3_concepts", "v3_1_concepts", "semantic_violations",
        "response_model_id", "reasoning_token_usage", "latency",
        *MANUAL_FIELDS,
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            record_id = record["stable_id"]
            result, source = latest.get(record_id, {}), sources[record_id]
            previous = old.get(record_id, {})
            writer.writerow({
                "stable_id": record_id,
                "modality": record["modality"],
                "selection_reasons": "|".join(record["selection_reasons"]),
                "source_caption": source["source_caption"],
                "v3_concepts": json.dumps(record["v3_concepts"], ensure_ascii=False),
                "v3_1_concepts": json.dumps(result.get("parsed_concepts") or [], ensure_ascii=False),
                "semantic_violations": "|".join(result.get("semantic_violations") or []),
                "response_model_id": result.get("response_model_id"),
                "reasoning_token_usage": result.get("reasoning_token_usage"),
                "latency": result.get("latency"),
                **{field: previous.get(field, "") for field in MANUAL_FIELDS},
            })


def yes(value: str) -> bool:
    return value.strip().casefold() in {"yes", "true", "1", "y"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default=str(ARTIFACT))
    args = parser.parse_args()
    artifact = Path(args.artifact_root)
    prompt, _, config, manifest = runner.load_contract(artifact)
    sources, _ = runner.load_sources(artifact, config)
    history, latest, _ = runner.load_history(artifact / "v3_1_results.jsonl")
    records = manifest["records"]
    selected_ids = [row["stable_id"] for row in records]
    active = {
        record_id: latest[record_id]
        for record_id in selected_ids
        if record_id in latest
        and runner.active_success(latest[record_id], sources[record_id], prompt, config)
    }
    rows = [active[record_id] for record_id in selected_ids if record_id in active]
    uncertain = sum(
        any(value.startswith("uncertain:") for value in row["parsed_concepts"])
        for row in rows
    )
    negated = sum(
        any(runner.NEGATED.search(value) for value in row["parsed_concepts"])
        for row in rows
    )
    contradictions = sum(bool(runner.contradiction_hits(row["parsed_concepts"])) for row in rows)
    artifacts = sum(
        any(runner.FORBIDDEN.search(value) or value in runner.GENERIC for value in row["parsed_concepts"])
        for row in rows
    )
    backgrounds = sum(
        "background_organ_enumeration" in runner.semantic_violations(row["parsed_concepts"])
        for row in rows
    )
    duplicate = sum(
        len(row["parsed_concepts"]) != len(set(row["parsed_concepts"]))
        for row in rows
    )
    gates = {
        "api_success_50_of_50": len(rows) == 50,
        "json_and_1_to_8_100pct": len(rows) == 50 and all(1 <= len(row["parsed_concepts"]) <= 8 for row in rows),
        "reasoning_tokens_zero": len(rows) == 50 and all(int(row.get("reasoning_token_usage", 0) or 0) == 0 for row in rows),
        "reasoning_content_absent": len(rows) == 50 and all(row.get("reasoning_content_present") is False for row in rows),
        "uncertain_prefix_zero": uncertain == 0,
        "negated_concept_zero": negated == 0,
        "mutually_exclusive_pair_zero": contradictions == 0,
        "roi_artifact_zero": artifacts == 0,
        "background_organ_enumeration_zero": backgrounds == 0,
        "exact_duplicate_zero": duplicate == 0,
    }

    human_path = artifact / "v3_1_pilot_human_audit_50.csv"
    write_human_csv(human_path, records, latest, sources)
    manual = existing_manual(human_path)
    reviewed = [
        row for row in manual.values()
        if all((row.get(field) or "").strip() for field in MANUAL_FIELDS[:-1])
    ]
    manual_gates = {
        "all_50_reviewed": len(reviewed) == 50,
        "caption_supported_ge_98pct": len(reviewed) == 50 and sum(yes(row["caption_supported"]) for row in reviewed) >= 49,
        "negative_removed_50_of_50": len(reviewed) == 50 and all(yes(row["negative_removed"]) for row in reviewed),
        "contradiction_resolved_50_of_50": len(reviewed) == 50 and all(yes(row["contradiction_resolved"]) for row in reviewed),
        "background_enumeration_absent_50_of_50": len(reviewed) == 50 and all(yes(row["background_enumeration_absent"]) for row in reviewed),
        "important_positive_omission_zero": len(reviewed) == 50 and not any(yes(row["important_positive_omitted"]) for row in reviewed),
    }
    machine_pass = all(gates.values())
    status = (
        "PASS" if machine_pass and all(manual_gates.values())
        else "PENDING_HUMAN_REVIEW" if machine_pass
        else "FAIL"
    )
    concepts = [len(row["parsed_concepts"]) for row in rows]
    prompt_tokens = [int(row.get("prompt_tokens", 0) or 0) for row in rows]
    cached_tokens = [int(row.get("cached_tokens", 0) or 0) for row in rows]
    completion_tokens = [int(row.get("completion_tokens", 0) or 0) for row in rows]
    output = {
        "status": status,
        "full_run_blocked": status != "PASS",
        "selected": 50,
        "active_success": len(rows),
        "attempt_rows_total": len(history),
        "api_attempt_rows_total": sum(row.get("api_call_performed") is not False for row in history),
        "deterministic_recertification_rows": sum(row.get("api_call_performed") is False for row in history),
        "machine_gates": gates,
        "manual_gates": manual_gates,
        "human_reviewed": len(reviewed),
        "metrics": {
            "concept_count_histogram": dict(sorted(collections.Counter(concepts).items())),
            "mean_concept_count": statistics.fmean(concepts) if concepts else None,
            "cap8_count": sum(value == 8 for value in concepts),
            "uncertain_prefix_count": uncertain,
            "negated_concept_count": negated,
            "mutually_exclusive_pair_count": contradictions,
            "roi_artifact_count": artifacts,
            "background_organ_enumeration_count": backgrounds,
            "exact_duplicate_count": duplicate,
            "mean_latency": statistics.fmean(float(row.get("latency", 0) or 0) for row in rows) if rows else None,
            "mean_prompt_tokens": statistics.fmean(prompt_tokens) if prompt_tokens else None,
            "mean_cached_tokens": statistics.fmean(cached_tokens) if cached_tokens else None,
            "mean_completion_tokens": statistics.fmean(completion_tokens) if completion_tokens else None,
            "reasoning_tokens_total": sum(int(row.get("reasoning_token_usage", 0) or 0) for row in rows),
            "response_model_distribution": dict(collections.Counter(row.get("response_model_id") for row in rows)),
        },
        "human_audit_path": str(human_path),
        "prompt_hash": prompt["prompt_hash"],
        "schema_hash": prompt["schema_hash"],
        "run_config_hash": config["run_config_hash"],
    }
    runner.atomic_write_json(artifact / "v3_1_pilot_acceptance.json", output)
    runner.atomic_write_json(artifact / "v3_1_pilot_quality_report.json", output)
    (artifact / "v3_1_pilot_quality_report.md").write_text(
        f"# Kimi K2.6-routing v3.1 targeted Pilot\n\n"
        f"Status: **{status}**\n\n"
        f"Active results: {len(rows)}/50. Machine gates: {'PASS' if machine_pass else 'FAIL'}. "
        f"Human reviewed: {len(reviewed)}/50. Formal batches remain blocked until status is PASS.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": status,
        "machine_status": "PASS" if machine_pass else "FAIL",
        "human_reviewed": len(reviewed),
        "full_run_blocked": status != "PASS",
        "api_called": False,
    }))


if __name__ == "__main__":
    main()
