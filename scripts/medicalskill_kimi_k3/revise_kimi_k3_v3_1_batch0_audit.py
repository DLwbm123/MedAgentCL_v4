#!/usr/bin/env python3
"""Append-only batch-0 audit correction after the anatomy-split false positive."""
from __future__ import annotations

import collections
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_kimi_k3_v3_1 as runner  # noqa: E402
import v3_1_quality_policy as quality  # noqa: E402

ARTIFACT = Path("/root/MedAgentCL_v4/artifacts/medicalskill_cl_kimi_k3_v3_1_k2_6_routing")
REPORT_LABEL = "cycle_01_batch_00"
REVISION_LABEL = "cycle_01_batch_00_audit_revision_001"


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    artifact = ARTIFACT
    results_path = artifact / "v3_1_results.jsonl"
    state_path = artifact / "v3_1_resume_state.json"
    report_dir = artifact / "batch_reports"
    event_path = report_dir / "batch0_audit_revision_events.jsonl"
    prior_events = runner.read_results(event_path) if event_path.exists() else []
    if any(row.get("revision_label") == REVISION_LABEL for row in prior_events):
        print(json.dumps({"status": "ALREADY_APPLIED", "api_called": False}))
        return

    prompt, _, config, _ = runner.load_contract(artifact)
    sources, stable_ids = runner.load_sources(artifact, config)
    selected_ids = stable_ids[:runner.BATCH_SIZE]
    history, latest, ordinals = runner.load_history(results_path)
    old_report_path = report_dir / f"{REPORT_LABEL}.json"
    old_report = json.loads(old_report_path.read_text())
    original_run_id = old_report["run_id"]
    run_rows = [row for row in history if row.get("run_id") == original_run_id]

    revision_run_id = (
        datetime.now(timezone.utc).strftime("v3.1-batch0-audit-r001-%Y%m%dT%H%M%SZ-")
        + uuid.uuid4().hex[:12]
    )
    revisions = []
    for record_id in selected_ids:
        row = latest.get(record_id)
        if not runner.terminal_rejected(row) or row.get("output_source") != "retry_model":
            continue
        concepts = list(row.get("retry_model_concepts") or [])
        audited = quality.audit_concepts(
            concepts,
            sources[record_id]["source_caption"],
            sources[record_id].get("modality", ""),
        )
        if audited["retry_reasons"]:
            continue
        revised = {
            **row,
            "run_id": revision_run_id,
            "mode": "batch0_quality_reclassification_revision_001",
            "final_concepts": concepts,
            "parsed_concepts": concepts,
            "audit_flags": audited["audit_flags"],
            "audit_details": audited["audit_details"],
            "semantic_violations": audited["audit_flags"],
            "quality_status": "PASS",
            "cache_status": "active_v3_1_success",
            "error_type": None,
            "repair_or_retry_reason": list(row.get("repair_or_retry_reason") or [])
            + ["audit_policy_correction_allow_anatomy_plus_localized_lesion"],
            "api_call_performed": False,
            "api_attempt_kind": "quality_reclassification",
            "attempt_count": ordinals[record_id],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "audit_policy_revision": 1,
        }
        revisions.append(revised)
        latest[record_id] = revised

    append_rows(results_path, revisions)
    history, latest, _ = runner.load_history(results_path)
    state = runner.state_payload(
        "BATCH0_AUDIT_REVISED",
        selected_ids,
        latest,
        sources,
        prompt,
        config,
        results_path,
        "formal_batch",
        "cycle_01",
        0,
    )
    runner.atomic_write_json(state_path, state)

    api_rows = [row for row in run_rows if row.get("api_call_performed") is not False]
    original_rows = [row for row in run_rows if row.get("api_attempt_kind") == "original"]
    retry_required_rows = [row for row in run_rows if row.get("quality_status") == "RETRY_REQUIRED"]
    triggered_ids = {row["stable_id"] for row in retry_required_rows}
    direct_ids = {
        row["stable_id"] for row in run_rows
        if row.get("quality_status") == "PASS" and row.get("output_source") == "raw_model"
    }
    accepted_rows = [
        latest[record_id] for record_id in selected_ids
        if runner.active_success(latest.get(record_id), sources[record_id], prompt, config)
    ]
    rejected_ids = [
        record_id for record_id in selected_ids
        if runner.terminal_rejected(latest.get(record_id))
    ]

    corrected_initial = {
        row["stable_id"]: quality.audit_concepts(
            row.get("raw_model_concepts") or [],
            sources[row["stable_id"]]["source_caption"],
            sources[row["stable_id"]].get("modality", ""),
        )
        for row in original_rows
    }
    corrected_trigger_ids = {
        record_id for record_id, audit in corrected_initial.items()
        if audit["retry_reasons"]
    }
    initial_flags = collections.Counter(
        flag for audit in corrected_initial.values() for flag in audit["retry_reasons"]
    )
    final_flags = collections.Counter(
        flag for row in accepted_rows for flag in (row.get("audit_flags") or [])
    )
    counts = [len(row["final_concepts"]) for row in accepted_rows]
    modalities: dict[str, dict[str, int]] = {}
    for record_id in selected_ids:
        modality = sources[record_id].get("modality", "unknown")
        bucket = modalities.setdefault(modality, {"selected": 0, "accepted": 0, "rejected": 0})
        bucket["selected"] += 1
        if record_id in rejected_ids:
            bucket["rejected"] += 1
        else:
            bucket["accepted"] += 1

    usage_keys = ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_token_usage")
    usage = {key: sum(int(row.get(key, 0) or 0) for row in api_rows) for key in usage_keys}
    latencies = sorted(float(row.get("latency", 0) or 0) for row in api_rows)
    checks = {
        "prompt_hash_unchanged": prompt["prompt_hash"] == "fb63d4b1963d5b9c93188ed65b7ec5677eeaf7ecb8e38b86bb73e640f516ae68",
        "schema_hash_unchanged": prompt["schema_hash"] == "14830fcd9b334ac7ce24c17cd0704a373ccec247da03bf1435dad139a8d38aea",
        "run_config_hash_unchanged": config["run_config_hash"] == "382703cd787c9c49cce0ccdc2cda71615ccbb16f35820cc4a9ad290243d230a1",
        "all_api_calls_http_200": len(api_rows) == 1288 and all(row.get("http_status") == 200 for row in api_rows),
        "reasoning_tokens_zero": usage["reasoning_token_usage"] == 0,
        "all_selected_terminal": len(accepted_rows) + len(rejected_ids) == 1000,
        "all_selected_accepted": len(accepted_rows) == 1000,
        "active_retry_violations_zero": not any(
            set(row.get("audit_flags") or []) & quality.RETRY_FLAGS for row in accepted_rows
        ),
        "resume_state_valid": runner.valid_state(state_path, prompt, config, results_path),
        "batch_1_requires_user_authorization": (
            runner.latest_acceptance_revision(artifact) or {}
        ).get("remaining_cycle_batches_authorized") is False,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "HOLD",
        "review_semantics": "machine_assisted_semantic_audit",
        "human_review_status": "NOT_VERIFIED_BY_USER",
        "api_called_by_revision": False,
        "original_batch_run_id": original_run_id,
        "audit_revision_run_id": revision_run_id,
        "selected": len(selected_ids),
        "cache_hits_at_start": 1,
        "api_attempts": len(api_rows),
        "api_http_200": sum(row.get("http_status") == 200 for row in api_rows),
        "api_success_rate": sum(row.get("http_status") == 200 for row in api_rows) / len(api_rows),
        "direct_qualified_records": len(direct_ids),
        "direct_qualified_rate": len(direct_ids) / len(original_rows),
        "actual_retry_triggered_records": len(triggered_ids),
        "actual_retry_trigger_rate": len(triggered_ids) / len(original_rows),
        "corrected_policy_retry_triggered_records": len(corrected_trigger_ids),
        "corrected_policy_retry_trigger_rate": len(corrected_trigger_ids) / len(original_rows),
        "superseded_policy_false_positive_retry_records": len(triggered_ids - corrected_trigger_ids),
        "retry_success_records_after_audit_revision": len(triggered_ids) - len(rejected_ids),
        "retry_success_rate_after_audit_revision": (
            (len(triggered_ids) - len(rejected_ids)) / len(triggered_ids)
        ),
        "final_accepted_records": len(accepted_rows),
        "final_rejected_records": len(rejected_ids),
        "final_rejection_rate": len(rejected_ids) / len(selected_ids),
        "quality_reclassification_rows_appended": len(revisions),
        "concept_count_histogram": dict(sorted(collections.Counter(counts).items())),
        "cap8_count": sum(value == 8 for value in counts),
        "cap8_ratio": sum(value == 8 for value in counts) / len(counts),
        "modality_statistics": dict(sorted(modalities.items())),
        "corrected_initial_retry_audit_hit_counts": dict(sorted(initial_flags.items())),
        "final_audit_hit_counts": dict(sorted(final_flags.items())),
        "usage": usage,
        "latency": {
            "mean_seconds": sum(latencies) / len(latencies),
            "p95_seconds": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
            "total_seconds": sum(latencies),
        },
        "quota": {
            "cycle_hard_api_call_limit": runner.CYCLE_HARD_LIMIT,
            "cycle_api_calls_used": len(api_rows),
            "cycle_api_calls_remaining": runner.CYCLE_HARD_LIMIT - len(api_rows),
            "manual_remaining_weekly_quota_percent": None,
            "manual_remaining_two_hour_quota_percent": None,
        },
        "checks": checks,
        "prompt_hash": prompt["prompt_hash"],
        "schema_hash": prompt["schema_hash"],
        "run_config_hash": config["run_config_hash"],
        "original_batch_report": {
            "path": str(old_report_path),
            "sha256": runner.sha_file(old_report_path),
        },
        "results_hash": runner.sha_file(results_path),
        "resume_state_hash": runner.sha_file(state_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    revised_path = report_dir / f"{REVISION_LABEL}.json"
    runner.atomic_write_json(revised_path, report)
    markdown = (
        f"# Batch 0 audit revision 001\n\n"
        f"Status: **{report['status']}**\n\n"
        "Review type: machine-assisted semantic audit; "
        "user verification status: **NOT_VERIFIED_BY_USER**.\n\n"
        f"API: {report['api_http_200']}/{report['api_attempts']} HTTP 200; "
        f"accepted: {report['final_accepted_records']}/1000; "
        f"final rejected: {report['final_rejected_records']}.\n\n"
        f"The superseded parent/child audit caused "
        f"{report['superseded_policy_false_positive_retry_records']} unnecessary retries. "
        "No concepts were rewritten and this revision made no API calls.\n"
    )
    (report_dir / f"{REVISION_LABEL}.md").write_text(markdown, encoding="utf-8")
    runner.atomic_write_json(report_dir / f"{REVISION_LABEL}.hash.json", {
        "report_sha256": runner.sha_file(revised_path),
        "results_sha256": runner.sha_file(results_path),
        "resume_state_sha256": runner.sha_file(state_path),
        "original_report_sha256": runner.sha_file(old_report_path),
    })
    event = {
        "revision_label": REVISION_LABEL,
        "timestamp": report["timestamp"],
        "reason": "Correct parent/child false positives while preserving the frozen anatomy-split contract.",
        "quality_reclassification_rows_appended": len(revisions),
        "api_called": False,
        "report_path": str(revised_path),
        "report_sha256": runner.sha_file(revised_path),
    }
    append_rows(event_path, [event])
    print(json.dumps({
        "status": report["status"],
        "accepted": len(accepted_rows),
        "rejected": len(rejected_ids),
        "reclassified": len(revisions),
        "api_called": False,
    }))


if __name__ == "__main__":
    main()
