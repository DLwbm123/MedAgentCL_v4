#!/usr/bin/env python3
"""Append-only correction of v3.1 review semantics and output provenance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_kimi_k3_v3_1 as runner  # noqa: E402
import v3_1_quality_policy as quality  # noqa: E402

ARTIFACT = Path("/root/MedAgentCL_v4/artifacts/medicalskill_cl_kimi_k3_v3_1_k2_6_routing")
ORIGINAL_FILES = (
    "v3_1_pilot_human_audit_50.csv",
    "v3_1_pilot_acceptance.json",
    "v3_1_pilot_quality_report.json",
    "v3_1_pilot_quality_report.md",
    "v3_1_closure_report.md",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def stages(path: Path) -> set[str]:
    return {row.get("revision_stage") for row in runner.read_results(path)} if path.exists() else set()


def original_api_row(history: list[dict[str, Any]], record_id: str) -> dict[str, Any]:
    rows = [row for row in history if row.get("stable_id") == record_id and row.get("api_call_performed") is not False and row.get("http_status") == 200]
    if not rows:
        raise RuntimeError(f"missing_original_api_row:{record_id}")
    return rows[0]


def append_lineage_annotations(artifact: Path, prompt: dict[str, Any], config: dict[str, Any],
                               pilot: dict[str, Any], sources: dict[str, dict[str, Any]]) -> int:
    results_path = artifact / "v3_1_results.jsonl"
    history, latest, ordinals = runner.load_history(results_path)
    if any(row.get("mode") == "lineage_annotation_revision_001" for row in history):
        return 0
    run_id = datetime.now(timezone.utc).strftime("v3.1-lineage-r001-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
    rows = []
    for record in pilot["records"]:
        record_id = record["stable_id"]
        original = original_api_row(history, record_id)
        current = latest[record_id]
        raw_concepts = original.get("raw_model_concepts") or original.get("raw_parsed_concepts") or original.get("parsed_concepts") or []
        audited = quality.audit_concepts(raw_concepts, sources[record_id]["source_caption"], sources[record_id].get("modality", ""))
        deterministic = record_id in runner.SELECTIVE_RETRY_IDS
        final_concepts = (current.get("parsed_concepts") or []) if deterministic else raw_concepts
        row = {
            **current,
            "run_id": run_id,
            "mode": "lineage_annotation_revision_001",
            "raw_response": original.get("raw_response", ""),
            "original_raw_response": original.get("raw_response", ""),
            "raw_model_concepts": raw_concepts,
            "retry_model_concepts": [],
            "final_concepts": final_concepts,
            "repair_or_retry_reason": current.get("parser_repairs") or (["legacy_deterministic_semantic_repair"] if deterministic else []),
            "output_source": "deterministic_repair" if deterministic else "raw_model",
            "original_run_id": original.get("run_id"),
            "retry_run_id": None,
            "audit_flags": audited["audit_flags"],
            "audit_details": audited["audit_details"],
            "parsed_concepts": final_concepts,
            "raw_parsed_concepts": raw_concepts,
            "semantic_violations": audited["audit_flags"],
            "quality_status": "LEGACY_DETERMINISTIC_REPAIR_AWAITING_RETRY" if deterministic else "PASS_LEGACY_AUDITED",
            "cache_status": "legacy_deterministic_repair_not_active" if deterministic else "active_v3_1_success",
            "api_call_performed": False,
            "api_attempt_kind": "lineage_annotation",
            "attempt_count": ordinals[record_id],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lineage_revision": 1,
        }
        rows.append(row)
        latest[record_id] = row
    with results_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    selected = [row["stable_id"] for row in pilot["records"]]
    state = runner.state_payload("AUDIT_SEMANTICS_REVISED", selected, latest, sources, prompt, config, results_path, "pilot_50", None, None)
    runner.atomic_write_json(artifact / "v3_1_resume_state.json", state)
    return len(rows)


def write_codex_review(artifact: Path, revision: int, pilot: dict[str, Any],
                       latest: dict[str, dict[str, Any]], sources: dict[str, dict[str, Any]]) -> Path:
    old_path = artifact / "v3_1_pilot_human_audit_50.csv"
    old = {}
    if old_path.exists():
        with old_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                old[row.get("stable_id") or row.get("\ufeffstable_id")] = row
    path = artifact / f"v3_1_codex_semantic_review_50_revision_{revision:03d}.csv"
    fields = [
        "stable_id", "modality", "raw_model_concepts", "final_concepts", "output_source",
        "repair_or_retry_reason", "audit_flags", "quality_status", "codex_caption_supported",
        "codex_negative_check", "codex_contradiction_check", "codex_background_check",
        "codex_important_positive_omitted", "codex_notes", "user_verified",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in pilot["records"]:
            record_id = record["stable_id"]
            row = latest[record_id]
            prior = old.get(record_id, {})
            writer.writerow({
                "stable_id": record_id,
                "modality": sources[record_id].get("modality"),
                "raw_model_concepts": json.dumps(row.get("raw_model_concepts") or [], ensure_ascii=False),
                "final_concepts": json.dumps(row.get("final_concepts") or [], ensure_ascii=False),
                "output_source": row.get("output_source"),
                "repair_or_retry_reason": "|".join(row.get("repair_or_retry_reason") or []),
                "audit_flags": "|".join(row.get("audit_flags") or []),
                "quality_status": row.get("quality_status"),
                "codex_caption_supported": prior.get("caption_supported", "not_reassessed"),
                "codex_negative_check": prior.get("negative_removed", "not_reassessed"),
                "codex_contradiction_check": prior.get("contradiction_resolved", "not_reassessed"),
                "codex_background_check": prior.get("background_enumeration_absent", "not_reassessed"),
                "codex_important_positive_omitted": prior.get("important_positive_omitted", "not_reassessed"),
                "codex_notes": prior.get("reviewer_notes", "") or "Machine-assisted semantic audit; not user verified.",
                "user_verified": "no",
            })
        handle.flush()
        os.fsync(handle.fileno())
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pre_retry", "post_retry", "post_retry_reaudit"), required=True)
    parser.add_argument("--artifact-root", default=str(ARTIFACT))
    args = parser.parse_args()
    artifact = Path(args.artifact_root)
    prompt, _, config, pilot = runner.load_contract(artifact)
    sources, _ = runner.load_sources(artifact, config)
    events_path = artifact / "v3_1_audit_semantics_revision_events.jsonl"
    revisions_path = artifact / "v3_1_pilot_acceptance_revisions.jsonl"
    if args.stage in stages(events_path):
        print(json.dumps({"status": "ALREADY_APPLIED", "stage": args.stage, "api_called": False}))
        return
    annotations = append_lineage_annotations(artifact, prompt, config, pilot, sources) if args.stage == "pre_retry" else 0
    history, latest, _ = runner.load_history(artifact / "v3_1_results.jsonl")
    quality_reclassifications = 0
    if args.stage == "post_retry_reaudit":
        results_path = artifact / "v3_1_results.jsonl"
        run_id = datetime.now(timezone.utc).strftime("v3.1-quality-r003-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
        revisions = []
        for record_id in runner.SELECTIVE_RETRY_IDS:
            row = latest[record_id]
            if row.get("output_source") != "retry_model" or row.get("quality_status") != "PASS":
                continue
            evaluated = quality.evaluate_response(
                row.get("raw_response", ""), sources[record_id]["source_caption"],
                sources[record_id].get("modality", ""), row.get("finish_reason"),
            )
            if not evaluated["retry_reasons"]:
                continue
            revised = {
                **row,
                "run_id": run_id,
                "mode": "quality_reclassification_revision_003",
                "retry_model_concepts": evaluated["raw_model_concepts"],
                "final_concepts": [],
                "parsed_concepts": [],
                "audit_flags": evaluated["audit_flags"],
                "audit_details": evaluated["audit_details"],
                "semantic_violations": evaluated["audit_flags"],
                "quality_status": "REJECTED_AFTER_RETRY",
                "cache_status": "rejected_after_retry",
                "error_type": "quality_rejected_after_retry",
                "api_call_performed": False,
                "api_attempt_kind": "quality_reclassification",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            revisions.append(revised)
            latest[record_id] = revised
        if revisions:
            with results_path.open("a", encoding="utf-8") as handle:
                for row in revisions:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            quality_reclassifications = len(revisions)
            selected = list(runner.SELECTIVE_RETRY_IDS)
            state = runner.state_payload("SELECTIVE_RETRY_REAUDITED", selected, latest, sources, prompt, config, results_path, "selective_retry_5", None, None)
            runner.atomic_write_json(artifact / "v3_1_resume_state.json", state)
            history, latest, _ = runner.load_history(results_path)
    retry_terminal = {record_id: runner.selective_retry_terminal(latest.get(record_id)) for record_id in runner.SELECTIVE_RETRY_IDS}
    revision = {"pre_retry": 1, "post_retry": 2, "post_retry_reaudit": 3}[args.stage]
    review_path = write_codex_review(artifact, revision, pilot, latest, sources)
    technical = "PASS" if args.stage in {"post_retry", "post_retry_reaudit"} and all(retry_terminal.values()) else "PENDING_SELECTIVE_RETRY_5"
    acceptance = {
        "revision": revision,
        "revision_stage": args.stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "technical_acceptance_status": technical,
        "human_review": {"status": "NOT_VERIFIED_BY_USER", "verified_rows": 0},
        "codex_semantic_review": {"status": "COMPLETED", "rows": 50, "path": str(review_path)},
        "formal_batch_0_authorized": True,
        "remaining_cycle_batches_authorized": False,
        "selective_retry_ids": list(runner.SELECTIVE_RETRY_IDS),
        "selective_retry_terminal": retry_terminal,
        "selective_retry_passed": [record_id for record_id in runner.SELECTIVE_RETRY_IDS if latest.get(record_id, {}).get("quality_status") == "PASS" and latest.get(record_id, {}).get("output_source") == "retry_model"],
        "selective_retry_rejected": [record_id for record_id in runner.SELECTIVE_RETRY_IDS if latest.get(record_id, {}).get("quality_status") == "REJECTED_AFTER_RETRY"],
        "prompt_hash": prompt["prompt_hash"],
        "schema_hash": prompt["schema_hash"],
        "run_config_hash": config["run_config_hash"],
        "original_artifacts_preserved": {name: {"sha256": sha(artifact / name), "bytes": (artifact / name).stat().st_size} for name in ORIGINAL_FILES if (artifact / name).exists()},
        "results_hash": runner.sha_file(artifact / "v3_1_results.jsonl"),
        "resume_state_hash": runner.sha_file(artifact / "v3_1_resume_state.json"),
    }
    append_jsonl(revisions_path, acceptance)
    runner.atomic_write_json(artifact / f"v3_1_pilot_acceptance_revision_{revision:03d}.json", acceptance)
    event = {
        "revision_stage": args.stage,
        "revision": revision,
        "timestamp": acceptance["timestamp"],
        "reason": "Correct Codex-authored review semantics and preserve raw/retry/final provenance without modifying prompt v3.1.",
        "lineage_annotation_rows_appended": annotations,
        "quality_reclassification_rows_appended": quality_reclassifications,
        "acceptance_revision_path": str(artifact / f"v3_1_pilot_acceptance_revision_{revision:03d}.json"),
        "codex_review_path": str(review_path),
        "human_review_status": "NOT_VERIFIED_BY_USER",
        "api_called": False,
    }
    append_jsonl(events_path, event)
    print(json.dumps({"status": technical, "stage": args.stage, "annotations": annotations, "quality_reclassifications": quality_reclassifications, "human_review": "NOT_VERIFIED_BY_USER", "api_called": False}))


if __name__ == "__main__":
    main()
