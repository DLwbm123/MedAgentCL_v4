#!/usr/bin/env python3
"""Finalize reports after the corrected-endpoint canary is rejected by its fixed temperature contract."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_kimi_k3_pilot"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    artifact = ARTIFACT
    attempts = read_jsonl(artifact / "pilot_results.jsonl")
    if len(attempts) != 2:
        raise RuntimeError(f"Expected exactly two preserved formal attempt rows, found {len(attempts)}")
    old, current = attempts
    if "401" not in str(old.get("error_category")) or "Invalid Authentication" not in str(old.get("error_category")):
        raise RuntimeError("Historical 401 row is missing or changed")
    if current.get("endpoint_hostname") != "api.kimi.com" or current.get("endpoint_base_path") != "/coding/v1":
        raise RuntimeError("Corrected endpoint audit is missing")
    if "400" not in str(current.get("error_category")) or "only 1 is allowed" not in str(current.get("error_category")):
        raise RuntimeError("Expected fixed-temperature contract rejection is missing")
    selection = json.loads((artifact / "pilot_selection_manifest.json").read_text(encoding="utf-8"))
    prompt = json.loads((artifact / "prompt_manifest.json").read_text(encoding="utf-8"))

    runtime = {
        "status": "FROZEN_FOR_CANARY",
        "effective_base_url": "https://api.kimi.com/coding/v1",
        "scheme": "https",
        "hostname": "api.kimi.com",
        "base_path": "/coding/v1",
        "chat_completions_url": "https://api.kimi.com/coding/v1/chat/completions",
        "base_url_source": current["base_url_source"],
        "credential_present": True,
        "credential_source": current["credential_source"],
        "model": "kimi-k3",
        "reasoning_effort": "low",
        "temperature_requested": 0,
        "max_completion_tokens": 512,
        "structured_output_mode": "json_schema_strict",
        "prompt_version": prompt["prompt_version"],
        "prompt_hash": prompt["prompt_hash"],
        "schema_version": prompt["schema_version"],
        "schema_hash": prompt["schema_hash"],
        "pilot_manifest_count": selection["total"],
        "pilot_unique_stable_record_ids": len(set(selection["stable_record_ids"])),
    }
    write_json(artifact / "runtime_api_contract.json", runtime)

    chain = {
        "status": "PRESERVED_BLOCKED_AT_FORMAL_CANARY",
        "events": [
            {
                "sequence": 1,
                "purpose": "historical_formal_canary",
                "stable_record_id": "caption_ct_train_005626",
                "endpoint": "https://api.moonshot.cn/v1",
                "http_status": 401,
                "error_type": "invalid_authentication_error",
                "error_message": "Invalid Authentication",
                "source": "pilot_results.jsonl preserved first row",
                "success_cache": False,
            },
            {
                "sequence": 2,
                "purpose": "connectivity_only",
                "endpoint": "https://api.kimi.com/coding/v1",
                "status": "API_TEST_SUCCESS",
                "returned_model": "kimi-k3",
                "finish_reason": "length",
                "total_tokens": 123,
                "source": "user-confirmed manual test",
                "success_cache": False,
            },
            {
                "sequence": 3,
                "purpose": "formal_structured_output_canary",
                "stable_record_id": current["stable_record_id"],
                "run_id": current["run_id"],
                "attempt_id": current["attempt_id"],
                "timestamp": current["created_timestamp"],
                "endpoint": current["endpoint_base_url"],
                "hostname": current["endpoint_hostname"],
                "base_path": current["endpoint_base_path"],
                "http_status": 400,
                "error_type": "invalid_request_error",
                "error_message": "invalid temperature: only 1 is allowed for this model",
                "model_requested": current["model_requested"],
                "reasoning_effort": current["reasoning_effort"],
                "max_completion_tokens": current["max_completion_tokens"],
                "contract_status": current["contract_status"],
                "token_usage": current["token_usage"],
                "success_cache": False,
            },
        ],
        "historical_rows_deleted_or_overwritten": False,
        "active_success_cache_count": 0,
    }
    write_json(artifact / "authentication_audit_chain.json", chain)

    write_json(
        artifact / "pilot_api_summary.json",
        {
            **runtime,
            "status": "BLOCKED_CANARY_PARAMETER_400",
            "historical_formal_attempts": 2,
            "manual_connectivity_tests": 1,
            "current_endpoint_formal_api_requests": 1,
            "successful_formal_api_requests": 0,
            "active_success_cache": 0,
            "automatic_retries": 0,
            "error_distribution": {
                "historical_http_401_invalid_authentication": 1,
                "current_http_400_invalid_temperature": 1,
            },
            "finish_reason_distribution": {},
            "schema_failures": 0,
            "json_parse_failures": 0,
            "empty_concepts": 0,
            "concept_count_over_8": 0,
            "prompt_source_schema_hash_mismatches": 0,
            "cache_hits": 0,
            "normal_success_duplicate_calls": 0,
            "full_pilot_started": False,
            "full_generation_started": False,
        },
    )
    write_json(
        artifact / "pilot_cache_index.json",
        {
            "status": "BLOCKED_NO_ACTIVE_SUCCESS",
            "attempt_rows": 2,
            "active_success_count": 0,
            "records": {
                "caption_ct_train_005626": {
                    "active_attempt_id": None,
                    "attempt_ids": [current["attempt_id"]],
                    "historical_pre_attempt_id_rows": 1,
                    "status": "no_contract_compliant_success",
                }
            },
        },
    )
    write_json(
        artifact / "pilot_auto_audit.json",
        {
            "status": "NOT_RUN_CANARY_PARAMETER_BLOCKED",
            "program_verified": {
                "frozen_inputs": 70_356,
                "pilot_manifest_records": 2_000,
                "pilot_unique_stable_record_ids": 2_000,
                "formal_canary_calls_after_endpoint_fix": 1,
                "formal_canary_success": 0,
                "remaining_pilot_calls": 0,
                "failure": "HTTP 400 invalid temperature: only 1 is allowed for this model",
            },
            "strict_structured_output_contract": "NOT_REACHED_API_PARAMETER_REJECTED",
            "machine_quality_gates": "NOT_EVALUABLE",
            "human_semantic_gates": "NOT_READY_NO_KIMI_OUTPUTS",
        },
    )
    write_json(
        artifact / "pilot_cost_report.json",
        {
            "status": "NO_SUCCESSFUL_FORMAL_COMPLETION",
            "formal_attempt_rows": 2,
            "manual_connectivity_total_tokens_not_in_pilot_cache": 123,
            "formal_prompt_tokens": 0,
            "formal_cached_prompt_tokens": 0,
            "formal_completion_tokens": 0,
            "formal_total_tokens": 0,
            "formal_token_statistics": {"mean": 0, "median": 0, "p95": 0, "max": 0},
            "formal_latency_seconds": {
                "mean": current["latency_seconds"],
                "median": current["latency_seconds"],
                "p95": current["latency_seconds"],
                "max": current["latency_seconds"],
            },
            "estimated_cost": None,
            "reason": "Both formal attempts were rejected before billable usage was reported; no unverified Kimi Code pricing was invented.",
        },
    )

    human_path = artifact / "pilot_human_audit_500.csv"
    with human_path.open("r", encoding="utf-8-sig", newline="") as handle:
        human_rows = list(csv.DictReader(handle))
    if len(human_rows) != 500:
        raise RuntimeError("The preserved stratified human audit does not contain 500 rows")
    manual_fields = [
        "caption_supported", "negation_correct", "uncertainty_correct", "generic_or_artifact",
        "semantic_duplicate", "preferred_output", "reviewer_notes",
    ]
    if any(any(row.get(field, "").strip() for field in manual_fields) for row in human_rows):
        raise RuntimeError("Human judgment fields must remain blank")

    (artifact / "pilot_quality_report.md").write_text(
        """# Kimi K3 MedicalSkill-CL Pilot quality report

Status: **BLOCKED AT FORMAL STRUCTURED-OUTPUT CANARY**

The endpoint and credential route are now correct: `https://api.kimi.com/coding/v1`, model `kimi-k3`. The frozen formal canary used `reasoning_effort=low`, temperature 0, strict JSON Schema, and `max_completion_tokens=512` for `caption_ct_train_005626`.

The API returned HTTP 400: `invalid temperature: only 1 is allowed for this model`. The request was rejected before model output, finish reason, JSON, schema, concepts, or token usage could be evaluated. Per the frozen stop policy for parameter 400 responses, no automatic parameter change or retry was made.

The earlier `.cn` HTTP 401 record and the user-confirmed connectivity-only test are preserved in `authentication_audit_chain.json`. The latter is not Pilot cache because it ended with `finish_reason=length` and did not validate strict structured output.

Pilot quality metrics and the 500-row semantic audit remain not ready: there are no contract-compliant Kimi outputs. The remaining 1,999 Pilot records and all non-Pilot records were not called.
""",
        encoding="utf-8",
    )
    write_json(
        artifact / "resume_state.json",
        {
            "status": "BLOCKED_CANARY_PARAMETER_400",
            "selected_unique_limit": 2_000,
            "attempt_rows": 2,
            "active_success_cache_count": 0,
            "current_endpoint_formal_calls": 1,
            "remaining_pilot_started": False,
            "full_generation_started": False,
            "next_action": "Await explicit user direction on the temperature contract; do not modify it or rerun automatically.",
        },
    )
    (artifact / "commands.txt").write_text(
        "# No-API configuration audit\n"
        "scripts/medicalskill_kimi_k3/run_kimi_k3_secure.sh --config-audit\n\n"
        "# Formal canary command used once after endpoint correction\n"
        "scripts/medicalskill_kimi_k3/run_kimi_k3_secure.sh --canary\n\n"
        "# BLOCKED: do not run the remaining Pilot or full generation until explicitly directed.\n",
        encoding="utf-8",
    )
    (artifact / "git_state_after.txt").write_text(git_state(), encoding="utf-8")
    targets = sorted(path for path in artifact.iterdir() if path.is_file() and path.name != "artifact_hashes.json")
    write_json(artifact / "artifact_hashes.json", {path.name: sha_file(path) for path in targets})
    print(json.dumps({"status": "BLOCKED_CANARY_PARAMETER_400", "formal_calls": 2, "success": 0, "remaining_pilot_started": False}))


if __name__ == "__main__":
    main()
