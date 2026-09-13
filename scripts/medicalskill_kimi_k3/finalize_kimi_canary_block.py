#!/usr/bin/env python3
"""Finalize required artifacts when the single Kimi canary blocks the pilot."""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_kimi_k3_pilot"


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


def stratified_500(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in records:
        groups[(row["pilot_stratum"], row["modality"], row["source_dataset"])].append(row)
    selected = []
    fractions = []
    for key, rows in groups.items():
        exact = 500 * len(rows) / len(records)
        count = int(exact)
        ranked = sorted(rows, key=lambda row: hashlib.sha256(f"42|blocked-human|{row['stable_record_id']}".encode()).hexdigest())
        selected.extend(ranked[:count])
        fractions.append((exact - count, key, ranked[count:]))
    needed = 500 - len(selected)
    for _, _, rows in sorted(fractions, reverse=True):
        if needed and rows:
            selected.append(rows[0])
            needed -= 1
    if len(selected) != 500:
        raise RuntimeError(f"Blocked audit sample size {len(selected)} != 500")
    return selected


def main() -> None:
    artifact = ARTIFACT
    selection = json.loads((artifact / "pilot_selection_manifest.json").read_text(encoding="utf-8"))
    frozen = {row["stable_record_id"]: row for row in read_jsonl(artifact / "input_freeze_manifest.jsonl")}
    results = list(read_jsonl(artifact / "pilot_results.jsonl"))
    if len(results) != 1 or results[0].get("status") != "contract_or_api_rejected":
        raise RuntimeError("This finalizer is only valid for the one-call canary rejection state")
    result = results[0]
    if "401" not in str(result.get("error_category")) or "Invalid Authentication" not in str(result.get("error_category")):
        raise RuntimeError("Unexpected canary rejection; refusing to generalize the report")

    write_json(
        artifact / "pilot_api_summary.json",
        {
            "status": "BLOCKED_CANARY_AUTHENTICATION",
            "pilot_selected_unique_records": 2_000,
            "api_calls": 1,
            "success": 0,
            "cache_success": 0,
            "error_distribution": {"http_401_invalid_authentication": 1},
            "retry_count": 0,
            "model_requested": "kimi-k3",
            "model_returned": None,
            "reasoning_effort": "low",
            "token_usage": {"prompt": 0, "cached": 0, "completion": 0, "total": 0},
            "credential_or_headers_recorded": False,
            "full_pilot_started": False,
        },
    )
    write_json(
        artifact / "pilot_cache_index.json",
        {
            "status": "BLOCKED_NO_SUCCESS_CACHE",
            "count": 1,
            "success_count": 0,
            "records": {
                result["stable_record_id"]: {
                    "status": result["status"],
                    "source_caption_sha256": result["source_caption_sha256"],
                    "prompt_hash": result["prompt_hash"],
                    "created_timestamp": result["created_timestamp"],
                }
            },
        },
    )
    write_json(
        artifact / "pilot_auto_audit.json",
        {
            "status": "NOT_RUN_CANARY_BLOCKED",
            "program_verified": {
                "frozen_inputs": 70_356,
                "pilot_selected_unique_records": 2_000,
                "canary_calls": 1,
                "canary_success": 0,
                "full_pilot_calls": 0,
                "failure": "HTTP 401 Invalid Authentication",
            },
            "machine_gates": "NOT_EVALUABLE",
            "requires_human_confirmation": "NOT_READY_NO_KIMI_OUTPUTS",
        },
    )
    write_json(
        artifact / "pilot_cost_report.json",
        {
            "status": "NO_SUCCESSFUL_BILLABLE_COMPLETION_OBSERVED",
            "api_calls": 1,
            "successful_completions": 0,
            "token_usage": {"prompt": 0, "cached": 0, "completion": 0, "total": 0},
            "estimated_cost": None,
            "reason": "The canary was rejected before token usage was reported; no unverified kimi-k3 pricing was invented.",
        },
    )

    audit_rows = stratified_500(selection["records"])
    fields = [
        "stable_record_id", "modality", "source_dataset", "source_caption", "qwen_v3_concepts",
        "kimi_k3_concepts", "pilot_stratum", "automatic_risk_flags", "caption_supported",
        "negation_correct", "uncertainty_correct", "generic_or_artifact", "semantic_duplicate",
        "preferred_output", "reviewer_notes",
    ]
    with (artifact / "pilot_human_audit_500.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in audit_rows:
            source = frozen[record["stable_record_id"]]
            writer.writerow(
                {
                    "stable_record_id": record["stable_record_id"],
                    "modality": record["modality"],
                    "source_dataset": record["source_dataset"],
                    "source_caption": source["source_caption"],
                    "qwen_v3_concepts": json.dumps(record["qwen_v3_concepts"], ensure_ascii=False),
                    "kimi_k3_concepts": "",
                    "pilot_stratum": record["pilot_stratum"],
                    "automatic_risk_flags": "canary_blocked_no_kimi_output",
                    "caption_supported": "",
                    "negation_correct": "",
                    "uncertainty_correct": "",
                    "generic_or_artifact": "",
                    "semantic_duplicate": "",
                    "preferred_output": "",
                    "reviewer_notes": "",
                }
            )

    (artifact / "pilot_quality_report.md").write_text(
        """# Kimi K3 pilot quality report

Status: **BLOCKED AT CANARY**

The frozen 70,356-record input and mutually exclusive 2,000-record selection both passed deterministic validation. The first selected record was sent exactly once with model `kimi-k3`, `reasoning_effort=low`, temperature 0, max completion tokens 128, and the frozen strict JSON Schema.

The service returned HTTP 401 `Invalid Authentication`. The error is non-retryable and occurred before any completion/token usage. The remaining 1,999 pilot records and the other 68,356 records were not called. No model, endpoint, reasoning mode, prompt, or response format was substituted.

Automatic quality gates and semantic human-review gates are not evaluable without Kimi outputs. `pilot_human_audit_500.csv` contains the requested stratified rows and blank review fields, but its Kimi output column is intentionally empty and it is not ready for review.

Decision: do not enter the full run. The credential or its authorization for the configured official endpoint/model must be corrected by the user, after which the same canary command can safely resume from this state.
""",
        encoding="utf-8",
    )
    write_json(
        artifact / "resume_state.json",
        {
            "status": "BLOCKED_CANARY_AUTHENTICATION",
            "selected_unique_limit": 2_000,
            "api_calls": 1,
            "result_unique_count": 1,
            "success_count": 0,
            "full_pilot_started": False,
            "full_generation_started": False,
            "next_action": "User corrects the credential/authorization, then reruns only the unchanged secure canary command.",
        },
    )
    (artifact / "git_state_after.txt").write_text(git_state(), encoding="utf-8")
    targets = sorted(path for path in artifact.iterdir() if path.is_file() and path.name != "artifact_hashes.json")
    write_json(artifact / "artifact_hashes.json", {path.name: sha_file(path) for path in targets})
    print(json.dumps({"status": "BLOCKED_CANARY_AUTHENTICATION", "api_calls": 1, "success": 0, "pilot_started": False}))


if __name__ == "__main__":
    main()
