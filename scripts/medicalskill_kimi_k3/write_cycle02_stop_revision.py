#!/usr/bin/env python3
"""Write an append-only audit revision for the Cycle 02 quota stop."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = (
    ROOT
    / "artifacts/medicalskill_cl_kimi_k3_v3_1_k2_6_routing"
)
CYCLE_DIR = ARTIFACT / "cycle_02_35k_closure"
RESULTS = ARTIFACT / "v3_1_results.jsonl"
EVENTS = CYCLE_DIR / "cycle_02_events.jsonl"
SUMMARY = CYCLE_DIR / "cycle_02_summary.json"
REPORT = (
    CYCLE_DIR
    / "batch_reports/cycle_02_batch_17_closure.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


rows = [
    json.loads(line)
    for line in RESULTS.open(encoding="utf-8")
    if line.strip()
]
cycle_rows = [
    row
    for row in rows
    if row.get("cycle_id") == "cycle_02"
    and row.get("api_call_performed") is not False
]
events = [
    json.loads(line)
    for line in EVENTS.open(encoding="utf-8")
    if line.strip()
]
summary = json.loads(SUMMARY.read_text())
report = json.loads(REPORT.read_text())

assert len(cycle_rows) == 8
assert {row.get("http_status") for row in cycle_rows} == {403}
assert all(row.get("quota_exhausted") for row in cycle_rows)
assert {
    row.get("provider_error_type") for row in cycle_rows
} == {"access_terminated_error"}
assert all(
    "billing cycle"
    in str(row.get("provider_error_message") or "").lower()
    for row in cycle_rows
)
reservations = sum(
    event.get("event_type") == "API_BUDGET_RESERVED"
    for event in events
)
assert reservations == 8

payload = {
    "revision": "cycle_02_hard_stop_revision_001",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "api_called_by_revision": False,
    "historical_summary_stop_reason": summary.get("stop_reason"),
    "corrected_stop_reason": "BILLING_CYCLE_QUOTA",
    "provider_error_type": "access_terminated_error",
    "provider_message_category": "billing_cycle_usage_limit",
    "http_status": 403,
    "cycle_api_result_rows": len(cycle_rows),
    "cycle_budget_reservations": reservations,
    "remaining_call_budget": 18_000 - reservations,
    "batch_17_success": report["success"],
    "batch_17_target": report["planned_unique_records"],
    "batch_17_remaining": (
        report["planned_unique_records"] - report["success"]
    ),
    "reasoning_token_total": sum(
        int(row.get("reasoning_token_usage") or 0)
        for row in cycle_rows
    ),
    "successful_cache_rows_overwritten": 0,
    "results_sha256": sha256(RESULTS),
    "events_sha256": sha256(EVENTS),
    "summary_sha256": sha256(SUMMARY),
    "batch_report_sha256": sha256(REPORT),
    "safe_resume_command": (
        "scripts/medicalskill_kimi_k3/"
        "run_kimi_k3_v3_1_cycle02_35k_secure.sh "
        "--run-cycle-02-35k --confirm-cycle-02-35k "
        "--concurrency 8 --requests-per-second 2.0 "
        "--timeout 120 --max-retries 2"
    ),
}
json_path = CYCLE_DIR / "cycle_02_hard_stop_revision_001.json"
md_path = CYCLE_DIR / "cycle_02_hard_stop_revision_001.md"
atomic_text(
    json_path,
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
)
atomic_text(
    md_path,
    "# Cycle 02 hard-stop revision 001\n\n"
    "Status: **BILLING_CYCLE_QUOTA**\n\n"
    "- HTTP 403 attempts: 8\n"
    "- Budget reservations: 8/18000\n"
    "- Remaining call budget: 17992\n"
    "- Batch 17 success remains: 725/1000\n"
    "- Successful cache rows overwritten: 0\n"
    "- API calls made by this revision: 0\n"
    f"- Results SHA-256: `{payload['results_sha256']}`\n",
)
print(json.dumps(payload, ensure_ascii=False))
