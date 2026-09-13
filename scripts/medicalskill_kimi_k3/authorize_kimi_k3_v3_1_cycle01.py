#!/usr/bin/env python3
"""Append the user-approved v3.1 Cycle 01 Batch 01-16 authorization."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_kimi_k3_v3_1_k2_6_routing"
REVISIONS = ARTIFACT / "v3_1_pilot_acceptance_revisions.jsonl"
REVISION_FILE = ARTIFACT / "v3_1_pilot_acceptance_revision_004.json"
BATCH0_REPORT = (
    ARTIFACT
    / "batch_reports/cycle_01_batch_00_audit_revision_001.json"
)
EXPECTED = {
    "prompt_hash": "fb63d4b1963d5b9c93188ed65b7ec5677eeaf7ecb8e38b86bb73e640f516ae68",
    "schema_hash": "14830fcd9b334ac7ce24c17cd0704a373ccec247da03bf1435dad139a8d38aea",
    "run_config_hash": "382703cd787c9c49cce0ccdc2cda71615ccbb16f35820cc4a9ad290243d230a1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"invalid_acceptance_revision_jsonl_line:{line_number}"
                    ) from error
    return rows


def atomic_write_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)


def append_jsonl(path: Path, value: dict) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def validate_batch0() -> tuple[dict, dict, str, str]:
    config = read_json(ARTIFACT / "run_config_v3_1.json")
    for key, expected in EXPECTED.items():
        actual = config.get(key) if key == "run_config_hash" else None
        if key != "run_config_hash":
            manifest = read_json(ARTIFACT / "prompt_v3_1_manifest.json")
            actual = manifest.get(key)
        if actual != expected:
            raise RuntimeError(f"frozen_contract_hash_mismatch:{key}")

    report = read_json(BATCH0_REPORT)
    checks = {
        "status": report.get("status") == "PASS",
        "selected": report.get("selected") == 1000,
        "accepted": report.get("final_accepted_records") == 1000,
        "rejected": report.get("final_rejected_records") == 0,
        "api_attempts": report.get("api_attempts") == 1288,
        "reasoning_tokens": report.get("usage", {}).get(
            "reasoning_token_usage"
        ) == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"batch0_preservation_check_failed:{','.join(failed)}")

    results_path = ARTIFACT / "v3_1_results.jsonl"
    resume_path = ARTIFACT / "v3_1_resume_state.json"
    results_hash = sha256(results_path)
    resume_hash = sha256(resume_path)
    resume = read_json(resume_path)
    if resume.get("results_hash") != results_hash:
        raise RuntimeError("resume_results_hash_mismatch")
    if report.get("results_hash") != results_hash:
        raise RuntimeError("batch0_report_results_hash_mismatch")
    return config, report, results_hash, resume_hash


def main() -> None:
    revisions = read_jsonl(REVISIONS)
    if not revisions:
        raise RuntimeError("missing_acceptance_revisions")
    latest = revisions[-1]
    if latest.get("revision") == 4:
        if (
            latest.get("authorized_formal_batch_indices") == list(range(17))
            and latest.get("batch_17_and_later_authorized") is False
            and latest.get("audit_policy_revision") == 1
        ):
            print(json.dumps({"status": "ALREADY_AUTHORIZED", "revision": 4}))
            return
        raise RuntimeError("conflicting_revision_004")
    if latest.get("revision") != 3:
        raise RuntimeError("revision_003_must_be_latest")
    if latest.get("technical_acceptance_status") != "PASS":
        raise RuntimeError("technical_acceptance_not_pass")
    if latest.get("human_review", {}).get("status") != "NOT_VERIFIED_BY_USER":
        raise RuntimeError("human_review_semantics_changed")

    config, report, results_hash, resume_hash = validate_batch0()
    revision = {
        "revision": 4,
        "revision_stage": "cycle_01_batches_01_16_authorization",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "technical_acceptance_status": "PASS",
        "human_review": {
            "status": "NOT_VERIFIED_BY_USER",
            "verified_rows": 0,
        },
        "codex_semantic_review": latest.get("codex_semantic_review"),
        "formal_batch_0_authorized": True,
        "authorized_formal_batch_indices": list(range(17)),
        "batch_17_and_later_authorized": False,
        "audit_policy_revision": 1,
        "semantic_qc_mode": "flag_only_no_retry_no_rewrite_no_reject",
        "technical_retry_only": True,
        "cycle_id": "cycle_01",
        "cycle_api_hard_limit": 18000,
        "cycle_api_calls_used_at_authorization": 1288,
        "cycle_api_calls_remaining_at_authorization": 16712,
        "authorized_new_record_batches": list(range(1, 17)),
        "authorized_new_record_count": 16000,
        "two_hour_rolling_call_limit": 6600,
        "prompt_hash": EXPECTED["prompt_hash"],
        "schema_hash": EXPECTED["schema_hash"],
        "run_config_hash": EXPECTED["run_config_hash"],
        "stable_id_set_sha256": config["stable_id_set_sha256"],
        "batch0_preserved": {
            "selected": report["selected"],
            "accepted": report["final_accepted_records"],
            "rejected": report["final_rejected_records"],
            "api_attempts": report["api_attempts"],
            "report_path": str(BATCH0_REPORT),
            "report_sha256": sha256(BATCH0_REPORT),
        },
        "results_hash_at_authorization": results_hash,
        "resume_state_hash_at_authorization": resume_hash,
        "authorization_source": "explicit_user_instruction",
    }
    atomic_write_json(REVISION_FILE, revision)
    append_jsonl(REVISIONS, revision)
    print(
        json.dumps(
            {
                "status": "AUTHORIZED",
                "revision": 4,
                "authorized_batches": list(range(1, 17)),
                "batch_17_and_later_authorized": False,
                "revision_file": str(REVISION_FILE),
            }
        )
    )


if __name__ == "__main__":
    main()
