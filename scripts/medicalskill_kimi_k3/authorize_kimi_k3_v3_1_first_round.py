#!/usr/bin/env python3
"""Append authorization for unattended frozen Batch 00-17 execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/root/MedAgentCL_v4")
SCRIPTS = ROOT / "scripts/medicalskill_kimi_k3"
sys.path.insert(0, str(SCRIPTS))

import v3_1_first_round_policy as policy  # noqa: E402


REVISIONS = policy.ARTIFACT / "v3_1_pilot_acceptance_revisions.jsonl"
REVISION_FILE = (
    policy.ARTIFACT / "v3_1_pilot_acceptance_revision_005.json"
)
POLICY_FILE = (
    policy.ARTIFACT / "first_round_runtime_policy_revision_002.json"
)


def main() -> None:
    policy.load_contract_files()
    stable_ids = policy.load_stable_ids()
    history = policy.read_jsonl(policy.RESULTS_PATH)
    revisions = policy.read_jsonl(REVISIONS)
    if not revisions:
        raise RuntimeError("missing_acceptance_revisions")
    if revisions[-1].get("revision") == 5:
        if (
            revisions[-1].get("authorized_formal_batch_indices")
            == list(range(18))
            and revisions[-1].get(
                "batch_18_and_later_authorized"
            ) is False
        ):
            print(json.dumps({
                "status": "ALREADY_AUTHORIZED",
                "revision": 5,
            }))
            return
        raise RuntimeError("conflicting_revision_005")
    if revisions[-1].get("revision") != 4:
        raise RuntimeError("revision_004_must_be_latest")

    classification = policy.classify_records(
        stable_ids, history, policy.load_permanent_registry()
    )
    if classification["counts"] != {
        "planned": 18000,
        "success": 1996,
        "permanent_failed": 0,
        "pending": 4,
        "not_started": 16000,
    }:
        raise RuntimeError(
            f"unexpected_pre_authorization_state:"
            f"{classification['counts']}"
        )
    used = policy.cycle_calls_used(history)
    if used != 2296:
        raise RuntimeError(
            f"unexpected_cycle_call_count:{used}"
        )

    runtime_policy = {
        "policy_revision": 2,
        "scope": "frozen_first_round_only",
        "batch_indices": list(range(18)),
        "stable_record_count": 18000,
        "stable_record_ids_sha256": policy.sha256(
            policy.ARTIFACT / "concept_35k_stable_ids.txt"
        ),
        "next_batch_authorized": False,
        "cycle_api_hard_limit": 18000,
        "cycle_api_calls_at_authorization": used,
        "cycle_api_calls_remaining_at_authorization": 18000 - used,
        "attempt_success_rate_is_monitoring_only": True,
        "minimum_unique_record_completion_rate": 0.99,
        "maximum_global_permanent_failure_rate": 0.01,
        "minimum_contract_valid_rate_among_http_200": 0.995,
        "http_429_immediate_retry": False,
        "http_429_burst_rules": {
            "consecutive_attempts": 3,
            "recent_window_attempts": 20,
            "recent_window_429": 5,
            "multiple_ids_short_window": True,
        },
        "cooldown_seconds": policy.COOLDOWN_SECONDS,
        "recovery_probe_records": 1,
        "same_window_max_attempts": 3,
        "lifetime_max_attempts": 6,
        "contract_failure_extra_attempts": 1,
        "temperature": 1.0,
        "top_p_sent": False,
        "reasoning_effort": "none",
        "request_model_id": "k3",
        "expected_routed_model": "kimi-k2.6",
        "returned_model_must_remain": "k3",
        "base_url": "https://api.kimi.com/coding/v1",
        "prompt_hash": policy.EXPECTED_HASHES["prompt_hash"],
        "schema_hash": policy.EXPECTED_HASHES["schema_hash"],
        "run_config_hash": policy.EXPECTED_HASHES[
            "run_config_hash"
        ],
        "source_hash": policy.EXPECTED_HASHES[
            "input_manifest_sha256"
        ],
        "existing_success_cache_grandfathered": True,
        "existing_success_cache_recalled": False,
        "historical_batch01": {
            "success": 996,
            "pending": 4,
            "api_attempts": 1008,
            "http_429": 12,
        },
        "created_at": policy.iso_now(),
    }
    policy.atomic_write_json(POLICY_FILE, runtime_policy)
    revision = {
        "revision": 5,
        "revision_stage": "unattended_first_round_00_17",
        "timestamp": policy.iso_now(),
        "technical_acceptance_status": "PASS",
        "human_review": {
            "status": "NOT_VERIFIED_BY_USER",
            "verified_rows": 0,
        },
        "authorized_formal_batch_indices": list(range(18)),
        "batch_18_and_later_authorized": False,
        "first_round_record_count": 18000,
        "runtime_policy_revision": 2,
        "runtime_policy_path": str(POLICY_FILE),
        "runtime_policy_sha256": policy.sha256(POLICY_FILE),
        "temperature": 1.0,
        "prompt_hash": policy.EXPECTED_HASHES["prompt_hash"],
        "schema_hash": policy.EXPECTED_HASHES["schema_hash"],
        "run_config_hash": policy.EXPECTED_HASHES[
            "run_config_hash"
        ],
        "stable_id_set_sha256": policy.EXPECTED_HASHES[
            "stable_id_set_sha256"
        ],
        "input_manifest_sha256": policy.EXPECTED_HASHES[
            "input_manifest_sha256"
        ],
        "cycle_api_hard_limit": 18000,
        "cycle_api_calls_used_at_authorization": used,
        "cycle_api_calls_remaining_at_authorization": 18000 - used,
        "authorization_source": "explicit_user_instruction",
    }
    policy.atomic_write_json(REVISION_FILE, revision)
    policy.append_jsonl(REVISIONS, revision)
    print(json.dumps({
        "status": "AUTHORIZED",
        "revision": 5,
        "authorized_batches": list(range(18)),
        "batch_18_and_later_authorized": False,
        "cycle_api_calls_used": used,
    }))


if __name__ == "__main__":
    main()
