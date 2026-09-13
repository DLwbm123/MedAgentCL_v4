#!/usr/bin/env python3
"""Append the user-approved K2.6 temperature=0.6 runtime contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/root/MedAgentCL_v4")
SCRIPTS = ROOT / "scripts/medicalskill_kimi_k3"
sys.path.insert(0, str(SCRIPTS))

import v3_1_first_round_policy as policy  # noqa: E402


REVISIONS = (
    policy.ARTIFACT / "v3_1_pilot_acceptance_revisions.jsonl"
)
REVISION_FILE = (
    policy.ARTIFACT
    / "v3_1_pilot_acceptance_revision_006.json"
)
POLICY_FILE = (
    policy.ARTIFACT
    / "first_round_runtime_policy_revision_003.json"
)


def main() -> None:
    policy.load_contract_files()
    stable_ids = policy.load_stable_ids()
    history = policy.read_jsonl(policy.RESULTS_PATH)
    revisions = policy.read_jsonl(REVISIONS)
    if not revisions:
        raise RuntimeError("missing_acceptance_revisions")
    if revisions[-1].get("revision") == 6:
        if (
            revisions[-1].get("temperature") == 0.6
            and revisions[-1].get(
                "authorized_formal_batch_indices"
            ) == list(range(18))
            and revisions[-1].get(
                "batch_18_and_later_authorized"
            ) is False
        ):
            print(json.dumps({
                "status": "ALREADY_AUTHORIZED",
                "revision": 6,
            }))
            return
        raise RuntimeError("conflicting_revision_006")
    if revisions[-1].get("revision") != 5:
        raise RuntimeError("revision_005_must_be_latest")

    classification = policy.classify_records(
        stable_ids,
        history,
        policy.load_permanent_registry(),
    )
    expected_counts = {
        "planned": 18000,
        "success": 1996,
        "permanent_failed": 0,
        "pending": 4,
        "not_started": 16000,
    }
    if classification["counts"] != expected_counts:
        raise RuntimeError(
            "unexpected_pre_authorization_state:"
            f"{classification['counts']}"
        )
    used = policy.cycle_calls_used(history)
    if used != 2298:
        raise RuntimeError(
            f"unexpected_cycle_call_count:{used}"
        )
    migrated = classification["pending_details"].get(
        "caption_ct_train_002493"
    )
    if not migrated or not all((
        migrated.get("latest_http_status") == 400,
        migrated.get("latest_temperature") == 1.0,
        migrated.get("latest_runtime_policy_revision") == 2,
        migrated.get("lifetime_attempts") == 5,
        migrated.get("recoverable") is True,
    )):
        raise RuntimeError(
            f"temperature_migration_state_mismatch:{migrated}"
        )

    runtime_policy = {
        "policy_revision": 3,
        "supersedes_policy_revision": 2,
        "scope": "frozen_first_round_only",
        "batch_indices": list(range(18)),
        "stable_record_count": 18000,
        "batch_18_and_later_authorized": False,
        "cycle_api_hard_limit": 18000,
        "cycle_api_calls_at_authorization": used,
        "cycle_api_calls_remaining_at_authorization": 18000 - used,
        "request_model_id": "k3",
        "expected_routed_model": "kimi-k2.6",
        "reasoning_effort": "none",
        "temperature": 0.6,
        "temperature_contract_reason": (
            "Kimi K3 with thinking disabled routes to K2.6; "
            "the provider requires temperature=0.6"
        ),
        "top_p_sent": False,
        "max_completion_tokens": 256,
        "base_url": "https://api.kimi.com/coding/v1",
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
        "historical_temperature_1_http_400_retained": True,
        "historical_temperature_1_http_400_superseded": True,
        "migrated_pending_id": "caption_ct_train_002493",
        "migrated_pending_lifetime_attempts": 5,
        "migrated_pending_remaining_attempts": 1,
        "same_window_max_attempts": 3,
        "lifetime_max_attempts": 6,
        "http_429_immediate_retry": False,
        "cooldown_seconds": policy.COOLDOWN_SECONDS,
        "existing_success_cache_recalled": False,
        "created_at": policy.iso_now(),
        "authorization_source": "explicit_user_instruction",
    }
    policy.atomic_write_json(POLICY_FILE, runtime_policy)

    revision = {
        "revision": 6,
        "revision_stage": "k2_6_temperature_0_6_migration",
        "timestamp": policy.iso_now(),
        "technical_acceptance_status": "PASS",
        "human_review": {
            "status": "NOT_VERIFIED_BY_USER",
            "verified_rows": 0,
        },
        "authorized_formal_batch_indices": list(range(18)),
        "batch_18_and_later_authorized": False,
        "first_round_record_count": 18000,
        "runtime_policy_revision": 3,
        "runtime_policy_path": str(POLICY_FILE),
        "runtime_policy_sha256": policy.sha256(POLICY_FILE),
        "temperature": 0.6,
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
        "revision": 6,
        "runtime_policy_revision": 3,
        "temperature": 0.6,
        "authorized_batches": list(range(18)),
        "batch_18_and_later_authorized": False,
        "cycle_api_calls_used": used,
    }))


if __name__ == "__main__":
    main()
