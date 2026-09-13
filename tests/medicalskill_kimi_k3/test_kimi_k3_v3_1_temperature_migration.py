#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path("/root/MedAgentCL_v4")
POLICY_PATH = (
    ROOT
    / "scripts/medicalskill_kimi_k3/v3_1_first_round_policy.py"
)


def load_policy():
    spec = importlib.util.spec_from_file_location(
        "v31_temperature_migration_test", POLICY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


policy = load_policy()


def failure_row(
    record_id: str,
    temperature: float,
    runtime_policy_revision: int,
) -> dict:
    return {
        "stable_id": record_id,
        "mode": "recovery_queue",
        "cycle_id": "cycle_01",
        "batch_index": 0,
        "api_call_performed": True,
        "cache_status": "failed_v3_1_attempt",
        "quality_status": "TRANSPORT_FAILED",
        "output_source": "raw_model",
        "http_status": 400,
        "error_type": "http_400",
        "temperature": temperature,
        "runtime_policy_revision": runtime_policy_revision,
        "prompt_hash": policy.EXPECTED_HASHES["prompt_hash"],
        "schema_hash": policy.EXPECTED_HASHES["schema_hash"],
        "run_config_hash": policy.EXPECTED_HASHES["run_config_hash"],
        "reasoning_content_present": False,
        "reasoning_token_usage": 0,
        "final_concepts": [],
        "timestamp": "2026-07-25T00:00:00+00:00",
    }


class TemperatureMigrationTest(unittest.TestCase):
    def setUp(self):
        self.ids = [
            f"stable_{index:05d}"
            for index in range(policy.FIRST_ROUND_SIZE)
        ]

    def test_runtime_contract_is_revision_3_temperature_0_6(self):
        self.assertEqual(policy.RUNTIME_POLICY_REVISION, 3)
        self.assertEqual(policy.TEMPERATURE, 0.6)

    def test_revision_2_temperature_1_http_400_is_recoverable(self):
        record_id = self.ids[0]
        classified = policy.classify_records(
            self.ids,
            [failure_row(record_id, 1.0, 2)],
            {},
        )
        details = classified["pending_details"][record_id]
        self.assertTrue(details["recoverable"])
        self.assertEqual(details["latest_temperature"], 1.0)
        self.assertEqual(
            details["latest_runtime_policy_revision"], 2
        )

    def test_revision_3_http_400_remains_hard_failure(self):
        record_id = self.ids[0]
        classified = policy.classify_records(
            self.ids,
            [failure_row(record_id, 0.6, 3)],
            {},
        )
        self.assertFalse(
            classified["pending_details"][record_id]["recoverable"]
        )


if __name__ == "__main__":
    unittest.main()
