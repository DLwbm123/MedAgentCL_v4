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
        "v31_state_classification_test", POLICY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


policy = load_policy()


def row(
    record_id: str,
    *,
    mode: str,
    batch_index: int | None,
    success: bool,
    api_call: bool,
) -> dict:
    return {
        "stable_id": record_id,
        "mode": mode,
        "cycle_id": (
            "cycle_01"
            if mode in policy.CYCLE_STATE_MODES
            else None
        ),
        "batch_index": batch_index,
        "api_call_performed": api_call,
        "cache_status": (
            "active_v3_1_success"
            if success
            else "failed_v3_1_attempt"
        ),
        "quality_status": "PASS" if success else "TRANSPORT_FAILED",
        "output_source": "raw_model",
        "http_status": 200 if success else 429,
        "error_type": None if success else "rate_limit",
        "prompt_hash": policy.EXPECTED_HASHES["prompt_hash"],
        "schema_hash": policy.EXPECTED_HASHES["schema_hash"],
        "run_config_hash": policy.EXPECTED_HASHES["run_config_hash"],
        "reasoning_content_present": False,
        "reasoning_token_usage": 0,
        "finish_reason": "stop",
        "final_concepts": ["computed tomography"] if success else [],
        "raw_model_concepts": ["computed tomography"] if success else [],
        "timestamp": "2026-07-25T00:00:00+00:00",
    }


class StateClassificationTest(unittest.TestCase):
    def setUp(self):
        self.ids = [
            f"stable_{index:05d}"
            for index in range(policy.FIRST_ROUND_SIZE)
        ]

    def test_unstarted_pilot_success_does_not_complete_future_batch(self):
        future = self.ids[17_000]
        classified = policy.classify_records(
            self.ids,
            [
                row(
                    future,
                    mode="pilot_50",
                    batch_index=None,
                    success=True,
                    api_call=True,
                )
            ],
            {},
        )
        self.assertIn(future, classified["not_started"])
        self.assertNotIn(future, classified["success"])

    def test_started_batch_inherits_valid_cache_only_record(self):
        cached = self.ids[0]
        starter = self.ids[1]
        classified = policy.classify_records(
            self.ids,
            [
                row(
                    cached,
                    mode="lineage_annotation_revision_001",
                    batch_index=None,
                    success=True,
                    api_call=False,
                ),
                row(
                    starter,
                    mode="formal_batch",
                    batch_index=0,
                    success=True,
                    api_call=True,
                ),
            ],
            {},
        )
        self.assertIn(cached, classified["success"])
        self.assertIn(starter, classified["success"])

    def test_reclassification_supersedes_old_formal_rejection(self):
        record_id = self.ids[0]
        classified = policy.classify_records(
            self.ids,
            [
                row(
                    record_id,
                    mode="formal_batch",
                    batch_index=0,
                    success=False,
                    api_call=True,
                ),
                row(
                    record_id,
                    mode="batch0_quality_reclassification_revision_001",
                    batch_index=0,
                    success=True,
                    api_call=False,
                ),
            ],
            {},
        )
        self.assertIn(record_id, classified["success"])
        self.assertNotIn(record_id, classified["pending"])

    def test_pilot_calls_do_not_consume_cycle_budget_or_lifetime(self):
        record_id = self.ids[0]
        pilot = row(
            record_id,
            mode="pilot_50",
            batch_index=None,
            success=True,
            api_call=True,
        )
        formal = row(
            record_id,
            mode="formal_batch",
            batch_index=0,
            success=False,
            api_call=True,
        )
        self.assertEqual(policy.cycle_calls_used([pilot, formal]), 1)
        self.assertEqual(policy.attempt_counts([pilot, formal])[record_id], 1)


if __name__ == "__main__":
    unittest.main()
