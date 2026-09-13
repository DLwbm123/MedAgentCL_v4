#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path("/root/MedAgentCL_v4")
SCRIPTS = ROOT / "scripts/medicalskill_kimi_k3"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


policy = load(
    "v31_first_round_policy_test",
    SCRIPTS / "v3_1_first_round_policy.py",
)
runner = load(
    "v31_first_round_runner_test",
    SCRIPTS / "run_kimi_k3_v3_1.py",
)


def success_row(record_id: str) -> dict:
    return {
        "stable_id": record_id,
        "mode": "formal_batch",
        "cycle_id": "cycle_01",
        "api_call_performed": True,
        "cache_status": "active_v3_1_success",
        "quality_status": "PASS",
        "output_source": "raw_model",
        "http_status": 200,
        "error_type": None,
        "prompt_hash": policy.EXPECTED_HASHES["prompt_hash"],
        "schema_hash": policy.EXPECTED_HASHES["schema_hash"],
        "run_config_hash": policy.EXPECTED_HASHES["run_config_hash"],
        "reasoning_content_present": False,
        "reasoning_token_usage": 0,
        "final_concepts": ["computed tomography"],
        "timestamp": "2026-07-23T00:00:00+00:00",
    }


class FirstRoundPolicyTest(unittest.TestCase):
    def setUp(self):
        self.ids = [
            f"stable_{index:05d}"
            for index in range(policy.FIRST_ROUND_SIZE + 1000)
        ]

    def test_first_round_is_exactly_batches_00_through_17(self):
        self.assertEqual(policy.FIRST_BATCH, 0)
        self.assertEqual(policy.LAST_BATCH, 17)
        self.assertEqual(policy.FIRST_ROUND_SIZE, 18_000)
        self.assertEqual(runner.MAX_AUTHORIZED_BATCH_INDEX, 17)

    def test_batch_17_allowed_and_batch_18_blocked(self):
        base = dict(
            confirm_formal_batch=True,
            cycle_id="cycle_01",
            cycle_start_batch=0,
            artifact_root="/nonexistent",
        )
        revision = {
            "authorized_formal_batch_indices": list(range(18)),
        }
        config = {"formal_batches_total": 35}
        with mock.patch.object(
            runner, "acceptance_pass", return_value=True
        ), mock.patch.object(
            runner, "latest_acceptance_revision",
            return_value=revision,
        ):
            selected = runner.formal_selection(
                argparse.Namespace(
                    formal_batch_index=17, **base
                ),
                self.ids,
                config,
            )
            self.assertEqual(len(selected), 1000)
            with self.assertRaisesRegex(
                RuntimeError, "batch_18_and_later_not_authorized"
            ):
                runner.formal_selection(
                    argparse.Namespace(
                        formal_batch_index=18, **base
                    ),
                    self.ids,
                    config,
                )

    def test_successful_cache_is_never_pending(self):
        record_id = self.ids[0]
        classified = policy.classify_records(
            self.ids,
            [success_row(record_id)],
            {},
        )
        self.assertIn(record_id, classified["success"])
        self.assertNotIn(record_id, classified["pending"])
        self.assertNotIn(record_id, classified["not_started"])

    def test_recovery_cannot_cross_first_round_boundary(self):
        args = argparse.Namespace(
            confirm_recovery=True,
            recovery_stable_id=[self.ids[18_000]],
            cycle_id="cycle_01",
            cycle_start_batch=0,
            artifact_root="/nonexistent",
        )
        with mock.patch.object(
            runner, "acceptance_pass", return_value=True
        ):
            with self.assertRaisesRegex(
                RuntimeError, "recovery_id_outside_first_round"
            ):
                runner.recovery_selection(args, self.ids, {})

    def test_temperature_is_explicit_and_top_p_absent(self):
        kwargs = runner.request_kwargs(
            {"source_caption": "Normal study."},
            {
                "system_prompt": "Return concepts.",
                "user_template": "{caption}",
            },
            {"type": "object"},
        )
        self.assertEqual(kwargs["temperature"], 0.6)
        self.assertNotIn("top_p", kwargs)

    def test_rate_limit_burst_three_consecutive(self):
        detector = policy.RateLimitBurstDetector()
        self.assertIsNone(detector.record(True, "a"))
        self.assertIsNone(detector.record(True, "b"))
        self.assertIsNotNone(detector.record(True, "c"))

    def test_rate_limit_burst_five_of_twenty(self):
        detector = policy.RateLimitBurstDetector()
        decision = None
        for index in range(20):
            hit = index in {0, 4, 8, 12, 16}
            decision = detector.record(hit, f"id_{index}") or decision
        self.assertIsNotNone(decision)

    def test_cycle_budget_counts_formal_and_recovery_only(self):
        rows = [
            {
                **success_row("a"),
                "mode": "formal_batch",
            },
            {
                **success_row("b"),
                "mode": "recovery_queue",
            },
            {
                **success_row("c"),
                "mode": "pilot_50",
            },
        ]
        self.assertEqual(policy.cycle_calls_used(rows), 2)
        self.assertEqual(policy.CYCLE_CALL_LIMIT, 18_000)

    def test_one_percent_permanent_failure_is_inclusive(self):
        allowed = int(
            policy.FIRST_ROUND_SIZE
            * policy.MAX_PERMANENT_FAILURE_RATE
        )
        self.assertEqual(allowed, 180)
        self.assertLessEqual(
            allowed / policy.FIRST_ROUND_SIZE,
            policy.MAX_PERMANENT_FAILURE_RATE,
        )
        self.assertGreater(
            (allowed + 1) / policy.FIRST_ROUND_SIZE,
            policy.MAX_PERMANENT_FAILURE_RATE,
        )


if __name__ == "__main__":
    unittest.main()

