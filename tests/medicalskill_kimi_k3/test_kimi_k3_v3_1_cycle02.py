import asyncio
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path("/root/MedAgentCL_v4")
RUNNER_PATH = (
    ROOT
    / "scripts/medicalskill_kimi_k3"
    / "run_kimi_k3_v3_1_cycle02_35k.py"
)
SPEC = importlib.util.spec_from_file_location(
    "kimi_v31_cycle02_runner", RUNNER_PATH
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class Cycle02PolicyTest(unittest.TestCase):
    def test_frozen_contract_and_batch_plan(self):
        prompt, schema, config, _ = runner.core.load_contract(
            runner.ARTIFACT
        )
        sources, stable_ids = runner.core.load_sources(
            runner.ARTIFACT, config
        )
        kwargs = runner.core.request_kwargs(
            sources[stable_ids[0]], prompt, schema
        )
        self.assertEqual(kwargs["temperature"], 0.6)
        self.assertNotIn("top_p", kwargs)
        plan = runner.batch_plan(stable_ids, set())
        self.assertEqual(len(plan), 18)
        self.assertEqual(plan[0]["batch_index"], 17)
        self.assertEqual(plan[-1]["batch_index"], 34)
        self.assertEqual(
            sum(len(entry["selected_ids"]) for entry in plan),
            18_000,
        )

    def test_budget_reservation_never_reaches_18001(self):
        async def exercise():
            with tempfile.TemporaryDirectory() as temp_dir:
                original_events = runner.EVENTS_PATH
                runner.EVENTS_PATH = Path(temp_dir) / "events.jsonl"
                try:
                    budget = runner.AtomicCycleBudget(
                        17_999, {"last"}
                    )
                    self.assertTrue(
                        await budget.reserve("last", False)
                    )
                    self.assertFalse(
                        await budget.reserve("last", True)
                    )
                    snapshot = await budget.snapshot()
                    self.assertEqual(
                        snapshot["cycle_api_attempts"], 18_000
                    )
                    self.assertEqual(
                        runner.budget_reservation_count(), 1
                    )
                finally:
                    runner.EVENTS_PATH = original_events

        asyncio.run(exercise())

    def test_initial_retry_margin_accounts_for_cache(self):
        async def exercise():
            with tempfile.TemporaryDirectory() as temp_dir:
                original_events = runner.EVENTS_PATH
                runner.EVENTS_PATH = Path(temp_dir) / "events.jsonl"
                try:
                    budget = runner.AtomicCycleBudget(
                        0,
                        {f"id-{index}" for index in range(17_255)},
                    )
                    snapshot = await budget.snapshot()
                    self.assertEqual(
                        snapshot["retry_margin"], 745
                    )
                finally:
                    runner.EVENTS_PATH = original_events

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
