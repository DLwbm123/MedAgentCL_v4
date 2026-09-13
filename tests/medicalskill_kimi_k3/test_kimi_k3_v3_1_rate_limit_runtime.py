#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import RateLimitError


ROOT = Path("/root/MedAgentCL_v4")
SCRIPTS = ROOT / "scripts/medicalskill_kimi_k3"
ARTIFACT = (
    ROOT
    / "artifacts/medicalskill_cl_kimi_k3_v3_1_k2_6_routing"
)
sys.path.insert(0, str(SCRIPTS))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


runner = load(
    "v31_rate_limit_runtime_test",
    SCRIPTS / "run_kimi_k3_v3_1.py",
)


class FakeLimiter:
    async def wait(self):
        return None

    def success(self):
        return None

    def limited(self):
        return None


class FakeBudget:
    def __init__(self):
        self.used = 0

    async def reserve(self):
        self.used += 1
        return True


class FailingCompletions:
    def __init__(self, error):
        self.error = error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        raise self.error


def rate_limit_error(message: str) -> RateLimitError:
    request = httpx.Request(
        "POST", "https://api.kimi.com/coding/v1/chat/completions"
    )
    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": "60"},
        json={"error": {"message": message}},
    )
    return RateLimitError(
        message,
        response=response,
        body={"error": {"message": message}},
    )


class RateLimitRuntimeTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt, cls.schema, cls.config, _ = runner.load_contract(
            ARTIFACT
        )
        sources, _ = runner.load_sources(ARTIFACT, cls.config)
        cls.source = sources["caption_ct_train_002493"]

    async def invoke(self, message: str):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        results = directory / "results.jsonl"
        state = directory / "state.json"
        results.touch()
        writer = runner.ResultWriter(results)
        source_id = self.source["stable_record_id"]
        tracker = runner.StateTracker(
            state,
            results,
            [source_id],
            {},
            {source_id: self.source},
            self.prompt,
            self.config,
            "recovery_queue",
            "cycle_01",
            None,
        )
        completions = FailingCompletions(
            rate_limit_error(message)
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        budget = FakeBudget()
        result = await runner.request_one(
            client,
            FakeLimiter(),
            budget,
            writer,
            tracker,
            self.source,
            self.prompt,
            self.schema,
            self.config,
            "mock-rate-limit",
            3,
            "recovery_queue",
            "cycle_01",
            None,
            2,
            None,
            1,
        )
        return result, completions.calls, budget

    async def test_429_is_queued_without_same_window_retry(self):
        result, calls, budget = await self.invoke(
            "temporary request rate limit"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(budget.used, 1)
        self.assertEqual(result["http_status"], 429)
        self.assertEqual(result["quality_status"], "TRANSPORT_FAILED")
        self.assertFalse(result["quota_exhausted"])
        self.assertEqual(
            result["provider_error_message"],
            "temporary request rate limit",
        )
        self.assertEqual(calls[0]["temperature"], 0.6)
        self.assertNotIn("top_p", calls[0])

    async def test_explicit_weekly_quota_is_classified(self):
        result, calls, _ = await self.invoke(
            "weekly quota exceeded"
        )
        self.assertEqual(len(calls), 1)
        self.assertTrue(result["quota_exhausted"])
        self.assertEqual(result["quota_scope"], "weekly")
        self.assertEqual(
            result["provider_error_message"],
            "weekly quota exceeded",
        )


if __name__ == "__main__":
    unittest.main()



