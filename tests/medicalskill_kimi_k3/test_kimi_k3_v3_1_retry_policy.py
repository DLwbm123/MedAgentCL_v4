#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/root/MedAgentCL_v4")
SCRIPTS = ROOT / "scripts/medicalskill_kimi_k3"
ARTIFACT = ROOT / "artifacts/medicalskill_cl_kimi_k3_v3_1_k2_6_routing"
sys.path.insert(0, str(SCRIPTS))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


runner = load("runner_v31_retry_test", SCRIPTS / "run_kimi_k3_v3_1.py")
quality = load("quality_v31_retry_test", SCRIPTS / "v3_1_quality_policy.py")


class FakeLimiter:
    async def wait(self):
        return None

    def success(self):
        return None

    def limited(self):
        return None


class FakeBudget:
    def __init__(self, limit=10):
        self.used = 0
        self.limit = limit

    async def reserve(self):
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            content, finish_reason = item
        else:
            content, finish_reason = item, "stop"
        usage = SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=6,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )
        return SimpleNamespace(
            model="k3",
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content, reasoning_content=None),
                finish_reason=finish_reason,
            )],
            usage=usage,
        )


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class TechnicalRetryPolicyTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt, cls.schema, cls.config, _ = runner.load_contract(ARTIFACT)
        sources, _ = runner.load_sources(ARTIFACT, cls.config)
        cls.source = sources["caption_xray_train_005620"]

    async def invoke(self, responses):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        directory = Path(temp.name)
        results = directory / "results.jsonl"
        state = directory / "state.json"
        results.touch()
        source_id = self.source["stable_record_id"]
        writer = runner.ResultWriter(results)
        tracker = runner.StateTracker(
            state, results, [source_id], {}, {source_id: self.source},
            self.prompt, self.config, "mock", None, None,
        )
        client = FakeClient(responses)
        row = await runner.request_one(
            client, FakeLimiter(), FakeBudget(), writer, tracker, self.source,
            self.prompt, self.schema, self.config, "mock-run", 0,
            "mock", None, None, 0, None,
        )
        return row, client.chat.completions.calls, runner.read_results(results)

    async def test_semantic_qc_never_retries_or_rewrites(self):
        raw = json.dumps({"concepts": ["pneumothorax", "absent lung markings"]})
        row, calls, history = await self.invoke([raw])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(history), 1)
        self.assertEqual(row["quality_status"], "PASS_WITH_QC_FLAGS")
        self.assertEqual(row["final_concepts"], ["pneumothorax", "absent lung markings"])
        self.assertIn("negated_concept", row["audit_flags"])
        self.assertEqual(row["output_source"], "raw_model")

    async def test_invalid_json_gets_one_technical_retry_with_same_contract(self):
        good = json.dumps({"concepts": ["pneumothorax", "pleural air"]})
        row, calls, history = await self.invoke(["not json", good])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(
            [item["quality_status"] for item in history],
            ["TECHNICAL_RETRY_REQUIRED", "PASS"],
        )
        self.assertEqual(row["output_source"], "retry_model")
        self.assertEqual(row["repair_or_retry_reason"], ["invalid_json"])

    async def test_second_invalid_json_is_terminal_without_third_call(self):
        row, calls, history = await self.invoke(["not json", "still not json"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(history), 2)
        self.assertEqual(row["quality_status"], "TECHNICAL_FAILED_AFTER_RETRY")
        self.assertEqual(row["final_concepts"], [])
        self.assertTrue(runner.terminal_rejected(row))

    async def test_length_finish_reason_gets_technical_retry(self):
        bad = (json.dumps({"concepts": ["pneumothorax"]}), "length")
        good = json.dumps({"concepts": ["pneumothorax", "pleural air"]})
        row, calls, _ = await self.invoke([bad, good])
        self.assertEqual(len(calls), 2)
        self.assertEqual(row["quality_status"], "PASS")
        self.assertIn("truncated_output", row["repair_or_retry_reason"])

    def test_parent_child_revision_one_contract(self):
        legal = (
            ["lung", "lung lesion"],
            ["liver", "liver lesion"],
            ["kidney", "renal lesion"],
            ["brain", "brain lesion"],
            ["abdomen", "abdominal lesion"],
            ["chest", "thoracic lesion"],
            ["pelvic bone", "pelvic bone lesion"],
        )
        for concepts in legal:
            audited = quality.audit_concepts(list(concepts))
            self.assertNotIn("parent_child_duplicate", audited["audit_flags"])
            self.assertEqual(audited["retry_reasons"], [])
        redundant = (
            ["lesion", "pelvic bone lesion"],
            ["tumor", "neuroendocrine tumor"],
            ["neuroendocrine tumor", "grade 2 neuroendocrine tumor"],
            ["malignancy", "breast carcinoma"],
        )
        for concepts in redundant:
            audited = quality.audit_concepts(list(concepts))
            self.assertIn("parent_child_duplicate", audited["audit_flags"])
            self.assertEqual(audited["retry_reasons"], [])

    def test_cap8_and_generic_are_qc_only(self):
        cap8 = quality.audit_concepts([f"concept {i}" for i in range(8)])
        self.assertIn("cap8_review", cap8["audit_flags"])
        self.assertEqual(cap8["retry_reasons"], [])
        generic = quality.audit_concepts(["lesion", "mass"])
        self.assertIn("generic_only", generic["audit_flags"])
        self.assertEqual(generic["retry_reasons"], [])

    def test_flagged_result_is_active(self):
        prompt, _, config, _ = runner.load_contract(ARTIFACT)
        source = self.source
        row = {
            "cache_status": "active_v3_1_success",
            "quality_status": "PASS_WITH_QC_FLAGS",
            "output_source": "raw_model",
            "stable_id": source["stable_record_id"],
            "source_caption_sha256": source["source_caption_sha256"],
            "request_model_id": runner.REQUEST_MODEL,
            "reasoning_effort": runner.EFFORT,
            "expected_routed_model": runner.ROUTED_MODEL,
            "prompt_hash": prompt["prompt_hash"],
            "schema_hash": prompt["schema_hash"],
            "run_config_hash": config["run_config_hash"],
            "reasoning_content_present": False,
            "reasoning_token_usage": 0,
            "http_status": 200,
            "error_type": None,
            "final_concepts": ["pneumothorax", "absent lung markings"],
        }
        self.assertTrue(runner.active_success(row, source, prompt, config))


if __name__ == "__main__":
    unittest.main()
