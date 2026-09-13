from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import unittest
from pathlib import Path


ROOT = Path("/root/MedAgentCL_v4/scripts/medicalskill_kimi_k3")


def load(name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / file_name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepare = load("kimi_prepare", "prepare_kimi_k3_pilot.py")
        cls.runner = load("kimi_runner", "run_kimi_k3_pilot.py")

    def test_schema_contract(self):
        schema = self.prepare.SCHEMA
        self.assertEqual(schema["properties"]["concepts"]["minItems"], 1)
        self.assertEqual(schema["properties"]["concepts"]["maxItems"], 8)
        self.assertFalse(schema["additionalProperties"])

    def test_normalization_and_validation(self):
        concepts = self.runner.validate_content(json.dumps({"concepts": [" Right upper lobe ", "CT"]}))
        self.assertEqual(concepts, ["right upper lobe", "ct"])

    def test_invalid_output_is_rejected_without_guessing(self):
        with self.assertRaisesRegex(ValueError, "concept_duplicate"):
            self.runner.validate_content(json.dumps({"concepts": ["CT", "ct"]}))
        with self.assertRaisesRegex(ValueError, "obvious_artifact"):
            self.runner.validate_content(json.dumps({"concepts": ["bounding box"]}))
        with self.assertRaisesRegex(ValueError, "markdown_code_fence"):
            self.runner.validate_content("```json\n{\"concepts\":[\"lung\"]}\n```")

    def test_endpoint_priority_and_normalization(self):
        default = self.runner.resolve_base_url({})
        self.assertEqual(default["base_url_source"], "default")
        self.assertEqual(default["chat_completions_url"], "https://api.kimi.com/coding/v1/chat/completions")
        old = self.runner.resolve_base_url({"MOONSHOT_BASE_URL": "https://api.kimi.com/coding/v1/"})
        self.assertEqual(old["base_url_source"], "MOONSHOT_BASE_URL")
        preferred = self.runner.resolve_base_url({
            "MOONSHOT_BASE_URL": "https://api.kimi.com/coding/v1",
            "KIMI_BASE_URL": "https://api.kimi.com/coding/v1/",
        })
        self.assertEqual(preferred["base_url_source"], "KIMI_BASE_URL")
        for invalid in (
            "https://api.moonshot.cn/v1",
            "https://api.kimi.com/coding/v1/v1",
            "https://api.kimi.com/coding/v1/chat/completions",
        ):
            with self.assertRaisesRegex(RuntimeError, "unsupported_base_url"):
                self.runner.resolve_base_url({"KIMI_BASE_URL": invalid})

    def test_credential_priority_and_ambiguity(self):
        self.assertEqual(self.runner.resolve_credential({"KIMI_API_KEY": "a"})[1], "KIMI_API_KEY")
        self.assertEqual(self.runner.resolve_credential({"MOONSHOT_API_KEY": "a"})[1], "MOONSHOT_API_KEY")
        self.assertEqual(self.runner.resolve_credential({"KIMI_API_KEY": "a", "MOONSHOT_API_KEY": "a"})[1], "KIMI_API_KEY")
        with self.assertRaisesRegex(RuntimeError, "ambiguous_credentials"):
            self.runner.resolve_credential({"KIMI_API_KEY": "a", "MOONSHOT_API_KEY": "b"})

    def test_secret_redaction(self):
        secret = "private-token-value-123456789"
        value = self.runner.redact_secrets(
            f"Authorization: Bearer {secret}; api_key={secret}", {"KIMI_API_KEY": secret}
        )
        self.assertNotIn(secret, value)
        self.assertIn("[REDACTED]", value)

    def test_active_cache_requires_full_contract(self):
        endpoint = self.runner.resolve_base_url({})
        source = {"stable_record_id": "x", "source_caption_sha256": "source"}
        prompt = {"prompt_hash": "prompt", "schema_hash": "schema"}
        cached = {
            "status": "success", "contract_status": "PASS", "stable_record_id": "x",
            "source_caption_sha256": "source", "prompt_hash": "prompt", "schema_hash": "schema",
            "model_returned": "kimi-k3", "endpoint_hostname": "api.kimi.com",
            "endpoint_base_path": "/coding/v1", "max_completion_tokens": 512,
            "finish_reason": "stop", "parsed_concepts": ["lung"],
        }
        self.assertTrue(self.runner.active_success(cached, source, prompt, endpoint))
        cached["max_completion_tokens"] = 128
        self.assertFalse(self.runner.active_success(cached, source, prompt, endpoint))

    def test_wire_request_uses_only_temperature_one(self):
        source = inspect.getsource(self.runner.request_one)
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "create"]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertIn("temperature", keywords)
        self.assertIsInstance(keywords["temperature"], ast.Constant)
        self.assertEqual(keywords["temperature"].value, 1.0)
        self.assertNotIn("top_p", keywords)

    def test_quota_is_exact(self):
        groups = {"a": [{}, {}, {}], "b": [{}, {}]}
        quotas = self.prepare.proportional_quotas(groups, 4)
        self.assertEqual(sum(quotas.values()), 4)


if __name__ == "__main__":
    unittest.main()
