#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import httpx
from openai import AsyncOpenAI

ROOT = Path("/root/MedAgentCL_v4")
SCRIPTS = ROOT / "scripts/medicalskill_kimi_k3"
V2 = ROOT / "artifacts/medicalskill_cl_kimi_k3_v2_pilot"
V3 = ROOT / "artifacts/medicalskill_cl_kimi_k3_v3_k2_6_routing"
sys.path.insert(0, str(SCRIPTS))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_v3_test", SCRIPTS / "prepare_kimi_k3_v3.py")
runner = load("runner_v3_test", SCRIPTS / "run_kimi_k3_v3.py")
audit = load("audit_v3_test", SCRIPTS / "audit_kimi_k3_v3.py")


class V3ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt, cls.schema, cls.config, cls.paired = runner.load_contract(V3)
        cls.sources, cls.stable_ids = runner.load_sources(V3, cls.config)

    def test_01_request_model_is_k3(self):
        self.assertEqual(runner.REQUEST_MODEL, "k3")
        self.assertNotEqual(runner.REQUEST_MODEL, "kimi-k2.6")

    def test_02_reasoning_effort_is_none(self):
        self.assertEqual(runner.EFFORT, "none")

    def test_03_expected_route_is_distinct(self):
        self.assertEqual(runner.ROUTED_MODEL, "kimi-k2.6")
        self.assertNotEqual(self.config["request_model_id"], self.config["expected_routed_model"])

    def test_04_generation_fields_are_omitted(self):
        kwargs = runner.request_kwargs(next(iter(self.sources.values())), self.prompt, self.schema)
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "thinking"):
            self.assertNotIn(key, kwargs)

    def test_05_reasoning_effort_is_top_level(self):
        kwargs = runner.request_kwargs(next(iter(self.sources.values())), self.prompt, self.schema)
        self.assertEqual(kwargs["reasoning_effort"], "none")

    def test_06_max_completion_tokens_is_256(self):
        kwargs = runner.request_kwargs(next(iter(self.sources.values())), self.prompt, self.schema)
        self.assertEqual(kwargs["max_completion_tokens"], 256)

    def test_07_strict_json_schema(self):
        value = runner.request_kwargs(next(iter(self.sources.values())), self.prompt, self.schema)["response_format"]
        self.assertEqual(value["type"], "json_schema")
        self.assertIs(value["json_schema"]["strict"], True)

    def test_08_schema_accepts_one_to_eight(self):
        runner.validate_content(json.dumps({"concepts": ["pulmonary nodule"]}))
        runner.validate_content(json.dumps({"concepts": [str(i) for i in range(8)]}))

    def test_09_schema_rejects_empty(self):
        with self.assertRaises(ValueError):
            runner.validate_content(json.dumps({"concepts": []}))

    def test_10_schema_rejects_nine(self):
        with self.assertRaises(ValueError):
            runner.validate_content(json.dumps({"concepts": [str(i) for i in range(9)]}))

    def test_11_schema_rejects_extra_properties(self):
        with self.assertRaises(ValueError):
            runner.validate_content(json.dumps({"concepts": ["x"], "explanation": "no"}))

    def test_12_exact_duplicates_are_normalized(self):
        raw = '{"concepts":["Pulmonary nodule"," pulmonary   nodule ","right lung"]}'
        self.assertEqual(runner.validate_content(raw), ["pulmonary nodule", "right lung"])

    def test_13_uncertain_prefix_is_forbidden(self):
        with self.assertRaisesRegex(ValueError, "uncertainty"):
            runner.validate_content('{"concepts":["uncertain: pulmonary nodule"]}')

    def test_14_roi_is_forbidden(self):
        with self.assertRaisesRegex(ValueError, "generic_or_artifact"):
            runner.validate_content('{"concepts":["region of interest"]}')

    def test_15_bbox_is_forbidden(self):
        with self.assertRaisesRegex(ValueError, "generic_or_artifact"):
            runner.validate_content('{"concepts":["bounding box"]}')

    def test_16_generic_disease_process_is_forbidden(self):
        with self.assertRaisesRegex(ValueError, "generic_or_artifact"):
            runner.validate_content('{"concepts":["disease process"]}')

    def test_17_compound_ct_axis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "axis_not_split"):
            runner.validate_content('{"concepts":["abdominal CT scan"]}')

    def test_18_compound_mri_axis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "axis_not_split"):
            runner.validate_content('{"concepts":["brain MRI"]}')

    def test_19_split_modality_and_anatomy_are_accepted(self):
        self.assertEqual(
            runner.validate_content('{"concepts":["computed tomography","abdomen"]}'),
            ["computed tomography", "abdomen"],
        )

    def test_20_localized_lesion_phrase_is_preserved(self):
        self.assertEqual(runner.validate_content('{"concepts":["pelvic bone lesion"]}'), ["pelvic bone lesion"])

    def test_21_localized_mass_phrase_is_preserved(self):
        self.assertEqual(runner.validate_content('{"concepts":["right renal mass"]}'), ["right renal mass"])

    def test_22_fixed_terms_are_preserved(self):
        concepts = ["ground-glass opacity", "pleural effusion", "invasive ductal carcinoma"]
        self.assertEqual(runner.validate_content(json.dumps({"concepts": concepts})), concepts)

    def test_23_parent_child_lesion_duplicate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "parent_child"):
            runner.validate_content('{"concepts":["pelvic bone lesion","lesion"]}')

    def test_24_parent_child_carcinoma_duplicate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "parent_child"):
            runner.validate_content('{"concepts":["invasive ductal carcinoma","carcinoma"]}')

    def test_standalone_lesion_mass_and_carcinoma_are_concrete(self):
        concepts = ["lesion", "mass", "carcinoma"]
        self.assertEqual(runner.validate_content(json.dumps({"concepts": concepts})), concepts)

    def test_25_possible_finding_positive_form_is_accepted(self):
        self.assertEqual(runner.validate_content('{"concepts":["pulmonary nodule"]}'), ["pulmonary nodule"])
        self.assertIn("possible pulmonary nodule", self.prompt["user_template"])

    def test_26_may_suggest_finding_positive_form_is_documented(self):
        self.assertIn("findings may suggest cerebral edema", self.prompt["user_template"])
        self.assertEqual(runner.validate_content('{"concepts":["cerebral edema"]}'), ["cerebral edema"])

    def test_27_downstream_speculation_rule_is_documented(self):
        text = self.prompt["user_template"]
        self.assertIn("may affect adjacent tissue", text)
        self.assertIn("does not establish adjacent-tissue involvement", text)

    def test_28_differential_rule_is_documented(self):
        self.assertIn("mutually exclusive differential diagnoses", self.prompt["user_template"])

    def test_29_priority_order_is_documented(self):
        self.assertIn("specific diagnosis/pathology > salient abnormal finding", self.prompt["user_template"])

    def test_30_frozen_35k_id_hash(self):
        text = (V3 / "concept_35k_stable_ids.txt").read_text()
        self.assertEqual(len(text.splitlines()), 35_000)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), self.config["stable_id_set_sha256"])

    def test_31_paired_manifest_is_200_and_all_modalities(self):
        self.assertEqual(len(self.paired["records"]), 200)
        self.assertEqual(len({row["stable_id"] for row in self.paired["records"]}), 200)
        self.assertEqual(set(self.paired["distribution_by_modality"]), {
            "ct", "dermoscopy", "endoscopy", "histopathology", "microscopy", "mri", "pet", "xray",
        })

    def test_32_paired_manifest_covers_risk_strata(self):
        strata = set(self.paired["distribution_by_stratum"])
        self.assertTrue({"long_caption", "repair_heavy", "k3_v2_exactly_8", "stratified_random"} <= strata)

    def test_33_v2_cache_is_not_reusable(self):
        with (V2 / "v2_pilot_results.jsonl").open(encoding="utf-8") as handle:
            old = json.loads(next(handle))
        source = self.sources[old["stable_record_id"]]
        self.assertFalse(runner.active_success(old, source, self.prompt, self.config))
        self.assertIs(self.config["v2_cache_reusable_as_v3"], False)

    def test_34_v3_result_field_contract(self):
        source = next(iter(self.sources.values()))
        row = runner.make_result(source, self.prompt, self.config, "mock", 1)
        required = {
            "stable_id", "source_caption", "source_caption_sha256", "raw_response", "parsed_concepts",
            "base_url", "request_model_id", "response_model_id", "expected_routed_model",
            "reasoning_effort", "reasoning_content_present", "reasoning_token_usage",
            "prompt_version", "prompt_hash", "schema_version", "schema_hash",
            "prompt_tokens", "cached_tokens", "completion_tokens", "latency", "attempt_count",
            "http_status", "error_type", "timestamp", "cache_status",
        }
        self.assertTrue(required <= set(row))

    def test_35_full_confirmation_and_acceptance_gates_exist(self):
        source = (SCRIPTS / "run_kimi_k3_v3.py").read_text()
        self.assertIn("--confirm-full-run", source)
        self.assertIn("canary_acceptance_not_pass", source)
        self.assertIn("paired_pilot_acceptance_not_pass", source)

    def test_36_403_is_not_retryable(self):
        self.assertNotIn(403, runner.RETRYABLE_STATUS)
        self.assertIn(429, runner.RETRYABLE_STATUS)
        self.assertIn(503, runner.RETRYABLE_STATUS)

    def test_37_v3_result_file_contract_when_present(self):
        path = V3 / "v3_results.jsonl"
        if not path.exists():
            return
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["base_url"], runner.BASE_URL)
            self.assertEqual(row["request_model_id"], runner.REQUEST_MODEL)
            self.assertNotIn("api_key", row)
            self.assertNotIn("authorization", row)
            self.assertNotIn("sk-", json.dumps(row).casefold())


    def test_38_audit_counts_exact_generic_terms(self):
        self.assertTrue(audit.is_generic_or_artifact("pathology"))
        self.assertTrue(audit.is_generic_or_artifact("abnormality"))
        self.assertTrue(audit.is_generic_or_artifact("region of interest"))
        self.assertFalse(audit.is_generic_or_artifact("pleural effusion"))
        self.assertFalse(audit.is_generic_or_artifact("lesion"))
        self.assertFalse(audit.is_generic_or_artifact("carcinoma"))


class SDKSerializationTest(unittest.IsolatedAsyncioTestCase):
    async def test_39_mock_transport_serializes_none_without_sampling_fields(self):
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "mock", "object": "chat.completion", "created": 0, "model": "k3",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"concepts":["pulmonary nodule"]}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            })

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncOpenAI(api_key="mock-not-a-real-key", base_url=runner.BASE_URL, http_client=http_client)
        prompt, schema, _, _ = runner.load_contract(V3)
        sources, _ = runner.load_sources(V3, json.loads((V3 / "v3_run_config.json").read_text()))
        response = await client.chat.completions.create(**runner.request_kwargs(next(iter(sources.values())), prompt, schema))
        await client.close()
        self.assertEqual(response.model, "k3")
        self.assertEqual(captured["model"], "k3")
        self.assertEqual(captured["reasoning_effort"], "none")
        self.assertEqual(captured["max_completion_tokens"], 256)
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "thinking"):
            self.assertNotIn(key, captured)
        self.assertTrue(captured["response_format"]["json_schema"]["strict"])


if __name__ == "__main__":
    unittest.main()
