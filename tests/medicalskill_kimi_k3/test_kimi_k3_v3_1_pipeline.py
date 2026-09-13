#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
from openai import AsyncOpenAI

ROOT = Path("/root/MedAgentCL_v4")
SCRIPTS = ROOT / "scripts/medicalskill_kimi_k3"
V3 = ROOT / "artifacts/medicalskill_cl_kimi_k3_v3_k2_6_routing"
V31 = ROOT / "artifacts/medicalskill_cl_kimi_k3_v3_1_k2_6_routing"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_v31_test", SCRIPTS / "prepare_kimi_k3_v3_1.py")
runner = load("runner_v31_test", SCRIPTS / "run_kimi_k3_v3_1.py")
quality = load("quality_v31_test", SCRIPTS / "v3_1_quality_policy.py")


class V31ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt, cls.schema, cls.config, cls.manifest = runner.load_contract(V31)
        cls.sources, cls.stable_ids = runner.load_sources(V31, cls.config)

    def test_01_request_contract_unchanged(self):
        self.assertEqual(runner.BASE_URL, "https://api.kimi.com/coding/v1")
        self.assertEqual(runner.REQUEST_MODEL, "k3")
        self.assertEqual(runner.EFFORT, "none")
        self.assertEqual(runner.ROUTED_MODEL, "kimi-k2.6")
        self.assertEqual(runner.MAX_TOKENS, 256)

    def test_02_temperature_fixed_and_optional_sampling_omitted(self):
        kwargs = runner.request_kwargs(next(iter(self.sources.values())), self.prompt, self.schema)
        self.assertEqual(kwargs["temperature"], 0.6)
        for key in ("top_p", "presence_penalty", "frequency_penalty", "thinking"):
            self.assertNotIn(key, kwargs)

    def test_03_reasoning_effort_top_level(self):
        kwargs = runner.request_kwargs(next(iter(self.sources.values())), self.prompt, self.schema)
        self.assertEqual(kwargs["reasoning_effort"], "none")

    def test_04_strict_schema(self):
        value = runner.request_kwargs(next(iter(self.sources.values())), self.prompt, self.schema)["response_format"]
        self.assertTrue(value["json_schema"]["strict"])
        self.assertEqual(value["json_schema"]["schema"]["properties"]["concepts"]["minItems"], 1)
        self.assertEqual(value["json_schema"]["schema"]["properties"]["concepts"]["maxItems"], 8)

    def test_05_prompt_hash_differs_from_v3(self):
        old = json.loads((V3 / "prompt_v3_manifest.json").read_text())
        self.assertNotEqual(old["prompt_hash"], self.prompt["prompt_hash"])

    def test_06_schema_hash_preserved(self):
        old = json.loads((V3 / "prompt_v3_manifest.json").read_text())
        self.assertEqual(old["schema_hash"], self.prompt["schema_hash"])

    def test_07_stable_ids_unchanged(self):
        old = (V3 / "concept_35k_stable_ids.txt").read_bytes()
        new = (V31 / "concept_35k_stable_ids.txt").read_bytes()
        self.assertEqual(hashlib.sha256(old).hexdigest(), hashlib.sha256(new).hexdigest())
        self.assertEqual(len(new.splitlines()), 35_000)

    def test_08_v3_cache_is_not_v31_cache(self):
        self.assertIs(self.config["v3_cache_reusable_as_v3_1"], False)
        self.assertNotEqual(V3 / "v3_results.jsonl", V31 / "v3_1_results.jsonl")

    def test_09_prompt_has_negation_rule(self):
        text = self.prompt["user_template"]
        for phrase in ("no necrosis", "absence of pleomorphism", "no evidence of metastasis", "without mitotic activity"):
            self.assertIn(phrase, text)

    def test_10_prompt_has_missing_lung_markings_exception(self):
        text = self.prompt["user_template"]
        self.assertIn("absent lung markings", text)
        self.assertIn("pneumothorax", text)
        self.assertIn("pleural air", text)

    def test_11_prompt_has_contradiction_rule(self):
        text = self.prompt["user_template"]
        self.assertIn("high-grade with low-grade", text)
        self.assertIn("benign with malignant", text)
        self.assertIn("well-differentiated with poorly differentiated", text)

    def test_12_prompt_has_background_rule(self):
        text = self.prompt["user_template"]
        self.assertIn("Do not enumerate every visible background organ", text)
        self.assertIn("upper abdominal lesion", text)

    def test_13_existing_semantics_preserved(self):
        text = self.prompt["user_template"]
        self.assertIn("Never output an `uncertain:` prefix", text)
        self.assertIn("abdominal CT scan", text)
        self.assertIn("pelvic bone lesion", text)

    def test_14_pilot_is_exactly_50(self):
        self.assertEqual(len(self.manifest["records"]), 50)
        self.assertEqual(len({row["stable_id"] for row in self.manifest["records"]}), 50)

    def test_15_pilot_covers_all_modalities(self):
        self.assertEqual(set(self.manifest["distribution_by_modality"]), {
            "ct", "dermoscopy", "endoscopy", "histopathology",
            "microscopy", "mri", "pet", "xray",
        })

    def test_16_all_v3_negative_outputs_selected(self):
        selected = {row["stable_id"] for row in self.manifest["records"]}
        v3_manifest = json.loads((V3 / "v3_paired_pilot_200_manifest.json").read_text())
        history = runner.read_results(V3 / "v3_results.jsonl")
        latest = {row["stable_id"]: row for row in history if row.get("cache_status") == "active_v3_success"}
        required = {
            row["stable_id"] for row in v3_manifest["records"]
            if any(runner.NEGATED.search(value) for value in latest[row["stable_id"]]["parsed_concepts"])
        }
        self.assertTrue(required <= selected)
        self.assertEqual(len(required), 7)

    def test_17_all_v3_conflicts_selected(self):
        selected = {row["stable_id"] for row in self.manifest["records"]}
        required = {
            row["stable_id"] for row in self.manifest["records"]
            if "v3_mutually_exclusive_output" in row["selection_reasons"]
        }
        self.assertEqual(len(required), 1)
        self.assertTrue(required <= selected)

    def test_18_all_v3_cap8_selected(self):
        selected_cap8 = [row for row in self.manifest["records"] if row["v3_concept_count"] == 8]
        self.assertEqual(len(selected_cap8), 31)

    def test_19_manifest_is_reference_only(self):
        self.assertIs(self.manifest["v3_results_are_reference_not_v3_1_cache"], True)
        self.assertTrue(all(row["v3_result_is_reference_only"] for row in self.manifest["records"]))

    def test_20_validate_one_to_eight(self):
        self.assertEqual(runner.validate_content('{"concepts":["lesion"]}'), ["lesion"])
        self.assertEqual(len(runner.validate_content(json.dumps({"concepts": [str(i) for i in range(8)]}))), 8)
        with self.assertRaises(ValueError):
            runner.validate_content('{"concepts":[]}')

    def test_21_negated_concept_detected(self):
        self.assertIn("negated_concept", runner.semantic_violations(["no necrosis"]))
        self.assertIn("negated_concept", runner.semantic_violations(["absence of colloid"]))
        self.assertNotIn("negated_concept", runner.semantic_violations(["pneumothorax"]))

    def test_negated_sign_is_deterministically_removed(self):
        raw = '{"concepts":["pneumothorax","pleural air","absent lung markings"]}'
        concepts, repairs, original = runner.parse_with_repairs(raw)
        self.assertEqual(original, ["pneumothorax", "pleural air", "absent lung markings"])
        self.assertEqual(concepts, ["pneumothorax", "pleural air"])
        self.assertIn("remove_explicitly_negated_concept", repairs)

    def test_obvious_background_enumeration_is_deterministically_removed(self):
        raw = json.dumps({"concepts": [
            "positron emission tomography", "increased fdg uptake", "upper abdominal lesion",
            "brain", "heart", "liver", "spleen",
        ]})
        concepts, repairs, original = runner.parse_with_repairs(raw)
        self.assertEqual(len(original), 7)
        self.assertEqual(concepts, [
            "positron emission tomography", "increased fdg uptake", "upper abdominal lesion",
        ])
        self.assertIn("remove_obvious_background_organ_enumeration", repairs)

    def test_22_high_low_conflict_detected(self):
        self.assertTrue(runner.contradiction_hits(["high-grade tumor", "low-grade tumor"]))

    def test_23_benign_malignant_conflict_detected(self):
        self.assertTrue(runner.contradiction_hits(["benign tumor", "carcinoma"]))

    def test_24_differentiation_conflict_detected(self):
        self.assertTrue(runner.contradiction_hits(["well-differentiated", "poorly differentiated"]))

    def test_25_non_conflicting_morphology_allowed(self):
        self.assertFalse(runner.contradiction_hits(["nuclear pleomorphism", "mitotic figures"]))

    def test_26_background_enumeration_detected(self):
        values = ["brain", "heart", "liver", "spleen", "positron emission tomography"]
        self.assertIn("background_organ_enumeration", runner.semantic_violations(values))

    def test_27_localized_anatomy_not_background_enumeration(self):
        values = ["positron emission tomography", "increased fdg uptake", "upper abdominal lesion", "liver"]
        self.assertNotIn("background_organ_enumeration", runner.semantic_violations(values))

    def test_28_no_full_mode(self):
        source = (SCRIPTS / "run_kimi_k3_v3_1.py").read_text()
        self.assertNotIn('add_argument("--full"', source)
        self.assertIn("--formal-batch-index", source)

    def test_29_batch_and_cycle_limits(self):
        self.assertEqual(runner.BATCH_SIZE, 1000)
        self.assertEqual(runner.CYCLE_HARD_LIMIT, 18_000)
        self.assertEqual(self.config["current_cycle_batch_count"], 18)

    def test_30_formal_gate_uses_append_only_acceptance_revision(self):
        revision = runner.latest_acceptance_revision(V31)
        expected = bool(revision) and revision.get("technical_acceptance_status") == "PASS" and revision.get("human_review", {}).get("status") == "NOT_VERIFIED_BY_USER"
        self.assertEqual(runner.acceptance_pass(V31), expected)

    def test_31_v31_result_contract_when_present(self):
        path = V31 / "v3_1_results.jsonl"
        if not path.exists():
            return
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["base_url"], runner.BASE_URL)
            self.assertEqual(row["prompt_hash"], self.prompt["prompt_hash"])
            self.assertNotIn("api_key", row)
            self.assertNotIn("authorization", row)

    def test_32_atomic_json_is_valid_and_fsynced_path_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            runner.atomic_write_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            self.assertFalse(list(Path(directory).glob(".*.tmp")))

    def test_33_interrupted_atomic_replace_preserves_old_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            runner.atomic_write_json(path, {"generation": 1})
            with mock.patch.object(runner.os, "replace", side_effect=RuntimeError("mock interruption")):
                with self.assertRaises(RuntimeError):
                    runner.atomic_write_json(path, {"generation": 2})
            self.assertEqual(json.loads(path.read_text()), {"generation": 1})
            self.assertFalse(list(Path(directory).glob(".*.tmp")))

    def _mock_contract_row(self):
        record_id = self.manifest["records"][0]["stable_id"]
        source = self.sources[record_id]
        row = runner.make_result(
            source, self.prompt, self.config, "mock-run", 1,
            "pilot_50", None, None,
            raw_model_concepts=["pleural effusion"],
            final_concepts=["pleural effusion"],
            output_source="raw_model",
            original_run_id="mock-run:original",
            retry_run_id=None,
            audit_flags=[],
            quality_status="PASS",
            response_model_id="k3",
            reasoning_content_present=False,
            reasoning_token_usage=0,
            http_status=200,
            error_type=None,
            finish_reason="stop",
            cache_status="active_v3_1_success",
        )
        return record_id, source, row

    def test_34_empty_state_rebuilds_from_results(self):
        record_id, source, row = self._mock_contract_row()
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.jsonl"
            state = Path(directory) / "state.json"
            results.write_text(json.dumps(row) + "\n")
            state.write_bytes(b"")
            _, latest, _ = runner.load_history(results)
            rebuilt = runner.rebuild_resume_state(
                state, results, [record_id], latest, {record_id: source},
                self.prompt, self.config, "pilot_50",
            )
            self.assertTrue(rebuilt["rebuilt_from_results"])
            self.assertEqual(rebuilt["completed"], [record_id])
            self.assertEqual(rebuilt["pending"], [])

    def test_35_corrupt_state_rebuilds_from_results(self):
        record_id, source, row = self._mock_contract_row()
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.jsonl"
            state = Path(directory) / "state.json"
            results.write_text(json.dumps(row) + "\n")
            state.write_text("{broken")
            _, latest, _ = runner.load_history(results)
            rebuilt = runner.rebuild_resume_state(
                state, results, [record_id], latest, {record_id: source},
                self.prompt, self.config, "pilot_50",
            )
            self.assertEqual(rebuilt["counts"]["completed"], 1)
            self.assertEqual(rebuilt["counts"]["pending"], 0)

    def test_36_successful_id_never_pending_when_state_empty(self):
        record_id, source, row = self._mock_contract_row()
        self.assertTrue(runner.active_success(row, source, self.prompt, self.config))
        state = runner.state_payload(
            "REBUILT", [record_id], {record_id: row}, {record_id: source},
            self.prompt, self.config, Path("/nonexistent/results"), "pilot_50", None, None, True,
        )
        self.assertNotIn(record_id, state["pending"])

    def test_37_state_contains_required_hashes_and_lists(self):
        record_id, source, row = self._mock_contract_row()
        state = runner.state_payload(
            "READY", [record_id], {record_id: row}, {record_id: source},
            self.prompt, self.config, Path("/nonexistent/results"), "pilot_50", None, None,
        )
        for key in ("completed", "failed", "pending", "last_update", "prompt_hash", "schema_hash", "run_config_hash", "results_hash"):
            self.assertIn(key, state)

    def test_38_formal_batch_outside_cycle_rejected(self):
        args = argparse.Namespace(
            confirm_formal_batch=True,
            formal_batch_index=18,
            cycle_id="cycle_01",
            cycle_start_batch=0,
            artifact_root=str(V31),
        )
        with mock.patch.object(runner, "acceptance_pass", return_value=True), mock.patch.object(
            runner, "latest_acceptance_revision", return_value={
                "authorized_formal_batch_indices": list(range(18))
            }
        ):
            with self.assertRaisesRegex(RuntimeError, "batch_18_and_later_not_authorized"):
                runner.formal_selection(args, self.stable_ids, self.config)

    def test_39_artifact_hash_manifest_exists(self):
        value = json.loads((V31 / "v3_1_artifact_hashes.json").read_text())
        self.assertEqual(value["status"], "PASS")
        self.assertIn("prompt_v3_1_manifest.json", value["files"])


    def test_40_audit_only_does_not_remove_negated_concept(self):
        raw = json.dumps({"concepts": ["pneumothorax", "absent lung markings"]})
        evaluated = quality.evaluate_response(raw, "Pneumothorax with absent lung markings.", "xray", "stop")
        self.assertEqual(evaluated["raw_model_concepts"], ["pneumothorax", "absent lung markings"])
        self.assertIn("negated_concept", evaluated["audit_flags"])

    def test_41_high_low_conflict_is_flagged_not_rewritten(self):
        values = ["high-grade tumor", "low-grade tumor", "breast"]
        audited = quality.audit_concepts(values, "Conflicting caption", "microscopy")
        self.assertEqual(values, ["high-grade tumor", "low-grade tumor", "breast"])
        self.assertIn("mutually_exclusive_diagnosis", audited["audit_flags"])

    def test_42_special_audit_cases(self):
        cases = [
            (["neuroendocrine tumor", "grade 2 neuroendocrine tumor"], "parent_child_duplicate"),
            (["breast cancer", "high-grade malignancy"], "parent_child_duplicate"),
            (["nerve sheath tumor", "neurofibroma", "granuloma"], "mutually_exclusive_differential"),
            (["villous adenoma", "tubular adenoma"], "mutually_exclusive_differential"),
            (["coronal section", "positron emission tomography"], "geometric_view_artifact"),
            (["high-grade malignant tumor", "well-differentiated tumor"], "mutually_exclusive_diagnosis"),
        ]
        for concepts, expected in cases:
            with self.subTest(concepts=concepts):
                self.assertIn(expected, quality.audit_concepts(concepts)["audit_flags"])


class V31SDKSerializationTest(unittest.IsolatedAsyncioTestCase):
    async def test_40_sdk_serialization(self):
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "mock",
                "object": "chat.completion",
                "created": 0,
                "model": "k3",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"concepts":["pneumothorax"]}'},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            })

        client = AsyncOpenAI(
            api_key="mock-not-a-real-key",
            base_url=runner.BASE_URL,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        prompt, schema, config, _ = runner.load_contract(V31)
        sources, _ = runner.load_sources(V31, config)
        response = await client.chat.completions.create(
            **runner.request_kwargs(next(iter(sources.values())), prompt, schema)
        )
        await client.close()
        self.assertEqual(response.model, "k3")
        self.assertEqual(captured["reasoning_effort"], "none")
        self.assertEqual(captured["max_completion_tokens"], 256)
        self.assertEqual(captured["temperature"], 0.6)
        for key in ("top_p", "presence_penalty", "frequency_penalty", "thinking"):
            self.assertNotIn(key, captured)


if __name__ == "__main__":
    unittest.main()



