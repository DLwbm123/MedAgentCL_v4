#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import unittest
from pathlib import Path

ROOT = Path("/root/MedAgentCL_v4")
SCRIPTS = ROOT / "scripts/medicalskill_kimi_k3"
V1 = ROOT / "artifacts/medicalskill_cl_kimi_k3_pilot"
V2 = ROOT / "artifacts/medicalskill_cl_kimi_k3_v2_pilot"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_kimi_k3_v2", SCRIPTS / "prepare_kimi_k3_v2.py")
runner = load("run_kimi_k3_v2", SCRIPTS / "run_kimi_k3_v2_pilot.py")
auditor = load("audit_kimi_k3_v2", SCRIPTS / "audit_kimi_k3_v2_pilot.py")


class KimiV2PipelineTest(unittest.TestCase):
    def test_v1_results_are_preserved(self):
        path = V1 / "pilot_results.jsonl"
        self.assertEqual(len(read_jsonl(path)), 2400)
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "9f95e3bc7b4234c5382b4a5b3f148bfbe5eda7340ed68e9d4340b2b7f094c6a2",
        )

    def test_waterfill_is_exact(self):
        counts = {
            "ct": 10001, "dermoscopy": 5989, "endoscopy": 549,
            "histopathology": 10341, "microscopy": 10605, "mri": 10916,
            "pet": 10984, "xray": 10347,
        }
        quotas, level = prepare.waterfill_quotas(counts, 35000)
        self.assertEqual(sum(quotas.values()), 35000)
        self.assertEqual(quotas["endoscopy"], 549)
        self.assertAlmostEqual(level, 4921.571428571428)

    def test_built_manifests_have_hard_limits(self):
        summary = json.loads((V2 / "concept_35k_candidate_summary.json").read_text())
        pilot = json.loads((V2 / "v2_pilot_selection_manifest.json").read_text())
        candidate_ids = {row["stable_record_id"] for row in read_jsonl(V2 / "concept_35k_candidate_manifest.jsonl")}
        reserve_ids = {row["stable_record_id"] for row in read_jsonl(V2 / "concept_35k_reserve_manifest.jsonl")}
        self.assertEqual(len(candidate_ids), 35000)
        self.assertEqual(len(reserve_ids), 5000)
        self.assertFalse(candidate_ids & reserve_ids)
        self.assertEqual(len(pilot["stable_record_ids"]), 500)
        self.assertEqual(len(set(pilot["stable_record_ids"])), 500)
        self.assertTrue(set(pilot["stable_record_ids"]) <= candidate_ids)
        self.assertEqual(summary["actual_total"], 35000)
        self.assertFalse((V2 / "full_generation_results.jsonl").exists())

    def test_prompt_contract(self):
        prompt = json.loads((V2 / "prompt_v2_manifest.json").read_text())
        schema = json.loads((V2 / "kimi_prompt_schema_v2.json").read_text())
        normalization = json.loads((V2 / "normalization_contract_v2.json").read_text())
        contract = {
            key: prompt[key]
            for key in (
                "prompt_version", "system_prompt", "user_template", "model",
                "reasoning_effort", "temperature", "max_completion_tokens",
                "normalization_contract",
            )
        }
        self.assertEqual(prompt["temperature"], 1.0)
        self.assertEqual(prompt["max_completion_tokens"], 1024)
        self.assertIs(prompt["top_p_sent"], False)
        self.assertEqual(runner.sha_text(runner.canonical_json(contract)), prompt["prompt_hash"])
        self.assertEqual(runner.sha_text(runner.canonical_json(schema)), prompt["schema_hash"])
        self.assertEqual(runner.sha_text(runner.canonical_json(normalization)), prompt["normalization_hash"])

    def test_wire_request_has_fixed_parameters(self):
        tree = ast.parse((SCRIPTS / "run_kimi_k3_v2_pilot.py").read_text())
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertEqual(keywords["temperature"].value, 1.0)
        self.assertEqual(keywords["max_completion_tokens"].id, "MAX_COMPLETION_TOKENS")
        self.assertNotIn("top_p", keywords)

    def test_normalization_only_exact_deduplicates(self):
        raw = '{"concepts":["Pulmonary nodule"," pulmonary   nodule ","Left lung"]}'
        self.assertEqual(runner.validate_content(raw), ["pulmonary nodule", "left lung"])
        with self.assertRaises(ValueError):
            runner.validate_content('{"concepts":[]}')

    def test_endpoint_and_credentials(self):
        endpoint = runner.resolve_base_url({})
        self.assertEqual(endpoint["base_url"], "https://api.kimi.com/coding/v1")
        with self.assertRaises(RuntimeError):
            runner.resolve_base_url({"KIMI_BASE_URL": "https://api.moonshot.cn/v1"})
        self.assertEqual(runner.resolve_credential({"KIMI_API_KEY": "x"}), ("x", "KIMI_API_KEY"))
        with self.assertRaises(RuntimeError):
            runner.resolve_credential({"KIMI_API_KEY": "x", "MOONSHOT_API_KEY": "y"})

    def test_risk_audit_patterns_are_live(self):
        row = {
            "stable_record_id": "x",
            "parsed_concepts": [
                "uncertain: disease process",
                "benign or malignant lesion",
                "adjacent tissue involvement",
            ],
        }
        counts, _ = auditor.risk_counts([row])
        self.assertEqual(counts["uncertain_concept"], 1)
        self.assertEqual(counts["benign_or_malignant"], 1)
        self.assertEqual(counts["pathological_or_disease_process"], 1)
        self.assertEqual(counts["adjacent_or_proximity"], 1)
        self.assertEqual(counts["involvement"], 1)

    def test_budget_and_stop_contract_are_present(self):
        source = (SCRIPTS / "run_kimi_k3_v2_pilot.py").read_text()
        self.assertEqual(runner.PILOT_LIMIT, 500)
        self.assertEqual(runner.MAX_COMPLETION_TOKENS, 1024)
        self.assertIn("args.max_api_calls > 650", source)
        self.assertIn("if stop.is_set():\n                return", source)
        self.assertIn("attempt_ordinal < 2", source)
        self.assertNotIn("contract_finish_reason_length\"}:", source)


if __name__ == "__main__":
    unittest.main()

