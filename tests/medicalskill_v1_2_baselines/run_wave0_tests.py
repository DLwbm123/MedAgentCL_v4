#!/usr/bin/env python3
"""Dependency-free Wave-0 unit tests for the formal baseline infrastructure."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, LlamaForCausalLM

from med_prism.baselines.octopus import (
    attach_historical_stage2_extras,
    canonical_base_weight_name,
    historical_extra_runtime_stats,
    inspect_octopus_stage2_serialization,
    octopus_layer_penalty,
    stage2_inference_stack,
    validate_gradient_manifest,
)
from scripts.medicalskill_v1_2_baselines.baseline_harness import (
    HistoricalMemory,
    TASK_ORDER,
    atomic_json,
    build_run_manifest,
    inspect_adapter,
    sha256_file,
    validate_dataset_lock,
    validate_runtime_record,
)
from scripts.medicalskill_v1_2_baselines.evaluate_cumulative_lora_v1_2 import (
    cumulative_adapter_manifest,
)
from scripts.medicalskill_v1_2_octopus.octopus_control import write_stage2_initialization

ROOT = Path("/root/MedAgentCL_v4")


def make_dataset(root: Path) -> Path:
    (root / "audits").mkdir(parents=True)
    (root / "manifests").mkdir()
    atomic_json(root / "audits" / "acceptance_matrix.json", {"status": "PASS"})
    selected = root / "manifests" / "selected_ids.jsonl"
    selected.write_text(
        "".join(json.dumps({"id": index}) + "\n" for index in range(10)),
        encoding="utf-8",
    )
    tasks = []
    for task_id, name in enumerate(TASK_ORDER, 1):
        directory = root / name
        directory.mkdir()
        splits = {}
        for split in ("train", "test"):
            path = directory / f"{split}.jsonl"
            path.write_text(json.dumps({"id": f"{task_id}-{split}"}) + "\n")
            splits[split] = {
                "file": f"{name}/{split}.jsonl",
                "records": 1,
                "sha256": sha256_file(path),
            }
        tasks.append({"task_id": task_id, "task_name": name, "splits": splits})
    atomic_json(
        root / "manifests" / "dataset_lock.json",
        {
            "dataset_version": "MedicalSkill-CL-v1.2-lite-10k1k",
            "seed": 42,
            "selected_id_count": 10,
            "selected_ids_file": "manifests/selected_ids.jsonl",
            "selected_ids_sha256": sha256_file(selected),
            "tasks": tasks,
        },
    )
    return root


def make_adapter(root: Path, value: float = 1.0) -> Path:
    root.mkdir(parents=True)
    atomic_json(
        root / "adapter_config.json",
        {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"]},
    )
    save_file(
        {
            "base_model.model.model.layer.q_proj.lora_A.weight": torch.full(
                (16, 4), value
            ),
            "base_model.model.model.layer.q_proj.lora_B.weight": torch.full(
                (4, 16), value
            ),
        },
        root / "adapter_model.safetensors",
    )
    return root


class Wave0Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="wave0_test_")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_01_dataset_lock(self):
        value = validate_dataset_lock(make_dataset(self.root / "dataset"))
        self.assertEqual(value["status"], "PASS")
        self.assertEqual(value["task_order"], TASK_ORDER)
        self.assertEqual(value["selected_id_count"], 10)

    def test_02_dataset_lock_detects_mutation(self):
        data = make_dataset(self.root / "dataset")
        with (data / TASK_ORDER[0] / "train.jsonl").open("a") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(RuntimeError, "Record-count mismatch"):
            validate_dataset_lock(data)

    def test_03_actual_parameter_count(self):
        value = inspect_adapter(make_adapter(self.root / "adapter"))
        self.assertEqual(value["adapter_parameters"], 128)
        self.assertEqual(value["adapter_tensor_count"], 2)
        self.assertEqual(value["r"], 16)
        self.assertEqual(value["target_modules"], ["q_proj", "v_proj"])

    def test_04_historical_memory_multilabel(self):
        value = HistoricalMemory(
            statistics_or_masks=True,
            historical_parameters=True,
            notes="gradient responses",
        ).manifest()
        self.assertFalse(value["raw_examples"])
        self.assertTrue(value["statistics_or_masks"])
        self.assertTrue(value["historical_parameters"])
        self.assertFalse(value["formally_privacy_preserving"])

    def test_05_run_manifest_contract(self):
        value = build_run_manifest(
            method="unit",
            method_family="unit",
            data_root=make_dataset(self.root / "dataset"),
            output_root=self.root / "output",
            capacity_policy={"class": "task_growing", "rank_per_task": 16},
            historical_memory=HistoricalMemory(historical_parameters=True),
            method_config={"target_modules": ["q_proj", "v_proj"]},
        )
        self.assertEqual(value["status"], "PASS")
        self.assertEqual(value["capacity_policy"]["rank_per_task"], 16)
        self.assertEqual(len(value["immutable_config_sha256"]), 64)

    def test_06_runtime_manifest(self):
        record = self.root / "runtime.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/medicalskill_v1_2_baselines/run_timed.py"),
                "--record",
                str(record),
                "--category",
                "unit",
                "--",
                sys.executable,
                "-c",
                "pass",
            ],
            cwd=ROOT,
            check=True,
        )
        value = validate_runtime_record(record, "unit")
        self.assertEqual(value["status"], "PASS")
        self.assertEqual(value["command_exit_code"], 0)
        self.assertGreaterEqual(value["wall_clock_seconds"], 0)
        self.assertEqual(value["gpu_count"], 0)

    def test_07_original_trace_equivalence(self):
        torch.manual_seed(42)
        a = torch.randn(3, 5)
        b = torch.randn(7, 3)
        gradient = torch.randn(7, 5)
        gradients_a = a @ gradient.T
        expected = torch.trace(
            (b / (torch.norm(b) + 1e-8))
            @ (gradients_a / (torch.norm(gradients_a) + 1e-8))
        ).abs()
        self.assertTrue(
            torch.allclose(octopus_layer_penalty(a, b, gradient), expected)
        )

    def test_07b_original_orthogonal_term_detaches_a_gradient(self):
        torch.manual_seed(43)
        a = torch.randn(3, 5, requires_grad=True)
        b = torch.randn(7, 3, requires_grad=True)
        gradient = torch.randn(7, 5)
        penalty = octopus_layer_penalty(a, b, gradient)
        penalty.backward()
        self.assertIsNone(a.grad)
        self.assertIsNotNone(b.grad)
        self.assertGreater(float(torch.linalg.vector_norm(b.grad)), 0.0)


    def test_07c_official_frozen_extras_forward_and_serialization(self):
        def tiny_lora():
            base = LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                )
            )
            return get_peft_model(
                base,
                LoraConfig(
                    task_type="CAUSAL_LM",
                    r=2,
                    lora_alpha=4,
                    lora_dropout=0.0,
                    bias="none",
                    target_modules=["q_proj", "v_proj"],
                ),
            )

        historical = tiny_lora()
        for name, parameter in historical.named_parameters():
            if ".lora_A." in name:
                parameter.data.fill_(0.02)
            elif ".lora_B." in name:
                parameter.data.fill_(0.05)
        historical_path = self.root / "historical_stage2"
        historical.save_pretrained(historical_path)

        current = tiny_lora().eval()
        input_ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            without_history = current(input_ids=input_ids).logits
        metadata = attach_historical_stage2_extras(current, [historical_path])
        with torch.no_grad():
            with_history = current(input_ids=input_ids).logits
        self.assertFalse(torch.equal(without_history, with_history))
        self.assertEqual(metadata["historical_adapter_count"], 1)
        self.assertEqual(metadata["trainable_extra_tensor_count"], 0)
        stats = historical_extra_runtime_stats(current)
        self.assertTrue(stats["all_historical_extras_active_in_forward"])
        self.assertEqual(stats["trainable_extra_tensor_count"], 0)

        saved = self.root / "current_stage2_saved"
        current.save_pretrained(saved)
        saved_state = load_file(saved / "adapter_model.safetensors")
        self.assertTrue(all("extras_A" not in key for key in saved_state))
        self.assertTrue(all("extras_B" not in key for key in saved_state))
        serialization = inspect_octopus_stage2_serialization(saved)
        self.assertTrue(serialization["contains_only_current_default_lora"])
        self.assertEqual(serialization["serialized_historical_extra_tensor_count"], 0)

    def test_08_peft_name_mapping(self):
        value = canonical_base_weight_name(
            "base_model.model.model.language_model.layers.0."
            "self_attn.q_proj.lora_B.default.weight"
        )
        self.assertEqual(
            value, "model.language_model.layers.0.self_attn.q_proj.weight"
        )

    def test_09_same_task_stage2_initialization(self):
        stage1 = make_adapter(self.root / "task2_stage1")
        value = write_stage2_initialization(
            self.root / "init.json",
            stage=2,
            stage1_checkpoint=stage1,
            mode="peft_from_pretrained_trainable_same_task_stage1",
        )
        self.assertEqual(value["source_task"], 2)
        self.assertEqual(value["source_role"], "same_task_stage1")
        self.assertEqual(
            value["source_adapter_weights_sha256"],
            inspect_adapter(stage1)["adapter_weights_sha256"],
        )

    def test_10_task1_stage2_retraining_contract(self):
        stage1 = make_adapter(self.root / "task1_stage1")
        initialization = write_stage2_initialization(
            self.root / "init.json",
            stage=1,
            stage1_checkpoint=stage1,
            mode="peft_from_pretrained_trainable_same_task_stage1",
        )
        stage2 = make_adapter(self.root / "task1_stage2")
        weights = stage2 / "adapter_model.safetensors"
        state = load_file(weights)
        first = sorted(state)[0]
        state[first] = state[first] + 1
        save_file(state, weights)
        self.assertEqual(initialization["source_task"], 1)
        self.assertEqual(initialization["source_role"], "same_task_stage1")
        self.assertNotEqual(
            inspect_adapter(stage1)["adapter_weights_sha256"],
            inspect_adapter(stage2)["adapter_weights_sha256"],
        )

    def test_11_gradient_artifact_hash_isolation(self):
        adapter = make_adapter(self.root / "stage2")
        info = inspect_adapter(adapter)
        dataset = self.root / "current.jsonl"
        dataset.write_text('{"id":"current"}\n')
        artifact = self.root / "gradients.safetensors"
        save_file({"model.layer.q_proj.weight": torch.ones(4, 4)}, artifact)
        manifest = self.root / "gradient_manifest.json"
        atomic_json(
            manifest,
            {
                "status": "PASS",
                "task_id": 2,
                "source_current_task_dataset_sha256": sha256_file(dataset),
                "target_modules": ["q_proj", "v_proj"],
                "gradient_artifact": str(artifact),
                "gradient_artifact_sha256": sha256_file(artifact),
                "historical_prefix_length": 1,
                "current_task_data_only": True,
                "historical_stage2_prefix": [
                    {
                        "task": 1,
                        "role": "stage2",
                        "checkpoint": str(adapter),
                        "adapter_weights_sha256": info["adapter_weights_sha256"],
                    }
                ],
            },
        )
        self.assertTrue(
            validate_gradient_manifest(
                manifest,
                current_dataset_sha256=sha256_file(dataset),
                target_modules=["q_proj", "v_proj"],
            )["current_task_data_only"]
        )
        artifact.write_bytes(artifact.read_bytes() + b"tamper")
        with self.assertRaisesRegex(RuntimeError, "artifact_hash"):
            validate_gradient_manifest(manifest)

    def test_12_historical_stage2_stack(self):
        completions = []
        for task in (1, 2):
            stage2 = make_adapter(
                self.root / f"task{task}_stage2", float(task)
            )
            completion = self.root / f"completion{task}.json"
            atomic_json(
                completion,
                {
                    "status": "PASS",
                    "stage": task,
                    "stage2_checkpoint": str(stage2),
                    "stage1_checkpoint": str(self.root / f"task{task}_stage1"),
                },
            )
            completions.append(completion)
        stack = stage2_inference_stack(completions)
        self.assertEqual([item["task"] for item in stack], [1, 2])
        self.assertEqual({item["role"] for item in stack}, {"stage2"})

    def test_13_evaluator_handoff_stage2_only(self):
        adapters = [
            make_adapter(self.root / "task1_stage2", 1.0),
            make_adapter(self.root / "task2_stage2", 2.0),
        ]
        value = cumulative_adapter_manifest(
            adapters, "octopus_qwen3vl_r16", expected_rank=16
        )
        self.assertEqual(value["adapter_load_count"], 2)
        self.assertFalse(value["stage1_adapters_active"])
        self.assertFalse(value["oracle_task_id"])
        self.assertEqual(
            [item["role"] for item in value["components"]],
            ["stage2", "stage2"],
        )
        self.assertEqual(value["active_adapter_parameters"], 256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
