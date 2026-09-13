import copy
import tempfile
import unittest
from pathlib import Path

import torch

from med_prism.adapters.injection import collect_language_targets
from med_prism.checkpoint.rank1_checkpoint import (
    MANIFEST_NAME,
    WEIGHTS_NAME,
    load_rank1_checkpoint,
    save_rank1_checkpoint,
    validate_manifest,
)
from med_prism.config import MODEL_REVISION, Rank1BankConfig
from tests.phase4.helpers import ToyQwen3VL, make_bank


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.model = make_bank(((1, 4, 4.0, True),))
        generator = torch.Generator().manual_seed(19)
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if ".experts." in name:
                    parameter.copy_(
                        torch.randn(
                            parameter.shape,
                            generator=generator,
                            dtype=parameter.dtype,
                        )
                    )
        self.config = Rank1BankConfig(
            current_task_id=1,
            experts_per_task=4,
            alpha=4.0,
            dropout=0.0,
            output_dir=str(self.output),
        )
        self.model.med_prism_config = self.config
        self.manifest = save_rank1_checkpoint(
            self.model, self.output, config=self.config
        )
        self.target_hash = collect_language_targets(self.model).sha256

    def tearDown(self):
        self.temp.cleanup()

    def test_save_contains_only_explicit_rank1_tensors(self):
        self.assertTrue((self.output / MANIFEST_NAME).is_file())
        self.assertTrue((self.output / WEIGHTS_NAME).is_file())
        self.assertEqual(self.manifest["wrapper_count"], 72)
        self.assertEqual(self.manifest["expert_count"], 72 * 4)
        self.assertEqual(self.manifest["expert_tensor_count"], 72 * 4 * 2)
        self.assertEqual(self.manifest["merge_state"], "unmerged")
        self.assertTrue(
            all(
                item["side"] in {"A", "B"}
                and ".experts.task_" in item["key"]
                for item in self.manifest["tensor_schema"]
            )
        )
        self.assertFalse(
            any("lora_A" in item["key"] or "lora_B" in item["key"]
                for item in self.manifest["tensor_schema"])
        )

    def test_save_reload_round_trip_is_exact(self):
        reloaded = ToyQwen3VL()
        loaded = load_rank1_checkpoint(
            reloaded,
            self.output / MANIFEST_NAME,
            expected_revision=MODEL_REVISION,
            trainable_task_id=None,
        )
        self.assertEqual(loaded["safetensors_sha256"], self.manifest["safetensors_sha256"])
        source = {
            name: parameter.detach()
            for name, parameter in self.model.named_parameters()
            if ".experts." in name
        }
        target = {
            name: parameter.detach()
            for name, parameter in reloaded.named_parameters()
            if ".experts." in name
        }
        self.assertEqual(set(source), set(target))
        for name in source:
            torch.testing.assert_close(source[name], target[name], rtol=0, atol=0)
            self.assertFalse(dict(reloaded.named_parameters())[name].requires_grad)

    def test_loader_requires_explicit_manifest_file(self):
        with self.assertRaises(FileNotFoundError):
            load_rank1_checkpoint(ToyQwen3VL(), self.output)

    def assert_manifest_rejected(self, mutate, message):
        candidate = copy.deepcopy(self.manifest)
        mutate(candidate)
        with self.assertRaisesRegex(ValueError, message):
            validate_manifest(
                candidate,
                expected_revision=MODEL_REVISION,
                expected_target_hash=self.target_hash,
            )

    def test_rejects_revision_target_method_and_merge_mismatch(self):
        cases = [
            (lambda m: m.update(immutable_revision="wrong"), "revision"),
            (lambda m: m.update(target_module_hash="wrong"), "Target module"),
            (lambda m: m.update(method="lora"), "not a Med-PRISM"),
            (lambda m: m.update(merge_state="merged"), "Merged"),
        ]
        for mutate, message in cases:
            with self.subTest(message=message):
                self.assert_manifest_rejected(mutate, message)

    def test_rejects_incomplete_duplicate_and_bad_scaling_schema(self):
        def duplicate_task(manifest):
            manifest["task_ids"].append(1)

        def incomplete(manifest):
            manifest["tensor_schema"].pop()

        def duplicate_tensor(manifest):
            manifest["tensor_schema"][1] = copy.deepcopy(manifest["tensor_schema"][0])

        def bad_scaling(manifest):
            manifest["task_banks"][0]["scaling"] = 9.0

        cases = [
            (duplicate_task, "Duplicate task"),
            (incomplete, "incomplete"),
            (duplicate_tensor, "Duplicate task/expert"),
            (bad_scaling, "scaling"),
        ]
        for mutate, message in cases:
            with self.subTest(message=message):
                self.assert_manifest_rejected(mutate, message)


if __name__ == "__main__":
    unittest.main()
