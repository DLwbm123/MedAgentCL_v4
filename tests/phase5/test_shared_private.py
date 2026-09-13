from __future__ import annotations

import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from med_prism.adapters.injection import (
    add_task_bank,
    collect_language_targets,
    freeze_shared_private_except,
    inject_shared_private_wrappers,
    iter_shared_private_wrappers,
)
from med_prism.checkpoint.shared_private_checkpoint import (
    compose_shared_private,
    load_private_component,
    load_shared_component,
    save_shared_private_components,
)
from med_prism.config import SharedPrivateConfig
from med_prism.projection.orth_losses import compute_rank1_orth_loss
from tests.phase4.helpers import ToyQwen3VL


def make_model(tasks=(1,), current=1):
    model = ToyQwen3VL()
    model.requires_grad_(False)
    inject_shared_private_wrappers(
        model,
        shared_rank=4,
        shared_alpha=4.0,
        shared_lr_scale=1.0,
        dropout=0.0,
        shared_trainable=True,
    )
    for task_id in tasks:
        add_task_bank(
            model,
            task_id=task_id,
            experts_per_task=4,
            alpha=4.0,
            trainable=task_id == current,
        )
    freeze_shared_private_except(model, current)
    for _, wrapper in iter_shared_private_wrappers(model):
        wrapper.set_active_tasks(tasks)
    return model


class SharedPrivateTest(unittest.TestCase):
    def test_target_inventory_is_language_qv_only(self):
        model = make_model()
        inventory = collect_language_targets(model)
        self.assertEqual(
            (inventory.language, inventory.q_proj, inventory.v_proj), (72, 36, 36)
        )
        self.assertEqual((inventory.vision, inventory.merger), (0, 0))

    def test_optimizer_membership_contract(self):
        model = make_model(tasks=(1, 2), current=2)
        trainable = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(".shared." in name or "task_0002__" in name for name in trainable)
        )
        self.assertFalse(any("task_0001__" in name for name in trainable))

    def test_old_private_forward_is_active_and_order_is_canonical(self):
        model = make_model(tasks=(2, 1), current=2)
        x = torch.ones(2, 4)
        for _, wrapper in iter_shared_private_wrappers(model):
            with torch.no_grad():
                wrapper.shared.B.fill_(0.01)
                for task_id in (1, 2):
                    for expert in wrapper.task_experts(task_id):
                        expert.A.fill_(0.02 * task_id)
                        expert.B.fill_(0.03 * task_id)
            wrapper.set_active_tasks((1,))
            old_only = wrapper(x)
            wrapper.set_active_tasks((1, 2))
            cumulative = wrapper(x)
            self.assertGreater(float((cumulative - old_only).abs().max()), 0.0)
            break

    def test_private_scaling_stays_alpha_over_experts(self):
        model = make_model(tasks=(1,), current=1)
        wrapper = next(iter(iter_shared_private_wrappers(model)))[1]
        before = [float(item.scaling) for item in wrapper.task_experts(1)]
        wrapper.add_task(2, experts_per_task=16, alpha=16.0, trainable=True)
        after = [float(item.scaling) for item in wrapper.task_experts(1)]
        self.assertEqual(before, after)
        self.assertEqual(before, [1.0] * 4)

    def test_orthogonal_gradient_reaches_only_current_private(self):
        model = make_model(tasks=(1, 2), current=2)
        for _, wrapper in iter_shared_private_wrappers(model):
            for task_id in (1, 2):
                for expert in wrapper.task_experts(task_id):
                    with torch.no_grad():
                        expert.A.fill_(0.1 + task_id * 0.01)
                        expert.B.fill_(0.2 + task_id * 0.01)
        orth = compute_rank1_orth_loss(model, current_task_id=2, loss_type="rms")
        orth.loss.backward()
        old = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "task_0001__" in name
        ]
        current = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "task_0002__" in name
        ]
        self.assertTrue(all(value is None for value in old))
        self.assertTrue(
            any(value is not None and value.abs().sum() > 0 for value in current)
        )

    def test_single_task_orthogonal_loss_is_zero(self):
        model = make_model()
        orth = compute_rank1_orth_loss(model, current_task_id=1, loss_type="rms")
        self.assertEqual(float(orth.loss), 0.0)

    def test_component_round_trip_and_deduplication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = make_model()
            config = SharedPrivateConfig(
                current_task_id=1,
                shared_rank=4,
                shared_alpha=4.0,
                shared_lr_scale=1.0,
                private_experts_per_task=4,
                private_alpha=4.0,
                shared_drift_lambda=0.0,
                shared_output_dir=str(root / "shared"),
                private_output_dir=str(root / "private"),
                output_dir=str(root / "run"),
            )
            save_shared_private_components(model, config)
            fresh = ToyQwen3VL()
            active = compose_shared_private(
                fresh,
                shared_manifest=root / "shared/shared_manifest.json",
                private_manifests=[root / "private/private_manifest.json"],
            )
            self.assertEqual(active["shared_load_count"], 1)
            self.assertEqual(active["private_task_ids"], [1])
            with self.assertRaisesRegex(ValueError, "Duplicate shared"):
                load_shared_component(fresh, root / "shared/shared_manifest.json")
            with self.assertRaisesRegex(ValueError, "Duplicate private"):
                load_private_component(
                    fresh,
                    root / "private/private_manifest.json",
                    trainable=False,
                )

    def test_bad_revision_and_schema_fail_fast(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = make_model()
            config = SharedPrivateConfig(
                current_task_id=1,
                shared_rank=4,
                shared_alpha=4.0,
                shared_lr_scale=1.0,
                private_experts_per_task=4,
                private_alpha=4.0,
                shared_output_dir=str(root / "shared"),
                private_output_dir=str(root / "private"),
            )
            save_shared_private_components(model, config)
            manifest_path = root / "shared/shared_manifest.json"
            payload = json.loads(manifest_path.read_text())
            payload["model_revision"] = "wrong"
            manifest_path.write_text(json.dumps(payload))
            fresh = ToyQwen3VL()
            inject_shared_private_wrappers(
                fresh,
                shared_rank=4,
                shared_alpha=4.0,
                shared_lr_scale=1.0,
                dropout=0.0,
                shared_trainable=False,
            )
            with self.assertRaisesRegex(ValueError, "revision"):
                load_shared_component(fresh, manifest_path)

    def test_registry_preserves_lora_rank1_and_shared_private(self):
        compatibility = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "phase3"
            / "callback_compat.py"
        )
        runpy.run_path(str(compatibility))
        from med_prism.swift_plugins import med_prism_rank1_plugin
        from med_prism.swift_plugins import med_prism_shared_private_plugin
        from swift.tuner_plugin import tuners_map

        self.assertIn("med_prism_rank1", tuners_map)
        self.assertIn("med_prism_shared_private", tuners_map)
        args = SimpleNamespace(tuner_type="lora", task_type="causal_lm")
        trainer = med_prism_shared_private_plugin.TrainerFactory.get_trainer_cls(args)
        self.assertIsNot(
            trainer, med_prism_shared_private_plugin.MedPrismSharedPrivateTrainer
        )
        self.assertIsNot(trainer, med_prism_rank1_plugin.MedPrismRank1Trainer)


if __name__ == "__main__":
    unittest.main()
