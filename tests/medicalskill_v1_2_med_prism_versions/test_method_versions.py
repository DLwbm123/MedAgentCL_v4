from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from med_prism.adapters.injection import (
    add_task_bank,
    freeze_shared_private_except,
    inject_shared_private_wrappers,
    iter_shared_private_wrappers,
)
from med_prism.adapters.shared_private import SharedAdapter
from med_prism.config import SharedPrivateConfig
from med_prism.evaluation.cl import component_task_ids
from med_prism.optimization.shared_private import (
    build_shared_private_optimizer_groups,
    optimizer_component_initial_lrs,
    optimizer_component_lrs,
)
from med_prism.projection.orth_losses import (
    compute_rank1_key_isolation_loss,
    compute_rank1_orth_loss,
    rank1_key_cosines,
    rank1_overlap_squared,
)
from med_prism.projection.shared_private import (
    compute_effective_shared_drift,
    compute_shared_drift_loss,
)
import scripts.phase3.callback_compat  # noqa: F401
from med_prism.swift_plugins.shared_private_trainer import (
    MedPrismSharedPrivateTrainer,
)
from tests.phase4.helpers import ToyQwen3VL


def make_model(tasks=(1,), current=1, experts=16):
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
            experts_per_task=experts,
            alpha=float(experts),
            trainable=task_id == current,
        )
    freeze_shared_private_except(model, current)
    for _, wrapper in iter_shared_private_wrappers(model):
        wrapper.set_active_tasks(tasks)
    return model


def config(tmp: Path, version: str):
    fields = {
        "current_task_id": 1,
        "shared_rank": 4,
        "shared_alpha": 4.0,
        "private_experts_per_task": 4,
        "private_alpha": 4.0,
        "shared_output_dir": str(tmp / "shared"),
        "private_output_dir": str(tmp / "private"),
    }
    if version == "1.1":
        fields.update(
            method_version="1.1",
            method_variant="shared_fix",
            shared_optimizer_mode="separate_lr_param_groups",
            shared_lr=1e-5,
            private_lr=1e-4,
            shared_gradient_hook_enabled=False,
            shared_drift_mode="effective_BA",
        )
    elif version == "1.2":
        fields.update(
            method_version="1.2",
            method_variant="shared_fix_plus_key_isolation",
            shared_optimizer_mode="separate_lr_param_groups",
            shared_lr=1e-5,
            private_lr=1e-4,
            shared_gradient_hook_enabled=False,
            shared_drift_mode="effective_BA",
            key_isolation_enabled=True,
            key_loss_mode="rms_cosine_A",
            key_lambda=0.1,
            key_history_scope="all_previous_tasks_same_layer",
        )
    return SharedPrivateConfig(**fields)


class LegacyRegressionTest(unittest.TestCase):
    def test_default_config_is_v1_0_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            value = SharedPrivateConfig(
                shared_output_dir=f"{temp}/shared",
                private_output_dir=f"{temp}/private",
            )
            value.validate()
            self.assertEqual(value.method_version, "1.0")
            self.assertEqual(value.shared_optimizer_mode, "legacy_gradient_hook")
            self.assertEqual(value.shared_drift_mode, "factor")
            self.assertFalse(value.key_isolation_enabled)

    def test_legacy_gradient_hook_scales_shared_gradients(self):
        torch.manual_seed(3)
        kwargs = dict(
            in_features=4,
            out_features=3,
            rank=2,
            alpha=2.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
            trainable=True,
        )
        reference = SharedAdapter(lr_scale=1.0, **kwargs)
        hooked = SharedAdapter(lr_scale=0.1, **kwargs)
        hooked.load_state_dict(reference.state_dict())
        with torch.no_grad():
            reference.B.fill_(0.2)
            hooked.B.copy_(reference.B)
        x = torch.randn(2, 4)
        reference(x, torch.nn.Identity()).sum().backward()
        hooked(x, torch.nn.Identity()).sum().backward()
        self.assertFalse(reference.gradient_hook_enabled)
        self.assertTrue(hooked.gradient_hook_enabled)
        self.assertTrue(torch.allclose(hooked.A.grad, reference.A.grad * 0.1))
        self.assertTrue(torch.allclose(hooked.B.grad, reference.B.grad * 0.1))

    def test_legacy_factor_drift_matches_original_formula(self):
        model = make_model(experts=4)
        wrappers = list(iter_shared_private_wrappers(model))
        denominator = sum(
            float(t.float().pow(2).sum())
            for _, wrapper in wrappers
            for t in (wrapper.shared.anchor_A, wrapper.shared.anchor_B)
        )
        wrappers[0][1].shared.B.data[0, 0] = 2.0
        observed = compute_shared_drift_loss(model)
        self.assertAlmostEqual(float(observed), 4.0 / max(denominator, 1e-8), places=6)


class SharedFixTest(unittest.TestCase):
    def test_v11_config_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            value = config(Path(temp), "1.1")
            value.validate()
            self.assertFalse(value.shared_gradient_hook_enabled)
            self.assertEqual(value.shared_drift_mode, "effective_BA")
            self.assertFalse(value.key_isolation_enabled)

    def test_effective_low_rank_drift_matches_materialized_ba(self):
        torch.manual_seed(7)
        model = make_model(tasks=(1, 2), current=2, experts=4)
        explicit_numerator = torch.tensor(0.0)
        explicit_denominator = torch.tensor(0.0)
        for _, wrapper in iter_shared_private_wrappers(model):
            shared = wrapper.shared
            with torch.no_grad():
                shared.anchor_A.copy_(torch.randn_like(shared.anchor_A))
                shared.anchor_B.copy_(torch.randn_like(shared.anchor_B))
                shared.A.copy_(torch.randn_like(shared.A))
                shared.B.copy_(torch.randn_like(shared.B))
                shared.scaling.fill_(2.0)
            scale = shared.scaling.float()
            current = scale * shared.B.float() @ shared.A.float()
            reference = scale * shared.anchor_B.float() @ shared.anchor_A.float()
            explicit_numerator += (current - reference).pow(2).sum()
            explicit_denominator += reference.pow(2).sum()
        result = compute_effective_shared_drift(model, current_task_id=2)
        expected = explicit_numerator / explicit_denominator.clamp_min(1e-8)
        self.assertTrue(torch.allclose(result.loss, expected, rtol=1e-5, atol=1e-6))
        self.assertAlmostEqual(
            result.effective_delta_norm,
            float(explicit_numerator.sqrt()),
            places=4,
        )

    def test_task1_effective_drift_is_exact_differentiable_zero(self):
        model = make_model(experts=4)
        result = compute_effective_shared_drift(model, current_task_id=1)
        self.assertEqual(float(result.loss), 0.0)
        self.assertTrue(result.loss.requires_grad)
        result.loss.backward()
        shared = [p for n, p in model.named_parameters() if ".shared." in n]
        self.assertTrue(all(p.grad is not None and float(p.grad.abs().sum()) == 0.0 for p in shared))

    def test_effective_drift_gradient_ownership(self):
        model = make_model(tasks=(1, 2), current=2, experts=4)
        wrapper = next(iter(iter_shared_private_wrappers(model)))[1]
        with torch.no_grad():
            wrapper.shared.anchor_B.fill_(0.2)
            wrapper.shared.B.fill_(0.3)
        loss = compute_effective_shared_drift(model, current_task_id=2).loss
        loss.backward()
        self.assertTrue(any(p.grad is not None for n, p in model.named_parameters() if ".shared." in n))
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if ".experts." in n))
        self.assertFalse(wrapper.shared.anchor_A.requires_grad)
        self.assertFalse(wrapper.shared.anchor_B.requires_grad)

    def test_optimizer_groups_and_scheduler_preserve_ratio_and_resume(self):
        model = make_model(tasks=(1, 2), current=2, experts=4)
        decay = {name for name, p in model.named_parameters() if p.requires_grad}
        groups, audit = build_shared_private_optimizer_groups(
            model,
            current_task_id=2,
            decay_parameter_names=decay,
            weight_decay=0.1,
            shared_lr=1e-5,
            private_lr=1e-4,
        )
        optimizer = torch.optim.AdamW(groups)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 0.0 if step == 0 else 0.5
        )
        shared_lr, private_lr = optimizer_component_lrs(optimizer)
        self.assertEqual((shared_lr, private_lr), (0.0, 0.0))
        initial_shared_lr, initial_private_lr = optimizer_component_initial_lrs(
            optimizer
        )
        self.assertAlmostEqual(initial_shared_lr, 1e-5)
        self.assertAlmostEqual(initial_private_lr, 1e-4)
        self.assertAlmostEqual(initial_shared_lr / initial_private_lr, 0.1)
        optimizer.step()
        scheduler.step()
        shared_lr, private_lr = optimizer_component_lrs(optimizer)
        self.assertAlmostEqual(shared_lr / private_lr, 0.1)
        self.assertEqual(audit.shared_private_lr_ratio, 0.1)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        old_ids = {id(p) for n, p in model.named_parameters() if "task_0001__" in n}
        self.assertFalse(old_ids & set(ids))
        state = optimizer.state_dict()
        resumed = torch.optim.AdamW(groups)
        resumed.load_state_dict(state)
        resumed_shared, resumed_private = optimizer_component_lrs(resumed)
        self.assertAlmostEqual(resumed_shared / resumed_private, 0.1)

    def test_optimizer_audit_accepts_zero_warmup_lrs(self):
        model = make_model(current=1, experts=4)
        decay = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        groups, _ = build_shared_private_optimizer_groups(
            model,
            current_task_id=1,
            decay_parameter_names=decay,
            weight_decay=0.1,
            shared_lr=1e-5,
            private_lr=1e-4,
        )
        optimizer = torch.optim.AdamW(groups)
        torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 0.0 if step == 0 else 1.0
        )
        with tempfile.TemporaryDirectory() as temp:
            trainer = object.__new__(MedPrismSharedPrivateTrainer)
            trainer.optimizer = optimizer
            trainer.model = model
            trainer.accelerator = SimpleNamespace(unwrap_model=lambda value: value)
            trainer.sp_config = config(Path(temp), "1.1")
            trainer.artifact_root = Path(temp)
            trainer.optimizer_audit = {}
            trainer._capture_optimizer_audit()
            audit = trainer.optimizer_audit
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["shared_lr"], 1e-5)
        self.assertEqual(audit["private_lr"], 1e-4)
        self.assertEqual(audit["actual_shared_lr"], 0.0)
        self.assertEqual(audit["actual_private_lr"], 0.0)
        self.assertTrue(audit["scheduled_lrs_compatible"])
        self.assertEqual(audit["failure_reasons"], [])

    def test_primary_cumulative_contract_is_unchanged(self):
        self.assertEqual(
            component_task_ids("primary_cumulative", after_task=5, eval_task=3),
            [1, 2, 3, 4, 5],
        )


class KeyIsolationTest(unittest.TestCase):
    def test_v12_config_contract_and_manifest_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            value = config(Path(temp), "1.2")
            value.validate()
            payload = value.to_dict()
            self.assertEqual(payload["method_version"], "1.2")
            self.assertEqual(payload["key_loss_mode"], "rms_cosine_A")
            self.assertEqual(payload["key_lambda"], 0.1)
            self.assertFalse(payload["shared_gradient_hook_enabled"])

        from scripts.medicalskill_v1_2_med_prism_versions.prepare_versioned import (
            manifest_method_fields,
            method_fields,
            refresh_failed_run_manifest,
        )

        v11 = manifest_method_fields(
            method_fields("1.1", shared_lr=1e-5, private_lr=1e-4, drift=0.01, key=0.1)
        )
        v12 = manifest_method_fields(
            method_fields("1.2", shared_lr=1e-5, private_lr=1e-4, drift=0.01, key=0.1)
        )
        self.assertEqual(v11["lambda_shared_drift"], 0.01)
        self.assertEqual(v11["lambda_key"], 0.0)
        self.assertEqual(v12["lambda_shared_drift"], 0.01)
        self.assertEqual(v12["lambda_key"], 0.1)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_manifest = root / "run_manifest.json"
            previous = {"source_provenance": {"source_hashes": {"old": "1"}}}
            replacement = {"source_provenance": {"source_hashes": {"new": "2"}}}
            run_manifest.write_text(
                __import__("json").dumps(previous) + "\n", encoding="utf-8"
            )
            archive = refresh_failed_run_manifest(
                run_manifest, previous, replacement
            )
            self.assertTrue(archive.is_file())
            self.assertEqual(
                __import__("json").loads(run_manifest.read_text()), replacement
            )
            completion = root / "med_prism/stage_01/completion.json"
            completion.parent.mkdir(parents=True)
            completion.write_text('{"status": "PASS"}\n', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                refresh_failed_run_manifest(
                    run_manifest, replacement, previous
                )

    def test_task1_key_loss_is_exact_differentiable_zero(self):
        model = make_model(experts=16)
        result = compute_rank1_key_isolation_loss(model, current_task_id=1)
        self.assertEqual((result.pair_count, float(result.loss)), (0, 0.0))
        self.assertTrue(result.loss.requires_grad)
        result.loss.backward()
        current_a = [
            p
            for name, p in model.named_parameters()
            if "task_0001__" in name and name.endswith(".A")
        ]
        current_b = [
            p
            for name, p in model.named_parameters()
            if "task_0001__" in name and name.endswith(".B")
        ]
        self.assertTrue(
            all(p.grad is not None and float(p.grad.abs().sum()) == 0.0 for p in current_a)
        )
        self.assertTrue(all(p.grad is None for p in current_b))

    def test_pair_counts_and_all_history_scope(self):
        task2 = make_model(tasks=(1, 2), current=2, experts=16)
        result2 = compute_rank1_key_isolation_loss(task2, current_task_id=2)
        self.assertEqual(result2.pair_count, 72 * 16 * 16)
        self.assertEqual(result2.old_task_count, 1)
        task3 = make_model(tasks=(1, 2, 3), current=3, experts=16)
        result3 = compute_rank1_key_isolation_loss(task3, current_task_id=3)
        self.assertEqual(result3.pair_count, 72 * 16 * 16 * 2)
        self.assertEqual(result3.old_task_count, 2)

    def test_key_gradient_reaches_only_current_a(self):
        torch.manual_seed(11)
        model = make_model(tasks=(1, 2), current=2, experts=4)
        for _, wrapper in iter_shared_private_wrappers(model):
            for task_id in (1, 2):
                for expert in wrapper.task_experts(task_id):
                    with torch.no_grad():
                        expert.A.copy_(torch.randn_like(expert.A))
        result = compute_rank1_key_isolation_loss(model, current_task_id=2)
        result.loss.backward()
        old = [(n, p) for n, p in model.named_parameters() if "task_0001__" in n]
        current_a = [p for n, p in model.named_parameters() if "task_0002__" in n and n.endswith(".A")]
        current_b = [p for n, p in model.named_parameters() if "task_0002__" in n and n.endswith(".B")]
        self.assertTrue(all(p.grad is None for _, p in old))
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in current_a))
        self.assertTrue(all(p.grad is None for p in current_b))

    def test_geometric_blind_spot_is_exposed(self):
        current_a = torch.tensor([[1.0, 0.0]])
        old_a = torch.tensor([[1.0, 0.0]])
        current_b = torch.tensor([[1.0, 0.0]])
        old_b = torch.tensor([[0.0, 1.0]])
        geometric = rank1_overlap_squared(current_a, current_b, old_a, old_b)
        key = rank1_key_cosines(current_a, old_a)
        self.assertLess(float(geometric), 1e-8)
        self.assertGreater(float(key.abs()), 0.99)

    def test_geo_orth_is_unchanged_and_key_is_added_once(self):
        model = make_model(tasks=(1, 2), current=2, experts=4)
        before = compute_rank1_orth_loss(model, current_task_id=2, loss_type="rms")
        key = compute_rank1_key_isolation_loss(model, current_task_id=2)
        after = compute_rank1_orth_loss(model, current_task_id=2, loss_type="rms")
        self.assertTrue(torch.equal(before.squared_loss, after.squared_loss))
        task = torch.tensor(2.0)
        total = task + 0.1 * after.loss + 0.01 * torch.tensor(0.5) + 0.1 * key.loss
        expected = task + 0.1 * after.loss + 0.005 + 0.1 * key.loss
        self.assertTrue(torch.allclose(total, expected))


if __name__ == "__main__":
    unittest.main()
