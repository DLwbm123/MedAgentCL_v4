import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from safetensors.torch import save_file

from med_prism.adapters.shared_private import SharedPrivateLinear
from med_prism.transport.analytic_transport import self_test, transport_keys
from med_prism.transport.banks import fingerprint, check_invariants, banks_for_wrapper, tensor_hash
from med_prism.transport.calibration import select_tokens, shuffled_target, ordered_rows
from med_prism.transport.checkpoint import (AdapterState, materialize_state, read_state,
    sha256_file, validate_transition, apply_tensors)
from med_prism.transport.config import TPMConfig, parse_rank
from med_prism.transport.engine import run_transport
from med_prism.transport.hooks import QwenBlockBridge
from med_prism.transport.safety import propose, measure


def make_model():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    cfg = Qwen3VLTextConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True})
    cfg._attn_implementation = "eager"
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Qwen3VLTextModel(cfg)
        def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
            visual = torch.zeros_like(input_ids, dtype=torch.bool)
            visual[:, 1:3] = True
            features = [torch.full((int(visual.sum()), 32), .003*(i+1), device=input_ids.device) for i in range(2)]
            return self.language_model(input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, visual_pos_masks=visual, deepstack_visual_embeds=features, **kwargs)
    model = nn.Module()
    model.model = Backbone()
    for index, layer in enumerate(model.model.language_model.layers):
        for key in ("q_proj", "v_proj"):
            name = f"model.language_model.layers.{index}.self_attn.{key}"
            wrapper = SharedPrivateLinear(getattr(layer.self_attn, key), module_identity=name,
                      shared_rank=4, shared_alpha=4, shared_lr_scale=1, dropout=0, shared_trainable=False)
            for task in (1, 2, 3):
                wrapper.add_private_task(task, private_rank=4, experts_per_task=4, alpha=4, trainable=False)
                for expert in wrapper.task_experts(task):
                    expert.B.data.normal_(std=.03)
            wrapper.shared.B.data.normal_(std=.03)
            setattr(layer.self_attn, key, wrapper)
    return model.eval().requires_grad_(False)


def source_state(model, root, task):
    bridge = QwenBlockBridge(model)
    targets = list(bridge.targets)
    params = dict(model.named_parameters())
    manifests = []
    for kind, i in [("shared", task)] + [("private", i) for i in range(1, task+1)]:
        folder = root / ("shared" if kind == "shared" else f"private{i}")
        folder.mkdir(parents=True)
        marker = ".shared." if kind == "shared" else f".experts.task_{i:04d}__"
        values = {k: v.detach().cpu().clone() for k, v in params.items() if marker in k}
        weights = folder / f"{kind}_adapter.safetensors"
        save_file(values, str(weights))
        header = {"format_version": "med_prism_shared_private_v1", "method": "med_prism_shared_private",
            "component_type": kind, "task_id": i, "rank": 4, "alpha": 4, "scaling": 1,
            "experts_per_task": 4, "private_adapter_type": "rank1_expert_bank", "merged": False,
            "model_id": "fake", "model_revision": "fake", "target_modules": targets,
            "target_module_hash": "fake", "weights_file": weights.name, "weights_sha256": sha256_file(weights),
            "tensor_count": len(values), "tensor_schema": [{"key": k, "shape": list(v.shape), "dtype": str(v.dtype).removeprefix("torch.")} for k,v in values.items()]}
        path = folder / f"{kind}_manifest.json"
        path.write_text(json.dumps(header))
        manifests.append(str(path))
    return AdapterState(task, manifests[0], manifests[1:], str(root)).validate()


def records():
    return [{"id": str(i), "split": "fit" if i < 2 else "holdout", "batch": {
        "input_ids": torch.tensor([[3, 4, 5, 6+i, 9, 10, 11, 12]]),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
        "labels": torch.tensor([[-100]*5+[10, 11, 12]])}} for i in range(4)]


class TransportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_supplied_fifteen_checks(self):
        report = self_test()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["checks"]), 15)

    def test_configuration_and_rank_controls(self):
        self.assertIsNone(parse_rank("none"))
        self.assertIsNone(parse_rank("unbudgeted"))
        self.assertEqual(parse_rank("0"), 0)
        for kwargs in ({"eta": 0}, {"repair_rank": -1}, {"calibration_samples": 0}, {"solve_dtype": "bfloat16"}):
            with self.assertRaises(ValueError):
                TPMConfig(**kwargs)

    def test_deterministic_padding_and_types(self):
        mask = torch.tensor([1, 1, 1, 1, 1, 0, 0])
        labels = torch.tensor([-100, -100, -100, 2, 3, -100, -100])
        visual = torch.tensor([0, 1, 0, 0, 0, 0, 0])
        a, stats = select_tokens(mask, labels, visual, count=4, seed=42)
        b, _ = select_tokens(mask, labels, visual, count=4, seed=42)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue((a < 5).all())
        self.assertEqual(set(stats["selected_types"]), {"vision", "prompt", "answer"})
        _, fallback = select_tokens(mask, None, None, count=8, seed=42)
        self.assertEqual(fallback["selected_types"], {"unknown_other": 5})

    def test_shuffle_preserves_exact_token_multiset(self):
        z = torch.arange(64).view(16,4)
        shuffled, permutation = shuffled_target(z, 42)
        self.assertFalse(torch.equal(shuffled,z))
        self.assertEqual(sorted(permutation.tolist()), list(range(16)))
        self.assertTrue(torch.equal(shuffled, z[permutation]))
        with self.assertRaises(ValueError):
            ordered_rows([{"id": "same"}]*2,42)

    def test_postcast_cap_objective_and_rank_zero(self):
        A = torch.randn(4, 24).bfloat16().float()
        B = torch.randn(16,4)
        X = torch.randn(32,24)
        Z = (X + .2*torch.randn_like(X)) @ A.T
        config = TPMConfig()
        candidate, report = propose(A,B,X,Z,X[:8],Z[:8],torch.bfloat16,config)
        self.assertEqual(candidate.dtype, torch.bfloat16)
        self.assertLessEqual(report["solve_delta_effective_rank"], 2)
        if report["commit"]:
            self.assertLessEqual(report["post_cast"]["effective_relative_edit"], .050001)
            self.assertLessEqual(report["post_cast"]["regularized_objective_after"], report["post_cast"]["regularized_objective_before"] + 1e-5)
        zero, zr = propose(A,B,X,Z,X[:8],Z[:8],torch.bfloat16,replace(config,repair_rank=0))
        self.assertTrue(torch.equal(zero.float(), A))
        self.assertTrue(zr["commit"])

    def test_explicit_cholesky_fallback_no_jitter(self):
        from med_prism.transport import safety
        actual = safety.transport_keys
        calls = []
        def flaky(*args, **kwargs):
            calls.append(args[0].dtype)
            if args[0].dtype == torch.float32:
                raise RuntimeError("Cholesky failed. test")
            return actual(*args, **kwargs)
        A,B,X = torch.randn(2,4), torch.randn(3,2), torch.randn(8,4)
        Z = X @ A.T
        with patch.object(safety, "transport_keys", flaky):
            _, r = propose(A,B,X,Z,X,Z,torch.float32,TPMConfig(fallback_fp64=True))
        self.assertEqual(calls, [torch.float32, torch.float64])
        self.assertIsNotNone(r["fp64_retry_reason"])

    def bf16_boundary_fixture(self):
        # Real FP32 solve with a small, deterministic BF16 cap overshoot.
        torch.manual_seed(4)
        A = torch.randn(4, 24).bfloat16().float()
        B = torch.randn(16, 4)
        X = torch.randn(32, 24)
        Z = (X + .5 * torch.randn_like(X)) @ A.T
        return A, B, X, Z

    def test_bf16_deployment_gamma_backtracking(self):
        A, B, X, Z = self.bf16_boundary_fixture()
        originals = [t.clone() for t in (A, B, X, Z)]
        cfg = TPMConfig()
        solved = transport_keys(A, B, X, Z, eta=cfg.eta, repair_rank=2,
                                max_relative_edit=cfg.max_relative_edit)
        fp32 = measure(A, solved.A_new, B, X, Z, cfg.eta)
        initial = measure(A, solved.A_new.bfloat16(), B, X, Z, cfg.eta)
        self.assertLessEqual(fp32["effective_relative_edit"], .05)
        self.assertGreater(initial["effective_relative_edit"], .05 + cfg.edit_tolerance)
        self.assertLess(initial["effective_relative_edit"], .0501)
        with patch("med_prism.transport.safety.measure", wraps=measure) as measured:
            candidate, r = propose(A, B, X, Z, X[:8], Z[:8], torch.bfloat16, cfg)
        self.assertTrue(r["commit"], r["reject_reason"])
        self.assertGreater(r["backtracking_steps"], 0)
        self.assertLessEqual(r["backtracking_steps"], 16)
        self.assertEqual(measured.call_count, 3 + 2 * r["backtracking_steps"])
        self.assertEqual(r["initial_gamma"], solved.diagnostics["damping"])
        self.assertLess(r["final_gamma"], r["initial_gamma"])
        self.assertEqual(r["initial_post_cast_relative_edit"], initial["effective_relative_edit"])
        final = measure(A, candidate, B, X, Z, cfg.eta)
        self.assertEqual(final, r["post_cast"])
        self.assertEqual(final["effective_relative_edit"], r["final_post_cast_relative_edit"])
        self.assertLessEqual(final["effective_relative_edit"], .05)
        self.assertTrue(final["finite"])
        for key in ("read_error", "regularized_objective"):
            self.assertLessEqual(final[key + "_after"], final[key + "_before"])
        self.assertTrue(all(torch.equal(t, old) for t, old in zip((A, B, X, Z), originals)))

    def test_backtracking_is_bounded_and_preserves_other_gates(self):
        A, B, X, Z = self.bf16_boundary_fixture()
        for extra_failure in (False, True):
            with self.subTest(extra_failure=extra_failure):
                def blocked(*args):
                    stats = measure(*args)
                    if args[1].dtype == torch.bfloat16:
                        # Simulate an unresolvable quantization plateau.
                        stats["effective_relative_edit"] = .050064
                        if extra_failure:
                            stats["regularized_objective_after"] = stats["regularized_objective_before"] + 1
                    return stats
                with patch("med_prism.transport.safety.measure", side_effect=blocked) as measured:
                    _, r = propose(A, B, X, Z, X[:8], Z[:8], torch.bfloat16, TPMConfig())
                self.assertFalse(r["commit"])
                self.assertEqual(r["backtracking_steps"], 0 if extra_failure else 16)
                self.assertEqual(measured.call_count, 3 + 2 * r["backtracking_steps"])
                self.assertIn("post_cast:edit_cap", r["reject_reason"])
                if extra_failure:
                    self.assertIn("post_cast:objective_increase", r["reject_reason"])

    def test_forbidden_mutation_detected(self):
        model = nn.Linear(3,2)
        before = fingerprint(model)
        model.weight.data.add_(1)
        with self.assertRaisesRegex(RuntimeError, "non-approved"):
            check_invariants(before, fingerprint(model), set())

    def test_grouped_development_exclusion_across_tasks(self):
        from med_prism.transport.pilot_data import build_split, group_aliases
        rows={t:[{"id":f"{t}:{i}","images":[f"/tmp/test_image_{i}.png"],
                    "metadata":{"source_dataset":"fake","patient_id":str(i//2)}}
                  for i in range(30)] for t in (1,2,3)}
        splits,report=build_split(rows,train_count=12,dev_count=4,seed=42)
        self.assertTrue(report["global_group_disjoint"])
        ta={k for train,_ in splits.values() for r in train for k in group_aliases(r)}
        da={k for _,dev in splits.values() for r in dev for k in group_aliases(r)}
        self.assertFalse(ta&da)
        self.assertTrue(all(len(a)==12 and len(b)==4 for a,b in splits.values()))
        self.assertEqual(report,build_split(rows,train_count=12,dev_count=4,seed=42)[1])

    def test_no_geo_training_recipe_and_accepted_paths(self):
        from med_prism.transport.pilot import training_config,training_command
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dataset=root/"train.jsonl"
            dataset.write_text('{}\n')
            previous=AdapterState(2,"/new/accepted/shared/shared_manifest.json",
                                  ["/new/accepted/private/task_0001/private_manifest.json",
                                   "/new/accepted/private/task_0002/private_manifest.json"],"accepted")
            cfg=training_config(root/"stage3",3,dataset,root/"manifest.json",previous,42)
            self.assertEqual(cfg.private_source_manifests,tuple(previous.private_manifests))
            self.assertEqual((cfg.orth_lambda,cfg.key_lambda,cfg.shared_drift_lambda),(0,.1,.01))
            self.assertEqual((cfg.shared_lr,cfg.private_lr),(1e-5,1e-4))
            cmd=training_command(root/"stage3",dataset,"0,1",42)
            self.assertEqual(cmd[cmd.index("--gradient_accumulation_steps")+1],"8")
            self.assertFalse(any("transport" in x for x in cmd))

    def test_sequential_native_invariants_roundtrip_and_next_key(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = source_state(model, root/"teacher",2)
            for name, p in model.named_parameters():
                if ".shared.B" in name:
                    p.add_(torch.randn_like(p)*.06)
            student = source_state(model, root/"student",3)
            validate_transition(teacher,student)
            before = fingerprint(model)
            config = TPMConfig(calibration_samples=2,holdout_samples=2,tokens_per_sample=6,max_relative_edit=.2)
            result = run_transport(model,teacher,student,records(),config,root/"diagnostics")
            self.assertEqual(result["status"],"PASS")
            self.assertTrue(result["invariants"]["changed_tensors"])
            self.assertTrue(all("task_0003" not in k and k.endswith(".A") for k in result["invariants"]["changed_tensors"]))
            rows = [json.loads(s) for s in (root/"diagnostics/tpm_bank_diagnostics.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows),12)
            self.assertTrue(all(r["n_fit_tokens"] == 12 and r["n_holdout_tokens"] == 12 for r in rows))
            # Explicitly re-run final model and compare every normalized projection
            # input to what that bank's solver used. This fails for stale-input repair.
            final_bridge=QwenBlockBridge(model)
            final_records=records()
            observed={name:[] for name in final_bridge.targets}
            for record in final_records[:2]:
                final_bridge.collect_teacher(record,3,config)
                handles=[]
                for name,wrapper in final_bridge.targets.items():
                    def capture(module,args,_name=name):
                        observed[_name].append(args[0][0,record["positions"]].detach().float().clone())
                    handles.append(wrapper.register_forward_pre_hook(capture))
                try:
                    final_bridge.backbone(**final_bridge.model_inputs(record["batch"]),use_cache=False,return_dict=True)
                finally:
                    for handle in handles:
                        handle.remove()
            for row in rows:
                X=torch.cat(observed[row["module_name"]])/row["reference_rms"]
                self.assertEqual(tensor_hash(X),row["student_fit_input_sha256"])
            saved = materialize_state(student,root/"accepted",model=model,accepted=True)
            self.assertEqual(read_state(root/"accepted",require_accepted=True).task_id,3)
            materialize_state(student,root/"pre_only",accepted=False)
            with self.assertRaisesRegex(ValueError,"must be ACCEPTED"):
                read_state(root/"pre_only",require_accepted=True)
            after = saved.tensors()
            actual = dict(model.named_parameters())
            self.assertTrue(all(torch.equal(v,actual[k]) for k,v in after.items()))
            self.assertTrue(any(not torch.equal(v,student.tensors()[k]) for k,v in after.items()))
            with self.assertRaises(FileExistsError):
                materialize_state(student,root/"accepted",accepted=True)
            fresh = make_model()
            apply_tensors(fresh,read_state(root/"accepted").tensors())
            from med_prism.projection.orth_losses import compute_rank1_key_isolation_loss
            for name, wrapper in QwenBlockBridge(fresh).targets.items():
                wrapper.add_private_task(4, private_rank=4, experts_per_task=4, alpha=4, trainable=True)
                for bank in banks_for_wrapper(name,wrapper):
                    if bank.task_id < 4:
                        for pname,factor in zip(bank.names,bank.factors):
                            self.assertTrue(torch.equal(factor.A,after[pname]))
            # Existing live-row key loss reads accepted transported A, no original-key cache.
            loss = compute_rank1_key_isolation_loss(fresh,current_task_id=4)
            self.assertEqual(loss.old_task_count,3)
            self.assertGreater(loss.pair_count,0)
            self.assertEqual(sum(v["numel"] for v in before.values()),sum(v["numel"] for v in fingerprint(model).values()))

    def test_off_no_change_and_strict_rollback(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            teacher=source_state(model,root/"t",2)
            student=source_state(model,root/"s",3)
            before=fingerprint(model)
            run_transport(model,teacher,student,[],TPMConfig(mode="off"),root/"off")
            self.assertEqual(before,fingerprint(model))
            with patch("med_prism.transport.engine.propose", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaisesRegex(RuntimeError,"Strict TPM rejection"):
                    run_transport(model,teacher,student,records(),TPMConfig(calibration_samples=2,holdout_samples=2,strict=True),root/"strict")
            self.assertEqual(before,fingerprint(model))
            self.assertTrue((root/"strict/failure.json").is_file())

    def test_unbudgeted_and_shuffled_boundary_controls(self):
        for rank,shuffle in ((None,False),(2,True)):
            with self.subTest(rank=rank,shuffle=shuffle), tempfile.TemporaryDirectory() as tmp:
                model=make_model()
                root=Path(tmp)
                teacher=source_state(model,root/"teacher",2)
                for name,p in model.named_parameters():
                    if ".shared.B" in name:
                        p.add_(torch.randn_like(p)*.06)
                student=source_state(model,root/"student",3)
                cfg=TPMConfig(calibration_samples=2,holdout_samples=2,tokens_per_sample=4,
                              repair_rank=rank,shuffle_teacher_pairing=shuffle)
                summary=run_transport(model,teacher,student,records(),cfg,root/"diag")
                self.assertEqual(summary["status"],"PASS")
                rows=[json.loads(l) for l in (root/"diag/tpm_bank_diagnostics.jsonl").read_text().splitlines()]
                self.assertTrue(all(r["n_fit_tokens"]==8 and r["n_holdout_tokens"]==8 for r in rows))
                self.assertTrue(all(bool(r["pairing_permutation_sha256"])==shuffle for r in rows))
                self.assertTrue(all(("matched_fit_diagnostic" in r)==shuffle for r in rows))


if __name__ == "__main__":
    unittest.main()
