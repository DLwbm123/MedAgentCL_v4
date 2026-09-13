"""CPU-only evaluation wiring tests; no real generation or test split writes."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location('tpm_retention',
    Path(__file__).resolve().parents[2] / 'scripts/medicalskill_v2_tpm/evaluate_retention.py')
ret = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ret)


class RetentionTests(unittest.TestCase):
    def test_state_loader_uses_explicit_banks_and_checks_values(self):
        model = torch.nn.Linear(2, 2, bias=False)
        expected = model.weight.detach().clone()
        state = SimpleNamespace(shared_manifest='/accepted/shared.json',
             private_manifests=[f'/accepted/task{i}.json' for i in (1, 2, 3)],
             tensors=lambda: {'weight': expected})
        def base_loader(stage):
            self.assertEqual(stage, 3)
            active = ret.evaluator.compose_shared_private(model, shared_manifest='/old/shared',
                                                          private_manifests=['/old/task1'])
            return model, 'processor', active
        with patch.object(ret.evaluator, 'load_model', side_effect=base_loader), \
             patch.object(ret.evaluator, 'compose_shared_private', return_value={'loaded': True}) as compose, \
             patch.object(ret.evaluator, 'set_active') as active, \
             patch.object(ret.evaluator, 'reload_probe', side_effect=AssertionError('Old logits prohibited')):
            ret.load_evaluation_state(state)
            compose.assert_called_once_with(model, shared_manifest=state.shared_manifest,
                                            private_manifests=state.private_manifests)
            active.assert_called_once_with(model, [1, 2, 3])
            expected.add_(1)
            with self.assertRaisesRegex(RuntimeError, 'Loaded state tensor differs'):
                ret.load_evaluation_state(state)

    def test_contract_survives_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ret, 'DATA', Path(tmp)):
            for task in (1, 2, 3):
                p = Path(tmp)/ret.evaluator.TASK_DIRS[task]/'test.jsonl'
                p.parent.mkdir()
                p.write_text(json.dumps({'id': str(task)})+'\n')
            value = ret.evaluation_contract()
            self.assertEqual(value, json.loads(json.dumps(value)))
            self.assertEqual(value['active_private_tasks'], [1, 2, 3])
            self.assertEqual(value['eval_limit'], 0)

    def test_paired_orchestration_reuses_generation_metrics_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            states = {}
            for name in ('pre_tpm', 'post_tpm'):
                p = root/(name+'.json'); p.write_text('{}')
                states[name] = SimpleNamespace(origin=str(p), shared_manifest=name+'/shared',
                                               private_manifests=[name+f'/task{i}' for i in (1, 2, 3)])
            datasets = {}
            for task in (1, 2, 3):
                p = root/f'data{task}.jsonl'
                row = {'id': str(task), 'messages': [{'role': 'user', 'content': 'Question'},
                                                     {'role': 'assistant', 'content': 'aorta'}]}
                p.write_text(json.dumps(row)+'\n')
                datasets[str(task)] = {'path': str(p), 'total': 1, 'sha256': ret.sha256_file(p),
                                      'ordered_ids_sha256': ret.digest([str(task)])}
            common = {'datasets': datasets}
            with patch.object(ret, 'audit_pair', return_value=(states, {'status': 'PASS'})), \
                 patch.object(ret, 'evaluation_contract', return_value=common), \
                 patch.object(ret, 'load_evaluation_state', return_value=('model', 'processor', {})), \
                 patch.object(ret.cell, 'batched_generate', return_value=['aorta']) as generate, \
                 patch.object(ret.evaluator, 'evaluate_rows', wraps=ret.evaluator.evaluate_rows) as metric:
                for name in states:
                    ret.evaluate(root, name)
                self.assertEqual(generate.call_count, 6)
                self.assertEqual(metric.call_count, 6)
                self.assertEqual([c.args[3] for c in generate.call_args_list], [1, 2, 3]*2)
                self.assertTrue(all(c.args[4] == 4 for c in generate.call_args_list))
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    ret.summarize(root)
                self.assertIn('state,task1,task2,task3', output.getvalue())
                self.assertIn('delta,0.0000000000,0.0000000000,0.0000000000', output.getvalue())
                with self.assertRaises(FileExistsError):
                    ret.evaluate(root, 'pre_tpm')
                p = root/'eval_post_tpm/task_01.predictions.jsonl'
                p.write_text('{}\n')
                with self.assertRaisesRegex(ValueError, 'Invalid cell'):
                    ret.summarize(root)
