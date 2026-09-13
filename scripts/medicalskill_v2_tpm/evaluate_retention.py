#!/usr/bin/env python3
"""Task3 paired retention only; reuse v1 evaluator, never training/repair/oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LEGACY = ROOT / 'scripts/medicalskill_v1_2_med_prism'
sys.path.insert(0, str(LEGACY))
import torch
import evaluate_formal_v1_2 as evaluator
import evaluate_formal_v1_2_cell as cell
import evaluation_contract_v1_2 as contract
from med_prism.transport.checkpoint import read_state, read_component, sha256_file

DATA = Path('/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def audit_pair(run):
    states = {name: read_state(run / folder / 'state.json', require_accepted=name == 'post_tpm')
              for name, folder in (('pre_tpm', 'pre_tpm'), ('post_tpm', 'accepted'))}
    pre, post = states.values()
    pindex = json.loads(Path(pre.origin).read_text())
    qindex = json.loads(Path(post.origin).read_text())
    if pindex['status'] != 'PRE_TPM' or pre.task_id != 3 or post.task_id != 3:
        raise ValueError('Requires Task3 PRE_TPM/ACCEPTED pair')
    if Path(qindex['source_state']).resolve() != Path(pre.origin):
        raise ValueError('Accepted state is not derived from this pre-TPM state')
    a, b = pre.tensors(), post.tensors()
    if a.keys() != b.keys():
        raise ValueError('Pre/post tensor schemas differ')
    changed = []
    for key in a:
        if a[key].dtype != b[key].dtype or a[key].shape != b[key].shape:
            raise ValueError('Pre/post tensor shape/dtype differs: ' + key)
        if not torch.equal(a[key], b[key]):
            if not (key.endswith('.A') and any(t in key for t in ('task_0001__', 'task_0002__'))):
                raise ValueError('Non-historical-A mutation: ' + key)
            changed.append(key)
    for left, right in zip(pre.components(), post.components()):
        h, _ = read_component(left)
        j, _ = read_component(right)
        for field in ('component_type', 'task_id', 'model_id', 'model_revision', 'target_modules',
                      'target_module_hash', 'rank', 'alpha', 'scaling', 'experts_per_task'):
            if h.get(field) != j.get(field):
                raise ValueError('Pre/post component metadata differs: ' + field)
        if (h['model_id'], h['model_revision']) != (evaluator.MODEL_ID, evaluator.MODEL_REVISION):
            raise ValueError('Evaluator base identity differs from state')
    invariants = json.loads((run / 'diagnostics/invariants.json').read_text())
    if invariants['status'] != 'PASS' or set(invariants['changed_tensors']) != set(changed):
        raise ValueError('Checkpoint changes disagree with recorded full-model invariants')
    return states, {'status': 'PASS', 'adapter_tensors': len(a), 'changed_historical_A': len(changed),
                    'other_adapter_tensors_equal': True, 'recorded_full_model_invariants': 'PASS'}


def evaluation_contract():
    datasets = {}
    for task in (1, 2, 3):
        path = DATA / evaluator.TASK_DIRS[task] / 'test.jsonl'
        rows = evaluator.read_jsonl(path, 0)
        if not rows:
            raise ValueError('Empty test split')
        datasets[str(task)] = {'path': str(path), 'sha256': sha256_file(path), 'total': len(rows),
                               'ordered_ids_sha256': digest([r['id'] for r in rows])}
    import transformers
    return {'stage': 3, 'tasks': [1, 2, 3], 'mode': 'primary_cumulative',
            'active_private_tasks': [1, 2, 3], 'seed': 42, 'batch_size': 4, 'eval_limit': 0,
            'model_id': evaluator.MODEL_ID, 'model_revision': evaluator.MODEL_REVISION,
            'snapshot': str(evaluator.SNAPSHOT), 'dtype': 'bfloat16', 'attention': 'sdpa',
            'generation': json.loads(json.dumps(contract.GENERATION_CONFIG)),
            'image_preprocessing': contract.IMAGE_PREPROCESSING,
            'datasets': datasets, 'torch': torch.__version__, 'transformers': transformers.__version__,
            'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in
                             (Path(__file__).resolve(), LEGACY/'evaluate_formal_v1_2.py',
                              LEGACY/'evaluate_formal_v1_2_cell.py', LEGACY/'evaluation_contract_v1_2.py',
                              ROOT/'med_prism/checkpoint/shared_private_checkpoint.py',
                              ROOT/'med_prism/adapters/shared_private.py')}}


def load_evaluation_state(state):
    # Reuse the original model/processor loading recipe, but replace its legacy
    # manifest path selection. No call to load_model(0) and no Stage0 evaluation.
    compose = evaluator.compose_shared_private
    def from_state(model, **unused_legacy_paths):
        return compose(model, shared_manifest=state.shared_manifest,
                       private_manifests=state.private_manifests)
    with patch.object(evaluator, 'compose_shared_private', side_effect=from_state):
        model, processor, active = evaluator.load_model(3)
    params = dict(model.named_parameters())
    for key, expected in state.tensors().items():
        actual = params[key].detach().cpu()
        if actual.dtype != expected.dtype or not torch.equal(actual, expected):
            raise RuntimeError('Loaded state tensor differs: ' + key)
    evaluator.set_active(model, [1, 2, 3])
    # Original training logits intentionally not used: TPM changes those logits.
    return model, processor, active


def evaluate(run, name):
    states, audit = audit_pair(run)
    common = evaluation_contract()
    output = run / ('eval_' + name)
    state = states[name]
    provenance = {'state': name, 'state_path': state.origin, 'state_sha256': sha256_file(state.origin),
                  'shared_manifest': state.shared_manifest, 'private_manifests': state.private_manifests,
                  'pair_state_sha256': {k: sha256_file(v.origin) for k, v in states.items()},
                  'contract': common, 'audit': audit}
    other = run / ('eval_post_tpm' if name == 'pre_tpm' else 'eval_pre_tpm') / 'provenance.json'
    if other.exists():
        previous = json.loads(other.read_text())
        if previous['contract'] != common or previous['pair_state_sha256'] != provenance['pair_state_sha256']:
            raise ValueError('Paired evaluation inputs/settings changed')
    output.mkdir(exist_ok=False)  # Never silently reuse another state or overwrite results.
    cell.write_json(output/'provenance.json', provenance)
    torch.manual_seed(42)
    model, processor, active = load_evaluation_state(state)
    cell.write_json(output/'loaded_state.json', {'status': 'PASS', 'verification': 'exact adapter tensors',
                                               'active_component_manifest': active})
    scores = {}
    for task in (1, 2, 3):
        rows = evaluator.read_jsonl(Path(common['datasets'][str(task)]['path']), 0)
        predictions = cell.batched_generate(model, processor, rows, task, 4)
        if len(predictions) != len(rows):
            raise RuntimeError('Prediction count mismatch')
        metric, details = evaluator.evaluate_rows(task, rows, predictions)
        path = output / f'task_{task:02d}.predictions.jsonl'
        cell.atomic_write(path, ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in details))
        metric.update(state=name, stage=3, eval_task=task, mode='primary_cumulative',
                      active_private_tasks=[1, 2, 3], predictions=str(path), predictions_sha256=sha256_file(path))
        cell.write_json(output/f'task_{task:02d}.summary.json', metric)
        if metric['status'] != 'PASS':
            raise RuntimeError('Evaluator reported failure; see task summary')
        scores[str(task)] = metric['primary_score']
        print(f'{name} R3{task}={scores[str(task)]}', flush=True)
    if evaluation_contract() != common:
        raise RuntimeError('Evaluation inputs/settings changed during evaluation')
    cell.write_json(output/'completion.json', {'status': 'PASS', 'scores': scores})


def summarize(run):
    states, _ = audit_pair(run)
    expected = evaluation_contract()
    pair_hashes = {k: sha256_file(v.origin) for k, v in states.items()}
    scores = []
    for name in ('pre_tpm', 'post_tpm'):
        root = run / ('eval_' + name)
        provenance = json.loads((root/'provenance.json').read_text())
        if (provenance['contract'] != expected or provenance['pair_state_sha256'] != pair_hashes
                or provenance['state'] != name or provenance['state_sha256'] != pair_hashes[name]):
            raise ValueError('Paired result provenance mismatch')
        completion = json.loads((root/'completion.json').read_text())
        if completion['status'] != 'PASS':
            raise ValueError('Incomplete evaluation')
        row = []
        for task in (1, 2, 3):
            m = json.loads((root/f'task_{task:02d}.summary.json').read_text())
            predictions = root/f'task_{task:02d}.predictions.jsonl'
            details = evaluator.read_jsonl(predictions, 0)
            data = expected['datasets'][str(task)]
            if (m['status'] != 'PASS' or m['state'] != name or m['eval_task'] != task
                    or m['total'] != data['total'] or len(details) != data['total']
                    or m['predictions_sha256'] != sha256_file(predictions)
                    or digest([r['id'] for r in details]) != data['ordered_ids_sha256']):
                raise ValueError('Invalid cell or sample order')
            row.append(m['primary_score'])
        scores.append(row)
    # Only tabulate the existing evaluator's primary_score; do not recalculate metrics.
    lines = ['state,task1,task2,task3']
    for name, row in zip(('pre_tpm', 'post_tpm', 'delta'),
                         [*scores, [b-a for a, b in zip(*scores)]]):
        lines.append(name + ',' + ','.join(f'{v:.10f}' for v in row))
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--state', choices=('pre_tpm', 'post_tpm'))
    action.add_argument('--check-only', action='store_true')
    action.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    run = args.run_root.resolve()
    torch.set_num_threads(1)
    if args.check_only:
        states, audit = audit_pair(run)
        print(json.dumps({'audit': audit, 'states': {k: vars(v) for k, v in states.items()},
                          'contract': evaluation_contract()}, indent=2))
    elif args.summarize:
        summarize(run)
    else:
        evaluate(run, args.state)


if __name__ == '__main__':
    main()
