"""Frozen-state teacher-forced read audit; no fitting, generation or checkpoint writes."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from contextlib import ExitStack
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from med_prism.transport.cli import load_calibration_model, DEFAULT_NO_GEO, DEFAULT_DATA, TASKS
from med_prism.transport.checkpoint import read_state, legacy_state, apply_tensors, validate_transition, sha256_file
from med_prism.transport.config import TPMConfig
from med_prism.transport.calibration import stable_id
from med_prism.transport.hooks import QwenBlockBridge, cleanup_hooks
from med_prism.transport.analytic_transport import _read_error
from med_prism.transport.banks import fingerprint, check_invariants
from med_prism.transport.diagnostics import write_json


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


@torch.no_grad()
def student_reads(bridge, record, rms):
    stats = {}
    def hook_for(name):
        def capture(module, args):
            x = args[0][0, record['positions'].to(bridge.device)].float().double() / rms[name]
            for bank in bridge.banks[name]:
                if bank.task_id >= 3:
                    continue
                a, b = bank.matrices(device=bridge.device)
                z = record['teacher'][name]['targets'][bank.task_id].to(bridge.device).double() / rms[name]
                error = _read_error(a.double(), b.double(), x, z).clamp_min(0)
                if not bool(torch.isfinite(error)):
                    raise RuntimeError('Nonfinite diagnostic')
                stats[(name, bank.task_id)] = float(error)
        return capture
    with cleanup_hooks() as handles:
        for name, module in bridge.targets.items():
            handles.append(module.register_forward_pre_hook(hook_for(name)))
        output = bridge.backbone(**bridge.model_inputs(record['batch']), use_cache=False, return_dict=True)
        if not bool(torch.isfinite(output.last_hidden_state).all()):
            raise RuntimeError('Nonfinite forward')
    if len(stats) != 2 * len(bridge.targets):
        raise RuntimeError('Incomplete bank capture')
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--pilot-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    run, pilot, out = args.run_root.resolve(), args.pilot_root.resolve(), args.output_root.resolve()
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    states = {'teacher': legacy_state(DEFAULT_NO_GEO, 2), 'pre': read_state(run/'pre_tpm'),
              'post': read_state(run/'accepted', require_accepted=True)}
    request = json.loads((run/'run_request.json').read_text())
    if states['teacher'].origin != request['teacher']:
        raise ValueError('Teacher differs from original repair')
    validate_transition(states['teacher'], states['pre'])
    config = TPMConfig(**request['tpm'])
    diag = read(run/'diagnostics/tpm_bank_diagnostics.jsonl')
    rms = {r['module_name']: r['reference_rms'] for r in diag}
    if len(rms) != 72 or any(r['reference_rms'] != rms[r['module_name']] for r in diag):
        raise ValueError('Original RMS inconsistent')
    metadata = {(r['module_name'], r['historical_bank_id']):
                {k:r[k] for k in ('module_name','historical_bank_id','projection','depth','block')} for r in diag}
    files = [run/'run_request.json', run/'diagnostics/tpm_bank_diagnostics.jsonl',
             pilot/'data/pilot_data_manifest.json']
    for state in states.values():
        if Path(state.origin).is_file(): files.append(Path(state.origin))
        for path in state.components():
            files.extend([Path(path), Path(path).parent/json.loads(Path(path).read_text())['weights_file']])
    samples = []
    membership = {}
    for task in (1,2):
        path = pilot/'data'/TASKS[task]/'development.jsonl'
        train = Path(DEFAULT_DATA)/TASKS[task]/'train.jsonl'
        files.extend([path,train])
        train_ids = {stable_id(r) for r in read(train)}
        rows = read(path)
        membership[str(task)] = {'count':len(rows), 'in_original_training':sum(stable_id(r) in train_ids for r in rows)}
        samples.extend((f'dev_task{task}', r) for r in rows)
    path = Path(DEFAULT_DATA)/TASKS[2]/'test.jsonl'
    files.append(path)
    predictions = []
    for name in ('eval_pre_tpm','eval_post_tpm'):
        f = run/name/'task_02.predictions.jsonl'; files.append(f)
        predictions.append({r['id']:r for r in read(f)})
    groups = {}
    for row in read(path):
        sid = stable_id(row); a,b = [v[sid]['correct'] for v in predictions]
        group = 'wrong_to_right' if not a and b else 'right_to_wrong' if a and not b else 'unchanged'
        groups[sid] = {'group':group,'pre_correct':bool(a),'post_correct':bool(b)}
        samples.append(('test_task2_'+group, row))
    hashes = {str(f):sha256_file(f) for f in files}
    out.mkdir(parents=True)
    write_json(out/'provenance.json', {'source_sha256':hashes,'dev_membership':membership,
        'teacher':states['teacher'].origin,'pre':states['pre'].origin,'post':states['post'].origin,
        'samples':len(samples),'tokens_per_sample':config.tokens_per_sample,
        'rms':'frozen original Task3 teacher fit RMS; no old-data estimation',
        'scope':'diagnostic only; test responses never fitted; all states frozen',
        'aggregation':'sum over 144 banks of token-weighted mean read errors',
        'wrapper_sha256':sha256_file(__file__)})
    started = time.time()
    model, template = load_calibration_model(states['post'], request['max_length'])
    bridge = QwenBlockBridge(model)
    original = fingerprint(model)
    tensors = {k:v.tensors() for k,v in states.items()}
    def activate(name):
        apply_tensors(model,tensors[name])
        for wrapper in bridge.targets.values(): wrapper.set_active_tasks([1,2] if name=='teacher' else [1,2,3])
    rejected = []
    all_rows = []
    with ExitStack() as stack:
        # Fail closed if an accidental fitting/checkpoint call is introduced.
        for target in ('med_prism.transport.analytic_transport.transport_keys',
                       'med_prism.transport.safety.propose', 'med_prism.transport.checkpoint.materialize_state'):
            stack.enter_context(patch(target, side_effect=RuntimeError('Forbidden in frozen diagnostic')))
        for offset in range(0,len(samples),16):
            records = []
            from swift.template import MaxLengthError
            for group,row in samples[offset:offset+16]:
                try:
                    encoded = template.encode(copy.deepcopy({k:row[k] for k in ('messages','images','videos') if k in row}))
                except MaxLengthError:
                    rejected.append({'group':group,'id':stable_id(row),'reason':'MaxLengthError'}); continue
                batch = template.data_collator([encoded])
                if 'attention_mask' not in batch: batch['attention_mask'] = torch.ones_like(batch['input_ids'])
                if 'labels' not in batch or batch['input_ids'].shape[0] != 1: raise ValueError('Template mismatch')
                records.append({'id':stable_id(row),'group':group,'batch':batch})
            activate('teacher')
            for rec in records: bridge.collect_teacher(rec,3,config)
            for name in ('pre','post'):
                activate(name)
                for rec in records: rec[name] = student_reads(bridge,rec,rms)
            with (out/'bank_reads.jsonl').open('a') as f, (out/'sample_reads.jsonl').open('a') as sf:
                for rec in records:
                    rows = []
                    for key, pre in rec['pre'].items():
                        row = {'id':rec['id'],'group':rec['group'],**metadata[key],
                               'tokens':len(rec['positions']),'pre':pre,'post':rec['post'][key]}
                        f.write(json.dumps(row)+'\n'); rows.append(row)
                    all_rows.extend(rows)
                    summary = {'id':rec['id'],'group':rec['group'],'tokens':len(rec['positions']),
                               'read_error_pre':sum(r['pre'] for r in rows),'read_error_post':sum(r['post'] for r in rows),
                               'token_selection':rec['selection']}
                    if rec['group'].startswith('test_'): summary.update(groups[rec['id']])
                    sf.write(json.dumps(summary)+'\n')
            print(f'DIAGNOSTIC {min(offset+16,len(samples))}/{len(samples)} elapsed={time.time()-started:.1f}s',flush=True)
    activate('post')
    invariant = check_invariants(original,fingerprint(model),set())
    if any(sha256_file(f)!=h for f,h in hashes.items()): raise RuntimeError('Input files changed')
    result = {'status':'PASS','elapsed_seconds':time.time()-started,'invariants':invariant,'rejected':rejected,'groups':{}}
    for group in sorted({r['group'] for r in all_rows}):
        subset = [r for r in all_rows if r['group']==group]
        def summarize(rows):
            # Bank-wise weighted means then sum, including variable selected-token counts.
            totals = {}
            for row in rows:
                key=(row['module_name'],row['historical_bank_id']); v=totals.setdefault(key,[0.,0.,0])
                v[0]+=row['pre']*row['tokens'];v[1]+=row['post']*row['tokens'];v[2]+=row['tokens']
            pre=sum(v[0]/v[2] for v in totals.values()); post=sum(v[1]/v[2] for v in totals.values())
            return {'samples':len({r['id'] for r in rows}),'read_error_pre':pre,'read_error_post':post,
                    'relative_improvement':(pre-post)/pre if pre else None}
        entry = summarize(subset)
        entry['breakdown']={f'{key}:{value}':summarize([r for r in subset if r[key]==value])
            for key in ('projection','depth','historical_bank_id') for value in sorted({r[key] for r in subset})}
        result['groups'][group]=entry
    write_json(out/'summary.json',result)
    print(json.dumps(result['groups'],indent=2),flush=True)


if __name__=='__main__':
    with torch.no_grad(): main()
