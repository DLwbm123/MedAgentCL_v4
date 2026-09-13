"""Operational continuation: finish training first, then parallel locked-lite cells.

The frozen RGPA trainer, solver, sidecars and old run contract are not edited.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from med_prism.rgpa.run import (REPO, BASELINE, initialize, environment, call,
                                boundary, read, copy_artifacts)
from med_prism.rgpa.state import write_json
from med_prism.transport.checkpoint import sha256_file, legacy_state

CELLS = [(3, 1), (3, 2), (3, 3), (2, 1), (2, 2), (1, 1)]


def eval_env(root, output, gpu):
    manifest = read(root / 'run_manifest.json')
    data = manifest['data']['dataset_root']
    if Path(data).name != 'MedicalSkill-CL-v1.2-lite-10k1k':
        raise ValueError('Expected the frozen lite-10k1k dataset')
    e = environment(root, str(gpu), read(root/'rgpa_contract.json')['slack_enabled'])
    e.update(MED_PRISM_DATA_ROOT=data, MED_PRISM_EVALUATION_ROOT=str(output))
    return e


def validate_cell(root, output, stage, task):
    spec = read(root/'run_manifest.json')['data']['tasks'][str(task)]
    path = output/f'stage_{stage:02d}/primary_cumulative_task_{task:02d}.summary.json'
    s = read(path)
    prediction = path.with_name(path.name.replace('.summary.json', '.predictions.jsonl'))
    if not (s['status'] == 'PASS' and s['stage'] == stage and s['eval_task'] == task
            and s['mode'] == 'primary_cumulative'
            and s['dataset'] == spec['test'] and s['dataset_sha256'] == spec['test_sha256']
            and s['total'] == spec['test_count'] == 1000
            and s['nonempty_predictions'] == 1000 and s['eval_limit'] == 0
            and s['active_private_tasks'] == list(range(1, stage+1))
            and s['predictions_sha256'] == sha256_file(prediction)):
        raise ValueError(f'Incorrect evaluation contract: {path}')
    return s


def gpu_available(gpu):
    # Do not reserve memory or kill other users' processes. Recheck before every cell.
    line = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
           '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    used, utilization = map(int, line.strip().split(','))
    return used < 2048 and utilization < 10


def train(root, output, train_gpus='0'):
    if train_gpus not in ('0', '0,1'):
        raise ValueError('Supported training GPUs: 0 or 0,1')
    gpu_ids = [int(g) for g in train_gpus.split(',')]
    initialize(root, read(root/'rgpa_contract.json')['slack_enabled'])
    legacy_state(root, 2)
    for i in (1, 2):
        if not (root/f'rgpa/task_{i:02d}.json').is_file():
            raise ValueError('Required source-task salience is missing')
    stage = root/'med_prism/stage_03'
    if not (stage/'completion.json').exists():
        if not all(gpu_available(g) for g in gpu_ids):
            raise RuntimeError('Requested GPU is occupied; refusing to start over another job')
        env = environment(root, train_gpus, read(root/'rgpa_contract.json')['slack_enabled'])
        checkpoints = sorted((stage/'train').glob('checkpoint-*'), key=lambda p:int(p.name.split('-')[-1]))
        if checkpoints:
            cp = checkpoints[-1]
            if not all((cp/n).exists() for n in ('rgpa_state.json', 'optimizer.pt', 'rgpa_trainable.safetensors')):
                raise ValueError('Incomplete RGPA training checkpoint')
            env['MED_PRISM_RGPA_RESUME'] = str(cp)
        elif any((stage/'train').iterdir()):
            raise ValueError('Nonempty failed stage without a checkpoint: explicit recovery needed')
        # Use exactly the already-generated stage script; only GPU count changes.
        start = time.monotonic()
        call(['bash', root/'rgpa_train_stage.sh', '--method-version', '1.2',
              '--ablation-name', 'no_geo_orth', '--orth-lambda', '0', '--output-root', root,
              '--gpus', train_gpus, '--start-stage', '3', '--end-stage', '3'], output/'train_task3.log', env)
        write_json(output/'training_time.json', dict(elapsed_seconds=time.monotonic()-start,
                   train_gpus=gpu_ids, per_device_batch=1, gradient_accumulation_steps=16//len(gpu_ids), effective_batch=16))
    if read(stage/'completion.json')['status'] != 'PASS':
        raise RuntimeError('Task3 completion audit failed')
    legacy_state(root, 3)
    # Source-task boundary operation, not evaluation; no old samples used here.
    if not (root/'rgpa/task_03.json').exists():
        boundary(root, 3, environment(root, '0', True), slack=True)
    write_json(output/'training_complete.json', dict(status='PASS', stage=3))


def evaluate(root, output):
    for stage in (1, 2, 3):
        legacy_state(root, stage)
    manifest = read(root/'run_manifest.json')
    for task in (1, 2, 3):
        spec = manifest['data']['tasks'][str(task)]
        if sha256_file(spec['test']) != spec['test_sha256']:
            raise ValueError('Lite test hash mismatch')
    jobs = queue.Queue()
    for stage, task in CELLS:
        path = output/f'stage_{stage:02d}/primary_cumulative_task_{task:02d}.summary.json'
        if path.exists():
            validate_cell(root, output, stage, task)
        else:
            jobs.put((stage, task))
    stop = threading.Event()
    def worker(gpu):
        try:
            while not stop.is_set() and not jobs.empty():
                if not gpu_available(gpu):
                    stop.wait(30)
                    continue
                try:
                    stage, task = jobs.get_nowait()
                except queue.Empty:
                    break
                print(f'EVAL GPU{gpu}: stage{stage}/task{task}', flush=True)
                call([sys.executable, REPO/'scripts/medicalskill_v1_2_med_prism/evaluate_formal_v1_2_cell.py',
                      '--stage', stage, '--task', task, '--mode', 'primary_cumulative',
                      '--batch-size', '4', '--eval-limit', '0'], output/f'eval_s{stage}_t{task}_gpu{gpu}.log',
                     eval_env(root, output, gpu))
                validate_cell(root, output, stage, task)
                jobs.task_done()
        except Exception:
            stop.set()
            raise
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu) for gpu in (0, 1)]
        for future in futures:
            future.result()
    scores = {(s,t):validate_cell(root, output, s,t)['primary_score'] for s,t in CELLS}
    v = [scores[3,i] for i in (1,2,3)]
    with (BASELINE/'primary_cumulative_matrix.csv').open() as f:
        b = list(csv.DictReader(f))
    baseline = [float(b[2][k]) for k in ('VQA','Diagnosis','Concept')]
    current = dict(R31=v[0], R32=v[1], R33=v[2], Avg3=sum(v)/3,
                   BWT3=((v[0]-scores[1,1])+(v[1]-scores[2,2]))/2)
    reference = dict(R31=baseline[0], R32=baseline[1], R33=baseline[2], Avg3=sum(baseline)/3,
                     BWT3=((baseline[0]-float(b[0]['VQA']))+(baseline[1]-float(b[1]['Diagnosis'])))/2)
    delta = {k:100*(current[k]-reference[k]) for k in current}
    safe = delta['R33'] >= -.3 and min(delta['R31'], delta['R32']) >= -1
    decision = ('GO-A' if safe and delta['Avg3'] >= .5 else
                'GO-B' if safe and delta['Avg3'] >= .3 and delta['BWT3'] >= .5 else
                'GO-C' if safe and delta['Avg3'] >= 0 and delta['R32'] >= 1 else 'NO-GO')
    write_json(output/'summary.json', dict(status='PASS', dataset=manifest['data']['dataset_root'],
               scale='0..1', scores={f'R{s}{t}':x for (s,t),x in scores.items()},
               current=current, baseline=reference, delta_pp=delta, decision=decision))
    lines = ['# Matched lite evaluation', '', '|Metric|Baseline|RGPA|Delta pp|', '|---|---:|---:|---:|']
    for k in current:
        lines.append(f'|{k}|{100*reference[k]:.5f}|{100*current[k]:.5f}|{delta[k]:+.5f}|')
    lines += ['', f'Decision: {decision}. No automatic T4/T5 or Candidate B.',
              'Task3 metric is all-concept micro-F1. Old full-test results are excluded.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('train','eval','all'), default='all')
    p.add_argument('--train-gpus', choices=('0', '0,1'), default='0')
    a = p.parse_args()
    a.root, a.output = a.root.resolve(), a.output.resolve()
    if a.output.parent != a.root or not a.output.name.startswith('evaluation_lite_'):
        p.error('Use a dedicated evaluation_lite_* subdirectory')
    a.output.mkdir(exist_ok=True)
    provenance = dict(mode=a.mode, train_gpus=[int(g) for g in a.train_gpus.split(',')], evaluation_gpus=[0,1],
        cells=CELLS, source_sha256=sha256_file(__file__), old_contract_sha256=sha256_file(a.root/'rgpa_contract.json'),
        note='Operational split and explicit lite dataset. Frozen method source is unchanged.')
    write_json(a.output/f'workflow_{a.mode}.json', provenance)
    try:
        if a.mode in ('train','all'):
            train(a.root, a.output, a.train_gpus)
        if a.mode in ('eval','all'):
            evaluate(a.root, a.output)
        write_json(a.output/f'completion_{a.mode}.json', dict(status='PASS'))
    except Exception as exc:
        write_json(a.output/f'completion_{a.mode}.json', dict(status='FAIL', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
