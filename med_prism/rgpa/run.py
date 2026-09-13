"""Small sequential coordinator around the hash-locked existing training script."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from med_prism.rgpa.state import write_json
from med_prism.transport.checkpoint import sha256_file, legacy_state

REPO = Path('/root/MedAgentCL_v4')
BASELINE = Path('/remote-home/wangbomin/med_prism_v1_2_ablation_no_geo_orth_medicalskill_v1_2_lite_10k1k_seed42')
TRAIN = REPO / 'scripts/medicalskill_v1_2_med_prism_versions/run_versioned_train.sh'


def read(path):
    return json.loads(Path(path).read_text())


def copy_artifacts(source, target):
    """Byte-identical copies without unsupported cross-NFS extended attributes."""
    source, target = Path(source), Path(target)
    files = sorted(p for p in source.rglob('*') if p.is_file()) if source.is_dir() else [source]
    for item in files:
        destination = target / item.relative_to(source) if source.is_dir() else target
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(item, destination)
        if sha256_file(item) != sha256_file(destination):
            raise IOError('Artifact read-back checksum mismatch: ' + str(destination))


def call(command, log, env):
    print(shlex.join(map(str, command)), flush=True)
    with Path(log).open('a') as handle:
        subprocess.run(list(map(str, command)), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def environment(root, gpus, slack):
    e = os.environ.copy()
    e.pop('MED_PRISM_RGPA_RESUME', None)
    e.update(HF_HOME='/remote-home/wangbomin/huggingface_cache',
             HF_HUB_CACHE='/remote-home/wangbomin/huggingface_cache/hub',
             HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=str(REPO),
             CUDA_VISIBLE_DEVICES=gpus, MED_PRISM_RGPA_ROOT=str(root),
             MED_PRISM_RGPA_SLACK='1' if slack else '0', MED_PRISM_ARTIFACT_ROOT=str(root))
    return e


def initialize(root, slack):
    if root.exists():
        contract = read(root / 'rgpa_contract.json')
        if contract['slack_enabled'] != slack or contract['baseline_manifest_hash'] != sha256_file(BASELINE / 'run_manifest.json'):
            raise ValueError('Existing run contract mismatch')
        for name, expected in contract['rgpa_source_hashes'].items():
            if sha256_file(REPO / name) != expected:
                raise ValueError('RGPA source changed; use a NEW run root: ' + name)
        return
    baseline = read(BASELINE / 'run_manifest.json')
    hashes = baseline['source_provenance']['source_hashes']
    if any(sha256_file(REPO / n) != h for n, h in hashes.items()):
        raise ValueError('Current-best source has changed; audit required')
    state = legacy_state(BASELINE, 1)  # validates both component tensors and metadata
    command = shlex.split((BASELINE / 'med_prism/stage_01/train_command.txt').read_text())
    def flag(name):
        return command[command.index(name) + 1]
    if (flag('--gradient_accumulation_steps') != '8' or flag('--per_device_train_batch_size') != '1'
            or flag('--num_train_epochs') != '1' or flag('--seed') != '42' or flag('--data_seed') != '42'):
        raise ValueError('T1 training semantics are not the audited dual-GPU baseline')
    root.mkdir(parents=True)
    env = environment(root, '0', slack)
    call([sys.executable, REPO / 'scripts/medicalskill_v1_2_med_prism_versions/prepare_versioned.py',
          '--method-version', '1.2', '--ablation-name', 'no_geo_orth', '--orth-lambda', '0',
          '--output-root', root], root / 'prepare.log', env)
    # Copy only the immutable completed T1 boundary, not optimizer checkpoints.
    source = BASELINE / 'med_prism/stage_01'
    target = root / 'med_prism/stage_01'
    for folder in ('shared', 'private', 'runtime_audits'):
        copy_artifacts(source / folder, target / folder)
    for filename in ('completion.json', 'target_module_audit.json', 'train_command.txt'):
        copy_artifacts(source / filename, target / filename)
    legacy_state(root, 1)
    own = {str(p.relative_to(REPO)): sha256_file(p) for p in sorted((REPO / 'med_prism/rgpa').glob('*.py'))}
    write_json(root / 'rgpa_contract.json', dict(method_version='2.4', rho=.25, slack_enabled=slack,
               baseline=str(BASELINE), baseline_manifest_hash=sha256_file(BASELINE / 'run_manifest.json'),
               baseline_source_hashes=hashes, rgpa_source_hashes=own, reused_task1=True,
               task1_shared_hash=read(state.shared_manifest)['weights_sha256'],
               task1_private_hash=read(state.private_manifests[0])['weights_sha256'],
               go_thresholds_pp=dict(GO_A_avg=.5, GO_B_avg=.3, GO_B_bwt=.5,
                                      GO_C_avg=0., GO_C_task2=1., current_drop_budget=.3, history_drop_budget=1.)))
    copy_artifacts(REPO / 'docs/med_prism_v2_4_rgpa/IMPLEMENTATION_REPORT.md', root / 'IMPLEMENTATION_REPORT.md')


def training_script(root):
    text = TRAIN.read_text()
    marker = '    "$ROOT/scripts/med_prism_real_5skill/reload_probe_plugin.py"'
    if text.count(marker) != 1:
        raise ValueError('Unexpected baseline plugin list')
    text = text.replace(marker, marker + '\n    "$ROOT/med_prism/rgpa/plugin.py"')
    marker = 'if [[ -d "$TRAIN_ROOT" ]] && find'
    if text.count(marker) != 1:
        raise ValueError('Unexpected baseline resume guard')
    text = text.replace(marker, 'if [[ -z "${MED_PRISM_RGPA_RESUME:-}" && -d "$TRAIN_ROOT" ]] && find')
    path = root / 'rgpa_train_stage.sh'
    path.write_text(text)
    return path


def boundary(root, task, env, *, timing=False, slack=True):
    command = [sys.executable, '-m', 'med_prism.rgpa.boundary', '--root', root, '--task', task]
    if timing:
        command += ['--timing-only']
    if not slack:
        command += ['--without-slack']
    call(command, root / f'boundary_task{task}{"_timing" if timing else ""}.log', env)


def report(root):
    def matrix(path):
        with path.open() as handle:
            return list(csv.DictReader(handle))
    base = matrix(BASELINE / 'primary_cumulative_matrix.csv')
    def score(stage, task):
        s = read(root / f'evaluation/stage_{stage:02d}/primary_cumulative_task_{task:02d}.summary.json')
        # Reuse the official scalar chosen by the evaluator, never Concept accuracy.
        return float(s['primary_score'])
    values = [score(3, i) for i in (1, 2, 3)]
    prior = [score(1, 1), score(2, 2)]
    baseline = [float(base[2][k]) for k in ('VQA', 'Diagnosis', 'Concept')]
    bwt = sum(values[i] - prior[i] for i in (0, 1)) / 2
    base_bwt = ((baseline[0] - float(base[0]['VQA'])) + (baseline[1] - float(base[1]['Diagnosis']))) / 2
    a, b = [*baseline, sum(baseline)/3, base_bwt], [*values, sum(values)/3, bwt]
    delta = [100*(y-x) for x, y in zip(a, b)]
    safe = delta[2] >= -.3 and min(delta[:2]) >= -1
    decision = ('GO-A' if safe and delta[3] >= .5 else
                'GO-B' if safe and delta[3] >= .3 and delta[4] >= .5 else
                'GO-C' if safe and delta[3] >= 0 and delta[1] >= 1 else 'NO-GO')
    rows = list(zip(('R31', 'R32', 'R33_micro_F1', 'Avg3', 'BWT3'), a, b, delta))
    write_json(root / 'summary.json', dict(status='PASS', decision=decision, rows=rows,
               timing=read(root/'rgpa/timing_task_01.json'), thresholds=read(root/'rgpa_contract.json')['go_thresholds_pp']))
    with (root / 'results.csv').open('w') as f:
        f.write('metric,baseline,rgpa,delta_pp\n')
        for key, x, y, d in rows:
            f.write(f'{key},{100*x},{100*y},{d}\n')
    next_command = (f'bash scripts/medicalskill_v2_4_rgpa/run.sh --output-root {root} --start-stage 4 --end-stage 5'
                    if decision.startswith('GO') else
                    'bash scripts/medicalskill_v2_4_rgpa/run.sh --without-slack --output-root /remote-home/wangbomin/MedPRISM_RGPA_B_$(date +%Y%m%d_%H%M%S)_${RANDOM}')
    with (root / 'IMPLEMENTATION_REPORT.md').open('a') as f:
        f.write('\n## G. Completed 3-task results (0–100)\n\n|Metric|Baseline|RGPA|Delta pp|\n|---|---:|---:|---:|\n')
        for key, x, y, d in rows:
            f.write(f'|{key}|{100*x:.6f}|{100*y:.6f}|{d:+.6f}|\n')
        f.write(f'\n## H. Decision\n\n{decision}\n\n## I. Next command (NOT automatically run)\n\n```bash\n{next_command}\n```\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--gpus', default='0')
    p.add_argument('--without-slack', action='store_true')
    p.add_argument('--initialize-only', action='store_true')
    p.add_argument('--start-stage', type=int, default=2)
    p.add_argument('--end-stage', type=int, default=3)
    a = p.parse_args()
    root = a.output_root.resolve()
    if not root.name.startswith('MedPRISM_RGPA_') or root.parent != Path('/remote-home/wangbomin'):
        p.error('Use a new dedicated /remote-home/wangbomin/MedPRISM_RGPA_* directory')
    if not 2 <= a.start_stage <= a.end_stage <= 5:
        p.error('Allowed stages 2..5, default stops at3')
    initialize(root, not a.without_slack)
    if a.initialize_only:
        return
    env = environment(root, a.gpus, not a.without_slack)
    boundary_env = dict(env, CUDA_VISIBLE_DEVICES=a.gpus.split(',')[0])
    if not (root / 'rgpa/timing_task_01.json').exists():
        boundary(root, 1, boundary_env, timing=True, slack=not a.without_slack)
    timing = read(root / 'rgpa/timing_task_01.json')
    # Engineering wall-time guard, not a model hyperparameter or research gate.
    if timing['estimated_64_seconds'] > 3600:
        raise RuntimeError('Boundary estimate exceeds one hour; STOP and ask the user, do not reduce m')
    write_json(root / 'runtime_approval.json', dict(approved=True, limit_seconds=3600,
               estimated_64_seconds=timing['estimated_64_seconds'], m_unchanged=64))
    stage_script = training_script(root)
    for stage in range(a.start_stage, a.end_stage + 1):
        for previous in range(1, stage):
            sidecar = root / f'rgpa/task_{previous:02d}.json'
            if not sidecar.exists():
                if previous != stage - 1:
                    raise ValueError('Missing past boundary: refusing retrospective old-data access')
                boundary(root, previous, boundary_env, slack=not a.without_slack)
        stage_root = root / f'med_prism/stage_{stage:02d}'
        if not (stage_root / 'completion.json').exists():
            checkpoints = sorted((stage_root/'train').glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
            stage_env = dict(env)
            if checkpoints:
                cp = checkpoints[-1]
                if not (cp/'rgpa_state.json').exists() or not (cp/'optimizer.pt').exists():
                    raise ValueError('Incomplete checkpoint; explicit recovery required')
                stage_env['MED_PRISM_RGPA_RESUME'] = str(cp)
            started = time.monotonic()
            call(['bash', stage_script, '--method-version', '1.2', '--ablation-name', 'no_geo_orth',
                  '--orth-lambda', '0', '--output-root', root, '--gpus', a.gpus,
                  '--start-stage', stage, '--end-stage', stage], root/f'train_task{stage}.log', stage_env)
            write_json(root/f'train_task{stage}_timing.json', dict(elapsed_seconds=time.monotonic()-started))
        if read(stage_root/'completion.json')['status'] != 'PASS':
            raise ValueError('Training completion failed')
        for task in range(1, stage + 1):
            call([sys.executable, REPO/'scripts/medicalskill_v1_2_med_prism/evaluate_formal_v1_2_cell.py',
                  '--stage', stage, '--mode', 'primary_cumulative', '--task', task,
                  '--batch-size', '4'], root/f'eval_stage{stage}_task{task}.log', boundary_env)
        # T3 sidecar is legal at its boundary and allows later T4 without old-data revisits.
        if not (root/f'rgpa/task_{stage:02d}.json').exists():
            boundary(root, stage, boundary_env, slack=not a.without_slack)
    if a.end_stage == 3:
        call([sys.executable, REPO/'scripts/medicalskill_v1_2_med_prism/evaluate_formal_v1_2_cell.py',
              '--stage', '1', '--mode', 'primary_cumulative', '--task', '1', '--batch-size', '4'],
             root/'eval_stage1_task1.log', boundary_env)
        report(root)
    write_json(root/'run_completion.json', dict(status='PASS', end_stage=a.end_stage))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
