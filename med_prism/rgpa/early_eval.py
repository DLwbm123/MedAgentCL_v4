"""GPU1 lite cells for finished stages; never signal the training process group."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import sys
import time

from med_prism.rgpa.separated import (REPO, call, eval_env, gpu_available,
                                    legacy_state, read, sha256_file,
                                    validate_cell, write_json)

CELLS = [(1, 1), (2, 1), (2, 2)]


@contextmanager
def hold_scheduler(pid, root, output):
    cmd = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    if not all(x in cmd for x in ('med_prism.rgpa.separated', str(root), str(output), 'all')):
        raise ValueError('Refusing to signal an unrelated process')
    try:
        os.kill(pid, signal.SIGSTOP)  # Single coordinator PID, NOT its children.
        for _ in range(100):
            status = Path(f'/proc/{pid}/status').read_text()
            if any(line.startswith('State:') and 'T' in line for line in status.splitlines()):
                break
            time.sleep(.01)
        else:
            raise RuntimeError('Coordinator did not stop')
        if (output/'training_complete.json').exists():
            raise RuntimeError('Main evaluation may already be active; refusing duplicate work')
        yield
    finally:
        os.kill(pid, signal.SIGCONT)
        print(f'RESUMED coordinator PID {pid}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--coordinator-pid', type=int, required=True)
    a = p.parse_args()
    root, output = a.root.resolve(), a.output.resolve()
    if output.parent != root or not output.name.startswith('evaluation_lite_'):
        p.error('Expected existing dedicated lite evaluation directory')
    if not gpu_available(1):
        raise RuntimeError('GPU1 occupied; no process was paused')
    for stage in (1, 2):
        legacy_state(root, stage)
        spec = read(root/'run_manifest.json')['data']['tasks'][str(stage)]
        if sha256_file(spec['test']) != spec['test_sha256']:
            raise ValueError('Lite test hash mismatch')
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with hold_scheduler(a.coordinator_pid, root, output):
            write_json(output/'early_gpu1_started.json', dict(pid=os.getpid(),
                coordinator_pid=a.coordinator_pid, cells=CELLS, gpu=1,
                source_sha256=sha256_file(__file__), note='Training children remain running'))
            for stage, task in CELLS:
                summary = output/f'stage_{stage:02d}/primary_cumulative_task_{task:02d}.summary.json'
                if not summary.exists():
                    while not gpu_available(1):
                        time.sleep(30)
                    print(f'EVAL GPU1: stage{stage}/task{task}', flush=True)
                    call([sys.executable, REPO/'scripts/medicalskill_v1_2_med_prism/evaluate_formal_v1_2_cell.py',
                          '--stage', stage, '--task', task, '--mode', 'primary_cumulative',
                          '--batch-size', '4', '--eval-limit', '0'],
                         output/f'early_s{stage}_t{task}_gpu1.log', eval_env(root, output, 1))
                validate_cell(root, output, stage, task)
                print(f'PASS R{stage}{task}', flush=True)
        write_json(output/'early_gpu1_completion.json', dict(status='PASS', cells=CELLS))
    except BaseException as exc:
        write_json(output/'early_gpu1_completion.json', dict(status='FAIL', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
