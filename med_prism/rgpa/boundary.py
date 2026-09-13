"""Source-task-only finite group removal. No sample-level state is exported."""
import argparse
import json
from pathlib import Path
import time

import torch
from med_prism.ms_crc.data import encode
from med_prism.ms_crc.evaluator import token_scores
from med_prism.transport.calibration import ordered_rows
from med_prism.transport.checkpoint import legacy_state, sha256_file
from med_prism.transport.cli import load_calibration_model
from med_prism.transport.banks import fingerprint
from med_prism.rgpa.core import allocation, mapping, digest, group_off
from med_prism.rgpa.state import write_json


def measure(model, template, path, task, count):
    from swift.template import MaxLengthError
    rows = ordered_rows([json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()], 42)
    total = torch.zeros(16, dtype=torch.float32)
    valid = skipped = 0
    started = time.monotonic()
    for row in rows:
        try:
            batch = encode(row, template)
        except MaxLengthError:
            skipped += 1
            continue
        except ValueError as exc:
            if 'No answer tokens' not in str(exc):
                raise
            skipped += 1
            continue
        scores, logits = token_scores(model, batch)
        reference = scores['nll_sum'] / scores['answer_tokens']
        del logits
        for group in range(16):
            with group_off(model, task, group):
                scores, logits = token_scores(model, batch)
            total[group] += max(scores['nll_sum'] / scores['answer_tokens'] - reference, 0.)
            del logits
        valid += 1
        if valid == count:
            break
    if not valid:
        raise ValueError('No valid source-task answers')
    return total / valid, valid, skipped, time.monotonic() - started


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task', type=int, choices=range(1, 6), required=True)
    p.add_argument('--timing-only', action='store_true')
    p.add_argument('--without-slack', action='store_true')
    a = p.parse_args()
    manifest = json.loads((a.root / 'run_manifest.json').read_text())
    dataset = Path(manifest['data']['tasks'][str(a.task)]['train'])
    if dataset.name != 'train.jsonl' or sha256_file(dataset) != manifest['data']['tasks'][str(a.task)]['train_sha256']:
        raise ValueError('Only the hash-locked current task TRAIN is permitted')
    target = a.root / 'rgpa' / (f'timing_task_{a.task:02d}.json' if a.timing_only else f'task_{a.task:02d}.json')
    if target.exists():
        raise FileExistsError(target)
    state = legacy_state(a.root, a.task)
    torch.manual_seed(42)
    model, template = load_calibration_model(state, 4096 if a.task == 5 else 1024)
    template.processor.image_processor.min_pixels = 200704
    template.processor.image_processor.max_pixels = 200704
    before = fingerprint(model)
    salience, valid, skipped, seconds = measure(model, template, dataset, a.task, 8 if a.timing_only else 64)
    if fingerprint(model) != before:
        raise RuntimeError('Boundary modified persistent model tensors')
    common = dict(valid_sample_count=valid, skipped_count=skipped, elapsed_seconds=seconds,
                  persistent_tensor_invariant='PASS', dataset_hash=sha256_file(dataset), calibration_seed=42)
    if a.timing_only:
        # Timing samples NEVER become a formal salience sidecar.
        write_json(target, dict(**common, timing_only=True, estimated_64_seconds=seconds / valid * 64))
    else:
        s, z, w = allocation(salience)
        write_json(target, dict(**common, method_version='2.4', rho=.25, slack_enabled=not a.without_slack,
                    num_groups=16, salience=s.tolist(), z=z.tolist(), w=w.tolist(),
                    group_map_hash=digest(mapping(model, a.task)),
                    source_private_sha256=sha256_file(state.private_manifests[-1]),
                    min_weight=float(w.min()), max_weight=float(w.max()), mean_weight=float(w.mean())))
    print(json.dumps(dict(output=str(target), **common)), flush=True)


if __name__ == '__main__':
    main()
