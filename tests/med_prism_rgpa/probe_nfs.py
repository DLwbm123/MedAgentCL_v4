"""Bounded I/O diagnosis on a fresh artifact, never alters source files."""
from pathlib import Path
import hashlib
import os
import shutil
import tempfile
import traceback

source = Path('/remote-home/wangbomin/med_prism_v1_2_ablation_no_geo_orth_medicalskill_v1_2_lite_10k1k_seed42/med_prism/stage_01/shared/shared_manifest.json')
root = Path(tempfile.mkdtemp(prefix='MedPRISM_RGPA_IO_probe_', dir='/remote-home/wangbomin'))
target = root / source.name
print('Probe artifact:', root, flush=True)
for name, operation in (
    ('copy_file_contents', lambda: shutil.copyfile(source, target)),
    ('readback_hash', lambda: hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(source.read_bytes()).hexdigest()),
    ('copy_metadata', lambda: shutil.copystat(source, target)),
):
    try:
        print(name, 'PASS', operation(), flush=True)
    except Exception:
        print(name, 'FAIL', flush=True)
        traceback.print_exc()
