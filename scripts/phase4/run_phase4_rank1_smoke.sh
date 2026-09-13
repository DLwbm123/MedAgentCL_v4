#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/MedAgentCL_v4
OUT=$ROOT/output/phase4_rank1
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset TRANSFORMERS_CACHE

cd "$ROOT"
mkdir -p "$OUT"
df -h "$HF_HOME" > "$OUT/cache_filesystem_before_phase4.txt"
"$PYTHON" -m unittest discover -s tests/phase4 -v 2>&1 | tee "$OUT/cpu_tests.log"

bash "$ROOT/scripts/phase4/train_phase4_rank1_task.sh" \
  1 "$OUT/task_01"
test -f "$OUT/task_01/train/checkpoint-5/rank1_manifest.json"

bash "$ROOT/scripts/phase4/train_phase4_rank1_task.sh" \
  2 "$OUT/task_02"
test -f "$OUT/task_02/train/checkpoint-5/rank1_manifest.json"

"$PYTHON" "$ROOT/scripts/phase4/validate_phase4_checkpoint.py" \
  --manifest "$OUT/task_02/train/checkpoint-5/rank1_manifest.json" \
  --output "$OUT/task_02/real_reload_audit.json" \
  --load-model

"$PYTHON" "$ROOT/scripts/phase4/build_phase4_report.py" \
  --output-root "$OUT" \
  --json-output "$OUT/phase4_summary.json" \
  --markdown-output "$ROOT/docs/phase4_smoke_report.md"

echo "Phase 4 smoke completed: $OUT/phase4_summary.json"
