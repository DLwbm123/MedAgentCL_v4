#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/MedAgentCL_v4
ARTIFACT=$ROOT/artifacts/medicalskill_cl_v1_1_and_medprism_smoke
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub
export HF_DATASETS_CACHE=/remote-home/wangbomin/huggingface_cache/datasets
export XDG_CACHE_HOME=/remote-home/wangbomin/huggingface_cache/xdg
export TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset TRANSFORMERS_CACHE
cd "$ROOT"

for closure in concept_v3_summary.json cap8_vs_cap12_pilot.json cross_split_phash_only_summary.json; do test -f "$ARTIFACT/$closure"; done
test -f /remote-home/wangbomin/MedicalSkill-CL-v1.1/audits/strict_validation_v1_1.json

"$PYTHON" scripts/med_prism_real_5skill/prepare_real_5stage_smoke.py
"$PYTHON" scripts/med_prism_real_5skill/evaluate_real_5stage.py --stage 0
for stage in 1 2 3 4 5; do
  bash scripts/med_prism_real_5skill/train_real_5stage_task.sh "$stage"
  "$PYTHON" scripts/med_prism_real_5skill/evaluate_real_5stage.py --stage "$stage"
done
"$PYTHON" scripts/med_prism_real_5skill/aggregate_real_5stage.py
