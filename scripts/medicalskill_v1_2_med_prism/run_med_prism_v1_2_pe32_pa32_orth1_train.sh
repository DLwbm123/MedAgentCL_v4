#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/MedAgentCL_v4
BASE_TRAIN="$ROOT/scripts/medicalskill_v1_2_med_prism/run_med_prism_v1_2_train.sh"

# Frozen tuning contract. These are passed explicitly to the generic trainer
# and are persisted in run_manifest.json plus every configs/stage_XX.json.
PRIVATE_EXPERTS_PER_TASK=32
PRIVATE_ALPHA=32
ORTH_LAMBDA=1

DATA_ROOT=/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k
OUTPUT_ROOT=/remote-home/wangbomin/med_prism_medicalskill_v1_2_lite_10k1k_pe32_pa32_orth1_seed42
GPUS=0,1
SEED=42

exec bash "$BASE_TRAIN" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --gpus "$GPUS" \
  --seed "$SEED" \
  --private-experts-per-task "$PRIVATE_EXPERTS_PER_TASK" \
  --private-alpha "$PRIVATE_ALPHA" \
  --orth-lambda "$ORTH_LAMBDA" \
  "$@"
