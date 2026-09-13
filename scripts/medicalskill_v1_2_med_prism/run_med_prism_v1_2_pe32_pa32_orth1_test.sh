#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/MedAgentCL_v4
BASE_TEST="$ROOT/scripts/medicalskill_v1_2_med_prism/run_med_prism_v1_2_test.sh"

DATA_ROOT=/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k
OUTPUT_ROOT=/remote-home/wangbomin/med_prism_medicalskill_v1_2_lite_10k1k_pe32_pa32_orth1_seed42
BASE_EVAL_ROOT=/remote-home/wangbomin/med_prism_medicalskill_v1_2_lite_10k1k_seed42
GPUS=0,1

# No --with-oracle and no --with-stage0: reuse the canonical Stage 0 and
# generate only the 15 Primary lower-triangular cells.
exec bash "$BASE_TEST" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --base-eval-root "$BASE_EVAL_ROOT" \
  --gpus "$GPUS" \
  --batch-size 4 \
  "$@"
