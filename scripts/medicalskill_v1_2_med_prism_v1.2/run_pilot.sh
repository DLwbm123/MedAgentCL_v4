#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/MedAgentCL_v4
PILOT_ROOT=/remote-home/wangbomin/med_prism_v1.2_medicalskill_v1_2_pilot_seed42
CHECK_ONLY=0
for ARG in "$@"; do [[ "$ARG" == "--check-only" ]] && CHECK_ONLY=1; done
cd "$ROOT"
/root/anaconda3/envs/medagentcl_v4/bin/python scripts/medicalskill_v1_2_med_prism_versions/prepare_pilot.py \
  --source-data-root /remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k \
  --output-data-root "$PILOT_ROOT/data" --seed 42 \
  --pilot-train-limit 1000 --pilot-diagnostic-limit 200 \
  $([[ "$CHECK_ONLY" == 1 ]] && printf '%s' '--check-only')
if (( CHECK_ONLY == 1 )); then exit 0; fi
bash scripts/medicalskill_v1_2_med_prism_versions/run_versioned_train.sh \
  --method-version 1.2 --pilot-mode --data-root "$PILOT_ROOT/data" \
  --output-root "$PILOT_ROOT/run" --gpus 0,1 --end-stage 5 \
  --shared-lr 1e-5 --private-lr 1e-4 --lambda-drift 0.01 --orth-lambda 0.1 --lambda-key 0.1 "$@"
bash scripts/medicalskill_v1_2_med_prism_versions/run_pilot_evaluation.sh \
  --method-version 1.2 --data-root "$PILOT_ROOT/data" --output-root "$PILOT_ROOT/run" --gpus 0,1
