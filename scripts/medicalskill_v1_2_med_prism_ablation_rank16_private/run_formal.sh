#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/MedAgentCL_v4
cd "$ROOT"
bash scripts/medicalskill_v1_2_med_prism_ablation_rank16_private/run_train.sh
bash scripts/medicalskill_v1_2_med_prism_ablation_rank16_private/run_test.sh
