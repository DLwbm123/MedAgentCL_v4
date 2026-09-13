#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/MedAgentCL_v4
exec bash scripts/medicalskill_v1_2_med_prism/run_med_prism_v1_2_test.sh \
  --data-root /remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k \
  --output-root /remote-home/wangbomin/med_prism_v1_2_ablation_rank16_private_medicalskill_v1_2_lite_10k1k_seed42 \
  --base-eval-root /remote-home/wangbomin/med_prism_medicalskill_v1_2_lite_10k1k_seed42 \
  --gpus 0,1 "$@"
