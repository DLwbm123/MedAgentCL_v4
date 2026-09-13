#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/MedAgentCL_v4
exec bash scripts/medicalskill_v1_2_med_prism_versions/run_versioned_train.sh \
  --method-version 1.1 \
  --data-root /remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k \
  --output-root /remote-home/wangbomin/med_prism_v1.1_medicalskill_v1_2_lite_10k1k_seed42 \
  --shared-lr 1e-5 --private-lr 1e-4 \
  --lambda-drift 0.01 --orth-lambda 0.1 \
  "$@"
