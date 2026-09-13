#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/MedAgentCL_v4
exec /root/anaconda3/envs/medagentcl_v4/bin/python \
  scripts/medicalskill_v1_2_med_prism/aggregate_formal_v1_2.py \
  --data-root /remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k \
  --output-root /remote-home/wangbomin/med_prism_v1.1_medicalskill_v1_2_lite_10k1k_seed42 \
  "$@"
