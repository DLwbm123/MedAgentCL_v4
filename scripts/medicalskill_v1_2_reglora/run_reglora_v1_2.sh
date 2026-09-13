#!/usr/bin/env bash
set -Eeuo pipefail
exec /root/MedAgentCL_v4/scripts/medicalskill_v1_2_wave1/run_wave1_v1_2.sh --method reglora --output-root /remote-home/wangbomin/reglora_r16_medicalskill_v1_2_lite_10k1k_seed42 "$@"
