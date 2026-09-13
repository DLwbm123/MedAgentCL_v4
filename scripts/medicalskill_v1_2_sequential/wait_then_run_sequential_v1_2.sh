#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/MedAgentCL_v4
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
MEDPRISM_OUT=/remote-home/wangbomin/med_prism_medicalskill_v1_2_lite_10k1k_seed42
SEQUENTIAL_OUT=/remote-home/wangbomin/sequential_medicalskill_v1_2_lite_10k1k_seed42
RUNNER="$ROOT/scripts/medicalskill_v1_2_sequential/run_sequential_v1_2.sh"
LOG="$SEQUENTIAL_OUT/wait_then_run.log"
THRESHOLD_MIB=30720
POLL_SECONDS=30
REQUIRED_CONSECUTIVE=3

mkdir -p "$SEQUENTIAL_OUT"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] waiting for Med-PRISM test PASS and both GPUs >30 GiB free"
consecutive=0
while true; do
  summary="$MEDPRISM_OUT/formal_experiment_summary.json"
  test_pass=0
  if [[ -s "$summary" ]] && "$PYTHON" -c 'import json,sys;raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="PASS" else 1)' "$summary"; then
    test_pass=1
  fi
  mapfile -t free_mib < <(nvidia-smi -i 0,1 --query-gpu=memory.free --format=csv,noheader,nounits)
  gpu0=${free_mib[0]// /}; gpu1=${free_mib[1]// /}
  if (( test_pass == 1 && gpu0 > THRESHOLD_MIB && gpu1 > THRESHOLD_MIB )); then
    consecutive=$((consecutive + 1))
  else
    consecutive=0
  fi
  echo "[$(date -Is)] medprism_test_pass=$test_pass gpu0_free=${gpu0}MiB gpu1_free=${gpu1}MiB ready=$consecutive/$REQUIRED_CONSECUTIVE"
  (( consecutive >= REQUIRED_CONSECUTIVE )) && break
  sleep "$POLL_SECONDS"
done
echo "[$(date -Is)] launching sequential baseline"
exec bash "$RUNNER" \
  --data-root /remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k \
  --output-root "$SEQUENTIAL_OUT" \
  --gpus 0,1 --seed 42 --eval-limit 0 --restart-incomplete-stage
