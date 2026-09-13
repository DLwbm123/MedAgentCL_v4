#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/MedAgentCL_v4
TRAIN_SCRIPT=$ROOT/scripts/medicalskill_v1_2_med_prism/run_med_prism_v1_2_train.sh
DATA_ROOT=/remote-home/wangbomin/MedicalSkill-CL-v1.2
OUTPUT_ROOT=/remote-home/wangbomin/med_prism_medicalskill_v1_2_seed42
GPUS=0,1
SEED=42
POLL_SECONDS=20
MAX_WAIT_SECONDS=86400
FREE_MEMORY_USED_MIB=1000
FREE_UTILIZATION_PERCENT=10

started=$(date +%s)
consecutive=0
echo "[$(date -Is)] waiting for GPUs $GPUS to become idle"
while true; do
  now=$(date +%s)
  if (( now - started >= MAX_WAIT_SECONDS )); then
    echo "[$(date -Is)] timeout waiting for GPUs" >&2
    exit 4
  fi
  ready=1
  snapshot=()
  IFS=',' read -r -a ids <<< "$GPUS"
  for gpu in "${ids[@]}"; do
    line=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)
    IFS=',' read -r used free util <<< "$line"
    used=${used// /}; free=${free// /}; util=${util// /}
    snapshot+=("gpu${gpu}:used=${used}MiB,free=${free}MiB,util=${util}%")
    if (( used >= FREE_MEMORY_USED_MIB || util >= FREE_UTILIZATION_PERCENT )); then
      ready=0
    fi
  done
  echo "[$(date -Is)] ${snapshot[*]} consecutive_ready=$consecutive"
  if (( ready == 1 )); then
    consecutive=$((consecutive + 1))
  else
    consecutive=0
  fi
  if (( consecutive >= 3 )); then
    break
  fi
  sleep "$POLL_SECONDS"
done

if pgrep -af 'run_med_prism_v1_2_train.sh' | grep -v "$$" | grep -q .; then
  echo "Another formal Med-PRISM launcher is already running" >&2
  pgrep -af 'run_med_prism_v1_2_train.sh' >&2 || true
  exit 5
fi

echo "[$(date -Is)] both GPUs are idle; starting two-GPU DDP training"
exec bash "$TRAIN_SCRIPT" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --gpus "$GPUS" \
  --seed "$SEED" \
  --start-stage 1 \
  --end-stage 5 \
  --restart-incomplete-stage