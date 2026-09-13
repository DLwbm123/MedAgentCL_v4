#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/MedAgentCL_v4
OUT=${1:?Usage: launch_detached.sh OUTPUT_ROOT [GPUS] [EXISTING_TIMING_PID]}
GPUS=${2:-0,1}
TIMING_PID=${3:-0}
[[ "$OUT" == /remote-home/wangbomin/MedPRISM_RGPA_* && -f "$OUT/rgpa_contract.json" ]] || exit 2
[[ "$TIMING_PID" =~ ^[0-9]+$ && "$GPUS" =~ ^[01](,[01])?$ ]] || exit 2
[[ ! -e "$OUT/run.pid" && ! -e "$OUT/run.log" ]] || { echo 'Existing launch artifacts; refusing overwrite' >&2; exit 3; }
nohup bash "$ROOT/scripts/medicalskill_v2_4_rgpa/detached_worker.sh" "$OUT" "$GPUS" "$TIMING_PID" > "$OUT/run.log" 2>&1 < /dev/null &
RGPA_JOB_PID=$!
printf '%s\n' "$RGPA_JOB_PID" > "$OUT/run.pid"
printf 'PID=%s\nOUTPUT=%s\nLOG=%s/run.log\n' "$RGPA_JOB_PID" "$OUT" "$OUT"
