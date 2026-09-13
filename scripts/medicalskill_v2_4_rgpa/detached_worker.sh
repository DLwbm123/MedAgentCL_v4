#!/usr/bin/env bash
set -Eeuo pipefail
OUT=${1:?}
GPUS=${2:?}
TIMING_PID=${3:-0}
trap 'RGPA_EXIT_CODE=$?; printf "%s\n" "$RGPA_EXIT_CODE" > "$OUT/run.exit"' EXIT
echo "[$(date -Is)] RGPA-A pipeline: root=$OUT GPUs=$GPUS"
if (( TIMING_PID > 0 )); then
  echo "Waiting for the already-running 8-sample timing process $TIMING_PID. No formal training before the runtime guard passes."
  while [[ -r "/proc/$TIMING_PID/cmdline" ]]; do
    RGPA_DEPENDENCY_CMD=$(tr '\0' ' ' < "/proc/$TIMING_PID/cmdline" 2>/dev/null || true)
    [[ "$RGPA_DEPENDENCY_CMD" == *med_prism.rgpa.boundary* && "$RGPA_DEPENDENCY_CMD" == *"$OUT"* ]] || break
    sleep 10
  done
fi
echo "[$(date -Is)] Starting guarded sequential runner"
bash /root/MedAgentCL_v4/scripts/medicalskill_v2_4_rgpa/run.sh --output-root "$OUT" --gpus "$GPUS"
