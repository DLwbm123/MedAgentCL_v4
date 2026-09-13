#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/MedAgentCL_v4
RUN=/remote-home/wangbomin/med_prism_medicalskill_v1_2_seed42
TEST_SCRIPT=$ROOT/scripts/medicalskill_v1_2_med_prism/run_med_prism_v1_2_test.sh
POLL_SECONDS=60
MAX_WAIT_SECONDS=604800
started=$(date +%s)

echo "[$(date -Is)] waiting for five PASS training completions"
while true; do
  if (( $(date +%s) - started >= MAX_WAIT_SECONDS )); then
    echo "[$(date -Is)] timeout waiting for training" >&2
    exit 4
  fi
  if /root/anaconda3/envs/medagentcl_v4/bin/python - "$RUN" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for stage in range(1,6):
 p=root/'med_prism'/f'stage_{stage:02d}'/'completion.json'
 if not p.is_file() or json.loads(p.read_text()).get('status')!='PASS':
  raise SystemExit(1)
PY
  then
    break
  fi
  completed=0
  for stage in 01 02 03 04 05; do
    if test -s "$RUN/med_prism/stage_$stage/completion.json"; then completed=$((completed+1)); fi
  done
  echo "[$(date -Is)] training completions present: $completed/5"
  sleep "$POLL_SECONDS"
done

echo "[$(date -Is)] all five stages PASS; waiting for GPUs 0 and 1 to become idle"
consecutive=0
while (( consecutive < 3 )); do
  all_ready=1
  for gpu in 0 1; do
    line=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)
    IFS=',' read -r used free util <<< "$line"
    used=${used// /}; free=${free// /}; util=${util// /}
    echo "[$(date -Is)] gpu${gpu}:used=${used}MiB,free=${free}MiB,util=${util}% consecutive_ready=$consecutive"
    if (( used >= 1000 || util >= 10 )); then all_ready=0; fi
  done
  if (( all_ready == 1 )); then consecutive=$((consecutive+1)); else consecutive=0; fi
  (( consecutive >= 3 )) || sleep 20
done

echo "[$(date -Is)] starting complete formal evaluation"
exec bash "$TEST_SCRIPT" \
  --data-root /remote-home/wangbomin/MedicalSkill-CL-v1.2 \
  --output-root "$RUN" \
  --gpus 0,1 \
  --eval-limit 0