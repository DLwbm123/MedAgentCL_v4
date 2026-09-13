#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/MedAgentCL_v4
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
OUTPUT_ROOT=/remote-home/wangbomin/med_prism_medicalskill_v1_2_lite_10k1k_pe32_pa32_orth1_seed42
TEST_SCRIPT="$ROOT/scripts/medicalskill_v1_2_med_prism/run_med_prism_v1_2_pe32_pa32_orth1_test.sh"
LOG="$OUTPUT_ROOT/auto_primary_test_waiter.log"
POLL_SECONDS=${POLL_SECONDS:-60}
FREE_THRESHOLD_MIB=${FREE_THRESHOLD_MIB:-30720}
REQUIRED_CONSECUTIVE=${REQUIRED_CONSECUTIVE:-3}

mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.auto_primary_test_waiter.lock"
if ! flock -n 9; then
  echo "Another tuning test waiter is already running." >&2
  exit 2
fi
exec >>"$LOG" 2>&1

echo "[$(date -Is)] waiter started"
echo "[$(date -Is)] output=$OUTPUT_ROOT threshold=${FREE_THRESHOLD_MIB}MiB consecutive=$REQUIRED_CONSECUTIVE"

test_already_passed() {
  "$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = root / "formal_experiment_summary.json"
config = root / "evaluation_run_config.json"
if not summary.is_file() or not config.is_file():
    raise SystemExit(1)
s = json.loads(summary.read_text())
c = json.loads(config.read_text())
ok = (
    s.get("status") == "PASS"
    and c.get("generation_cell_count") == 15
    and c.get("oracle_evaluation_enabled") is False
    and c.get("stage0_evaluation_enabled") is False
    and c.get("base_zero_shot_reused") is True
)
raise SystemExit(0 if ok else 1)
PY
}

training_contract_passed() {
  "$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for stage in range(1, 6):
    padded = f"{stage:02d}"
    completion = root / "med_prism" / f"stage_{padded}" / "completion.json"
    config = root / "configs" / f"stage_{padded}.json"
    if not completion.is_file() or not config.is_file():
        raise SystemExit(1)
    done = json.loads(completion.read_text())
    cfg = json.loads(config.read_text())
    if done.get("status") != "PASS":
        raise SystemExit(1)
    if not (
        cfg.get("private_experts_per_task") == 32
        and float(cfg.get("private_alpha", -1)) == 32.0
        and float(cfg.get("orth_lambda", -1)) == 1.0
    ):
        raise SystemExit(2)
raise SystemExit(0)
PY
}

training_process_alive() {
  local pid args
  while read -r pid args; do
    [[ "$pid" == "$$" ]] && continue
    if [[ "$args" == *"$OUTPUT_ROOT"* ]] && \
       [[ "$args" == *"run_med_prism_v1_2_train.sh"* || \
          "$args" == *"swift sft"* || "$args" == *"swift/cli/sft.py"* ]]; then
      return 0
    fi
  done < <(ps -eo pid=,args=)
  return 1
}

if test_already_passed; then
  echo "[$(date -Is)] Primary 15-cell evaluation already PASS; nothing to do."
  exit 0
fi

ready_count=0
while true; do
  contract_pass=0
  training_contract_passed && contract_pass=1

  mapfile -t free_mib < <(
    nvidia-smi -i 0,1 --query-gpu=memory.free --format=csv,noheader,nounits
  )
  if (( ${#free_mib[@]} != 2 )); then
    echo "[$(date -Is)] unable to read both GPU free-memory values"
    exit 3
  fi
  gpu0=${free_mib[0]// /}
  gpu1=${free_mib[1]// /}

  if (( contract_pass == 1 && gpu0 > FREE_THRESHOLD_MIB && gpu1 > FREE_THRESHOLD_MIB )); then
    ready_count=$((ready_count + 1))
  else
    ready_count=0
  fi
  echo "[$(date -Is)] training_contract_pass=$contract_pass gpu0_free=${gpu0}MiB gpu1_free=${gpu1}MiB ready=$ready_count/$REQUIRED_CONSECUTIVE"

  if (( ready_count >= REQUIRED_CONSECUTIVE )); then
    break
  fi
  if (( contract_pass == 0 )) && ! training_process_alive; then
    echo "[$(date -Is)] ERROR: training process stopped before all five stages passed."
    echo "[$(date -Is)] Test was not started; inspect the latest stage train.log."
    exit 4
  fi
  sleep "$POLL_SECONDS"
done

echo "[$(date -Is)] all stages PASS and both GPUs are stably free; starting Primary 15-cell test"
exec bash "$TEST_SCRIPT"
