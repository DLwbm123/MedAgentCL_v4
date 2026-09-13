#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/MedAgentCL_v4
PY=/root/anaconda3/envs/medagentcl_v4/bin/python
DATA=/remote-home/wangbomin/MedicalSkill-CL-v1.2

task_id=${1:?Usage: medicalskill_v1_2_evaluate_task.sh TASK_ID PREDICTIONS_JSONL OUTPUT_JSON}
predictions=${2:?Missing predictions JSONL}
output=${3:?Missing output JSON}

exec "$PY" "$ROOT/scripts/skill_incremental_v1/evaluate_medicalskill_predictions_v1.py" \
  --task "$task_id" \
  --predictions "$predictions" \
  --data-root "$DATA" \
  --output "$output"
