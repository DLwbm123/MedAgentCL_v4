#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/MedAgentCL_v4
ARTIFACT="$ROOT/artifacts/phase6a"
OUTPUT=/remote-home/wangbomin/medagentcl_v4_phase6a/pilot
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub
export HF_DATASETS_CACHE=/remote-home/wangbomin/huggingface_cache/datasets
export XDG_CACHE_HOME=/remote-home/wangbomin/huggingface_cache/xdg
export TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
export MODELSCOPE_CACHE=/remote-home/wangbomin/huggingface_cache/modelscope
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset TRANSFORMERS_CACHE

cd "$ROOT"
mkdir -p "$ARTIFACT/predictions" "$OUTPUT"

train_task() {
  local task=$1
  local exit_file="$OUTPUT/runs/task_$task/exit_code.json"
  if [[ -f "$exit_file" ]] && "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("swift_sft_exit_code")==0 and d.get("shared_manifest_exists") and d.get("private_manifest_exists") else 1)' "$exit_file"
  then
    echo "SKIP completed pilot training task $task"
    return
  fi
  bash scripts/phase6a/train_shared_private_task.sh pilot "$task"
}

evaluate_cell() {
  local mode=$1
  local after_task=$2
  local eval_task=$3
  local stem="$mode"_after_task"$after_task"_eval_task"$eval_task"
  local prediction="$ARTIFACT/predictions/$stem.jsonl"
  local summary="$ARTIFACT/predictions/$stem.summary.json"
  if [[ -f "$summary" ]] && "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("status")=="PASS" else 1)' "$summary"
  then
    echo "SKIP completed evaluation $stem"
    return
  fi
  "$PYTHON" scripts/phase6a/evaluate_shared_private.py --mode "$mode" --after-task "$after_task" --eval-task "$eval_task" --output "$prediction"
}

train_task 1
evaluate_cell primary_cumulative 1 1
train_task 2
evaluate_cell primary_cumulative 2 1
evaluate_cell primary_cumulative 2 2
evaluate_cell oracle_skill_aware 2 1
evaluate_cell oracle_skill_aware 2 2
"$PYTHON" scripts/phase6a/build_pilot_matrices.py

echo "Phase 6A pilot completed"
