#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/MedAgentCL_v4
OUTPUT=/remote-home/wangbomin/medagentcl_v4_phase6b
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python

if [[ $# -ne 1 || ! "$1" =~ ^(prepare|base_zero_shot|sequential_native_lora_r48|pure_rank1_e16|shared32_private16|all)$ ]]; then
  echo "Usage: $0 {prepare|base_zero_shot|sequential_native_lora_r48|pure_rank1_e16|shared32_private16|all}" >&2
  exit 2
fi
REQUEST=$1

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub
export HF_DATASETS_CACHE=/remote-home/wangbomin/huggingface_cache/datasets
export XDG_CACHE_HOME=/remote-home/wangbomin/huggingface_cache/xdg
export TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset TRANSFORMERS_CACHE

cd "$ROOT"
if [[ ! -f "$ROOT/artifacts/phase6a/phase6b_data_summary.json" ]]; then
  "$PYTHON" scripts/phase6a/prepare_phase6b_data.py
fi
for task in 1 2; do
  test -s "$OUTPUT/data/task_${task}_train.jsonl"
  test -s "$OUTPUT/data/task_${task}_test.jsonl"
done
if [[ "$REQUEST" == prepare ]]; then
  exit 0
fi

summary_complete() {
  local summary=$1
  local dataset=$2
  [[ -f "$summary" ]] && "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); n=sum(1 for line in open(sys.argv[2]) if line.strip()); raise SystemExit(0 if d.get("status")=="PASS" and d.get("total")==n else 1)' "$summary" "$dataset"
}

evaluate_cell() {
  local method=$1
  local mode=$2
  local after_task=$3
  local eval_task=$4
  local dataset="$OUTPUT/data/task_${eval_task}_test.jsonl"
  local predictions="$OUTPUT/$method/predictions"
  local stem
  if [[ "$method" == base_zero_shot ]]; then
    stem="zero_shot_eval_task$eval_task"
  else
    stem="${mode}_after_task${after_task}_eval_task${eval_task}"
  fi
  local output="$predictions/$stem.jsonl"
  local summary="$predictions/$stem.summary.json"
  if summary_complete "$summary" "$dataset"; then
    echo "SKIP completed evaluation $method $stem"
    return
  fi
  mkdir -p "$predictions"
  local args=(
    --method "$method"
    --mode "$mode"
    --after-task "$after_task"
    --eval-task "$eval_task"
    --dataset "$dataset"
    --output "$output"
  )
  case "$method" in
    base_zero_shot)
      ;;
    sequential_native_lora_r48)
      local adapter
      adapter=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$OUTPUT/$method/task_$after_task/completion.json")
      args+=(--adapter "$adapter")
      ;;
    pure_rank1_e16)
      local checkpoint
      checkpoint=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$OUTPUT/$method/task_$after_task/completion.json")
      args+=(--rank1-manifest "$checkpoint/rank1_manifest.json")
      ;;
    shared32_private16)
      args+=(--shared-manifest "$OUTPUT/$method/shared/after_task_$after_task/shared_manifest.json")
      if [[ "$mode" == primary_cumulative ]]; then
        for task in $(seq 1 "$after_task"); do
          args+=(--private-manifest "$OUTPUT/$method/private/task_$task/private_manifest.json")
        done
      else
        args+=(--private-manifest "$OUTPUT/$method/private/task_$eval_task/private_manifest.json")
      fi
      ;;
  esac
  "$PYTHON" scripts/phase6a/evaluate_phase6b.py "${args[@]}"
}

run_method() {
  local method=$1
  if [[ "$method" == base_zero_shot ]]; then
    evaluate_cell "$method" primary_cumulative 0 1
    evaluate_cell "$method" primary_cumulative 0 2
  else
    bash scripts/phase6a/train_phase6b_task.sh "$method" 1
    evaluate_cell "$method" primary_cumulative 1 1
    bash scripts/phase6a/train_phase6b_task.sh "$method" 2
    evaluate_cell "$method" primary_cumulative 2 1
    evaluate_cell "$method" primary_cumulative 2 2
    if [[ "$method" == shared32_private16 ]]; then
      evaluate_cell "$method" oracle_skill_aware 2 1
      evaluate_cell "$method" oracle_skill_aware 2 2
    fi
  fi
  "$PYTHON" scripts/phase6a/build_phase6b_matrices.py --method "$method"
}

if [[ "$REQUEST" == all ]]; then
  for method in base_zero_shot sequential_native_lora_r48 pure_rank1_e16 shared32_private16; do
    run_method "$method"
  done
else
  run_method "$REQUEST"
fi
