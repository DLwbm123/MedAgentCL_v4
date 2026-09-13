#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/MedAgentCL_v4
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
SWIFT=/root/anaconda3/envs/medagentcl_v4/bin/swift
DATA_ROOT=/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k
OUTPUT_ROOT=/remote-home/wangbomin/replay_lora_medicalskill_v1_2_lite_10k1k_seed42
GPUS=0,1
SEED=42
RANK=48
REPLAY_BUDGET=1000
BUFFER_POLICY=balanced_per_task_stratified_stable_sha256_v1
MAX_STAGES=5
TRAIN_LIMIT=0
EVAL_LIMIT=0
CHECK_ONLY=0
REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b

usage() {
  cat <<'EOF'
Usage: run_replay_lora_v1_2.sh [options]
  --data-root PATH --output-root PATH --gpus 0,1 --seed 42
  --rank 48 --replay-budget {100,500,1000}
  --buffer-policy balanced_per_task_stratified_stable_sha256_v1
  --max-stages N --train-limit N --eval-limit N --check-only
One epoch is one pass over full current-task data plus the bounded historical buffer.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT=$2; shift 2;; --output-root) OUTPUT_ROOT=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;; --seed) SEED=$2; shift 2;; --rank) RANK=$2; shift 2;;
    --replay-budget) REPLAY_BUDGET=$2; shift 2;; --buffer-policy) BUFFER_POLICY=$2; shift 2;;
    --max-stages) MAX_STAGES=$2; shift 2;; --train-limit) TRAIN_LIMIT=$2; shift 2;;
    --eval-limit) EVAL_LIMIT=$2; shift 2;; --check-only) CHECK_ONLY=1; shift;;
    -h|--help) usage; exit 0;; *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done
[[ "$RANK" == 48 ]] || { echo "Replay+LoRA rank is frozen at 48" >&2; exit 2; }
[[ "$BUFFER_POLICY" == balanced_per_task_stratified_stable_sha256_v1 ]] || { echo "Unsupported buffer policy" >&2; exit 2; }
[[ "$REPLAY_BUDGET" =~ ^(100|500|1000)$ ]] || exit 2
[[ "$MAX_STAGES" =~ ^[1-5]$ ]] || exit 2
IFS=',' read -r -a GPU_IDS <<< "$GPUS"; NPROC_PER_NODE=${#GPU_IDS[@]}
(( NPROC_PER_NODE >= 1 && NPROC_PER_NODE <= 2 && 16 % NPROC_PER_NODE == 0 )) || exit 2
GRAD_ACC=$((16 / NPROC_PER_NODE))

export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub
export HF_DATASETS_CACHE=/remote-home/wangbomin/huggingface_cache/datasets
export XDG_CACHE_HOME=/remote-home/wangbomin/huggingface_cache/xdg
export TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE
unset TRANSFORMERS_CACHE
mkdir -p "$TMPDIR"; cd "$ROOT"

CONTROL=("$PYTHON" scripts/medicalskill_v1_2_replay_lora/replay_lora_control.py)
COMMON=(--data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --replay-budget "$REPLAY_BUDGET" --seed "$SEED" --train-limit "$TRAIN_LIMIT" --eval-limit "$EVAL_LIMIT" --max-stages "$MAX_STAGES")
"${CONTROL[@]}" check "${COMMON[@]}" $([[ "$CHECK_ONLY" == 0 ]] && echo --write)
(( CHECK_ONLY == 0 )) || { echo "Replay+LoRA contract check PASS"; exit 0; }

for ((STAGE=1; STAGE<=MAX_STAGES; STAGE++)); do
  PADDED=$(printf '%02d' "$STAGE"); STAGE_ROOT="$OUTPUT_ROOT/replay_lora/stage_$PADDED"
  COMPLETION="$STAGE_ROOT/completion.json"
  if [[ -s "$COMPLETION" ]] && "$PYTHON" -c 'import json,sys;raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="PASS" else 1)' "$COMPLETION"; then
    echo "[Replay+LoRA stage $STAGE] already PASS"; continue
  fi
  if [[ -e "$STAGE_ROOT" ]]; then
    echo "Incomplete stage exists; choose a new output root or archive it explicitly: $STAGE_ROOT" >&2; exit 3
  fi
  "${CONTROL[@]}" prepare-stage "${COMMON[@]}" --stage "$STAGE" --global-batch 16
  MIXTURE="$STAGE_ROOT/training_mixture.jsonl"; TRAIN_ROOT="$STAGE_ROOT/train"
  COUNT=$(wc -l < "$MIXTURE"); STEPS=$(( (COUNT + 15) / 16 )); mkdir -p "$TRAIN_ROOT"
  CMD=("$SWIFT" sft --model Qwen/Qwen3-VL-8B-Instruct --model_revision "$REVISION" --template qwen3_vl
    --dataset "$MIXTURE" --tuner_type lora --tuner_backend peft --target_modules q_proj v_proj
    --lora_rank 48 --lora_alpha 96 --lora_dropout 0.05 --lora_bias none
    --freeze_llm false --freeze_vit true --freeze_aligner true --torch_dtype bfloat16 --attn_impl sdpa
    --max_length 4096 --max_pixels 200704 --per_device_train_batch_size 1 --gradient_accumulation_steps "$GRAD_ACC"
    --learning_rate 1e-4 --num_train_epochs 1 --warmup_ratio 0.03 --logging_steps 10 --logging_first_step true
    --save_strategy epoch --save_total_limit 1 --save_only_model false --eval_strategy no --gradient_checkpointing true
    --vit_gradient_checkpointing false --ddp_find_unused_parameters false --dataloader_num_workers 0 --dataset_num_proc 1
    --dataset_shuffle true --split_dataset_ratio 0 --packing false --use_hf true --seed "$SEED" --data_seed "$SEED"
    --report_to none --add_version false --create_checkpoint_symlink false --logging_dir "$TRAIN_ROOT/logs" --output_dir "$TRAIN_ROOT"
    --external_plugins "$ROOT/scripts/phase3/callback_compat.py")
  if (( STAGE > 1 )); then
    PREV=$(printf '%02d' $((STAGE-1)))
    PREVIOUS_CHECKPOINT=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint"])' "$OUTPUT_ROOT/replay_lora/stage_$PREV/completion.json")
    CMD+=(--adapters "$PREVIOUS_CHECKPOINT" --load_args false)
  fi
  printf '%q ' "${CMD[@]}" > "$STAGE_ROOT/train_command.txt"; printf '\n' >> "$STAGE_ROOT/train_command.txt"
  "$PYTHON" scripts/medicalskill_v1_2_baselines/run_timed.py --record "$STAGE_ROOT/train_runtime.json" --category replay_lora_train --gpus "$GPUS" --number-of-samples "$COUNT" --training-steps "$STEPS" -- "${CMD[@]}" 2>&1 | tee "$STAGE_ROOT/train.log"
  CHECKPOINT=$(find "$TRAIN_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)
  test -n "$CHECKPOINT"
  "${CONTROL[@]}" update-buffer "${COMMON[@]}" --stage "$STAGE"
  "${CONTROL[@]}" complete "${COMMON[@]}" --stage "$STAGE" --checkpoint "$CHECKPOINT" --runtime "$STAGE_ROOT/train_runtime.json"
done

evaluate_stage() {
  local gpu=$1 stage=$2 padded checkpoint
  padded=$(printf '%02d' "$stage")
  checkpoint=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint"])' "$OUTPUT_ROOT/replay_lora/stage_$padded/completion.json")
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/medicalskill_v1_2_replay_lora/evaluate_replay_lora_v1_2.py --stage "$stage" --adapter "$checkpoint" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --eval-limit "$EVAL_LIMIT"
}
for ((STAGE=1; STAGE<=MAX_STAGES; STAGE++)); do evaluate_stage "${GPU_IDS[$(((STAGE-1)%NPROC_PER_NODE))]}" "$STAGE"; done
if (( MAX_STAGES == 5 )); then
  "$PYTHON" scripts/medicalskill_v1_2_replay_lora/aggregate_replay_lora_v1_2.py --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --eval-limit "$EVAL_LIMIT"
fi
echo "Replay+LoRA requested stages PASS: $OUTPUT_ROOT"
