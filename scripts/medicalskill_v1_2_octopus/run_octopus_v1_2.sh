#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/MedAgentCL_v4
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
SWIFT=/root/anaconda3/envs/medagentcl_v4/bin/swift
DATA_ROOT=/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k
OUTPUT_ROOT=/remote-home/wangbomin/octopus_medicalskill_v1_2_lite_10k1k_seed42
GPUS=0,1
EVAL_LIMIT=0
GRADIENT_SAMPLES=256
LAMBDA_1=0.01
LAMBDA_2=0.01
LEARNING_RATE=1e-4
RESTART_INCOMPLETE=0
CHECK_ONLY=0
REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
METHOD=octopus_qwen3vl_r16

usage() {
  cat <<'EOF'
Usage: run_octopus_v1_2.sh [options]
  --data-root PATH
  --output-root PATH
  --gpus LIST                 Default: 0,1
  --eval-limit N              0 means full formal test splits
  --gradient-samples N        Default: 256 current-task samples
  --lambda-1 FLOAT            Default: 0.01
  --lambda-2 FLOAT            Default: 0.01
  --learning-rate FLOAT       Default: 1e-4
  --restart-incomplete-stage  Archive one explicit incomplete stage and restart it
  --check-only                Validate contract; no output/GPU work

Runs the audited two-stage Octopus lifecycle, then hands cumulative Stage-2
adapters to the shared formal 15-cell evaluator. It never modifies completed
Sequential or Med-PRISM outputs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT=$2; shift 2 ;;
    --output-root) OUTPUT_ROOT=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --eval-limit) EVAL_LIMIT=$2; shift 2 ;;
    --gradient-samples) GRADIENT_SAMPLES=$2; shift 2 ;;
    --lambda-1) LAMBDA_1=$2; shift 2 ;;
    --lambda-2) LAMBDA_2=$2; shift 2 ;;
    --learning-rate) LEARNING_RATE=$2; shift 2 ;;
    --restart-incomplete-stage) RESTART_INCOMPLETE=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$EVAL_LIMIT" =~ ^[0-9]+$ ]] || { echo "Invalid eval limit" >&2; exit 2; }
[[ "$GRADIENT_SAMPLES" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid gradient sample count" >&2; exit 2;
}
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NPROC_PER_NODE=${#GPU_IDS[@]}
(( NPROC_PER_NODE >= 1 && NPROC_PER_NODE <= 2 )) || {
  echo "Use one or two GPUs" >&2; exit 2;
}
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID" >&2; exit 2; }
done
(( 16 % NPROC_PER_NODE == 0 )) || {
  echo "Global batch 16 must divide by GPU count" >&2; exit 2;
}
TRAIN_NPROC_PER_NODE=$NPROC_PER_NODE
GRADIENT_ACCUMULATION_STEPS=$((16 / TRAIN_NPROC_PER_NODE))
GRADIENT_GPU=${GPU_IDS[0]}

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub
export HF_DATASETS_CACHE=/remote-home/wangbomin/huggingface_cache/datasets
export XDG_CACHE_HOME=/remote-home/wangbomin/huggingface_cache/xdg
export TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset TRANSFORMERS_CACHE
mkdir -p "$TMPDIR"
cd "$ROOT"

CONTROL=scripts/medicalskill_v1_2_octopus/octopus_control.py
COLLECTOR=scripts/medicalskill_v1_2_octopus/collect_octopus_gradients.py
TIMED=scripts/medicalskill_v1_2_baselines/run_timed.py
EVALUATOR=scripts/medicalskill_v1_2_baselines/evaluate_cumulative_lora_v1_2.py
AGGREGATOR=scripts/medicalskill_v1_2_baselines/aggregate_baseline_v1_2.py

for required in "$CONTROL" "$COLLECTOR" "$TIMED" "$EVALUATOR" "$AGGREGATOR"   "$DATA_ROOT/audits/acceptance_matrix.json"   "$DATA_ROOT/manifests/dataset_lock.json"; do
  test -s "$required"
done

INIT_ARGS=(
  "$PYTHON" "$CONTROL" init-run
  --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"
  --lambda-1 "$LAMBDA_1" --lambda-2 "$LAMBDA_2"
  --gradient-samples "$GRADIENT_SAMPLES"
  --learning-rate "$LEARNING_RATE"
)
if (( CHECK_ONLY == 1 )); then
  "${INIT_ARGS[@]}" --dry-run
  echo "Octopus immutable contract check PASS; no files written"
  exit 0
fi
"${INIT_ARGS[@]}"

latest_checkpoint() {
  local root=$1 checkpoint
  checkpoint=$(find "$root" -mindepth 1 -maxdepth 1 -type d     -name 'checkpoint-*' -print | sort -V | tail -n 1)
  test -n "$checkpoint"
  test -s "$checkpoint/adapter_config.json"
  [[ -s "$checkpoint/adapter_model.safetensors" ||
     -s "$checkpoint/adapter_model.bin" ]]
  printf '%s\n' "$checkpoint"
}

task_name() {
  case "$1" in
    1) echo task_01_vqa ;;
    2) echo task_02_diagnosis_classification ;;
    3) echo task_03_concept_recognition ;;
    4) echo task_04_visual_grounding ;;
    5) echo task_05_reasoning_vqa ;;
  esac
}

for STAGE in 1 2 3 4 5; do
  PADDED=$(printf '%02d' "$STAGE")
  TASK_NAME=$(task_name "$STAGE")
  DATASET="$DATA_ROOT/$TASK_NAME/train.jsonl"
  N_SAMPLES=$("$PYTHON" -c     'import sys;print(sum(bool(x.strip()) for x in open(sys.argv[1],encoding="utf-8")))'     "$DATASET")
  MAX_LENGTH=1024
  (( STAGE == 5 )) && MAX_LENGTH=4096
  STAGE_ROOT="$OUTPUT_ROOT/octopus/stage_$PADDED"
  STAGE1_ROOT="$STAGE_ROOT/stage1/train"
  STAGE2_ROOT="$STAGE_ROOT/stage2/train"
  STAGE2_INIT="$STAGE_ROOT/stage2_initialization.json"

  if "$PYTHON" "$CONTROL" completion-ok       --output-root "$OUTPUT_ROOT" --stage "$STAGE" >/dev/null 2>&1; then
    echo "[Octopus task $STAGE] already PASS; skipping"
    continue
  fi
  "$PYTHON" "$CONTROL" stage-ready     --output-root "$OUTPUT_ROOT" --stage "$STAGE"

  if [[ -d "$STAGE_ROOT" ]] &&
      find "$STAGE_ROOT" -mindepth 1 -print -quit | grep -q .; then
    if (( RESTART_INCOMPLETE != 1 )); then
      echo "Incomplete stage exists: $STAGE_ROOT" >&2
      echo "Use --restart-incomplete-stage to archive this exact stage." >&2
      exit 3
    fi
    ARCHIVE="$OUTPUT_ROOT/failed_attempts/stage_${PADDED}_$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$OUTPUT_ROOT/failed_attempts"
    mv "$STAGE_ROOT" "$ARCHIVE"
    printf '%s\n' "$ARCHIVE" >> "$OUTPUT_ROOT/failed_attempts/archive_index.txt"
  fi
  mkdir -p "$STAGE1_ROOT"

  STAGE1_CMD=(
    "$SWIFT" sft
    --model Qwen/Qwen3-VL-8B-Instruct --model_revision "$REVISION"
    --template qwen3_vl --dataset "$DATASET"
    --tuner_type lora --tuner_backend peft
    --target_modules q_proj v_proj
    --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --lora_bias none
    --freeze_llm false --freeze_vit true --freeze_aligner true
    --torch_dtype bfloat16 --attn_impl sdpa
    --max_length "$MAX_LENGTH" --max_pixels 200704
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE" --num_train_epochs 1
    --warmup_ratio 0.03 --logging_steps 10 --logging_first_step true
    --save_strategy epoch --save_total_limit 1 --save_only_model false
    --eval_strategy no --gradient_checkpointing true
    --vit_gradient_checkpointing false --ddp_find_unused_parameters false
    --dataloader_num_workers 0 --dataset_num_proc 1 --dataset_shuffle true
    --split_dataset_ratio 0 --packing false --use_hf true
    --seed 42 --data_seed 42 --report_to none --add_version false
    --create_checkpoint_symlink false
    --logging_dir "$STAGE1_ROOT/logs" --output_dir "$STAGE1_ROOT"
    --external_plugins "$ROOT/scripts/phase3/callback_compat.py"
  )
  if (( STAGE > 1 )); then
    PREVIOUS_STAGE1=$("$PYTHON" "$CONTROL" print-checkpoint       --output-root "$OUTPUT_ROOT" --stage $((STAGE - 1)) --role stage1)
    STAGE1_CMD+=(--adapters "$PREVIOUS_STAGE1" --load_args false)
  fi
  printf '%q ' "${STAGE1_CMD[@]}" > "$STAGE_ROOT/stage1_command.txt"
  printf '\n' >> "$STAGE_ROOT/stage1_command.txt"
  export CUDA_VISIBLE_DEVICES="$GPUS"
  export NPROC_PER_NODE="$TRAIN_NPROC_PER_NODE"
  set +e
  "$PYTHON" "$TIMED" --record "$STAGE_ROOT/stage1_runtime.json"     --category stage1_train --gpus "$GPUS"     --number-of-samples "$N_SAMPLES" -- "${STAGE1_CMD[@]}"     2>&1 | tee "$STAGE_ROOT/stage1_train.log"
  CODE=${PIPESTATUS[0]}
  set -e
  (( CODE == 0 )) || { echo "Stage-1 failed for task $STAGE" >&2; exit "$CODE"; }
  STAGE1_CHECKPOINT=$(latest_checkpoint "$STAGE1_ROOT")

  GRADIENT_MANIFESTS=()
  GRADIENT_RUNTIMES=()
  STAGE2_RUNTIME_ARGS=()
  HISTORICAL_STAGE2_ADAPTERS=()
  if (( STAGE > 1 )); then
    for ((HISTORICAL=1; HISTORICAL<STAGE; HISTORICAL++)); do
      HISTORICAL_STAGE2_ADAPTERS+=("$("$PYTHON" "$CONTROL" print-checkpoint \
        --output-root "$OUTPUT_ROOT" --stage "$HISTORICAL" --role stage2)")
    done
    for ((PREFIX=1; PREFIX<STAGE; PREFIX++)); do
      PREFIX_PADDED=$(printf '%02d' "$PREFIX")
      ARTIFACT_DIR="$STAGE_ROOT/gradient_artifacts/prefix_$PREFIX_PADDED"
      RUNTIME_PATH="$STAGE_ROOT/gradient_artifacts/prefix_${PREFIX_PADDED}_runtime.json"
      COLLECT_CMD=(
        "$PYTHON" "$COLLECTOR" --task-id "$STAGE" --dataset "$DATASET"
        --output-dir "$ARTIFACT_DIR" --samples "$GRADIENT_SAMPLES"
        --batch-size 1 --max-length "$MAX_LENGTH"
        --target-module q_proj --target-module v_proj
      )
      for ((HISTORICAL=1; HISTORICAL<=PREFIX; HISTORICAL++)); do
        COLLECT_CMD+=(--historical-adapter \
          "${HISTORICAL_STAGE2_ADAPTERS[$((HISTORICAL - 1))]}")
      done
      mkdir -p "$STAGE_ROOT/gradient_artifacts"
      printf '%q ' "${COLLECT_CMD[@]}" > "$STAGE_ROOT/gradient_artifacts/prefix_${PREFIX_PADDED}_command.txt"
      printf '\n' >> "$STAGE_ROOT/gradient_artifacts/prefix_${PREFIX_PADDED}_command.txt"
      export CUDA_VISIBLE_DEVICES="$GRADIENT_GPU"
      export NPROC_PER_NODE=1
      set +e
      "$PYTHON" "$TIMED" --record "$RUNTIME_PATH" \
        --category historical_gradient_collection --gpus "$GRADIENT_GPU" \
        --number-of-samples "$GRADIENT_SAMPLES" -- "${COLLECT_CMD[@]}" \
        2>&1 | tee "$STAGE_ROOT/gradient_artifacts/prefix_${PREFIX_PADDED}.log"
      CODE=${PIPESTATUS[0]}
      set -e
      (( CODE == 0 )) || {
        echo "Gradient prefix $PREFIX failed for task $STAGE" >&2; exit "$CODE";
      }
      GRADIENT_MANIFESTS+=("$ARTIFACT_DIR/gradient_manifest.json")
      GRADIENT_RUNTIMES+=("$RUNTIME_PATH")
    done

    VALIDATE_ARGS=(
      "$PYTHON" "$CONTROL" validate-gradients
      --output-root "$OUTPUT_ROOT" --data-root "$DATA_ROOT" --stage "$STAGE"
    )
    for value in "${GRADIENT_MANIFESTS[@]}"; do
      VALIDATE_ARGS+=(--manifest "$value")
    done
    "${VALIDATE_ARGS[@]}"
  fi
  "$PYTHON" "$CONTROL" record-stage2-init --stage "$STAGE" \
    --stage1-checkpoint "$STAGE1_CHECKPOINT" --manifest "$STAGE2_INIT"

  GRADIENT_JSON=$("$PYTHON" -c \
    'import json,sys;print(json.dumps(sys.argv[1:]))' \
    "${GRADIENT_MANIFESTS[@]}")
  HISTORICAL_STAGE2_JSON=$("$PYTHON" -c \
    'import json,sys;print(json.dumps(sys.argv[1:]))' \
    "${HISTORICAL_STAGE2_ADAPTERS[@]}")
  export OCTOPUS_STAGE2_ENABLED=1
  export OCTOPUS_STAGE2_TASK_ID="$STAGE"
  export OCTOPUS_GRADIENT_MANIFESTS="$GRADIENT_JSON"
  export OCTOPUS_HISTORICAL_STAGE2_ADAPTERS="$HISTORICAL_STAGE2_JSON"
  if (( STAGE == 1 )); then
    export OCTOPUS_LAMBDA_1=0.0 OCTOPUS_LAMBDA_2=0.0
  else
    export OCTOPUS_LAMBDA_1="$LAMBDA_1" OCTOPUS_LAMBDA_2="$LAMBDA_2"
  fi
  export OCTOPUS_AUDIT_DIR="$STAGE_ROOT/runtime_audits"
  mkdir -p "$STAGE2_ROOT"
  STAGE2_CMD=(
    "$SWIFT" sft
    --model Qwen/Qwen3-VL-8B-Instruct --model_revision "$REVISION"
    --template qwen3_vl --dataset "$DATASET"
    --tuner_type lora --tuner_backend peft
    --target_modules q_proj v_proj
    --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --lora_bias none
    --adapters "$STAGE1_CHECKPOINT" --load_args false
    --freeze_llm false --freeze_vit true --freeze_aligner true
    --torch_dtype bfloat16 --attn_impl sdpa
    --max_length "$MAX_LENGTH" --max_pixels 200704
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE" --num_train_epochs 1
    --warmup_ratio 0.03 --logging_steps 10 --logging_first_step true
    --save_strategy epoch --save_total_limit 1 --save_only_model false
    --eval_strategy no --gradient_checkpointing true
    --vit_gradient_checkpointing false --ddp_find_unused_parameters false
    --dataloader_num_workers 0 --dataset_num_proc 1 --dataset_shuffle true
    --split_dataset_ratio 0 --packing false --use_hf true
    --seed 42 --data_seed 42 --report_to none --add_version false
    --create_checkpoint_symlink false
    --logging_dir "$STAGE2_ROOT/logs" --output_dir "$STAGE2_ROOT"
    --external_plugins "$ROOT/scripts/phase3/callback_compat.py" \
      "$ROOT/med_prism/swift_plugins/octopus_plugin.py"
  )
  printf '%q ' "${STAGE2_CMD[@]}" > "$STAGE_ROOT/stage2_command.txt"
  printf '\n' >> "$STAGE_ROOT/stage2_command.txt"
  export CUDA_VISIBLE_DEVICES="$GPUS"
  export NPROC_PER_NODE="$TRAIN_NPROC_PER_NODE"
  set +e
  "$PYTHON" "$TIMED" --record "$STAGE_ROOT/stage2_runtime.json" \
    --category stage2_train --gpus "$GPUS" --number-of-samples "$N_SAMPLES" \
    -- "${STAGE2_CMD[@]}" 2>&1 | tee "$STAGE_ROOT/stage2_train.log"
  CODE=${PIPESTATUS[0]}
  set -e
  unset OCTOPUS_STAGE2_ENABLED OCTOPUS_STAGE2_TASK_ID
  unset OCTOPUS_GRADIENT_MANIFESTS OCTOPUS_HISTORICAL_STAGE2_ADAPTERS
  unset OCTOPUS_LAMBDA_1 OCTOPUS_LAMBDA_2 OCTOPUS_AUDIT_DIR
  (( CODE == 0 )) || { echo "Stage-2 failed for task $STAGE" >&2; exit "$CODE"; }
  STAGE2_CHECKPOINT=$(latest_checkpoint "$STAGE2_ROOT")
  STAGE2_RUNTIME_ARGS=(--stage2-runtime "$STAGE_ROOT/stage2_runtime.json")

  FINALIZE_ARGS=(
    "$PYTHON" "$CONTROL" finalize
    --output-root "$OUTPUT_ROOT" --data-root "$DATA_ROOT" --stage "$STAGE"
    --stage1-checkpoint "$STAGE1_CHECKPOINT"
    --stage2-checkpoint "$STAGE2_CHECKPOINT"
    --stage2-initialization "$STAGE2_INIT"
    --stage1-runtime "$STAGE_ROOT/stage1_runtime.json"
    "${STAGE2_RUNTIME_ARGS[@]}"
  )
  for value in "${GRADIENT_MANIFESTS[@]}"; do
    FINALIZE_ARGS+=(--gradient-manifest "$value")
  done
  for value in "${GRADIENT_RUNTIMES[@]}"; do
    FINALIZE_ARGS+=(--gradient-runtime "$value")
  done
  "${FINALIZE_ARGS[@]}"
  echo "[Octopus task $STAGE] PASS"
done

for STAGE in 1 2 3 4 5; do
  PADDED=$(printf '%02d' "$STAGE")
  mapfile -t STAGE2_STACK < <(
    "$PYTHON" "$CONTROL" print-stack       --output-root "$OUTPUT_ROOT" --stage "$STAGE"
  )
  EVAL_ARGS=(
    "$PYTHON" "$EVALUATOR" --method "$METHOD" --stage "$STAGE"
    --expected-rank 16 --data-root "$DATA_ROOT"
    --output-root "$OUTPUT_ROOT" --eval-limit "$EVAL_LIMIT"
  )
  for adapter in "${STAGE2_STACK[@]}"; do
    EVAL_ARGS+=(--adapter "$adapter")
  done
  export CUDA_VISIBLE_DEVICES="$GRADIENT_GPU"
  export NPROC_PER_NODE=1
  "$PYTHON" "$TIMED"     --record "$OUTPUT_ROOT/evaluation/stage_$PADDED/runtime.json"     --category evaluation --gpus "$GRADIENT_GPU" -- "${EVAL_ARGS[@]}"     2>&1 | tee "$OUTPUT_ROOT/evaluation_stage_$PADDED.log"
done

"$PYTHON" "$AGGREGATOR" --method "$METHOD" --stage-root-name octopus   --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"   --eval-limit "$EVAL_LIMIT" 2>&1 | tee "$OUTPUT_ROOT/aggregate.log"
echo "Octopus train + 15-cell evaluation PASS: $OUTPUT_ROOT"
