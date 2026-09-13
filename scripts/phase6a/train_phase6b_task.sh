#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/MedAgentCL_v4
OUTPUT=/remote-home/wangbomin/medagentcl_v4_phase6b
SWIFT=/root/anaconda3/envs/medagentcl_v4/bin/swift
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b

if [[ $# -ne 2 || ! "$1" =~ ^(sequential_native_lora_r48|pure_rank1_e16|shared32_private16)$ || ! "$2" =~ ^[12]$ ]]; then
  echo "Usage: $0 METHOD TASK_ID(1|2)" >&2
  exit 2
fi
METHOD=$1
TASK_ID=$2
DATA="$OUTPUT/data/task_${TASK_ID}_train.jsonl"
OUT="$OUTPUT/$METHOD/task_$TASK_ID"
COMPLETION="$OUT/completion.json"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub
export HF_DATASETS_CACHE=/remote-home/wangbomin/huggingface_cache/datasets
export XDG_CACHE_HOME=/remote-home/wangbomin/huggingface_cache/xdg
export TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset TRANSFORMERS_CACHE

cd "$ROOT"
test -f "$DATA"
if [[ "$TASK_ID" == 2 ]]; then
  test -f "$OUTPUT/$METHOD/task_1/completion.json"
  "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("status")=="PASS" else 1)' "$OUTPUT/$METHOD/task_1/completion.json"
fi
if [[ -f "$COMPLETION" ]] && "$PYTHON" -c 'import json,pathlib,sys; d=json.load(open(sys.argv[1])); p=d.get("checkpoint"); raise SystemExit(0 if d.get("status")=="PASS" and p and pathlib.Path(p).is_dir() else 1)' "$COMPLETION"
then
  echo "SKIP completed $METHOD Task $TASK_ID"
  exit 0
fi

mkdir -p "$OUT/train"
df -h "$HF_HOME" > "$OUT/cache_filesystem_before.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader > "$OUT/gpu_before_training.txt"

COMMON=(
  "$SWIFT" sft
  --model Qwen/Qwen3-VL-8B-Instruct
  --model_revision "$REVISION"
  --template qwen3_vl
  --dataset "$DATA"
  --freeze_llm false
  --freeze_vit true
  --freeze_aligner true
  --torch_dtype bfloat16
  --attn_impl sdpa
  --max_length 1024
  --max_pixels 802816
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 1
  --learning_rate 1e-4
  --num_train_epochs 1
  --warmup_ratio 0
  --logging_steps 10
  --logging_first_step true
  --save_strategy epoch
  --save_total_limit 1
  --save_only_model false
  --eval_strategy no
  --gradient_checkpointing true
  --vit_gradient_checkpointing false
  --ddp_find_unused_parameters false
  --dataloader_num_workers 0
  --dataset_num_proc 1
  --dataset_shuffle false
  --split_dataset_ratio 0
  --packing false
  --use_hf true
  --seed 42
  --data_seed 42
  --report_to none
  --add_version false
  --create_checkpoint_symlink false
  --logging_dir "$OUT/train/logs"
  --output_dir "$OUT/train"
)

case "$METHOD" in
  sequential_native_lora_r48)
    CMD=(
      "${COMMON[@]}"
      --tuner_type lora
      --tuner_backend peft
      --target_modules q_proj v_proj
      --lora_rank 48
      --lora_alpha 96
      --lora_dropout 0.05
      --lora_bias none
      --external_plugins
      "$ROOT/scripts/phase3/callback_compat.py"
    )
    if [[ "$TASK_ID" == 2 ]]; then
      TASK1_CHECKPOINT=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$OUTPUT/$METHOD/task_1/completion.json")
      CMD+=(--adapters "$TASK1_CHECKPOINT" --load_args false)
    fi
    ;;
  pure_rank1_e16)
    CONFIG="$ROOT/configs/phase6b/rank1_task_$TASK_ID.json"
    test -f "$CONFIG"
    export MED_PRISM_CONFIG="$CONFIG"
    CMD=(
      "${COMMON[@]}"
      --tuner_type med_prism_rank1
      --external_plugins
      "$ROOT/scripts/phase3/callback_compat.py"
      "$ROOT/med_prism/swift_plugins/med_prism_rank1_plugin.py"
    )
    ;;
  shared32_private16)
    CONFIG="$ROOT/configs/phase6b/shared_private_task_$TASK_ID.json"
    test -f "$CONFIG"
    export MED_PRISM_SHARED_PRIVATE_CONFIG="$CONFIG"
    CMD=(
      "${COMMON[@]}"
      --tuner_type med_prism_shared_private
      --external_plugins
      "$ROOT/scripts/phase3/callback_compat.py"
      "$ROOT/med_prism/swift_plugins/med_prism_rank1_plugin.py"
      "$ROOT/med_prism/swift_plugins/med_prism_shared_private_plugin.py"
    )
    ;;
esac

printf '%q ' "${CMD[@]}" > "$OUT/train_command.txt"
printf '\n' >> "$OUT/train_command.txt"
set +e
"${CMD[@]}" 2>&1 | tee "$OUT/train.log"
swift_exit_code=${PIPESTATUS[0]}
set -e
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader > "$OUT/gpu_after_training.txt"
checkpoint=$(find "$OUT/train" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)
status=FAIL
if [[ "$swift_exit_code" == 0 && -n "$checkpoint" && -d "$checkpoint" ]]; then
  status=PASS
fi
if [[ "$status" == PASS ]]; then
  case "$METHOD" in
    sequential_native_lora_r48)
      test -f "$checkpoint/adapter_config.json" || status=FAIL
      if [[ ! -f "$checkpoint/adapter_model.safetensors" && ! -f "$checkpoint/adapter_model.bin" ]]; then
        status=FAIL
      fi
      ;;
    pure_rank1_e16)
      test -f "$checkpoint/rank1_manifest.json" || status=FAIL
      if ! "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("status")=="PASS" else 1)' "$OUT/orth_gradient_audit.json"; then
        status=FAIL
      fi
      ;;
    shared32_private16)
      test -f "$OUTPUT/$METHOD/shared/after_task_$TASK_ID/shared_manifest.json" || status=FAIL
      test -f "$OUTPUT/$METHOD/private/task_$TASK_ID/private_manifest.json" || status=FAIL
      if ! "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("status")=="PASS" else 1)' "$OUTPUT/$METHOD/runtime_audits/task${TASK_ID}_parameter_audit.json"; then
        status=FAIL
      fi
      ;;
  esac
fi
"$PYTHON" -c 'import json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({"status":sys.argv[2],"swift_sft_exit_code":int(sys.argv[3]),"checkpoint":sys.argv[4] or None,"model_id":"Qwen/Qwen3-VL-8B-Instruct","model_revision":sys.argv[5],"epochs":1,"oversampling":False},indent=2)+"\n")' "$COMPLETION" "$status" "$swift_exit_code" "$checkpoint" "$REVISION"
if [[ "$status" != PASS ]]; then
  exit 1
fi
