#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/MedAgentCL_v4
OUT="${PHASE3_OUTPUT_DIR:-$ROOT/output/phase3_native_lora}"
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
SWIFT=/root/anaconda3/envs/medagentcl_v4/bin/swift
REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export PHASE3_OUTPUT_DIR="$OUT"

cd "$ROOT"
mkdir -p "$OUT/train"
: > "$OUT/training_trace.jsonl"

TRAIN_CMD=(
  "$SWIFT" sft
  --model Qwen/Qwen3-VL-8B-Instruct
  --model_revision "$REVISION"
  --template qwen3_vl
  --dataset "$OUT/data/smoke_train.jsonl"
  --tuner_type lora
  --tuner_backend peft
  --target_modules q_proj v_proj
  --lora_rank 48
  --lora_alpha 96
  --lora_dropout 0.05
  --lora_bias none
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
  --max_steps 10
  --warmup_ratio 0
  --logging_steps 1
  --logging_first_step true
  --save_strategy steps
  --save_steps 10
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
  --external_plugins "$ROOT/scripts/phase3/callback_compat.py" "$ROOT/scripts/phase3/phase3_audit_plugin.py" "$ROOT/scripts/phase3/gradient_checkpointing_audit_plugin.py"
  --callbacks phase3_audit phase3_gc_audit
)

printf '%q ' "${TRAIN_CMD[@]}" > "$OUT/train_command.txt"
printf '\n' >> "$OUT/train_command.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader > "$OUT/gpu_before_training.txt"
set +e
"${TRAIN_CMD[@]}" 2>&1 | tee "$OUT/train.log"
swift_exit_code=${PIPESTATUS[0]}
set -e
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader > "$OUT/gpu_after_training.txt"
"$PYTHON" -c 'import json, pathlib, sys; pathlib.Path(sys.argv[2]).write_text(json.dumps({"swift_sft_exit_code": int(sys.argv[1]), "checkpoint_exists": pathlib.Path(sys.argv[3]).is_dir()}, indent=2) + "\n")' "$swift_exit_code" "$OUT/phase3_exit_code.json" "$OUT/train/checkpoint-10"
exit "$swift_exit_code"
