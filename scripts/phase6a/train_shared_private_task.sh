#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/MedAgentCL_v4
SWIFT=/root/anaconda3/envs/medagentcl_v4/bin/swift
REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b

if [[ $# -ne 2 || ! "$1" =~ ^(closure|pilot)$ || ! "$2" =~ ^[12]$ ]]; then
  echo "Usage: $0 (closure|pilot) TASK_ID(1|2)" >&2
  exit 2
fi
SCOPE=$1
TASK_ID=$2
CONFIG="$ROOT/configs/phase6a/"$SCOPE"_task_"$TASK_ID".json"
DATA="$ROOT/artifacts/phase6a/data/pilot_train_task"$TASK_ID".jsonl"
OUT="/remote-home/wangbomin/medagentcl_v4_phase6a/"$SCOPE"/runs/task_"$TASK_ID

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
export NPROC_PER_NODE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MED_PRISM_SHARED_PRIVATE_CONFIG="$CONFIG"
unset TRANSFORMERS_CACHE

cd "$ROOT"
test -f "$CONFIG"
test -f "$DATA"
mkdir -p "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" "$MODELSCOPE_CACHE"
mkdir -p "$OUT/train"
df -h "$HF_HOME" > "$OUT/cache_filesystem_before.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader \
  > "$OUT/gpu_before_training.txt"

TRAIN_CMD=(
  "$SWIFT" sft
  --model Qwen/Qwen3-VL-8B-Instruct
  --model_revision "$REVISION"
  --template qwen3_vl
  --dataset "$DATA"
  --tuner_type med_prism_shared_private
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
  --warmup_ratio 0
  --logging_steps 1
  --logging_first_step true
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
  --external_plugins
  "$ROOT/scripts/phase3/callback_compat.py"
  "$ROOT/med_prism/swift_plugins/med_prism_rank1_plugin.py"
  "$ROOT/med_prism/swift_plugins/med_prism_shared_private_plugin.py"
)
if [[ "$SCOPE" == closure ]]; then
  TRAIN_CMD+=(--max_steps 3 --save_strategy steps --save_steps 3)
else
  TRAIN_CMD+=(--num_train_epochs 1 --save_strategy epoch)
fi

printf '%q ' "${TRAIN_CMD[@]}" > "$OUT/train_command.txt"
printf '\n' >> "$OUT/train_command.txt"
set +e
"${TRAIN_CMD[@]}" 2>&1 | tee "$OUT/train.log"
swift_exit_code=$?
set -e
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader \
  > "$OUT/gpu_after_training.txt"
python -c 'import json,pathlib,sys; pathlib.Path(sys.argv[2]).write_text(json.dumps({"swift_sft_exit_code":int(sys.argv[1]),"shared_manifest_exists":pathlib.Path(sys.argv[3]).is_file(),"private_manifest_exists":pathlib.Path(sys.argv[4]).is_file()},indent=2)+"\n")' \
  "$swift_exit_code" \
  "$OUT/exit_code.json" \
  "/remote-home/wangbomin/medagentcl_v4_phase6a/$SCOPE/shared/after_task_$TASK_ID/shared_manifest.json" \
  "/remote-home/wangbomin/medagentcl_v4_phase6a/$SCOPE/private/task_$TASK_ID/private_manifest.json"
exit "$swift_exit_code"
