#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/MedAgentCL_v4
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
SWIFT=/root/anaconda3/envs/medagentcl_v4/bin/swift
DATA_ROOT=/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k
OUTPUT_ROOT=/remote-home/wangbomin/mrlora_medicalskill_v1_2_lite_10k1k_seed42
GPUS=0,1; SEED=42; RANK=16; ROUTER_RANK=32; ROUTER_MEMORY_PER_TASK=20; ROUTER_EPOCHS=30
MAX_STAGES=5; TRAIN_LIMIT=0; EVAL_LIMIT=0; CHECK_ONLY=0
REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
usage(){ cat <<'EOF'
Usage: run_mrlora_v1_2.sh [options]
  --data-root PATH --output-root PATH --gpus 0,1 --seed 42
  --rank 16 --router-rank 32 --router-memory-per-task 20 --router-epochs 30
  --max-stages N --train-limit N --eval-limit N --check-only
Each task expert starts independently from the locked base. Each stage router is
trained from the locked base on cumulative router-only image-question memory.
EOF
}
while [[ $# -gt 0 ]];do case "$1" in
 --data-root)DATA_ROOT=$2;shift 2;;--output-root)OUTPUT_ROOT=$2;shift 2;;--gpus)GPUS=$2;shift 2;;--seed)SEED=$2;shift 2;;
 --rank)RANK=$2;shift 2;;--router-rank)ROUTER_RANK=$2;shift 2;;--router-memory-per-task)ROUTER_MEMORY_PER_TASK=$2;shift 2;;--router-epochs)ROUTER_EPOCHS=$2;shift 2;;
 --max-stages)MAX_STAGES=$2;shift 2;;--train-limit)TRAIN_LIMIT=$2;shift 2;;--eval-limit)EVAL_LIMIT=$2;shift 2;;--check-only)CHECK_ONLY=1;shift;;-h|--help)usage;exit 0;;*)echo "Unknown argument: $1" >&2;exit 2;;esac;done
[[ "$RANK" == 16 && "$ROUTER_MEMORY_PER_TASK" == 20 && "$MAX_STAGES" =~ ^[1-5]$ ]]||exit 2
IFS=',' read -r -a GPU_IDS <<< "$GPUS";NPROC_PER_NODE=${#GPU_IDS[@]};((NPROC_PER_NODE>=1&&NPROC_PER_NODE<=2&&16%NPROC_PER_NODE==0))||exit 2;GRAD_ACC=$((16/NPROC_PER_NODE))
export HF_HOME=/remote-home/wangbomin/huggingface_cache HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub HF_DATASETS_CACHE=/remote-home/wangbomin/huggingface_cache/datasets XDG_CACHE_HOME=/remote-home/wangbomin/huggingface_cache/xdg TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE;unset TRANSFORMERS_CACHE;mkdir -p "$TMPDIR";cd "$ROOT"
CONTROL=("$PYTHON" scripts/medicalskill_v1_2_mrlora/mrlora_control.py);COMMON=(--data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --rank "$RANK" --router-rank "$ROUTER_RANK" --router-memory-per-task "$ROUTER_MEMORY_PER_TASK" --router-epochs "$ROUTER_EPOCHS" --seed "$SEED" --train-limit "$TRAIN_LIMIT" --eval-limit "$EVAL_LIMIT" --max-stages "$MAX_STAGES")
if ((CHECK_ONLY));then "${CONTROL[@]}" check "${COMMON[@]}";echo "MR-LoRA contract check PASS";exit 0;else "${CONTROL[@]}" check "${COMMON[@]}" --write;fi
for ((STAGE=1;STAGE<=MAX_STAGES;STAGE++));do
 PADDED=$(printf '%02d' "$STAGE");STAGE_ROOT="$OUTPUT_ROOT/mrlora/stage_$PADDED";COMPLETION="$STAGE_ROOT/completion.json"
 if [[ -s "$COMPLETION" ]]&&"$PYTHON" -c 'import json,sys;raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="PASS" else 1)' "$COMPLETION";then echo "[MR-LoRA stage $STAGE] already PASS";continue;fi
 if [[ -e "$STAGE_ROOT" ]];then echo "Incomplete stage exists; choose a new output root or archive explicitly: $STAGE_ROOT" >&2;exit 3;fi
 "${CONTROL[@]}" prepare-expert "${COMMON[@]}" --stage "$STAGE"
 EXPERT_DATA="$STAGE_ROOT/expert_train.jsonl";EXPERT_COUNT=$(wc -l < "$EXPERT_DATA");EXPERT_ROOT="$STAGE_ROOT/expert_train"
 EXPERT_CMD=("$SWIFT" sft --model Qwen/Qwen3-VL-8B-Instruct --model_revision "$REVISION" --template qwen3_vl --dataset "$EXPERT_DATA" --tuner_type lora --tuner_backend peft --target_modules q_proj v_proj --lora_rank "$RANK" --lora_alpha $((2*RANK)) --lora_dropout 0.05 --lora_bias none --freeze_llm false --freeze_vit true --freeze_aligner true --torch_dtype bfloat16 --attn_impl sdpa --max_length 4096 --max_pixels 200704 --per_device_train_batch_size 1 --gradient_accumulation_steps "$GRAD_ACC" --learning_rate 1e-4 --num_train_epochs 1 --warmup_ratio 0.03 --logging_steps 10 --logging_first_step true --save_strategy epoch --save_total_limit 1 --save_only_model false --eval_strategy no --gradient_checkpointing true --vit_gradient_checkpointing false --ddp_find_unused_parameters false --dataloader_num_workers 0 --dataset_num_proc 1 --dataset_shuffle true --split_dataset_ratio 0 --packing false --use_hf true --seed "$SEED" --data_seed "$SEED" --report_to none --add_version false --create_checkpoint_symlink false --logging_dir "$EXPERT_ROOT/logs" --output_dir "$EXPERT_ROOT" --external_plugins "$ROOT/scripts/phase3/callback_compat.py")
 printf '%q ' "${EXPERT_CMD[@]}" > "$STAGE_ROOT/expert_train_command.txt";printf '\n' >> "$STAGE_ROOT/expert_train_command.txt"
 "$PYTHON" scripts/medicalskill_v1_2_baselines/run_timed.py --record "$STAGE_ROOT/expert_train_runtime.json" --category mrlora_expert_train --gpus "$GPUS" --number-of-samples "$EXPERT_COUNT" --training-steps $(((EXPERT_COUNT+15)/16)) -- "${EXPERT_CMD[@]}" 2>&1|tee "$STAGE_ROOT/expert_train.log"
 EXPERT_CHECKPOINT=$(find "$EXPERT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print|sort -V|tail -n 1);test -n "$EXPERT_CHECKPOINT"
 "${CONTROL[@]}" prepare-router "${COMMON[@]}" --stage "$STAGE"
 ROUTER_DATA="$STAGE_ROOT/router_memory/router_train.jsonl";ROUTER_COUNT=$(wc -l < "$ROUTER_DATA");ROUTER_ROOT="$STAGE_ROOT/router_train"
 ROUTER_CMD=("$SWIFT" sft --model Qwen/Qwen3-VL-8B-Instruct --model_revision "$REVISION" --template qwen3_vl --dataset "$ROUTER_DATA" --tuner_type lora --tuner_backend peft --target_modules q_proj v_proj --lora_rank "$ROUTER_RANK" --lora_alpha $((2*ROUTER_RANK)) --lora_dropout 0.05 --lora_bias none --freeze_llm false --freeze_vit true --freeze_aligner true --torch_dtype bfloat16 --attn_impl sdpa --max_length 1024 --max_pixels 200704 --per_device_train_batch_size 1 --gradient_accumulation_steps "$GRAD_ACC" --learning_rate 2e-5 --num_train_epochs "$ROUTER_EPOCHS" --warmup_ratio 0.03 --logging_steps 1 --logging_first_step true --save_strategy epoch --save_total_limit 1 --save_only_model false --eval_strategy no --gradient_checkpointing true --vit_gradient_checkpointing false --ddp_find_unused_parameters false --dataloader_num_workers 0 --dataset_num_proc 1 --dataset_shuffle true --split_dataset_ratio 0 --packing false --use_hf true --seed "$SEED" --data_seed "$SEED" --report_to none --add_version false --create_checkpoint_symlink false --logging_dir "$ROUTER_ROOT/logs" --output_dir "$ROUTER_ROOT" --external_plugins "$ROOT/scripts/phase3/callback_compat.py")
 printf '%q ' "${ROUTER_CMD[@]}" > "$STAGE_ROOT/router_train_command.txt";printf '\n' >> "$STAGE_ROOT/router_train_command.txt"
 "$PYTHON" scripts/medicalskill_v1_2_baselines/run_timed.py --record "$STAGE_ROOT/router_train_runtime.json" --category mrlora_router_train --gpus "$GPUS" --number-of-samples "$ROUTER_COUNT" -- "${ROUTER_CMD[@]}" 2>&1|tee "$STAGE_ROOT/router_train.log"
 ROUTER_CHECKPOINT=$(find "$ROUTER_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print|sort -V|tail -n 1);test -n "$ROUTER_CHECKPOINT"
 "${CONTROL[@]}" complete "${COMMON[@]}" --stage "$STAGE" --expert-checkpoint "$EXPERT_CHECKPOINT" --router-checkpoint "$ROUTER_CHECKPOINT" --expert-runtime "$STAGE_ROOT/expert_train_runtime.json" --router-runtime "$STAGE_ROOT/router_train_runtime.json"
done
for ((STAGE=1;STAGE<=MAX_STAGES;STAGE++));do PADDED=$(printf '%02d' "$STAGE");ROUTER=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["router_checkpoint"])' "$OUTPUT_ROOT/mrlora/stage_$PADDED/completion.json");ARGS=();for ((TASK=1;TASK<=STAGE;TASK++));do P=$(printf '%02d' "$TASK");E=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["expert_checkpoint"])' "$OUTPUT_ROOT/mrlora/stage_$P/completion.json");ARGS+=(--expert "$E");done;CUDA_VISIBLE_DEVICES="${GPU_IDS[$(((STAGE-1)%NPROC_PER_NODE))]}" "$PYTHON" scripts/medicalskill_v1_2_mrlora/evaluate_mrlora_v1_2.py --stage "$STAGE" --router "$ROUTER" "${ARGS[@]}" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --eval-limit "$EVAL_LIMIT";done
if ((MAX_STAGES==5));then "$PYTHON" scripts/medicalskill_v1_2_mrlora/aggregate_mrlora_v1_2.py --output-root "$OUTPUT_ROOT";fi
echo "MR-LoRA requested stages PASS: $OUTPUT_ROOT"
