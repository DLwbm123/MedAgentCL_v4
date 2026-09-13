#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/MedAgentCL_v4
SWIFT=/root/anaconda3/envs/medagentcl_v4/bin/swift
DATA=/remote-home/wangbomin/MedicalSkill-CL-v1
OUT=/remote-home/wangbomin/medicalskill_cl_v1_runs
REV=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
declare -A TASKS=([1]=task_01_vqa [2]=task_02_diagnosis_classification [3]=task_03_concept_recognition [4]=task_04_visual_grounding [5]=task_05_reasoning_vqa)
[[ $# -eq 1 && -n "${TASKS[$1]:-}" ]] || { echo "usage: $0 TASK_ID(1-5)" >&2;exit 2; }
task=$1;name=${TASKS[$task]}
export HF_HOME=/remote-home/wangbomin/huggingface_cache HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp NPROC_PER_NODE=1
cd "$ROOT";mkdir -p "$OUT/$name"
exec "$SWIFT" sft --model Qwen/Qwen3-VL-8B-Instruct --model_revision "$REV" --template qwen3_vl --dataset "$DATA/$name/train.jsonl" --tuner_type lora --tuner_backend peft --target_modules q_proj v_proj --lora_rank 48 --lora_alpha 96 --lora_dropout 0.05 --freeze_llm false --freeze_vit true --freeze_aligner true --torch_dtype bfloat16 --attn_impl sdpa --max_length 1024 --max_pixels 802816 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 1e-4 --num_train_epochs 1 --warmup_ratio 0.03 --logging_steps 10 --save_strategy epoch --save_total_limit 1 --eval_strategy no --gradient_checkpointing true --dataloader_num_workers 0 --dataset_num_proc 1 --dataset_shuffle true --split_dataset_ratio 0 --packing false --use_hf true --seed 42 --data_seed 42 --report_to none --add_version false --create_checkpoint_symlink false --output_dir "$OUT/$name"
