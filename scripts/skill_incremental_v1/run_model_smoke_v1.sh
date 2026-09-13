#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/MedAgentCL_v4
PY=/root/anaconda3/envs/medagentcl_v4/bin/python
SWIFT=/root/anaconda3/envs/medagentcl_v4/bin/swift
DATA=/remote-home/wangbomin/MedicalSkill-CL-v1
OUT=$ROOT/artifacts/medicalskill_cl_v1_closure/model_smoke
REV=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
export HF_HOME=/remote-home/wangbomin/huggingface_cache HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} NPROC_PER_NODE=1
cd "$ROOT";mkdir -p "$OUT"
for source in "$DATA"/task_*;do
 name=$(basename "$source");run="$OUT/$name";mkdir -p "$run"
 "$PY" -c 'import json,pathlib,sys;rows=[json.loads(x) for x in pathlib.Path(sys.argv[1]).read_text().splitlines() if x.strip()];rows.sort(key=lambda x:(len(x.get("images") or []),sum(len(str(m.get("content") or "")) for m in x.get("messages") or [])));pathlib.Path(sys.argv[2]).write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in rows[:2])+"\n")' "$source/train.jsonl" "$run/smoke_train.jsonl"
 printf '{"status":"PENDING"}\n' >"$run/target_module_audit.json";: >"$run/training_trace.jsonl";export MEDICALSKILL_V1_SMOKE_OUTPUT="$run"
 nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader >"$run/gpu_before.txt"
 cmd=("$SWIFT" sft --model Qwen/Qwen3-VL-8B-Instruct --model_revision "$REV" --template qwen3_vl --dataset "$run/smoke_train.jsonl" --tuner_type lora --tuner_backend peft --target_modules q_proj v_proj --lora_rank 48 --lora_alpha 96 --lora_dropout 0.05 --lora_bias none --freeze_llm false --freeze_vit true --freeze_aligner true --torch_dtype bfloat16 --attn_impl sdpa --max_length 1024 --max_pixels 802816 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 1e-4 --max_steps 2 --warmup_ratio 0 --logging_steps 1 --logging_first_step true --save_strategy no --eval_strategy no --gradient_checkpointing true --dataloader_num_workers 0 --dataset_num_proc 1 --dataset_shuffle false --split_dataset_ratio 0 --packing false --use_hf true --seed 42 --data_seed 42 --report_to none --add_version false --create_checkpoint_symlink false --output_dir "$run/train" --external_plugins "$ROOT/scripts/phase3/callback_compat.py" "$ROOT/scripts/skill_incremental_v1/model_smoke_audit_plugin.py" --callbacks medicalskill_v1_audit)
 printf '%q ' "${cmd[@]}" >"$run/command.txt";printf '\n' >>"$run/command.txt"
 set +e;"${cmd[@]}" >"$run/train.log" 2>&1;code=$?;set -e
 nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader >"$run/gpu_after.txt"
 "$PY" -c 'import json,pathlib,sys;pathlib.Path(sys.argv[2]).write_text(json.dumps({"exit_code":int(sys.argv[1])},indent=2)+"\n")' "$code" "$run/exit_code.json"
 [[ $code -eq 0 ]] || exit "$code"
done
exec "$PY" "$ROOT/scripts/skill_incremental_v1/aggregate_model_smoke.py" --root "$OUT" --output "$ROOT/artifacts/medicalskill_cl_v1_closure/model_smoke_summary.json"
