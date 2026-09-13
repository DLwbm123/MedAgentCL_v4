#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/MedAgentCL_v4
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
SWIFT=/root/anaconda3/envs/medagentcl_v4/bin/swift
DATA_ROOT=/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k
OUTPUT_ROOT=
METHOD=
GPUS=0,1
TASKS=5
TRAIN_LIMIT=0
EVAL_LIMIT=0
MAX_STEPS=0
CHECK_ONLY=0
RESTART_INCOMPLETE=0
REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
usage(){ echo "Usage: $0 --method {olora|reglora|moelora} --output-root PATH [--gpus 0,1] [--tasks 1..5] [--train-limit N] [--eval-limit N] [--max-steps N] [--check-only] [--restart-incomplete-stage]"; }
while [[ $# -gt 0 ]]; do case "$1" in
 --method) METHOD=$2;shift 2;; --output-root) OUTPUT_ROOT=$2;shift 2;; --data-root) DATA_ROOT=$2;shift 2;; --gpus) GPUS=$2;shift 2;; --tasks) TASKS=$2;shift 2;; --train-limit) TRAIN_LIMIT=$2;shift 2;; --eval-limit) EVAL_LIMIT=$2;shift 2;; --max-steps) MAX_STEPS=$2;shift 2;; --check-only) CHECK_ONLY=1;shift;; --restart-incomplete-stage) RESTART_INCOMPLETE=1;shift;; -h|--help) usage;exit 0;; *) echo "Unknown: $1" >&2;usage >&2;exit 2;; esac; done
[[ $METHOD =~ ^(olora|reglora|moelora)$ && -n $OUTPUT_ROOT ]] || { usage >&2;exit 2; }
[[ $TASKS =~ ^[1-5]$ && $TRAIN_LIMIT =~ ^[0-9]+$ && $EVAL_LIMIT =~ ^[0-9]+$ && $MAX_STEPS =~ ^[0-9]+$ ]] || exit 2
IFS=',' read -r -a GPU_IDS <<< "$GPUS"; NPROC=${#GPU_IDS[@]}; ((NPROC>=1&&NPROC<=2)) || exit 2; ((16%NPROC==0)) || exit 2; ACCUM=$((16/NPROC)); ((TRAIN_LIMIT>0)) && ACCUM=1; EVAL_GPU=${GPU_IDS[0]}
export HF_HOME=/remote-home/wangbomin/huggingface_cache HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub HF_DATASETS_CACHE=/remote-home/wangbomin/huggingface_cache/datasets XDG_CACHE_HOME=/remote-home/wangbomin/huggingface_cache/xdg TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_NVLS_ENABLE=0 PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}; unset TRANSFORMERS_CACHE; mkdir -p "$TMPDIR";cd "$ROOT"
CONTROL=scripts/medicalskill_v1_2_wave1/wave1_control.py; TIMED=scripts/medicalskill_v1_2_baselines/run_timed.py; CUM_EVAL=scripts/medicalskill_v1_2_baselines/evaluate_cumulative_lora_v1_2.py; MOE_EVAL=scripts/medicalskill_v1_2_wave1/evaluate_moelora_v1_2.py
INIT=("$PYTHON" "$CONTROL" init-run --method "$METHOD" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"); if ((CHECK_ONLY));then "${INIT[@]}" --dry-run;echo "Wave-1 $METHOD contract PASS; no files/GPU work";exit 0;fi; "${INIT[@]}"
task_name(){ case $1 in 1)echo task_01_vqa;;2)echo task_02_diagnosis_classification;;3)echo task_03_concept_recognition;;4)echo task_04_visual_grounding;;5)echo task_05_reasoning_vqa;;esac; }
method_id(){ case $METHOD in olora)echo olora_qwen3vl_r16;;reglora)echo reglora_sefe_component_qwen3vl_r16;;moelora)echo coin_moelora_qwen3vl_total_r48_e4;;esac; }
latest_checkpoint(){ find "$1" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n1; }
json_list(){ "$PYTHON" -c 'import json,sys;print(json.dumps(sys.argv[1:]))' "$@"; }
for STAGE in $(seq 1 "$TASKS");do
 PAD=$(printf '%02d' "$STAGE"); TASK=$(task_name "$STAGE"); STAGE_ROOT="$OUTPUT_ROOT/$METHOD/stage_$PAD"; TRAIN_ROOT="$STAGE_ROOT/train"; COMPLETION="$STAGE_ROOT/completion.json"
 if "$PYTHON" "$CONTROL" completion-ok --method "$METHOD" --output-root "$OUTPUT_ROOT" --stage "$STAGE" >/dev/null 2>&1;then echo "[$METHOD stage $STAGE] already PASS";continue;fi
 "$PYTHON" "$CONTROL" stage-ready --method "$METHOD" --output-root "$OUTPUT_ROOT" --stage "$STAGE"
 if [[ -d $STAGE_ROOT ]]&&find "$STAGE_ROOT" -mindepth 1 -print -quit|grep -q .;then if ((RESTART_INCOMPLETE==0));then echo "Incomplete stage exists: $STAGE_ROOT" >&2;exit 3;fi; ARCHIVE="$OUTPUT_ROOT/failed_attempts/${METHOD}_stage_${PAD}_$(date -u +%Y%m%dT%H%M%SZ)";mkdir -p "$(dirname "$ARCHIVE")";mv "$STAGE_ROOT" "$ARCHIVE";fi
 mkdir -p "$TRAIN_ROOT"; DATASET="$DATA_ROOT/$TASK/train.jsonl"; NSAMPLES=$(wc -l < "$DATASET")
 if ((TRAIN_LIMIT>0));then DATASET="$STAGE_ROOT/smoke_inputs/train_first_${TRAIN_LIMIT}.jsonl";mkdir -p "$(dirname "$DATASET")";"$PYTHON" - "$DATA_ROOT/$TASK/train.jsonl" "$DATASET" "$TRAIN_LIMIT" <<'PY'
import hashlib,json,pathlib,sys
src,dst,n=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),int(sys.argv[3]);rows=[]
with src.open(encoding='utf-8') as f:
 for line in f:
  if line.strip(): rows.append(line)
  if len(rows)==n: break
if len(rows)!=n: raise RuntimeError('not enough smoke rows')
dst.write_text(''.join(rows),encoding='utf-8');h=hashlib.sha256(dst.read_bytes()).hexdigest();(dst.parent/'subset_manifest.json').write_text(json.dumps({'status':'PASS','source':str(src),'rows':n,'sha256':h,'paper_result':False},indent=2)+'\n')
PY
 NSAMPLES=$TRAIN_LIMIT;fi
 HISTORY=(); if ((STAGE>1));then for H in $(seq 1 $((STAGE-1)));do HPAD=$(printf '%02d' "$H"); HISTORY+=("$($PYTHON -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint"])' "$OUTPUT_ROOT/$METHOD/stage_$HPAD/completion.json")");done;fi
 RANK=48;ALPHA=96;PLUGIN=; unset OLORA_ENABLED OLORA_TASK_ID OLORA_HISTORY OLORA_AUDIT_DIR OLORA_LAMBDA_1 OLORA_LAMBDA_2 REGLORA_ENABLED REGLORA_TASK_ID REGLORA_HISTORY REGLORA_MASK REGLORA_MASK_META REGLORA_AUDIT_DIR REGLORA_LAMBDA MOELORA_ENABLED MOELORA_TASK_ID MOELORA_AUDIT_DIR
 case $METHOD in
  olora) RANK=16;ALPHA=32;PLUGIN="$ROOT/med_prism/swift_plugins/olora_plugin.py";export OLORA_ENABLED=1 OLORA_TASK_ID=$STAGE OLORA_HISTORY="$(json_list "${HISTORY[@]}")" OLORA_AUDIT_DIR="$STAGE_ROOT/runtime_audits" OLORA_LAMBDA_1=0.5 OLORA_LAMBDA_2=0;;
  reglora) RANK=16;ALPHA=32;PLUGIN="$ROOT/med_prism/swift_plugins/reglora_plugin.py";export REGLORA_ENABLED=1 REGLORA_TASK_ID=$STAGE REGLORA_HISTORY="$(json_list "${HISTORY[@]}")" REGLORA_AUDIT_DIR="$STAGE_ROOT/runtime_audits" REGLORA_LAMBDA=2500;if ((STAGE>1));then PREV=$(printf '%02d' $((STAGE-1)));export REGLORA_MASK="$OUTPUT_ROOT/reglora/stage_$PREV/importance_mask.pt" REGLORA_MASK_META="$OUTPUT_ROOT/reglora/stage_$PREV/mask_build.json";fi;;
  moelora) PLUGIN="$ROOT/med_prism/swift_plugins/moelora_plugin.py";export MOELORA_ENABLED=1 MOELORA_TASK_ID=$STAGE MOELORA_AUDIT_DIR="$STAGE_ROOT/runtime_audits";;
 esac
 MAX_LENGTH=1024;((STAGE==5))&&MAX_LENGTH=4096;((TRAIN_LIMIT>0))&&MAX_LENGTH=512
 CMD=("$SWIFT" sft --model Qwen/Qwen3-VL-8B-Instruct --model_revision "$REVISION" --template qwen3_vl --dataset "$DATASET" --tuner_type lora --tuner_backend peft --target_modules q_proj v_proj --lora_rank "$RANK" --lora_alpha "$ALPHA" --lora_dropout 0.05 --lora_bias none --freeze_llm false --freeze_vit true --freeze_aligner true --torch_dtype bfloat16 --attn_impl sdpa --max_length "$MAX_LENGTH" --max_pixels 200704 --per_device_train_batch_size 1 --gradient_accumulation_steps "$ACCUM" --learning_rate 1e-4 --num_train_epochs 1 --warmup_ratio 0.03 --logging_steps 1 --logging_first_step true --save_strategy epoch --save_total_limit 1 --save_only_model false --eval_strategy no --gradient_checkpointing true --vit_gradient_checkpointing false --ddp_find_unused_parameters false --dataloader_num_workers 0 --dataset_num_proc 1 --dataset_shuffle true --split_dataset_ratio 0 --packing false --use_hf true --seed 42 --data_seed 42 --report_to none --add_version false --create_checkpoint_symlink false --logging_dir "$TRAIN_ROOT/logs" --output_dir "$TRAIN_ROOT" --external_plugins "$ROOT/scripts/phase3/callback_compat.py" "$PLUGIN")
 if [[ $METHOD == moelora && $STAGE -gt 1 ]];then CMD+=(--adapters "${HISTORY[-1]}" --load_args false);fi;((MAX_STEPS>0))&&CMD+=(--max_steps "$MAX_STEPS")
 printf '%q ' "${CMD[@]}" > "$STAGE_ROOT/train_command.txt";printf '\n' >> "$STAGE_ROOT/train_command.txt";export CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE=$NPROC
 "$PYTHON" "$TIMED" --record "$STAGE_ROOT/train_runtime.json" --category main_train --gpus "$GPUS" --number-of-samples "$NSAMPLES" -- "${CMD[@]}" 2>&1|tee "$STAGE_ROOT/train.log";CHECKPOINT=$(latest_checkpoint "$TRAIN_ROOT");test -n "$CHECKPOINT"
 MASK_ARGS=();if [[ $METHOD == reglora ]];then PREVIOUS=;((STAGE>1))&&PREVIOUS="$OUTPUT_ROOT/reglora/stage_$(printf '%02d' $((STAGE-1)))/importance_mask.pt";"$PYTHON" - "$CHECKPOINT" "$STAGE_ROOT/importance_mask.pt" "$PREVIOUS" "$STAGE_ROOT/mask_build.json" <<'PY'
import json,pathlib,sys
from med_prism.baselines.reglora import build_importance_mask
from scripts.medicalskill_v1_2_baselines.baseline_harness import atomic_json
ck,out,prev,record=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3],pathlib.Path(sys.argv[4]);v=build_importance_mask(ck,out,pathlib.Path(prev) if prev else None);atomic_json(record,v);print(json.dumps(v,indent=2))
PY
 MASK_ARGS=(--mask "$STAGE_ROOT/importance_mask.pt" --mask-build-record "$STAGE_ROOT/mask_build.json");fi
 INFERENCE=("${HISTORY[@]}" "$CHECKPOINT");EVAL_SUMMARY="$OUTPUT_ROOT/evaluation/stage_$PAD/stage_evaluation_summary.json";export CUDA_VISIBLE_DEVICES="$EVAL_GPU" NPROC_PER_NODE=1
 if [[ $METHOD == moelora ]];then EVAL_CMD=("$PYTHON" "$MOE_EVAL" --stage "$STAGE" --checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --eval-limit "$EVAL_LIMIT");INFERENCE=("$CHECKPOINT");else EVAL_CMD=("$PYTHON" "$CUM_EVAL" --method "$(method_id)" --stage "$STAGE" --expected-rank "$RANK" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --eval-limit "$EVAL_LIMIT");for P in "${INFERENCE[@]}";do EVAL_CMD+=(--adapter "$P");done;fi
 printf '%q ' "${EVAL_CMD[@]}" > "$STAGE_ROOT/eval_command.txt";printf '\n' >> "$STAGE_ROOT/eval_command.txt";"$PYTHON" "$TIMED" --record "$STAGE_ROOT/eval_runtime.json" --category evaluation --gpus "$EVAL_GPU" -- "${EVAL_CMD[@]}" 2>&1|tee "$STAGE_ROOT/eval.log"
 FINAL=("$PYTHON" "$CONTROL" finalize --method "$METHOD" --output-root "$OUTPUT_ROOT" --stage "$STAGE" --checkpoint "$CHECKPOINT" --train-runtime "$STAGE_ROOT/train_runtime.json" --eval-runtime "$STAGE_ROOT/eval_runtime.json" --evaluation-summary "$EVAL_SUMMARY" "${MASK_ARGS[@]}");for P in "${INFERENCE[@]}";do FINAL+=(--inference-checkpoint "$P");done;"${FINAL[@]}";echo "[$METHOD stage $STAGE] PASS"
done
if ((TASKS==5));then "$PYTHON" scripts/medicalskill_v1_2_wave1/aggregate_wave1_v1_2.py --method "$METHOD" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --eval-limit "$EVAL_LIMIT";fi
echo "Wave-1 $METHOD completed requested $TASKS stage(s)"
