#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/MedAgentCL_v4
ARTIFACT=$ROOT/artifacts/medicalskill_cl_v1_1_and_medprism_smoke
SWIFT=/root/anaconda3/envs/medagentcl_v4/bin/swift
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b

if [[ $# -ne 1 || ! "$1" =~ ^[1-5]$ ]]; then
  echo "Usage: $0 STAGE_ID(1..5)" >&2
  exit 2
fi
STAGE=$1
PADDED=$(printf '%02d' "$STAGE")
OUT=$ARTIFACT/med_prism/stage_$PADDED
DATA=$ARTIFACT/smoke_data/task_${PADDED}_train_100.jsonl
CONFIG=$ARTIFACT/configs/stage_$PADDED.json
COMPLETION=$OUT/completion.json

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub
export HF_DATASETS_CACHE=/remote-home/wangbomin/huggingface_cache/datasets
export XDG_CACHE_HOME=/remote-home/wangbomin/huggingface_cache/xdg
export TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MED_PRISM_SHARED_PRIVATE_CONFIG=$CONFIG MED_PRISM_RELOAD_PROBE=1
unset TRANSFORMERS_CACHE
cd "$ROOT"

test -s "$DATA"; test -f "$CONFIG"
if (( STAGE > 1 )); then
  PREV=$(printf '%02d' $((STAGE - 1)))
  test -f "$ARTIFACT/med_prism/stage_$PREV/completion.json"
  test -f "$ARTIFACT/med_prism/stage_$PREV/shared/shared_manifest.json"
fi
if [[ -f "$COMPLETION" ]] && "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("status")=="PASS" else 1)' "$COMPLETION"; then
  echo "SKIP completed stage $STAGE"
  exit 0
fi
mkdir -p "$OUT/train"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader > "$OUT/gpu_before_training.txt"

CMD=(
  "$SWIFT" sft --model Qwen/Qwen3-VL-8B-Instruct --model_revision "$REVISION"
  --template qwen3_vl --dataset "$DATA" --tuner_type med_prism_shared_private
  --freeze_llm false --freeze_vit true --freeze_aligner true
  --torch_dtype bfloat16 --attn_impl sdpa --max_length 1024 --max_pixels 200704
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1
  --learning_rate 1e-4 --max_steps 100 --warmup_ratio 0
  --logging_steps 5 --logging_first_step true --save_strategy steps --save_steps 100
  --save_total_limit 1 --save_only_model false --eval_strategy no
  --gradient_checkpointing true --vit_gradient_checkpointing false
  --ddp_find_unused_parameters false --dataloader_num_workers 0 --dataset_num_proc 1
  --dataset_shuffle false --split_dataset_ratio 0 --packing false --use_hf true
  --seed 42 --data_seed 42 --report_to none --add_version false
  --create_checkpoint_symlink false --logging_dir "$OUT/train/logs" --output_dir "$OUT/train"
  --external_plugins "$ROOT/scripts/phase3/callback_compat.py"
  "$ROOT/med_prism/swift_plugins/med_prism_rank1_plugin.py"
  "$ROOT/med_prism/swift_plugins/med_prism_shared_private_plugin.py"
  "$ROOT/scripts/med_prism_real_5skill/reload_probe_plugin.py"
)
printf '%q ' "${CMD[@]}" > "$OUT/train_command.txt"; printf '\n' >> "$OUT/train_command.txt"
set +e
"${CMD[@]}" 2>&1 | tee "$OUT/train.log"
CODE=${PIPESTATUS[0]}
set -e
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader > "$OUT/gpu_after_training.txt"
CHECKPOINT=$(find "$OUT/train" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)
STATUS=FAIL
if [[ "$CODE" == 0 && -n "$CHECKPOINT" && -f "$OUT/shared/shared_manifest.json" && -f "$OUT/private/private_manifest.json" && -f "$OUT/runtime_audits/task${STAGE}_pre_reload_probe.json" ]]; then STATUS=PASS; fi
"$PYTHON" - "$COMPLETION" "$STATUS" "$CODE" "$CHECKPOINT" "$REVISION" <<'PY'
import json,pathlib,sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({"status":sys.argv[2],"swift_sft_exit_code":int(sys.argv[3]),"checkpoint":sys.argv[4] or None,"model_id":"Qwen/Qwen3-VL-8B-Instruct","model_revision":sys.argv[5],"max_steps":100,"seed":42,"method":"med_prism_shared_private"},indent=2)+"\n")
PY
[[ "$STATUS" == PASS ]]
