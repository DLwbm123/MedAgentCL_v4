#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/MedAgentCL_v4
SOURCE="$ROOT/output/phase3_native_lora"
OUT="$ROOT/output/phase4_rank1/phase3_closure"
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
CACHE=/remote-home/wangbomin/huggingface_cache/hub

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE="$CACHE"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1

cd "$ROOT"
if [[ -e "$OUT/train/checkpoint-10" ]]; then
  echo "Refusing to overwrite existing Phase 3 closure checkpoint: $OUT/train/checkpoint-10" >&2
  exit 2
fi

mkdir -p "$OUT/data"
cp "$SOURCE/data/smoke_train.jsonl" "$OUT/data/"
cp "$SOURCE/data/smoke_eval.jsonl" "$OUT/data/"
cp "$SOURCE/data/image_token_grid_stats.json" "$OUT/data/"
cp "$SOURCE/data/smoke_data_manifest.json" "$OUT/data/"
cp "$SOURCE/data/smoke_schema_report.json" "$OUT/data/"
cp "$SOURCE/target_modules_pre_injection.txt" "$OUT/"

"$PYTHON" scripts/phase3/native_lora_tools.py targets \
  --module-names "$OUT/target_modules_pre_injection.txt" \
  --output-dir "$OUT"

set +e
PHASE3_OUTPUT_DIR="$OUT" \
  bash scripts/phase3/train_phase3_native_lora.sh \
  2>&1 | tee "$OUT/native_lora_closure.log"
train_exit=${PIPESTATUS[0]}
set -e
if [[ "$train_exit" -ne 0 ]]; then
  echo "Phase 3 closure training failed with exit code $train_exit" >&2
  exit "$train_exit"
fi

{
  "$PYTHON" scripts/phase3/native_lora_tools.py adapter \
    --adapter-dir "$OUT/train/checkpoint-10" \
    --output-dir "$OUT" --rank 48 --alpha 96 --dropout 0.05
  printf '%s\n' "$OUT/train/checkpoint-10" > "$OUT/adapter_path.txt"
  "$PYTHON" scripts/phase3/native_lora_tools.py infer \
    --variant base --run-index 1 \
    --eval-jsonl "$OUT/data/smoke_eval.jsonl" \
    --output-dir "$OUT" --cache-dir "$CACHE" --device cuda:0
  "$PYTHON" scripts/phase3/native_lora_tools.py infer \
    --variant adapter --run-index 1 \
    --adapter-dir "$OUT/train/checkpoint-10" \
    --eval-jsonl "$OUT/data/smoke_eval.jsonl" \
    --output-dir "$OUT" --cache-dir "$CACHE" --device cuda:0
  "$PYTHON" scripts/phase3/native_lora_tools.py infer \
    --variant adapter --run-index 2 \
    --adapter-dir "$OUT/train/checkpoint-10" \
    --eval-jsonl "$OUT/data/smoke_eval.jsonl" \
    --output-dir "$OUT" --cache-dir "$CACHE" --device cuda:0
  "$PYTHON" scripts/phase3/native_lora_tools.py aggregate \
    --output-dir "$OUT" --reload-tolerance 1e-5
} 2>&1 | tee -a "$OUT/native_lora_closure.log"

"$PYTHON" scripts/phase3/phase3_closure_report.py --output-dir "$OUT" \
  2>&1 | tee -a "$OUT/native_lora_closure.log"
