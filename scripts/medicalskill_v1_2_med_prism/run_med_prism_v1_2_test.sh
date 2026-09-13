#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/MedAgentCL_v4
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
DATA_ROOT=/remote-home/wangbomin/MedicalSkill-CL-v1.2
OUTPUT_ROOT=/remote-home/wangbomin/med_prism_medicalskill_v1_2_seed42
GPUS=0,1
EVAL_LIMIT=0
BATCH_SIZE=4
SEED=42
WITH_ORACLE=0
WITH_STAGE0=0
BASE_EVAL_ROOT=
PLAN_ONLY=0

usage() {
  cat <<'EOF'
Usage: run_med_prism_v1_2_test.sh [options]
  --data-root PATH       Dataset root
  --output-root PATH     Must match the training output root
  --gpus LIST            Comma-separated GPUs; default: 0,1
  --gpu ID               Backward-compatible single-GPU alias
  --eval-limit N         Per-task limit; 0 means complete test split
  --batch-size N         Per-GPU generation batch size; default: 4
  --base-eval-root PATH  Canonical compatible Stage 0 output to reuse
  --with-stage0          Explicitly generate five Stage 0 cells
  --with-oracle          Explicitly generate 15 Oracle diagnostic cells
  --plan-only            Validate and print the plan; no model loading

Default evaluation generates only the 15 Primary lower-triangular cells. Stage 0
is reused and Oracle is disabled. Counts: default=15, +Oracle=30,
+Stage0=20, +Stage0+Oracle=35. If --base-eval-root is omitted, OUTPUT_ROOT is
used as the canonical Stage 0 source and must already contain compatible cells.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT=$2; shift 2 ;;
    --output-root) OUTPUT_ROOT=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --gpu) GPUS=$2; shift 2 ;;
    --eval-limit) EVAL_LIMIT=$2; shift 2 ;;
    --batch-size) BATCH_SIZE=$2; shift 2 ;;
    --base-eval-root) BASE_EVAL_ROOT=$2; shift 2 ;;
    --with-stage0) WITH_STAGE0=1; shift ;;
    --with-oracle) WITH_ORACLE=1; shift ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$EVAL_LIMIT" =~ ^[0-9]+$ ]] || { echo "--eval-limit must be non-negative" >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "--batch-size must be positive" >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
(( ${#GPU_IDS[@]} >= 1 && ${#GPU_IDS[@]} <= 2 )) || { echo "--gpus requires one or two GPU IDs" >&2; exit 2; }
for GPU_ID in "${GPU_IDS[@]}"; do
  [[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: $GPU_ID" >&2; exit 2; }
done
if (( ${#GPU_IDS[@]} == 2 )) && [[ "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
  echo "--gpus requires distinct GPU IDs" >&2
  exit 2
fi
if (( WITH_STAGE0 == 0 )) && [[ -z "$BASE_EVAL_ROOT" ]]; then
  BASE_EVAL_ROOT=$OUTPUT_ROOT
fi

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub
export HF_DATASETS_CACHE=/remote-home/wangbomin/huggingface_cache/datasets
export XDG_CACHE_HOME=/remote-home/wangbomin/huggingface_cache/xdg
export TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MED_PRISM_ARTIFACT_ROOT="$OUTPUT_ROOT"
export MED_PRISM_DATA_ROOT="$DATA_ROOT"
export MED_PRISM_EVAL_LIMIT="$EVAL_LIMIT"
unset TRANSFORMERS_CACHE
mkdir -p "$TMPDIR"
cd "$ROOT"

RUNNER_ARGS=(--gpus "$GPUS" --eval-limit "$EVAL_LIMIT" --batch-size "$BATCH_SIZE")
RUNNER_ARGS+=(--recovery-records-log "$OUTPUT_ROOT/parallel_evaluation.log")
(( WITH_ORACLE == 1 )) && RUNNER_ARGS+=(--with-oracle)
(( WITH_STAGE0 == 1 )) && RUNNER_ARGS+=(--with-stage0)
(( WITH_STAGE0 == 0 )) && RUNNER_ARGS+=(--base-eval-root "$BASE_EVAL_ROOT")
(( PLAN_ONLY == 1 )) && RUNNER_ARGS+=(--plan-only)

if (( PLAN_ONLY == 0 )); then
  mkdir -p "$OUTPUT_ROOT/evaluation"
  "$PYTHON" scripts/medicalskill_v1_2_med_prism/prepare_formal_v1_2.py \
    --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --seed "$SEED" --check-only
  for STAGE in 1 2 3 4 5; do
    PADDED=$(printf '%02d' "$STAGE")
    test -s "$OUTPUT_ROOT/med_prism/stage_$PADDED/completion.json"
    test -s "$OUTPUT_ROOT/med_prism/stage_$PADDED/shared/shared_manifest.json"
    test -s "$OUTPUT_ROOT/med_prism/stage_$PADDED/private/private_manifest.json"
    "$PYTHON" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="PASS" else 1)' \
      "$OUTPUT_ROOT/med_prism/stage_$PADDED/completion.json"
  done
fi

"$PYTHON" scripts/medicalskill_v1_2_med_prism/run_parallel_formal_v1_2.py "${RUNNER_ARGS[@]}" \
  2>&1 | tee -a "${OUTPUT_ROOT}/parallel_evaluation.log"

if (( PLAN_ONLY == 0 )); then
  "$PYTHON" scripts/medicalskill_v1_2_med_prism/aggregate_formal_v1_2.py \
    --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --eval-limit "$EVAL_LIMIT" \
    2>&1 | tee "$OUTPUT_ROOT/aggregate.log"
  echo "Formal evaluation completed. Output: $OUTPUT_ROOT"
fi
