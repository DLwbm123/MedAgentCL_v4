#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/MedAgentCL_v4
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
METHOD_VERSION=
DATA_ROOT=
OUTPUT_ROOT=
GPUS=0,1
BATCH_SIZE=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method-version) METHOD_VERSION=$2; shift 2 ;;
    --data-root) DATA_ROOT=$2; shift 2 ;;
    --output-root) OUTPUT_ROOT=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --batch-size) BATCH_SIZE=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$METHOD_VERSION" == "1.1" || "$METHOD_VERSION" == "1.2" ]] || exit 2
[[ -n "$DATA_ROOT" && -n "$OUTPUT_ROOT" ]] || exit 2
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub
export HF_DATASETS_CACHE=/remote-home/wangbomin/huggingface_cache/datasets
export XDG_CACHE_HOME=/remote-home/wangbomin/huggingface_cache/xdg
export TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MED_PRISM_ARTIFACT_ROOT="$OUTPUT_ROOT"
export MED_PRISM_DATA_ROOT="$DATA_ROOT"
export MED_PRISM_EVAL_LIMIT=0
mkdir -p "$OUTPUT_ROOT/evaluation" "$TMPDIR"
cd "$ROOT"
if [[ "$METHOD_VERSION" == "1.1" ]]; then
  IFS=',' read -r -a IDS <<< "$GPUS"
  GPU=${IDS[0]}
  for SPEC in "2 primary_cumulative 2" "3 primary_cumulative 2" "3 oracle_skill_aware 2" "3 primary_cumulative 3" "3 oracle_skill_aware 3"; do
    read -r STAGE MODE TASK <<< "$SPEC"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/medicalskill_v1_2_med_prism/evaluate_formal_v1_2_cell.py \
      --stage "$STAGE" --mode "$MODE" --task "$TASK" --eval-limit 0 --batch-size "$BATCH_SIZE"
  done
else
  "$PYTHON" scripts/medicalskill_v1_2_med_prism/run_parallel_formal_v1_2.py \
    --gpus "$GPUS" --eval-limit 0 --batch-size "$BATCH_SIZE" --with-oracle \
    --base-eval-root /remote-home/wangbomin/med_prism_medicalskill_v1_2_lite_10k1k_seed42 \
    --recovery-records-log "$OUTPUT_ROOT/pilot_parallel_evaluation.log" \
    2>&1 | tee -a "$OUTPUT_ROOT/pilot_parallel_evaluation.log"
fi
"$PYTHON" scripts/medicalskill_v1_2_med_prism_versions/summarize_pilot.py \
  --method-version "$METHOD_VERSION" --output-root "$OUTPUT_ROOT"
