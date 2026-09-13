#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/MedAgentCL_v4
PY=/root/anaconda3/envs/medagentcl_v4/bin/python
DIR=$ROOT/scripts/skill_incremental_v1
ART=$ROOT/artifacts/medicalskill_cl_v1_closure
LOG=$ART/closure_pipeline.log
export HF_HOME=/remote-home/wangbomin/huggingface_cache HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
exec > >(tee -a "$LOG") 2>&1
echo "closure_resume_from_strict=$(date -Is)"
echo "stage=strict_validation_optimized $(date -Is)"
"$PY" "$DIR/validate_medicalskill_cl_v1.py" --reuse-path-validation "$ART/path_validation_monotonicity.json"
echo "stage=swift_schema $(date -Is)"
cd "$ROOT";"$PY" "$DIR/swift_schema_smoke_v1.py"
echo "stage=model_smoke $(date -Is)"
CUDA_VISIBLE_DEVICES=0 bash "$DIR/run_model_smoke_v1.sh"
echo "stage=finalize $(date -Is)"
"$PY" "$DIR/finalize_closure.py"
echo "closure_complete=$(date -Is)"
