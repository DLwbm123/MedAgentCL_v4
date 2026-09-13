#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/MedAgentCL_v4
PY=/root/anaconda3/envs/medagentcl_v4/bin/python
DIR=$ROOT/scripts/skill_incremental_v1
ART=$ROOT/artifacts/medicalskill_cl_v1_closure
LOG=$ART/closure_pipeline.log
export HF_HOME=/remote-home/wangbomin/huggingface_cache HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
exec > >(tee -a "$LOG") 2>&1
echo "closure_resume_from_phash=$(date -Is)"
echo "stage=phash_repair_optimized $(date -Is)"
"$PY" "$DIR/phash_closure.py" --apply-confirmed-repair --io-workers 16
cp "$ART/phash_candidate_classification.json" "$ART/phash_candidate_classification_before_repair.json"
echo "stage=phash_verify_optimized $(date -Is)"
"$PY" "$DIR/phash_closure.py" --io-workers 16
echo "stage=unit_and_static $(date -Is)"
"$PY" -m unittest discover -s "$ROOT/tests/skill_incremental_v1" -v
"$PY" -m compileall -q "$DIR" "$ROOT/tests/skill_incremental_v1"
bash -n "$DIR"/*.sh
echo "stage=strict_validation $(date -Is)"
"$PY" "$DIR/validate_medicalskill_cl_v1.py"
echo "stage=swift_schema $(date -Is)"
cd "$ROOT";"$PY" "$DIR/swift_schema_smoke_v1.py"
echo "stage=model_smoke $(date -Is)"
CUDA_VISIBLE_DEVICES=0 bash "$DIR/run_model_smoke_v1.sh"
echo "stage=finalize $(date -Is)"
"$PY" "$DIR/finalize_closure.py"
echo "closure_complete=$(date -Is)"
