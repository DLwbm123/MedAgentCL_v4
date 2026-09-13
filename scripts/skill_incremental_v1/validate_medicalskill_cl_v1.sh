#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/MedAgentCL_v4
PY=/root/anaconda3/envs/medagentcl_v4/bin/python
DIR=$ROOT/scripts/skill_incremental_v1
export HF_HOME=/remote-home/wangbomin/huggingface_cache HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"
"$PY" -m unittest discover -s tests/skill_incremental_v1 -v
"$PY" "$DIR/validate_medicalskill_cl_v1.py"
"$PY" "$DIR/swift_schema_smoke_v1.py"
