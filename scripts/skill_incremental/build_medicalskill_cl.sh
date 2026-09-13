#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/MedAgentCL_v4
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
DIR="$ROOT/scripts/skill_incremental"
DATA=/remote-home/wangbomin/MedicalSkill-CL
REV=b968826d9c46dd6066d109eabc6255188de91218
MODEL=/remote-home/wangbomin/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/$REV
export HF_HOME=${HF_HOME:-/remote-home/wangbomin/huggingface_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=${TMPDIR:-$HF_HOME/tmp}
mkdir -p "$DATA/_concept" "$TMPDIR"
mode=${1:-all}; shift || true
case "$mode" in
 dry-run) exec "$PYTHON" "$DIR/build_medicalskill_cl.py" --dry-run "$@";;
 pilot) exec "$PYTHON" "$DIR/extract_medtrinity_concepts.py" --input /root/MedAgentCL/data/MedTrinity_ConceptCaption_CL/concept_train.jsonl --output "$DATA/_concept/pilot.jsonl" --qa-report "$DATA/_concept/pilot.qa.json" --model-path "$MODEL" --revision "$REV" --limit 500 "$@";;
 extract) exec "$PYTHON" "$DIR/extract_medtrinity_concepts.py" --input /root/MedAgentCL/data/MedTrinity_ConceptCaption_CL/concept_train.jsonl --input /root/MedAgentCL/data/MedTrinity_ConceptCaption_CL/concept_test.jsonl --output "$DATA/_concept/full.jsonl" --model-path "$MODEL" --revision "$REV" --resume "$@";;
 build) exec "$PYTHON" "$DIR/build_medicalskill_cl.py" "$@";;
 validate) exec "$PYTHON" "$DIR/validate_medicalskill_cl.py" --resume "$@";;
 all)
  "$0" pilot --resume
  "$0" extract
  "$0" build "$@"
  exec "$0" validate
  ;;
 *) echo "usage: $0 {dry-run|pilot|extract|build|validate|all}" >&2; exit 2;;
esac
