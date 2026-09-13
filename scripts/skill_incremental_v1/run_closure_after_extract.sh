#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/MedAgentCL_v4
PY=/root/anaconda3/envs/medagentcl_v4/bin/python
DIR=$ROOT/scripts/skill_incremental_v1
DATA=/remote-home/wangbomin/MedicalSkill-CL-v1
ART=$ROOT/artifacts/medicalskill_cl_v1_closure
LOG=$ART/closure_pipeline.log
export HF_HOME=/remote-home/wangbomin/huggingface_cache HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
exec > >(tee -a "$LOG") 2>&1
echo "watcher_started=$(date -Is)"
while pgrep -f 'extract_medtrinity_concepts_v1.py --input .*full_input_part_' >/dev/null;do
 echo "waiting_for_extraction $(date -Is) $(wc -l "$DATA"/_concept/full_output_part_*.jsonl.partial 2>/dev/null | tail -1 || true)"
 sleep 120
done
for part in 00 01;do
 file="$DATA/_concept/full_output_part_${part}.jsonl"
 [[ -f "$file" ]] || { echo "missing $file";exit 3; }
 [[ $(wc -l <"$file") -eq 35178 ]] || { echo "wrong count $file";exit 4; }
done
echo "stage=reparse $(date -Is)"
"$PY" "$DIR/reparse_concepts_v1.py" --input "$DATA/_concept/full_output_part_00.jsonl" --input "$DATA/_concept/full_output_part_01.jsonl"
echo "stage=build $(date -Is)"
"$PY" "$DIR/build_medicalskill_cl_v1.py" --concept-extraction "$DATA/_concept/full_output_part_00.jsonl" --concept-extraction "$DATA/_concept/full_output_part_01.jsonl"
echo "stage=phash_repair $(date -Is)"
"$PY" "$DIR/phash_closure.py" --apply-confirmed-repair
cp "$ART/phash_candidate_classification.json" "$ART/phash_candidate_classification_before_repair.json"
echo "stage=phash_verify $(date -Is)"
"$PY" "$DIR/phash_closure.py"
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
