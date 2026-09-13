#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/MedAgentCL_v4
DIR=$ROOT/scripts/skill_incremental_v1
PY=/root/anaconda3/envs/medagentcl_v4/bin/python
DATA=/remote-home/wangbomin/MedicalSkill-CL-v1
ART=$ROOT/artifacts/medicalskill_cl_v1_closure
REV=b968826d9c46dd6066d109eabc6255188de91218
MODEL=/remote-home/wangbomin/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/$REV
export HF_HOME=/remote-home/wangbomin/huggingface_cache HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
mkdir -p "$DATA/_concept/logs" "$ART" "$TMPDIR"
mode=${1:-all}
case "$mode" in
 prepare) exec "$PY" "$DIR/prepare_medtrinity_v1_inputs.py";;
 pilot) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} exec "$PY" "$DIR/extract_medtrinity_concepts_v1.py" --input "$DATA/_concept/pilot500_input.jsonl" --output "$DATA/_concept/pilot500_output.jsonl" --model-path "$MODEL" --revision "$REV" --batch-size 48 --max-new-tokens 512 --seed 42 --resume --qa-report "$ART/concept_semantic_audit_500.json";;
 split) "$PY" -c 'import pathlib,sys;src=pathlib.Path(sys.argv[1]);hs=[pathlib.Path(x).open("w") for x in sys.argv[2:]];[(hs[i%2].write(line)) for i,line in enumerate(src.open()) if line.strip()];[h.close() for h in hs]' "$DATA/_concept/merged_input.jsonl" "$DATA/_concept/full_input_part_00.jsonl" "$DATA/_concept/full_input_part_01.jsonl";;
 extract)
  "$0" split
  CUDA_VISIBLE_DEVICES=0 "$PY" "$DIR/extract_medtrinity_concepts_v1.py" --input "$DATA/_concept/full_input_part_00.jsonl" --output "$DATA/_concept/full_output_part_00.jsonl" --model-path "$MODEL" --revision "$REV" --batch-size 48 --max-new-tokens 512 --seed 42 --resume --qa-report "$ART/concept_full_part_00_qa.json" >"$DATA/_concept/logs/full_part_00.log" 2>&1 & p0=$!
  CUDA_VISIBLE_DEVICES=1 "$PY" "$DIR/extract_medtrinity_concepts_v1.py" --input "$DATA/_concept/full_input_part_01.jsonl" --output "$DATA/_concept/full_output_part_01.jsonl" --model-path "$MODEL" --revision "$REV" --batch-size 48 --max-new-tokens 512 --seed 43 --resume --qa-report "$ART/concept_full_part_01_qa.json" >"$DATA/_concept/logs/full_part_01.log" 2>&1 & p1=$!
  wait "$p0";wait "$p1";;
 reparse) exec "$PY" "$DIR/reparse_concepts_v1.py" --input "$DATA/_concept/full_output_part_00.jsonl" --input "$DATA/_concept/full_output_part_01.jsonl";;
 build) exec "$PY" "$DIR/build_medicalskill_cl_v1.py" --concept-extraction "$DATA/_concept/full_output_part_00.jsonl" --concept-extraction "$DATA/_concept/full_output_part_01.jsonl";;
 phash) exec "$PY" "$DIR/phash_closure.py" --apply-confirmed-repair;;
 validate) "$PY" "$DIR/validate_medicalskill_cl_v1.py";exec "$PY" "$DIR/swift_schema_smoke_v1.py";;
 all) "$0" prepare;"$0" pilot;"$0" extract;"$0" reparse;"$0" build;"$0" phash;exec "$0" validate;;
 *) echo "usage: $0 {prepare|pilot|split|extract|reparse|build|phash|validate|all}" >&2;exit 2;;
esac
