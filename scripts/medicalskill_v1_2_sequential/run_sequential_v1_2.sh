#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/MedAgentCL_v4
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
SWIFT=/root/anaconda3/envs/medagentcl_v4/bin/swift
DATA_ROOT=/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k
OUTPUT_ROOT=/remote-home/wangbomin/sequential_medicalskill_v1_2_lite_10k1k_seed42
GPUS=0,1
SEED=42
EVAL_LIMIT=0
RESTART_INCOMPLETE=0
CHECK_ONLY=0
REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
METHOD=sequential_native_lora_r48

usage() {
  cat <<'EOF'
Usage: run_sequential_v1_2.sh [options]
  --data-root PATH       Frozen MedicalSkill dataset root
  --output-root PATH     Independent sequential baseline output
  --gpus LIST            One or two GPU IDs; default: 0,1
  --seed N               Default: 42
  --eval-limit N         Per-task test limit; 0 means full test split
  --restart-incomplete-stage
                         Archive and restart an incomplete stage
  --check-only           Validate the complete contract without GPU work

The run trains one cumulative native LoRA through Tasks 1..5. Stage N starts
from base + the Stage N-1 cumulative adapter. After all training stages PASS,
it evaluates exactly the 15 seen-task lower-triangular cells.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT=$2; shift 2 ;;
    --output-root) OUTPUT_ROOT=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --eval-limit) EVAL_LIMIT=$2; shift 2 ;;
    --restart-incomplete-stage) RESTART_INCOMPLETE=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Invalid seed" >&2; exit 2; }
[[ "$EVAL_LIMIT" =~ ^[0-9]+$ ]] || { echo "Invalid eval limit" >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NPROC_PER_NODE=${#GPU_IDS[@]}
(( NPROC_PER_NODE >= 1 && NPROC_PER_NODE <= 2 )) || { echo "Use one or two GPUs" >&2; exit 2; }
for gpu in "${GPU_IDS[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || exit 2; done
(( 16 % NPROC_PER_NODE == 0 )) || { echo "Global batch 16 must divide by GPU count" >&2; exit 2; }
GRADIENT_ACCUMULATION_STEPS=$((16 / NPROC_PER_NODE))

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub
export HF_DATASETS_CACHE=/remote-home/wangbomin/huggingface_cache/datasets
export XDG_CACHE_HOME=/remote-home/wangbomin/huggingface_cache/xdg
export TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset TRANSFORMERS_CACHE
mkdir -p "$TMPDIR"
cd "$ROOT"

for required in \
  "$DATA_ROOT/audits/acceptance_matrix.json" \
  "$DATA_ROOT/manifests/dataset_lock.json" \
  "$ROOT/scripts/medicalskill_v1_2_sequential/evaluate_sequential_v1_2.py" \
  "$ROOT/scripts/medicalskill_v1_2_sequential/aggregate_sequential_v1_2.py"; do
  test -s "$required"
done
for stage in 1 2 3 4 5; do
  case "$stage" in
    1) task=task_01_vqa ;;
    2) task=task_02_diagnosis_classification ;;
    3) task=task_03_concept_recognition ;;
    4) task=task_04_visual_grounding ;;
    5) task=task_05_reasoning_vqa ;;
  esac
  test -s "$DATA_ROOT/$task/train.jsonl"
  test -s "$DATA_ROOT/$task/test.jsonl"
done

"$PYTHON" - "$DATA_ROOT" "$OUTPUT_ROOT" "$SEED" "$REVISION" "$METHOD" "$CHECK_ONLY" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
data,output,seed,revision,method,check_only=pathlib.Path(sys.argv[1]).resolve(),pathlib.Path(sys.argv[2]).resolve(),int(sys.argv[3]),sys.argv[4],sys.argv[5],int(sys.argv[6])
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
acceptance=json.loads((data/'audits/acceptance_matrix.json').read_text())
lock_path=data/'manifests/dataset_lock.json'; lock=json.loads(lock_path.read_text())
selected=data/lock['selected_ids_file']
assert acceptance.get('status')=='PASS'
assert lock.get('dataset_version')=='MedicalSkill-CL-v1.2-lite-10k1k'
assert sha(selected)==lock['selected_ids_sha256']
task_contract={}
for task in lock['tasks']:
 task_contract[task['task_name']]={}
 for split in ('train','test'):
  expected=task['splits'][split]; path=data/expected['file']
  count=sum(bool(line.strip()) for line in path.open(encoding='utf-8'))
  assert count==expected['records'],(path,count,expected['records'])
  assert sha(path)==expected['sha256'],path
  task_contract[task['task_name']][split]={'records':count,'sha256':expected['sha256']}
payload={'status':'PASS','method':method,'ordinary_lora':True,'continual_learning_constraints':False,'replay':False,'historical_adapter_stack':False,'adapter_semantics':'one cumulative LoRA continued from the previous stage checkpoint','model_id':'Qwen/Qwen3-VL-8B-Instruct','model_revision':revision,'seed':seed,'data_root':str(data),'dataset_version':lock['dataset_version'],'dataset_lock_sha256':sha(lock_path),'selected_ids_sha256':lock['selected_ids_sha256'],'selected_id_count':lock['selected_id_count'],'dataset_tasks':task_contract,'output_root':str(output),'task_order':[t['task_name'] for t in lock['tasks']],'training_epochs_per_task':1,'evaluation_protocol':'exactly_15_lower_triangular_seen_task_cells_no_base'}
if check_only: print(json.dumps(payload,indent=2)); raise SystemExit(0)
output.mkdir(parents=True,exist_ok=True); path=output/'run_manifest.json'
if path.exists() and json.loads(path.read_text())!=payload: raise RuntimeError(f'Existing run contract differs: {path}')
if not path.exists():
 fd,tmp=tempfile.mkstemp(prefix='.run_manifest.',dir=output)
 with os.fdopen(fd,'w') as f: json.dump(payload,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
print(json.dumps(payload,indent=2))
PY
if (( CHECK_ONLY == 1 )); then
  echo "Sequential contract check PASS"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export NPROC_PER_NODE
echo "Sequential training on GPUs $GPUS; DDP processes=$NPROC_PER_NODE; global batch=16"

for STAGE in 1 2 3 4 5; do
  PADDED=$(printf '%02d' "$STAGE")
  case "$STAGE" in
    1) TASK_NAME=task_01_vqa ;;
    2) TASK_NAME=task_02_diagnosis_classification ;;
    3) TASK_NAME=task_03_concept_recognition ;;
    4) TASK_NAME=task_04_visual_grounding ;;
    5) TASK_NAME=task_05_reasoning_vqa ;;
  esac
  STAGE_ROOT="$OUTPUT_ROOT/sequential/stage_$PADDED"
  TRAIN_ROOT="$STAGE_ROOT/train"
  COMPLETION="$STAGE_ROOT/completion.json"
  DATASET="$DATA_ROOT/$TASK_NAME/train.jsonl"
  MAX_LENGTH=1024
  (( STAGE == 5 )) && MAX_LENGTH=4096

  if [[ -s "$COMPLETION" ]] && "$PYTHON" -c 'import json,pathlib,sys;x=json.load(open(sys.argv[1]));p=x.get("checkpoint");raise SystemExit(0 if x.get("status")=="PASS" and p and pathlib.Path(p).is_dir() else 1)' "$COMPLETION"; then
    echo "[stage $STAGE] already PASS; skipping"
    continue
  fi
  PREVIOUS_CHECKPOINT=
  PREVIOUS_COMPLETION=
  if (( STAGE > 1 )); then
    PREVIOUS=$(printf '%02d' $((STAGE - 1)))
    PREVIOUS_COMPLETION="$OUTPUT_ROOT/sequential/stage_$PREVIOUS/completion.json"
    test -s "$PREVIOUS_COMPLETION"
    PREVIOUS_CHECKPOINT=$("$PYTHON" -c 'import json,pathlib,sys;x=json.load(open(sys.argv[1]));p=x.get("checkpoint");ok=x.get("status")=="PASS" and p and pathlib.Path(p).is_dir();print(p) if ok else sys.exit(1)' "$PREVIOUS_COMPLETION")
  fi
  if [[ -d "$STAGE_ROOT" ]] && find "$STAGE_ROOT" -mindepth 1 -print -quit | grep -q .; then
    if (( RESTART_INCOMPLETE != 1 )); then
      echo "[stage $STAGE] incomplete directory exists: $STAGE_ROOT" >&2
      echo "Rerun with --restart-incomplete-stage" >&2
      exit 3
    fi
    ARCHIVE="$OUTPUT_ROOT/failed_attempts/stage_${PADDED}_$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$OUTPUT_ROOT/failed_attempts"
    mv "$STAGE_ROOT" "$ARCHIVE"
    printf '%s\n' "$ARCHIVE" >> "$OUTPUT_ROOT/failed_attempts/archive_index.txt"
    echo "[stage $STAGE] archived incomplete attempt: $ARCHIVE"
  fi
  mkdir -p "$TRAIN_ROOT"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader > "$STAGE_ROOT/gpu_before_training.txt"
  CMD=(
    "$SWIFT" sft
    --model Qwen/Qwen3-VL-8B-Instruct
    --model_revision "$REVISION"
    --template qwen3_vl
    --dataset "$DATASET"
    --tuner_type lora
    --tuner_backend peft
    --target_modules q_proj v_proj
    --lora_rank 48
    --lora_alpha 96
    --lora_dropout 0.05
    --lora_bias none
    --freeze_llm false
    --freeze_vit true
    --freeze_aligner true
    --torch_dtype bfloat16
    --attn_impl sdpa
    --max_length "$MAX_LENGTH"
    --max_pixels 200704
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate 1e-4
    --num_train_epochs 1
    --warmup_ratio 0.03
    --logging_steps 10
    --logging_first_step true
    --save_strategy epoch
    --save_total_limit 1
    --save_only_model false
    --eval_strategy no
    --gradient_checkpointing true
    --vit_gradient_checkpointing false
    --ddp_find_unused_parameters false
    --dataloader_num_workers 0
    --dataset_num_proc 1
    --dataset_shuffle true
    --split_dataset_ratio 0
    --packing false
    --use_hf true
    --seed "$SEED"
    --data_seed "$SEED"
    --report_to none
    --add_version false
    --create_checkpoint_symlink false
    --logging_dir "$TRAIN_ROOT/logs"
    --output_dir "$TRAIN_ROOT"
    --external_plugins "$ROOT/scripts/phase3/callback_compat.py"
  )
  if (( STAGE > 1 )); then
    CMD+=(--adapters "$PREVIOUS_CHECKPOINT" --load_args false)
  fi
  printf '%q ' "${CMD[@]}" > "$STAGE_ROOT/train_command.txt"; printf '\n' >> "$STAGE_ROOT/train_command.txt"
  echo "[stage $STAGE] $TASK_NAME initialization: $([[ $STAGE == 1 ]] && echo base || echo base+stage_$((STAGE-1))_cumulative_adapter)"
  set +e
  "${CMD[@]}" 2>&1 | tee "$STAGE_ROOT/train.log"
  CODE=${PIPESTATUS[0]}
  set -e
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader > "$STAGE_ROOT/gpu_after_training.txt"
  CHECKPOINT=$(find "$TRAIN_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)
  "$PYTHON" - "$COMPLETION" "$STAGE" "$CODE" "$CHECKPOINT" "$PREVIOUS_COMPLETION" "$PREVIOUS_CHECKPOINT" "$REVISION" "$SEED" <<'PY'
import hashlib,json,os,pathlib,sys,tempfile
path,stage,code,checkpoint,previous_completion,previous_checkpoint,revision,seed=pathlib.Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),pathlib.Path(sys.argv[4]) if sys.argv[4] else None,sys.argv[5],sys.argv[6],sys.argv[7],int(sys.argv[8])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
weights=None
if checkpoint:
 for name in ('adapter_model.safetensors','adapter_model.bin'):
  candidate=checkpoint/name
  if candidate.is_file(): weights=candidate; break
config=checkpoint/'adapter_config.json' if checkpoint else None
ok=code==0 and checkpoint is not None and checkpoint.is_dir() and config.is_file() and weights is not None
previous_hash=None
if previous_completion:
 previous=json.load(open(previous_completion)); previous_hash=previous.get('adapter_weights_sha256')
payload={'status':'PASS' if ok else 'FAIL','stage':stage,'method':'sequential_native_lora_r48','ordinary_lora':True,'continual_learning_constraints':False,'checkpoint':str(checkpoint) if checkpoint else None,'swift_sft_exit_code':code,'model_revision':revision,'seed':seed,'initialization':'base_only' if stage==1 else 'base_plus_previous_cumulative_adapter','previous_completion':previous_completion or None,'previous_checkpoint':previous_checkpoint or None,'previous_adapter_weights_sha256':previous_hash,'adapter_config_sha256':sha(config) if ok else None,'adapter_weights_file':str(weights) if weights else None,'adapter_weights_sha256':sha(weights) if ok else None,'historical_adapter_stack':False,'cumulative_adapter_count_loaded':0 if stage==1 else 1}
path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.completion.',dir=path.parent)
with os.fdopen(fd,'w') as f:json.dump(payload,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,path);print(json.dumps(payload,indent=2));raise SystemExit(0 if ok else 1)
PY
  echo "[stage $STAGE] PASS"
done

for STAGE in 1 2 3 4 5; do
  PADDED=$(printf '%02d' "$STAGE")
  test -s "$OUTPUT_ROOT/sequential/stage_$PADDED/completion.json"
  "$PYTHON" -c 'import json,sys;raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="PASS" else 1)' "$OUTPUT_ROOT/sequential/stage_$PADDED/completion.json"
done

evaluate_stage() {
  local gpu=$1 stage=$2 padded adapter
  padded=$(printf '%02d' "$stage")
  adapter=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint"])' "$OUTPUT_ROOT/sequential/stage_$padded/completion.json")
  echo "[evaluation stage $stage] GPU $gpu, cells 1..$stage"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/medicalskill_v1_2_sequential/evaluate_sequential_v1_2.py \
    --stage "$stage" --adapter "$adapter" --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" --eval-limit "$EVAL_LIMIT" \
    2>&1 | tee "$OUTPUT_ROOT/evaluation_stage_${padded}.log"
}
run_lane() { local gpu=$1; shift; local stage; for stage in "$@"; do evaluate_stage "$gpu" "$stage"; done; }

if (( NPROC_PER_NODE == 1 )); then
  run_lane "${GPU_IDS[0]}" 1 2 3 4 5
else
  run_lane "${GPU_IDS[0]}" 2 5 & LANE0=$!
  run_lane "${GPU_IDS[1]}" 1 3 4 & LANE1=$!
  STATUS=0; wait "$LANE0" || STATUS=1; wait "$LANE1" || STATUS=1
  (( STATUS == 0 )) || { echo "Sequential evaluation lane failed" >&2; exit 1; }
fi

"$PYTHON" scripts/medicalskill_v1_2_sequential/aggregate_sequential_v1_2.py \
  --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --eval-limit "$EVAL_LIMIT" \
  2>&1 | tee "$OUTPUT_ROOT/aggregate.log"
echo "Sequential train+15-cell evaluation PASS: $OUTPUT_ROOT"
