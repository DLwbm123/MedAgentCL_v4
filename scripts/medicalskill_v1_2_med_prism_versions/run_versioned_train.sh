#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/MedAgentCL_v4
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
SWIFT=/root/anaconda3/envs/medagentcl_v4/bin/swift
DATA_ROOT=/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k
OUTPUT_ROOT=
GPUS=0
SEED=42
START_STAGE=1
END_STAGE=5
RESTART_INCOMPLETE=0
REFRESH_SOURCE_PROVENANCE=0
REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PRIVATE_EXPERTS_PER_TASK=16
PRIVATE_RANK=16
PRIVATE_ADAPTER_TYPE=rank1_expert_bank
PRIVATE_ALPHA=16
ORTH_LAMBDA=0.1
METHOD_VERSION=
SHARED_LR=1e-5
PRIVATE_LR=1e-4
LAMBDA_DRIFT=0.01
LAMBDA_KEY=0.1
GRADIENT_DIAGNOSTICS_INTERVAL=0
PILOT_MODE=0
CHECK_ONLY=0
ABLATION_NAME=

usage() {
  cat <<'EOF'
Usage: run_versioned_train.sh --method-version 1.1|1.2 --output-root PATH [options]
  --ablation-name NAME   no_geo_orth, no_shared_drift, or rank16_private
  --data-root PATH       Default: MedicalSkill-CL-v1.2-lite-10k1k
  --output-root PATH     Required and version-specific
  --gpu ID               Single-GPU compatibility option
  --gpus LIST            Comma-separated GPUs; default: 0 (use 0,1 for DDP)
  --seed N               Default: 42
  --start-stage N        Default: 1
  --end-stage N          Default: 5
  --restart-incomplete-stage
                         Archive and restart an incomplete selected stage
  --refresh-source-provenance-after-failed-run
                         Archive stale provenance; requires restart-incomplete
  --private-experts-per-task N  Default: 16
  --private-alpha FLOAT          Default: 16
  --orth-lambda FLOAT            Default: 0.1

The script resumes only at completed stage boundaries. If a stage is interrupted,
move its incomplete stage directory aside before rerunning that stage.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method-version) METHOD_VERSION=$2; shift 2 ;;
    --ablation-name) ABLATION_NAME=$2; shift 2 ;;
    --data-root) DATA_ROOT=$2; shift 2 ;;
    --output-root) OUTPUT_ROOT=$2; shift 2 ;;
    --gpu) GPUS=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --start-stage) START_STAGE=$2; shift 2 ;;
    --end-stage) END_STAGE=$2; shift 2 ;;
    --restart-incomplete-stage) RESTART_INCOMPLETE=1; shift ;;
    --refresh-source-provenance-after-failed-run) REFRESH_SOURCE_PROVENANCE=1; shift ;;
    --private-experts-per-task) PRIVATE_EXPERTS_PER_TASK=$2; shift 2 ;;
    --private-alpha) PRIVATE_ALPHA=$2; shift 2 ;;
    --orth-lambda) ORTH_LAMBDA=$2; shift 2 ;;
    --shared-lr) SHARED_LR=$2; shift 2 ;;
    --private-lr) PRIVATE_LR=$2; shift 2 ;;
    --lambda-drift) LAMBDA_DRIFT=$2; shift 2 ;;
    --lambda-key) LAMBDA_KEY=$2; shift 2 ;;
    --gradient-diagnostics-interval) GRADIENT_DIAGNOSTICS_INTERVAL=$2; shift 2 ;;
    --pilot-mode) PILOT_MODE=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$METHOD_VERSION" == "1.1" || "$METHOD_VERSION" == "1.2" ]] || { echo "--method-version must be 1.1 or 1.2" >&2; exit 2; }
[[ -n "$OUTPUT_ROOT" ]] || { echo "--output-root is required" >&2; exit 2; }
case "$ABLATION_NAME" in
  "")
    EXPECTED_ORTH=0.1; EXPECTED_DRIFT=0.01; EXPECTED_KEY=0.1 ;;
  no_geo_orth)
    EXPECTED_ORTH=0; EXPECTED_DRIFT=0.01; EXPECTED_KEY=0.1 ;;
  no_shared_drift)
    EXPECTED_ORTH=0.1; EXPECTED_DRIFT=0; EXPECTED_KEY=0.1 ;;
  rank16_private)
    EXPECTED_ORTH=0; EXPECTED_DRIFT=0.01; EXPECTED_KEY=0
    PRIVATE_EXPERTS_PER_TASK=1
    PRIVATE_ADAPTER_TYPE=standard_rank16_lora ;;
  *) echo "Unknown --ablation-name: $ABLATION_NAME" >&2; exit 2 ;;
esac
if [[ "$METHOD_VERSION" == "1.1" ]]; then
  [[ -z "$ABLATION_NAME" ]] || { echo "Ablations require v1.2" >&2; exit 2; }
  EXPECTED_KEY=0
fi
[[ "$ORTH_LAMBDA" == "$EXPECTED_ORTH" ]] || { echo "Ablation locks orth lambda to $EXPECTED_ORTH" >&2; exit 2; }
[[ "$LAMBDA_DRIFT" == "$EXPECTED_DRIFT" ]] || { echo "Ablation locks drift lambda to $EXPECTED_DRIFT" >&2; exit 2; }
[[ "$LAMBDA_KEY" == "$EXPECTED_KEY" ]] || { echo "Ablation locks key lambda to $EXPECTED_KEY" >&2; exit 2; }
if [[ "$PRIVATE_ADAPTER_TYPE" == rank1_expert_bank ]]; then
  (( PRIVATE_EXPERTS_PER_TASK == 16 )) || { echo "Rank-1 bank is locked to 16 experts" >&2; exit 2; }
else
  (( PRIVATE_EXPERTS_PER_TASK == 1 && PRIVATE_RANK == 16 )) || exit 2
fi
[[ "$PRIVATE_ALPHA" == "16" ]] || { echo "Private alpha is locked to 16" >&2; exit 2; }
[[ "$GRADIENT_DIAGNOSTICS_INTERVAL" =~ ^[0-9]+$ ]] || { echo "Invalid gradient diagnostic interval" >&2; exit 2; }
if (( REFRESH_SOURCE_PROVENANCE == 1 && RESTART_INCOMPLETE != 1 )); then
  echo "--refresh-source-provenance-after-failed-run requires --restart-incomplete-stage" >&2
  exit 2
fi
if [[ "$METHOD_VERSION" == "1.1" ]]; then LAMBDA_KEY=0; fi
[[ "$START_STAGE" =~ ^[1-5]$ && "$END_STAGE" =~ ^[1-5]$ && "$START_STAGE" -le "$END_STAGE" ]] || { echo "Invalid stage range" >&2; exit 2; }
[[ "$PRIVATE_EXPERTS_PER_TASK" =~ ^[1-9][0-9]*$ ]] || { echo "--private-experts-per-task must be positive" >&2; exit 2; }
"$PYTHON" - "$SHARED_LR" "$PRIVATE_LR" "$LAMBDA_DRIFT" "$LAMBDA_KEY" <<'PY'
import math, sys
values = list(map(float, sys.argv[1:]))
raise SystemExit(0 if all(math.isfinite(v) and v >= 0 for v in values) and values[0] > 0 and values[1] > 0 else 2)
PY
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NPROC_PER_NODE=${#GPU_IDS[@]}
(( NPROC_PER_NODE >= 1 && NPROC_PER_NODE <= 2 )) || { echo "This formal script supports one or two GPUs" >&2; exit 2; }
(( 16 % NPROC_PER_NODE == 0 )) || { echo "Global batch 16 is not divisible by GPU count" >&2; exit 2; }
GRADIENT_ACCUMULATION_STEPS=$((16 / NPROC_PER_NODE))

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub
export HF_DATASETS_CACHE=/remote-home/wangbomin/huggingface_cache/datasets
export XDG_CACHE_HOME=/remote-home/wangbomin/huggingface_cache/xdg
export TMPDIR=/remote-home/wangbomin/huggingface_cache/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPUS"
export NPROC_PER_NODE
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset TRANSFORMERS_CACHE
mkdir -p "$TMPDIR"
cd "$ROOT"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES; DDP processes: $NPROC_PER_NODE; gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Med-PRISM v$METHOD_VERSION: shared_lr=$SHARED_LR private_lr=$PRIVATE_LR drift=$LAMBDA_DRIFT orth=$ORTH_LAMBDA key=$LAMBDA_KEY"
PREP_ARGS=(
  --method-version "$METHOD_VERSION"
  --data-root "$DATA_ROOT"
  --output-root "$OUTPUT_ROOT"
  --seed "$SEED"
  --shared-lr "$SHARED_LR"
  --private-lr "$PRIVATE_LR"
  --orth-lambda "$ORTH_LAMBDA"
  --lambda-drift "$LAMBDA_DRIFT"
  --lambda-key "$LAMBDA_KEY"
  --gradient-diagnostics-interval "$GRADIENT_DIAGNOSTICS_INTERVAL"
)
[[ -n "$ABLATION_NAME" ]] && PREP_ARGS+=(--ablation-name "$ABLATION_NAME")
(( PILOT_MODE == 1 )) && PREP_ARGS+=(--pilot-mode)
(( REFRESH_SOURCE_PROVENANCE == 1 )) && PREP_ARGS+=(--refresh-source-provenance-after-failed-run)
(( CHECK_ONLY == 1 )) && PREP_ARGS+=(--check-only)
"$PYTHON" scripts/medicalskill_v1_2_med_prism_versions/prepare_versioned.py "${PREP_ARGS[@]}"
if (( CHECK_ONLY == 1 )); then
  echo "Versioned contract check passed; no training launched."
  exit 0
fi

for STAGE in $(seq "$START_STAGE" "$END_STAGE"); do
  PADDED=$(printf '%02d' "$STAGE")
  TASK_DIR=$(printf 'task_%02d' "$STAGE")
  case "$STAGE" in
    1) TASK_NAME=task_01_vqa ;;
    2) TASK_NAME=task_02_diagnosis_classification ;;
    3) TASK_NAME=task_03_concept_recognition ;;
    4) TASK_NAME=task_04_visual_grounding ;;
    5) TASK_NAME=task_05_reasoning_vqa ;;
  esac
  MAX_LENGTH=1024
  if (( STAGE == 5 )); then
    # Reasoning VQA contains up to eight images plus long rationales. Keeping
    # 1024 causes the multimodal collator to delete nearly every candidate.
    MAX_LENGTH=4096
  fi
  STAGE_ROOT="$OUTPUT_ROOT/med_prism/stage_$PADDED"
  TRAIN_ROOT="$STAGE_ROOT/train"
  CONFIG="$OUTPUT_ROOT/configs/stage_$PADDED.json"
  DATASET="$DATA_ROOT/$TASK_NAME/train.jsonl"
  COMPLETION="$STAGE_ROOT/completion.json"

  if [[ -f "$COMPLETION" ]] && "$PYTHON" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="PASS" else 1)' "$COMPLETION"; then
    echo "[stage $STAGE] already PASS; skipping"
    continue
  fi
  if (( STAGE > 1 )); then
    PREVIOUS=$(printf '%02d' $((STAGE - 1)))
    PREVIOUS_ROOT="$OUTPUT_ROOT/med_prism/stage_$PREVIOUS"
    test -s "$PREVIOUS_ROOT/completion.json"
    test -s "$PREVIOUS_ROOT/shared/shared_manifest.json"
    "$PYTHON" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="PASS" else 1)' "$PREVIOUS_ROOT/completion.json"
  fi
  if [[ -d "$TRAIN_ROOT" ]] && find "$TRAIN_ROOT" -mindepth 1 -print -quit | grep -q .; then
    if (( RESTART_INCOMPLETE != 1 )); then
      echo "[stage $STAGE] incomplete training directory exists: $TRAIN_ROOT" >&2
      echo "Rerun with --restart-incomplete-stage to archive this failed attempt." >&2
      exit 3
    fi
    FAILED_ROOT="$OUTPUT_ROOT/failed_attempts"
    ARCHIVE="$FAILED_ROOT/stage_${PADDED}_$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$FAILED_ROOT"
    mv "$STAGE_ROOT" "$ARCHIVE"
    printf '%s\n' "$ARCHIVE" >> "$FAILED_ROOT/archive_index.txt"
    echo "[stage $STAGE] archived incomplete attempt to $ARCHIVE"
  fi

  mkdir -p "$TRAIN_ROOT" "$STAGE_ROOT/runtime_audits"
  export MED_PRISM_SHARED_PRIVATE_CONFIG="$CONFIG"
  export MED_PRISM_RELOAD_PROBE=1
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader > "$STAGE_ROOT/gpu_before_training.txt"

  CMD=(
    "$SWIFT" sft
    --model Qwen/Qwen3-VL-8B-Instruct
    --model_revision "$REVISION"
    --template qwen3_vl
    --dataset "$DATASET"
    --tuner_type med_prism_shared_private
    --freeze_llm false
    --freeze_vit true
    --freeze_aligner true
    --torch_dtype bfloat16
    --attn_impl sdpa
    --max_length "$MAX_LENGTH"
    --max_pixels 200704
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$PRIVATE_LR"
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
    --external_plugins
    "$ROOT/scripts/phase3/callback_compat.py"
    "$ROOT/med_prism/swift_plugins/med_prism_rank1_plugin.py"
    "$ROOT/med_prism/swift_plugins/med_prism_shared_private_plugin.py"
    "$ROOT/scripts/med_prism_real_5skill/reload_probe_plugin.py"
  )
  printf '%q ' "${CMD[@]}" > "$STAGE_ROOT/train_command.txt"
  printf '\n' >> "$STAGE_ROOT/train_command.txt"
  echo "[stage $STAGE] training $TASK_NAME with max_length=$MAX_LENGTH"
  set +e
  "${CMD[@]}" 2>&1 | tee "$STAGE_ROOT/train.log"
  CODE=${PIPESTATUS[0]}
  set -e
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader > "$STAGE_ROOT/gpu_after_training.txt"
  if (( CODE != 0 )); then
    echo "[stage $STAGE] swift sft failed with exit code $CODE" >&2
    exit "$CODE"
  fi

  CHECKPOINT=$(find "$TRAIN_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)
  "$PYTHON" - "$STAGE_ROOT" "$STAGE" "$CHECKPOINT" "$REVISION" "$SEED" "$METHOD_VERSION" <<'PY'
import hashlib, json, os, sys, tempfile
from pathlib import Path
root, stage = Path(sys.argv[1]), int(sys.argv[2])
checkpoint, revision, seed = sys.argv[3], sys.argv[4], int(sys.argv[5])
method_version = sys.argv[6]
def load(path): return json.loads(path.read_text(encoding="utf-8"))
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()
shared_path=root/"shared/shared_manifest.json"
private_path=root/"private/private_manifest.json"
shared, private=load(shared_path), load(private_path)
config=load(root.parents[1]/"configs"/f"stage_{stage:02d}.json")
checks={
    "checkpoint_exists": bool(checkpoint) and Path(checkpoint).is_dir(),
    "shared_manifest": shared.get("method")=="med_prism_shared_private" and shared.get("task_id")==stage,
    "private_manifest": private.get("method")=="med_prism_shared_private" and private.get("task_id")==stage,
    "shared_hash": sha(shared_path.parent/shared["weights_file"])==shared["weights_sha256"],
    "private_hash": sha(private_path.parent/private["weights_file"])==private["weights_sha256"],
    "target_module_audit": load(root/"target_module_audit.json").get("status")=="PASS",
    "optimizer_audit": load(root/"runtime_audits"/f"task{stage}_optimizer_audit.json").get("status")=="PASS",
    "parameter_audit": load(root/"runtime_audits"/f"task{stage}_parameter_audit.json").get("status")=="PASS",
    "shared_hash_audit": load(root/"runtime_audits/shared_hash_audit.json").get("status")=="PASS",
    "private_hash_audit": load(root/"runtime_audits/private_hash_audit.json").get("status")=="PASS",
    "orth_gradient_audit": load(root/"runtime_audits/orth_gradient_audit.json").get("status")=="PASS",
    "shared_drift_audit": load(root/"runtime_audits/shared_drift_gradient_audit.json").get("status")=="PASS",
    "key_isolation_audit": load(root/"runtime_audits/key_isolation_audit.json").get("status")=="PASS",
}

payload={"status":"PASS" if all(checks.values()) else "FAIL","stage":stage,"method":"Med-PRISM","method_version":method_version,"method_variant":config["method_variant"],"ablation_name":config.get("ablation_name"),"enabled_components":{"geometric_rank1_orthogonality":config["orth_lambda"]>0,"effective_BA_shared_drift":config["shared_drift_lambda"]>0,"rank1_key_isolation":config["key_isolation_enabled"],"explicit_rank1_private_experts":config["private_adapter_type"]=="rank1_expert_bank","conventional_rank16_task_private_lora":config["private_adapter_type"]=="standard_rank16_lora"},"private_adapter_type":config["private_adapter_type"],"private_rank":config["private_rank"],"private_units_per_task":config["private_experts_per_task"],"tuner_type":"med_prism_shared_private","ordinary_lora":False,"model_revision":revision,"seed":seed,"checkpoint":checkpoint or None,"checks":checks,"shared_weights_sha256":shared["weights_sha256"],"private_weights_sha256":private["weights_sha256"]}
fd,tmp=tempfile.mkstemp(prefix=".completion.",dir=root)
with os.fdopen(fd,"w",encoding="utf-8") as f:
    json.dump(payload,f,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,root/"completion.json")
if payload["status"] != "PASS": raise SystemExit(json.dumps(payload,indent=2))
print(json.dumps(payload,indent=2))
PY
  echo "[stage $STAGE] PASS"
done

echo "Training stages $START_STAGE..$END_STAGE completed. Output: $OUTPUT_ROOT"
