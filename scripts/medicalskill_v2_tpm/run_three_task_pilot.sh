#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$HF_HOME/xdg"
export TMPDIR="$HF_HOME/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${TPM_GPU:-0}"
unset TRANSFORMERS_CACHE
cd "$ROOT"
exec /root/anaconda3/envs/medagentcl_v4/bin/python -u -m med_prism.transport.pilot "$@"
