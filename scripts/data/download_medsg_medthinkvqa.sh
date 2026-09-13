#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT="/root/MedAgentCL_v4"
readonly PYTHON="/root/anaconda3/envs/medagentcl_v4/bin/python"
readonly SCRIPT="${ROOT}/scripts/data/download_medsg_medthinkvqa.py"
readonly CACHE_ROOT="/remote-home/wangbomin/huggingface_cache"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CACHE_ROOT}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

mkdir -p \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}"

cd "${ROOT}"
exec "${PYTHON}" "${SCRIPT}" "$@"
