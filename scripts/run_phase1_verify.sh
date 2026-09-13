#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/medagentcl_v4/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/phase1}"

test -x "${PYTHON_BIN}"
mkdir -p "${OUTPUT_DIR}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_phase1.py" \
  --repo-root "${REPO_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/phase1_verification.log"
