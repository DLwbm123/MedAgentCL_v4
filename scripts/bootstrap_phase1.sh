#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/root/anaconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-medagentcl_v4}"
ENV_PREFIX="${ENV_PREFIX:-/root/anaconda3/envs/medagentcl_v4}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
PIP_BIN="${PIP_BIN:-${ENV_PREFIX}/bin/pip}"
UPSTREAM_COMMIT="98a09c18cdf95ff07051324b9b8cc90f5184b24b"

test -x "${CONDA_BIN}"
test -d "${REPO_ROOT}/swift"
test "$(git -C "${REPO_ROOT}" rev-parse v4.4.1^{commit})" = "${UPSTREAM_COMMIT}"
git -C "${REPO_ROOT}" merge-base --is-ancestor "${UPSTREAM_COMMIT}" HEAD

if "${CONDA_BIN}" env list --json | grep -Fq "${ENV_PREFIX}"; then
  echo "Refusing to overwrite existing Conda environment: ${ENV_PREFIX}" >&2
  exit 2
fi

"${CONDA_BIN}" create -y -n "${ENV_NAME}" -c conda-forge -c defaults \
  python=3.12.7 pip \
  'decord=0.6.0=np2py312h48de876_2' \
  'numpy=2.5.1=py312h33ff503_0'

PYTHON_BIN="${PYTHON_BIN}" PIP_BIN="${PIP_BIN}" \
  "${SCRIPT_DIR}/install_pytorch_cu124.sh"

"${PIP_BIN}" install --no-cache-dir --index-url https://pypi.org/simple \
  -r "${REPO_ROOT}/requirements-bootstrap.txt"

"${PIP_BIN}" install --no-deps --no-build-isolation -e "${REPO_ROOT}"
"${PIP_BIN}" check

echo "Phase 1 environment bootstrap complete: ${ENV_PREFIX}"
