#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/medagentcl_v4/bin/python}"
PIP_BIN="${PIP_BIN:-/root/anaconda3/envs/medagentcl_v4/bin/pip}"

test -d "${REPO_ROOT}/swift"
test -x "${PYTHON_BIN}"
test -x "${PIP_BIN}"

"${PIP_BIN}" install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

"${PYTHON_BIN}" - <<'PY'
import torch
import torchaudio
import torchvision

assert torch.__version__ == '2.5.1+cu124', torch.__version__
assert torch.version.cuda == '12.4', torch.version.cuda
assert torchvision.__version__ == '0.20.1+cu124', torchvision.__version__
assert torchaudio.__version__ == '2.5.1+cu124', torchaudio.__version__
print('PyTorch CUDA 12.4 stack verified.')
PY
