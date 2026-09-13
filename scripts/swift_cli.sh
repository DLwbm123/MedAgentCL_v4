#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SWIFT_BIN="${SWIFT_BIN:-/root/anaconda3/envs/medagentcl_v4/bin/swift}"

test -d "${REPO_ROOT}/swift"
test -x "${SWIFT_BIN}"

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
usage: swift <command> [options]

available commands:
  pt
  sft
  infer
  merge-lora
  web-ui
  deploy
  rollout
  rlhf
  sample
  export
  eval
  app

Run 'swift <command> --help' for command-specific options.
EOF
  exit 0
fi

exec "${SWIFT_BIN}" "$@"
