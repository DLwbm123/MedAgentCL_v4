#!/usr/bin/env bash
set -euo pipefail

umask 077
SECRETS_FILE=/root/.config/medicalskill-cl/secrets/kimi.env
PYTHON=/root/anaconda3/envs/medagentcl_v4/bin/python
SCRIPT=/root/MedAgentCL_v4/scripts/medicalskill_kimi_k3/run_kimi_k3_pilot.py

if [[ ! -r "$SECRETS_FILE" ]]; then
  printf 'ERROR: Kimi credential file is not readable.\n' >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$SECRETS_FILE"
set +a

if [[ -n "${KIMI_API_KEY:-}" && -n "${MOONSHOT_API_KEY:-}" && "$KIMI_API_KEY" != "$MOONSHOT_API_KEY" ]]; then
  printf 'ERROR: ambiguous_credentials\n' >&2
  exit 2
fi

if [[ -z "${KIMI_API_KEY:-${MOONSHOT_API_KEY:-}}" ]]; then
  printf 'ERROR: Kimi credential is empty.\n' >&2
  exit 2
fi

exec "$PYTHON" "$SCRIPT" "$@"
