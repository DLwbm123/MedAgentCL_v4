#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/MedAgentCL_v4
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/root/MedAgentCL_v4${PYTHONPATH:+:$PYTHONPATH}
exec /root/anaconda3/envs/medagentcl_v4/bin/python -u -m med_prism.rgpa.run "$@"
