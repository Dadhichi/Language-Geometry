#!/usr/bin/env bash
# Colab VM (L4). Needs /root/.hf_token (Llama-3.1 gate accepted). Output /content/wl_out.tar
set -euo pipefail
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token)
export PYTHONUNBUFFERED=1
cd /content/Language-Geometry
git rev-parse HEAD > /content/jobWL_commit.txt
echo "== debug $(date)"
python explore/steer_within_llama/pairstates_l.py --pairs explore/steer_ov/pairs.jsonl --out /content/wl_debug --limit 3
echo "DEBUG ok"
echo "== pairs $(date)"
python explore/steer_within_llama/pairstates_l.py --pairs explore/steer_ov/pairs.jsonl --out /content/wl_out
cp /content/jobWL_commit.txt /content/wl_out/
tar -cf /content/wl_out.tar -C /content wl_out
echo "JOBWL_DONE $(date)"
