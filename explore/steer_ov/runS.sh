#!/usr/bin/env bash
# Colab VM (L4): OV steering run. Needs /root/.hf_token and /root/steer_dirs.npz. Output /content/steer_out.tar
set -euo pipefail
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token)
export PYTHONUNBUFFERED=1
cd /content/Language-Geometry
git rev-parse HEAD > /content/jobS_commit.txt
echo "== steer $(date)"
python explore/steer_ov/steer_score.py --pairs explore/steer_ov/pairs.jsonl --dirs /root/steer_dirs.npz \
    --out /content/steer_out --model Qwen/Qwen2.5-7B
cp /content/jobS_commit.txt /content/steer_out/
tar -cf /content/steer_out.tar -C /content steer_out
echo "JOBS_DONE $(date)"
