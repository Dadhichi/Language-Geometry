#!/usr/bin/env bash
# Runs ON the Colab VM after run34.sh (same session; model already cached). Log -> /content/jobT.log.
# Needs /root/.hf_token. Output: /content/tok_out.tar
set -euo pipefail
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token)
export PYTHONUNBUFFERED=1
cd /content/Language-Geometry
git pull -q
git rev-parse HEAD > /content/jobT_commit.txt
echo "== extract_tokens $(date)"
python extract_tokens.py --model Qwen/Qwen2.5-7B --out /content/tok_out --span explore/prereg_belief/span12.npz \
    --split devtest --flores hf --batch_tokens 8192
cp /content/jobT_commit.txt /content/tok_out/
tar -cf /content/tok_out.tar -C /content tok_out
echo "JOBT_DONE $(date)"
