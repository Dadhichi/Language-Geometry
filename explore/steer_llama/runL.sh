#!/usr/bin/env bash
# Colab VM (L4). Needs /root/.hf_token (Llama-3.1 gate accepted), /root/steer_dirsL.npz. Output /content/l_out.tar
set -euo pipefail
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token)
export PYTHONUNBUFFERED=1
cd /content/Language-Geometry
git rev-parse HEAD > /content/jobL_commit.txt
pip -q install stanza langid
python - <<'PY'
import stanza
for l in ["de", "nl", "ru", "uk", "pl", "hr", "en", "es", "ko"]:
    stanza.download(l, verbose=False)
print("stanza models ready")
PY
echo "== debug $(date)"
python explore/steer_llama/gen_llama.py --prompts explore/steer_gen2/prompts2.jsonl --dirs /root/steer_dirsL.npz \
    --out /content/l_debug --limit 2 --n_rand 1
echo "DEBUG ok"
echo "== gen $(date)"
python explore/steer_llama/gen_llama.py --prompts explore/steer_gen2/prompts2.jsonl --dirs /root/steer_dirsL.npz --out /content/l_out
cp /content/jobL_commit.txt /content/l_out/
tar -cf /content/l_out.tar -C /content l_out
echo "JOBL_DONE $(date)"
