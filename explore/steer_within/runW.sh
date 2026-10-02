#!/usr/bin/env bash
# Colab VM (L4). Needs /root/.hf_token, /root/steer_dirs2.npz. Output /content/w_out.tar
set -euo pipefail
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token)
export PYTHONUNBUFFERED=1
cd /content/Language-Geometry
git rev-parse HEAD > /content/jobW_commit.txt
pip -q install stanza langid
python - <<'PY'
import stanza
for l in ["de", "nl", "ru", "uk", "pl", "hr", "en", "es", "ko"]:
    stanza.download(l, verbose=False)
print("stanza models ready")
PY
echo "== debug $(date)"
python explore/steer_within/pairstates.py --pairs explore/steer_ov/pairs.jsonl --dirs2 /root/steer_dirs2.npz --out /content/w_debug --limit 3
python explore/steer_within/gen_w.py --prompts explore/steer_gen2/prompts2.jsonl --dirs /content/w_debug/steer_dirsW.npz --out /content/w_debug/gen --limit 2
echo "DEBUG ok"
echo "== pairs $(date)"
python explore/steer_within/pairstates.py --pairs explore/steer_ov/pairs.jsonl --dirs2 /root/steer_dirs2.npz --out /content/w_out
echo "== gen $(date)"
python explore/steer_within/gen_w.py --prompts explore/steer_gen2/prompts2.jsonl --dirs /content/w_out/steer_dirsW.npz --out /content/w_out/gen
cp /content/jobW_commit.txt /content/w_out/
tar -cf /content/w_out.tar -C /content w_out
echo "JOBW_DONE $(date)"
