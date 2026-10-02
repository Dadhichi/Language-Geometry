#!/usr/bin/env bash
# Colab VM (L4). Needs /root/.hf_token, /root/steer_dirs2.npz. Output /content/gen2_out.tar
set -euo pipefail
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token)
export PYTHONUNBUFFERED=1
cd /content/Language-Geometry
git rev-parse HEAD > /content/jobG2_commit.txt
pip -q install stanza langid
python - <<'PY'
import stanza
for l in ["de", "nl", "ru", "uk", "pl", "hr", "en", "es", "ko"]:
    stanza.download(l, verbose=False)          # default package (includes mwt only where the language has one)
print("stanza models ready")
PY
echo "== debug $(date)"
python explore/steer_gen2/gen_ov2.py --prompts explore/steer_gen2/prompts2.jsonl --dirs /root/steer_dirs2.npz \
    --out /content/gen2_debug --limit 2 --n_rand 1
python - <<'PY'
import json
G = [json.loads(l) for l in open("/content/gen2_debug/gen.jsonl", encoding="utf-8")]
print("DEBUG rows", len(G), "| matched", sum(g["match"] for g in G), "| noun pairs", sum(g["nom_ov"] + g["nom_vo"] for g in G))
PY
echo "== gen $(date)"
python explore/steer_gen2/gen_ov2.py --prompts explore/steer_gen2/prompts2.jsonl --dirs /root/steer_dirs2.npz --out /content/gen2_out
cp /content/jobG2_commit.txt /content/gen2_out/
tar -cf /content/gen2_out.tar -C /content gen2_out
echo "JOBG2_DONE $(date)"
