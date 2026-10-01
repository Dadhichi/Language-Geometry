#!/usr/bin/env bash
# Colab VM (L4). usage: runG.sh [debug]. Needs /root/.hf_token, /root/steer_dirs.npz. Output /content/gen_out.tar
set -euo pipefail
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token)
export PYTHONUNBUFFERED=1
cd /content/Language-Geometry
git rev-parse HEAD > /content/jobG_commit.txt
pip -q install stanza langid
python - <<'PY'
import stanza
for l in ["en", "de", "fr", "ru", "hi", "tr", "ja", "zh-hans"]:
    stanza.download(l, verbose=False)          # default package (includes mwt only where the language has one)
print("stanza models ready")
PY
if [ "${1:-}" = "debug" ]; then
  python explore/steer_gen/gen_ov.py --prompts explore/steer_gen/prompts.jsonl --dirs /root/steer_dirs.npz \
      --out /content/gen_debug --limit 3 --n_rand 1
  echo "JOBG_DEBUG_DONE $(date)"
  exit 0
fi
echo "== gen $(date)"
python explore/steer_gen/gen_ov.py --prompts explore/steer_gen/prompts.jsonl --dirs /root/steer_dirs.npz --out /content/gen_out
cp /content/jobG_commit.txt /content/gen_out/
tar -cf /content/gen_out.tar -C /content gen_out
echo "JOBG_DONE $(date)"
