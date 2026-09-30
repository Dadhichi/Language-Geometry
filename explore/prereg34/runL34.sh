#!/usr/bin/env bash
# Llama-3.1-8B replication of run34.sh (Colab VM, L4). Log -> /content/jobL34.log. Output /content/derivedL34.tar
set -euo pipefail
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token)
export PYTHONUNBUFFERED=1
cd /content/Language-Geometry
git pull -q || true
git rev-parse HEAD > /content/jobL34_commit.txt
pip -q install -r requirements.txt
M=meta-llama/Llama-3.1-8B
LANGS=eng_Latn,deu_Latn,nld_Latn,swe_Latn,fra_Latn,spa_Latn,por_Latn,rus_Cyrl,ukr_Cyrl,pol_Latn,hrv_Latn,srp_Cyrl,\
hin_Deva,urd_Arab,mar_Deva,pes_Arab,arb_Arab,heb_Hebr,mlt_Latn,tur_Latn,azj_Latn,kaz_Cyrl,fin_Latn,ekk_Latn,\
cmn_Hans,cmn_Hant,jpn_Jpan,kor_Hang,ind_Latn,fil_Latn,vie_Latn,khm_Khmr,tam_Taml,tel_Telu
echo "== extract $(date)"
python extract.py --model $M --tag l34 --out /content/outL --tmp /content/outL --langs "$LANGS" --splits dev,devtest --flores hf
echo "== covgamma $(date)"
python explore/prereg34/covgamma_hf.py $M 128000 /content/covgamma_llama_f32.npy
echo "== derive $(date)"
python explore/prereg34/derive34.py /content/outL/l34 /content/derivedL34 /content/covgamma_llama_f32.npy --B 50
cp /content/jobL34_commit.txt /content/derivedL34/
cp /content/outL/l34/meta.json /content/derivedL34/extract_meta.json
for s in dev devtest; do cp /content/outL/l34/ntok_$s.npy /content/outL/l34/sentences_$s.json /content/derivedL34/; done
tar -cf /content/derivedL34.tar -C /content derivedL34
echo "JOBL34_DONE $(date)"
