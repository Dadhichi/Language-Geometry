#!/usr/bin/env bash
# Runs ON the Colab VM (L4). Launched in the background by the Colab-CLI driver; log -> /content/job34.log.
# Needs /root/.hf_token and /root/covgamma_f32.npy uploaded first. Output: /content/derived34.tar
set -euo pipefail
export HF_TOKEN=$(tr -d '[:space:]' < /root/.hf_token)
cd /content/Language-Geometry
git rev-parse HEAD > /content/job34_commit.txt
pip -q install -r requirements.txt
LANGS=eng_Latn,deu_Latn,nld_Latn,swe_Latn,fra_Latn,spa_Latn,por_Latn,rus_Cyrl,ukr_Cyrl,pol_Latn,hrv_Latn,srp_Cyrl,\
hin_Deva,urd_Arab,mar_Deva,pes_Arab,arb_Arab,heb_Hebr,mlt_Latn,tur_Latn,azj_Latn,kaz_Cyrl,fin_Latn,ekk_Latn,\
cmn_Hans,cmn_Hant,jpn_Jpan,kor_Hang,ind_Latn,fil_Latn,vie_Latn,khm_Khmr,tam_Taml,tel_Telu
echo "== extract $(date)"
python extract.py --model Qwen/Qwen2.5-7B --tag q34 --out /content/out --tmp /content/out \
    --langs "$LANGS" --splits dev,devtest --flores hf
echo "== derive $(date)"
python explore/prereg34/derive34.py /content/out/q34 /content/derived34 /root/covgamma_f32.npy --B 50
cp /content/job34_commit.txt /content/derived34/
cp /content/out/q34/meta.json /content/derived34/extract_meta.json
for s in dev devtest; do cp /content/out/q34/ntok_$s.npy /content/out/q34/sentences_$s.json /content/derived34/; done
tar -cf /content/derived34.tar -C /content derived34
echo "JOB34_DONE $(date)"
