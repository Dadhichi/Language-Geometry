"""EXPLORATORY: published per-language MGSM accuracy of the BASE models against our mid-depth geometry.
Accuracies: EMMA-500 Gen 2 (Ji et al., arXiv 2506.00469), Table 26 (MGSM, 3-shot CoT, lm-eval-harness, flexible match),
as transcribed by the literature check for PROPOSAL.md. Llama-3.1-8B ja/zh (2.8, 1.6) look like answer-extraction failures
and are excluded. Common Crawl shares: CC-MAIN-2023-50 (commoncrawl.github.io/cc-crawl-statistics). n is tiny
(7 and 5 languages): with 7 languages only |rho| > ~0.85 would be detectable, so this is a sanity look, not a test."""
import json, os
import numpy as np
from scipy.stats import spearmanr
o = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "offsets34.json")))
IDX = {"deu": 1, "fra": 4, "spa": 5, "rus": 7, "jpn": 26, "zho": 24, "tel": 33}
ACC = {"qwen": {"eng": 81.6, "deu": 69.6, "fra": 68.0, "spa": 73.6, "rus": 74.8, "jpn": 60.0, "zho": 72.8, "tel": 9.2},
       "llama": {"eng": 52.8, "deu": 42.0, "fra": 34.8, "spa": 42.8, "rus": 37.2, "tel": 5.6}}
CC = {"deu": 5.4499, "fra": 4.3933, "spa": 4.5391, "rus": 6.0303, "jpn": 5.1508, "zho": 5.0798, "tel": 0.0161}
for m in ACC:
    langs = [l for l in ACC[m] if l != "eng"]
    gap = np.array([np.log(ACC[m]["eng"] / (100 - ACC[m]["eng"])) - np.log(ACC[m][l] / (100 - ACC[m][l])) for l in langs])
    fert = np.array([o[m]["fert"][IDX[l]] for l in langs])
    cc = np.array([CC[l] for l in langs])
    print(f"{m}: languages {langs}; logit gap {np.round(gap, 2)}")
    print(f"   spearman(gap, log fert) {spearmanr(gap, np.log(fert))[0]:+.2f}; spearman(gap, -log CC share) {spearmanr(gap, -np.log(cc))[0]:+.2f}")
    for geom in ("causal", "lda05"):
        d2 = np.array([o[m][geom]["window"]["d2en"][IDX[l]] for l in langs])
        print(f"   {geom}: spearman(gap, d2en window) {spearmanr(gap, d2)[0]:+.2f}; without Telugu {spearmanr(gap[:-1], d2[:-1])[0]:+.2f}")
