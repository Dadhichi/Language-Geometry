"""EXPLORATORY: does a language's web-data share predict its mid-depth distance from English beyond tokenizer fertility?
Common Crawl CC-MAIN-2023-50 primary-language shares (CLD2; commoncrawl.github.io/cc-crawl-statistics/plots/languages),
as transcribed by the literature check for PROPOSAL.md. One Chinese code (zho) serves both scripts; fil uses tgl;
pes uses fas. Partial Spearman = Spearman of rank residuals after regressing ranks on log fertility. Run
cpu_offsets34.py first. Nothing pre-registered; CC share is a proxy, not the models' real data mix."""
import json, os
import numpy as np
from scipy.stats import spearmanr, rankdata
o = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "offsets34.json")))
CC = [44.4285, 5.4499, 1.9746, 0.6637, 4.3933, 4.5391, 1.7631, 6.0303, 0.3845, 1.7490, 0.2014, 0.2206, 0.1749, 0.0265,
      0.0212, 0.6795, 0.5867, 0.2091, 0.0042, 0.9804, 0.0528, 0.0221, 0.3680, 0.1339, 5.0798, 5.0798, 5.1508, 0.7288,
      0.8596, 0.0093, 1.0291, 0.0097, 0.0423, 0.0161]


def partial_spearman(y, x, z):
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    A = np.column_stack([np.ones(len(rz)), rz])
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    return np.corrcoef(ey, ex)[0, 1]


lcc = -np.log(np.array(CC))
for m in ("qwen", "llama"):
    lf = np.log(o[m]["fert"])
    print(f"{m}: spearman(log fert, -log CC) over 33 = {spearmanr(lf[1:], lcc[1:])[0]:+.2f}")
    for geom in ("causal", "lda05"):
        d = np.array(o[m][geom]["window"]["d2en"])
        print(f"   {geom}: spearman(d2en, -log CC) {spearmanr(d[1:], lcc[1:])[0]:+.2f}; partial on log fert "
              f"{partial_spearman(d[1:], lcc[1:], lf[1:]):+.2f}; partial of log fert on -log CC {partial_spearman(d[1:], lf[1:], lcc[1:]):+.2f}")
