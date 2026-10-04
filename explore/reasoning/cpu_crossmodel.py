"""EXPLORATORY: does the distance of a language from English at mid-depth follow its tokenizer fertility ACROSS models?
Within a language, Qwen2.5-7B and Llama-3.1-8B differ in fertility (e.g. Hebrew 39.6 vs 96.2 tokens/sentence, Hindi
118.8 vs 66.7). If mid-depth distance to English is driven by tokenization rather than by the language itself, the
model difference in relative d2en should track the model difference in log fertility. Reads offsets34.json (run
cpu_offsets34.py first). Pre-registered window, relative units (d2en / mean pair D2). Nothing pre-registered."""
import json, os
import numpy as np
from scipy.stats import spearmanr, pearsonr
o = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "offsets34.json")))
L = ["eng","deu","nld","swe","fra","spa","por","rus","ukr","pol","hrv","srp","hin","urd","mar","pes","arb","heb","mlt",
     "tur","azj","kaz","fin","ekk","zhs","zht","jpn","kor","ind","fil","vie","khm","tam","tel"]
dlf = np.log(o["qwen"]["fert"]) - np.log(o["llama"]["fert"])
rng = np.random.default_rng(0)
for geom in ("causal", "lda05", "euc"):
    q = np.array(o["qwen"][geom]["window"]["d2en"]); l = np.array(o["llama"][geom]["window"]["d2en"])
    dd = np.log(q[1:]) - np.log(l[1:])
    rho = spearmanr(dd, dlf[1:])[0]; r = pearsonr(dd, dlf[1:])[0]
    null = [spearmanr(dd, rng.permutation(dlf[1:]))[0] for _ in range(10000)]
    p = (1 + sum(abs(x) >= abs(rho) for x in null)) / 10001
    print(f"{geom}: across 33 languages, spearman(dlog d2en, dlog fert) = {rho:+.2f} (perm p {p:.4f}), pearson {r:+.2f}")
    top = np.argsort(-np.abs(dlf[1:]))[:6]
    print("   largest fertility differences:", " ".join(f"{L[i+1]}(dlogf {dlf[i+1]:+.2f}, dlog d2en {dd[i]:+.2f})" for i in top))
