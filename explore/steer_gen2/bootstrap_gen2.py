"""DESCRIPTIVE (not pre-registered): prompt-bootstrap 95% interval of the pooled word-order slope for H1 and H2.
Resamples the 150 sentences with replacement (the same draw for every language and condition), recomputes the
noun-object shares and the pooled slope with the pre-registered inclusion rule (analysis_gen2.py).
usage: python bootstrap_gen2.py OUT_DIR [--B 2000]   -> bootstrap_gen2.json next to this script"""
import argparse, json, os
import numpy as np
from analysis_gen2 import K, MIN_PAIRS, SETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("D")
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    conds = json.load(open(os.path.join(args.D, "conditions.json")))
    find = lambda kind, k: next(i for i, c in enumerate(conds) if c["kind"] == kind and abs(c["k"] - k) < 1e-9 and c["r"] == -1)
    cis = {"0": 0, "+": find("ov", K), "-": find("ov", -K)}
    inv = {v: k for k, v in cis.items()}
    sents = sorted({json.loads(l)["sent"] for l in open(os.path.join(args.D, "gen.jsonl"), encoding="utf-8")})
    si = {s: i for i, s in enumerate(sents)}
    langs = sorted({l for L in SETS.values() for l in L})
    C = {(l, c): np.zeros((len(sents), 2)) for l in langs for c in cis}           # [sentence, (ov, vo)] noun counts
    for line in open(os.path.join(args.D, "gen.jsonl"), encoding="utf-8"):
        g = json.loads(line)
        if g["cond"] in inv and g["lang"] in langs and g["match"]:
            C[(g["lang"], inv[g["cond"]])][si[g["sent"]]] += (g["nom_ov"], g["nom_vo"])

    def pooled(L, w):
        b = []
        for l in L:
            t = {c: w @ C[(l, c)] for c in cis}
            if min(v.sum() for v in t.values()) < MIN_PAIRS:
                continue
            b.append((t["+"][0] / t["+"].sum() - t["-"][0] / t["-"].sum()) / (2 * K))
        return np.mean(b) if b else np.nan
    rng = np.random.default_rng(0)
    out = {}
    for H, L in SETS.items():
        bs = [pooled(L, np.bincount(rng.integers(0, len(sents), len(sents)), minlength=len(sents)).astype(float))
              for _ in range(args.B)]
        out[H] = [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))]
        print(f"{H}: point {pooled(L, np.ones(len(sents))):+.4f} | 95% prompt bootstrap [{out[H][0]:+.4f}, {out[H][1]:+.4f}]")
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bootstrap_gen2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
