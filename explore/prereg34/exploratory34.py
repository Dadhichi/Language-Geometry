"""EXPLORATORY (not pre-registered; written after seeing results34.json). Robustness of the primary claims to:
(1) geography: great-circle distance between WALS coordinates added to the base;
(2) leave-one-family-out: drop each family (and each OV family) and redo H1 / OV|H1 on the remaining languages;
(3) OV|H1 with the Glottolog splits AND geography in the base.
Primary Gram (L8-L20, causal + lda05). usage: python exploratory34.py DATA_DIR [--n_perm 2000]"""
import argparse, json, os
import numpy as np
import lib34 as T
import analysis34 as A

# WALS coordinates (lat, lon) from cldf-datasets/wals languages.csv (explore/prereg34 lookup, 2026-09-30)
WALS = {"eng": (52.0, 0.0), "deu": (52.0, 10.0), "nld": (52.5, 6.0), "swe": (60.0, 15.0), "fra": (48.0, 2.0),
        "spa": (40.0, -4.0), "por": (39.0, -8.0), "rus": (56.0, 38.0), "ukr": (49.0, 33.0), "pol": (52.0, 20.0),
        "hrv": (44.0, 19.0), "srp": (44.0, 19.0), "hin": (25.0, 77.0), "urd": (25.0, 67.0), "mar": (19.0, 76.0),
        "pes": (32.0, 54.0), "arb": (25.0, 42.0), "heb": (31.5, 34.8333333333), "mlt": (35.9166666667, 14.4166666667),
        "tur": (39.0, 35.0), "azj": (40.5, 48.5), "kaz": (50.0, 70.0), "fin": (62.0, 25.0), "ekk": (59.0, 26.0),
        "cmn_Hans": (34.0, 110.0), "cmn_Hant": (34.0, 110.0), "jpn": (37.0, 140.0), "kor": (37.5, 128.0),
        "ind": (0.0, 106.0), "fil": (15.0, 121.0), "vie": (10.5, 106.5), "khm": (12.5, 105.0), "tam": (11.0, 78.5),
        "tel": (16.0, 79.0)}
FAMILIES = {"IE": T.GLOTTO[0], "Semitic": T.G("arb", "heb", "mlt"), "Turkic": T.G("tur", "azj", "kaz"),
            "Finnic": T.G("fin", "ekk"), "Sinitic": T.G("cmn_Hans", "cmn_Hant"), "Austronesian": T.G("ind", "fil"),
            "Austroasiatic": T.G("vie", "khm"), "Dravidian": T.G("tam", "tel"), "Japanese": T.G("jpn"),
            "Korean": T.G("kor"), "Indo-Iranian": T.G("hin", "urd", "mar", "pes")}


def geo_matrix():
    ll = np.radians(np.array([WALS[l.split("_")[0] if not l.startswith("cmn") else l] for l in T.LANGS]))
    lat, lon = ll[:, 0], ll[:, 1]
    c = (np.sin(lat[:, None]) * np.sin(lat[None]) + np.cos(lat[:, None]) * np.cos(lat[None]) * np.cos(lon[:, None] - lon[None]))
    return np.arccos(np.clip(c, -1, 1))


def restrict(keep, splits):
    m = {o: n for n, o in enumerate(keep)}
    out = [tuple(sorted(m[i] for i in S if i in m)) for S in splits]
    return [S for S in out if 1 < len(S) < len(keep) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--n_perm", type=int, default=2000)
    args = ap.parse_args()
    grams, tok = A.load(args.data)
    geo = geo_matrix()
    tokd = 1 - tok["tok_hist_int"]; np.fill_diagonal(tokd, 0)
    lf = np.log(tok["fert"]); fert = np.abs(lf[:, None] - lf[None, :])
    out = {}
    for m in A.PRIMARY_METRICS:
        K = A.avg_gram(grams, m)
        D2 = T.d2_from_gram(K)
        res = {}

        def run(keep, tag, with_geo):
            keep = list(keep)
            sp = T.Space(len(keep))
            sub = lambda M: M[np.ix_(keep, keep)]
            covs = [sub(tokd), sub(fert)] + ([sub(geo)] if with_geo else [])
            scr = restrict(keep, T.SCRIPT_SPLITS)
            base = np.column_stack([sp.star] + [sp.col(S) for S in scr] + [sp.covcols(covs)])
            gl = restrict(keep, T.GLOTTO)
            ov = tuple(sorted({o: n for n, o in enumerate(keep)}[i] for i in T.OV if i in keep))
            y = sp.vec(sub(D2))
            r = {"n": len(keep)}
            if gl:
                r["H1_gain"], r["H1_p"], _ = T.perm_relabel(sp, y, base, gl, args.n_perm, 5)
            if 1 < len(ov) < len(keep) - 1:
                b_gl = np.column_stack([base] + [sp.col(S) for S in gl]) if gl else base
                r["OV|H1_gain"], r["OV|H1_p"], _ = T.perm_subset(sp, y, b_gl, ov, args.n_perm, 6)
            res[tag] = r
            print(f"  {m:6s} {tag:28s} n={len(keep):2d} " + " ".join(f"{k}={v:.4f}" for k, v in r.items() if k != "n"),
                  flush=True)

        allk = range(T.N)
        run(allk, "all + geography", True)
        for fam, S in FAMILIES.items():
            run([i for i in allk if i not in S], f"drop {fam} + geography", True)
        out[m] = res
    json.dump(out, open("exploratory34.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
