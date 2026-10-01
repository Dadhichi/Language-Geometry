"""PRE-REGISTERED (PREREG.md): (A) typology bundle (83A OV vs 85A adpositions vs 86A/87A); (B) lexical control for
genealogy (romanised FLORES character 3-grams; ASJP LDND). Uses explore/prereg34 Grams + lib34 machinery.
usage: python analysis_tl.py DERIVED_DIR PRIMARY_WINDOW(a-b) ASJP_DIR OUT.json [--n_perm 10000]"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prereg34"))
import lib34 as T
import analysis34 as A

G = T.G
OV = T.OV
POST = G("hin", "urd", "mar", "tur", "azj", "kaz", "fin", "ekk", "jpn", "kor", "tam", "tel")
GENN = G("swe", "hin", "urd", "mar", "tur", "azj", "kaz", "fin", "ekk", "cmn_Hans", "cmn_Hant", "jpn", "kor", "tam", "tel")
NADJ = G("fra", "spa", "por", "pes", "arb", "heb", "mlt", "ind", "vie", "khm")
TYPO = {"OV": OV, "POST": POST, "GENN": GENN, "NADJ": NADJ}
ASJP = {"eng_Latn": "ENGLISH", "deu_Latn": "STANDARD_GERMAN", "nld_Latn": "DUTCH", "swe_Latn": "SWEDISH",
        "fra_Latn": "FRENCH", "spa_Latn": "SPANISH", "por_Latn": "PORTUGUESE", "rus_Cyrl": "RUSSIAN",
        "ukr_Cyrl": "UKRAINIAN", "pol_Latn": "POLISH", "hrv_Latn": "CROATIAN", "srp_Cyrl": "SERBOCROATIAN",
        "hin_Deva": "HINDI", "urd_Arab": "URDU", "mar_Deva": "MARATHI", "pes_Arab": "PERSIAN", "arb_Arab": "STANDARD_ARABIC",
        "heb_Hebr": "MODERN_HEBREW", "mlt_Latn": "MALTESE", "tur_Latn": "TURKISH", "azj_Latn": "AZERBAIJANI_NORTH",
        "kaz_Cyrl": "KAZAKH", "fin_Latn": "FINNISH", "ekk_Latn": "ESTONIAN", "cmn_Hans": "MANDARIN", "cmn_Hant": "MANDARIN",
        "jpn_Jpan": "JAPANESE", "kor_Hang": "KOREAN", "ind_Latn": "INDONESIAN", "fil_Latn": "TAGALOG",
        "vie_Latn": "VIETNAMESE", "khm_Khmr": "KHMER", "tam_Taml": "TAMIL", "tel_Telu": "TELUGU"}


def lex_text(derived):
    from unidecode import unidecode
    sims = np.zeros((T.N, T.N)); n = 0
    for s in ("dev", "devtest"):
        S = json.load(open(os.path.join(derived, f"sentences_{s}.json"), encoding="utf-8"))["text"]
        grams = [[{t[i:i + 3] for i in range(len(t) - 2)} for t in (unidecode(x).lower() for x in S[l])] for l in T.LANGS]
        m = len(grams[0])
        for i in range(T.N):
            for j in range(i, T.N):
                v = np.mean([len(a & b) / max(1, len(a | b)) for a, b in zip(grams[i], grams[j])])
                sims[i, j] += v * m; sims[j, i] = sims[i, j]
        n += m
    D = 1 - sims / n
    np.fill_diagonal(D, 0)
    return D


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lex_asjp(asjp_dir):
    import pandas as pd
    P = pd.read_csv(os.path.join(asjp_dir, "parameters.csv"))
    forty = set(P.ID[P.Name.str.startswith("*")])
    F = pd.read_csv(os.path.join(asjp_dir, "forms.csv"), usecols=["Language_ID", "Parameter_ID", "Form"])
    F = F[F.Language_ID.isin(set(ASJP.values())) & F.Parameter_ID.isin(forty)]
    clean = lambda w: "".join(ch for ch in str(w) if ch not in '*"~$ ')
    words = {lid: {} for lid in ASJP.values()}
    for r in F.itertuples():
        words[r.Language_ID].setdefault(r.Parameter_ID, []).append(clean(r.Form))
    ldn = lambda u, v: min(lev(a, b) / max(len(a), len(b), 1) for a in u for b in v)
    D = np.zeros((T.N, T.N))
    for i, li in enumerate(T.LANGS):
        for j in range(i + 1, T.N):
            wa, wb = words[ASJP[li]], words[ASJP[T.LANGS[j]]]
            shared = sorted(set(wa) & set(wb))
            same = np.mean([ldn(wa[c], wb[c]) for c in shared])
            diff = np.mean([ldn(wa[c], wb[d]) for c in shared for d in shared if c != d])
            D[i, j] = D[j, i] = same / diff
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("derived"); ap.add_argument("window"); ap.add_argument("asjp"); ap.add_argument("out")
    ap.add_argument("--n_perm", type=int, default=10000)
    args = ap.parse_args()
    a, b = map(int, args.window.split("-"))
    A.LAYERS_PRIMARY = list(range(a, b + 1))
    grams, tok = A.load(args.derived)
    sp = T.Space()
    base, _ = A.bases(sp, tok)
    cache = os.path.join(os.path.dirname(os.path.abspath(args.out)), "lexdist.npz")
    if os.path.exists(cache):
        z = np.load(cache); LT, LA_ = z["lex_text"], z["lex_asjp"]
    else:
        LT, LA_ = lex_text(args.derived), lex_asjp(args.asjp)
        np.savez(cache, lex_text=LT, lex_asjp=LA_)
    gl = [sp.col(S) for S in T.GLOTTO]
    res = {"window": args.window}
    for m in A.PRIMARY_METRICS:
        y = sp.vec(T.d2_from_gram(A.avg_gram(grams, m)))
        r = {}
        bg = np.column_stack([base] + gl)
        r["A1_OV|POST_gain"], r["A1_OV|POST_p"], _ = T.perm_subset(sp, y, np.column_stack([bg, sp.col(POST)]), OV, args.n_perm, 31)
        r["A2_POST|OV_gain"], r["A2_POST|OV_p"], _ = T.perm_subset(sp, y, np.column_stack([bg, sp.col(OV)]), POST, args.n_perm, 32)
        for k, S in TYPO.items():
            others = [sp.col(X) for kk, X in TYPO.items() if kk != k]
            r[f"A3_{k}_alone"] = T.gain(sp, y, bg, [S])
            r[f"A3_{k}_beyond_others"] = T.gain(sp, y, np.column_stack([bg] + others), [S])
        b1 = np.column_stack([base, sp.covcols([LT])])
        b2 = np.column_stack([base, sp.covcols([LT, LA_])])
        r["B1_H1|LEXTEXT_gain"], r["B1_H1|LEXTEXT_p"], _ = T.perm_relabel(sp, y, b1, T.GLOTTO, args.n_perm, 33)
        r["B2_H1|LEXTEXT+ASJP_gain"], r["B2_H1|LEXTEXT+ASJP_p"], _ = T.perm_relabel(sp, y, b2, T.GLOTTO, args.n_perm, 34)
        r["H1_gain_plainbase"] = T.gain(sp, y, base, T.GLOTTO)
        res[m] = r
        print(f"{m:6s} A1 OV|POST gain {r['A1_OV|POST_gain']:.3f} p={r['A1_OV|POST_p']:.4f} | A2 POST|OV gain "
              f"{r['A2_POST|OV_gain']:.3f} p={r['A2_POST|OV_p']:.4f} | alone OV {r['A3_OV_alone']:.3f} POST {r['A3_POST_alone']:.3f} "
              f"GENN {r['A3_GENN_alone']:.3f} NADJ {r['A3_NADJ_alone']:.3f} | beyond others OV {r['A3_OV_beyond_others']:.3f} "
              f"POST {r['A3_POST_beyond_others']:.3f} GENN {r['A3_GENN_beyond_others']:.3f} NADJ {r['A3_NADJ_beyond_others']:.3f}", flush=True)
        print(f"       B genealogy gain: plain {r['H1_gain_plainbase']:.3f} | +LEX_TEXT {r['B1_H1|LEXTEXT_gain']:.3f} "
              f"p={r['B1_H1|LEXTEXT_p']:.4f} | +LEX_TEXT+ASJP {r['B2_H1|LEXTEXT+ASJP_gain']:.3f} p={r['B2_H1|LEXTEXT+ASJP_p']:.4f}", flush=True)
    # descriptive depth profile: genealogy gain with / without LEX_TEXT, every layer
    prof = []
    b1 = np.column_stack([base, sp.covcols([LT])])
    for l in range(len(grams)):
        for m in A.PRIMARY_METRICS:
            K = 0.5 * (grams[l][f"{m}_dt"] + grams[l][f"{m}_dt"].T)
            y = sp.vec(T.d2_from_gram(K / np.trace(K)))
            prof.append(dict(layer=l, metric=m, gl_plain=T.gain(sp, y, base, T.GLOTTO), gl_lex=T.gain(sp, y, b1, T.GLOTTO),
                             ov_plain=T.gain(sp, y, base, [OV]), ov_lex=T.gain(sp, y, b1, [OV])))
    res["profile"] = prof
    res["lex_corr"] = dict(text_vs_asjp=float(np.corrcoef(sp.vec(LT), sp.vec(LA_))[0, 1]))
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    print("profile (causal): " + " ".join(f"L{p['layer']}:{p['gl_plain']:.2f}->{p['gl_lex']:.2f}" for p in prof if p["metric"] == "causal"))


if __name__ == "__main__":
    main()
