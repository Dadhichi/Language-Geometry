"""Assemble figdata.json for the write-up from the committed results and local derived data.
usage: python build_data.py OUT.json"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(EX, "prereg34"))
import lib34 as T
import analysis34 as A

DATA = "C:/Users/ASUS/Documents/lang-geom"
FAM = {}
for name, members in [("Germanic", "eng deu nld swe"), ("Romance", "fra spa por"), ("Slavic", "rus ukr pol hrv srp"),
                      ("Indo-Iranian", "hin urd mar pes"), ("Semitic", "arb heb mlt"), ("Turkic", "tur azj kaz"),
                      ("Finnic", "fin ekk"), ("Sinitic", "cmn_Hans cmn_Hant"), ("Japonic", "jpn"), ("Koreanic", "kor"),
                      ("Austronesian", "ind fil"), ("Austroasiatic", "vie khm"), ("Dravidian", "tam tel")]:
    for m in members.split():
        FAM[T.LANGS[T.IX[m]]] = name
NAMES = dict(eng="English", deu="German", nld="Dutch", swe="Swedish", fra="French", spa="Spanish", por="Portuguese",
             rus="Russian", ukr="Ukrainian", pol="Polish", hrv="Croatian", srp="Serbian", hin="Hindi", urd="Urdu",
             mar="Marathi", pes="Persian", arb="Arabic", heb="Hebrew", mlt="Maltese", tur="Turkish", azj="Azerbaijani",
             kaz="Kazakh", fin="Finnish", ekk="Estonian", jpn="Japanese", kor="Korean", ind="Indonesian", fil="Filipino",
             vie="Vietnamese", khm="Khmer", tam="Tamil", tel="Telugu")
POST = T.G("hin", "urd", "mar", "tur", "azj", "kaz", "fin", "ekk", "jpn", "kor", "tam", "tel")


def lname(code):
    if code == "cmn_Hans":
        return "Chinese (Simpl.)"
    if code == "cmn_Hant":
        return "Chinese (Trad.)"
    return NAMES[code.split("_")[0]]


def langs_meta():
    return [dict(code=c, name=lname(c), family=FAM[c], script=c.split("_")[1], ov=i in T.OV, post=i in POST,
                 ie=i in T.GLOTTO[0]) for i, c in enumerate(T.LANGS)]


def residual_map(derived, win, metric="lda05"):
    a, b = map(int, win.split("-")); A.LAYERS_PRIMARY = list(range(a, b + 1))
    g, tok = A.load(derived); sp = T.Space(); base, _ = A.bases(sp, tok)
    y = sp.vec(T.d2_from_gram(A.avg_gram(g, metric)))
    bb, _ = T.nnls(base, y); r = y - base @ bb
    R = np.zeros((T.N, T.N)); R[sp.iu] = r; R = R + R.T
    R = R - R.min() * (1 - np.eye(T.N))
    J = np.eye(T.N) - 1 / T.N; B = -0.5 * J @ R @ J
    w, V = np.linalg.eigh(B); w, V = w[::-1], V[:, ::-1]
    X = V[:, :2] * np.sqrt(np.clip(w[:2], 0, None))
    if np.mean(X[list(T.OV), 0]) < 0:                     # orient: OV to the right
        X[:, 0] *= -1
    return dict(xy=X.round(5).tolist(), var_share=float(np.clip(w[:2], 0, None).sum() / np.clip(w, 0, None).sum()))


def profiles(path):
    r = json.load(open(path))
    out = {}
    for row in r["per_layer"]:
        out.setdefault(row["metric"], []).append(dict(layer=row["layer"], gen=row["H1_gain"], ov=row["H2_gain"],
                                                      p_gen=row["H1_p"], p_ov=row["H2_p"]))
    prim = {m: {k: r[f"primary_{m}"][k] for k in ("H1_gain", "H1_p", "H2_gain", "H2_p", "H1|OV_p", "OV|H1_p", "H3_p",
                                                   "R2_star", "R2_base", "H3_pair_pct")} for m in ("causal", "lda05")}
    return dict(per_layer=out, primary=prim, window=r["prereg"]["layers"])


def steering():
    D = f"{DATA}/steer/steer_out"
    rows = [json.loads(l) for l in open(os.path.join(EX, "steer_ov", "pairs.jsonl"), encoding="utf-8")]
    conds = json.load(open(f"{D}/conditions.json")); z = np.load(f"{D}/scores.npz")
    lang = np.array([r["lang"] for r in rows]); LG = sorted(set(lang))
    m0 = z["cond_0"][:, 0] - z["cond_0"][:, 1]
    find = lambda L, kind, k, r=-1: next(i for i, c in enumerate(conds) if c["layer"] == L and c["kind"] == kind and c["k"] == k and c["r"] == r)
    dmean = lambda ci: float(np.mean([((z[f"cond_{ci}"][:, 0] - z[f"cond_{ci}"][:, 1]) - m0)[lang == g].mean() for g in LG]))
    out = {"layers": {}}
    for L in (8, 14, 20):
        out["layers"][L] = {kind: {k: dmean(find(L, kind, k)) for k in (-2, -1, 1, 2)} for kind in ("ov", "ie")}
    nr = sum(c["kind"] == "rand" and c["k"] == 2 for c in conds)
    out["rand"] = [{k: dmean(find(14, "rand", k, r)) for k in (-2, 2)} for r in range(nr)]
    res = json.load(open(os.path.join(EX, "steer_ov", "results_steer.json")))
    out["primary"] = {k: res["primary"][k] for k in ("b_ov", "rand_mean", "rand_sd", "z", "p_emp", "per_language", "rand_slopes")}
    out["b_ie"] = res["ie_L14"]["b"]; out["ci"] = res["b_ov_boot95"]
    s2, s_2 = z[f"cond_{find(14, 'ov', 2)}"], z[f"cond_{find(14, 'ov', -2)}"]
    out["per_language"] = [dict(code=g, name=lname(g if g != "zho_Hans" else "cmn_Hans").replace(" (Simpl.)", ""),
                                base=float(m0[lang == g].mean()), plus2=float((s2[:, 0] - s2[:, 1])[lang == g].mean()),
                                minus2=float((s_2[:, 0] - s_2[:, 1])[lang == g].mean()), n=int((lang == g).sum()))
                           for g in LG]
    pick = {"eng_Latn": 1, "hin_Deva": 0, "zho_Hans": 1, "tur_Latn": 1}
    out["examples"] = []
    for g, k in pick.items():
        cand = [r for r in rows if r["lang"] == g and len(r["ov"]) < 70]
        out["examples"].append(cand[k] if len(cand) > k else cand[0])
    info = json.load(open(os.path.join(EX, "steer_ov", "steer_dirs_info.json")))
    out["heldout_proj"] = {g: info[f"L14_{g}"]["heldout_proj"] for g in LG}
    out["n_pairs"] = len(rows)
    return out


def belief():
    D = f"{DATA}/tok/tok_out"
    meta = json.load(open(f"{D}/meta.json")); s = meta["s_dim"]; L = len(meta["langs"])
    idx = np.load(f"{D}/index.npz"); ci = np.load(f"{D}/cs_index.npz")
    cst = np.load(f"{D}/cs_tagscore.f16.npy").astype(np.float64)
    out = {}
    for layer in (4, 14, 24):
        Zn = np.load(f"{D}/states_L{layer}.f16.npy").astype(np.float32)[:, :s]
        C = np.stack([Zn[idx["lang"] == k].mean(0) for k in range(L)])
        Zc = np.load(f"{D}/cs_states_L{layer}.f16.npy").astype(np.float32)[:, :s]
        a, b = ci["a"], ci["b"]; dv = C[b] - C[a]
        lam = ((Zc - C[a]) * dv).sum(1) / (dv ** 2).sum(1)
        order = np.lexsort((ci["pos"], ci["seq"]))
        seq, pos, nA = ci["seq"][order], ci["pos"][order], ci["nA"][order]
        lam_o = lam[order]; d_ = cst[order, 1] - cst[order, 0]
        llr = np.zeros(len(order)); st = np.r_[0, np.flatnonzero(np.diff(seq)) + 1, len(seq)]
        for x, y in zip(st[:-1], st[1:]):
            llr[x:y] = np.cumsum(d_[x:y])
        rel = pos - 1 - nA                                   # 0 = first token of segment B
        traj, bayes = {}, {}
        for t in range(-6, 13):
            m = rel == t
            traj[t] = float(lam_o[m].mean()); bayes[t] = float((1 / (1 + np.exp(-np.clip(llr[m], -50, 50)))).mean())
        out[layer] = dict(observed=traj, bayes=bayes)
    res = json.load(open(os.path.join(EX, "prereg_belief", "results_belief.json")))
    out["P1"] = {l: res["P1"][str(l)] for l in (4, 8, 14, 20)}
    out["P1_primary"] = dict(diff=res["P1"]["primary_mean_diff"], p=res["P1"]["primary_p"])
    out["P2"] = {k: v for k, v in res["P2"].items() if not k.isdigit()}
    return out


def main():
    out = dict(langs=langs_meta())
    out["map"] = {"qwen": residual_map(f"{DATA}/q34/derived34", "8-20"), "llama": residual_map(f"{DATA}/l34/derivedL34", "9-23")}
    out["profiles"] = {"qwen": profiles(os.path.join(EX, "prereg34", "results34.json")),
                       "llama": profiles(os.path.join(EX, "prereg34", "results34_llama.json"))}
    out["steering"] = steering()
    out["belief"] = belief()
    for name, f in (("typology_lex_qwen", "typology_lex/results_tl_qwen.json"), ("typology_lex_llama", "typology_lex/results_tl_llama.json"),
                    ("freegen", "steer_gen/results_gen.json"), ("freegen_x", "steer_gen/exploratory_gen.json"),
                    ("freegen_obj", "steer_gen/objtype.json")):
        p = os.path.join(EX, f)
        if os.path.exists(p):
            out[name] = json.load(open(p))
    def clean(o):                                  # NaN/inf are not valid JSON for the browser's JSON.parse
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    json.dump(clean(out), open(sys.argv[1], "w"), allow_nan=False)
    print("written", sys.argv[1], round(os.path.getsize(sys.argv[1]) / 1e3), "KB; sections:", list(out))


if __name__ == "__main__":
    main()
