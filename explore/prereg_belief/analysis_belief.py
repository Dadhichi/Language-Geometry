"""PRE-REGISTERED analysis of the token-level data (PREREG.md). Frozen before data; later additions go elsewhere.
usage: python analysis_belief.py DATA_DIR [--boot 10000] [--out results_belief.json]"""
import argparse, json, os
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

ALPHAS = np.logspace(-3, 4, 15)
MID_P1 = [8, 10, 12, 14, 16]
MID_P2 = [8, 12, 14, 16]


def ridge_mse(Xf, yf, Xt, yt):
    m = RidgeCV(alphas=ALPHAS, cv=5).fit(Xf, yf)
    # BUGFIX after first run (no change to the statistic): RidgeCV returns (n,) for a (n,1) target, which broadcast
    # to (n,n); reshape the prediction to the target's shape.
    return ((m.predict(Xt).reshape(yt.shape) - yt) ** 2).sum(1)              # per-row squared error


def boot_mean_p(per_unit, n_boot, rs):
    """one-sided p for mean > 0, bootstrap over units"""
    per_unit = np.asarray(per_unit)
    means = np.array([per_unit[rs.randint(0, len(per_unit), len(per_unit))].mean() for _ in range(n_boot)])
    return float(per_unit.mean()), float((1 + (means <= 0).sum()) / (1 + n_boot))


def posterior(D):
    ti = np.load(os.path.join(D, "tagindex.npz"))
    ts = np.load(os.path.join(D, "tagscore.f16.npy")).astype(np.float64)
    order = np.lexsort((ti["pos"], ti["sent"], ti["lang"]))
    lang, sent, pos, ts = ti["lang"][order], ti["sent"][order], ti["pos"][order], ts[order]
    grp = lang * 100000 + sent
    cum = np.zeros_like(ts)
    start = np.r_[0, np.flatnonzero(np.diff(grp)) + 1, len(grp)]
    for a, b in zip(start[:-1], start[1:]):
        cum[a:b] = np.cumsum(ts[a:b], 0)
    cum -= cum.max(1, keepdims=True)
    pi = np.exp(cum); pi /= pi.sum(1, keepdims=True)
    return {(int(l), int(s), int(p)): k for k, (l, s, p) in enumerate(zip(lang, sent, pos))}, pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="results_belief.json")
    args = ap.parse_args()
    D = args.data
    meta = json.load(open(os.path.join(D, "meta.json")))
    L = len(meta["langs"])
    idx = np.load(os.path.join(D, "index.npz"))
    s_dim = meta["s_dim"]
    rs = np.random.RandomState(0)
    perm = rs.permutation(meta["n_sent"])
    test_sent = np.zeros(meta["n_sent"], bool); test_sent[perm[meta["n_sent"] // 2:]] = True
    key, pi = posterior(D)
    rows = np.array([key.get((int(l), int(s), int(p)), -1) for l, s, p in zip(idx["lang"], idx["sent"], idx["pos"])])
    inwin = rows >= 0
    P = pi[rows[inwin]]
    amb = P.max(1) < 0.9
    is_test = test_sent[idx["sent"][inwin]]
    unit = (idx["lang"][inwin] * 100000 + idx["sent"][inwin])
    onehot = np.eye(L)[P.argmax(1)]
    true1 = np.eye(L)[idx["lang"][inwin]]
    tok0 = np.load(os.path.join(D, "states_L0.f16.npy")).astype(np.float32)[inwin]
    posoh = np.eye(meta["tag_len"] + 1)[np.minimum(idx["pos"][inwin], meta["tag_len"])]
    res = {"n_window": int(inwin.sum()), "n_ambiguous_test": int((amb & is_test).sum()), "P1": {}, "P2": {}}
    print(f"tag-window positions {inwin.sum()}, ambiguous test positions {(amb & is_test).sum()}", flush=True)
    diffs = {}
    for l in meta["layers"]:
        Z = np.load(os.path.join(D, f"states_L{l}.f16.npy")).astype(np.float32)[inwin][:, :s_dim]
        f, t = ~is_test, is_test & amb
        e_post = ridge_mse(P[f], Z[f], P[t], Z[t])
        e_oh = ridge_mse(onehot[f], Z[f], onehot[t], Z[t])
        e_true = ridge_mse(true1[f], Z[f], true1[t], Z[t])
        Xtok = np.hstack([tok0, posoh])
        e_tok = ridge_mse(Xtok[f], Z[f], Xtok[t], Z[t])
        e_tokpost = ridge_mse(np.hstack([Xtok, P])[f], Z[f], np.hstack([Xtok, P])[t], Z[t])
        u = unit[t]
        per_unit = {k: [] for k in np.unique(u)}
        for k, v in zip(u, e_oh - e_post):
            per_unit[k].append(v)
        diffs[l] = {k: np.mean(v) for k, v in per_unit.items()}
        res["P1"][l] = dict(mse_post=float(e_post.mean()), mse_onehot=float(e_oh.mean()), mse_true=float(e_true.mean()),
                            mse_tok=float(e_tok.mean()), mse_tok_post=float(e_tokpost.mean()))
        print(f"L{l:2d} P1 ambiguous-test MSE: posterior {e_post.mean():.4f} | argmax one-hot {e_oh.mean():.4f} | "
              f"true lang {e_true.mean():.4f} | token+pos {e_tok.mean():.4f} | token+pos+posterior {e_tokpost.mean():.4f}",
              flush=True)
    units = sorted(set.intersection(*[set(diffs[l]) for l in MID_P1]))
    per = [np.mean([diffs[l][k] for l in MID_P1]) for k in units]
    res["P1"]["primary_mean_diff"], res["P1"]["primary_p"] = boot_mean_p(per, args.boot, np.random.RandomState(1))
    print(f"P1 PRIMARY (onehot - posterior MSE, L8-16): {res['P1']['primary_mean_diff']:.5f}  p={res['P1']['primary_p']:.4f}")

    # ---------------- P2 code-switch
    ci = np.load(os.path.join(D, "cs_index.npz"))
    cst = np.load(os.path.join(D, "cs_tagscore.f16.npy")).astype(np.float64)
    order = np.lexsort((ci["pos"], ci["seq"]))
    seq, pos, seg, a, b, nA, sent = (ci[k][order] for k in ("seq", "pos", "seg", "a", "b", "nA", "sent"))
    llr = np.zeros(len(order))
    start = np.r_[0, np.flatnonzero(np.diff(seq)) + 1, len(seq)]
    d_ = cst[order, 1] - cst[order, 0]
    for s0, s1 in zip(start[:-1], start[1:]):
        llr[s0:s1] = np.cumsum(d_[s0:s1])
    sig = 1 / (1 + np.exp(-np.clip(llr, -50, 50)))
    cs_test = test_sent[sent]
    lag_rows, fit_rows = {}, {}
    for l in sorted(set(MID_P2) | {x for x in (2, 4, 20, 24) if x in meta.get("cs_layers", [])}):
        path = os.path.join(D, f"cs_states_L{l}.f16.npy")
        if not os.path.exists(path):
            continue
        Zc = np.load(path).astype(np.float32)[order][:, :s_dim]
        Zn = np.load(os.path.join(D, f"states_L{l}.f16.npy")).astype(np.float32)[:, :s_dim]
        fitn = ~test_sent[idx["sent"]]
        C = np.stack([Zn[fitn & (idx["lang"] == k)].mean(0) for k in range(L)])
        dv = C[b] - C[a]
        lam = ((Zc - C[a]) * dv).sum(1) / (dv ** 2).sum(1)
        f, t = ~cs_test, cs_test
        e_sig = ridge_mse(sig[f, None], lam[f, None], sig[t, None], lam[t, None])
        e_step = ridge_mse(seg[f, None].astype(float), lam[f, None], seg[t, None].astype(float), lam[t, None])
        u = seq[t]
        per_unit = {}
        for k, v in zip(u, e_step - e_sig):
            per_unit.setdefault(k, []).append(v)
        fit_rows[l] = {k: np.mean(v) for k, v in per_unit.items()}
        # lag: first post-switch token index with lambda > 0.5 (censored at segment length)
        lags, prefix = [], []
        for s0, s1 in zip(start[:-1], start[1:]):
            post = np.flatnonzero(seg[s0:s1] == 1)
            if len(post) == 0 or not cs_test[s0]:
                continue
            hit = np.flatnonzero(lam[s0:s1][post] > 0.5)
            lags.append(hit[0] if len(hit) else len(post)); prefix.append(nA[s0])
        rho = spearmanr(prefix, lags)[0]
        lag_rows[l] = (np.array(lags), np.array(prefix))
        res["P2"][l] = dict(mse_sigmoid_llr=float(e_sig.mean()), mse_step=float(e_step.mean()), lag_spearman=float(rho),
                            mean_lag=float(np.mean(lags)))
        print(f"L{l:2d} P2 MSE sigmoid(LLR) {e_sig.mean():.4f} vs step {e_step.mean():.4f} | lag~prefix Spearman "
              f"{rho:+.3f} | mean lag {np.mean(lags):.2f} tokens", flush=True)
    mids = [l for l in MID_P2 if l in fit_rows]
    units = sorted(set.intersection(*[set(fit_rows[l]) for l in mids]))
    per = [np.mean([fit_rows[l][k] for l in mids]) for k in units]
    res["P2"]["primary_fit_diff"], res["P2"]["primary_fit_p"] = boot_mean_p(per, args.boot, np.random.RandomState(2))
    rs2 = np.random.RandomState(3)
    rhos = []
    n_seq = len(lag_rows[mids[0]][0])
    for _ in range(min(args.boot, 2000)):
        ii = rs2.randint(0, n_seq, n_seq)
        rhos.append(np.mean([spearmanr(lag_rows[l][1][ii], lag_rows[l][0][ii])[0] for l in mids]))
    rho_obs = float(np.mean([res["P2"][l]["lag_spearman"] for l in mids]))
    res["P2"]["primary_lag_rho"] = rho_obs
    res["P2"]["primary_lag_p"] = float((1 + (np.array(rhos) <= 0).sum()) / (1 + len(rhos)))
    res["P2"]["claim"] = bool(res["P2"]["primary_fit_p"] < 0.025 and res["P2"]["primary_lag_p"] < 0.025)
    res["P1"]["claim"] = bool(res["P1"]["primary_p"] < 0.025)
    print(f"P2 PRIMARY: step - sigmoid MSE {res['P2']['primary_fit_diff']:.5f} p={res['P2']['primary_fit_p']:.4f} | "
          f"lag rho {rho_obs:+.3f} p={res['P2']['primary_lag_p']:.4f}")
    print(f"CLAIM P1 (belief-affine): {res['P1']['claim']} | CLAIM P2 (Bayesian code-switch): {res['P2']['claim']}")
    json.dump(res, open(args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
