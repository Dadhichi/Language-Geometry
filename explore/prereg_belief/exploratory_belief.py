"""EXPLORATORY (after the pre-registered P2 failed): is the code-switch trajectory a LEAKY Bayesian integrator?
lambda_t ~ a + b * sigmoid(c * L_h(t)),  L_h(t) = sum_{tau<=t} rho^(t-tau) d_tau,  rho = 2^(-1/h)  (h = half-life in
tokens; h = inf is the pre-registered cumulative LLR). (h, c) chosen on the fit half by MSE, scored on the test half
against the step model. Same lambda, split and layers as analysis_belief.py.
usage: python exploratory_belief.py DATA_DIR"""
import json, os, sys
import numpy as np
import analysis_belief as AB

D = sys.argv[1]
meta = json.load(open(os.path.join(D, "meta.json")))
L, s_dim = len(meta["langs"]), meta["s_dim"]
idx = np.load(os.path.join(D, "index.npz"))
rs = np.random.RandomState(0)
perm = rs.permutation(meta["n_sent"])
test_sent = np.zeros(meta["n_sent"], bool); test_sent[perm[meta["n_sent"] // 2:]] = True
ci = np.load(os.path.join(D, "cs_index.npz"))
cst = np.load(os.path.join(D, "cs_tagscore.f16.npy")).astype(np.float64)
order = np.lexsort((ci["pos"], ci["seq"]))
seq, seg, a, b, sent, nA = (ci[k][order] for k in ("seq", "seg", "a", "b", "sent", "nA"))
d_ = cst[order, 1] - cst[order, 0]
start = np.r_[0, np.flatnonzero(np.diff(seq)) + 1, len(seq)]
H = [0.5, 1, 2, 3, 5, 8, 13, 21, np.inf]


def leaky(h):
    rho = 0.0 if h == 0 else (1.0 if np.isinf(h) else 2 ** (-1 / h))
    out = np.zeros(len(d_))
    for s0, s1 in zip(start[:-1], start[1:]):
        acc = 0.0
        for k in range(s0, s1):
            acc = rho * acc + d_[k]
            out[k] = acc
    return out


Ls = {h: leaky(h) for h in H}
cs_test = test_sent[sent]
f, t = ~cs_test, cs_test
res = {}
for l in [8, 12, 14, 16]:
    Zc = np.load(os.path.join(D, f"cs_states_L{l}.f16.npy")).astype(np.float32)[order][:, :s_dim]
    Zn = np.load(os.path.join(D, f"states_L{l}.f16.npy")).astype(np.float32)[:, :s_dim]
    fitn = ~test_sent[idx["sent"]]
    C = np.stack([Zn[fitn & (idx["lang"] == k)].mean(0) for k in range(L)])
    dv = C[b] - C[a]
    lam = ((Zc - C[a]) * dv).sum(1) / (dv ** 2).sum(1)
    best = None
    for h in H:
        for c in [0.03, 0.1, 0.3, 1.0, 3.0]:
            x = 1 / (1 + np.exp(-np.clip(c * Ls[h], -50, 50)))
            e_fit = AB.ridge_mse(x[f, None], lam[f, None], x[f, None], lam[f, None]).mean()
            if best is None or e_fit < best[0]:
                best = (e_fit, h, c)
    _, h, c = best
    x = 1 / (1 + np.exp(-np.clip(c * Ls[h], -50, 50)))
    e_leaky = AB.ridge_mse(x[f, None], lam[f, None], x[t, None], lam[t, None]).mean()
    e_step = AB.ridge_mse(seg[f, None].astype(float), lam[f, None], seg[t, None].astype(float), lam[t, None]).mean()
    both = np.column_stack([x, seg])
    e_both = AB.ridge_mse(both[f], lam[f, None], both[t], lam[t, None]).mean()
    res[l] = dict(best_half_life=float(h), best_scale=c, mse_leaky=float(e_leaky), mse_step=float(e_step),
                  mse_step_plus_leaky=float(e_both))
    print(f"L{l:2d} best half-life {h} tokens (scale {c}) | test MSE leaky {e_leaky:.4f} vs step {e_step:.4f} | "
          f"step+leaky {e_both:.4f}", flush=True)
json.dump(res, open("exploratory_belief.json", "w"), indent=1)
