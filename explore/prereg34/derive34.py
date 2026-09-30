"""Runs on the Colab VM after extract.py: raw activations -> compact inputs of analysis34.py.

usage: python derive34.py EXTRACT_DIR OUT_DIR COVGAMMA_F32_NPY [--B 50]
Per layer (mean pooling): centroid offsets a_i = mu_i - mean_i mu_i on dev and devtest, and Grams under
  euc    : u.v
  lda01  : u^T W_a^-1 v, W = pooled within-language covariance (dev), shrinkage a = 0.1 toward (trW/d) I
  lda05  : same, a = 0.5
  causal : u^T C v, C = diag(g) Cov(gamma) diag(g) (Park et al. causal inner product; from explore/E_tree/covgamma.py)
keys {m}_dd, {m}_tt, {m}_dt (cross-split, unbiased), {m}_bdt [B] (sentence bootstrap, same indices across
languages), {m}_dt_sq (sqrt(ntok)-rescaled pooling variant).  Also centroids.npz and tokstats.npz
(tok_hist_int = histogram intersection of Qwen token-frequency distributions over dev+devtest text; fert = mean
content tokens per sentence).
"""
import argparse, json, os, time
from collections import Counter
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("covgamma")
ap.add_argument("--B", type=int, default=50)
args = ap.parse_args()
os.makedirs(os.path.join(args.out, "grams"), exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta = json.load(open(os.path.join(args.src, "meta.json")))
langs, n_layers, d = meta["langs"], meta["n_layers"], meta["d"]
N = len(langs)
C = torch.as_tensor(np.load(args.covgamma).astype(np.float32), device=dev)
ntok = {s: torch.as_tensor(np.load(os.path.join(args.src, f"ntok_{s}.npy")).astype(np.float32), device=dev)
        for s in ("dev", "devtest")}
gscale = torch.cat([ntok["dev"].flatten(), ntok["devtest"].flatten()]).mean()


def load(split, layer):
    X = np.load(os.path.join(args.src, f"mean_{split}_L{layer}.f16.npy"))
    return torch.as_tensor(X.astype(np.float32), device=dev)                     # [N, n, d]


def offsets(X, cnt=None):
    mu = X.mean(1) if cnt is None else torch.einsum("bn,lnd->bld", cnt, X)
    return mu - mu.mean(-2, keepdim=True)


cents = {}
for layer in range(n_layers):
    t0 = time.time()
    rs = np.random.RandomState(1000 + layer)
    Xd, Xt = load("dev", layer), load("devtest", layer)
    Ad, At = offsets(Xd), offsets(Xt)
    cnt = lambda n: torch.as_tensor(rs.multinomial(n, np.ones(n) / n, size=args.B).astype(np.float32) / n, device=dev)
    Adb, Atb = offsets(Xd, cnt(Xd.shape[1])), offsets(Xt, cnt(Xt.shape[1]))
    sq = lambda X, s: X * torch.sqrt(ntok[s] / gscale)[..., None]
    Ads, Ats = offsets(sq(Xd, "dev")), offsets(sq(Xt, "devtest"))
    # pooled within-language covariance on dev
    Y = Xd - Xd.mean(1, keepdim=True)
    W = torch.einsum("lnd,lne->de", Y, Y).double() / (N * (Xd.shape[1] - 1))
    del Y
    w, V = torch.linalg.eigh(W)
    w = w.clamp(min=0)
    res = {}

    def put(name, f):
        zd, zt, zdb, ztb, zds, zts = (f(a) for a in (Ad, At, Adb, Atb, Ads, Ats))
        g = lambda P, Q: (P.double() @ Q.double().transpose(-1, -2)).cpu().numpy()
        res[f"{name}_dd"], res[f"{name}_tt"], res[f"{name}_dt"] = g(zd, zd), g(zt, zt), g(zd, zt)
        res[f"{name}_bdt"] = g(zdb, ztb)
        res[f"{name}_dt_sq"] = g(zds, zts)

    put("euc", lambda A: A)
    for a, nm in ((0.1, "lda01"), (0.5, "lda05")):
        s = torch.sqrt((1 - a) * w + a * w.mean()).float()
        Vf = V.float()
        put(nm, lambda A, s=s, Vf=Vf: (A @ Vf) / s)
    # causal: K = A C A^T -> write as (A C) . A
    Cf = C
    for key, (P, Q) in {"dd": (Ad, Ad), "tt": (At, At), "dt": (Ad, At), "bdt": (Adb, Atb), "dt_sq": (Ads, Ats)}.items():
        res[f"causal_{key}"] = ((P @ Cf).double() @ Q.double().transpose(-1, -2)).cpu().numpy()
    res["W_eig"] = w.flip(0).float().cpu().numpy()
    np.savez(os.path.join(args.out, "grams", f"L{layer}.npz"), **res)
    cents[f"dev_L{layer}"], cents[f"devtest_L{layer}"] = Ad.cpu().numpy(), At.cpu().numpy()
    cents[f"dev_sq_L{layer}"], cents[f"devtest_sq_L{layer}"] = Ads.cpu().numpy(), Ats.cpu().numpy()
    del Xd, Xt, W, V
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    print(f"L{layer} {time.time() - t0:.0f}s", flush=True)
np.savez(os.path.join(args.out, "centroids.npz"), **cents)

# token statistics from text (Qwen tokenizer)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(meta["model"], token=os.environ.get("HF_TOKEN"))
dist = []
for lang in langs:
    c = Counter()
    for s in ("dev", "devtest"):
        txt = json.load(open(os.path.join(args.src, f"sentences_{s}.json"), encoding="utf-8"))["text"][lang]
        for ids in tok(txt, add_special_tokens=False)["input_ids"]:
            c.update(ids)
    tot = sum(c.values())
    dist.append({k: v / tot for k, v in c.items()})
H = np.zeros((N, N))
for i in range(N):
    for j in range(i, N):
        a, b = dist[i], dist[j]
        H[i, j] = H[j, i] = sum(min(v, b.get(k, 0.0)) for k, v in a.items())
fert = np.concatenate([np.load(os.path.join(args.src, f"ntok_{s}.npy")) for s in ("dev", "devtest")], 1).mean(1)
np.savez(os.path.join(args.out, "tokstats.npz"), tok_hist_int=H, fert=fert, langs=np.array(langs))
json.dump({"langs": langs, "n_layers": n_layers, "d": d, "B": args.B, "model": meta["model"]},
          open(os.path.join(args.out, "meta.json"), "w"), indent=1)
print("done")
