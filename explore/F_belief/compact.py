"""Per layer: per-sentence language offsets o_{i,s} = x_{i,s} - (1/12) sum_i' x_{i',s} (content removed exactly),
projected onto a dev-fitted 64-d basis Q = [U_a (11-d span of dev centroids) | top-53 PCs of the dev interaction
e = o - a_i, taken orthogonal to U_a].  Variants: raw mean pooling, and sqrt(ntok)-rescaled.
Saves layers/L{l}_{variant}.npz with Z_dev [12,n,64], Z_test [12,m,64], full-d energies, centroid Gram."""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = r"C:\Users\ASUS\Documents\lang-geom"
OUT = os.path.join(HERE, "layers")
os.makedirs(OUT, exist_ok=True)
L, K = 12, 64


def load(split, layer, variant):
    X = np.asarray(np.load(os.path.join(DATA, f"mean_{split}_L{layer}.f16.npy"), mmap_mode="r"), dtype=np.float32)
    if variant == "sqrtn":
        nt = np.load(os.path.join(DATA, f"ntok_{split}.npy")).astype(np.float32)
        X *= np.sqrt(nt)[:, :, None]
    X -= X.mean(0, keepdims=True)          # offsets: remove the per-sentence cross-language mean (content)
    return X


def do(layer, variant):
    t0 = time.time()
    O = load("dev", layer, variant)
    n, d = O.shape[1], O.shape[2]
    a = O.mean(1, dtype=np.float64)                         # [12,d] dev centroids of offsets (sum to 0)
    u, sv, vt = np.linalg.svd(a, full_matrices=False)
    r = int((sv > 1e-6 * sv[0]).sum())
    Ua = vt[:r].T                                            # [d, 11]
    Ce = np.zeros((d, d))
    a32 = a.astype(np.float32)
    for i in range(L):
        E = O[i] - a32[i]
        Ce += (E.T @ E).astype(np.float64)
    P = np.eye(d) - Ua @ Ua.T
    Ce = P @ Ce @ P
    w, V = np.linalg.eigh(Ce)
    V = V[:, ::-1][:, : K - r]
    del Ce, P
    Q = np.concatenate([Ua, V], 1).astype(np.float32)       # [d, 64] orthonormal
    res = dict(Q_sv=sv, e_eig_top=w[::-1][:K].astype(np.float32))
    res["Z_dev"] = np.stack([O[i] @ Q for i in range(L)])
    res["o2_dev"] = np.square(O, dtype=np.float32).sum(2)
    res["gram_a_dev"] = a @ a.T
    del O
    Ot = load("devtest", layer, variant)
    at = Ot.mean(1, dtype=np.float64)
    res["Z_test"] = np.stack([Ot[i] @ Q for i in range(L)])
    res["o2_test"] = np.square(Ot, dtype=np.float32).sum(2)
    res["gram_a_test"] = at @ at.T
    res["gram_a_x"] = a @ at.T
    del Ot
    np.savez(os.path.join(OUT, f"L{layer}_{variant}.npz"), **res)
    print(f"L{layer} {variant}: rank {r}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(0, 29, 2))
    for layer in layers:
        for variant in ("raw", "sqrtn"):
            if not os.path.exists(os.path.join(OUT, f"L{layer}_{variant}.npz")):
                do(layer, variant)
