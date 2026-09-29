"""Centroid norms (relative to mean) and eng-zho distance rank, per layer: is the dominant-language pair central?"""
import numpy as np
import flib as F

for l in (0, 4, 8, 14, 20, 24, 26, 28):
    for v in ("raw", "sqrtn"):
        z = F.load_layer(l, v)
        G = z["gram_a_dev"]
        nrm = np.sqrt(np.diag(G))
        D = np.sqrt(np.maximum(np.diag(G)[:, None] + np.diag(G)[None] - 2 * G, 0))
        iu = np.triu_indices(12, 1)
        rank = (D[iu] < D[0, 7]).sum() + 1
        print(l, v, " ".join(f"{a}:{b:.2f}" for a, b in zip(F.LANGS, nrm / nrm.mean())),
              f"| eng-zho dist rank {rank}/66, d/mean {D[0, 7] / D[iu].mean():.2f}")
