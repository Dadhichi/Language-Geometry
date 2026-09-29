"""Cov(gamma) of Qwen2.5-7B unembedding rows (Park et al. causal inner product), from range-fetched bf16 bytes.
gamma rows restricted to the tokenizer's real vocab (151665; padding rows dropped).  Two versions:
  C_raw  = Cov(gamma)                       (acts on the final-RMSNorm output)
  C_norm = diag(g) Cov(gamma) diag(g)       (acts on the residual stream before the final RMSNorm; g = model.norm.weight)
Saved float64 [3584,3584] under C:/Users/ASUS/Documents/lang-geom/."""
import os, time
import numpy as np

DATA = r"C:\Users\ASUS\Documents\lang-geom"
V, d, VREAL = 152064, 3584, 151665


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


t0 = time.time()
W = np.memmap(os.path.join(DATA, "lm_head.bf16"), dtype=np.uint16, mode="r", shape=(V, d))
g = bf16_to_f32(np.fromfile(os.path.join(DATA, "norm_weight.bf16"), dtype=np.uint16))
CH = 4096
s = np.zeros(d)
for a in range(0, VREAL, CH):
    s += bf16_to_f32(np.asarray(W[a:min(a + CH, VREAL)])).sum(0, dtype=np.float64)
mean = s / VREAL
m32 = mean.astype(np.float32)
C = np.zeros((d, d))
for a in range(0, VREAL, CH):
    X = bf16_to_f32(np.asarray(W[a:min(a + CH, VREAL)])) - m32
    C += (X.T @ X).astype(np.float64)
    del X
C /= (VREAL - 1)
pad = bf16_to_f32(np.asarray(W[VREAL:]))
print("pad rows norm mean", float(np.linalg.norm(pad, axis=1).mean()), "real mean-norm", float(np.linalg.norm(mean)))
np.save(os.path.join(DATA, "covgamma_raw.npy"), C)
np.save(os.path.join(DATA, "gamma_mean.npy"), mean)
np.save(os.path.join(DATA, "norm_weight.npy"), g)
Cn = C * g[:, None] * g[None, :]
np.save(os.path.join(DATA, "covgamma_norm.npy"), Cn)
w = np.linalg.eigvalsh(Cn)[::-1]
print("C_norm eig top5", w[:5], "PR", w.sum() ** 2 / (w ** 2).sum(), "min", w[-1])
print("g stats", g.min(), g.mean(), g.max())
print(f"done {time.time() - t0:.0f}s")
