"""Specificity of the conditional tests: H1|OV (genealogy beyond word order) and OV|H1 (word order beyond
genealogy) in planted null / genealogy / OV / both worlds (same generator as synth_check34.py)."""
import sys
import numpy as np
import lib34 as T
import analysis34 as A
import synth_check34 as S


def both(c, rs, dim=32):
    K1 = S.world("glotto", c, rs, dim)
    return K1


def main(reps=60, n_perm=200):
    rs = np.random.RandomState(1)
    tok = S.tokstats(rs)
    base, _ = A.bases(S.sp, tok)
    b_ov = np.column_stack([base, S.sp.col(T.OV)])
    b_gl = np.column_stack([base] + [S.sp.col(X) for X in T.GLOTTO])
    for kind, c in (("null", 0.0), ("glotto", 0.1), ("ov", 0.1), ("glotto", 0.2), ("ov", 0.2)):
        r1 = r2 = 0
        for k in range(reps):
            K = S.world(kind, c, rs)
            y = S.sp.vec(T.d2_from_gram(K / np.trace(K)))
            r1 += T.perm_relabel(S.sp, y, b_ov, T.GLOTTO, n_perm, k)[1] < 0.05
            r2 += T.perm_subset(S.sp, y, b_gl, T.OV, n_perm, k)[1] < 0.05
        print(f"{kind:6s} c={c:.1f}: reject H1|OV {r1 / reps:.2f}  OV|H1 {r2 / reps:.2f}", flush=True)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
