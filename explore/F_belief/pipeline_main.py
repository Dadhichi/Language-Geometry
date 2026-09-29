"""The instrument pipeline (identical for synthetic and real data)."""
import numpy as np
import flib as F

INSTR = ("e_tokc", "e_char", "B_bag", "B_pre", "B_first", "B_early", "B_late", "e_tokc_LL", "B_bag_LL")


def prep_LL(x):
    """within-Latin specificity: for Latin sources, demean over Latin targets only; everything else 0"""
    x = x.astype(np.float64)
    out = np.zeros_like(x)
    for i in F.LATIN:
        js = [j for j in F.LATIN if j != i]
        sub = x[i, js]
        out[i, js] = sub - sub.mean(0, keepdims=True)
    out = out - out.mean(2, keepdims=True)
    for i in range(F.L):
        for j in range(F.L):
            if i == j or i not in F.LATIN or j not in F.LATIN:
                out[i, j] = 0
    return out


class Ctx:
    """Everything that does not depend on the (possibly synthetic) displacements."""

    def __init__(self, Idev, Itest, A):
        self.A = A                                        # [12,11] dev centroids in span coords
        self.xt = {}
        for sp, I in (("dev", Idev), ("test", Itest)):
            self.xt[sp] = {nm: (prep_LL(I[nm[:-3]]) if nm.endswith("_LL") else F.prep(I[nm])) for nm in INSTR}
        self.Dh = {sp: {nm: F.dhat(self.xt[sp][nm], A) for nm in INSTR} for sp in self.xt}
        self.Zc = {"dev": F.covariates(Idev), "test": F.covariates(Itest)}


SETS = (("e_tokc",), ("e_char",), ("B_bag",), ("B_pre",), ("e_tokc_LL",), ("B_bag_LL",), ("B_bag", "B_pre"),
        ("e_tokc", "B_pre"), ("B_early", "B_late", "B_pre"), ("B_first", "B_pre"), ("e_tokc", "e_tokc_LL"))


def run(ctx, d_dev, d_test, sets=SETS, nperm=100, seed=0, covs=True):
    rng = np.random.RandomState(seed)
    d_dev, d_test = F.centre(d_dev), F.centre(d_test)
    Zd = ctx.Zc["dev"] if covs else None
    Zt = ctx.Zc["test"] if covs else None
    M = F.metric(F.residualize(d_dev, Zd))
    out = {}
    for names in sets:
        Dd = [ctx.Dh["dev"][nm] for nm in names]
        Dt = [ctx.Dh["test"][nm] for nm in names]
        g, Gam = F.gfit(d_dev, Dd, Zd, M)
        rho, R2, r_t = F.heldout(d_test, Dt, Zt, Gam, g, M)
        g_test = F.gfit(d_test, Dt, Zt, M)[0]
        key = "+".join(names)
        res = dict(g_dev=g, g_test=g_test, rho=rho, R2=R2)
        if len(names) == 1 and nperm:
            rr = F.ip(r_t, r_t, M)
            nm = names[0]
            ns, nj = [], []
            for _ in range(nperm):
                Ds = F.shuffle_rows(Dt[0], rng)
                ns.append(F.ip(r_t, Ds, M) / np.sqrt(rr * F.ip(Ds, Ds, M)))
                Dj = F.dhat(F.jperm(ctx.xt["test"][nm], rng), ctx.A)
                nj.append(F.ip(r_t, Dj, M) / np.sqrt(rr * F.ip(Dj, Dj, M)))
            ns, nj = np.array(ns), np.array(nj)
            res.update(z_shuf=(rho[0] - ns.mean()) / ns.std(), z_jperm=(rho[0] - nj.mean()) / nj.std(),
                       null_shuf_sd=ns.std(), null_jperm_mean=nj.mean(), null_jperm_sd=nj.std())
        out[key] = res
    return out


def fmt(out):
    lines = []
    for k, r in out.items():
        s = f"  {k:14s} g_dev={np.array2string(np.asarray(r['g_dev']), precision=3)} " \
            f"g_test={np.array2string(np.asarray(r['g_test']), precision=3)} rho={np.array2string(r['rho'], precision=4)} " \
            f"R2={r['R2']:.4f}"
        if "z_shuf" in r:
            s += f" z_shuf={r['z_shuf']:.1f} z_jperm={r['z_jperm']:.1f}"
        lines.append(s)
    return "\n".join(lines)
