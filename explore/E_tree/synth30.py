"""Power of the split-gain test for the proposed 34-language FLORES extraction vs the current 12.
Same planted BM model as synth.py (isotropic increments in `dim` dims, lognormal leaf lengths with the real CV and a
Hindi-like outlier), internal share f.  Worlds: 'glotto' (planted genealogy) and 'script' (planted script clades).
Bases: star, star+script-different (+ a token proxy = script-different + within-script noise).  Also the exact rank of
the same-language/different-script pairs (hin/urd, hrv/srp, zho Hans/Hant) among all pairs."""
import itertools, json, os, sys, time
import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["eng_Latn", "deu_Latn", "nld_Latn", "swe_Latn", "fra_Latn", "spa_Latn", "por_Latn",
         "rus_Cyrl", "ukr_Cyrl", "pol_Latn", "hrv_Latn", "srp_Cyrl",
         "hin_Deva", "urd_Arab", "mar_Deva", "pes_Arab",
         "arb_Arab", "heb_Hebr", "mlt_Latn", "tur_Latn", "azj_Latn", "kaz_Cyrl", "fin_Latn", "est_Latn",
         "zho_Hans", "zho_Hant", "jpn_Jpan", "kor_Hang", "ind_Latn", "tgl_Latn", "vie_Latn", "khm_Khmr",
         "tam_Taml", "tel_Telu"]
N = len(LANGS)
ix = {l.split("_")[0] + ("_" + l.split("_")[1] if l.startswith("zho") else ""): i for i, l in enumerate(LANGS)}
G = lambda *a: tuple(sorted(ix[x] for x in a))
GLOTTO30 = [G("eng", "deu", "nld", "swe", "fra", "spa", "por", "rus", "ukr", "pol", "hrv", "srp", "hin", "urd", "mar", "pes"),
            G("eng", "deu", "nld", "swe"), G("eng", "deu", "nld"), G("fra", "spa", "por"), G("spa", "por"),
            G("rus", "ukr", "pol", "hrv", "srp"), G("rus", "ukr"), G("hrv", "srp"),
            G("hin", "urd", "mar", "pes"), G("hin", "urd", "mar"), G("hin", "urd"),
            G("arb", "heb", "mlt"), G("arb", "mlt"), G("tur", "azj", "kaz"), G("tur", "azj"), G("fin", "est"),
            G("zho_Hans", "zho_Hant"), G("ind", "tgl"), G("vie", "khm"), G("tam", "tel")]
SCR = [l.split("_")[1].replace("Hant", "Hans") for l in LANGS]
SCRIPT30 = [tuple(i for i in range(N) if SCR[i] == s) for s in sorted(set(SCR)) if sum(x == s for x in SCR) > 1]
PAIRS_SAMELANG = [G("hin", "urd"), G("hrv", "srp"), G("zho_Hans", "zho_Hant")]

GL12 = [(0, 1, 2, 3, 4, 5), (0, 1), (2, 3)]
SCR12 = ["Latn"] * 4 + ["Cyrl", "Deva", "Arab", "Hans", "Hans", "Latn", "Latn", "Latn"]
SCRIPT12 = [(0, 1, 2, 3, 9, 10, 11), (7, 8)]


class Design:
    def __init__(self, n, scripts):
        self.n = n
        self.iu = np.triu_indices(n, 1)
        m = len(self.iu[0])
        self.star = np.zeros((m, n))
        self.star[np.arange(m), self.iu[0]] = 1
        self.star[np.arange(m), self.iu[1]] = 1
        sd = np.array([scripts[i] != scripts[j] for i, j in zip(*self.iu)], float)
        self.bases = {"star": self.star, "star+script": np.column_stack([self.star, sd])}
        self.scripts = scripts

    def add_script_splits(self, script_splits):
        self.bases["star+scriptsplits"] = np.column_stack([self.star] + [self.col(S) for S in script_splits])

    def col(self, S):
        v = np.zeros(self.n, bool); v[list(S)] = True
        return (v[self.iu[0]] ^ v[self.iu[1]]).astype(float)

    def gain(self, y, base, splits):
        s0 = nnls(base, y)[1] ** 2
        return 1 - nnls(np.column_stack([base] + [self.col(S) for S in splits]), y)[1] ** 2 / s0

    def perm_p(self, y, base, splits, n_perm, rs):
        obs = self.gain(y, base, splits)
        null = [self.gain(y, base, [tuple(p[i] for i in S) for S in splits])
                for p in (rs.permutation(self.n) for _ in range(n_perm))]
        return obs, (1 + np.sum(np.array(null) >= obs - 1e-12)) / (1 + n_perm)

    def pair_rank(self, y, base, pair):
        s0 = nnls(base, y)[1] ** 2
        g = {P: 1 - nnls(np.column_stack([base, self.col(P)]), y)[1] ** 2 / s0
             for P in itertools.combinations(range(self.n), 2)}
        return float(np.mean([v >= g[pair] - 1e-12 for v in g.values()]))


def leaves(n, rs, hin=None):
    b = np.exp(rs.randn(n) * 0.35)          # real star-fit leaf lengths: CV ~0.35 apart from Hindi
    if hin is not None:
        b[hin] *= 3.0
    return b


def world(des, tree, f, leaf, rs, dim, c=None):
    X = rs.randn(des.n, dim) * np.sqrt(leaf / dim)[:, None]
    star_sum = des.star.sum(0) @ leaf
    target = f / (1 - f) * star_sum if c is None else 0
    s = target / sum(des.col(S).sum() for S in tree) if c is None else c * leaf.mean()
    for S in tree:
        X[list(S)] += rs.randn(dim) * np.sqrt(s / dim)
    K = X @ X.T
    dg = np.diag(K)
    return (dg[:, None] + dg[None, :] - 2 * K)[des.iu]


def cherries(tree):
    return [S for S in tree if len(S) == 2]


def run(reps=50, n_perm=200, dim=16, cs=(0.1, 0.2)):
    """per-edge parametrisation: every internal edge has length c * mean leaf length (same c for both designs)"""
    rs = np.random.RandomState(7)
    out = []
    for name, n, gl, sc, scr, hin in (("12", 12, GL12, SCRIPT12, SCR12, 5), ("34", N, GLOTTO30, SCRIPT30, SCR, ix["hin"])):
        des = Design(n, scr)
        des.add_script_splits(sc)
        for wname, tree in (("glotto", gl), ("script", sc)):
            for c in cs:
                t0 = time.time()
                acc = {k: [] for k in ("full|star", "full|scr", "cherry|scr")}
                fs = []
                for r in range(reps):
                    leaf = leaves(n, rs, hin)
                    y = world(des, tree, None, leaf, rs, dim, c)
                    fs.append(c * leaf.mean() * sum(des.col(S).sum() for S in tree) / y.sum())
                    acc["full|star"].append(des.perm_p(y, des.bases["star"], gl, n_perm, rs)[1])
                    acc["full|scr"].append(des.perm_p(y, des.bases["star+scriptsplits"], gl, n_perm, rs)[1])
                    acc["cherry|scr"].append(des.perm_p(y, des.bases["star+scriptsplits"], cherries(gl), n_perm, rs)[1])
                row = dict(n=name, world=wname, c=c, f_mean=float(np.mean(fs)), dim=dim, reps=reps,
                           **{f"power_{k}": float(np.mean(np.array(v) < 0.05)) for k, v in acc.items()})
                out.append(row)
                print(row, f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(os.path.join(HERE, f"synth30c_d{dim}.json"), "w"), indent=1)


if __name__ == "__main__":
    run(dim=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
