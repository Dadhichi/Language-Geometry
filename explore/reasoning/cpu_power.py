"""Power calculations for the reasoning-gap proposal (PROPOSAL.md, section 'Measurement').  CPU only, seconds.

1. Paired accuracy (McNemar / exact sign test on discordant pairs) for an intervention on the same problems:
   n problems per language (MGSM: 250), baseline accuracy p0, net gain g, discordance rate q (share of problems that
   change correctness in either direction). Simulated power at alpha = 0.05 two-sided, per language and pooled.
2. Cross-language correlation: smallest |r| detectable with 80% power at alpha 0.05 (two-sided, Fisher z), for the
   number of languages a benchmark gives (MGSM-overlap 8, MGSM 10, PolyMath-overlap 13, 17, 34).
3. Per-problem failure prediction: logistic regression of failure on a standardised divergence score; simulated power
   to detect an odds ratio per SD (1.3, 1.5, 2.0) with n problems (250 one language, 2,500 pooled with language
   fixed effects approximated by centring).
4. Teacher-forced log-likelihood (continuous, paired): detectable standardised mean shift d = (z_a + z_b)/sqrt(n).
5. Random-direction null: with R random directions the smallest empirical p is 1/(R+1).
"""
import numpy as np
from scipy.stats import norm, binomtest

rng = np.random.default_rng(0)
ALPHA = 0.05


def mcnemar_power(n, p0, gain, q, sims=4000):
    """q = discordance; among discordant, up = (q + gain)/2, down = (q - gain)/2."""
    up, down = (q + gain) / 2, (q - gain) / 2
    if down < 0:
        return np.nan
    hits = 0
    for _ in range(sims):
        u = rng.binomial(n, up)
        d = rng.binomial(n - u, down / (1 - up))
        m = u + d
        if m == 0:
            continue
        # exact two-sided binomial test on discordant pairs
        if binomtest(u, m, 0.5).pvalue < ALPHA:
            hits += (u > d)
    return hits / sims


def r_detectable(n, power=0.8):
    z = norm.ppf(1 - ALPHA / 2) + norm.ppf(power)
    return np.tanh(z / np.sqrt(n - 3))


def logistic_power(n, base_fail, or_per_sd, sims=600):
    import warnings
    from scipy.optimize import minimize
    hits = 0
    b1 = np.log(or_per_sd)
    b0 = np.log(base_fail / (1 - base_fail))
    for _ in range(sims):
        x = rng.standard_normal(n)
        p = 1 / (1 + np.exp(-(b0 + b1 * x)))
        y = rng.random(n) < p
        X = np.column_stack([np.ones(n), x])

        def nll(b):
            eta = X @ b
            return np.sum(np.logaddexp(0, eta) - y * eta)

        def grad(b):
            eta = X @ b
            return X.T @ (1 / (1 + np.exp(-eta)) - y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(nll, np.zeros(2), jac=grad, method="BFGS")
        b = res.x
        eta = X @ b
        w = 1 / (1 + np.exp(-eta))
        w = w * (1 - w)
        cov = np.linalg.inv(X.T @ (X * w[:, None]))
        z = b[1] / np.sqrt(cov[1, 1])
        hits += (z > norm.ppf(1 - ALPHA / 2))
    return hits / sims


def main():
    print("1. Paired accuracy, exact McNemar, alpha .05 two-sided (power)")
    print("   n     p0   gain  discord  power")
    for n in (250, 500, 2500):
        for gain in (0.03, 0.05, 0.10):
            for q in (0.15, 0.25):
                print(f"   {n:5d} 0.40 {gain:5.2f} {q:7.2f}  {mcnemar_power(n, 0.4, gain, q, sims=2000 if n > 500 else 4000):.2f}")
    print("\n2. Smallest |r| detectable across languages (80% power, alpha .05 two-sided)")
    for nl in (8, 10, 13, 17, 34):
        print(f"   n_lang = {nl:2d}: |r| >= {r_detectable(nl):.2f}")
    print("\n3. Per-problem logistic regression of failure on a standardised divergence score (power)")
    for n in (250, 2500):
        for orsd in (1.3, 1.5, 2.0):
            print(f"   n = {n:5d}, base failure 0.5, OR/SD {orsd}: power {logistic_power(n, 0.5, orsd, sims=300 if n > 500 else 600):.2f}")
    print("\n4. Paired continuous outcome (teacher-forced log-lik): detectable standardised shift d (80% power)")
    for n in (100, 250, 2500):
        print(f"   n = {n:5d}: d >= {(norm.ppf(1 - ALPHA / 2) + norm.ppf(0.8)) / np.sqrt(n):.3f}")
    print("\n5. Random-direction null: R = 24 -> min p 0.040; R = 32 -> 0.030; R = 99 -> 0.010")


if __name__ == "__main__":
    main()
