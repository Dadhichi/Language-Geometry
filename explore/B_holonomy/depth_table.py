"""depth_table.py -- compact cross-layer table from out/summary_*.json + cross-layer overlap of the ambient
excess-curvature subspace (real minus O(k) null, top-8 eigenvectors)."""
import glob
import json
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
rows = []
for f in sorted(glob.glob(os.path.join(OUT, "sum_*.json"))):
    for o in json.load(open(f)):
        rows.append(o)
seen = set()
print(f"{'L':>3} {'k':>4} {'tag':>7} | {'dataset':8s} {'c_proc':>7} {'c_top1/2':>8} {'q4frac':>6} {'c_ridge':>7} {'rho':>6} "
      f"{'thRMS':>6} {'thMax':>6} {'PRpl':>5} {'comm':>5} {'ncc>.5':>6} {'tricorr':>7}")
for o in sorted(rows, key=lambda o: (o["tag"], o["k"], o["layer"])):
    key = (o["layer"], o["k"], o["tag"])
    if key in seen:
        continue
    seen.add(key)
    for nm, r in o["rows"].items():
        tc = o["tri_corr"].get(nm, np.nan)
        print(f"{o['layer']:>3} {o['k']:>4} {o['tag']:>7} | {nm:8s} {r['c_proc']:7.4f} {r['c_tophalf']:8.4f} "
              f"{r['prof_quart_frac'][3]:6.3f} {r['c_ridge']:7.4f} {r['rho']:6.3f} {r['theta_rms_deg']:6.2f} "
              f"{r['theta_max_deg']:6.1f} {r['pr_planes']:5.1f} {r['comm_unw']:5.3f} {r.get('n_cc_gt05', np.nan):6.1f} {tc:7.2f}")
    print()

# cross-layer overlap of excess subspaces (vs O(k) null), k=64, dev fit
V = {}
for f in glob.glob(os.path.join(OUT, "L*_k64_dev_excessV_O64r0.npy")):
    L = int(os.path.basename(f).split("_")[0][1:])
    V[L] = np.load(f)[:, :8].astype(np.float64)
Ls = sorted(V)
if Ls:
    print("cross-layer overlap (mean cos^2, top-8 excess eigvecs vs O64 null), chance ~ 8/3584 = 0.002")
    print("     " + "".join(f"{b:>7d}" for b in Ls))
    for a in Ls:
        print(f"{a:>4d} " + "".join(f"{((V[a].T @ V[b]) ** 2).sum() / 8:7.3f}" for b in Ls))
Vt = {}
for tag in ("dev", "devtest"):
    f = os.path.join(OUT, f"L14_k64_{tag}_excessV_O64r0.npy")
    if os.path.exists(f):
        Vt[tag] = np.load(f).astype(np.float64)
if len(Vt) == 2:
    for r in (2, 4, 8, 16):
        print(f"split replication dev vs devtest fit, top-{r}: {((Vt['dev'][:, :r].T @ Vt['devtest'][:, :r]) ** 2).sum() / r:.3f}")
