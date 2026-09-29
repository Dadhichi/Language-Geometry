"""Collect res_*.json -> tables.md (by layer)."""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for f in glob.glob(os.path.join(HERE, "res_*.json")):
    R.update(json.load(open(f)))


def key(k):
    l, v = k.split("_")
    return (v, int(l[1:]))


out = []
for variant in ("raw", "sqrtn"):
    ks = sorted([k for k in R if k.endswith(variant)], key=key)
    if not ks:
        continue
    out.append(f"\n## {variant}: geometry and single-instrument pull (g fitted on dev / refit on devtest; rho held-out; "
               f"z vs sentence-shuffle / target-permutation nulls)\n")
    out.append("| L | disp/edge | e in span | LDA acc | own w p5-p95 | e_tokc g | rho | z_sh/z_jp | B_bag g | rho | z_sh/z_jp "
               "| B_pre g | rho | z_sh/z_jp | e_tokc_LL g | z_sh |")
    out.append("|" + "---|" * 16)
    for k in ks:
        r = R[k]; I = r["instr"]
        row = [k.split("_")[0][1:], f"{r['disp_over_edge']:.3f}", f"{r['e_inspan']:.3f}", f"{r['lda_acc']:.3f}",
               f"{r['own_w_p05']:.2f}-{r['own_w_p95']:.2f}"]
        for nm in ("e_tokc", "B_bag", "B_pre"):
            x = I[nm]
            row += [f"{x['g_dev'][0]:.3f}/{x['g_test'][0]:.3f}", f"{x['rho'][0]:.3f}", f"{x['z_shuf']:.0f}/{x['z_jperm']:.0f}"]
        x = I["e_tokc_LL"]
        row += [f"{x['g_dev'][0]:.3f}/{x['g_test'][0]:.3f}", f"{x['z_shuf']:.1f}"]
        out.append("| " + " | ".join(row) + " |")
    out.append(f"\n### {variant}: joint attribution (dev g; devtest g) and pair groups (dev g, devtest rho)\n")
    out.append("| L | bag, pre | early, late, pre | first, pre | e_tokc, e_tokc_LL | B_bag grp LatLat/zj/rest (g) | "
               "B_pre grp LatLat/zj/rest (g) | no-cov g e/bag/pre |")
    out.append("|" + "---|" * 8)
    f2 = lambda a: ",".join(f"{v:.2f}" for v in a)
    for k in ks:
        r = R[k]; I = r["instr"]
        row = [k.split("_")[0][1:]]
        for s in ("B_bag+B_pre", "B_early+B_late+B_pre", "B_first+B_pre", "e_tokc+e_tokc_LL"):
            row.append(f"{f2(I[s]['g_dev'])}; {f2(I[s]['g_test'])}")
        for nm in ("B_bag", "B_pre"):
            G = r["groups"][nm]
            row.append("/".join(f"{G[g]['g_dev']:.2f}" for g in ("LatLat", "zho-jpn", "rest")))
        nc = r["instr_nocov"]
        row.append("/".join(f"{nc[s]['g_dev'][0]:.2f}" for s in ("e_tokc", "B_bag", "B_pre")))
        out.append("| " + " | ".join(row) + " |")
    out.append(f"\n### {variant}: script-level instruments (partial r, controls log ntok/punct/digit; dev / devtest) and "
               f"surface sharing\n")
    out.append("| L | jpn kanji->zho | kanji->eng ctrl | zho shared-char->jpn | nonLatin Latin-frac->Latin face | "
               "surface cos~overlap all/LatLat |")
    out.append("|" + "---|" * 6)
    for k in ks:
        r = R[k]; a, b = r["script_dev"], r["script_test"]
        row = [k.split("_")[0][1:]]
        for s in ("jpn_kanji->zho", "jpn_kanji->eng_ctrl", "zho_sharedchar->jpn", "nonLatin_latinfrac->LatinFace"):
            row.append(f"{a[s]:.2f} / {b[s]:.2f}")
        row.append(f"{r['surface_sharing_test'][0]:.3f} / {r['surface_sharing_test'][1]:.3f}")
        out.append("| " + " | ".join(row) + " |")
    out.append(f"\n### {variant}: packing (Spearman Mantel of -centroid distance vs text similarity; p; partial|script r, p)\n")
    out.append("| L | euclid~tok_hist | euclid~B_bag | euclid~e_tokc | euclid~script | mahal~tok_hist | mahal~B_bag | "
               "mahal~script | nearest |")
    out.append("|" + "---|" * 9)
    for k in ks:
        r = R[k]; p = r["packing"]
        f4 = lambda t: f"{t[0]:.2f} ({t[1]:.3f}); {t[2]:.2f} ({t[3]:.3f})"
        f22 = lambda t: f"{t[0]:.2f} ({t[1]:.3f})"
        nn = ",".join(f"{a}>{b}" for a, b in r["nearest"].items())
        out.append("| " + " | ".join([k.split("_")[0][1:], f4(p["euclid~tok_hist_int"]), f4(p["euclid~B_bag"]),
                                      f4(p["euclid~e_tokc"]), f22(p["euclid~same_script"]),
                                      f4(p["mahal~tok_hist_int"]), f4(p["mahal~B_bag"]), f22(p["mahal~same_script"]),
                                      nn]) + " |")
open(os.path.join(HERE, "tables.md"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
