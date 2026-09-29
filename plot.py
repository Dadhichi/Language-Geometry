#!/usr/bin/env python
"""
plot.py -- depth curves and pivot heatmap from one fit.py results directory.

    python plot.py results/qwen25_7b_mean [--split devtest] [--pivot_k 128]

Writes {dir}/depth_{split}.png and {dir}/pivots_{split}.png.

depth: one panel per statistic, x = layer, one line per k (solid = real, dashed = consistent null in the
same colour).  The composition panels show c / null_max, so 1 is the null edge.
pivots: layer x pivot language.  Shows pivot_X - null_pivot_X when the null columns exist (the excess
over what estimation noise alone gives that pivot); otherwise the raw delta, which is all < 0 and partly
tracks per-language estimation precision.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # first three categorical slots (validated all-pairs)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#1f5fae", "#2a78d6", "#f0efec", "#e34948", "#b3302f"])

# (column, null column or None, title, reference line or None)
PANELS = [
    ("rho_procrustes", "null_rho_procrustes", "Procrustes residual (rho)", None),
    ("rho_ridge", None, "Ridge residual (rho)", None),
    ("p1_procrustes", None, "Procrustes retrieval P@1", None),
    ("c_proc_ratio", None, "Procrustes composition c / null max", 1.0),
    ("c_ridge_ratio", None, "Ridge composition c / null max", 1.0),
    ("cocycle", "null_cocycle_min", "Cocycle (1 = consistent)", None),
]


def style(ax, title):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def depth(s, split, out, name):
    s = s.assign(c_proc_ratio=s.c_proc_mean / s.null_c_proc_max, c_ridge_ratio=s.c_ridge_mean / s.null_c_ridge_max)
    ks = list(dict.fromkeys(s.k.astype(str)))[: len(SERIES)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURFACE)
    for ax, (col, ncol, title, ref) in zip(axes.flat, PANELS):
        style(ax, title)
        if ref is not None:
            ax.axhline(ref, color=INK2, linewidth=1, linestyle=":")
        for color, k in zip(SERIES, ks):
            t = s[s.k.astype(str) == k].sort_values("layer")
            ax.plot(t.layer, t[col], color=color, linewidth=2, label=f"k={k}")
            if ncol and ncol in t:
                ax.plot(t.layer, t[ncol], color=color, linewidth=1.5, linestyle="--", alpha=0.8)
        ax.set_xlabel("layer", fontsize=8, color=INK2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK2, linewidth=1.5, linestyle="--"))
    labels.append("consistent null")
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(f"{name} - {split}", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(out, f"depth_{split}.png")
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def pivots(s, split, k, out, name):
    t = s[s.k.astype(str) == str(k)].sort_values("layer").set_index("layer")
    piv = [c for c in t.columns if c.startswith("pivot_")]
    has_null = all("null_" + c in t for c in piv)
    M = pd.DataFrame({c[6:]: t[c] - t["null_" + c] if has_null else t[c] for c in piv})
    M = M[M.mean().sort_values(ascending=False).index]          # best pivot (on average) first
    lim = float(np.nanmax(np.abs(M.values)))
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * M.shape[1], 0.8 + 0.22 * M.shape[0]), facecolor=SURFACE)
    im = ax.imshow(M.values, aspect="auto", cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim))
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=45, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(M.shape[0]), M.index, fontsize=7, color=INK2)
    ax.set_ylabel("layer", fontsize=8, color=INK2)
    for side in ax.spines.values():
        side.set_visible(False)
    what = "pivot delta - null (excess)" if has_null else "raw pivot delta (no null in this run)"
    ax.set_title(f"{name} - {split}, k={k}: {what}\n> 0: routing through the pivot beats direct (red), < 0: hurts (blue)",
                 fontsize=9, color=INK, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    path = os.path.join(out, f"pivots_{split}.png")
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="a fit.py --out directory containing summary.csv")
    ap.add_argument("--split", default=None, help="test split (default: every split in summary.csv)")
    ap.add_argument("--pivot_k", default=None, help="k for the pivot heatmap (default: the first k)")
    args = ap.parse_args()
    s = pd.read_csv(os.path.join(args.results, "summary.csv"))
    name = os.path.basename(os.path.normpath(args.results))
    for split in [args.split] if args.split else list(dict.fromkeys(s.split)):
        ss = s[s.split == split]
        k = args.pivot_k or str(ss.k.iloc[0])
        print(depth(ss, split, args.results, name))
        print(pivots(ss, split, k, args.results, name))


if __name__ == "__main__":
    main()
