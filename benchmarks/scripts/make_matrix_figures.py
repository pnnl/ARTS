#!/usr/bin/env python3
"""make_matrix_figures.py -- full-matrix strong-scaling figures.

Renders one figure set per app-placement variant (default / optimized) from
performance_harness strong runs:

  (a) <variant>_abs_scaling.png   -- per-app absolute e2e over 1/2/4 nodes,
      one line per runtime series.  TIMEOUT cells sit at the wall budget with
      an open marker (true e2e is above the budget line).
  (b) <variant>_speedup.png       -- per-app strong speedup t(1n)/t(Nn) with
      the y=x ideal.  Cells whose 1n or Nn median is censored are open.
  (c) ab_default_vs_optimized.png -- per-app, per-runtime ratio
      default/optimized at each node count (>1 = the optimized variant is
      faster), once both variants' data exist.

The VAL family appears twice when a nocomb run is supplied: the plain build
(combining mitigation compiled out -- validation in its bare form, dashed) and
the +comb build (solid).  Palette/labels come from make_scaling_figures
(single source of truth; do not restyle here).

Usage:
  make_matrix_figures.py --out <dir> \
      --opt-all8 <dir> [--opt-valnocomb <dir>] \
      [--def-all8 <dir>] [--def-valnocomb <dir>]
"""
from __future__ import annotations
import argparse, csv, math, statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from make_scaling_figures import RT_COLOR, RT_LABEL, HINTED_APPS, INK, MUTED, HINT_INK, _style

BUDGET = 60.0
GEOS = [("1n_sc", 1), ("2n_sc", 2), ("4n_sc", 4)]

# Series = (key, source, runtime-in-csv, linestyle).  The nocomb VAL pair maps
# onto the same family colors, dashed, so combining reads as a style axis.
SERIES = [
    ("ocr_val_wt_pure",  "valnocomb", "ocr_val_wt",      "--"),
    ("ocr_val_wt",       "all8",      "ocr_val_wt",      "-"),
    ("ocr_val_wb_pure",  "valnocomb", "ocr_val_wb",      "--"),
    ("ocr_val_wb",       "all8",      "ocr_val_wb",      "-"),
    ("ocr_inv_wt",       "all8",      "ocr_inv_wt",      "-"),
    ("ocr_inv_wb",       "all8",      "ocr_inv_wb",      "-"),
    ("ocr_excl_purge",   "all8",      "ocr_excl_purge",  "-"),
    ("ocr_excl_retain",  "all8",      "ocr_excl_retain", "-"),
    ("xsocr",            "all8",      "xsocr",           "-"),
]
# ocr-vx is excluded from the QUICK-LOOK sweep only (not permanently): ARTS
# INV-RETAIN covers the same protocol family more strongly here, and ocr-vx's
# pervasive budget timeouts dominate wall-clock without adding signal.  Its
# measured data stays on disk; re-add the series line below to restore it:
#   ("ocrvx",            "all8",      "ocrvx",           "-"),
S_COLOR = {k: RT_COLOR[rt] for k, _, rt, _ in SERIES}
S_LABEL = {k: (RT_LABEL[rt] + (" (no comb)" if k.endswith("_pure") else
               (" +comb" if rt.startswith("ocr_val") else "")))
           for k, _, rt, _ in SERIES}
S_STYLE = {k: ls for k, _, _, ls in SERIES}


def load(dir_: str | None):
    if not dir_:
        return []
    p = Path(dir_) / "results.csv"
    return list(csv.DictReader(open(p, newline=""))) if p.exists() else []


def aggregate(rows_by_src):
    """(bench, geo, series_key) -> {med, lo, hi, censored} over OK+TIMEOUT rows."""
    A = {}
    for key, src, rt, _ in SERIES:
        for r in rows_by_src.get(src, []):
            if r["runtime"] != rt or r["config"] not in dict(GEOS):
                continue
            st_ = r["status"]
            if st_ == "OK":
                w = float(r["e2e_ns"]) / 1e9 if r["e2e_ns"] else float(r["wall"])
                cen = False
            elif st_ == "TIMEOUT":
                w, cen = BUDGET, True
            else:
                continue
            A.setdefault((r["bench"], r["config"], key), []).append((w, cen))
    out = {}
    for k, samples in A.items():
        ws = sorted(w for w, _ in samples)
        out[k] = {"med": ws[len(ws) // 2], "lo": ws[0], "hi": ws[-1],
                  "censored": any(c for _, c in samples)}
    return out


def benches_of(A):
    return sorted({b for b, _, _ in A})


def series_present(A):
    have = {k for _, _, k in A}
    return [s[0] for s in SERIES if s[0] in have]


def legend_handles(keys, budget_line=True):
    hs = [Line2D([0], [0], color=S_COLOR[k], ls=S_STYLE[k], lw=2, label=S_LABEL[k])
          for k in keys]
    if budget_line:
        hs.append(Line2D([0], [0], ls=":", color="#c0392b", label=f"wall budget ({BUDGET:.0f}s)"))
        hs.append(Line2D([0], [0], marker="o", color=MUTED, markerfacecolor="none", lw=0,
                         label="open marker = TIMEOUT (true e2e above budget)"))
        hs.append(Line2D([0], [0], marker="^", color=MUTED, markerfacecolor="none", lw=0,
                         label="^ at top edge = off-axis"))
    return hs


def fig_abs_scaling(A, title, path):
    bs, keys = benches_of(A), series_present(A)
    if not bs:
        return
    ncol = 4
    nrow = (len(bs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for idx, b in enumerate(bs):
        ax = axes[idx // ncol][idx % ncol]
        oky = [A[(b, g, k)]["med"] for g, _ in GEOS for k in keys
               if (b, g, k) in A and not A[(b, g, k)]["censored"]]
        if oky:
            med = st.median(oky)
            core = [y for y in oky if y <= 5 * med]
            ymax = 1.2 * max(core)
        else:
            ymax = BUDGET * 1.08
        ax.set_ylim(0, ymax)
        for k in keys:
            xs, ys, cens = [], [], []
            for g, nn in GEOS:
                c = A.get((b, g, k))
                if not c:
                    continue
                xs.append(nn); ys.append(c["med"]); cens.append(c["censored"])
            if not xs:
                continue
            ax.plot(xs, ys, S_STYLE[k], color=S_COLOR[k], lw=1.4, zorder=3, clip_on=True)
            for x, y, cn in zip(xs, ys, cens):
                if y > ymax:
                    ax.plot(x, ymax * 0.965, marker="^", ms=5, color=S_COLOR[k],
                            markerfacecolor=("none" if cn else S_COLOR[k]), lw=0, zorder=5)
                else:
                    ax.plot(x, y, "o", ms=4, color=S_COLOR[k],
                            markerfacecolor=("none" if cn else S_COLOR[k]), zorder=4)
        if BUDGET <= ymax:
            ax.axhline(BUDGET, color="#c0392b", lw=0.6, ls=":")
        hint = b in HINTED_APPS
        ax.set_title(b, fontsize=8, color=HINT_INK if hint else INK,
                     weight="bold" if hint else "normal")
        ax.set_xticks([1, 2, 4]); ax.set_xticklabels(["1", "2", "4"], fontsize=7)
        ax.tick_params(labelsize=7)
        _style(ax, axis="y")
    for j in range(len(bs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title, y=1.005, fontsize=12, weight="bold", color=INK, x=0.02, ha="left")
    fig.legend(handles=legend_handles(keys), fontsize=8, frameon=False,
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def fig_speedup(A, title, path):
    bs, keys = benches_of(A), series_present(A)
    if not bs:
        return
    ncol = 4
    nrow = (len(bs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for idx, b in enumerate(bs):
        ax = axes[idx // ncol][idx % ncol]
        ax.plot([1, 4], [1, 4], color=GRID_IDEAL, lw=1.0, ls="--", zorder=1)
        top = 1.0
        for k in keys:
            base = A.get((b, "1n_sc", k))
            if not base:
                continue
            xs, ys, cens = [], [], []
            for g, nn in GEOS:
                c = A.get((b, g, k))
                if not c:
                    continue
                sp = base["med"] / c["med"] if c["med"] > 0 else 0
                xs.append(nn); ys.append(sp)
                cens.append(c["censored"] or base["censored"])
            if not xs:
                continue
            top = max(top, max(ys))
            ax.plot(xs, ys, S_STYLE[k], color=S_COLOR[k], lw=1.4, zorder=3)
            for x, y, cn in zip(xs, ys, cens):
                ax.plot(x, y, "o", ms=4, color=S_COLOR[k],
                        markerfacecolor=("none" if cn else S_COLOR[k]), zorder=4)
        ax.set_ylim(0, min(max(4.4, 1.15 * top), 8))
        hint = b in HINTED_APPS
        ax.set_title(b, fontsize=8, color=HINT_INK if hint else INK,
                     weight="bold" if hint else "normal")
        ax.set_xticks([1, 2, 4]); ax.set_xticklabels(["1", "2", "4"], fontsize=7)
        ax.tick_params(labelsize=7)
        _style(ax, axis="y")
    for j in range(len(bs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    hs = legend_handles(keys, budget_line=False)
    hs.append(Line2D([0], [0], ls="--", color=GRID_IDEAL, label="ideal (y=x)"))
    hs.append(Line2D([0], [0], marker="o", color=MUTED, markerfacecolor="none", lw=0,
                     label="open marker = a censored median in the ratio"))
    fig.suptitle(title, y=1.005, fontsize=12, weight="bold", color=INK, x=0.02, ha="left")
    fig.legend(handles=hs, fontsize=8, frameon=False, loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


GRID_IDEAL = "#9ca3af"


def fig_variant_ratio(Ad, Ao, title, path):
    """default/optimized ratio per app x runtime x geometry (>1 = optimized wins)."""
    keys = [k for k in series_present(Ao) if k in {s[0] for s in SERIES}]
    bs = sorted(set(benches_of(Ad)) & set(benches_of(Ao)))
    if not bs or not keys:
        return
    ncol = 4
    nrow = (len(bs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for idx, b in enumerate(bs):
        ax = axes[idx // ncol][idx % ncol]
        ax.axhline(1.0, color=GRID_IDEAL, lw=1.0, ls="--", zorder=1)
        for k in keys:
            xs, ys, cens = [], [], []
            for g, nn in GEOS:
                cd, co = Ad.get((b, g, k)), Ao.get((b, g, k))
                if not cd or not co or co["med"] <= 0:
                    continue
                xs.append(nn); ys.append(cd["med"] / co["med"])
                cens.append(cd["censored"] or co["censored"])
            if not xs:
                continue
            ax.plot(xs, ys, S_STYLE[k], color=S_COLOR[k], lw=1.4, zorder=3)
            for x, y, cn in zip(xs, ys, cens):
                ax.plot(x, y, "o", ms=4, color=S_COLOR[k],
                        markerfacecolor=("none" if cn else S_COLOR[k]), zorder=4)
        ax.set_yscale("log")
        hint = b in HINTED_APPS
        ax.set_title(b, fontsize=8, color=HINT_INK if hint else INK,
                     weight="bold" if hint else "normal")
        ax.set_xticks([1, 2, 4]); ax.set_xticklabels(["1", "2", "4"], fontsize=7)
        ax.tick_params(labelsize=7)
        _style(ax, axis="y")
    for j in range(len(bs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    hs = legend_handles(keys, budget_line=False)
    hs.append(Line2D([0], [0], ls="--", color=GRID_IDEAL, label="parity (default == optimized)"))
    fig.suptitle(title, y=1.005, fontsize=12, weight="bold", color=INK, x=0.02, ha="left")
    fig.legend(handles=hs, fontsize=8, frameon=False, loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--opt-all8"); ap.add_argument("--opt-valnocomb")
    ap.add_argument("--def-all8"); ap.add_argument("--def-valnocomb")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    Ao = aggregate({"all8": load(a.opt_all8), "valnocomb": load(a.opt_valnocomb)})
    Ad = aggregate({"all8": load(getattr(a, "def_all8")),
                    "valnocomb": load(getattr(a, "def_valnocomb"))})
    if Ao:
        fig_abs_scaling(Ao, "strong scaling, e2e seconds -- OPTIMIZED app variant (cbgpu02, 12w/node)",
                        out / "optimized_abs_scaling.png")
        fig_speedup(Ao, "strong speedup t(1n)/t(Nn) -- OPTIMIZED app variant",
                    out / "optimized_speedup.png")
    if Ad:
        fig_abs_scaling(Ad, "strong scaling, e2e seconds -- DEFAULT (as-born) app variant (cbgpu02, 12w/node)",
                        out / "default_abs_scaling.png")
        fig_speedup(Ad, "strong speedup t(1n)/t(Nn) -- DEFAULT (as-born) app variant",
                    out / "default_speedup.png")
    if Ao and Ad:
        fig_variant_ratio(Ad, Ao, "default / optimized e2e ratio (log; >1 = optimized faster)",
                          out / "ab_default_vs_optimized.png")
    print(f"[figures] wrote {sum(1 for _ in out.glob('*.png'))} png(s) -> {out}")


if __name__ == "__main__":
    main()
