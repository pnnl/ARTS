#!/usr/bin/env python3
"""make_figures.py — publication-quality figures for the coherence-protocol perf
matrix (6 apps x {fix,strong,weak} x 9 runtimes).

Inputs : <matrix_dir>/results.csv  +  <matrix_dir>/metrics.json
Outputs: <matrix_dir>/analysis/figures/*.pdf and *.png  (300 dpi)

Figure groups:
  A. e2e scaling        (strong / weak / fix, one 2x3 app panel each)
  B. strong speedup     (vs ideal linear)
  C. protocol compare   (eager-vs-lazy penalty; arts vs xsocr/ocrvx)
  D. network comm       (remote bytes & sends vs nodes; eager-vs-lazy traffic;
                         comm-intensity; e2e-vs-bytes correlation)
  E. load balance       (per-rank EDT_FINISH imbalance)
  F. summary heatmaps   (apps x runtimes, per experiment)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------- style ----
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.titlesize": 11.5, "axes.labelsize": 10.5, "axes.linewidth": 0.8,
    "legend.fontsize": 8.0, "legend.frameon": False,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "lines.linewidth": 1.7, "lines.markersize": 5.5,
    "axes.prop_cycle": plt.cycler(color=["#333"]),
})

# runtime -> (color, linestyle, marker, display label). Protocol family = color;
# lazy = solid, eager = dashed; WRF_RCU = dash-dot; references = dotted.
RT_STYLE = {
    "ocr_rcu_lazy":  ("#1f77b4", "-",  "o", "RCU-lazy"),
    "ocr_rcu_eager": ("#1f77b4", "--", "o", "RCU-eager"),
    "ocr_rwlock_lazy":   ("#d62728", "-",  "^", "RWLOCK-lazy"),
    "ocr_rwlock_eager":  ("#d62728", "--", "^", "RWLOCK-eager"),
    "wrf_rcu_eager":        ("#9467bd", "-.", "D", "WRF_RCU"),
    "xsocr":       ("#000000", ":",  "x", "xsocr"),
    "ocrvx":       ("#7f7f7f", ":",  "+", "ocr-vx"),
}
RT_ORDER = list(RT_STYLE)
APP_LABEL = {
    "graph500": "graph500 (BFS)",
    "hpcg_intel": "HPCG (CG)",
    "nekbone": "Nekbone (SEM-CG)",
    "RSBench_intel_sharedDB": "RSBench (MC)",
    "Stencil2D_intel_chandra": "Stencil2D (PRK)",
    "hpgmg": "HPGMG (multigrid)",
}
APP_ORDER = ["graph500", "RSBench_intel_sharedDB", "Stencil2D_intel_chandra",
             "hpcg_intel", "nekbone", "hpgmg"]
SC_NODES = {"1n_sc": 1, "2n_sc": 2, "4n_sc": 4, "8n_sc": 8}
FIX_ORDER = ["1n", "2", "4", "8", "16", "2n_io", "4n_io", "8n_io"]
FIX_X = {c: i for i, c in enumerate(FIX_ORDER)}
CAP = 300.0  # timeout wall (s)


def _num(v):
    if isinstance(v, dict):
        return v.get("value", v.get("value_ms", 0))
    return v or 0


def load(md):
    md = Path(md)
    rows = list(csv.DictReader(open(md / "results.csv")))
    metrics = json.loads((md / "metrics.json").read_text())
    # E[(exp,bench,cfg,rt)] = e2e_s (None if FAIL)
    E, ST = {}, {}
    for r in rows:
        k = (r["experiment"], r["bench"], r["config"], r["runtime"])
        ST[k] = r["status"]
        E[k] = (float(r["e2e_ns"]) / 1e9) if (r["status"] != "FAIL" and r["e2e_ns"]) else None
    # M[(exp,bench,cfg,rt)] = {"bytes":, "sends":, "fins":[per-rank], "e2e":}
    M = {}
    for m in metrics:
        pr = m.get("per_rank", {})
        M[(m["experiment"], m["bench"], m["config"], m["runtime"])] = {
            "bytes": sum(_num(pr[r].get("BYTES_REMOTE_SENT")) for r in pr),
            "recv": sum(_num(pr[r].get("BYTES_REMOTE_RECEIVED")) for r in pr),
            "sends": sum(_num(pr[r].get("NUM_REMOTE_SEND")) for r in pr),
            "fins": [_num(pr[r].get("NUM_EDT_FINISH")) for r in pr],
            "e2e": (m.get("e2e_ns") or 0) / 1e9,
        }
    return E, ST, M


def savefig(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print("  wrote", name)


def _timeout_bar(ax, x, w, color):
    """Draw a 'broken' bar that runs off the top of the axis with a wavy (~) cap and
    a vertical 'timeout' label — the paper convention for a value that did not
    complete (ran past the wall-clock cap)."""
    top = ax.get_ylim()[1]
    ax.bar(x, top, w, color=color, alpha=0.28, hatch="////", edgecolor=color,
           linewidth=0.9, zorder=1)
    xx = np.linspace(x - w / 2, x + w / 2, 80)
    yb = top * 0.895
    wave = yb + (top * 0.02) * np.sin((xx - (x - w / 2)) / w * 5 * np.pi)
    ax.plot(xx, wave, color="white", lw=3.4, solid_capstyle="round", zorder=3)
    ax.plot(xx, wave, color=color, lw=1.2, zorder=4)
    ax.text(x, top * 0.46, "timeout", rotation=90, ha="center", va="center",
            fontsize=7.2, color=color, fontweight="bold", zorder=5)


def _mark_timeouts_on_lines(ax, tos, speedup=False):
    """After the real lines are drawn, put each timed-out point on the TOP edge of
    the axis (a colored caret 'escaping' the frame) + one italic 'timeout' note."""
    if not tos:
        return
    ax.autoscale(enable=True, axis="y")
    lo, hi = ax.get_ylim()
    for tx, col in tos:
        ax.plot([tx], [hi], marker="^", color=col, ms=8, mec="white", mew=0.6,
                clip_on=False, zorder=8)
    ax.annotate("△ = timeout", xy=(0.03, 0.965), xycoords="axes fraction",
                fontsize=7, style="italic", color="0.30", va="top")


# ---------------------------------------------------- A. e2e scaling ----
def fig_scaling(E, ST, outdir, exp, ylabel="end-to-end time (s)", speedup=False):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6))
    is_sc = exp in ("strong", "weak")
    xmap = SC_NODES if is_sc else FIX_X
    cfgs = list(SC_NODES) if is_sc else FIX_ORDER
    for ax, app in zip(axes.flat, APP_ORDER):
        tos = []                                     # (x, color) timed-out points
        for rt in RT_ORDER:
            col, ls, mk, lab = RT_STYLE[rt]
            xs, ys = [], []
            base = E.get((exp, app, cfgs[0], rt))
            for c in cfgs:
                k = (exp, app, c, rt)
                if k not in ST:
                    continue
                v = E.get(k)
                if v is None:                       # timeout -> escapes off the top
                    tos.append((xmap[c], col))
                    continue
                y = (base / v) if (speedup and base) else v
                xs.append(xmap[c]); ys.append(y)
            if xs:
                ax.plot(xs, ys, color=col, ls=ls, marker=mk, label=lab,
                        markeredgewidth=0.6)
        _mark_timeouts_on_lines(ax, tos, speedup)
        if speedup:
            xi = [xmap[c] for c in cfgs]
            ax.plot(xi, [x / xi[0] for x in xi], color="0.4", ls=(0, (1, 1)),
                    lw=1.1, label="ideal")
        ax.set_title(APP_LABEL[app])
        if is_sc:
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(SC_NODES.values()))
            ax.set_xticklabels([str(v) for v in SC_NODES.values()])
            ax.set_xlabel("nodes (4 workers each)")
        else:
            ax.set_xticks(list(FIX_X.values()))
            ax.set_xticklabels(FIX_ORDER, rotation=45, ha="right", fontsize=7.5)
            ax.set_xlabel("config (48 cores total)")
        if not speedup:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylabel("speedup vs 1 node" if speedup else ylabel)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.045))
    ttl = {"strong": "Strong scaling", "weak": "Weak scaling",
           "fix": "Capacity (48-core) — distribution overhead"}[exp]
    if speedup:
        ttl = "Strong-scaling speedup (x = red = timeout)"
    fig.suptitle(ttl, y=1.10, fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, outdir, ("fig_speedup_strong" if speedup else f"fig_scaling_{exp}"))


# ---------------------------------------------- C. protocol compare ----
def fig_eager_lazy(E, outdir):
    """Eager/Lazy e2e ratio at a multinode config, per protocol family, per app.
    An eager cell that timed out has no finite ratio (eager/lazy -> infinity); it is
    drawn as a broken bar running off the top of the axis with a 'timeout' label."""
    cfg, exp = "2n_sc", "strong"
    fams = [("RCU", "ocr_rcu_eager", "ocr_rcu_lazy"),
            ("RWLOCK", "ocr_rwlock_eager", "ocr_rwlock_lazy")]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = np.arange(len(APP_ORDER)); w = 0.26
    R, real_max = {}, 1.0            # (i,j) -> ratio | None(timeout) | nan(missing)
    for i, (fam, e, l) in enumerate(fams):
        for j, app in enumerate(APP_ORDER):
            ve, vl = E.get((exp, app, cfg, e)), E.get((exp, app, cfg, l))
            if ve and vl:
                R[(i, j)] = ve / vl; real_max = max(real_max, ve / vl)
            elif vl and (exp, app, cfg, e) in E:      # eager ran but FAILed
                R[(i, j)] = None
            else:
                R[(i, j)] = np.nan
    ax.set_ylim(0, real_max * 1.28)
    for i, (fam, e, l) in enumerate(fams):
        col = RT_STYLE[l][0]
        ax.bar([], [], color=col, alpha=0.85, label=fam)   # legend proxy
        for j, app in enumerate(APP_ORDER):
            xx = x[j] + (i - 1) * w
            r = R[(i, j)]
            if r is None:
                _timeout_bar(ax, xx, w, col)
            elif not (isinstance(r, float) and np.isnan(r)):
                ax.bar(xx, r, w, color=col, alpha=0.85)
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([APP_LABEL[a] for a in APP_ORDER],
                                          rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("eager / lazy  e2e ratio")
    ax.set_title(f"EAGER writeback penalty vs LAZY  (strong @ {cfg}; >1 = eager slower; "
                 "broken bar = eager timed out)")
    ax.legend(title="protocol")
    fig.tight_layout(); savefig(fig, outdir, "fig_eager_vs_lazy")


def fig_arts_vs_ref(E, outdir):
    """Best arts-lazy vs xsocr vs ocr-vx at the strong baseline (1n_sc) and 8n_sc.
    A runtime that timed out is drawn as a broken bar off the top with 'timeout'."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    def best_lazy(app, cfg):
        ks = [("strong", app, cfg, rt) for rt in ("ocr_rcu_lazy", "ocr_rwlock_lazy")]
        vs = [E.get(k) for k in ks if k in E]
        done = [v for v in vs if v]
        return min(done) if done else (None if vs else np.nan)

    def ref(app, cfg, rt):
        k = ("strong", app, cfg, rt)
        if k not in E:
            return np.nan
        return E[k] if E[k] else None               # None = ran but timed out

    for ax, cfg in zip(axes, ("1n_sc", "8n_sc")):
        x = np.arange(len(APP_ORDER)); w = 0.26
        series = [("arts (best lazy)", "#1f77b4", [best_lazy(a, cfg) for a in APP_ORDER]),
                  ("xsocr", "#000000", [ref(a, cfg, "xsocr") for a in APP_ORDER]),
                  ("ocr-vx", "#7f7f7f", [ref(a, cfg, "ocrvx") for a in APP_ORDER])]
        real = [v for _, _, ys in series for v in ys
                if isinstance(v, (int, float)) and v and not np.isnan(v)]
        ax.set_ylim(0, (max(real) if real else 1) * 1.28)
        for i, (lab, col, ys) in enumerate(series):
            ax.bar([], [], color=col, alpha=0.85, label=lab)      # legend proxy
            for j, v in enumerate(ys):
                xx = x[j] + (i - 1) * w
                if v is None:
                    _timeout_bar(ax, xx, w, col)
                elif isinstance(v, (int, float)) and not np.isnan(v):
                    ax.bar(xx, v, w, color=col, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([APP_LABEL[a].split()[0] for a in APP_ORDER],
                                              rotation=25, ha="right", fontsize=8.5)
        ax.set_ylabel("end-to-end time (s)")
        ax.set_title(f"strong @ {cfg}")
        ax.legend()
    fig.suptitle("ARTS vs reference runtimes (lower is better; broken bar = timeout)",
                 y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout(); savefig(fig, outdir, "fig_arts_vs_reference")


# ----------------------------------------------- D. network comm ----
def fig_network(M, outdir, field, ylabel, name, unit=1.0):
    """field-vs-nodes (log y), 2x3 app panel, one line per runtime (that emits counters)."""
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6))
    exp = "strong"
    for ax, app in zip(axes.flat, APP_ORDER):
        for rt in RT_ORDER:
            if rt in ("xsocr", "ocrvx"):            # no per-rank counters for refs
                continue
            col, ls, mk, lab = RT_STYLE[rt]
            xs, ys = [], []
            for c, n in SC_NODES.items():
                d = M.get((exp, app, c, rt))
                if d and d[field] > 0:
                    xs.append(n); ys.append(d[field] / unit)
            if xs:
                ax.plot(xs, ys, color=col, ls=ls, marker=mk, label=lab,
                        markeredgewidth=0.6)
        ax.set_title(APP_LABEL[app]); ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(list(SC_NODES.values()))
        ax.set_xticklabels([str(v) for v in SC_NODES.values()])
        ax.set_xlabel("nodes"); ax.set_ylabel(ylabel)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Inter-node communication — {ylabel} (strong)", y=1.09,
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); savefig(fig, outdir, name)


def fig_comm_correlation(E, ST, M, outdir):
    """e2e vs total remote bytes, all arts multinode cells, colored by app."""
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    cmap = plt.get_cmap("tab10")
    for ai, app in enumerate(APP_ORDER):
        xs, ys = [], []
        for (exp, b, c, rt), d in M.items():
            if b != app or d["bytes"] <= 0:
                continue
            e = E.get((exp, b, c, rt))
            if e:
                xs.append(d["bytes"] / 1e6); ys.append(e)
        if xs:
            ax.scatter(xs, ys, s=22, color=cmap(ai), alpha=0.7,
                       label=APP_LABEL[app].split()[0], edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total inter-node bytes sent (MB)")
    ax.set_ylabel("end-to-end time (s)")
    ax.set_title("Communication volume drives runtime\n(all arts multinode cells)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); savefig(fig, outdir, "fig_comm_vs_e2e")


def fig_comm_intensity(M, E, outdir):
    """remote MB/s (bytes/e2e) per app per node count, ocr_rcu_lazy — comm pressure."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(APP_ORDER)); w = 0.2
    for i, (c, n) in enumerate(SC_NODES.items()):
        vals = []
        for app in APP_ORDER:
            d = M.get(("strong", app, c, "ocr_rcu_lazy"))
            e = E.get(("strong", app, c, "ocr_rcu_lazy"))
            vals.append((d["bytes"] / 1e6 / e) if (d and e and d["bytes"] > 0) else 0)
        ax.bar(x + (i - 1.5) * w, vals, w, label=f"{n}n", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([APP_LABEL[a].split()[0] for a in APP_ORDER],
                                          rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("inter-node MB / second")
    ax.set_title("Communication intensity (RCU-lazy, strong)")
    ax.legend(title="nodes", ncol=4)
    fig.tight_layout(); savefig(fig, outdir, "fig_comm_intensity")


# ------------------------------------------------ E. load balance ----
def fig_load_balance(M, outdir):
    """EDT_FINISH imbalance = max/mean across ranks, vs nodes, ocr_rcu_lazy strong."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for app in APP_ORDER:
        xs, ys = [], []
        for c, n in SC_NODES.items():
            d = M.get(("strong", app, c, "ocr_rcu_lazy"))
            if d and d["fins"] and n > 1:
                f = d["fins"]; mean = sum(f) / len(f)
                if mean > 0:
                    xs.append(n); ys.append(max(f) / mean)
        if xs:
            ax.plot(xs, ys, marker="o", label=APP_LABEL[app].split()[0])
    ax.axhline(1.0, color="0.4", ls="--", lw=1, label="perfect balance")
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8])
    ax.set_xticklabels(["2", "4", "8"])
    ax.set_xlabel("nodes"); ax.set_ylabel("EDT load imbalance (max/mean per rank)")
    ax.set_title("Work-distribution balance (RCU-lazy, strong)")
    ax.legend(ncol=2, fontsize=8.5)
    fig.tight_layout(); savefig(fig, outdir, "fig_load_balance")


# --------------------------------------------- F. summary heatmap ----
def fig_heatmap(E, ST, outdir, exp):
    cfg = {"strong": "8n_sc", "weak": "8n_sc", "fix": "16"}[exp]
    base_cfg = {"strong": "1n_sc", "weak": "1n_sc", "fix": "1n"}[exp]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    Z = np.full((len(APP_ORDER), len(RT_ORDER)), np.nan)
    for i, app in enumerate(APP_ORDER):
        for j, rt in enumerate(RT_ORDER):
            v = E.get((exp, app, cfg, rt)); b = E.get((exp, app, base_cfg, rt))
            if v and b:
                Z[i, j] = b / v                    # speedup base->this config
    im = ax.imshow(Z, cmap="RdYlGn", aspect="auto", vmin=0, vmax=8)
    for i in range(len(APP_ORDER)):
        for j in range(len(RT_ORDER)):
            k = (exp, APP_ORDER[i], cfg, RT_ORDER[j])
            txt = "TO" if ST.get(k) == "FAIL" else (f"{Z[i,j]:.1f}" if not np.isnan(Z[i, j]) else "-")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="black")
    ax.set_xticks(range(len(RT_ORDER)))
    ax.set_xticklabels([RT_STYLE[r][3] for r in RT_ORDER], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(APP_ORDER)))
    ax.set_yticklabels([APP_LABEL[a] for a in APP_ORDER], fontsize=8.5)
    ax.set_title(f"{exp}: speedup {base_cfg} -> {cfg}  (green=scales, red=stalls, TO=timeout)")
    fig.colorbar(im, ax=ax, label="speedup", shrink=0.85)
    fig.tight_layout(); savefig(fig, outdir, f"fig_heatmap_{exp}")


def main():
    md = sys.argv[1] if len(sys.argv) > 1 else \
        "benchmarks/scripts/logs/perf/matrix6app"
    E, ST, M = load(md)
    outdir = Path(md) / "analysis" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    print("[figures] ->", outdir)
    for exp in ("strong", "weak", "fix"):
        fig_scaling(E, ST, outdir, exp)
    fig_scaling(E, ST, outdir, "strong", speedup=True)
    fig_eager_lazy(E, outdir)
    fig_arts_vs_ref(E, outdir)
    fig_network(M, outdir, "bytes", "remote bytes sent (MB)", "fig_net_bytes", unit=1e6)
    fig_network(M, outdir, "sends", "remote sends (count)", "fig_net_sends")
    fig_comm_correlation(E, ST, M, outdir)
    fig_comm_intensity(M, E, outdir)
    fig_load_balance(M, outdir)
    for exp in ("strong", "weak", "fix"):
        fig_heatmap(E, ST, outdir, exp)
    print("[figures] done")


if __name__ == "__main__":
    main()
