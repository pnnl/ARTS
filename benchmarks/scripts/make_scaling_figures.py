#!/usr/bin/env python3
"""make_scaling_figures.py -- 2026-07-06 scaling-campaign figures.

Absolute seconds are NOT comparable across apps (each app is sized differently),
so the DEFAULT figures are NORMALIZED:

  (a) a_runtime_1n_normalized  -- per app, each runtime's median e2e / the app's
      ARTS mrnew_lazy median (mrnew_lazy == 1.0).  Log x.  Whiskers propagate as
      ratios (min/max over that runtime's iters / the mrnew_lazy median).  A
      TIMEOUT is a lower bound (>= 30 s / base): hatched bar + '>' arrow.  A
      GEOMEAN group (geometric mean ratio per runtime, OK cells only, TIMEOUTs
      excluded) summarises across apps.
  (b) b_protocols_1n_normalized -- same, the 5 ARTS coherence variants.  MRMW is
      hatched where the app is MRMW-contract-ineligible (correctness_harness
      mrmw_skip).
  (c) c_concurrency_speedup -- 48w vs 12w speedup = t(12w)/t(48w) (>1 = 48w
      faster), per runtime; ideal = 4x (4x the cores).
  (d1/d2) strong/weak speedup panels -- per app, speedup vs 1 node = t(1n_sc)/
      t(Nn_sc) over nodes 1/2/4, with the y=x ideal (strong) / y=1 weak-ideal;
      TIMEOUT = open marker (a 30 s floor -> an UPPER bound on speedup).

Absolute-second originals are preserved under figures/absolute/.  dataviz-skill
palette (ARTS #2a78d6 + cool variants; xsocr #1baf7a; ocr-vx #eda100), fixed
order, recessive grid.  Robust to partial data; safe to re-run.

Usage: make_scaling_figures.py --single <dir> --strong <dir> --weak <dir> --out <exp_dir>
"""
from __future__ import annotations
import argparse, csv, math, re, statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TIMEOUT_CAP = 30.0

RT_ORDER = ["mrnew_eager", "mrnew_lazy", "lock_eager", "lock_lazy", "mrmw", "xsocr", "ocrvx"]
RT_COLOR = {"mrnew_eager": "#1b5c9e", "mrnew_lazy": "#2a78d6", "lock_eager": "#5b3fb8",
            "lock_lazy": "#9b7fe0", "mrmw": "#c44fa0", "xsocr": "#1baf7a", "ocrvx": "#eda100"}
RT_LABEL = {"mrnew_eager": "ARTS MRNEW/eager", "mrnew_lazy": "ARTS MRNEW/lazy",
            "lock_eager": "ARTS LOCK/eager", "lock_lazy": "ARTS LOCK/lazy",
            "mrmw": "ARTS MRMW", "xsocr": "xsocr (OCR)", "ocrvx": "ocr-vx (TBB)"}
ARTS_VARIANTS = RT_ORDER[:5]
CROSS = ["mrnew_lazy", "xsocr", "ocrvx"]
INK, MUTED, GRID = "#1a1a1a", "#666666", "#dddddd"


def get_mrmw_ineligible(repo: Path) -> dict:
    src = (repo / "benchmarks" / "scripts" / "correctness_harness.py").read_text()
    out = {}
    for chunk in re.split(r"\bCase\(", src)[1:]:
        nm = re.search(r"[\"']([A-Za-z0-9_]+)[\"']", chunk)
        ms = re.search(r"mrmw_skip\s*=\s*[\"']([^\"']+)", chunk)
        if nm and ms:
            out[nm.group(1)] = ms.group(1)
    return out


def load(dir_: Path) -> list:
    p = dir_ / "results.csv"
    return list(csv.DictReader(open(p, newline=""))) if p.exists() else []


def agg(rows: list) -> dict:
    buck = {}
    for r in rows:
        key = (r["bench"], r["config"], r["runtime"])
        e, statv = r.get("e2e_ns"), r.get("status")
        if e:
            buck.setdefault(key, []).append((float(e) / 1e9, False))
        elif statv in ("TIMEOUT", "FAIL"):
            buck.setdefault(key, []).append((TIMEOUT_CAP, True))
    out = {}
    for key, vals in buck.items():
        ok = [v for v, c in vals if not c]
        if ok:
            out[key] = {"med": st.median(ok), "lo": min(ok), "hi": max(ok), "censored": False, "n": len(ok)}
        else:
            out[key] = {"med": TIMEOUT_CAP, "lo": TIMEOUT_CAP, "hi": TIMEOUT_CAP, "censored": True, "n": 0}
    return out


def geomean(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


def _style(ax, axis="x"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis=axis, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Normalized bar figures (a, b) + geomean
# ---------------------------------------------------------------------------
def _norm_ratios(A, benches, runtimes, base_rt="mrnew_lazy"):
    """Return (rows, geomeans): rows[b][rt] = (ratio, rlo, rhi, censored);
    geomeans[rt] over OK (non-censored) cells."""
    rows, per_rt = {}, {rt: [] for rt in runtimes}
    keep = []
    for b in benches:
        base = A.get((b, "1n", base_rt))
        if not base or base["censored"] or base["med"] <= 0:
            continue
        bm = base["med"]; d = {}
        for rt in runtimes:
            c = A.get((b, "1n", rt))
            if not c:
                continue
            d[rt] = (c["med"] / bm, c["lo"] / bm, c["hi"] / bm, c["censored"])
            if not c["censored"]:
                per_rt[rt].append(c["med"] / bm)
        if d:
            rows[b] = d; keep.append(b)
    return keep, rows, {rt: geomean(v) for rt, v in per_rt.items()}


def fig_normalized(A, benches, runtimes, base_rt, title, path, mrmw_bad):
    keep, rows, gm = _norm_ratios(A, benches, runtimes, base_rt)
    if not keep:
        return gm
    n, k = len(keep), len(runtimes)
    allr = [rows[b][rt][0] for b in keep for rt in runtimes if rt in rows[b]]
    floor = max(0.05, min(allr) * 0.7); right = max(allr + [1]) * 1.4
    fig, ax = plt.subplots(figsize=(9.5, max(3.5, 0.34 * (n + 2) * k / 2)))
    bw = 0.82 / k
    # geomean group sits below the apps, separated by a gap of 1.
    def ypos(i, j):
        return i + (j - k / 2) * bw + bw / 2
    for j, rt in enumerate(runtimes):
        for i, b in enumerate(keep):
            if rt not in rows[b]:
                continue
            ratio, rlo, rhi, cen = rows[b][rt]
            y = ypos(i, j)
            hatch = "///" if cen or (rt == "mrmw" and b in mrmw_bad) else ""
            ax.barh(y, ratio - floor, left=floor, height=bw, color=RT_COLOR[rt], zorder=3,
                    edgecolor="white", linewidth=0.4, hatch=hatch,
                    xerr=[[max(0, ratio - rlo)], [max(0, rhi - ratio)]] if not cen else None,
                    error_kw=dict(ecolor=MUTED, lw=0.6, capsize=1.5))
            if cen:
                ax.annotate(">", (ratio, y), fontsize=7, color=RT_COLOR[rt], va="center", ha="left", weight="bold")
    ygeo = n + 1
    for j, rt in enumerate(runtimes):
        if gm.get(rt):
            ax.barh(ygeo + (j - k / 2) * bw + bw / 2, gm[rt] - floor, left=floor, height=bw,
                    color=RT_COLOR[rt], zorder=3, edgecolor=INK, linewidth=0.7, label=RT_LABEL[rt])
    ax.axvline(1.0, color=INK, lw=1.0, ls="-", zorder=2)
    ax.set_xscale("log"); ax.set_xlim(floor, right)
    ax.set_yticks(list(range(n)) + [ygeo])
    ax.set_yticklabels(keep + ["GEOMEAN"], fontsize=7)
    for t in ax.get_yticklabels():
        if t.get_text() == "GEOMEAN":
            t.set_weight("bold")
    ax.set_xlabel(f"normalized e2e  (x ARTS mrnew_lazy;  1.0 = mrnew_lazy;  <1 faster)  [log]", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", weight="bold")
    _style(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.text(0.01, -0.01, "hatched = TIMEOUT lower bound (>=30s/base, '>' arrow) or MRMW-ineligible; excluded from GEOMEAN.",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return gm


# ---------------------------------------------------------------------------
# (e) timing axis isolated: eager vs lazy per protocol
# ---------------------------------------------------------------------------
def fig_eager_vs_lazy(A, benches, path):
    """Per app, the eager/lazy median-e2e ratio for the MRNEW pair and the LOCK
    pair (>1 = eager slower).  Log x, 1.0 parity baseline, GEOMEAN group -- shows
    whether an eager-vs-lazy gap is uniform across apps or driven by a few."""
    PAIRS = [("MRNEW  eager/lazy", "mrnew_eager", "mrnew_lazy", RT_COLOR["mrnew_lazy"]),
             ("LOCK  eager/lazy", "lock_eager", "lock_lazy", RT_COLOR["lock_eager"])]
    labels = [p[0] for p in PAIRS]; colors = {p[0]: p[3] for p in PAIRS}
    rows, per, keep = {}, {lab: [] for lab in labels}, []
    for b in benches:
        d = {}
        for lab, eg, lz, _ in PAIRS:
            e, l = A.get((b, "1n", eg)), A.get((b, "1n", lz))
            if not e or not l or l["med"] <= 0:
                continue
            d[lab] = (e["med"] / l["med"], e["censored"], l["censored"])
            if not (e["censored"] or l["censored"]):
                per[lab].append(e["med"] / l["med"])
        if d:
            rows[b] = d; keep.append(b)
    if not keep:
        return {}
    gm = {lab: geomean(v) for lab, v in per.items()}
    n, k = len(keep), len(PAIRS)
    allr = [rows[b][lab][0] for b in keep for lab in labels if lab in rows[b]]
    floor = max(0.05, min(allr + [1]) * 0.7); right = max(allr + [1]) * 1.4
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.34 * (n + 2) * k / 2)))
    bw = 0.82 / k
    for j, lab in enumerate(labels):
        for i, b in enumerate(keep):
            if lab not in rows[b]:
                continue
            ratio, ce, cl = rows[b][lab]
            y = i + (j - k / 2) * bw + bw / 2
            ax.barh(y, ratio - floor, left=floor, height=bw, color=colors[lab], zorder=3,
                    edgecolor="white", linewidth=0.4, hatch=("///" if (ce or cl) else ""))
            if ce and not cl:
                ax.annotate(">", (ratio, y), fontsize=7, color=colors[lab], va="center", ha="left", weight="bold")
            elif cl and not ce:
                ax.annotate("<", (floor, y), fontsize=7, color=colors[lab], va="center", ha="right", weight="bold")
    ygeo = n + 1
    for j, lab in enumerate(labels):
        if gm.get(lab):
            ax.barh(ygeo + (j - k / 2) * bw + bw / 2, gm[lab] - floor, left=floor, height=bw,
                    color=colors[lab], zorder=3, edgecolor=INK, linewidth=0.7, label=lab)
    ax.axvline(1.0, color=INK, lw=1.0, zorder=2)
    ax.set_xscale("log"); ax.set_xlim(floor, right)
    ax.set_yticks(list(range(n)) + [ygeo]); ax.set_yticklabels(keep + ["GEOMEAN"], fontsize=7)
    for t in ax.get_yticklabels():
        if t.get_text() == "GEOMEAN":
            t.set_weight("bold")
    ax.set_xlabel("eager / lazy  median e2e @48w   (>1 = eager slower;  1.0 = parity)  [log]", color=INK, fontsize=9)
    ax.set_title("(e) Timing axis isolated: eager vs lazy per ARTS protocol", color=INK, fontsize=11, loc="left", weight="bold")
    _style(ax); ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.text(0.01, -0.01, "hatched = TIMEOUT bound ('>' eager>=30s: lower bound; '<' lazy>=30s: upper bound); excluded from GEOMEAN.",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return gm


# ---------------------------------------------------------------------------
# (f) same-protocol implementation duel: one runtime / another, per app
# ---------------------------------------------------------------------------
def fig_pair_ratio(A, benches, num_rt, den_rt, num_lbl, den_lbl, color, title, xlabel, path):
    """Per app, ratio num_rt/den_rt of 48w (1n) median e2e (>1 = num slower).
    Returns (geomean_over_OK, sorted rows list of (bench, ratio, num_cen, den_cen),
    ascending by ratio).  num TIMEOUT -> lower bound ('>'); den TIMEOUT -> upper
    bound ('<'); both TIMEOUT -> excluded."""
    rows = []
    for b in benches:
        cn, cd = A.get((b, "1n", num_rt)), A.get((b, "1n", den_rt))
        if not cn or not cd or cd["med"] <= 0 or cn["med"] <= 0:
            continue
        if cn["censored"] and cd["censored"]:
            continue
        rows.append((b, cn["med"] / cd["med"], cn["censored"], cd["censored"]))
    if not rows:
        return None, []
    rows.sort(key=lambda r: r[1])
    ok = [r[1] for r in rows if not (r[2] or r[3])]
    gm = geomean(ok)
    keep = [r[0] for r in rows]; n = len(keep)
    allr = [r[1] for r in rows]
    floor = max(0.03, min(allr + [1]) * 0.7); right = max(allr + [1]) * 1.4
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.24 * (n + 3))))
    for i, (b, ratio, nc, dc) in enumerate(rows):
        ax.barh(i, ratio - floor, left=floor, height=0.7, color=color, zorder=3,
                edgecolor="white", linewidth=0.4, hatch=("///" if (nc or dc) else ""))
        if nc and not dc:
            ax.annotate(">", (ratio, i), fontsize=7, color=color, va="center", ha="left", weight="bold")
        elif dc and not nc:
            ax.annotate("<", (floor, i), fontsize=7, color=color, va="center", ha="right", weight="bold")
    ygeo = n + 1
    if gm:
        ax.barh(ygeo, gm - floor, left=floor, height=0.7, color=color, zorder=3,
                edgecolor=INK, linewidth=0.8, label=f"GEOMEAN {gm:.2f}")
    ax.axvline(1.0, color=INK, lw=1.0, zorder=2)
    ax.set_xscale("log"); ax.set_xlim(floor, right)
    ax.set_yticks(list(range(n)) + [ygeo]); ax.set_yticklabels(keep + ["GEOMEAN"], fontsize=7)
    for t in ax.get_yticklabels():
        if t.get_text() == "GEOMEAN":
            t.set_weight("bold")
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", weight="bold")
    ax.text(floor, ygeo + 1.3, f"<- {den_lbl} faster", color=MUTED, fontsize=7, ha="left")
    ax.text(right, ygeo + 1.3, f"{num_lbl} faster ->", color=MUTED, fontsize=7, ha="right")
    _style(ax); ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.text(0.01, -0.01, "hatched = TIMEOUT bound ('>' num>=30s lower bound; '<' den>=30s upper bound); both TIMEOUT excluded; geomean over OK only.",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return gm, rows


# ---------------------------------------------------------------------------
# (c) concurrency speedup 48w/12w
# ---------------------------------------------------------------------------
def fig_concurrency_speedup(A, benches, runtimes, path):
    bs = [b for b in benches if any((b, "1n", rt) in A and (b, "1n_sc", rt) in A for rt in runtimes)]
    if not bs:
        return
    n, k = len(bs), len(runtimes)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.34 * n * k / 2)))
    bw = 0.82 / k
    maxsp = 1.0
    for j, rt in enumerate(runtimes):
        for i, b in enumerate(bs):
            c48, c12 = A.get((b, "1n", rt)), A.get((b, "1n_sc", rt))
            if not c48 or not c12 or c48["med"] < 0.05 or c12["med"] < 0.05:
                continue  # skip fixtures/crashes with no real per-worker workload
            sp = c12["med"] / c48["med"]; cen = c48["censored"] or c12["censored"]
            maxsp = max(maxsp, sp)
            ax.barh(i + (j - k / 2) * bw + bw / 2, sp, height=bw, color=RT_COLOR[rt], zorder=3,
                    edgecolor="white", linewidth=0.4, hatch=("///" if cen else ""),
                    label=RT_LABEL[rt] if i == 0 else "")
    ax.axvline(1.0, color=MUTED, lw=0.8, ls="-", zorder=2)
    ax.axvline(4.0, color="#c0392b", lw=1.0, ls="--", zorder=2)
    ax.text(4.0, n - 0.3, "ideal 4x (4x cores)", color="#c0392b", fontsize=7, ha="center")
    ax.set_yticks(range(n)); ax.set_yticklabels(bs, fontsize=7)
    ax.set_xlabel("speedup  t(12w) / t(48w)   (>1 = 48 workers faster)", color=INK, fontsize=9)
    ax.set_title("(c) Within-node concurrency speedup: 48 vs 12 workers", color=INK, fontsize=11, loc="left", weight="bold")
    _style(ax); ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# (d) strong/weak speedup panels
# ---------------------------------------------------------------------------
def fig_speedup_panels(A, benches, title, path, weak=False, runtimes=RT_ORDER):
    # A crash that still printed an [E2E] marker records an implausibly tiny
    # e2e (< REAL_FLOOR) -> a bogus huge ratio; and micro-fixtures have no real
    # workload to scale.  Show only apps with a real 1n_sc base, and drop
    # individual crash-artifact points.  `runtimes` selects which series to draw
    # (ARTS-only for the default intra-ARTS panels; all 7 for the with_refs copy).
    REAL_FLOOR = 0.05
    YCAP = 6.0 if not weak else 2.5
    geos = [("1n_sc", 1), ("2n_sc", 2), ("4n_sc", 4)]
    def real_base(b, rt):
        c = A.get((b, "1n_sc", rt))
        return c and not c["censored"] and c["med"] > REAL_FLOOR
    bs = [b for b in benches if real_base(b, "mrnew_lazy")]
    if not bs:
        return
    ncol = 4; nrow = (len(bs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.5 * nrow), squeeze=False)
    for idx, b in enumerate(bs):
        ax = axes[idx // ncol][idx % ncol]
        pmax, clipped = 1.0, False
        for rt in runtimes:
            if not real_base(b, rt):
                continue
            b1 = A[(b, "1n_sc", rt)]["med"]; xs, ys, cens = [], [], []
            for g, nn in geos:
                c = A.get((b, g, rt))
                if not c or (not c["censored"] and c["med"] < REAL_FLOOR):
                    continue  # skip crash-artifact points (tiny e2e that isn't a TIMEOUT)
                xs.append(nn); ys.append(b1 / c["med"]); cens.append(c["censored"])
            if len(xs) < 2:
                continue
            yp = [min(y, YCAP) for y in ys]
            clipped = clipped or any(y > YCAP for y in ys)
            pmax = max(pmax, max(yp))
            ax.plot(xs, yp, "-", color=RT_COLOR[rt], lw=1.5, zorder=3)
            for x, y, yc, cn in zip(xs, ys, yp, cens):
                ax.plot(x, yc, "o", ms=5, color=RT_COLOR[rt],
                        markerfacecolor=("none" if cn else RT_COLOR[rt]), zorder=4)
                if cn or y > YCAP:  # TIMEOUT (upper bound) or clipped: mark with arrow
                    ax.annotate("", xy=(x, yc), xytext=(0, 11), textcoords="offset points",
                                arrowprops=dict(arrowstyle="->", color=RT_COLOR[rt], lw=0.9, alpha=0.85))
        if weak:
            ax.axhline(1.0, ls="--", color="#c0392b", lw=0.9, zorder=2)   # weak-ideal: flat efficiency = 1
        else:
            ax.plot([1, 4], [1, 4], ls=":", color=INK, lw=0.9, zorder=2)  # strong-ideal: speedup = N (y=x)
        ax.set_title(b + (" (clip)" if clipped else ""), fontsize=8, color=INK)
        ax.set_xticks([1, 2, 4]); ax.set_xticklabels(["1", "2", "4"], fontsize=7); ax.tick_params(labelsize=7)
        ax.set_ylim(0, max(1.25, min(YCAP, pmax * 1.15)))
        _style(ax, axis="y")
    for j in range(len(bs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles = [Line2D([0], [0], color=RT_COLOR[rt], lw=2, label=RT_LABEL[rt]) for rt in runtimes]
    if weak:
        handles.append(Line2D([0], [0], ls="--", color="#c0392b", label="weak-ideal: flat efficiency = 1"))
        ylab = "weak-scaling efficiency  t(1n_sc)/t(Nn_sc)   (1.0 = ideal; >1 super-linear)"
        foot = "weak: per-node work fixed & total grows with N -> time-invariant (efficiency 1.0) is ideal, NOT y=x."
    else:
        handles.append(Line2D([0], [0], ls=":", color=INK, label="strong-ideal: y=x (speedup = N)"))
        ylab = "speedup  t(1n_sc)/t(Nn_sc)   (N = ideal)"
        foot = "strong: fixed total work -> linear speedup (y=x) is ideal."
    handles.append(Line2D([0], [0], marker="o", color=MUTED, markerfacecolor="none", lw=0,
                          label="open + arrow = TIMEOUT (t(N)>=30s: plotted is an UPPER bound, true value lower)"))
    fig.suptitle(title, y=1.005, fontsize=12, weight="bold", color=INK, x=0.02, ha="left")
    fig.legend(handles=handles, fontsize=8, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.982))
    fig.text(-0.005, 0.5, ylab, rotation=90, va="center", ha="center", fontsize=9, color=INK)
    fig.text(0.5, -0.008, f"nodes 1/2/4 (12 workers each).  {foot}", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.955]); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Absolute-second originals (preserved under figures/absolute/)
# ---------------------------------------------------------------------------
def fig_abs_bars(A, benches, runtimes, title, path, mrmw_bad):
    bs = [b for b in benches if any((b, "1n", rt) in A for rt in runtimes)]
    if not bs:
        return
    n, k = len(bs), len(runtimes)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.30 * n * k / 2)))
    bw = 0.8 / k
    for j, rt in enumerate(runtimes):
        for i, b in enumerate(bs):
            c = A.get((b, "1n", rt))
            if not c:
                continue
            y = i + (j - k / 2) * bw + bw / 2
            ax.barh(y, c["med"], height=bw, color=RT_COLOR[rt], zorder=3, edgecolor="white", linewidth=0.4,
                    hatch=("///" if c["censored"] or (rt == "mrmw" and b in mrmw_bad) else ""),
                    xerr=[[c["med"] - c["lo"]], [c["hi"] - c["med"]]],
                    error_kw=dict(ecolor=MUTED, lw=0.6, capsize=1.5),
                    label=RT_LABEL[rt] if i == 0 else "")
    ax.axvline(TIMEOUT_CAP, color="#c0392b", lw=0.8, ls=":", zorder=2)
    ax.set_yticks(range(n)); ax.set_yticklabels(bs, fontsize=7)
    ax.set_xlabel("e2e (s) -- absolute (NOT cross-app comparable)", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", weight="bold")
    _style(ax); ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def fig_abs_scaling(A, benches, title, path):
    geos = [("1n_sc", 1), ("2n_sc", 2), ("4n_sc", 4)]
    bs = [b for b in benches if any((b, g, rt) in A for g, _ in geos for rt in RT_ORDER)]
    if not bs:
        return
    ncol = 4; nrow = (len(bs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for idx, b in enumerate(bs):
        ax = axes[idx // ncol][idx % ncol]
        for rt in RT_ORDER:
            xs, ys, cens = [], [], []
            for g, nn in geos:
                c = A.get((b, g, rt))
                if not c:
                    continue
                xs.append(nn); ys.append(c["med"]); cens.append(c["censored"])
            if not xs:
                continue
            ax.plot(xs, ys, "-", color=RT_COLOR[rt], lw=1.4, zorder=3)
            for x, y, cn in zip(xs, ys, cens):
                ax.plot(x, y, "o", ms=4, color=RT_COLOR[rt], markerfacecolor=("none" if cn else RT_COLOR[rt]), zorder=4)
        ax.axhline(TIMEOUT_CAP, color="#c0392b", lw=0.6, ls=":")
        ax.set_title(b, fontsize=8, color=INK)
        ax.set_xticks([1, 2, 4]); ax.set_xticklabels(["1", "2", "4"], fontsize=7); ax.tick_params(labelsize=7)
        _style(ax, axis="y")
    for j in range(len(bs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title, y=1.01, fontsize=12, weight="bold", color=INK, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=125, bbox_inches="tight"); plt.close(fig)


def write_summary_csv(A, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bench", "config", "runtime", "med_e2e_s", "min_e2e_s", "max_e2e_s", "censored", "iters"])
        for (b, cfg, rt), v in sorted(A.items()):
            w.writerow([b, cfg, rt, f"{v['med']:.3f}", f"{v['lo']:.3f}", f"{v['hi']:.3f}", int(v["censored"]), v["n"]])


def write_readme(out, A_s, A_st, A_wk, order, mrmw_bad, ph, gm_a, gm_b, gm_e, gm_f, rows_f):
    def med(A, b, cfg, rt):
        c = A.get((b, cfg, rt)); return c["med"] if c else None
    L = ["# ARTS scaling campaign -- 2026-07-06 (localhost cbgpu02, 48 cores)\n"]
    L.append("Cross-runtime + cross-protocol perf on **one 48-core box**; multinode is "
             "**localhost-simulated** (libfabric loopback) -- absolute MN numbers are comm-inflated; "
             "**real multinode requires the junction cluster**. Within-node (1n 48w / 1n_sc 12w) is real.\n")
    L.append("**Normalized by design**: absolute seconds are not comparable across apps, so figures a/b "
             "report **e2e / the app's ARTS mrnew_lazy median (mrnew_lazy = 1.0)**, c/d report **speedup**. "
             "Metric = [E2E] compute span; a TIMEOUT is a `>=30 s` lower bound (hatched, excluded from geomean). "
             "Absolute-second originals are in `figures/absolute/`.\n")
    L.append("## Figures\n")
    L.append("- **a_runtime_1n_normalized** -- 1n 48w, each runtime / mrnew_lazy (log x) + GEOMEAN group.")
    L.append("- **b_protocols_1n_normalized** -- the 5 ARTS variants / mrnew_lazy + GEOMEAN; MRMW hatched where ineligible.")
    L.append("- **e_eager_vs_lazy** -- timing axis isolated: per-app eager/lazy median-e2e ratio for the MRNEW pair and "
             "the LOCK pair (>1 = eager slower), log x, 1.0 parity line + GEOMEAN. Reads whether an eager-vs-lazy gap is "
             "uniform or driven by a few apps.")
    L.append("- **f_lock_eager_vs_xsocr** -- **same protocol, two implementations**: ARTS LOCK/eager and xsocr are the "
             "SAME coherence protocol (home-centric lock arbitration + release-time eager writeback), so this is an "
             "implementation duel. Per-app ratio LOCK/eager / xsocr (>1 = the ARTS implementation is slower), log x, "
             "1.0 parity line, GEOMEAN.")
    L.append("- **c_concurrency_speedup** -- t(12w)/t(48w) speedup (ideal 4x).")
    L.append("- **d1_strong_speedup** -- per-app **strong speedup** = t(1n_sc)/t(Nn_sc) vs nodes 1/2/4; ideal **y=x**. "
             "**ARTS 5 protocols only** (intra-ARTS scaling comparison); ARTS-variant colors as in (b).")
    L.append("- **d2_weak_speedup** -- per-app **weak-scaling efficiency** = t(1n_sc)/t(Nn_sc); ideal **flat 1.0** "
             "(per-node work fixed -> constant time is ideal, NOT y=x); >1 = super-linear. **ARTS 5 protocols only**.")
    L.append("- Both d1/d2: a TIMEOUT point is an **open marker + arrow** -- t(N)>=30 s means the plotted t1/tN (tN=30) is an UPPER bound; the true value is lower.")
    L.append("- `figures/with_refs/` -- d1/d2 including the xsocr + ocr-vx reference series (all 7 runtimes).")
    L.append("- `figures/absolute/` -- the raw-second bars/curves.\n")
    if any(gm_e.values()):
        L.append("### (e) eager-vs-lazy GEOMEAN @48w (>1 = eager slower)\n")
        L.append("| protocol pair | geomean eager/lazy |")
        L.append("|---|---|")
        for lab, v in gm_e.items():
            if v:
                L.append(f"| {lab.strip()} | {v:.2f} |")
        L.append("")
    if gm_f and rows_f:
        L.append("### (f) same-protocol implementation duel: ARTS LOCK/eager vs xsocr @48w\n")
        L.append("ARTS LOCK/eager and xsocr implement the **same** coherence protocol (home-centric lock arbitration + "
                 "release-time eager writeback), so this isolates *implementation* quality. Ratio = LOCK/eager / xsocr "
                 f"(>1 = ARTS slower). **GEOMEAN = {gm_f:.2f}** over OK cells "
                 f"({'ARTS faster on average' if gm_f < 1 else 'xsocr faster on average'}).\n")
        def _fmt(r):
            b, ratio, nc, dc = r
            return f"`{b}` {ratio:.2f}" + ("(>=,ARTS TO)" if nc else "(<=,xsocr TO)" if dc else "")
        arts_wins = [r for r in rows_f if not (r[2] or r[3])][:5]                  # lowest ratio
        xsocr_wins = [r for r in rows_f if not (r[2] or r[3])][-5:][::-1]          # highest ratio
        L.append("- **ARTS LOCK/eager wins most (lowest ratio):** " + ", ".join(_fmt(r) for r in arts_wins))
        L.append("- **xsocr wins most (highest ratio):** " + ", ".join(_fmt(r) for r in xsocr_wins))
        L.append("")
    # GEOMEAN table
    L.append("## GEOMEAN normalized e2e @48w (x mrnew_lazy; lower = faster; mrnew_lazy = 1.00)\n")
    L.append("| runtime | geomean (a: cross-runtime) | geomean (b: protocols) |")
    L.append("|---|---|---|")
    for rt in RT_ORDER:
        a = f"{gm_a[rt]:.2f}" if gm_a.get(rt) else "-"
        b = f"{gm_b[rt]:.2f}" if gm_b.get(rt) else "-"
        if a == "-" and b == "-":
            continue
        L.append(f"| {RT_LABEL[rt]} | {a} | {b} |")
    L.append("\n(geomean over apps with an OK mrnew_lazy base; TIMEOUT/censored cells excluded -- including them "
             "would only push the lower-bound ratios further and understate the reference runtimes' true cost.)\n")
    # win/loss @1n
    L.append("## Single-node 48w win/loss (median e2e s; `*` = TIMEOUT/censored)\n")
    L.append("| app | ARTS mrnew_lazy | xsocr | ocr-vx | fastest |")
    L.append("|---|---|---|---|---|")
    wins = {}
    for b in order:
        a1, x1, o1 = med(A_s, b, "1n", "mrnew_lazy"), med(A_s, b, "1n", "xsocr"), med(A_s, b, "1n", "ocrvx")
        if a1 is None and x1 is None and o1 is None:
            continue
        avail = {k: v for k, v in {"ARTS": a1, "xsocr": x1, "ocr-vx": o1}.items()
                 if v is not None and not A_s.get((b, "1n", {"ARTS": "mrnew_lazy", "xsocr": "xsocr", "ocr-vx": "ocrvx"}[k]), {}).get("censored")}
        best = min(avail, key=avail.get) if avail else "-"
        wins[best] = wins.get(best, 0) + 1
        def s(v, rt):
            return "-" if v is None else f"{v:.2f}" + ("*" if A_s.get((b, "1n", rt), {}).get("censored") else "")
        L.append(f"| {b} | {s(a1,'mrnew_lazy')} | {s(x1,'xsocr')} | {s(o1,'ocrvx')} | {best} |")
    L.append(f"\n**Fastest-at-48w tally (OK cells):** ARTS {wins.get('ARTS',0)}, xsocr {wins.get('xsocr',0)}, ocr-vx {wins.get('ocr-vx',0)}.\n")
    L.append("## MRMW-contract-ineligible (hatched; number shown, correctness NOT guaranteed under MRMW)\n")
    L.append(", ".join(f"`{k}`" for k in sorted(mrmw_bad)) + " -- same verdict as the correctness harness `mrmw_skip`.\n")
    exempt = sorted(getattr(ph, "_WEAK_EXEMPT", {}).items())
    if exempt:
        L.append("## Weak-scaling exemptions (no ~2x-scalable work knob; weak == strong there)\n")
        for k, v in exempt:
            L.append(f"- `{k}` -- {v}")
        L.append("")
    L.append("## Caveats\n")
    L.append("- **Localhost-simulated multinode** (loopback comm): comm-bound / NULL-HINT apps anti-scale and many "
             "diverge to TIMEOUT -- a placement/coherence stress signal, NOT a real-cluster verdict. Rerun on junction.")
    L.append("- **30 s wall budget** (chosen for a ~13 h campaign vs ~22 h at 60 s): cells finishing in 30-60 s record "
             "as TIMEOUT; every `>=30 s`/open marker means \"at least this\".")
    L.append("- Data: `data/summary_{single,strong,weak}.csv` (median/min/max/censored per cell).\n")
    Path(out / "README.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True); ap.add_argument("--strong", required=True)
    ap.add_argument("--weak", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent.parent
    out = Path(a.out); F = out / "figures"; FA = F / "absolute"; FR = F / "with_refs"
    FA.mkdir(parents=True, exist_ok=True); FR.mkdir(parents=True, exist_ok=True); (out / "data").mkdir(exist_ok=True)
    import sys; sys.path.insert(0, str(repo / "benchmarks" / "scripts"))
    import performance_harness as ph
    order = [b.name for b in ph.BENCHES]
    mrmw_bad = get_mrmw_ineligible(repo)
    A_s, A_st, A_wk = agg(load(Path(a.single))), agg(load(Path(a.strong))), agg(load(Path(a.weak)))
    for name, A in (("single", A_s), ("strong", A_st), ("weak", A_wk)):
        write_summary_csv(A, out / "data" / f"summary_{name}.csv")

    # normalized (default)
    gm_a = fig_normalized(A_s, order, CROSS, "mrnew_lazy",
                          "(a) Single-node 48w, normalized to ARTS mrnew_lazy (=1.0)",
                          F / "a_runtime_1n_normalized.png", mrmw_bad)
    gm_b = fig_normalized(A_s, order, ARTS_VARIANTS, "mrnew_lazy",
                          "(b) ARTS protocols @48w, normalized to MRNEW/lazy (=1.0)",
                          F / "b_protocols_1n_normalized.png", mrmw_bad)
    gm_e = fig_eager_vs_lazy(A_s, order, F / "e_eager_vs_lazy.png")
    # (f) same-protocol implementation duel: ARTS LOCK/eager vs xsocr (both are
    # home-centric lock arbitration + release-time eager writeback).
    gm_f, rows_f = fig_pair_ratio(A_s, order, "lock_eager", "xsocr", "xsocr", "ARTS LOCK/eager",
                                  RT_COLOR["lock_eager"],
                                  "(f) Same protocol, two implementations: ARTS LOCK/eager vs xsocr @48w",
                                  "ARTS LOCK/eager / xsocr  median e2e   (>1 = ARTS slower;  1.0 = parity)  [log]",
                                  F / "f_lock_eager_vs_xsocr.png")
    fig_concurrency_speedup(A_s, order, CROSS, F / "c_concurrency_speedup.png")
    # d1/d2 default = ARTS 5 protocols only (intra-ARTS scaling comparison).
    fig_speedup_panels(A_st, order, "(d1) STRONG scaling speedup -- ARTS protocols (fixed total work)",
                       F / "d1_strong_speedup.png", weak=False, runtimes=ARTS_VARIANTS)
    fig_speedup_panels(A_wk, order, "(d2) WEAK scaling efficiency -- ARTS protocols (per-node work fixed)",
                       F / "d2_weak_speedup.png", weak=True, runtimes=ARTS_VARIANTS)
    # with_refs copies = all 7 runtimes (arts variants + xsocr + ocr-vx).
    fig_speedup_panels(A_st, order, "(d1+refs) STRONG scaling speedup -- all runtimes",
                       FR / "d1_strong_speedup.png", weak=False, runtimes=RT_ORDER)
    fig_speedup_panels(A_wk, order, "(d2+refs) WEAK scaling efficiency -- all runtimes",
                       FR / "d2_weak_speedup.png", weak=True, runtimes=RT_ORDER)
    # absolute originals (preserved)
    fig_abs_bars(A_s, order, CROSS, "(a-abs) 48w ARTS vs xsocr vs ocr-vx (seconds)", FA / "a_runtime_1n.png", mrmw_bad)
    fig_abs_bars(A_s, order, ARTS_VARIANTS, "(b-abs) ARTS protocols @48w (seconds)", FA / "b_protocols_1n.png", mrmw_bad)
    fig_abs_scaling(A_st, order, "(d1-abs) STRONG e2e vs nodes", FA / "d1_strong_scaling.png")
    fig_abs_scaling(A_wk, order, "(d2-abs) WEAK e2e vs nodes", FA / "d2_weak_scaling.png")

    write_readme(out, A_s, A_st, A_wk, order, mrmw_bad, ph, gm_a, gm_b, gm_e, gm_f, rows_f)
    gm_line = " ".join(f"{rt}={gm_a[rt]:.2f}" for rt in CROSS if gm_a.get(rt))
    gm_eline = " | ".join(f"{lab}={v:.2f}" for lab, v in gm_e.items() if v)
    xs_win = ", ".join(f"{b}({r:.2f})" for b, r, _, _ in rows_f[:5])
    arts_win = ", ".join(f"{b}({r:.2f})" for b, r, _, _ in rows_f[-5:][::-1])
    print(f"[figures] geomean(a): {gm_line}; geomean(e): {gm_eline}; "
          f"geomean(f lock_eager/xsocr): {gm_f:.2f}\n  ARTS-wins-most(low): {xs_win}\n  xsocr-wins-most(high): {arts_win}")


if __name__ == "__main__":
    main()
