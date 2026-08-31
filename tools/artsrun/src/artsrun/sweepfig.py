"""Digest a sweep run: the long-form CSV always, the figure panels when
matplotlib is available.

The CSV is the datacube — one row per measured cell with the point's knobs
and every extracted scalar — and the figures are views of it.  Both are
rebuilt from the run directory alone (spec + track), so a detached or
finished run digests the same as a live one.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median

from artsrun.model.sweep import SweepSpec
from artsrun.report import RT_COLOR, RT_LABEL, rt_key

# Scalars that plot on a log axis read better in microseconds/ops-per-second
# as floats; everything the probe prints is numeric.


def _rows(run_dir: Path, spec: SweepSpec) -> list[dict]:
    by_point = {p.name: p for p in spec.points}
    finished: dict[str, dict] = {}
    track = run_dir / "track.jsonl"
    if track.is_file():
        for line in track.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") == "finished":
                finished[row.get("cell", "")] = row  # later attempt wins

    out: list[dict] = []
    for key, row in finished.items():
        # <app>+<point>:base@<n>n/<arm>#<rep>
        try:
            app_part, rest = key.split("@", 1)
            nodes_part, rest = rest.split("/", 1)
            arm, rep = rest.rsplit("#", 1)
            point_name = app_part.split("+", 1)[1].split(":", 1)[0]
        except (ValueError, IndexError):
            continue
        point = by_point.get(point_name)
        if point is None:
            continue
        knobs = spec.knobs_for(point)
        rec = {
            "point": point_name,
            "group": point.group,
            "arm": arm,
            "repeat": int(rep),
            "nodes": int(nodes_part.rstrip("n")),
            "status": row.get("status", ""),
            "wall_s": row.get("wall_s"),
            "e2e_s": row.get("e2e_s"),
        }
        for k, v in knobs.items():
            rec[f"knob_{k}"] = v
        for k, v in (row.get("extra") or {}).items():
            try:
                rec[k] = float(v)
            except (TypeError, ValueError):
                rec[k] = v
        out.append(rec)
    out.sort(key=lambda r: (r["group"], r["point"], r["arm"], r["repeat"]))
    return out


def _write_csv(rows: list[dict], path: Path) -> None:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _rho(rec: dict) -> float | None:
    r, w = rec.get("knob_R"), rec.get("knob_W")
    try:
        r, w = float(r), float(w)
        return w / (r + w) if (r + w) > 0 else None
    except (TypeError, ValueError):
        return None


def _ok(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("status") == "ok"]


def _series(rows, xkey, ykey):
    """(x, median y, min y, max y) per x, for rows that carry both."""
    per_x: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        x, y = r.get(xkey), r.get(ykey)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and y >= 0:
            per_x[float(x)].append(float(y))
    xs = sorted(per_x)
    return (xs, [median(per_x[x]) for x in xs],
            [min(per_x[x]) for x in xs], [max(per_x[x]) for x in xs])


def _rho_label(x: float) -> str:
    f = Fraction(x).limit_denominator(64)
    if f == 0:
        return "0"
    if f == 1:
        return "1"
    return f"{f.numerator}/{f.denominator}"


def _rho_axis(ax, rhos: list[float]) -> None:
    """Write-share axis: symlog spacing so the read-heavy tail spreads out,
    ticks at exactly the measured mixes, labeled as the fractions they are."""
    ax.set_xscale("symlog", linthresh=1.0 / 64.0)
    ticks = sorted(set(rhos))
    ax.set_xticks(ticks)
    ax.set_xticklabels([_rho_label(t) for t in ticks], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel("write share ρ")


def _style(arm: str) -> dict:
    color = RT_COLOR.get(rt_key(arm), "#666666")
    dashed = arm.endswith(("_wt", "_wt_purge")) and "excl" not in arm
    return {"color": color,
            "linestyle": "--" if dashed else "-",
            "label": RT_LABEL.get(rt_key(arm), arm)}


def _fig_ratio(rows: list[dict], out: Path, plt) -> Path | None:
    """Panel (a): per-class and aggregate throughput against the write share."""
    usable = [dict(r, rho=_rho(r)) for r in _ok(rows)]
    usable = [r for r in usable if r["rho"] is not None]
    if not usable:
        return None
    arms = sorted({r["arm"] for r in usable})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    panels = [("XPUT_TOTAL", "aggregate ops/s"),
              ("RRATE", "per-reader-chain ops/s"),
              ("WRATE", "per-writer-chain ops/s")]
    for r in usable:
        rx, wx = r.get("RXPUT"), r.get("WXPUT")
        if isinstance(rx, float) and isinstance(wx, float):
            r["XPUT_TOTAL"] = max(rx, 0.0) + max(wx, 0.0)
        # Population-normalized rates: R and W change along the gradient, so
        # the aggregate conflates head-count with speed; the per-chain rate
        # is what one actor experiences (starvation zeros included).
        try:
            R_, W_ = float(r.get("knob_R") or 0), float(r.get("knob_W") or 0)
        except (TypeError, ValueError):
            R_, W_ = 0.0, 0.0
        if isinstance(rx, float) and R_ > 0:
            r["RRATE"] = max(rx, 0.0) / R_
        if isinstance(wx, float) and W_ > 0:
            r["WRATE"] = max(wx, 0.0) / W_
    all_rhos = sorted({r["rho"] for r in usable})
    for ax, (ykey, title) in zip(axes, panels):
        for arm in arms:
            xs, med, lo, hi = _series(
                [r for r in usable if r["arm"] == arm], "rho", ykey)
            if not xs:
                continue
            st = _style(arm)
            ax.plot(xs, med, marker="o", markersize=3, **st)
            ax.fill_between(xs, lo, hi, alpha=0.15, color=st["color"],
                            linewidth=0)
        _rho_axis(ax, all_rhos)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("ops/s (log)")
    axes[-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    path = out / "fig_a_ratio_xput.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _fig_plane(rows: list[dict], out: Path, plt) -> Path | None:
    """Panel (b): read-side vs write-side per-op overhead, per-arm trajectory
    over ρ; Pareto comparisons are per-ρ (same workload), never global.

    A rep whose populated writer class measured under 10 ops per chain is
    STARVED: its writer quantile describes a different phenomenon (whether
    the class ran at all), so it leaves the plane and is counted in the
    legend instead of stretching the axis by four decades."""
    usable = [dict(r, rho=_rho(r)) for r in _ok(rows)]
    usable = [r for r in usable if r["rho"] not in (None, 0.0, 1.0)]
    starved: dict[str, int] = defaultdict(int)
    plane: list[dict] = []
    for r in usable:
        try:
            w_chains = float(r.get("knob_W") or 0)
        except (TypeError, ValueError):
            w_chains = 0.0
        wops = r.get("WOPS")
        if (w_chains > 0 and isinstance(wops, float)
                and wops < 10.0 * w_chains):
            starved[r["arm"]] += 1
            continue
        if (isinstance(r.get("RTOT_P99"), float) and r["RTOT_P99"] > 0
                and isinstance(r.get("WTOT_P99"), float)
                and r["WTOT_P99"] > 0):
            plane.append(r)
    if not plane:
        return None
    arms = sorted({r["arm"] for r in plane})
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    med_points: dict[tuple[str, float], tuple[float, float]] = {}
    for arm in arms:
        mine = [r for r in plane if r["arm"] == arm]
        per_rho: dict[float, tuple[list[float], list[float]]] = defaultdict(
            lambda: ([], []))
        for r in mine:
            per_rho[r["rho"]][0].append(r["RTOT_P99"])
            per_rho[r["rho"]][1].append(r["WTOT_P99"])
        rhos = sorted(per_rho)
        xs = [median(per_rho[q][0]) for q in rhos]
        ys = [median(per_rho[q][1]) for q in rhos]
        st = _style(arm)
        if starved.get(arm):
            st["label"] += f" ({starved[arm]} starved reps off-plane)"
        ax.plot(xs, ys, marker="o", markersize=4, alpha=0.85, **st)
        for q, x, y in zip(rhos, xs, ys):
            med_points[(arm, q)] = (x, y)
            ax.annotate(_rho_label(q), (x, y), fontsize=5, alpha=0.6,
                        textcoords="offset points", xytext=(3, 3))
    # Per-ρ Pareto set: same workload only — a ring marks every arm no other
    # arm beats on BOTH classes at that mix.
    all_rhos = sorted({q for _, q in med_points})
    for q in all_rhos:
        pts = {a: xy for (a, qq), xy in med_points.items() if qq == q}
        for a, (x, y) in pts.items():
            dominated = any(ox < x and oy < y for oa, (ox, oy) in pts.items()
                            if oa != a)
            if not dominated:
                ax.plot([x], [y], marker="o", markersize=9, mfc="none",
                        mec="#333333", mew=0.8, linestyle="none", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reader per-op overhead p99 (µs)")
    ax.set_ylabel("writer per-op overhead p99 (µs)")
    ax.set_title("rings = per-ρ Pareto set (no arm better on both classes "
                 "at that mix)", fontsize=8)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    path = out / "fig_b_tradeoff_plane.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _winners(rows: list[dict], out: Path) -> Path | None:
    """Per-point winner ledger, per CLASS as well as aggregate: a single
    aggregate scalar crowns whoever serves the majority class, which is
    exactly the tradeoff the probe exists to expose — so the reader-side and
    writer-side winners are reported beside it.  A pair whose rep ranges
    overlap is a tie; a winner the whiskers don't back is never forced."""
    usable = _ok(rows)
    per: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for r in usable:
        rx, wx = r.get("RXPUT"), r.get("WXPUT")
        if not (isinstance(rx, float) and isinstance(wx, float)):
            continue
        cell = per[r["point"]][r["arm"]]
        # A measured zero from a POPULATED class is starvation — real data
        # that must weigh the median down; only a class absent by design
        # (zero chains) contributes nothing.
        def _knob(k):
            try:
                return float(r.get(k) or 0)
            except (TypeError, ValueError):
                return 0.0
        if _knob("knob_R") > 0:
            cell["read"].append(max(rx, 0.0))
        if _knob("knob_W") > 0:
            cell["write"].append(max(wx, 0.0))
        cell["aggregate"].append(max(rx, 0.0) + max(wx, 0.0))
    if not per:
        return None

    def best(point: str, metric: str) -> str:
        arms = {a: v[metric] for a, v in per[point].items() if v.get(metric)}
        if len(arms) < 2:
            return f"{'-':>34s}"
        ranked = sorted(arms, key=lambda a: -median(arms[a]))
        b, s = ranked[0], ranked[1]
        margin = median(arms[b]) / max(median(arms[s]), 1e-9)
        tie = "~" if min(arms[b]) <= max(arms[s]) else " "
        return f"{b:>24s} {margin:5.2f}x{tie}"

    lines = [
        "winner per point and class (margin = median best/second; ~ = rep "
        "ranges overlap, treat as a tie)",
        f"{'point':16s} {'aggregate':>34s} {'reader side':>34s} "
        f"{'writer side':>34s}",
    ]
    for point in sorted(per):
        lines.append(f"{point:16s} {best(point, 'aggregate')} "
                     f"{best(point, 'read')} {best(point, 'write')}")
    path = out / "winners.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


# The plane is a fractional factorial (EXCL forbids WT, WB forbids PURGE), so
# the honest sensitivity unit is the within-family pair: each contrast below
# changes exactly ONE design axis.  The family contrast fixes the second axis
# at WB x RETAIN (the flagship triple) and at WT x RETAIN.
_CONTRASTS = [
    ("WT->WB (VAL)", "arts_val_wt", "arts_val_wb"),
    ("WT->WB (INV)", "arts_inv_wt", "arts_inv_wb"),
    ("PURGE->RETAIN (EXCL)", "arts_excl_purge", "arts_excl_retain"),
    ("PURGE->RETAIN (VAL/WT)", "arts_val_wt_purge", "arts_val_wt"),
    ("PURGE->RETAIN (INV/WT)", "arts_inv_wt_purge", "arts_inv_wt"),
    ("VAL->INV (WB)", "arts_val_wb", "arts_inv_wb"),
    ("VAL->EXCL (WB.RETAIN)", "arts_val_wb", "arts_excl_retain"),
    ("VAL->INV (WT)", "arts_val_wt", "arts_inv_wt"),
]


def _contrasts(rows: list[dict], out: Path) -> Path | None:
    """One-axis-at-a-time contrast ledger: for every point and every metric,
    how much flipping ONE design axis buys (ratio of medians, >1 = the
    flip helps).  This is what separates 'the family matters here' from
    'the write policy matters here' without eight-line spaghetti."""
    usable = _ok(rows)
    per: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in usable:
        for m in ("RXPUT", "WXPUT"):
            v = r.get(m)
            if isinstance(v, float) and v > 0:
                per[(r["point"], r["arm"])][m].append(v)

    points = sorted({p for p, _ in per})
    if not points:
        return None
    lines = ["one-axis contrasts (ratio of medians, flip-to/flip-from; "
             ">1 = the flip helps that class)",
             f"{'point':12s} " + " ".join(f"{name:>24s}" for name, _, _ in
                                          _CONTRASTS)]
    for m in ("RXPUT", "WXPUT"):
        lines.append(f"-- {m} --")
        for point in points:
            cols = []
            for _, a, b in _CONTRASTS:
                va, vb = per.get((point, a), {}).get(m), \
                    per.get((point, b), {}).get(m)
                if va and vb:
                    cols.append(f"{median(vb) / median(va):>24.2f}")
                else:
                    cols.append(f"{'-':>24s}")
            lines.append(f"{point:12s} " + " ".join(cols))
    path = out / "contrasts.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def _batching(rows: list[dict], out: Path) -> Path | None:
    """Turn-batching signature: p99/p50 of the per-op overhead, per class.

    A turn-based arm bills its cost to the TURN HEAD (the op that waits a
    whole opposing phase out) while batch-mates ride free — a wide p99/p50.
    A per-op-billed arm (validation RTTs, invalidation rounds) charges every
    op similarly — a narrow one.  The ratio makes 'EXCL clumps' a measured
    number instead of an anecdote."""
    usable = _ok(rows)
    per: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in usable:
        for cls, p50k, p99k in (("R", "RTOT_P50", "RTOT_P99"),
                                ("W", "WTOT_P50", "WTOT_P99")):
            p50, p99 = r.get(p50k), r.get(p99k)
            if (isinstance(p50, float) and p50 > 0
                    and isinstance(p99, float) and p99 > 0):
                per[(r["point"], r["arm"])][cls].append(p99 / p50)
    if not per:
        return None
    arms = sorted({a for _, a in per})
    points = sorted({p for p, _ in per})
    lines = ["p99/p50 of per-op overhead (batching signature: wide = costs "
             "bill to turn heads, narrow = per-op billing)",
             f"{'point':12s} " + " ".join(f"{a.removeprefix('arts_'):>18s}"
                                          for a in arms)]
    for cls in ("R", "W"):
        lines.append(f"-- {cls} --")
        for point in points:
            cols = []
            for a in arms:
                v = per.get((point, a), {}).get(cls)
                cols.append(f"{median(v):>18.1f}" if v else f"{'-':>18s}")
            lines.append(f"{point:12s} " + " ".join(cols))
    path = out / "batching.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_dat(rows: list[dict], out: Path) -> list[Path]:
    """Whitespace-column per-arm series for pgfplots injection (the paper's
    figures are LaTeX-native; these files are what `\\addplot table` reads,
    so integration swaps data files, never redraws)."""
    usable = [dict(r, rho=_rho(r)) for r in _ok(rows)]
    usable = [r for r in usable if r["rho"] is not None]
    metrics = ["RXPUT", "WXPUT", "RTOT_P50", "RTOT_P99", "WTOT_P50",
               "WTOT_P99", "RACQ_P99", "RREL_P99", "WACQ_P99", "WREL_P99"]
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)
    written: list[Path] = []
    for arm in sorted({r["arm"] for r in usable}):
        mine = [r for r in usable if r["arm"] == arm]
        per_rho: dict[float, list[dict]] = defaultdict(list)
        for r in mine:
            per_rho[r["rho"]].append(r)
        lines = ["rho " + " ".join(
            f"{m.lower()}_med {m.lower()}_lo {m.lower()}_hi" for m in metrics)]
        for rho in sorted(per_rho):
            cols = [f"{rho:.6g}"]
            for m in metrics:
                vals = [r[m] for r in per_rho[rho]
                        if isinstance(r.get(m), float) and r[m] >= 0]
                if vals:
                    cols += [f"{median(vals):.4g}", f"{min(vals):.4g}",
                             f"{max(vals):.4g}"]
                else:
                    cols += ["nan", "nan", "nan"]
            lines.append(" ".join(cols))
        p = data_dir / f"{arm}.dat"
        p.write_text("\n".join(lines) + "\n")
        written.append(p)
    return written


def digest(run_dir: Path, out_dir: Path | None = None) -> list[Path]:
    spec = SweepSpec.model_validate(
        json.loads((run_dir / "sweep.json").read_text()))
    rows = _rows(run_dir, spec)
    out = out_dir or (run_dir / "figures")
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    csv_path = out / "sweep.csv"
    _write_csv(rows, csv_path)
    written.append(csv_path)
    winners = _winners(rows, out)
    if winners is not None:
        written.append(winners)
    contrasts = _contrasts(rows, out)
    if contrasts is not None:
        written.append(contrasts)
    batching = _batching(rows, out)
    if batching is not None:
        written.append(batching)
    written += _write_dat(rows, out)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        (out / "NO-FIGURES.txt").write_text(
            "matplotlib is not installed; only sweep.csv was written.\n"
            "install it with: uv pip install -e 'tools/artsrun[fig]'\n")
        written.append(out / "NO-FIGURES.txt")
        return written
    for fn in (_fig_ratio, _fig_plane, _fig_index):
        p = fn(rows, out, plt)
        if p is not None:
            written.append(p)
    return written


# ---------------------------------------------------------------- atlas ----
_FAMILY_COLOR = {"excl": "#4C72B0", "inv": "#DD8452", "val": "#55A868"}
_ARM_ABBR = {"excl_purge": "E/P", "excl_retain": "E/R", "inv_wt": "I/WT",
             "inv_wb": "I/WB", "inv_wt_purge": "I/WTp", "val_wt": "V/WT",
             "val_wb": "V/WB", "val_wt_purge": "V/WTp"}


def _family(arm: str) -> str:
    for fam in _FAMILY_COLOR:
        if fam in arm:
            return fam
    return "val"


def atlas(run_dirs: list[Path], out_dir: Path) -> list[Path]:
    """The regime atlas: winner-PAIR labels on physical axes.

    Cells of a (write share x DB size) grid, each split diagonally — the
    upper-left triangle is the READER-side winner's family color, the
    lower-right the WRITER-side winner's; the arm abbreviation is printed in
    each half, greyed when the runner-up's rep range overlaps (a tie the
    whiskers won't back).  Discovery is clustering by winner pair; the
    presentation stays on knobs a reader can feel.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    cells: dict[tuple[float, float], dict[str, dict[str, list[float]]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rd in run_dirs:
        spec = SweepSpec.model_validate(
            json.loads((rd / "sweep.json").read_text()))
        for r in _ok(_rows(rd, spec)):
            rho = _rho(r)
            try:
                size = float(r.get("knob_BYTES") or 0)
                d_knob = float(r.get("knob_D") or 1)
                hs = float(r.get("knob_HSPREAD") or 1)
                rs = float(r.get("knob_RSPREAD") or 0)
                ws = float(r.get("knob_WSPREAD") or 0)
                R_, W_ = float(r.get("knob_R") or 0), float(r.get("knob_W") or 0)
            except (TypeError, ValueError):
                continue
            # the atlas plane is the anchor geometry: D=1, default spreads
            if rho is None or d_knob != 1 or hs != 1 or rs != 0 or ws != 0:
                continue
            if int(r.get("nodes") or 8) != 8:
                continue
            cell = cells[(rho, size)][r["arm"]]
            rx, wx = r.get("RXPUT"), r.get("WXPUT")
            if isinstance(rx, float) and R_ > 0:
                cell["r"].append(max(rx, 0.0) / R_)
            if isinstance(wx, float) and W_ > 0:
                cell["w"].append(max(wx, 0.0) / W_)

    if not cells:
        return []
    rhos = sorted({k[0] for k in cells})
    sizes = sorted({k[1] for k in cells})
    fig, ax = plt.subplots(
        figsize=(1.35 * len(rhos) + 2.4, 1.1 * len(sizes) + 1.6))

    def winner(cell, klass):
        """Family-level verdict: the atlas asks WHICH FAMILY owns the cell,
        so the runner-up for the tie test is the best arm of another family
        (in-family variants tying among themselves is not a tie)."""
        arms = {a: v[klass] for a, v in cell.items() if v.get(klass)}
        if len(arms) < 2:
            return None
        ranked = sorted(arms, key=lambda a: -median(arms[a]))
        b = ranked[0]
        rival = next((a for a in ranked[1:]
                      if _family(a) != _family(b)), None)
        if rival is None:
            return b, 1.0, False
        tie = min(arms[b]) <= max(arms[rival])
        return b, median(arms[b]) / max(median(arms[rival]), 1e-9), tie

    for xi, rho in enumerate(rhos):
        for yi, size in enumerate(sizes):
            cell = cells.get((rho, size))
            if not cell:
                continue
            for klass, tri in (("r", [(xi, yi), (xi + 1, yi), (xi, yi + 1)]),
                               ("w", [(xi + 1, yi), (xi + 1, yi + 1),
                                      (xi, yi + 1)])):
                won = winner(cell, klass)
                if won is None:
                    continue
                arm, margin, tie = won
                base = arm.removeprefix("arts_")
                color = "#BBBBBB" if tie else _FAMILY_COLOR[_family(base)]
                alpha = 0.35 if tie else min(0.35 + 0.13 * margin, 0.95)
                ax.add_patch(Polygon(tri, closed=True, facecolor=color,
                                     alpha=alpha, edgecolor="white", lw=1.2))
                cx = xi + (0.32 if klass == "r" else 0.68)
                cy = yi + (0.30 if klass == "r" else 0.70)
                ax.text(cx, cy, _ARM_ABBR.get(base, base), ha="center",
                        va="center", fontsize=6.5,
                        color="#333333" if tie else "white",
                        fontweight="bold")
    ax.set_xlim(0, len(rhos))
    ax.set_ylim(0, len(sizes))
    ax.set_xticks([i + 0.5 for i in range(len(rhos))])
    ax.set_xticklabels([_rho_label(q) for q in rhos], fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(len(sizes))])
    ax.set_yticklabels([f"{int(s // 1024)}K" if s < 1 << 20
                        else f"{int(s // (1 << 20))}M" for s in sizes],
                       fontsize=8)
    ax.set_xlabel("write share ρ")
    ax.set_ylabel("DB size")
    ax.set_title("regime atlas — upper-left: reader-side winner, "
                 "lower-right: writer-side winner\n(family hue; grey = "
                 "runner-up's rep range overlaps)", fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig_c_regime_atlas.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return [path]


def atlas_regions(run_dirs: list[Path], out_dir: Path) -> list[Path]:
    """The readable headline form of the atlas: one panel per CLASS, each a
    3-color family-region map over (write share x DB size), contiguous
    same-winner regions merged and labeled ONCE.  A cell whose best arm's
    rep range overlaps the best rival family's is a tie and stays pale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cells: dict[tuple[float, float], dict[str, dict[str, list[float]]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rd in run_dirs:
        spec = SweepSpec.model_validate(
            json.loads((rd / "sweep.json").read_text()))
        for r in _ok(_rows(rd, spec)):
            rho = _rho(r)
            try:
                size = float(r.get("knob_BYTES") or 0)
                if (float(r.get("knob_D") or 1) != 1
                        or float(r.get("knob_HSPREAD") or 1) != 1
                        or float(r.get("knob_RSPREAD") or 0) != 0
                        or float(r.get("knob_WSPREAD") or 0) != 0):
                    continue
                R_, W_ = float(r.get("knob_R") or 0), float(r.get("knob_W") or 0)
            except (TypeError, ValueError):
                continue
            if rho is None or int(r.get("nodes") or 8) != 8:
                continue
            cell = cells[(rho, size)][r["arm"]]
            rx, wx = r.get("RXPUT"), r.get("WXPUT")
            if isinstance(rx, float) and R_ > 0:
                cell["r"].append(max(rx, 0.0) / R_)
            if isinstance(wx, float) and W_ > 0:
                cell["w"].append(max(wx, 0.0) / W_)
    if not cells:
        return []
    rhos = sorted({k[0] for k in cells})
    sizes = sorted({k[1] for k in cells})

    def verdict(cell, klass):
        arms = {a: v[klass] for a, v in cell.items() if v.get(klass)}
        if len(arms) < 2:
            return None
        ranked = sorted(arms, key=lambda a: -median(arms[a]))
        b = ranked[0]
        rival = next((a for a in ranked[1:] if _family(a) != _family(b)), None)
        if rival is None:
            return _family(b), b, 1.0, False
        tie = min(arms[b]) <= max(arms[rival])
        return (_family(b), b,
                median(arms[b]) / max(median(arms[rival]), 1e-9), tie)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), sharey=True)
    titles = {"r": "who wins the READERS", "w": "who wins the WRITERS"}
    for ax, klass in zip(axes, ("r", "w")):
        grid: dict[tuple[int, int], tuple] = {}
        for xi, rho in enumerate(rhos):
            for yi, size in enumerate(sizes):
                cell = cells.get((rho, size))
                v = verdict(cell, klass) if cell else None
                if v:
                    grid[(xi, yi)] = v
                    fam, _arm, margin, tie = v
                    color = _FAMILY_COLOR[fam]
                    alpha = 0.22 if tie else min(0.35 + 0.11 * margin, 0.9)
                    ax.add_patch(Rectangle((xi, yi), 1, 1, facecolor=color,
                                           alpha=alpha, edgecolor="white",
                                           lw=1.5))
        # label each contiguous same-family region once, at its centroid
        seen: set[tuple[int, int]] = set()
        for start, v in sorted(grid.items()):
            if start in seen:
                continue
            fam = v[0]
            stack, blob = [start], []
            while stack:
                c = stack.pop()
                if c in seen or grid.get(c, (None,))[0] != fam:
                    continue
                seen.add(c)
                blob.append(c)
                x, y = c
                stack += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            cx = sum(x for x, _ in blob) / len(blob) + 0.5
            cy = sum(y for _, y in blob) / len(blob) + 0.5
            arms_in = {grid[c][1].removeprefix("arts_") for c in blob
                       if not grid[c][3]}
            margins = [grid[c][2] for c in blob if not grid[c][3]]
            if not margins:
                continue  # all-tie blob: color says "close", no claim made
            label = {"excl": "EXCL", "inv": "INV", "val": "VAL"}[fam]
            detail = "/".join(sorted({_ARM_ABBR.get(a, a).split("/", 1)[1]
                                      for a in arms_in}))
            ax.text(cx, cy, f"{label}\n{detail}\n×{median(margins):.1f}",
                    ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.25", fc=_FAMILY_COLOR[fam],
                              ec="white", alpha=0.9))
        ax.set_xlim(0, len(rhos))
        ax.set_ylim(0, len(sizes))
        ax.set_xticks([i + 0.5 for i in range(len(rhos))])
        ax.set_xticklabels([_rho_label(q) for q in rhos], fontsize=8)
        ax.set_xlabel("write share ρ")
        ax.set_title(titles[klass], fontsize=11)
    axes[0].set_yticks([i + 0.5 for i in range(len(sizes))])
    axes[0].set_yticklabels([f"{int(s // 1024)}K" if s < 1 << 20
                             else f"{int(s // (1 << 20))}M" for s in sizes],
                            fontsize=8)
    axes[0].set_ylabel("DB size")
    fig.suptitle("pale = the best rival family's rep range overlaps (too "
                 "close to call)", fontsize=8, y=0.99)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig_c2_region_map.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return [path]


# Coherence-free floor, derived from the cell's own knobs: a chain's op can
# never beat hold + think + the uncontended local machine cost (measured
# ~1.2 us at 1 chain/worker; single-rank cells across sweeps sit within a
# few percent of this bound).  A constant floor would misgrade sweeps whose
# spins differ.
RS_LOCAL_US = 1.2


def _floor(hold_us: float, think_us: float) -> float:
    return 1e6 / (hold_us + think_us + RS_LOCAL_US)


def _bp_index(rec: dict) -> float | None:
    """Balanced-progress index: an application finishes only if BOTH classes
    advance, so a cell's one number is the geometric mean of each populated
    class's per-chain rate against its coherence-free floor.  Starvation
    multiplies in as ~zero instead of hiding behind the majority class."""
    try:
        R_, W_ = float(rec.get("knob_R") or 0), float(rec.get("knob_W") or 0)
        eth = float(rec.get("knob_E_THINK") or 20)
        ehr = float(rec.get("knob_E_HOLD_R") or 1)
        ehw = float(rec.get("knob_E_HOLD_W") or 1)
        rx, wx = rec.get("RXPUT"), rec.get("WXPUT")
    except (TypeError, ValueError):
        return None
    parts = []
    if R_ > 0 and isinstance(rx, float):
        parts.append(max(rx, 0.0) / R_ / _floor(ehr, eth))
    if W_ > 0 and isinstance(wx, float):
        parts.append(max(wx, 0.0) / W_ / _floor(ehw, eth))
    if not parts:
        return None
    prod = 1.0
    for p in parts:
        prod *= p
    return min(prod ** (1.0 / len(parts)), 1.0)


def _fig_index(rows: list[dict], out: Path, plt) -> Path | None:
    """H1: one panel, one line per arm — the balanced-progress index over the
    write share.  Crossovers and starvation cliffs carry the whole story
    without a class split."""
    usable = []
    for r in _ok(rows):
        rho = _rho(r)
        idx = _bp_index(r)
        if rho is not None and idx is not None:
            usable.append(dict(r, rho=rho, BPI=idx))
    if not usable:
        return None
    arms = sorted({r["arm"] for r in usable})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for arm in arms:
        xs, med, lo, hi = _series(
            [r for r in usable if r["arm"] == arm], "rho", "BPI")
        if not xs:
            continue
        st = _style(arm)
        ax.plot(xs, med, marker="o", markersize=3.5, **st)
        ax.fill_between(xs, lo, hi, alpha=0.13, color=st["color"], linewidth=0)
    _rho_axis(ax, sorted({r["rho"] for r in usable}))
    ax.set_yscale("log")
    ax.set_ylabel("balanced progress (1 = coherence-free floor)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.set_title("both classes must advance: geomean of per-chain progress "
                 "vs the no-coherence floor", fontsize=9)
    fig.tight_layout()
    path = out / "fig_h1_balanced_progress.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def atlas_index(run_dirs: list[Path], out_dir: Path) -> list[Path]:
    """H2: ONE region map, no class split — cells colored by the family whose
    best arm has the highest balanced-progress index, labeled per contiguous
    region with the winning arm and its margin over the best rival family."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cells: dict[tuple[float, float], dict[str, list[float]]] = \
        defaultdict(lambda: defaultdict(list))
    for rd in run_dirs:
        spec = SweepSpec.model_validate(
            json.loads((rd / "sweep.json").read_text()))
        for r in _ok(_rows(rd, spec)):
            rho = _rho(r)
            idx = _bp_index(r)
            try:
                size = float(r.get("knob_BYTES") or 0)
                if (float(r.get("knob_D") or 1) != 1
                        or float(r.get("knob_HSPREAD") or 1) != 1
                        or float(r.get("knob_RSPREAD") or 0) != 0
                        or float(r.get("knob_WSPREAD") or 0) != 0):
                    continue
            except (TypeError, ValueError):
                continue
            if rho is None or idx is None or int(r.get("nodes") or 8) != 8:
                continue
            cells[(rho, size)][r["arm"]].append(idx)
    if not cells:
        return []
    rhos = sorted({k[0] for k in cells})
    sizes = sorted({k[1] for k in cells})

    def verdict(cell):
        if len(cell) < 2:
            return None
        ranked = sorted(cell, key=lambda a: -median(cell[a]))
        b = ranked[0]
        rival = next((a for a in ranked[1:] if _family(a) != _family(b)), None)
        if rival is None:
            return _family(b), b, 1.0, False
        tie = min(cell[b]) <= max(cell[rival])
        return (_family(b), b,
                median(cell[b]) / max(median(cell[rival]), 1e-9), tie)

    fig, ax = plt.subplots(figsize=(1.5 * len(rhos) + 2.2,
                                    1.15 * len(sizes) + 1.8))
    grid: dict[tuple[int, int], tuple] = {}
    for xi, rho in enumerate(rhos):
        for yi, size in enumerate(sizes):
            cell = cells.get((rho, size))
            v = verdict(cell) if cell else None
            if v is None:
                continue
            grid[(xi, yi)] = v
            fam, _arm, margin, tie = v
            color = _FAMILY_COLOR[fam]
            alpha = 0.2 if tie else min(0.3 + 0.12 * margin, 0.92)
            ax.add_patch(Rectangle((xi, yi), 1, 1, facecolor=color,
                                   alpha=alpha, edgecolor="white", lw=1.5))
    seen: set[tuple[int, int]] = set()
    for start, v in sorted(grid.items()):
        if start in seen:
            continue
        fam = v[0]
        stack, blob = [start], []
        while stack:
            c = stack.pop()
            if c in seen or grid.get(c, (None,))[0] != fam:
                continue
            seen.add(c)
            blob.append(c)
            x, y = c
            stack += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        margins = [grid[c][2] for c in blob if not grid[c][3]]
        if not margins or median(margins) < 1.15:
            continue
        cx = sum(x for x, _ in blob) / len(blob) + 0.5
        cy = sum(y for _, y in blob) / len(blob) + 0.5
        arms_in = sorted({_ARM_ABBR.get(grid[c][1].removeprefix("arts_"),
                                        grid[c][1]) for c in blob
                          if not grid[c][3]})
        label = {"excl": "EXCL", "inv": "INV", "val": "VAL"}[fam]
        ax.text(cx, cy, f"{label}\n{'/'.join(a.split('/', 1)[1] for a in arms_in)}"
                        f"\n×{median(margins):.1f}",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc=_FAMILY_COLOR[fam],
                          ec="white", alpha=0.92))
    ax.set_xlim(0, len(rhos))
    ax.set_ylim(0, len(sizes))
    ax.set_xticks([i + 0.5 for i in range(len(rhos))])
    ax.set_xticklabels([_rho_label(q) for q in rhos], fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(len(sizes))])
    ax.set_yticklabels([f"{int(s // 1024)}K" if s < 1 << 20
                        else f"{int(s // (1 << 20))}M" for s in sizes],
                       fontsize=8)
    ax.set_xlabel("write share ρ")
    ax.set_ylabel("DB size")
    ax.set_title("who keeps the application moving — winner by balanced "
                 "progress (readers AND writers)\npale = too close to call",
                 fontsize=10)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig_h2_index_map.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return [path]


# The paper's scenario vignettes: plain-language columns, each backed by one
# sweep point.  Resolved against whatever runs are handed in; absent columns
# are skipped, so the ladder grows as campaigns land.
# Column vocabulary is ONE controlled pair of axes: the R:W mix
# (read-heavy / mixed / write-heavy) and the DB size (small=64K / 4M / 16M);
# "producer" marks the write-locality variant (every writer on one rank,
# readers spread — the broadcast pattern); "thin x64" is the 64-rank
# 1-worker shape (coherence structure with no compute to hide behind).
_LADDER_GROUPS = [
    # (group label, [(column label, sweep, point), ...])
    # Core factorial: R:W mix x DB size, everything else at the anchor.
    ("R:W mix \u00d7 size (factorial core)", [
        ("read-heavy\n64K", "s1-ratio", "q1-16"),
        ("mixed\n64K", "s1-ratio", "q1-2"),
        ("write-heavy\n64K", "s1-ratio", "q3-4"),
        ("read-heavy\n4M", "s2-size", "s4m-q1-16"),
        ("mixed\n4M", "s2-size", "s4m-q1-4"),
        ("write-heavy\n4M", "s2-size", "s4m-q3-4"),
        ("read-heavy\n16M", "s2c-16m", "s16m-q1-16"),
        ("mixed\n16M", "s2c-16m", "s16m-q1-2"),
        ("write-only\n64K", "s1-ratio", "q100"),
    ]),
    # Structure extremes: each column flips exactly ONE structural variable.
    ("structure extremes (one variable each)", [
        ("producer\n16M", "s2d-producer", "p-q1-8"),
        ("scan\n64 DBs", "s10-scan", "a-d64-q1-2"),
        ("thin \u00d764\nmixed", "s9-thin", "n64"),
        ("inval bomb\n64K", "s11-bomb", "b-64k"),
        ("inval bomb\n16M", "s11-bomb", "b-16m"),
    ]),
]
_LADDER = [(lbl, sw, pt) for _g, cols in _LADDER_GROUPS for lbl, sw, pt in cols]


def ladder(run_dirs: list[Path], out_dir: Path) -> list[Path]:
    """H3: the scenario ladder — rows are arms, columns are plain-language
    workload vignettes.  The cell metric is AGGREGATE throughput (every
    column runs the identical actor mix, so total ops/s is apples-to-apples
    end-to-end work), normalized to the column's best.  A geomean cell would
    quietly let the minority class decide a column named after the majority;
    instead, starvation is its own mark: a warning corner when either
    populated class runs under 5% of the column's best per-chain rate for
    that class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_sweep: dict[str, list[Path]] = defaultdict(list)
    for rd in run_dirs:
        spec = SweepSpec.model_validate(
            json.loads((rd / "sweep.json").read_text()))
        by_sweep[spec.name].append(rd)

    cols: list[tuple[str, dict[str, float], dict[str, bool]]] = []
    for label, sweep_name, point in _LADDER:
        vals: dict[str, list[float]] = defaultdict(list)
        rrate: dict[str, list[float]] = defaultdict(list)
        wrate: dict[str, list[float]] = defaultdict(list)
        for rd in by_sweep.get(sweep_name, []):
            spec = SweepSpec.model_validate(
                json.loads((rd / "sweep.json").read_text()))
            for r in _ok(_rows(rd, spec)):
                if r["point"] != point:
                    continue
                try:
                    R_ = float(r.get("knob_R") or 0)
                    W_ = float(r.get("knob_W") or 0)
                    rx = max(float(r.get("RXPUT") or 0), 0.0)
                    wx = max(float(r.get("WXPUT") or 0), 0.0)
                except (TypeError, ValueError):
                    continue
                vals[r["arm"]].append(rx + wx)
                if R_ > 0:
                    rrate[r["arm"]].append(rx / R_)
                if W_ > 0:
                    wrate[r["arm"]].append(wx / W_)
        if len(vals) < 2:
            continue
        med = {a: median(v) for a, v in vals.items()}
        starve: dict[str, bool] = {}
        best_r = max((median(v) for v in rrate.values()), default=0.0)
        best_w = max((median(v) for v in wrate.values()), default=0.0)
        for a in med:
            starve[a] = (
                (a in rrate and best_r > 0
                 and median(rrate[a]) < 0.05 * best_r)
                or (a in wrate and best_w > 0
                    and median(wrate[a]) < 0.05 * best_w))
        cols.append((label, med, starve))
    if not cols:
        return []

    arms = sorted({a for _, med, _s in cols for a in med})
    fig, ax = plt.subplots(figsize=(1.35 * len(cols) + 3.4,
                                    0.55 * len(arms) + 1.9))
    for ci, (label, med, starve) in enumerate(cols):
        best = max(med.values())
        for ri, arm in enumerate(arms):
            v = med.get(arm)
            if v is None:
                continue
            rel = v / best if best > 0 else 0.0
            fam = _family(arm.removeprefix("arts_"))
            ax.add_patch(plt.Rectangle(
                (ci, ri), 1, 1,
                facecolor=_FAMILY_COLOR[fam] if rel > 0.85 else "#888888",
                alpha=0.9 if rel > 0.85 else 0.06 + 0.5 * rel,
                edgecolor="white", lw=1.2))
            text = "1.0" if rel > 0.999 else (
                f"{1 / rel:.1f}x" if rel > 0.005 else "stall")
            ax.text(ci + 0.5, ri + 0.5, text, ha="center", va="center",
                    fontsize=8.5,
                    fontweight="bold" if rel > 0.85 else "normal",
                    color="white" if rel > 0.85 else "#333333")
            if starve.get(arm):
                # one class starved (under 5% of the column's best per-chain
                # rate for that class): a corner wedge, not a metric penalty
                ax.add_patch(plt.Polygon(
                    [(ci + 1, ri), (ci + 0.72, ri), (ci + 1, ri + 0.28)],
                    closed=True, facecolor="#C0392B", edgecolor="white",
                    lw=0.5))
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(arms))
    ax.set_xticks([i + 0.5 for i in range(len(cols))])
    ax.set_xticklabels([c[0] for c in cols], fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(len(arms))])
    ax.set_yticklabels([RT_LABEL.get(rt_key(a), a) for a in arms], fontsize=8)
    ax.invert_yaxis()
    _ladder_group_bands(ax, [c[0] for c in cols], len(arms))
    ax.set_title("how many times slower than the scenario's winner "
                 "(aggregate throughput; colored = within 15% of the winner; "
                 "red corner = one class starved, under 5% of the column's "
                 "best per-chain rate)", fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig_h3_scenario_ladder.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return [path]


def scaling_fig(run_dirs: list[Path], out_dir: Path) -> list[Path]:
    """H4: strong-scaling of the coherence structure itself — balanced
    progress against rank count, per-PD load held constant.  Two panels
    because the two rank shapes answer different questions: fat ranks
    (15w+1p) ask how coherence waits amplify through a rank's concurrent
    outstanding ops; thin ranks (1w+1p) ask what survives when there is no
    local compute to hide wire latency behind."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = {"s5-nodes": ("fat ranks (15w+1p)", defaultdict(lambda: defaultdict(list))),
              "s9-thin": ("thin ranks (1w+1p)", defaultdict(lambda: defaultdict(list)))}
    for rd in run_dirs:
        spec = SweepSpec.model_validate(
            json.loads((rd / "sweep.json").read_text()))
        if spec.name not in panels:
            continue
        for r in _ok(_rows(rd, spec)):
            if r.get("group") in ("thinloc",):
                continue  # producer twins live in the ladder, not here
            idx = _bp_index(r)
            if idx is None:
                continue
            panels[spec.name][1][r["arm"]][int(r["nodes"])].append(idx)
    live = [(title, data) for name, (title, data) in panels.items() if data]
    if not live:
        return []
    fig, axes = plt.subplots(1, len(live), figsize=(6.2 * len(live), 4.4),
                             sharey=True)
    if len(live) == 1:
        axes = [axes]
    for ax, (title, data) in zip(axes, live):
        for arm in sorted(data):
            ns = sorted(data[arm])
            med = [median(data[arm][n]) for n in ns]
            lo = [min(data[arm][n]) for n in ns]
            hi = [max(data[arm][n]) for n in ns]
            st = _style(arm)
            ax.plot(ns, med, marker="o", markersize=4, **st)
            ax.fill_between(ns, lo, hi, alpha=0.12, color=st["color"],
                            linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sorted({n for d in data.values() for n in d}))
        ax.set_xticklabels([str(n) for n in
                            sorted({n for d in data.values() for n in d})])
        ax.minorticks_off()
        ax.set_xlabel("ranks")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("balanced progress (1 = no-coherence floor)")
    axes[-1].legend(fontsize=6.5, ncol=2, loc="best")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig_h4_strong_scaling.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return [path]


def tail_ladder(run_dirs: list[Path], out_dir: Path) -> list[Path]:
    """H3b: the bill behind the throughput crown — the same scenario columns,
    but the cell is the WRITER's p99 per-op overhead relative to the column's
    best (lower is better).  Aggregate throughput structurally cannot carry
    write-tail pain (writers are the minority in most mixes); this companion
    is where a blocking round's cost stops hiding."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_sweep: dict[str, list[Path]] = defaultdict(list)
    for rd in run_dirs:
        spec = SweepSpec.model_validate(
            json.loads((rd / "sweep.json").read_text()))
        by_sweep[spec.name].append(rd)

    cols = []
    for label, sweep_name, point in _LADDER:
        vals: dict[str, list[float]] = defaultdict(list)
        for rd in by_sweep.get(sweep_name, []):
            spec = SweepSpec.model_validate(
                json.loads((rd / "sweep.json").read_text()))
            for r in _ok(_rows(rd, spec)):
                if r["point"] != point:
                    continue
                try:
                    if float(r.get("knob_W") or 0) <= 0:
                        continue
                    v = float(r.get("WTOT_P99") or -1)
                except (TypeError, ValueError):
                    continue
                if v > 0:
                    vals[r["arm"]].append(v)
        if len(vals) >= 2:
            cols.append((label, {a: median(v) for a, v in vals.items()}))
    if not cols:
        return []
    arms = sorted({a for _, med in cols for a in med})
    fig, ax = plt.subplots(figsize=(1.35 * len(cols) + 3.4,
                                    0.55 * len(arms) + 1.9))
    for ci, (label, med) in enumerate(cols):
        best = min(med.values())
        for ri, arm in enumerate(arms):
            v = med.get(arm)
            if v is None:
                ax.text(ci + 0.5, ri + 0.5, "starved", ha="center",
                        va="center", fontsize=7.5, color="#888888")
                continue
            rel = v / best if best > 0 else float("inf")
            fam = _family(arm.removeprefix("arts_"))
            good = rel < 1.15
            ax.add_patch(plt.Rectangle(
                (ci, ri), 1, 1,
                facecolor=_FAMILY_COLOR[fam] if good else "#888888",
                alpha=0.9 if good else min(0.06 + 0.09 * rel, 0.55),
                edgecolor="white", lw=1.2))
            ax.text(ci + 0.5, ri + 0.5,
                    "1.0" if rel < 1.001 else f"{rel:.0f}x" if rel >= 10
                    else f"{rel:.1f}x",
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if good else "normal",
                    color="white" if good else "#333333")
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(arms))
    ax.set_xticks([i + 0.5 for i in range(len(cols))])
    ax.set_xticklabels([c[0] for c in cols], fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(len(arms))])
    ax.set_yticklabels([RT_LABEL.get(rt_key(a), a) for a in arms], fontsize=8)
    ax.invert_yaxis()
    _ladder_group_bands(ax, [c[0] for c in cols], len(arms))
    ax.set_title("write-tail bill: writer per-op overhead p99, times the "
                 "scenario's best (lower is better; colored = within 15%)",
                 fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fig_h3b_tail_ladder.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return [path]


def _ladder_group_bands(ax, col_labels: list[str], n_rows: int) -> None:
    """Draw the scenario-group separators and headers: the x-axis is designed,
    not collected — a factorial core (mix x size at the anchor) and a set of
    one-variable structural extremes, and the bands say which is which."""
    pos = 0
    for glabel, cols in _LADDER_GROUPS:
        present = [c for c in cols if c[0] in col_labels]
        if not present:
            continue
        width = len(present)
        if pos > 0:
            ax.axvline(pos, color="#333333", lw=2.0)
        ax.text(pos + width / 2, -0.45, glabel, ha="center", va="bottom",
                fontsize=8.5, fontstyle="italic", color="#555555")
        pos += width


# ---------------------------------------------------------------------------
# The e2e master table: one figure over the fixed-work e2e campaign — every
# cell is a fixed job and the number is the runtime's own end-to-end stamp,
# so a single unit spans the whole suite.  Columns name the sweep run and
# point they come from; the latest run of each sweep wins unless an explicit
# run list narrows the search.

E2E_MASTER_COLUMNS = [
    ("s17-e2e-priv", "rd-64k", "own re-read 64K"),
    ("s17-e2e-priv", "rd-16m", "own re-read 16M"),
    ("s17-e2e-priv", "wr-64k", "own re-write 64K"),
    ("s17-e2e-priv", "wr-16m", "own re-write 16M"),
    ("s18-e2e-steady", "funnel-4m", "one-home funnel 4M"),
    ("s18-e2e-steady", "purity-16m", "audienceless publish 16M"),
    ("s18b-e2e-mixed", "freerun-rh-64k", "free-run read-heavy 64K"),
    ("s18c-e2e-pipe", "pipe-1m", "pipeline converge 1M"),
    ("s18b-e2e-mixed", "bomb-64k", "alternating bomb 64K"),
    ("s14b-kill-fixed", "c5-e2e", "resident mill 1M"),
    ("s19-e2e-last", "a9-migrate", "migration gauntlet 64K"),
    ("s19-e2e-last", "c6-slow", "slow churn dial 16M"),
]

E2E_ARMS = ["arts_val_wt", "arts_val_wb", "arts_val_wt_purge", "arts_inv_wt",
            "arts_inv_wb", "arts_inv_wt_purge", "arts_excl_purge",
            "arts_excl_retain"]


def _e2e_collect(run_dir: Path, point: str) -> dict[str, list[float]]:
    import json as _json
    import re as _re
    out: dict[str, list[float]] = {}
    track = run_dir / "track.jsonl"
    if not track.exists():
        return out
    for line in track.read_text().splitlines():
        ev = _json.loads(line)
        if ev.get("event") != "finished" or ev.get("status") != "ok":
            continue
        m = _re.match(r".+?\+(.+?):.*/(arts_\w+)#", ev["cell"])
        if not m or m.group(1) != point:
            continue
        if ev.get("e2e_s") is not None:
            out.setdefault(m.group(2), []).append(float(ev["e2e_s"]))
    return out


def e2e_master(exp_root: Path, out_dir: Path,
               runs: list[Path] | None = None) -> list[Path]:
    import statistics
    from artsrun.report import RT_LABEL

    out_dir.mkdir(parents=True, exist_ok=True)
    table: list[tuple[str, dict[str, float | None]]] = []
    for sweep, point, label in E2E_MASTER_COLUMNS:
        cands = (runs if runs is not None
                 else sorted(exp_root.glob(f"*sweep-{sweep}*")))
        cands = [c for c in cands if c.name.find(sweep) >= 0]
        if not cands:
            continue
        cells = _e2e_collect(cands[-1], point)
        med = {a: (statistics.median(v) if (v := cells.get(a)) else None)
               for a in E2E_ARMS}
        if any(v is not None for v in med.values()):
            table.append((label, med))

    if not table:
        raise SystemExit("no e2e master runs found under the experiment root")
    written: list[Path] = []
    csv = out_dir / "e2e-master.csv"
    with csv.open("w") as f:
        f.write("column,arm,e2e_s,ratio_to_best\n")
        for label, med in table:
            ok = [v for v in med.values() if v]
            best = min(ok) if ok else 1.0
            for a in E2E_ARMS:
                v = med[a]
                f.write(f"{label},{a},"
                        f"{'' if v is None else f'{v:.4f}'},"
                        f"{'' if v is None else f'{v / best:.3f}'}\n")
    written.append(csv)

    dat = out_dir / "e2e-master.dat"
    with dat.open("w") as f:
        f.write("# col arm e2e_s ratio\n")
        for ci, (label, med) in enumerate(table):
            ok = [v for v in med.values() if v]
            best = min(ok) if ok else 1.0
            for ai, a in enumerate(E2E_ARMS):
                v = med[a]
                if v is not None:
                    f.write(f"{ci} {ai} {v:.4f} {v / best:.3f}\n")
    written.append(dat)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_r, n_c = len(E2E_ARMS), len(table)
    ratio = np.full((n_r, n_c), np.nan)
    for ci, (_, med) in enumerate(table):
        ok = [v for v in med.values() if v]
        best = min(ok) if ok else 1.0
        for ai, a in enumerate(E2E_ARMS):
            if med[a] is not None:
                ratio[ai, ci] = med[a] / best
    fig, ax = plt.subplots(figsize=(2.1 + 1.15 * n_c, 0.62 * n_r + 2.2))
    im = ax.imshow(np.log10(ratio), cmap="RdYlGn_r", vmin=0,
                   vmax=np.log10(np.nanmax(ratio)), aspect="auto")
    ax.set_xticks(range(n_c))
    ax.set_xticklabels([lbl for lbl, _ in table], rotation=35, ha="right",
                       fontsize=8)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels([RT_LABEL.get(a, a) for a in E2E_ARMS], fontsize=9)
    for ai in range(n_r):
        for ci in range(n_c):
            r = ratio[ai, ci]
            if np.isnan(r):
                ax.text(ci, ai, "off", ha="center", va="center", fontsize=7)
            else:
                ax.text(ci, ai, "1.0" if r < 1.005 else
                        (f"{r:.1f}x" if r < 100 else f"{r:.0f}x"),
                        ha="center", va="center", fontsize=7,
                        color="black")
    fig.colorbar(im, ax=ax, shrink=0.75,
                 label="log10( e2e / column best )")
    ax.set_title("Fixed-work e2e across the coherence plane "
                 "(each column one job; 1.0 = column crown)")
    fig.tight_layout()
    png = out_dir / "e2e-master.png"
    fig.savefig(png, dpi=170)
    plt.close(fig)
    written.append(png)
    return written
