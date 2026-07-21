#!/usr/bin/env python3
"""analyze_perf.py — turn a matrix results.csv (+ metrics.json) into:
  (1) per-app scaling tables (runtime x config grid of e2e seconds),
  (2) protocol cross-comparison (baseline ranking + eager/lazy + arts/ref ratios),
  (3) coherence-counter analysis (per-rank EDT spread + network bytes -> comm-cliff),
  (4) a plot-friendly summary CSV + a matplotlib plotting script.

Usage: python3 analyze_perf.py <matrix_dir>   (dir with results.csv + metrics.json)
Writes <matrix_dir>/analysis/{report.txt, summary.csv, plot_scaling.py}.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

RUNTIME_ORDER = ["ocr_rcu_eager", "ocr_rcu_lazy", "wrf_rcu_eager",
                 "ocr_rwlock_eager", "ocr_rwlock_lazy", "xsocr", "ocrvx"]
CFG_ORDER = {
    "strong": ["1n_sc", "2n_sc", "4n_sc", "8n_sc"],
    "weak":   ["1n_sc", "2n_sc", "4n_sc", "8n_sc"],
    "fix":    ["1n", "2", "4", "8", "16", "2n_io", "4n_io", "8n_io"],
}
EXPERIMENTS = ["fix", "strong", "weak"]


def _num(v):
    """Counter value may be a scalar or a {'value': x}/{'value_ms': x} dict."""
    if isinstance(v, dict):
        return v.get("value", v.get("value_ms", 0))
    return v or 0


def load(matrix_dir):
    md = Path(matrix_dir)
    rows = list(csv.DictReader(open(md / "results.csv")))
    metrics = []
    mp = md / "metrics.json"
    if mp.exists():
        try:
            metrics = json.loads(mp.read_text())
        except ValueError:
            metrics = []
    return rows, metrics


def e2e_s(r):
    if r["status"] == "FAIL" or not r["e2e_ns"]:
        return None
    try:
        return float(r["e2e_ns"]) / 1e9
    except ValueError:
        return None


def cell(rows_by):
    """rows_by[(exp,bench,cfg,rt)] -> the e2e in s (or None if FAIL/absent)."""
    d = {}
    for r in rows_by:
        k = (r["experiment"], r["bench"], r["config"], r["runtime"])
        d[k] = e2e_s(r)
    return d


def fmt(v):
    if v is None:
        return "   TO"          # timeout / fail
    return f"{v:6.1f}"


def section(title):
    return f"\n{'='*78}\n{title}\n{'='*78}\n"


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def report_single(rows, outdir):
    """The "single" experiment: every app at 1n x 48 workers, arts(ocr_rcu_lazy)
    vs xsocr vs ocr-vx.  Rows carry one line per accepted ITERATION, so each
    cell aggregates to median/min/max.  Emits a per-app comparison table
    sorted by the worst arts-vs-ref ratio, plus the arts teardown gap
    (median wall - median e2e) which the e2e span excludes."""
    per = defaultdict(list)          # (bench, rt) -> [(e2e_s, wall_s, status)]
    for r in rows:
        if r["experiment"] != "single":
            continue
        e = e2e_s(r)
        try:
            w = float(r["wall"])
        except (TypeError, ValueError):
            w = None
        per[(r["bench"], r["runtime"])].append((e, w, r["status"]))

    benches = sorted({b for (b, _) in per})
    rts = ["ocr_rcu_lazy", "xsocr", "ocrvx"]

    def agg(bench, rt):
        entries = per.get((bench, rt), [])
        es = [e for (e, _, _) in entries if e is not None]
        ws = [w for (_, w, s) in entries if w is not None]
        statuses = [s for (_, _, s) in entries]
        return {
            "med": _median(es), "min": min(es) if es else None,
            "max": max(es) if es else None, "wall_med": _median(ws),
            "n_ok": len(es), "n": len(statuses),
            "verdict": ("OK" if es else
                        ("TIMEOUT" if "TIMEOUT" in statuses else
                         ("FAIL" if statuses else "-"))),
        }

    def f(v, w=8):
        return f"{v:{w}.2f}" if v is not None else " " * (w - 4) + "  - "

    lines = [section("SINGLE-NODE 48T 3-RUNTIME COMPARISON (median e2e s; "
                     "x/A = runtime/arts ratio; teardown = arts wall-e2e)")]
    hdr = (f"  {'bench':<28}{'arts':>8}{'xsocr':>8}{'x/A':>7}{'ocrvx':>8}"
           f"{'o/A':>7}{'teardown':>9}  notes")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    def ratio(a, b):
        return (b / a) if (a and b) else None

    scored = []
    for bench in benches:
        A, X, O = (agg(bench, rt) for rt in rts)
        rx, ro = ratio(A["med"], X["med"]), ratio(A["med"], O["med"])
        # sort key: worst case for arts (smallest ref/arts ratio) first
        worst = min([r for r in (rx, ro) if r is not None], default=None)
        scored.append((worst if worst is not None else 99.0, bench, A, X, O, rx, ro))
    scored.sort()

    for _, bench, A, X, O, rx, ro in scored:
        notes = []
        for rt, G in (("arts", A), ("xsocr", X), ("ocrvx", O)):
            if G["verdict"] in ("FAIL", "TIMEOUT"):
                notes.append(f"{rt}:{G['verdict']}")
            elif 0 < G["n_ok"] < G["n"]:
                notes.append(f"{rt}:{G['n_ok']}/{G['n']}ok")
        tear = (A["wall_med"] - A["med"]) if (A["wall_med"] and A["med"]) else None
        lines.append(
            f"  {bench:<28}{f(A['med'])}{f(X['med'])}"
            f"{(f'{rx:5.1f}x' if rx else '    - '):>7}{f(O['med'])}"
            f"{(f'{ro:5.1f}x' if ro else '    - '):>7}"
            f"{f(tear, 9)}  {' '.join(notes)}")
    return "\n".join(lines)


def main():
    matrix_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "benchmarks/scripts/logs/perf/matrix6app"
    rows, metrics = load(matrix_dir)
    outdir = Path(matrix_dir) / "analysis"
    outdir.mkdir(exist_ok=True)

    if any(r["experiment"] == "single" for r in rows):
        report = report_single(rows, outdir)
        (outdir / "report_single.txt").write_text(report)
        print(report)
        print(f"\n[analysis] wrote {outdir}/report_single.txt")
        if all(r["experiment"] == "single" for r in rows):
            return

    benches = sorted({r["bench"] for r in rows})
    E = cell(rows)                          # (exp,bench,cfg,rt) -> e2e_s or None
    present = {(r["experiment"], r["bench"], r["config"], r["runtime"]) for r in rows}

    out = []

    # ---- (1)+(2) per-app scaling grid (runtime x config) = also protocol compare ----
    out.append(section("(1)+(2)  PER-APP SCALING GRID  (e2e seconds; 'TO'=300s timeout; "
                        "'-'=not run)"))
    for bench in benches:
        for exp in EXPERIMENTS:
            cfgs = CFG_ORDER[exp]
            # skip if this bench/exp has no rows
            if not any((exp, bench, c, rt) in present for c in cfgs for rt in RUNTIME_ORDER):
                continue
            out.append(f"\n### {bench}  [{exp}]   (columns = node configs ->)")
            hdr = "  runtime        " + "".join(f"{c:>8}" for c in cfgs)
            out.append(hdr)
            out.append("  " + "-" * (len(hdr) - 2))
            for rt in RUNTIME_ORDER:
                cells = []
                any_here = False
                for c in cfgs:
                    k = (exp, bench, c, rt)
                    if k in present:
                        any_here = True
                        cells.append(fmt(E.get(k)))
                    else:
                        cells.append("     -")
                if any_here:
                    out.append(f"  {rt:<14}" + "".join(f"{x:>8}" for x in cells))

    # ---- (2b) baseline protocol ranking + eager/lazy + arts-vs-ref ratios ----
    out.append(section("(2b)  PROTOCOL CROSS-COMPARISON  (baseline config per experiment)"))
    base_cfg = {"strong": "1n_sc", "weak": "1n_sc", "fix": "1n"}
    for exp in EXPERIMENTS:
        bc = base_cfg[exp]
        out.append(f"\n### {exp}  @ {bc}   e2e(s), and eager/lazy ratio, arts_lazy vs xsocr/ocrvx")
        out.append(f"  {'bench':<26}{'rcu_l':>9}{'rcu_e':>9}{'e/l':>6}"
                   f"{'lock_l':>9}{'lock_e':>9}{'xsocr':>9}{'ocrvx':>9}")
        for bench in benches:
            g = lambda rt: E.get((exp, bench, bc, rt))
            ml, me = g("ocr_rcu_lazy"), g("ocr_rcu_eager")
            ll, le = g("ocr_rwlock_lazy"), g("ocr_rwlock_eager")
            xs, ov = g("xsocr"), g("ocrvx")
            el = f"{me/ml:.1f}x" if (ml and me) else "  -"
            out.append(f"  {bench:<26}{fmt(ml):>9}{fmt(me):>9}{el:>6}"
                       f"{fmt(ll):>9}{fmt(le):>9}{fmt(xs):>9}{fmt(ov):>9}")

    # ---- (3) coherence-counter analysis: network + EDT spread -> comm-cliff ----
    out.append(section("(3)  COHERENCE COUNTERS  (arts ocr_rcu_lazy; total remote bytes & "
                       "per-rank EDT spread)"))
    # index metrics by (exp,bench,cfg,rt)
    M = {(m["experiment"], m["bench"], m["config"], m["runtime"]): m for m in metrics}
    for bench in benches:
        for exp in ("strong", "weak"):
            cfgs = CFG_ORDER[exp]
            hdr_done = False
            for c in cfgs:
                m = M.get((exp, bench, c, "ocr_rcu_lazy"))
                if not m:
                    continue
                if not hdr_done:
                    out.append(f"\n### {bench}  [{exp}]  (ocr_rcu_lazy)")
                    out.append(f"  {'config':>7}{'e2e_s':>8}{'remote_MB':>11}"
                               f"{'remote_sends':>13}{'edt_finish(min..max/rank)':>28}")
                    hdr_done = True
                pr = m.get("per_rank", {})
                tot_bytes = sum(_num(pr[r].get("BYTES_REMOTE_SENT")) for r in pr)
                tot_sends = sum(_num(pr[r].get("NUM_REMOTE_SEND")) for r in pr)
                fins = [_num(pr[r].get("NUM_EDT_FINISH")) for r in pr] or [0]
                e = e2e_s({"status": "OK", "e2e_ns": m.get("e2e_ns") or 0})
                spread = f"{min(fins)}..{max(fins)} ({len(pr)}r)"
                out.append(f"  {c:>7}{(f'{e:.1f}' if e else '-'):>8}"
                           f"{tot_bytes/1e6:>11.1f}{tot_sends:>13}{spread:>28}")

    report = "\n".join(out)
    (outdir / "report.txt").write_text(report)

    # ---- (4a) plot-friendly summary CSV ----
    with open(outdir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "bench", "config", "runtime", "e2e_s",
                    "status", "kernel2_s", "mteps"])
        for r in rows:
            w.writerow([r["experiment"], r["bench"], r["config"], r["runtime"],
                        (f"{e2e_s(r):.3f}" if e2e_s(r) is not None else ""),
                        r["status"], r.get("kernel2_ns", ""), r.get("mteps", "")])

    # ---- (4b) plotting script ----
    (outdir / "plot_scaling.py").write_text(PLOT_SCRIPT)

    print(report)
    print(f"\n[analysis] wrote {outdir}/report.txt, summary.csv, plot_scaling.py")


PLOT_SCRIPT = '''#!/usr/bin/env python3
"""Scaling-curve plots from summary.csv. One PNG per (bench, experiment):
x = node config, y = e2e (s), one line per runtime (timeouts drop out)."""
import csv, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG = {"strong": ["1n_sc","2n_sc","4n_sc","8n_sc"],
       "weak": ["1n_sc","2n_sc","4n_sc","8n_sc"],
       "fix": ["1n","2","4","8","16","2n_io","4n_io","8n_io"]}
here = sys.argv[1] if len(sys.argv) > 1 else "summary.csv"
rows = list(csv.DictReader(open(here)))
data = defaultdict(dict)  # (exp,bench,rt) -> {cfg: e2e}
for r in rows:
    if r["e2e_s"]:
        data[(r["experiment"], r["bench"], r["runtime"])][r["config"]] = float(r["e2e_s"])
benches = sorted({r["bench"] for r in rows})
exps = ["fix", "strong", "weak"]
for bench in benches:
    for exp in exps:
        cfgs = CFG[exp]
        series = {rt: data[(exp, bench, rt)] for _, b, rt in
                  {(exp, bench, r["runtime"]) for r in rows if r["bench"]==bench}
                  if data.get((exp, bench, rt))}
        if not series:
            continue
        plt.figure(figsize=(7, 4.5))
        for rt, d in sorted(series.items()):
            xs = [c for c in cfgs if c in d]
            ys = [d[c] for c in xs]
            if xs:
                plt.plot(range(len(xs)), ys, marker="o", label=rt)
        plt.xticks(range(len(cfgs)), cfgs, rotation=30)
        plt.ylabel("e2e (s)"); plt.title(f"{bench} [{exp}]")
        plt.legend(fontsize=7, ncol=2); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"scaling_{bench}_{exp}.png", dpi=110)
        plt.close()
print("wrote scaling_<bench>_<exp>.png")
'''


if __name__ == "__main__":
    main()
