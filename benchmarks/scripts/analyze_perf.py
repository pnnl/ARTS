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

RUNTIME_ORDER = ["mrnew_eager", "mrnew_lazy", "mrsw_eager", "mrsw_lazy", "mrmw",
                 "lock_eager", "lock_lazy", "xsocr", "ocrvx"]
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


def main():
    matrix_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "benchmarks/scripts/logs/perf/matrix6app"
    rows, metrics = load(matrix_dir)
    outdir = Path(matrix_dir) / "analysis"
    outdir.mkdir(exist_ok=True)

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
        out.append(f"  {'bench':<26}{'mrnew_l':>9}{'mrnew_e':>9}{'e/l':>6}"
                   f"{'lock_l':>9}{'lock_e':>9}{'xsocr':>9}{'ocrvx':>9}")
        for bench in benches:
            g = lambda rt: E.get((exp, bench, bc, rt))
            ml, me = g("mrnew_lazy"), g("mrnew_eager")
            ll, le = g("lock_lazy"), g("lock_eager")
            xs, ov = g("xsocr"), g("ocrvx")
            el = f"{me/ml:.1f}x" if (ml and me) else "  -"
            out.append(f"  {bench:<26}{fmt(ml):>9}{fmt(me):>9}{el:>6}"
                       f"{fmt(ll):>9}{fmt(le):>9}{fmt(xs):>9}{fmt(ov):>9}")

    # ---- (3) coherence-counter analysis: network + EDT spread -> comm-cliff ----
    out.append(section("(3)  COHERENCE COUNTERS  (arts mrnew_lazy; total remote bytes & "
                       "per-rank EDT spread)"))
    # index metrics by (exp,bench,cfg,rt)
    M = {(m["experiment"], m["bench"], m["config"], m["runtime"]): m for m in metrics}
    for bench in benches:
        for exp in ("strong", "weak"):
            cfgs = CFG_ORDER[exp]
            hdr_done = False
            for c in cfgs:
                m = M.get((exp, bench, c, "mrnew_lazy"))
                if not m:
                    continue
                if not hdr_done:
                    out.append(f"\n### {bench}  [{exp}]  (mrnew_lazy)")
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
