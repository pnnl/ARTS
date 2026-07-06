#!/usr/bin/env python3
"""recal_check.py <single_results.csv> -- flag apps whose 1n (48w) workload is
misconfigured, judged ONLY by the calibration-baseline protocol mrnew_lazy.

User rule (2026-07-06): an app whose mrnew_lazy 1n cell TIMEOUTs (or whose
mrnew_lazy 1n median falls far outside the ~3 s target: <1.5 s or >6 s) was
sized wrong from the start -> retune to a natural number, invalidate its cells,
re-measure.  BUT if mrnew_lazy is fine (~3 s) and only an eager variant or a
reference runtime (xsocr/ocr-vx) TIMEOUTs, that is a protocol/runtime
measurement, NOT a parameter problem -> keep it (do not retune; the baseline is
mrnew_lazy).

This script only REPORTS; it prints a flag table and writes recal_flags.txt.
Retuning is a human/orchestrator judgement (pick the natural value, edit
fix_args, invalidate rows) -- see the ledger recalibration table.
"""
import csv, sys, statistics as st
from pathlib import Path

LO, HI, TIMEOUT_S = 1.5, 6.0, None  # TIMEOUT detected by status, not a number


def main():
    csv_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "logs/perf/recal_flags.txt"
    base = {}   # bench -> list of mrnew_lazy 1n e2e seconds (OK cells)
    to = {}     # bench -> saw a mrnew_lazy 1n TIMEOUT/FAIL
    other_to = {}  # bench -> runtimes (non-mrnew_lazy) that TIMEOUT at 1n
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["config"] != "1n":
                continue
            rt, statv = r["runtime"], r["status"]
            if rt == "mrnew_lazy":
                if statv in ("TIMEOUT", "FAIL"):
                    to[r["bench"]] = True
                elif r["e2e_ns"]:
                    base.setdefault(r["bench"], []).append(float(r["e2e_ns"]) / 1e9)
            elif statv in ("TIMEOUT", "FAIL"):
                other_to.setdefault(r["bench"], set()).add(rt)

    flags = []
    for bench in sorted(set(base) | set(to)):
        if to.get(bench):
            flags.append((bench, "TIMEOUT", "mrnew_lazy 1n TIMEOUT/FAIL -> retune"))
            continue
        es = base.get(bench, [])
        if not es:
            continue
        med = st.median(es)
        if med > HI:
            flags.append((bench, f"{med:.1f}s", f"mrnew_lazy 1n median > {HI}s -> retune"))
        elif med < LO:
            flags.append((bench, f"{med:.1f}s", f"mrnew_lazy 1n median < {LO}s -> (fixed-size? else retune)"))

    lines = ["# recal flags (baseline = mrnew_lazy 1n 48w)"]
    if not flags:
        lines.append("NONE -- all mrnew_lazy 1n medians within [1.5, 6] s; no parameter retune.")
    for b, val, why in flags:
        lines.append(f"FLAG {b}\t{val}\t{why}")
    # informational: protocol-only timeouts (NOT parameter issues, do NOT retune)
    proto_only = {b: rts for b, rts in other_to.items() if not to.get(b) and base.get(b)}
    if proto_only:
        lines.append("\n# protocol/runtime-only 1n TIMEOUTs (measurement, NOT a param issue -- keep):")
        for b, rts in sorted(proto_only.items()):
            lines.append(f"  {b}: {','.join(sorted(rts))}")
    Path(out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
