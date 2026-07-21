#!/usr/bin/env python3
"""audit_rank_distribution.py — flag EDT/DB placement skew across ranks.

Consumes the perf harness's metrics.json (per-cell ARTS counter snapshots,
{"per_rank": {rank: {metric: value}}}) and reports, for every multi-rank cell,
how evenly work landed across ranks. Placement policy (hint -> rank, no-hint ->
round-robin) should keep creation counts near-uniform; a heavy skew means an
app is funneling work to one rank (e.g. caller-rank-only wiring) or a placement
regression.

Verdicts per (bench, config, runtime), medianed across iters:
  BALANCED   max_share <= warn threshold
  SKEW       max rank holds > warn-factor x mean (default 2x)
  DEAD-RANK  some rank finished 0 EDTs while others worked (worst skew)

Usage: audit_rank_distribution.py <metrics.json|dir>... [--out report.txt]
                                  [--warn-factor 2.0]
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path

METRICS = ["NUM_EDT_CREATE", "NUM_EDT_FINISH", "NUM_DB_CREATE"]


def _load_rows(paths):
    rows = []
    for p in paths:
        p = Path(p)
        f = p / "metrics.json" if p.is_dir() else p
        if not f.exists():
            continue
        try:
            rows.extend(json.loads(f.read_text()))
        except (OSError, ValueError) as e:
            print(f"[audit] unreadable {f}: {e}", file=sys.stderr)
    return rows


def _cell_shares(row, metric):
    per_rank = row.get("per_rank") or {}
    vals = {int(r): d.get(metric) for r, d in per_rank.items()
            if isinstance(d, dict) and d.get(metric) is not None}
    if len(vals) < 2:
        return None
    total = sum(vals.values())
    if total == 0:
        return None
    return {r: v / total for r, v in vals.items()}, vals


def audit(rows, warn_factor):
    cells = {}
    for row in rows:
        key = (row["experiment"], row["bench"], row["config"], row["runtime"])
        cells.setdefault(key, []).append(row)

    results, n_single = [], 0
    for key, its in sorted(cells.items()):
        per_metric = {}
        for metric in METRICS:
            max_shares, dead = [], False
            nrank = 0
            for row in its:
                got = _cell_shares(row, metric)
                if not got:
                    continue
                shares, vals = got
                nrank = len(shares)
                max_shares.append(max(shares.values()))
                if metric == "NUM_EDT_FINISH" and min(vals.values()) == 0:
                    dead = True
            if not max_shares:
                continue
            med = statistics.median(max_shares)
            mean_share = 1.0 / nrank
            verdict = "BALANCED"
            if dead:
                verdict = "DEAD-RANK"
            elif med > warn_factor * mean_share:
                verdict = "SKEW"
            per_metric[metric] = (verdict, med, nrank)
        if not per_metric:
            n_single += 1
            continue
        worst = max(per_metric.values(),
                    key=lambda t: {"BALANCED": 0, "SKEW": 1, "DEAD-RANK": 2}[t[0]])
        results.append((key, worst[0], per_metric))
    return results, n_single


def main():
    argv = sys.argv[1:]
    out_path, warn = None, 2.0
    if "--out" in argv:
        i = argv.index("--out"); out_path = argv[i + 1]; del argv[i:i + 2]
    if "--warn-factor" in argv:
        i = argv.index("--warn-factor"); warn = float(argv[i + 1]); del argv[i:i + 2]

    rows = _load_rows(argv)
    results, n_single = audit(rows, warn)

    lines = [f"RANK DISTRIBUTION AUDIT  ({len(rows)} metric rows; "
             f"{n_single} single-rank cells skipped; warn-factor {warn})", ""]
    flagged = [r for r in results if r[1] != "BALANCED"]
    lines.append(f"== FLAGGED ({len(flagged)}) ==")
    for (exp, bench, cfg, rt), verdict, pm in flagged:
        det = "  ".join(f"{m.split('NUM_')[1]}:{v[0]} max-share {v[1]:.2f}/{1/v[2]:.2f}"
                        for m, v in pm.items())
        lines.append(f"{verdict:9s} {exp}/{bench}/{cfg}/{rt}  {det}")
    lines.append("")
    lines.append(f"== ALL MULTI-RANK CELLS ({len(results)}) ==")
    for (exp, bench, cfg, rt), verdict, pm in results:
        m = pm.get("NUM_EDT_CREATE") or next(iter(pm.values()))
        lines.append(f"{verdict:9s} {exp}/{bench}/{cfg}/{rt}  "
                     f"EDT max-share {m[1]:.2f} (uniform {1/m[2]:.2f}, {m[2]} ranks)")
    text = "\n".join(lines) + "\n"
    if out_path:
        Path(out_path).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
