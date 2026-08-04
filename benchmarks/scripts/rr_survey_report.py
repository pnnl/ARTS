#!/usr/bin/env python3
"""rr_survey_report.py -- build logs/perf/rr_divergence_survey.md from an
RR-divergence survey run dir (results.csv + metrics.json).

The survey runs EVERY app across {1n_sc, 2n_sc, 4n_sc} under the no-hint RR EDT
placement policy (experiment "strong", so a non-scaling app broadcasts its
fix_args = a constant workload across the geometry).  We measure whether the
compute span DIVERGES when the same fixed work is spread over more (localhost-
simulated) nodes.

Verdict metric: the pure compute span [E2E] (main-EDT-runnable -> shutdown),
NOT wall -- wall carries a fixed multi-rank launcher/reap overhead (~3-5 s) that
would swamp sub-second apps and manufacture false divergence.  A cell that never
prints [E2E] (rc==124 TIMEOUT, or FAIL) is DIVERGENT by definition (it did not
finish the fixed work within the 45 s budget the 1n_sc point met).

Thresholds (r = max over geos of e2e(geo)/e2e(1n_sc)):
    TIMEOUT/FAIL at 2n_sc or 4n_sc  -> DIVERGENT
    r > 3.0                          -> DIVERGENT
    1.5 < r <= 3.0                   -> DEGRADED
    r <= 1.5                         -> SCALES (flat or improving)
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

R_DIVERGENT, R_DEGRADED = 3.0, 1.5
GEOS = ["1n_sc", "2n_sc", "4n_sc"]
PROTOS = ["ocr_val_wb", "ocr_excl_retain"]


def load(dir_):
    rows = {}
    p = Path(dir_) / "results.csv"
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["bench"], r["runtime"], r["config"])
            e = r["e2e_ns"]
            rows[key] = {
                "e2e": (float(e) / 1e9 if e not in ("", None) else None),
                "wall": float(r["wall"]) if r["wall"] else None,
                "status": r["status"], "rc": r["rc"],
            }
    metrics = {}
    mp = Path(dir_) / "metrics.json"
    if mp.exists():
        try:
            for m in json.loads(mp.read_text()):
                metrics[(m["bench"], m["runtime"], m["config"])] = m.get("per_rank", {})
        except (OSError, ValueError):
            pass
    return rows, metrics


def verdict(rows, bench, proto):
    """Return (verdict, ratio_or_None, base_e2e, per-geo cell dicts).

    A cell that was never run (absent from results.csv) is INCOMPLETE data, not
    divergence -- only a cell that ran and TIMEOUT/FAIL'd (or a ratio > 3x)
    counts as DIVERGENT."""
    cells = {g: rows.get((bench, proto, g)) for g in GEOS}
    base = cells["1n_sc"]
    base_e2e = base["e2e"] if base else None
    worst = 1.0
    diverged = incomplete = False
    for g in ("2n_sc", "4n_sc"):
        c = cells[g]
        if c is None:                     # not run yet
            incomplete = True
            continue
        if c["status"] in ("TIMEOUT", "FAIL") or c["e2e"] is None:
            diverged = True               # ran, but did not finish the work
            continue
        if base_e2e and base_e2e > 0:
            worst = max(worst, c["e2e"] / base_e2e)
    if diverged:
        return "DIVERGENT", (worst if worst > 1.0 else None), base_e2e, cells
    if base is None or (incomplete and worst == 1.0):
        return ("NO_BASE" if base is None else "INCOMPLETE"), None, base_e2e, cells
    if worst > R_DIVERGENT:
        return "DIVERGENT", worst, base_e2e, cells
    if worst > R_DEGRADED:
        return "DEGRADED", worst, base_e2e, cells
    return "SCALES", worst, base_e2e, cells


def cause(metrics, bench, proto="ocr_val_wb", geo="2n_sc"):
    """Counter-based cause-family estimate at the given geo.
    caller-rank-only  = EDTs stay on rank 0 (rank0 EDT_FINISH fraction ~1).
    coherence-roundtrip = EDTs spread but remote traffic per DB op is high."""
    pr = metrics.get((bench, proto, geo))
    if not pr or len(pr) < 2:
        return "-", {}
    def col(name):
        return {rk: float(v.get(name, 0) or 0) for rk, v in pr.items()}
    ef = col("NUM_EDT_FINISH")
    tot_ef = sum(ef.values())
    rank0_frac = (ef.get("0", 0) / tot_ef) if tot_ef else 1.0
    rsend = sum(col("NUM_REMOTE_SEND").values())
    owner = sum(col("NUM_OWNER_UPDATE_PERFORMED").values())
    dbacq = sum(col("NUM_DB_ACQUIRE_READ").values()) + sum(col("NUM_DB_ACQUIRE_WRITE").values())
    remote_per_db = rsend / dbacq if dbacq else 0.0
    ev = {"rank0_edt_frac": round(rank0_frac, 3), "remote_send": int(rsend),
          "owner_updates": int(owner), "db_acq": int(dbacq),
          "remote_per_db": round(remote_per_db, 2)}
    if rank0_frac > 0.85:
        return "caller-rank-only (EDTs stay on rank0)", ev
    if remote_per_db > 1.0 or owner > dbacq:
        return "coherence round-trips (remote DB home / ownership churn)", ev
    return "distributed; modest remote traffic", ev


def fmt(x, w=6):
    return ("%.2f" % x).rjust(w) if isinstance(x, (int, float)) else str(x).rjust(w)


def main():
    dir_ = sys.argv[1] if len(sys.argv) > 1 else "logs/perf/rr_survey"
    out = sys.argv[2] if len(sys.argv) > 2 else "logs/perf/rr_divergence_survey.md"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import performance_harness as ph
    skip = {b.name: b.scale_skip for b in ph.BENCHES}
    order = [b.name for b in ph.BENCHES]

    rows, metrics = load(dir_)
    L = []
    L.append("# RR-divergence survey (all apps x {ocr_val_wb, ocr_excl_retain} x {1n_sc,2n_sc,4n_sc})\n")
    L.append(f"Source: `{dir_}` (experiment=strong, 1 iter, retries=0, timeout=45 s).\n")
    L.append("## Criteria\n")
    L.append("Metric = compute span **[E2E]** (excludes fixed multi-rank launcher "
             "overhead). Ratio r = max(e2e(2n_sc), e2e(4n_sc)) / e2e(1n_sc).\n")
    L.append("- **DIVERGENT** = TIMEOUT/FAIL at 2n_sc or 4n_sc, OR r > 3.0\n")
    L.append("- **DEGRADED** = 1.5 < r <= 3.0\n")
    L.append("- **SCALES** = r <= 1.5 (flat or improving)\n")
    L.append("\nNon-scaling apps run the strong experiment by broadcasting their "
             "fix_args (constant workload across the geometry); `scale_skip` is "
             "carried as the *expected* reason and is NOT a run gate.\n")

    tally = {v: 0 for v in ("SCALES", "DEGRADED", "DIVERGENT", "NO_BASE", "INCOMPLETE")}
    L.append("\n## Per-app verdict (ocr_val_wb | ocr_excl_retain)\n")
    L.append("| app | expected(scale_skip) | ocr_val_wb e2e 1n/2n/4n | r | verdict | ocr_excl_retain e2e 1n/2n/4n | r | verdict |")
    L.append("|---|---|---|---|---|---|---|---|")
    verdicts = {}
    for bench in order:
        line = [bench, (skip.get(bench) or "SCALING-CORE")[:38]]
        for proto in PROTOS:
            v, r, base, cells = verdict(rows, bench, proto)
            def cell_s(g):
                c = cells[g]
                if not c:
                    return "-"
                if c["status"] in ("TIMEOUT", "FAIL"):
                    return c["status"][:4]
                return "%.2f" % c["e2e"] if c["e2e"] is not None else "noE2E"
            trip = "/".join(cell_s(g) for g in GEOS)
            rs = ("%.1fx" % r) if isinstance(r, float) else "-"
            line += [trip, rs, v]
            if proto == "ocr_val_wb":
                verdicts[bench] = v
                tally[v] = tally.get(v, 0) + 1
        L.append("| " + " | ".join(line) + " |")

    L.append(f"\n## Tally (ocr_val_wb): "
             + ", ".join(f"{k}={tally.get(k,0)}" for k in
                         ("SCALES","DEGRADED","DIVERGENT","NO_BASE","INCOMPLETE")) + "\n")

    # Cause-family spot-check on up to a few DIVERGENT apps (counter evidence).
    div = [b for b in order if verdicts.get(b) == "DIVERGENT"]
    L.append("## Cause-family (counter spot-check @2n_sc, ocr_val_wb)\n")
    L.append("| app | cause | rank0_edt_frac | remote_send | owner_upd | db_acq | remote/db |")
    L.append("|---|---|---|---|---|---|---|")
    shown = 0
    for bench in div:
        cz, ev = cause(metrics, bench)
        if not ev:
            continue
        L.append(f"| {bench} | {cz} | {ev['rank0_edt_frac']} | {ev['remote_send']} | "
                 f"{ev['owner_updates']} | {ev['db_acq']} | {ev['remote_per_db']} |")
        shown += 1
        if shown >= 12:
            break
    if shown == 0:
        L.append("| (no per-rank counter data yet) | | | | | | |")

    L.append("\n## Reading\n")
    L.append("- `caller-rank-only`: RR spread the EDTs but they still ran on rank 0 "
             "(rank0_edt_frac ~1) -- the app has no per-domain work to place elsewhere.\n")
    L.append("- `coherence round-trips`: EDTs DID spread, but each remote EDT's DB "
             "acquire pulls from a remote home (high remote_send / owner-update per DB "
             "acquire) -- loopback coherence traffic dominates.\n")
    L.append("- SCALING-CORE apps carry designed strong args; the rest broadcast "
             "fix_args, so their divergence is the pure RR-placement effect.\n")

    Path(out).write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(order)} apps; tally {tally})")


if __name__ == "__main__":
    main()
