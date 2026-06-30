#!/usr/bin/env python3
"""performance_harness.py — e2e wall-time + ARTS coherence-metric harness.
See docs/superpowers/specs/2026-06-29-performance-harness-design.md.

Counter-config selection (Step 1 finding): the enabled counter SET is
COMPILED IN, not read at runtime. `CMakeLists.txt` caches
`ARTS_COUNTER_CONFIG` (default `configs/counters.cfg`) and passes it as
`COUNTER_CONFIG_FILE` into `libs/include/internal/arts/counter/CMakeLists.txt`,
which parses the cfg at *configure* time and generates
`Preamble.h` (`ENABLE_*` / `COUNTER_MODE_*` / `COUNTER_LEVEL_*` /
`REDUCE_METHOD_*` macros + the `arts_counter_*_array`s) baked into the
build. There is no runtime knob to swap counter sets — a perf build that
wants `configs/perf_counters.cfg`'s counters must be *configured* with
`-DARTS_COUNTER_CONFIG=configs/perf_counters.cfg` and rebuilt. (Task 7:
the perf build dir is therefore a separate cmake configure/build from the
correctness `build_release_mrnew_lazy`, not a runtime flag on the same
binaries.)

Real counter JSON schema (Step 4 finding, from libs/src/core/counter/counter.c):
each emitted counter is a JSON object keyed by its counter name (e.g.
"TIME_TOTAL", "NUM_EDT_CREATE") with fields `captureMode`, `captureLevel`,
optionally `reduce_method`, a `value` field (uint64 — the brief's assumed
"final" key does not exist) and `value_ms` for time counters; PERIODIC
counters add a `captureHistory` array of `[epoch, value]` pairs. Per-node
files are named `n{node_id}.json` (NODE + CLUSTER level counters for that
rank, written by `arts_counter_write_node`); `cluster.json` (written by the
master rank only) holds CLUSTER-level counters reduced across all nodes
(`arts_counter_write` -> cluster aggregation). `TIME_TOTAL` is
ONCE,CLUSTER,MASTER in configs/perf_counters.cfg, so it lands in
cluster.json's `counters.TIME_TOTAL.value`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from harness_common import REPO, Runner, Runtime, RUNTIMES

LOGS_ROOT = REPO / "benchmarks" / "scripts" / "logs" / "perf"

# Set by main() once --build-dir is known; perf_runtime_eligible probes this
# to detect an absent ocrvx binary (some apps have no ocr-vx port).
RUNNER_APPS: Path | None = None


@dataclass
class PerfCase:
    name: str
    ocr_base: str
    args: object              # list[str] OR dict[int(node_geo) -> list[str]]
    completion_marker: str    # regex marking "result printed"


E2E_RE = re.compile(r"\[E2E\]\s+(\d+)")


def parse_reference_e2e(text: str):
    """Extract the reference-runtime (xsocr/ocr-vx/baseline) wall-time marker
    `[E2E] <nanoseconds>` from stdout. Returns None if absent."""
    m = E2E_RE.search(text)
    return float(m.group(1)) if m else None


def _counter_value(entry):
    """A counter's JSON object has a `value` field; tolerate a bare scalar
    too (defensive — not currently emitted, but cheap to allow)."""
    return entry.get("value") if isinstance(entry, dict) else entry


def parse_arts_counters(counter_dir) -> dict:
    """Parse one ARTS run's counter-output directory (the
    `output_folder` passed to `arts_counter_write`) into:
        {"per_rank": {rank:{metric:value}}, "periodic": {}}
    Robust to missing/partial n{id}.json files (e.g. a crashed or still-running
    node) -- returns partial data rather than raising.  End-to-end time is NOT
    a counter: it is the env-gated "[E2E] <ns>" stderr marker, parsed by
    parse_reference_e2e identically for all three runtimes.
    """
    counter_dir = Path(counter_dir)
    out = {"per_rank": {}, "periodic": {}}

    for nf in sorted(counter_dir.glob("n*.json")):
        m = re.match(r"n(\d+)\.json$", nf.name)
        if not m:
            continue
        rank = int(m.group(1))
        try:
            d = json.loads(nf.read_text())
        except (OSError, ValueError):
            continue
        ctrs = d.get("counters", {})
        out["per_rank"][rank] = {k: _counter_value(v) for k, v in ctrs.items()}

    return out


# Node configs exercised by the perf matrix (filled in by later tasks).
# The "*_sc" entries are the server-target single-config (_sc) cfgs added in
# Task 2: arts -> configs/local/server/{n}_sc.cfg, xsocr/ocr-vx ->
# configs/mpi/server/{n}_sc.cfg via mpirun.
PERF_NODE_CONFIGS = [
    "1n", 2, 4, 8, 16, "2n_io", "4n_io", "8n_io",
    "1n_sc", "2n_sc", "4n_sc", "8n_sc",
]

# Args are calibrated so each bench's 1n arts wall time lands in the 10-30s
# window: large enough that the run is compute-dominated (not startup/
# teardown noise), small enough to keep the full matrix tractable.
PERF_BENCHES: list = [
    # Strong scaling: SCALE=21 (total vertices=2^21) FIXED; the R*C worker grid
    # = node count so per-rank vertices = 2^21/(R*C) shrinks with node count.
    # R>=C (the app's grid convention). args = SCALE EDGEFACTOR R C.
    PerfCase("graph500", "graph500", {
        1:  ["21", "8", "1", "1"],
        2:  ["21", "8", "2", "1"],
        4:  ["21", "8", "2", "2"],
        8:  ["21", "8", "4", "2"],
        16: ["21", "8", "4", "4"],
    }, r"nodes \d+"),
    PerfCase("CoMD_sdsc2", "CoMD_sdsc2",
             ["-x", "40", "-y", "40", "-z", "40", "-N", "2"], r"Final energy"),
    # Strong scaling: total grid (npx*npy*npz * local_nx^3) FIXED at 1n's 160^3;
    # npx*npy*npz = node count, local_nx = round((160^3/node)^(1/3)) so per-rank
    # work shrinks with node count. args = npx npy npz local_nx iters.
    PerfCase("hpcg_intel", "hpcg_intel", {
        1:  ["1", "1", "1", "160", "5"],
        2:  ["2", "1", "1", "127", "5"],
        4:  ["2", "2", "1", "101", "5"],
        8:  ["2", "2", "2", "80", "5"],
        16: ["4", "2", "2", "64", "5"],
    }, r"final deviation"),
    PerfCase("hpgmg", "hpgmg", ["6", "64"], r"\|\|error\|\|"),
    # args = Rx Ry Rz Ex Ey Ez pDOF CGcount.  Strong scaling: TOTAL elements
    # (Rx*Ry*Rz * Ex*Ey*Ez) FIXED at 1n's 512 (=8^3); the Rx*Ry*Rz PD grid
    # (Rx>=Ry>=Rz, the app's ordering invariant) = node count, so per-rank
    # Ex*Ey*Ez = 512/node halves each step (8*8*8 -> 8*8*4 -> 8*4*4 -> 4*4*4
    # -> 4*4*2).  pDOF=8 (8th-order spectral basis) and CGcount=400 fixed.
    # Marker is FinalEDT (true completion) not rnorminit (which prints on the
    # first CG step, long before the CGcount loop finishes).
    PerfCase("nekbone", "nekbone", {
        1:  ["1", "1", "1", "8", "8", "8", "8", "400"],
        2:  ["2", "1", "1", "8", "8", "4", "8", "400"],
        4:  ["2", "2", "1", "8", "4", "4", "8", "400"],
        8:  ["2", "2", "2", "4", "4", "4", "8", "400"],
        16: ["4", "2", "2", "4", "4", "2", "8", "400"],
    }, r"FinalEDT"),
    # miniAMR_intel_bryan DROPPED from the perf core (8 -> 7): every EDT uses
    # ocrAffinityGetCurrent (caller-rank) with no PD-distribution code, so at 2n+
    # all EDTs pin to rank0 (measured EDT_FINISH n0/n1 = 356007/0) — effectively
    # single-node, unfit for multinode perf scaling. See memory
    # perf-affinity-rsbench-miniamr. Perf core is now: CoMD_sdsc2, hpcg_intel,
    # hpgmg, nekbone, RSBench_intel_sharedDB, Stencil2D_intel_chandra, graph500.
    # Strong scaling: lookups (-l) FIXED; -t (perThread SPMD EDT count) = node count
    # so the EDTs spread across policy domains via getPolicyDomainID_Cart1D. Needs
    # the SINGLE_RUN_ACROSS_PD build define (benchmarks/apps/CMakeLists.txt) to
    # compile in the affinity-hint code; without BOTH (build define + -t=node) all
    # RSBench EDTs pin to rank0 (verified: -t1 => n0/n1=612/0; -t2 => 315/303).
    PerfCase("RSBench_intel_sharedDB", "RSBench_intel_sharedDB", {
        1:  ["-l", "300000", "-t", "1"],
        2:  ["-l", "300000", "-t", "2"],
        4:  ["-l", "300000", "-t", "4"],
        8:  ["-l", "300000", "-t", "8"],
        16: ["-l", "300000", "-t", "16"],
    }, r"RS_CHECKSUM"),
    # Strong scaling: npoints (grid edge) and ntimesteps FIXED; nranks (2nd arg)
    # = node count so the grid is split across ranks. args = npoints nranks
    # ntimesteps.
    PerfCase("Stencil2D_intel_chandra", "Stencil2D_intel_chandra", {
        1:  ["10000", "1", "64"],
        2:  ["10000", "2", "64"],
        4:  ["10000", "4", "64"],
        8:  ["10000", "8", "64"],
        16: ["10000", "16", "64"],
    }, r"L1 norm"),
]


# ---------------------------------------------------------------------------
# Hang-detection FSM + retry wrapper (Task 6).
# ---------------------------------------------------------------------------

# Geometries (ranks) the "_sc" scalability series exercises.  1n_sc is a
# single-rank direct exec (no launcher); 2/4/8n_sc use the same launcher as
# the corresponding capacity geometry (arts self-fork / mpirun) but point at
# the dedicated server/{n}_sc.cfg pair (see configs/local/server,
# configs/mpi/server) instead of the capacity-matrix cfgs.
_SC_GEOS = {"1n_sc": 1, "2n_sc": 2, "4n_sc": 4, "8n_sc": 8}


def classify_run(rc: int, marker_seen: bool, proc_alive: bool) -> str:
    # A post-result hang: completion marker printed but process did not exit
    # (rc==124 wall-timeout, or reaped). e2e was already captured in-runtime.
    if marker_seen and (rc == 124 or proc_alive):
        return "SHUTDOWN_HANG"
    if rc == 0 and marker_seen:
        return "OK"
    if marker_seen:           # nonzero rc but result present → still usable e2e, but flag
        return "OK"
    return "COMPUTE_FAIL"


def _perf_dispatch(runner: Runner, rt: Runtime, case: PerfCase, node: object,
                   metrics_dir: str | None = None):
    """Run one (runtime, case, node) cell. Mirrors correctness_harness
    run_runtime's dispatch (capacity 1n / capacity MN), plus the "_sc"
    scalability-series family which needs an explicit cfg-path override
    (the capacity cfg maps in Runner only cover the capacity MN_RANKS keys).

    metrics_dir (arts only) overrides the `counter_folder` cfg key via its
    identically-named env override (config.c getenv(variable)), so each
    iteration writes its counter JSON to its own directory instead of all
    iterations colliding on the cfg-default `./counters`."""
    is_sc = isinstance(node, str) and node.endswith("_sc")
    if is_sc:
        geo = _SC_GEOS[node]
    elif node in ("1n", 1):
        geo = 1
    else:
        geo = int(str(node).split("n")[0])

    args = case.args.get(geo, []) if isinstance(case.args, dict) else case.args
    arts_env = {"counter_folder": metrics_dir} if (rt.kind == "arts" and metrics_dir) else None

    if is_sc:
        cfg_base = REPO / "configs"
        if geo == 1:
            # Single-rank direct exec, same shape as capacity 1n, but with
            # the "_sc" cfg (4 workers, no node_count) instead of the
            # capacity 1n cfg (48 workers).
            if rt.kind == "arts":
                return runner.run_ocr(
                    case.name, case.ocr_base, args, "arts", suffix=rt.suffix,
                    cfg_path=cfg_base / "local" / "server" / "1n_sc.cfg",
                    extra_env=arts_env)
            elif rt.kind == "xsocr":
                return runner.run_ocr(
                    case.name, case.ocr_base, args, "xsocr",
                    cfg_path=cfg_base / "mpi" / "server" / "1n_sc.cfg")
            else:  # ocrvx -- no cfg file consumed; np=1 already direct-execs
                return runner.run_ocrvx_mpi(case.name, case.ocr_base, args)
        else:
            if rt.kind == "arts":
                return runner.run_arts_mn(
                    case.name, case.ocr_base, args, geo, suffix=rt.suffix,
                    cfg_path=cfg_base / "local" / "server" / f"{geo}n_sc.cfg",
                    extra_env=arts_env)
            elif rt.kind == "xsocr":
                return runner.run_xsocr_mpi(
                    case.name, case.ocr_base, args, geo,
                    cfg_path=cfg_base / "mpi" / "server" / f"{geo}n_sc.cfg",
                    tpn=6)
            else:  # ocrvx -- np-driven, no cfg file
                return runner.run_ocrvx_mpi(case.name, case.ocr_base, args, np=geo)
    elif node in ("1n", 1):
        if rt.kind == "arts":
            return runner.run_ocr(case.name, case.ocr_base, args, "arts",
                                  suffix=rt.suffix, extra_env=arts_env)
        elif rt.kind == "xsocr":
            return runner.run_ocr(case.name, case.ocr_base, args, "xsocr")
        else:  # ocrvx
            return runner.run_ocrvx_mpi(case.name, case.ocr_base, args)
    else:
        if rt.kind == "arts":
            return runner.run_arts_mn(case.name, case.ocr_base, args, node,
                                      suffix=rt.suffix, extra_env=arts_env)
        elif rt.kind == "xsocr":
            return runner.run_xsocr_mpi(case.name, case.ocr_base, args, node)
        else:  # ocrvx
            return runner.run_ocrvx_mpi(case.name, case.ocr_base, args, np=geo)


def run_cell_with_retry(runner: Runner, rt: Runtime, case: PerfCase, node: object,
                        iters: int, retries: int) -> list:
    """Run (rt, case, node) for `iters` accepted iterations, retrying
    COMPUTE_FAIL up to `retries` times per iteration.

    A SHUTDOWN_HANG with a parsed e2e is accepted as-is (the result was
    captured in-runtime before the post-result hang); the hung process is
    reaped via Runner._reap_exe so it cannot interfere with the next run.
    Only COMPUTE_FAIL (no usable result at all) triggers a retry.

    Returns one dict per accepted iteration:
        {"iter", "e2e_ns", "rc", "wall", "status", "metrics_dir"}
    "metrics_dir" is the counter-output dir for arts runs, else None.
    """
    is_arts = rt.kind == "arts"
    bin_name = (f"{case.ocr_base}_arts_{rt.suffix}" if is_arts
                else f"{case.ocr_base}_{rt.kind}")
    exe_path = runner.apps_dir / bin_name

    results = []
    for i in range(iters):
        attempt = 0
        while True:
            # Keyed by attempt too: a COMPUTE_FAIL retry must not reuse the
            # previous attempt's counter-output dir, else a partially-written
            # (crashed mid-run) cluster.json/n*.json from the failed attempt
            # could be read back as this attempt's result.
            metrics_dir = None
            if is_arts:
                metrics_path = (runner.logdir / "metrics" /
                                 f"{case.name}__{node}__{rt.key}__iter{i}__try{attempt}")
                # The runtime's counter writer only mkdir()s the leaf
                # component of counter_folder (single-level, not -p): it
                # assumes the parent exists. Pre-create the full path here so
                # the leaf mkdir succeeds instead of silently failing closed
                # (fopen on a nonexistent dir returns NULL -> counters
                # dropped with no error).
                metrics_path.mkdir(parents=True, exist_ok=True)
                metrics_dir = str(metrics_path)
            r = _perf_dispatch(runner, rt, case, node, metrics_dir=metrics_dir)
            marker_seen = bool(re.search(case.completion_marker, r.stdout))
            proc_alive = False
            if r.rc == 124:
                # Wall-timeout sentinel: the process may have survived
                # graceful shutdown signaling; reap it and note whether it
                # was actually still alive (best-effort -- _reap_exe kills
                # unconditionally, so treat rc==124 itself as the alive
                # signal for classification, mirroring the brief's FSM).
                proc_alive = True
                runner._reap_exe(exe_path)

            status = classify_run(r.rc, marker_seen, proc_alive)

            # e2e is the env-gated "[E2E] <ns>" stderr marker for ALL three
            # runtimes (identical span definition: main app EDT runnable ->
            # shutdown recognition).  arts additionally emits per-rank coherence
            # metrics into its counter-output dir, summarized later from
            # metrics_dir; e2e itself is no longer a counter.
            e2e_ns = parse_reference_e2e(r.stdout)

            if status == "SHUTDOWN_HANG":
                # Accept if e2e parsed (captured pre-shutdown); otherwise
                # treat as an unusable iteration and retry like COMPUTE_FAIL.
                if e2e_ns is not None:
                    status = "OK"
                else:
                    status = "COMPUTE_FAIL"

            if status == "COMPUTE_FAIL" and attempt < retries:
                attempt += 1
                continue

            results.append({
                "iter": i,
                "e2e_ns": e2e_ns,
                "rc": r.rc,
                "wall": round(r.wall, 3),
                "status": status if status != "COMPUTE_FAIL" else "FAIL",
                "metrics_dir": metrics_dir,
            })
            break

    return results


# ---------------------------------------------------------------------------
# Eligibility + emitters (Task 7).
# ---------------------------------------------------------------------------

def perf_runtime_eligible(rt: Runtime, case: PerfCase, node: object) -> bool:
    """Whether (rt, case, node) is a cell the matrix should run at all.

    Reference runtimes (xsocr/ocr-vx) have no sender/receiver-split IO
    transport -- their N-node total already equals the plain N-node config,
    so the "_io" variants are arts-only and refs must be skipped there.
    ocr-vx is additionally skipped per-case when that app has no ocr-vx port
    (binary absent from the build's apps dir)."""
    is_io = isinstance(node, str) and node.endswith("_io")
    if rt.kind in ("xsocr", "ocrvx") and is_io:
        return False
    if rt.kind == "ocrvx" and not (RUNNER_APPS / f"{case.ocr_base}_ocrvx").exists():
        return False
    return True


def write_results_csv(rows: list, path) -> None:
    cols = ["bench", "config", "runtime", "iter", "e2e_ns", "rc", "wall", "status"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def write_metrics_json(metric_rows: list, path) -> None:
    Path(path).write_text(json.dumps(metric_rows, indent=2, default=str))


def _parse_node_arg(node_arg: str):
    """Mirror correctness_harness's --node parsing: "1n" stays a string,
    "Nn_io"/"Nn_sc" stay strings, plain digits become int."""
    if node_arg in ("1n",) or node_arg.endswith("_io") or node_arg.endswith("_sc"):
        return node_arg
    try:
        return int(node_arg.rstrip("n"))
    except ValueError:
        return node_arg


def main():
    global RUNNER_APPS, PERF_NODE_CONFIGS, PERF_BENCHES

    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", default="build_release_mrnew_lazy",
                   help="Build directory containing apps and configs")
    p.add_argument("--target", default="server", choices=["laptop", "server"],
                   help="Machine geometry: laptop (14-thread) or server (48-thread)")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated bench names to restrict to")
    p.add_argument("--node", type=str, default="",
                   help="Restrict to a single node-config, e.g. --node 1n")
    p.add_argument("--iters", type=int, default=5,
                   help="Accepted iterations per (bench, node, runtime) cell")
    p.add_argument("--retries", type=int, default=3,
                   help="Max COMPUTE_FAIL retries per iteration")
    p.add_argument("--mem-gb", type=int, default=4)
    p.add_argument("--timeout", type=int, default=90)
    args = p.parse_args()

    # All three runtimes (arts/xsocr/ocr-vx) emit their "[E2E] <ns>" span only
    # when this is set, so the measurement run enables it; every launched child
    # inherits it through os.environ. A normal (non-perf) run leaves it unset
    # and is never perturbed.
    os.environ["ARTS_E2E_MARKER"] = "1"

    build = Path(args.build_dir)
    if not build.is_absolute():
        build = REPO / build
    RUNNER_APPS = build / "benchmarks" / "apps"

    node_configs = PERF_NODE_CONFIGS
    if args.node:
        node_configs = [_parse_node_arg(args.node)]

    benches = PERF_BENCHES
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}
        benches = [b for b in benches if b.name in only]

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    logdir = LOGS_ROOT / ts
    runner = Runner(args.mem_gb, args.timeout, logdir, target=args.target, build=build)

    rows = []
    metric_rows = []
    for case in benches:
        for node in node_configs:
            for rt in RUNTIMES:
                if not perf_runtime_eligible(rt, case, node):
                    continue
                print(f"[perf] {case.name} x {node} x {rt.key} ...", flush=True)
                iters = run_cell_with_retry(runner, rt, case, node,
                                            args.iters, args.retries)
                for it in iters:
                    rows.append({
                        "bench": case.name,
                        "config": str(node),
                        "runtime": rt.key,
                        "iter": it["iter"],
                        "e2e_ns": it["e2e_ns"],
                        "rc": it["rc"],
                        "wall": it["wall"],
                        "status": it["status"],
                    })
                    if rt.kind == "arts" and it["metrics_dir"]:
                        counters = parse_arts_counters(it["metrics_dir"])
                        metric_rows.append({
                            "bench": case.name,
                            "config": str(node),
                            "runtime": rt.key,
                            "iter": it["iter"],
                            "e2e_ns": it["e2e_ns"],
                            "per_rank": counters["per_rank"],
                        })
                    print(f"    iter {it['iter']}: status={it['status']} "
                          f"e2e_ns={it['e2e_ns']} wall={it['wall']}", flush=True)

    logdir.mkdir(parents=True, exist_ok=True)
    write_results_csv(rows, logdir / "results.csv")
    write_metrics_json(metric_rows, logdir / "metrics.json")
    print(f"\n[perf] wrote {len(rows)} rows -> {logdir / 'results.csv'}")
    print(f"[perf] wrote {len(metric_rows)} metric rows -> {logdir / 'metrics.json'}")


if __name__ == "__main__":
    main()
