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
from dataclasses import dataclass, field
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
    completion_marker: str     # regex marking "result printed"
    fix_args: list             # one fixed arg-list (Cat A constant grid / Cat B fixed)
    strong_args: object        # list (constant across nodes) OR dict[int,list], grid=4N
    weak_args: object          # list OR dict[int,list], grid=4N, per-sub-domain fixed
    extra_scalars: dict = field(default_factory=dict)  # name -> regex (graph500 kernel2)


E2E_RE = re.compile(r"\[E2E\]\s+(\d+)")


def parse_reference_e2e(text: str):
    """Extract the reference-runtime (xsocr/ocr-vx/baseline) wall-time marker
    `[E2E] <nanoseconds>` from stdout. Returns None if absent."""
    m = E2E_RE.search(text)
    return float(m.group(1)) if m else None


def parse_extra_scalars(text: str, extra_scalars: dict) -> dict:
    """Extract app-specific secondary scalars by name->regex (e.g. graph500's
    kernel-2 time / MTEPS, which scale cleanly where e2e does not because of the
    NO_FILES redundant graph generation). Missing markers map to None."""
    out = {}
    for name, pat in extra_scalars.items():
        m = re.search(pat, text)
        out[name] = float(m.group(1)) if m else None
    return out


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


# fix = capacity family (48 cores constant, vary node count). _io are arts-only.
FIX_CONFIGS = ["1n", 2, 4, 8, 16, "2n_io", "4n_io", "8n_io"]
# strong AND weak share the scalability (_sc) family (4 workers/node, 1..8).
SCALE_CONFIGS = ["1n_sc", "2n_sc", "4n_sc", "8n_sc"]
EXPERIMENTS = {"fix": FIX_CONFIGS, "strong": SCALE_CONFIGS, "weak": SCALE_CONFIGS}


def select_perf_args(case: "PerfCase", experiment: str, geo: int) -> list:
    """Pick the argv for one (experiment, geo) cell.
    fix -> the single fixed list; strong/weak -> the geo entry of a dict, or a
    constant list broadcast to every geo (Category-B apps)."""
    if experiment == "fix":
        return case.fix_args
    a = case.strong_args if experiment == "strong" else case.weak_args
    return a[geo] if isinstance(a, dict) else a

PERF_BENCHES: list = [
    # graph500: SCALE EDGEFACTOR R C. R*C=grid (pow2). EDGEFACTOR=8 fixed.
    # fix R*C=32 (8x4); strong R*C=4N fixed SCALE; weak R*C=4N SCALE=base+log2(4N).
    PerfCase("graph500", "graph500", r"nodes \d+",
        fix_args=["24", "8", "8", "4"],
        strong_args={1: ["25","8","2","2"], 2: ["25","8","4","2"],
                     4: ["25","8","4","4"], 8: ["25","8","8","4"]},
        weak_args={1: ["22","8","2","2"], 2: ["23","8","4","2"],
                   4: ["24","8","4","4"], 8: ["25","8","8","4"]},
        extra_scalars={"kernel2_ns": r"\[kernel2 time ([0-9.eE+-]+)\]",
                       "mteps": r"mean MTEPS ([0-9.eE+-]+)"}),
    # CoMD_sdsc2: -x -y -z (cells) -N (steps). No grid arg (auto-distribute).
    # -N is the time lever (no memory growth). weak grows the box per node.
    PerfCase("CoMD_sdsc2", "CoMD_sdsc2", r"Final energy",
        fix_args=["-x","60","-y","60","-z","60","-N","4"],
        strong_args=["-x","40","-y","40","-z","40","-N","100"],
        weak_args={1: ["-x","40","-y","40","-z","40","-N","40"],
                   2: ["-x","80","-y","40","-z","40","-N","40"],
                   4: ["-x","80","-y","80","-z","40","-N","40"],
                   8: ["-x","80","-y","80","-z","80","-N","40"]}),
    # hpcg_intel: npx npy npz m iters. grid=npx*npy*npz. m rounds UP to x16.
    # fix grid=48 (4x4x3); strong grid=4N total~const; weak grid=4N m=64.
    PerfCase("hpcg_intel", "hpcg_intel", r"final deviation",
        fix_args=["4","4","3","64","48"],
        strong_args={1: ["2","2","1","144","50"], 2: ["2","2","2","112","50"],
                     4: ["4","2","2","96","50"], 8: ["4","4","2","80","50"]},
        weak_args={1: ["2","2","1","112","50"], 2: ["2","2","2","112","50"],
                   4: ["4","2","2","112","50"], 8: ["4","4","2","112","50"]}),
    # nekbone: Rx Ry Rz Ex Ey Ez pDOF CGcount. Rtotal=grid. Rx>=Ry>=Rz, Ex>=Ey>=Ez.
    # fix grid=48; strong grid=4N total=1024; weak grid=4N per-rank Etotal=256
    # (8x8x4, held constant across geos -> total grows with N). pDOF=12, CG=time lever.
    PerfCase("nekbone", "nekbone", r"FinalEDT",
        fix_args=["4","4","3","4","4","4","12","200"],
        strong_args={1: ["2","2","1","8","8","4","12","550"],
                     2: ["2","2","2","8","4","4","12","550"],
                     4: ["4","2","2","4","4","4","12","550"],
                     8: ["4","4","2","4","4","2","12","550"]},
        weak_args={1: ["2","2","1","8","8","4","12","200"],
                   2: ["2","2","2","8","8","4","12","200"],
                   4: ["4","2","2","8","8","4","12","200"],
                   8: ["4","4","2","8","8","4","12","200"]}),
    # RSBench_intel_sharedDB: -l (GLOBAL lookups, split among -t) -t (EDTs=grid).
    # fix -t=48; strong -t=4N -l fixed; weak -t=4N -l=base*4N.
    PerfCase("RSBench_intel_sharedDB", "RSBench_intel_sharedDB", r"RS_CHECKSUM",
        fix_args=["-l","14000000","-t","48"],
        strong_args={1: ["-l","6000000","-t","4"], 2: ["-l","6000000","-t","8"],
                     4: ["-l","6000000","-t","16"], 8: ["-l","6000000","-t","32"]},
        weak_args={1: ["-l","2560000","-t","4"], 2: ["-l","5120000","-t","8"],
                   4: ["-l","10240000","-t","16"], 8: ["-l","20480000","-t","32"]}),
    # Stencil2D_intel_chandra: npoints nranks ntimesteps. nranks=grid (2D split).
    # weak npoints=base*sqrt(4N). ntimesteps=time lever. L1=2*(nt+1).
    PerfCase("Stencil2D_intel_chandra", "Stencil2D_intel_chandra", r"L1 norm",
        fix_args=["20000","48","100"],
        strong_args={1: ["32000","4","64"], 2: ["32000","8","64"],
                     4: ["32000","16","64"], 8: ["32000","32","64"]},
        weak_args={1: ["20000","4","64"], 2: ["28284","8","64"],
                   4: ["40000","16","64"], 8: ["56569","32","64"]}),
    # hpgmg: log2_box_dim target_boxes. Boxes auto-home box%node_count (Cat B).
    # weak target_boxes=boxes_in_i^3 ~ proportional to N.
    PerfCase("hpgmg", "hpgmg", r"\|\|error\|\|",
        fix_args=["6","64"],
        strong_args=["6","216"],
        weak_args={1: ["6","64"], 2: ["6","216"], 4: ["6","512"], 8: ["6","1000"]}),
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


def finalize_status(status: str, e2e_ns) -> str:
    """Gate classify_run's verdict on e2e-marker presence: a run is only a
    valid measurement when the "[E2E] <ns>" marker was actually captured,
    whichever path produced "OK" -- a clean exit (rc==0), a post-result hang
    (SHUTDOWN_HANG: rc==124 or reaped), or a post-result teardown death
    (nonzero rc with the completion marker seen, e.g. rc==139 SIGSEGV; these
    three collapse to the same "OK" from classify_run). Missing e2e demotes
    to COMPUTE_FAIL so the caller's retry path can re-attempt (or the
    iteration is dropped as FAIL after retries exhaust) -- a run missing
    either marker never counts as valid, regardless of rc."""
    if status in ("OK", "SHUTDOWN_HANG"):
        return "OK" if e2e_ns is not None else "COMPUTE_FAIL"
    return status


def _perf_dispatch(runner: Runner, rt: Runtime, case: PerfCase, experiment: str,
                   node: object, metrics_dir: str | None = None):
    """Run one (runtime, case, experiment, node) cell. Mirrors correctness_harness
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

    args = select_perf_args(case, experiment, geo)
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
                return runner.run_ocrvx_mpi(case.name, case.ocr_base, args, tbb=4)
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
            else:  # ocrvx -- np-driven; _sc geometry = 4 workers / 6 cores per rank
                return runner.run_ocrvx_mpi(case.name, case.ocr_base, args, np=geo,
                                            tpn=6, tbb=4)
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


def run_cell_with_retry(runner: Runner, rt: Runtime, case: PerfCase, experiment: str,
                        node: object, iters: int, retries: int) -> list:
    """Run (rt, case, experiment, node) for `iters` accepted iterations, retrying
    COMPUTE_FAIL up to `retries` times per iteration.

    A post-result death with a parsed e2e is accepted as-is (the result was
    captured in-runtime before the process died) whether it presents as a
    SHUTDOWN_HANG (rc==124 or reaped-but-alive) or a teardown crash (nonzero
    rc, e.g. rc==139 SIGSEGV, process already exited) -- see
    finalize_status(). The dead/hung process is reaped via Runner._reap_exe
    so it cannot interfere with the next run. Only COMPUTE_FAIL (no usable
    result at all, including an OK-shaped run whose e2e marker never printed)
    triggers a retry.

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
            r = _perf_dispatch(runner, rt, case, experiment, node, metrics_dir=metrics_dir)
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
            extra = parse_extra_scalars(r.stdout, case.extra_scalars)

            # Gate on e2e presence uniformly across every "OK" path: a clean
            # exit, a post-result hang (SHUTDOWN_HANG), and a post-result
            # teardown death (nonzero rc, marker seen -> classify_run already
            # says "OK") all require the e2e marker to count as a valid
            # measurement; missing it demotes to COMPUTE_FAIL (retry).
            status = finalize_status(status, e2e_ns)

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
                **extra,
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
    cols = ["experiment", "bench", "config", "runtime", "iter", "e2e_ns",
            "rc", "wall", "status", "kernel2_ns", "mteps"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


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


RESULT_COLS = ["experiment", "bench", "config", "runtime", "iter", "e2e_ns",
               "rc", "wall", "status", "kernel2_ns", "mteps"]


def _load_done_cells(csv_path) -> set:
    """Read an existing results.csv into a set of (experiment, bench, config,
    runtime) keys already recorded, so a resumed run skips them. One saved row
    means that app+protocol+config cell is done (per-program resume)."""
    done = set()
    p = Path(csv_path)
    if not p.exists():
        return done
    try:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r.get("experiment", ""), r.get("bench", ""),
                          r.get("config", ""), r.get("runtime", "")))
    except (OSError, ValueError):
        pass
    return done


def _append_result_row(csv_path, row) -> None:
    """Append one cell's result row (writing the header if the file is new) and
    flush, so a kill loses at most the single in-flight cell."""
    p = Path(csv_path)
    new = (not p.exists()) or p.stat().st_size == 0
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLS)
        if new:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in RESULT_COLS})
        f.flush()


def main():
    global RUNNER_APPS

    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", default="build_release_mrnew_lazy",
                   help="Build directory containing apps and configs")
    p.add_argument("--target", default="server", choices=["laptop", "server"],
                   help="Machine geometry: laptop (14-thread) or server (48-thread)")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated bench names to restrict to")
    p.add_argument("--node", type=str, default="",
                   help="Restrict to a single node-config, e.g. --node 1n")
    p.add_argument("--experiment", type=str, default="fix,strong,weak",
                   help="Comma-separated experiments to run: fix,strong,weak")
    p.add_argument("--iters", type=int, default=5,
                   help="Accepted iterations per (bench, node, runtime) cell")
    p.add_argument("--retries", type=int, default=3,
                   help="Max COMPUTE_FAIL retries per iteration")
    p.add_argument("--mem-gb", type=int, default=4)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--out-dir", type=str, default="",
                   help="Stable results dir. Rows are appended per cell and any "
                        "cell already in its results.csv is skipped (per-program "
                        "resume). Default: a fresh timestamped dir (no resume).")
    args = p.parse_args()

    # All three runtimes (arts/xsocr/ocr-vx) emit their "[E2E] <ns>" span only
    # when this is set, so the measurement run enables it; every launched child
    # inherits it through os.environ. A normal (non-perf) run leaves it unset
    # and is never perturbed.
    os.environ["ARTS_E2E_MARKER"] = "1"

    # Perf runs cap EVERY runtime at the same wall budget (--timeout): a cell that
    # needs longer is "too slow" and should time out rather than consume the 8x
    # budget the correctness harness grants ocr-vx (whose protocol is structurally
    # slower).  Override the module's ocr-vx multiplier to 1 for this perf process
    # only; the correctness harness runs in a separate process and keeps its 8x.
    import harness_common
    harness_common._OCRVX_TIMEOUT_MULT = 1

    build = Path(args.build_dir)
    if not build.is_absolute():
        build = REPO / build
    RUNNER_APPS = build / "benchmarks" / "apps"

    benches = PERF_BENCHES
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}
        benches = [b for b in benches if b.name in only]

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    logdir = Path(args.out_dir) if args.out_dir else (LOGS_ROOT / ts)
    if not logdir.is_absolute():
        logdir = REPO / logdir
    logdir.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.mem_gb, args.timeout, logdir, target=args.target, build=build)

    csv_path = logdir / "results.csv"
    metrics_path = logdir / "metrics.json"
    # Per-program resume: skip any (experiment, bench, config, runtime) cell whose
    # row is already saved, and append each new cell immediately (flush) so a kill
    # or reboot loses at most the single in-flight program.
    done = _load_done_cells(csv_path)
    metric_rows = []
    if metrics_path.exists():
        try:
            metric_rows = json.loads(metrics_path.read_text())
        except (OSError, ValueError):
            metric_rows = []
    n_new = 0
    selected_exps = [e.strip() for e in args.experiment.split(",") if e.strip()]
    for experiment in selected_exps:
        node_configs = EXPERIMENTS[experiment]
        if args.node:
            node_configs = [_parse_node_arg(args.node)]
        for case in benches:
            for node in node_configs:
                for rt in RUNTIMES:
                    if not perf_runtime_eligible(rt, case, node):
                        continue
                    cell = (experiment, case.name, str(node), rt.key)
                    if cell in done:
                        print(f"[perf] SKIP done: {experiment} {case.name} x {node} "
                              f"x {rt.key}", flush=True)
                        continue
                    print(f"[perf] {experiment} {case.name} x {node} x {rt.key} ...",
                          flush=True)
                    iters = run_cell_with_retry(runner, rt, case, experiment, node,
                                                args.iters, args.retries)
                    for it in iters:
                        row = {
                            "experiment": experiment, "bench": case.name,
                            "config": str(node), "runtime": rt.key, "iter": it["iter"],
                            "e2e_ns": it["e2e_ns"], "rc": it["rc"], "wall": it["wall"],
                            "status": it["status"],
                            **{k: it.get(k) for k in case.extra_scalars},
                        }
                        _append_result_row(csv_path, row)
                        n_new += 1
                        if rt.kind == "arts" and it["metrics_dir"]:
                            counters = parse_arts_counters(it["metrics_dir"])
                            metric_rows.append({
                                "experiment": experiment, "bench": case.name,
                                "config": str(node), "runtime": rt.key,
                                "iter": it["iter"], "e2e_ns": it["e2e_ns"],
                                "per_rank": counters["per_rank"],
                            })
                            write_metrics_json(metric_rows, metrics_path)
                        print(f"    iter {it['iter']}: status={it['status']} "
                              f"e2e_ns={it['e2e_ns']} wall={it['wall']}", flush=True)
                    done.add(cell)

    print(f"\n[perf] appended {n_new} new rows -> {csv_path} "
          f"({len(done)} cells recorded)")
    print(f"[perf] {len(metric_rows)} metric rows -> {metrics_path}")


if __name__ == "__main__":
    main()
