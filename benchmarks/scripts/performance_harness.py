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
    fix_args: list             # single-experiment arg-list (48w / 12w two-point concurrency)
    strong_args: object        # None (not a scaling app) OR list/dict[int,list], grid grows
    weak_args: object          # None OR list/dict[int,list], per-node work fixed
    extra_scalars: dict = field(default_factory=dict)  # name -> regex (graph500 kernel2)
    # Analysis METADATA (not a run gate): when non-empty, documents why this app
    # is not expected to cross-node scale (no EDT affinity / fixed-size fixture /
    # variant of a core scaling app).  Every app still runs in strong/weak (with
    # fix_args broadcast when strong_args is None); scale_skip is carried into
    # the RR-divergence report so a reader knows the expected cause.
    scale_skip: str = ""


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


# strong AND weak share the scalability (_sc) family: 12 threads per node,
# 1/2/4 nodes on the 48-core box.
SCALE_CONFIGS = ["1n_sc", "2n_sc", "4n_sc"]
# single = the whole-app-suite family: every OCR pair at the full 48-worker
# single-node geometry PLUS the 12-worker 1n_sc geometry — a two-point
# within-node concurrency-scaling axis that every app (including the
# NULL-HINT single-node-only ones) can run.
EXPERIMENTS = {"strong": SCALE_CONFIGS, "weak": SCALE_CONFIGS,
               "single": ["1n", "1n_sc"]}


def select_perf_args(case: "PerfCase", experiment: str, geo: int) -> list:
    """Pick the argv for one (experiment, geo) cell.
    fix/single -> the single fixed list.  strong/weak -> the geo entry of a
    per-geo dict, or a constant list broadcast to every geo.  When an app has
    NO strong/weak args (strong_args/weak_args is None -- i.e. it was not
    designed as a scaling core app), fix_args is broadcast to every geo: a
    fixed workload run at 1n_sc/2n_sc/4n_sc, which is exactly the strong-scaling
    (constant-total-work) semantics and lets the RR-divergence survey measure
    EVERY app across the multinode geometry."""
    if experiment in ("fix", "single"):
        return case.fix_args
    a = case.strong_args if experiment == "strong" else case.weak_args
    if a is None:
        return case.fix_args
    return a[geo] if isinstance(a, dict) else a

SW_DATA = (REPO / "third_party" / "ocr-apps" / "apps" / "smithwaterman"
           / "datasets")
CHOLESKY_PERF_MAT = "/tmp/arts_cholesky_perf5k.mat"
BASIC_IO_DAT = "/tmp/arts_basicIO_test.dat"


def _single(name, marker, args,
            scale_skip="NULL_HINT: no cross-node speedup (RR-verified 2026-07-06)"):
    """A non-scaling app: only fix_args (the single-experiment two-point
    concurrency workload). scale_skip documents why it is excluded from the
    strong/weak cross-node matrix.  Empirically re-verified 2026-07-06: with the
    no-hint round-robin EDT placement policy in force, NULL_HINT apps still show
    no 1n_sc->2n_sc speedup -- their EDTs spread across ranks but the centralized
    DBs / dependency chains make cross-node comm dominate (fib 1.5->57 s,
    reduction 2->32 s, p2p 3.4->50 s at 2n_sc; Stencil1D/LCS stay flat).
    Fixed-size fixtures additionally have no workload knob."""
    return PerfCase(name, name, marker, fix_args=args,
                    strong_args=None, weak_args=None, scale_skip=scale_skip)


def _scal(name, marker, fix, strong, weak, extra=None):
    """A cross-node scaling app (real OCR_HINT_EDT_AFFINITY distribution).
    fix = the single-experiment workload (48w / 12w two points); strong =
    total work fixed while the grid grows 12->24->48 (wall should shrink);
    weak = per-node work fixed while total & grid grow with node count.
    Geo keys are 1/2/4 (the _sc geometry: 1n_sc, 2n_sc, 4n_sc)."""
    return PerfCase(name, name, marker, fix_args=fix,
                    strong_args=strong, weak_args=weak,
                    extra_scalars=extra or {})


# Cross-node scaling core (calibrated 2026-07-06 on arts_mrnew_lazy, post
# leak-fix, at the cbgpu02 _sc geometry: 12 workers/node, 1/2/4 nodes on the
# 48-core box).  On this LOCALHOST-simulated multinode (one physical box,
# libfabric loopback comm), compute-bound apps (graph500, hpcg, Stencil2D,
# RSBench, stream_dist) show real strong speedup; comm-bound apps (CoMD, hpgmg)
# anti-scale as loopback halo/reduction cost dominates -- both are honest
# results and are sized so every cell completes within the 45 s cell budget.
_SCALABLE: list = [
    # graph500: SCALE EDGEFACTOR R C. grid=R*C (must be pow2). strong: SCALE
    # fixed, grid 8/16/32; weak: SCALE grows with grid so per-worker work fixed.
    _scal("graph500", r"nodes \d+", ["22","8","8","4"],
          strong={1: ["22","8","4","2"], 2: ["22","8","4","4"], 4: ["22","8","8","4"]},
          weak={1: ["20","8","4","2"], 2: ["21","8","4","4"], 4: ["22","8","8","4"]},
          extra={"kernel2_ns": r"\[kernel2 time ([0-9.eE+-]+)\]",
                 "mteps": r"mean MTEPS ([0-9.eE+-]+)"}),
    # CoMD_sdsc2: -x -y -z (cells) -N (steps). Cells auto-distribute to ranks.
    # strong: box fixed; weak: box (and total cells) grow with node count.
    _scal("CoMD_sdsc2", r"Final energy",
          ["-x","40","-y","40","-z","40","-N","20"],
          strong={1: ["-x","44","-y","44","-z","44","-N","8"],
                  2: ["-x","44","-y","44","-z","44","-N","8"],
                  4: ["-x","44","-y","44","-z","44","-N","8"]},
          weak={1: ["-x","40","-y","40","-z","40","-N","6"],
                2: ["-x","56","-y","40","-z","40","-N","6"],
                4: ["-x","56","-y","56","-z","40","-N","6"]}),
    # hpcg_intel: npx npy npz m iters. grid=npx*npy*npz; m rounds UP to x16.
    # strong: local m shrinks as grid grows (total ~fixed); weak: m fixed.
    _scal("hpcg_intel", r"final deviation", ["4","4","3","64","15"],
          strong={1: ["2","2","3","96","15"], 2: ["4","2","3","80","15"],
                  4: ["4","4","3","64","15"]},
          weak={1: ["2","2","3","64","15"], 2: ["4","2","3","64","15"],
                4: ["4","4","3","64","15"]}),
    # nekbone: Rx Ry Rz Ex Ey Ez pDOF CGcount. Rtotal=grid.  The rank AND element
    # dims MUST be non-increasing (Rx>=Ry>=Rz, Ex>=Ey>=Ez) or setup errors out --
    # so grids are 3*2*2=12, 4*3*2=24, 4*4*3=48 (not 2*2*3).  strong: per-rank
    # Etotal shrinks as grid grows (total elements fixed); weak: per-rank fixed.
    _scal("nekbone", r"FinalEDT", ["4","4","3","4","4","4","12","50"],
          strong={1: ["3","2","2","8","8","4","12","50"],
                  2: ["4","3","2","8","4","4","12","50"],
                  4: ["4","4","3","4","4","4","12","50"]},
          weak={1: ["3","2","2","4","4","4","12","50"],
                2: ["4","3","2","4","4","4","12","50"],
                4: ["4","4","3","4","4","4","12","50"]}),
    # RSBench_intel_sharedDB: -l (global lookups, split among -t) -t (EDTs=grid).
    # strong: -l fixed, -t grows; weak: -l grows with -t (per-EDT work fixed).
    _scal("RSBench_intel_sharedDB", r"RS_CHECKSUM", ["-l","3000000","-t","48"],
          strong={1: ["-l","3000000","-t","12"], 2: ["-l","3000000","-t","24"],
                  4: ["-l","3000000","-t","48"]},
          weak={1: ["-l","750000","-t","12"], 2: ["-l","1500000","-t","24"],
                4: ["-l","3000000","-t","48"]}),
    # Stencil2D_intel_chandra: npoints nranks ntimesteps. nranks=grid (2D split).
    # strong: npoints fixed, nranks grows; weak: npoints~sqrt(nodes) (area/node fixed).
    _scal("Stencil2D_intel_chandra", r"L1 norm", ["20000","48","30"],
          strong={1: ["20000","12","30"], 2: ["20000","24","30"],
                  4: ["20000","48","30"]},
          weak={1: ["10000","12","30"], 2: ["14142","24","30"],
                4: ["20000","48","30"]}),
    # hpgmg: log2_box_dim target_boxes. Boxes auto-home box%node_count (Cat B).
    # strong: boxes fixed (32); weak: boxes grow with node count (8/16/32).
    _scal("hpgmg", r"\|\|error\|\|", ["6","27"],
          strong={1: ["6","32"], 2: ["6","32"], 4: ["6","32"]},
          weak={1: ["6","8"], 2: ["6","16"], 4: ["6","32"]}),
    # stream_dist: array_size num_threads ntimes (argvized 2026-07-06); the only
    # bandwidth kernel with OCR_HINT_EDT_AFFINITY (distributes across ranks).
    # strong: array fixed, threads=grid grows; weak: array grows with threads.
    _scal("stream_dist", r"STREAM checksum", ["8000000","48","1600"],
          strong={1: ["32000000","12","400"], 2: ["32000000","24","400"],
                  4: ["32000000","48","400"]},
          weak={1: ["8000000","12","400"], 2: ["16000000","24","400"],
                4: ["32000000","48","400"]}),
]


# ---------------------------------------------------------------------------
# Non-scaling apps: run only in the "single" experiment (whole OCR-pair suite
# at the 48-worker single-node geometry PLUS the 12-worker 1n_sc geometry -- a
# two-point within-node concurrency-scaling axis every app can run).  fix_args
# are calibrated (2026-07-06, post leak-fix) so the arts (mrnew_lazy) wall is
# ~3 s at 48 workers, with human-natural parameter values (powers of two /
# multiples of ten); worker/rank-grid parameters are pinned to 48 (or the
# nearest power of two the app demands).  Apps with no CLI workload knob run at
# their built-in size ("fixed") and their wall is reported as-is.  Every entry
# carries a scale_skip reason (why it is not in the strong/weak matrix).
# ---------------------------------------------------------------------------


_SINGLE_ONLY: list = [
    # --- calibrated kernels (CLI workload knob; ~3 s at 48 workers) ---
    _single("fibonacci", r"answer is\s*\d+", ["33"]),
    # No CLI board size yields ~3 s (15 -> 0.9 s, 16 -> 7 s); 16 is kept as the
    # meaningful full-workload point (its wall is reported as-is).
    _single("nqueens", r"sols:\s*\d+", ["16", "8"]),
    # ~41 GB live tile matrix at this size — feasible on arts since the
    # pool-lifecycle fix; the references need far beyond the budget here.
    _single("smithwaterman", r"score:\s*\d+",
            ["100", "100", f"{SW_DATA}/string1-large.txt",
             f"{SW_DATA}/string2-large.txt", f"{SW_DATA}/score-large.txt"],
            scale_skip="single-node wavefront (dataset-bound)"),
    _single("fft", r"FFT checksum", ["23"]),
    # No-arg triangle solves the full puzzle — its maximum problem size
    # (~0.8 s on arts at 48 workers; the fine-grained EDT tree is the point).
    _single("triangle", r"final count\s+\d+", [], scale_skip="fixed-size full-puzzle search"),
    _single("p2p", r"PASS checksum", ["48", "100", "100", "1400"]),
    _single("CoMD_sdsc", r"Final energy", ["-x","36","-y","36","-z","36","-N","2"],
            scale_skip="single-node CoMD variant"),
    _single("CoMD_intel_chandra", r"Initial energy",
            ["-x","60","-y","60","-z","60","-N","6","-n","1"],
            scale_skip="single-node CoMD variant"),
    _single("CoMD_intel_chandra_tiled", r"Final energy",
            ["-x","50","-y","50","-z","50","-N","4"],
            scale_skip="single-node CoMD variant"),
    # reduction-algorithm variants of the hpcg_intel scaling core; the MN matrix
    # is bounded to the core so the scaling story is one hpcg curve.
    _single("hpcg_intel_Eager", r"final deviation", ["4","4","3","64","15"],
            scale_skip="reduction variant of hpcg_intel (scaling core)"),
    _single("hpcg_intel_Eager_Collective", r"final deviation",
            ["4","4","3","64","15"],
            scale_skip="reduction variant of hpcg_intel (scaling core)"),
    _single("Stencil1D_intel_chandra", r"Solution validates",
            ["20000", "48", "30"]),
    _single("Stencil2D_intel_channelEVTs", r"Computed L1 norm",
            ["20000", "48", "30"],
            scale_skip="channel-comm variant of Stencil2D_intel_chandra (scaling core)"),
    _single("miniAMR_intel", r"Grand Total Checksum",
            ["--nx","64","--ny","64","--nz","64","--num_tsteps","6",
             "--num_objects","1"], scale_skip="single-node AMR"),
    # Cost is init-dominated (blocks of 16^3): timestep count barely moves the
    # wall; init_x/y/z set the decomposition.
    _single("miniAMR_intel_chandra", r"Done",
            ["--nx","16","--ny","16","--nz","16","--init_x","2","--init_y","2",
             "--init_z","2","--num_tsteps","10","--num_refine","1"],
            scale_skip="init-dominated fixed decomposition"),
    # OCR port parses workload flags but they do not change the computed
    # problem (verified byte-identical output) — effectively fixed-size.
    _single("miniAMR_intel_bryan", r"miniAMR complete", [],
            scale_skip="argv ignored; byte-identical fixed size"),
    # class A CG (~1.5 s).  Class B is size-fixed at ~43 s @48w / >170 s @12w
    # (every runtime times out at 1n_sc), so it is not in the perf matrix.
    _single("npb_cg", r"Verification SUCCESSFUL", ["-t","A","-b","25"]),
    _single("tempest", r"CROSS-CHECKING NEIGHBOR DATA EXCHANGE", ["96"],
            scale_skip="single-node comm sanity (no science scalar)"),
    # Wall saturates ~3 s regardless of maxX/tolerance (subdivision breadth
    # bounded by numRanks) — effectively a fixed ~3 s kernel at 48 ranks.
    _single("curvefit", r"SUCCESS", ["48","0.01","0.5","1000000"],
            scale_skip="wall saturates at numRanks"),
    _single("reduction_intel", r"T\d+ i1", ["48","2","20000"]),
    _single("reduction_intel_chandra", r"RESULT = ", ["650000"]),
    _single("LCS_distributed_ST", r"LCS length:", ["57344","1024","48"]),
    _single("LCS_shared", r"LCS length:", ["40960","1024","48"]),
    _single("LCS_all_db_distributed", r"LCS length:", ["65536","1024","48"]),
    _single("RSBench_intel", r"Lookups:", ["-l","400000"],
            scale_skip="non-shared RSBench (RSBench_intel_sharedDB is the scaling core)"),
    _single("XSBench_intel", r"XSBench grid checksum",
            ["-s","small","-g","1000","-l","400000"],
            scale_skip="single-node; memory axis fixed"),
    _single("XSBench_intel_sharedDB", r"Workload\s+\(unit\)",
            ["-s","small","-g","1000","-l","3500000"],
            scale_skip="single-node; memory axis fixed"),
    _single("uts", r"UTS Tree size",
            ["-g","1","-t","1","-a","2","-d","14","-b","7","-r","220"]),
    _single("cholesky", r"CHOLESKY trace",
            ["--ds","5000","--ts","100","--fi",CHOLESKY_PERF_MAT]),
    # The default workload (FANOUT=100 DEPTH=200, 20k DB churn) SIGSEGVs on
    # ALL THREE runtimes (vendor-app issue, not runtime-specific) — kept at
    # the correctness-scale workload.
    _single("dbctrl", r"Total time", ["6","6","1024"],
            scale_skip="fixed-size DB-tree stress"),
    # --- argvized bandwidth/sort kernels (single-node; no EDT affinity) ---
    _single("quicksort", r"Sorting Finished", ["4000000","1000000"],
            scale_skip="single-node sort, no EDT affinity"),
    _single("stream", r"STREAM_RESULT", ["4000000","48","4000"],
            scale_skip="single-node bandwidth kernel"),
    _single("highbw", r"HIGHBW_WORK_SUM", ["48","8388608","1000"],
            scale_skip="single-node bandwidth kernel"),
    _single("prodcon", r"MB/s", ["48","8000000","6000"],
            scale_skip="single-node producer/consumer kernel"),
    # --- fixed-size fixtures (no CLI workload knob; overhead floor) ---
    _single("multigen", r"End leaf1, result", [], scale_skip="fixed-size fixture"),
    _single("multigen_2", r"End leaf1, result", [], scale_skip="fixed-size fixture"),
    _single("globalsum_cgShim", r"CG0 T\d+\s+0 value", [], scale_skip="fixed-size fixture"),
    _single("globalsum_cgNoShim", r"CG0 T\d+\s+0 value", [], scale_skip="fixed-size fixture"),
    _single("globalsum_pcg", r"CG0 T\d+\s+0 value", [], scale_skip="fixed-size fixture"),
    _single("stencil1D_sticky", r"S3 i9 valu", [], scale_skip="fixed-size fixture"),
    _single("sar_large", r"SAR detects:", [], scale_skip="compile-time dataset"),
    _single("basicIO", r"BASICIO_CHK", ["0","10",BASIC_IO_DAT], scale_skip="IO fixture"),
    _single("printf", r"Hello from mainEdt", [], scale_skip="micro-fixture"),
    _single("testlibs", r"Testing strlen", [], scale_skip="micro-fixture"),
    _single("task_priorities", r"Hello from 9", [], scale_skip="micro-fixture"),
    _single("cache_offset", r"CACHE_OFFSET_CHK", [], scale_skip="fixed-size micro-fixture"),
    _single("xeonNumaSize", r"DONE!", ["-dcpu"], scale_skip="fixed-size NUMA probe"),
    _single("dbcreate_matrix", r"CELLS_OK=", [], scale_skip="DB-create capability probe"),
    _single("miniAMR_forkbomb", r"BLOCK 0 finished",
            ["--nx","1","--ny","1","--nz","1","--num_tsteps","3",
             "--num_refine","0"], scale_skip="fork-bomb stress"),
]


# ---------------------------------------------------------------------------
# Weak-scaling arg derivation for the non-core apps (user directive 2026-07-06).
# Scale each app's AUDIT work-axis so 1n_sc -> 2n_sc = 2x work, 4n_sc = 4x work
# (per-node work held constant, total grows with node count).  Linear knobs grow
# x2/x4; a log2 size knob (fft) or an exponential tree/recursion-depth knob
# (fib/uts) grows +1/+2 (~2x/~4x); a quadratic-work size knob (LCS wavefront)
# grows x1.41/x2.  Values snap to a natural number (power of two, or k*10^e).
# Apps with NO 2x-scalable work knob are WEAK-EXEMPT (weak_args stays None ->
# skipped from weak, since their weak would just duplicate the strong fix_args
# broadcast).  selector = positional index into fix_args, or a flag whose
# FOLLOWING token holds the value.
_WEAK_AXIS = {
    "fibonacci": (0, "exp"), "fft": (0, "log2"), "uts": ("-d", "exp"),
    "p2p": (3, "lin"), "tempest": (0, "lin"),
    "CoMD_sdsc": ("-N", "lin"), "CoMD_intel_chandra": ("-N", "lin"),
    "CoMD_intel_chandra_tiled": ("-N", "lin"),
    "hpcg_intel_Eager": (4, "lin"), "hpcg_intel_Eager_Collective": (4, "lin"),
    "Stencil1D_intel_chandra": (2, "lin"), "Stencil2D_intel_channelEVTs": (2, "lin"),
    "miniAMR_intel": ("--num_tsteps", "lin"),
    "miniAMR_intel_chandra": ("--num_tsteps", "lin"),
    "reduction_intel": (2, "lin"), "reduction_intel_chandra": (0, "lin"),
    "LCS_distributed_ST": (0, "quad"), "LCS_shared": (0, "quad"),
    "LCS_all_db_distributed": (0, "quad"),
    "RSBench_intel": ("-l", "lin"), "XSBench_intel": ("-l", "lin"),
    "XSBench_intel_sharedDB": ("-l", "lin"),
    "dbctrl": (2, "lin"), "quicksort": (0, "lin"), "stream": (2, "lin"),
    "highbw": (2, "lin"), "prodcon": (2, "lin"),
}
# Weak-exempt (no ~2x-scalable work knob): app -> reason (for ledger/README).
_WEAK_EXEMPT = {
    "nqueens": "board size super-exponential (16->17 ~15x); no ~2x step",
    "smithwaterman": "fixed input dataset (string lengths from file)",
    "triangle": "fixed full-puzzle size (no work knob)",
    "npb_cg": "discrete problem-class knob S/W/A/B/C; no ~2x step",
    "curvefit": "maxX/tol ignored, wall saturates at numRanks (no work knob)",
    "cholesky": "fixed dense input-matrix fixture (ds bound by --fi file)",
    "sar_large": "compile-time dataset", "basicIO": "IO fixture",
    "multigen": "fixed-size fixture", "multigen_2": "fixed-size fixture",
    "globalsum_cgShim": "fixed-size fixture", "globalsum_cgNoShim": "fixed-size fixture",
    "globalsum_pcg": "fixed-size fixture", "stencil1D_sticky": "fixed-size fixture",
    "printf": "micro-fixture", "testlibs": "micro-fixture",
    "task_priorities": "micro-fixture", "cache_offset": "fixed-size micro-fixture",
    "xeonNumaSize": "fixed-size NUMA probe", "dbcreate_matrix": "DB-create probe",
    "miniAMR_forkbomb": "fork-bomb stress (fixed)",
    "miniAMR_intel_bryan": "argv ignored (byte-identical fixed size)",
}


def _nat(x):
    """Snap x to the nearest natural number: a power of two, or k*10^e (k in
    {1,2,3,4,5,6,8})."""
    import math
    x = max(1.0, float(x))
    cands = []
    p = 1
    while p <= x * 2.5:
        cands.append(p)
        p *= 2
    e = 10 ** max(0, int(math.floor(math.log10(x))) - 1)
    for ee in (e, e * 10, e * 100):
        for k in (1, 2, 3, 4, 5, 6, 8):
            cands.append(k * ee)
    cands = sorted({int(c) for c in cands if c >= 1})
    return min(cands, key=lambda c: abs(math.log(c) - math.log(x)))


def _axis_get(args, sel):
    return float(args[sel] if isinstance(sel, int) else args[args.index(sel) + 1])


def _axis_set(args, sel, val):
    a = list(args)
    i = sel if isinstance(sel, int) else a.index(sel) + 1
    a[i] = str(val)
    return a


def _derive_weak(case, sel, kind):
    base = _axis_get(case.fix_args, sel)
    if kind == "lin":
        v2, v4 = _nat(base * 2), _nat(base * 4)
    elif kind == "quad":                 # work ~ param^2 -> param x1.41 / x2
        v2, v4 = _nat(base * 2 ** 0.5), _nat(base * 2)
    elif kind in ("log2", "exp"):        # log / exponential knob: +1 / +2
        v2, v4 = int(base) + 1, int(base) + 2
    else:
        raise ValueError(kind)
    return {1: list(case.fix_args),
            2: _axis_set(case.fix_args, sel, v2),
            4: _axis_set(case.fix_args, sel, v4)}


for _c in _SINGLE_ONLY:
    if _c.name in _WEAK_AXIS:
        _c.weak_args = _derive_weak(_c, *_WEAK_AXIS[_c.name])


# Unified table: every app is one PerfCase.  "single" and "strong" run all of
# BENCHES (strong broadcasts fix_args where no strong_args); "weak" runs every
# app that has a weak axis (designed core dict OR derived non-core dict) -- the
# weak-exempt fixed-workload apps are skipped there (their weak == their strong).
# PERF_BENCHES / SINGLE_BENCHES are kept as derived views.
BENCHES: list = _SCALABLE + _SINGLE_ONLY
SINGLE_BENCHES: list = BENCHES
PERF_BENCHES: list = _SCALABLE


def benches_for(experiment: str, only=None) -> list:
    """Apps to run for an experiment.  NO app is pre-excluded by any skip field
    (scale_skip / mrmw_skip / ... are analysis metadata only, never run gates).
      - single, strong: EVERY app (strong broadcasts fix_args where it has no
        designed strong_args -- a fixed workload = constant-total-work).
      - weak: every app that HAS a weak axis (the designed core dict OR a derived
        non-core dict); the weak-exempt fixed-workload apps (weak_args is None)
        are skipped because their weak would exactly duplicate their strong cell.
    `only` (a set of names) restricts the result when non-empty."""
    if experiment == "weak":
        table = [b for b in BENCHES if b.weak_args is not None]
    else:
        table = BENCHES
    return [b for b in table if b.name in only] if only else table


def stage_single_fixtures() -> None:
    """Input files the single-suite cases read.  The NxN diagonally-dominant
    SPD matrix gives cholesky a real decomposition workload (calibrated to
    ~3 s at 48 workers); basicIO's 10-u64 file matches the correctness harness
    fixture."""
    p = Path(CHOLESKY_PERF_MAT)
    if not p.exists():
        n = 5000
        with open(p, "w") as f:
            for r in range(n):
                row = ["1.0"] * n
                row[r] = f"{n + 1}.0"
                f.write(" ".join(row) + "\n")
    b = Path(BASIC_IO_DAT)
    if not b.exists():
        b.write_text("\n".join(str(i) for i in range(10)) + "\n")


# ---------------------------------------------------------------------------
# Hang-detection FSM + retry wrapper (Task 6).
# ---------------------------------------------------------------------------

# Geometries (ranks) the "_sc" scalability series exercises.  1n_sc is a
# single-rank direct exec (no launcher); 2/4/8n_sc use the same launcher as
# the corresponding capacity geometry (arts self-fork / mpirun) but point at
# the dedicated cbgpu02/{n}_sc.cfg pair (see configs/local/cbgpu02,
# configs/mpi/cbgpu02) instead of the capacity-matrix cfgs.
_SC_GEOS = {"1n_sc": 1, "2n_sc": 2, "4n_sc": 4}


def classify_run(rc: int, marker_seen: bool, proc_alive: bool) -> str:
    # A post-result hang: completion marker printed but process did not exit
    # (rc==124 wall-timeout, or reaped). e2e was already captured in-runtime.
    if marker_seen and (rc == 124 or proc_alive):
        return "SHUTDOWN_HANG"
    if rc == 0 and marker_seen:
        return "OK"
    if marker_seen:           # nonzero rc but result present → still usable e2e, but flag
        return "OK"
    if rc == 124:
        # Wall-timeout with no result: for a perf matrix this is a measurement
        # ("slower than the budget"), not a transient failure — retrying a
        # structurally-too-slow cell would burn iters*retries*budget for
        # nothing.  The caller distinguishes a one-off hang from a structural
        # timeout by allowing a single same-iteration retry.
        return "TIMEOUT"
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
    either marker never counts as valid, regardless of rc.  TIMEOUT (rc==124,
    no result at all) passes through: it is terminal-ish (single same-iter
    retry, then the cell stops) rather than retried like COMPUTE_FAIL."""
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
            # the "_sc" cfg (12 workers, no node_count) instead of the
            # capacity 1n cfg (48 workers).
            if rt.kind == "arts":
                return runner.run_ocr(
                    case.name, case.ocr_base, args, "arts", suffix=rt.suffix,
                    width=12,
                    cfg_path=cfg_base / "local" / "cbgpu02" / "1n_sc.cfg",
                    extra_env=arts_env)
            elif rt.kind == "xsocr":
                return runner.run_ocr(
                    case.name, case.ocr_base, args, "xsocr", width=12,
                    cfg_path=cfg_base / "mpi" / "cbgpu02" / "1n_sc.cfg")
            else:  # ocrvx -- no cfg file consumed; np=1 already direct-execs
                return runner.run_ocrvx_mpi(case.name, case.ocr_base, args, tbb=12)
        else:
            if rt.kind == "arts":
                return runner.run_arts_mn(
                    case.name, case.ocr_base, args, geo, suffix=rt.suffix,
                    cfg_path=cfg_base / "local" / "cbgpu02" / f"{geo}n_sc.cfg",
                    extra_env=arts_env)
            elif rt.kind == "xsocr":
                return runner.run_xsocr_mpi(
                    case.name, case.ocr_base, args, geo,
                    cfg_path=cfg_base / "mpi" / "cbgpu02" / f"{geo}n_sc.cfg",
                    tpn=12)
            else:  # ocrvx -- np-driven; _sc geometry = 12 threads per rank
                return runner.run_ocrvx_mpi(case.name, case.ocr_base, args, np=geo,
                                            tpn=12, tbb=12)
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
    cell_timed_out = False
    for i in range(iters):
        if cell_timed_out:
            break
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
            if status == "TIMEOUT":
                # Terminal immediately: a timeout IS the measurement ("slower
                # than the budget") — record one TIMEOUT row and stop the
                # whole cell; later iterations cannot get faster, and hang-vs-
                # slow triage happens offline from the recorded cell.
                cell_timed_out = True

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

    Reference runtimes (xsocr/ocr-vx) have no worker/progress-split IO
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
    p.add_argument("--target", default="cbgpu02", choices=["laptop", "cbgpu02"],
                   help="Machine geometry: laptop (14-thread) or cbgpu02 (48-thread)")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated bench names to restrict to")
    p.add_argument("--node", type=str, default="",
                   help="Restrict to a single node-config, e.g. --node 1n")
    p.add_argument("--experiment", type=str, default="fix,strong,weak",
                   help="Comma-separated experiments to run: fix,strong,weak,single")
    p.add_argument("--runtimes", type=str, default="",
                   help="Comma-separated runtime keys to restrict to "
                        "(e.g. mrnew_lazy,xsocr,ocrvx); default all")
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

    runtimes = RUNTIMES
    if args.runtimes:
        keys = {s.strip() for s in args.runtimes.split(",") if s.strip()}
        unknown = keys - {rt.key for rt in RUNTIMES}
        if unknown:
            p.error(f"unknown --runtimes keys: {sorted(unknown)}")
        runtimes = [rt for rt in RUNTIMES if rt.key in keys]

    only = {s.strip() for s in args.only.split(",") if s.strip()}

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
    if "single" in selected_exps:
        stage_single_fixtures()
    for experiment in selected_exps:
        node_configs = EXPERIMENTS[experiment]
        if args.node:
            node_configs = [_parse_node_arg(args.node)]
        for case in benches_for(experiment, only):
            for node in node_configs:
                for rt in runtimes:
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
