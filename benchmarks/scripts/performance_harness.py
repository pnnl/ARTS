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
correctness `build_release_ocr_val_wb`, not a runtime flag on the same
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
import shutil
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


E2E_RE = re.compile(r"\[E2E\]\s+(\d+)")


def parse_reference_e2e(text: str):
    """Extract the reference-runtime (xsocr/ocr-vx/baseline) wall-time marker
    `[E2E] <nanoseconds>` from stdout. Returns None if absent."""
    m = E2E_RE.search(text)
    return float(m.group(1)) if m else None


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_marker_scalar(text: str, marker: str):
    """Correctness-triage scalar: the last numeric token on the first line
    matching the app's completion marker.  Not a perf metric — captured so a
    wrong answer / silent divergence is at least LOGGED per cell and
    cross-runtime comparable post-hoc (see write_validation).  None when the
    marker never printed or its line carries no number (validation reports
    NO_SCALAR rather than guessing)."""
    m = re.search(marker, text)
    if not m:
        return None
    start = text.rfind("\n", 0, m.start()) + 1
    end = text.find("\n", m.start())
    if end == -1:
        end = len(text)
    nums = _NUM_RE.findall(text[start:end])
    return nums[-1] if nums else None


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
CHOLESKY_PERF_MAT_LG = "/tmp/arts_cholesky_perf7k5.mat"
# Weak-series fixtures: ds grows by 2^(1/3) per work doubling (flops ~ ds^3),
# rounded to a tile-size multiple. cholesky 5000->6300 (2.000x) -> 7900 (3.944x);
# cholesky_blas 7500 -> 9400 (1.968x) -> 11900 (3.994x).
CHOLESKY_WEAK_MATS = {
    "cholesky": {1: (5000, CHOLESKY_PERF_MAT), 2: (6300, "/tmp/arts_cholesky_perf6k3.mat"),
                 4: (7900, "/tmp/arts_cholesky_perf7k9.mat")},
    "cholesky_blas": {1: (7500, CHOLESKY_PERF_MAT_LG), 2: (9400, "/tmp/arts_cholesky_perf9k4.mat"),
                      4: (11900, "/tmp/arts_cholesky_perf11k9.mat")},
}
BASIC_IO_DAT = "/tmp/arts_basicIO_test.dat"

# SAR runtime-input assets: one huge pulse dataset (generated once by
# datagen-huge) serves every problem_size_scaling Parameter file -- the ladder
# varies only the image grid (Ix=Iy=400*(k+1)), so wall grows ~ Ix^2.
SAR_ROOT = REPO / "third_party" / "ocr-apps" / "apps" / "sar"
SAR_HUGE_DATA = SAR_ROOT / "datasets" / "huge"
SAR_PSS_PARAMS = SAR_ROOT / "ocr" / "problem_size_scaling"

def _sar_pss_args(p: int) -> list:
    return [str(SAR_HUGE_DATA / "Data.bin"),
            str(SAR_HUGE_DATA / "PlatformPosition.bin"),
            str(SAR_HUGE_DATA / "PulseTransmissionTime.bin"),
            "/tmp/arts_sar_detects.txt",
            str(SAR_PSS_PARAMS / f"Parameter{p}.txt")]

BASIC_IO_1M = str(REPO / "third_party" / "ocr-apps" / "apps" / "basicIO" / "ocr"
                  / "input_1000000.txt")
BASIC_IO_250K = "/tmp/arts_basicIO_250k.txt"
BASIC_IO_500K = "/tmp/arts_basicIO_500k.txt"
SW_XL = ("/tmp/arts_sw_string1_xl.txt", "/tmp/arts_sw_string2_xl.txt",
         "/tmp/arts_sw_score_xl.txt")
SW_XXL = ("/tmp/arts_sw_string1_xxl.txt", "/tmp/arts_sw_string2_xxl.txt",
          "/tmp/arts_sw_score_xxl.txt")


def _bench(name, marker, args):
    """An app declared with only fix_args.  Every experiment still runs it:
    strong broadcasts fix_args (a fixed workload = constant total work) and
    weak uses a derived axis where one exists (see the derivation tables
    below)."""
    return PerfCase(name, name, marker, fix_args=args,
                    strong_args=None, weak_args=None)


def _bench_axes(name, marker, fix, strong, weak, extra=None, ocr_base=None):
    """An app with hand-designed strong/weak arg tables.
    fix = the single-experiment workload (48w / 12w two points); strong =
    total work fixed while the grid grows 12->24->48 (wall should shrink);
    weak = per-node work fixed while total & grid grow with node count.
    Geo keys are 1/2/4 (the _sc geometry: 1n_sc, 2n_sc, 4n_sc).
    ocr_base overrides the binary basename when the display name differs."""
    return PerfCase(name, ocr_base or name, marker, fix_args=fix,
                    strong_args=strong, weak_args=weak,
                    extra_scalars=extra or {})


# Cross-node scaling core (calibrated 2026-07-06 on arts_val_wb(then 'val_wb'), post
# leak-fix, at the cbgpu02 _sc geometry: 12 workers/node, 1/2/4 nodes on the
# 48-core box).  On this LOCALHOST-simulated multinode (one physical box,
# libfabric loopback comm), compute-bound apps (graph500, hpcg, Stencil2D,
# RSBench, stream_dist) show real strong speedup; comm-bound apps (CoMD, hpgmg)
# anti-scale as loopback halo/reduction cost dominates -- both are honest
# results and are sized so every cell completes within the 45 s cell budget.
_DESIGNED_AXES: list = [
    # graph500: SCALE EDGEFACTOR R C. grid=R*C (must be pow2). strong: SCALE and
    # the process grid held IDENTICAL at every node count (total work fixed, result
    # decomposition-invariant) -- the runtime spreads the fixed grid (32=8*4,
    # dividing evenly into 1/2/4 nodes) over the physical nodes. weak: SCALE grows
    # with the grid so per-worker work stays fixed.
    # R*C is the only concurrency knob (per-level width; R, C powers of two).
    # 16*16 = 256 clears the concurrency floor at every geometry; the grid is
    # held fixed on the weak axis too (concurrency is a machine parameter,
    # not a node-count function) while SCALE grows with the node count.
    _bench_axes("graph500", r"nodes \d+", ["22","8","16","16"],
          strong=["22","8","16","16"],
          weak={1: ["21","8","16","16"], 2: ["22","8","16","16"],
                4: ["23","8","16","16"]},
          extra={"kernel2_ns": r"\[kernel2 time ([0-9.eE+-]+)\]",
                 "mteps": r"mean MTEPS ([0-9.eE+-]+)"}),
    # CoMD_sdsc2: -x -y -z (cells) -N (steps). Cells auto-distribute to ranks.
    # strong: box fixed; weak: box (and total cells) grow with node count.
    _bench_axes("CoMD_sdsc2", r"Final energy",
          ["-x","40","-y","40","-z","40","-N","20"],
          strong={1: ["-x","44","-y","44","-z","44","-N","8"],
                  2: ["-x","44","-y","44","-z","44","-N","8"],
                  4: ["-x","44","-y","44","-z","44","-N","8"]},
          weak={1: ["-x","40","-y","40","-z","40","-N","6"],
                2: ["-x","56","-y","40","-z","40","-N","6"],
                4: ["-x","56","-y","56","-z","40","-N","6"]}),
    # hpcg_intel: npx npy npz m iters. grid=npx*npy*npz; m rounds UP to x16.
    # strong: grid AND m held IDENTICAL at every node count so total DOF (=grid*m^3)
    # is EXACTLY constant and the "final deviation" scalar is decomposition-invariant
    # -- 48 logical subdomains (4*4*3) divide evenly into 1/2/4 nodes (48/24/12 per
    # node). m must be a multiple of 16; a per-node-varying grid cannot hold total DOF
    # fixed because m quantizes coarsely (this is why the old varying tuple drifted the
    # scalar +15-18%). weak: m fixed, grid grows so per-node DOF stays fixed.
    _bench_axes("hpcg_intel", r"final deviation", ["4","4","3","64","15"],
          strong=["4","4","3","64","15"],
          weak={1: ["2","2","3","64","15"], 2: ["4","2","3","64","15"],
                4: ["4","4","3","64","15"]}),
    # nekbone: Rx Ry Rz Ex Ey Ez pDOF CGcount. Rtotal=grid.  The rank AND element
    # dims MUST be non-increasing (Rx>=Ry>=Rz, Ex>=Ey>=Ez) or setup errors out.
    # strong: rank grid AND element grid held IDENTICAL at every node count so the
    # decomposition (and the FP-order-sensitive result checksum) is invariant -- 48
    # logical ranks (4*4*3) divide evenly into 1/2/4 nodes.  weak: per-rank elements
    # fixed, rank grid grows so per-node work stays fixed.
    _bench_axes("nekbone", r"FinalEDT", ["4","4","3","4","4","4","12","50"],
          strong=["4","4","3","4","4","4","12","50"],
          weak={1: ["3","2","2","4","4","4","12","50"],
                2: ["4","3","2","4","4","4","12","50"],
                4: ["4","4","3","4","4","4","12","50"]}),
    # RSBench_intel_sharedDB: -l (global lookups, split among -t) -t (EDTs=grid).
    # strong: -l AND -t held IDENTICAL at every node count so both the total lookup
    # count and the per-EDT lookup partition (hence the FP-order-sensitive checksum)
    # are invariant -- 48 EDTs divide evenly into 1/2/4 nodes (48/24/12 per node).
    # weak: -l grows with -t (per-EDT work fixed).
    _bench_axes("RSBench_intel_sharedDB", r"RS_CHECKSUM", ["-l","3000000","-t","48"],
          strong=["-l","3000000","-t","48"],
          weak={1: ["-l","750000","-t","12"], 2: ["-l","1500000","-t","24"],
                4: ["-l","3000000","-t","48"]}),
    # XSBench_dist: -l (global lookups, tiled across ranks then threads) -t
    # (PER-NODE SPMD-worker count; total SPMD width = nodes x -t, auto-derived
    # from the PD count, no -p).  strong: -l AND -t held IDENTICAL at every node
    # count so both the total lookups and the args are constant while total SPMD
    # width (nodes x -t) grows -- fixed work over more parallelism.  -t is set at
    # or below the per-node worker budget (<= 11 workers on the MN _sc cfgs) so no
    # config over-decomposes a worker.  weak: -l grows with the total worker count
    # (per-worker lookups ~fixed).  -s/-g size the per-node table replica (fixed
    # across the sweep).  -l 175M calibrated to ~5 s at 1n_sc.
    _bench_axes("XSBench_dist", r"XS_CHECKSUM",
          ["-s","small","-g","1000","-l","175000000","-t","11"],
          strong=["-s","small","-g","1000","-l","175000000","-t","11"],
          weak={1: ["-s","small","-g","1000","-l","43750000","-t","12"],
                2: ["-s","small","-g","1000","-l","87500000","-t","11"],
                4: ["-s","small","-g","1000","-l","175000000","-t","11"]},
          ocr_base="xsbench_dist"),
    # Stencil2D_intel_chandra: npoints nranks ntimesteps. nranks=grid (2D split).
    # strong: npoints AND nranks held IDENTICAL at every node count (total work fixed,
    # L1-norm result decomposition-invariant) -- 48 subdomains divide evenly into
    # 1/2/4 nodes.  weak: npoints~sqrt(nodes) (area/node fixed).
    _bench_axes("Stencil2D_intel_chandra", r"L1 norm", ["20000","48","30"],
          strong=["20000","48","30"],
          weak={1: ["10000","12","30"], 2: ["14142","24","30"],
                4: ["20000","48","30"]}),
    # hpgmg: log2_box_dim target_boxes. Boxes auto-home box%node_count (Cat B).
    # strong: boxes fixed (32); weak: boxes grow with node count (8/16/32).
    # target_boxes is the fine-level instantaneous width (boxes snap DOWN to
    # the largest cube).  216 boxes @ 32^3 keeps the old total volume of
    # 27 @ 64^3 while clearing the concurrency floor (box dim >= 16 hard min).
    _bench_axes("hpgmg", r"\|\|error\|\|", ["5","216"],
          strong={1: ["5","216"], 2: ["5","216"], 4: ["5","216"]},
          weak={1: ["5","64"], 2: ["5","125"], 4: ["5","216"]}),
    # stream_dist: array_size num_threads ntimes (argvized 2026-07-06); the only
    # bandwidth kernel with OCR_HINT_EDT_AFFINITY (distributes across ranks).
    # strong: array AND thread count held IDENTICAL at every node count (total work
    # fixed, checksum decomposition-invariant) -- 48 threads divide evenly into
    # 1/2/4 nodes.  weak: array grows with threads.
    _bench_axes("stream_dist", r"STREAM checksum", ["8000000","48","1600"],
          strong=["32000000","48","400"],
          weak={1: ["8000000","12","400"], 2: ["16000000","24","400"],
                4: ["32000000","48","400"]}),
    # quicksort_dist: array_size range [buckets] [chunks]. Sample-splitter
    # p-way partition (buckets round-robin over ranks) + bucket-local sort;
    # quicksort (single-DB recursion) stays as the 1n contrast twin.
    # strong: N fixed at every geo; weak: N grows with node count.
    # nbuckets/nchunks (args 3/4) are the concurrency width: their defaults
    # (8/rank and a flat 48) sit under the concurrency floor, so the width is
    # passed explicitly -- 384 = 8x the 48-core box, per-phase.
    _bench_axes("quicksort_dist", r"QSORT_VALID",
          ["20000000","1000000","384","384"],
          strong={1: ["20000000","1000000","384","384"],
                  2: ["20000000","1000000","384","384"],
                  4: ["20000000","1000000","384","384"]},
          weak={1: ["20000000","1000000","384","384"],
                2: ["40000000","1000000","384","384"],
                4: ["80000000","1000000","384","384"]}),
    # fft_dist: log2(N) [tiles]. Bailey four-step transpose FFT, tile
    # decomposition: the tile count (arg 2) is the app's whole concurrency
    # expression, independent of the PD count (tune per machine, not per
    # rank); segments are per-(producer,consumer) one-shot DBs, so wiring
    # grows as tiles^2 -- keep tiles at a small multiple of the core count.
    # fft (recursive monolith) stays as the 1n contrast twin.
    # strong: N fixed at every geo; weak: log2N +1 per node doubling.
    _bench_axes("fft_dist", r"FFT_DIST checksum", ["27","256"],
          strong={1: ["27","256"], 2: ["27","256"], 4: ["27","256"]},
          weak={1: ["25","256"], 2: ["26","256"], 4: ["27","256"]}),
]


# ---------------------------------------------------------------------------
# Non-scaling apps: run only in the "single" experiment (whole OCR-pair suite
# at the 48-worker single-node geometry PLUS the 12-worker 1n_sc geometry -- a
# two-point within-node concurrency-scaling axis every app can run).  fix_args
# are calibrated (2026-07-06, post leak-fix) so the arts (ocr_val_wb) wall is
# ~3 s at 48 workers, with human-natural parameter values (powers of two /
# multiples of ten); worker/rank-grid parameters are pinned to 48 (or the
# nearest power of two the app demands).  Apps with no CLI workload knob run at
# their built-in size ("fixed") and their wall is reported as-is.  Every entry
# ---------------------------------------------------------------------------


_DERIVED_AXES: list = [
    # --- calibrated kernels (CLI workload knob; ~3 s at 48 workers) ---
    _bench("fibonacci", r"answer is\s*\d+", ["33"]),
    # No CLI board size yields ~3 s (15 -> 0.9 s, 16 -> 7 s); 16 is kept as the
    # meaningful full-workload point (its wall is reported as-is).
    _bench("nqueens", r"sols:\s*\d+", ["16", "8"]),
    # ~41 GB live tile matrix at this size — feasible on arts since the
    # pool-lifecycle fix; the references need far beyond the budget here.
    _bench("smithwaterman", r"score:\s*\d+",
            ["100", "100", f"{SW_DATA}/string1-large.txt",
             f"{SW_DATA}/string2-large.txt", f"{SW_DATA}/score-large.txt"]),
    _bench("fft", r"FFT checksum", ["23"]),
    # No-arg triangle solves the full puzzle — its maximum problem size
    # (~0.8 s on arts at 48 workers; the fine-grained EDT tree is the point).
    _bench("triangle", r"final count\s+\d+", []),
    _bench("p2p", r"PASS checksum", ["48", "100", "100", "1400"]),
    _bench("CoMD_sdsc", r"Final energy", ["-x","36","-y","36","-z","36","-N","2"]),
    _bench("CoMD_intel_chandra", r"Initial energy",
            ["-x","60","-y","60","-z","60","-N","6","-n","1"]),
    # The SPMD tile grid (-i/-j/-k) defaults to 1x1x1 = one rank pinned to
    # PD 0, so without it every multinode cell degenerates to a single node.
    # 2x2x1 divides evenly into the 1/2/4-node PD grids; -x/-y/-z stay global
    # (the grid cuts a fixed volume -- strong scaling built in).
    # The rank grid is the concurrency width (per-rank EDT chains are
    # sequential): 4x4x4 = 64 chains clears the floor at every geometry and
    # turned the flat 4-rank scaling into a real strong-scaling curve; the
    # volume is resized so the wall stays above the signal floor (per-axis
    # constraint: cells >= ~3.2x the axis's rank count).
    _bench("CoMD_intel_chandra_tiled", r"Final energy",
            ["-x","112","-y","112","-z","112","-N","8",
             "-i","4","-j","4","-k","4"]),
    # reduction-algorithm variants of the hpcg_intel scaling core; the MN matrix
    # is bounded to the core so the scaling story is one hpcg curve.
    _bench("hpcg_intel_Eager", r"final deviation", ["4","4","3","64","15"]),
    _bench("hpcg_intel_Eager_Collective", r"final deviation",
            ["4","4","3","64","15"]),
    # PROBLEM_TYPE=1 true 1-D (npoints, tiles, iterations).  A 1-D domain has
    # far fewer active points per grid dimension than the 2-D sibling, so it
    # needs a much larger npoints to reach the same wall (~10 s at 48 workers):
    # calibrated to npoints=1.2e9 with tiles/iterations unchanged.  The derived
    # weak axis grows iterations (index 2), which stays valid in 1-D.
    _bench("Stencil1D_intel_chandra", r"Solution validates",
            ["1200000000", "48", "30"]),
    _bench("Stencil2D_intel_channelEVTs", r"Computed L1 norm",
            ["20000", "48", "30"]),
    # --npx/npy/npz is the block grid = the concurrency width (default 1x1x1
    # = ONE block); --nx is cells PER BLOCK, resized down as the grid grows.
    # 8x8x4 = 256 concurrent block chains (floor-clearing).  --uniform_refine
    # is broken in this port (negative-size allocation at init) -- concurrency
    # comes from the base grid alone.
    _bench("miniAMR_intel", r"Grand Total Checksum",
            ["--nx","16","--ny","16","--nz","16","--npx","8","--npy","8",
             "--npz","4","--num_tsteps","6","--num_objects","1"]),
    # Cost is init-dominated (blocks of 16^3): timestep count barely moves the
    # wall; init_x/y/z set the PER-RANK decomposition.  The SPMD rank grid
    # (--npx/npy/npz) defaults to 1x1x1 = the whole run on PD 0, so without it
    # every multinode cell degenerates to a single node; 2x2x1 divides evenly
    # into the 1/2/4-node PD grids (per-rank init blocks make the rank grid
    # multiply total work).
    _bench("miniAMR_intel_chandra", r"Done",
            ["--nx","12","--ny","12","--nz","12","--init_x","2","--init_y","2",
             "--init_z","2","--num_tsteps","10","--num_refine","1",
             "--npx","4","--npy","2","--npz","1"]),
    # OCR port parses workload flags but they do not change the computed
    # problem (verified byte-identical output) — effectively fixed-size.
    _bench("miniAMR_intel_bryan", r"miniAMR complete", []),
    # class A CG (~1.5 s).  Class B is size-fixed at ~43 s @48w / >170 s @12w
    # (every runtime times out at 1n_sc), so it is not in the perf matrix.
    _bench("npb_cg", r"Verification SUCCESSFUL", ["-t","A","-b","25"]),
    _bench("tempest", r"CROSS-CHECKING NEIGHBOR DATA EXCHANGE", ["96"]),
    # Wall saturates ~3 s regardless of maxX/tolerance (subdivision breadth
    # bounded by numRanks) — effectively a fixed ~3 s kernel at 48 ranks.
    _bench("curvefit", r"SUCCESS", ["48","0.01","0.5","1000000"]),
    _bench("reduction_intel", r"T\d+ i1", ["48","2","20000"]),
    _bench("reduction_intel_chandra", r"RESULT = ", ["650000"]),
    _bench("LCS_distributed_ST", r"LCS length:", ["57344","1024","48"]),
    _bench("LCS_shared", r"LCS length:", ["40960","1024","48"]),
    _bench("LCS_all_db_distributed", r"LCS length:", ["65536","1024","48"]),
    _bench("RSBench_intel", r"Lookups:", ["-l","400000"]),
    # "Lookups/s" prints only in the RESULTS section; the init-phase grid
    # checksum must NOT be the completion marker (a slow/hung init would then
    # classify as a post-result hang).
    _bench("XSBench_intel", r"Lookups/s",
            ["-s","small","-g","1000","-l","400000"]),
    _bench("cholesky", r"CHOLESKY trace",
            ["--ds","5000","--ts","100","--fi",CHOLESKY_PERF_MAT]),
    _bench("cholesky_blas", r"CHOLESKY trace",
            ["--ds","7500","--ts","100","--fi",CHOLESKY_PERF_MAT_LG]),
    # The default workload (FANOUT=100 DEPTH=200, 20k DB churn) SIGSEGVs on
    # ALL THREE runtimes (vendor-app issue, not runtime-specific) — kept at
    # the correctness-scale workload.
    _bench("dbctrl", r"Total time", ["6","6","1024"]),
    # --- argvized bandwidth/sort kernels (single-node; no EDT affinity) ---
    _bench("quicksort", r"Sorting Finished", ["4000000","1000000"]),
    # 1500 iterations: the cross-node geometries anti-scale (shared-array
    # coherence traffic), sized so 2n/4n clear the cell budget with margin
    # (~50/60 s) while 1n stays above the signal floor (~3 s).
    _bench("stream", r"STREAM_RESULT", ["4000000","48","1500"]),
    _bench("highbw", r"HIGHBW_WORK_SUM", ["48","8388608","1000"]),
    _bench("prodcon", r"MB/s", ["48","8000000","6000"]),
    # --- fixed-size fixtures (no CLI workload knob; overhead floor) ---
    _bench("multigen", r"End leaf1, result", []),
    _bench("multigen_2", r"End leaf1, result", []),
    _bench("globalsum_cgShim", r"CG0 T\d+\s+0 value", []),
    _bench("globalsum_cgNoShim", r"CG0 T\d+\s+0 value", []),
    _bench("globalsum_pcg", r"CG0 T\d+\s+0 value", []),
    _bench("stencil1D_sticky", r"S3 i9 valu", ["48","50","340000"]),
    _bench("stencil1D_channel", r"S3 i9 valu", ["48","50","340000"]),
    _bench("stencil1D_guid", r"S3 i9 valu", ["48","50","340000"]),
    _bench("stencil1D_once", r"S3 i9 valu", ["48","50","340000"]),
    _bench("stencil1D_stickyLG", r"9 49\s", ["48","50","340000"]),
    _bench("stencil1D_guidPI", r"9 49\s", ["48","50","340000"]),
    _bench("stencil1D_oncePI", r"9 49\s", ["48","50","340000"]),
    # stream_org/stream_sa: rate-only output (no correctness scalar; the
    # checksum-bearing ports of the same kernel are stream/stream_dist),
    # compile-time array size — perf/e2e cells only.
    _bench("stream_org", r"MB/s", []),
    _bench("stream_sa", r"MB/s", []),
    # SAR: the runtime-input problem_size_scaling app subsumes the fixed
    # incbin sizes for perf (same pipeline, selectable image grid over the
    # shared huge pulse dataset); tiny..large stay in the correctness matrix.
    # Strong = P2 (1200^2): the largest ladder rung whose slowest coherence
    # arm (ocr_val_wt) still clears the 4n_sc cell budget with margin
    # (measured ~38 s vs ~113 s at the next rung); the larger rungs are
    # cluster-scale problems, not for this host.
    PerfCase("sar_pss", "sar_problem_size_scaling", r"SAR detects:",
             _sar_pss_args(2), None, None, {}),
    _bench("basicIO", r"BASICIO_CHK", ["0","10",BASIC_IO_DAT]),
    _bench("printf", r"Hello from mainEdt", []),
    _bench("testlibs", r"Testing strlen", []),
    _bench("task_priorities", r"Hello from 9", []),
    _bench("cache_offset", r"CACHE_OFFSET_CHK", []),
    _bench("xeonNumaSize", r"DONE!", ["-dcpu"]),
    _bench("dbcreate_matrix", r"CELLS_OK=", []),
    # Spawn-throughput stress.  Root-block count is --nx*--ny*--nz and each
    # block refines every 50 timesteps with NO population cap, so more than one
    # root makes the wall bimodal -- block 0's shutdown races an unbounded
    # refinement storm (empirically 2 s or 40 s+ for identical args) and can
    # exhaust host memory.  A single root has no neighbor to refine against, so
    # it is a stable, memory-bounded serial EDT-spawn chain whose length (hence
    # wall) is num_tsteps: ~9 s at 48 workers for 1.2e6 steps.  num_refine=2
    # keeps the per-timestep refine-control EDT in the spawn mix.
    _bench("miniAMR_forkbomb", r"BLOCK 0 finished",
            ["--nx","1","--ny","1","--nz","1","--num_tsteps","1200000",
             "--num_refine","2"]),
]


# ---------------------------------------------------------------------------
# Weak-scaling arg derivation for the non-core apps (user directive 2026-07-06).
# Scale each app's AUDIT work-axis so 1n_sc -> 2n_sc = 2x work, 4n_sc = 4x work
# (per-node work held constant, total grows with node count).  Linear knobs grow
# x2/x4; a log2 size knob (fft) or an exponential tree/recursion-depth knob
# (fib) grows +1/+2 (~2x/~4x); a quadratic-work size knob (LCS wavefront)
# grows x1.41/x2.  Values snap to a natural number (power of two, or k*10^e).
# Apps with NO 2x-scalable work knob are WEAK-EXEMPT (weak_args stays None ->
# skipped from weak, since their weak would just duplicate the strong fix_args
# broadcast).  selector = positional index into fix_args, or a flag whose
# FOLLOWING token holds the value.
_WEAK_AXIS = {
    "fibonacci": (0, "exp"), "fft": (0, "log2"),
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
    "dbctrl": (2, "lin"), "quicksort": (0, "lin"), "stream": (2, "lin"),
    "highbw": (2, "lin"), "prodcon": (2, "lin"),
}
# Weak-exempt (no ~2x-scalable work knob): app -> reason (for ledger/README).
_WEAK_EXEMPT = {
    # Empty: every registered bench carries a weak series (designed dict,
    # derived axis, or _EXPLICIT_WEAK).  A future entry must document why the
    # app cannot scale its work by any knob, argv-ization, or dataset ladder.
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


# Designed weak series for apps whose knob lives outside a simple fix_args
# axis: argv-ized repeat/iteration counts (exact 2x/4x), dataset ladders, or
# multi-flag workloads.  Keys are the harness bench names.
_EXPLICIT_WEAK = {
    "multigen": {1: ["32"], 2: ["64"], 4: ["128"]},
    "multigen_2": {1: ["64"], 2: ["128"], 4: ["256"]},
    "globalsum_cgShim": {1: ["100"], 2: ["200"], 4: ["400"]},
    "globalsum_cgNoShim": {1: ["100"], 2: ["200"], 4: ["400"]},
    "globalsum_pcg": {1: ["100"], 2: ["200"], 4: ["400"]},
    "cache_offset": {1: ["1000"], 2: ["2000"], 4: ["4000"]},
    "task_priorities": {1: ["30"], 2: ["60"], 4: ["120"]},
    "printf": {1: ["100000"], 2: ["200000"], 4: ["400000"]},
    "testlibs": {1: ["1000"], 2: ["2000"], 4: ["4000"]},
    "dbcreate_matrix": {1: ["1000"], 2: ["2000"], 4: ["4000"]},
    "nqueens": {1: ["16", "8", "1"], 2: ["16", "8", "2"], 4: ["16", "8", "4"]},
    "triangle": {1: ["13", "1"], 2: ["13", "2"], 4: ["13", "4"]},
    # niter is the linear iteration knob; zeta converges, so the standard
    # verification tolerance still passes at higher iteration counts.
    "npb_cg": {1: ["-t", "A", "-b", "25"],
               2: ["-t", "A", "-b", "25", "-i", "30"],
               4: ["-t", "A", "-b", "25", "-i", "60"]},
    # delta halving doubles per-node sampling cost; tree shape unchanged.
    "curvefit": {1: ["48", "0.01", "0.5", "1000000"],
                 2: ["48", "0.01", "0.25", "1000000"],
                 4: ["48", "0.01", "0.125", "1000000"]},
    "basicIO": {1: ["0", "250000", BASIC_IO_250K],
                2: ["0", "500000", BASIC_IO_500K],
                4: ["0", "1000000", BASIC_IO_1M]},
    "xeonNumaSize": {1: ["-drRLoops=1000", "-drSize=67108864"],
                     2: ["-drRLoops=2000", "-drSize=67108864"],
                     4: ["-drRLoops=4000", "-drSize=67108864"]},
    "smithwaterman": {
        1: ["100", "100", f"{SW_DATA}/string1-large.txt",
            f"{SW_DATA}/string2-large.txt", f"{SW_DATA}/score-large.txt"],
        2: ["100", "100", SW_XL[0], SW_XL[1], SW_XL[2]],
        4: ["100", "100", SW_XXL[0], SW_XXL[1], SW_XXL[2]]},
    # Single root at every node count (growing the root count reintroduces the
    # unbounded/bimodal storm -- see the fix_args note); per-node spawn work
    # grows 2x/4x via the serial timestep chain.
    "miniAMR_forkbomb": {
        1: ["--nx", "1", "--ny", "1", "--nz", "1", "--num_tsteps", "1200000", "--num_refine", "2"],
        2: ["--nx", "1", "--ny", "1", "--nz", "1", "--num_tsteps", "2400000", "--num_refine", "2"],
        4: ["--nx", "1", "--ny", "1", "--nz", "1", "--num_tsteps", "4800000", "--num_refine", "2"]},
    # --max_time is the live timestep bound in this build; --npx/npy/npz set
    # the block count (the parsed --num_tsteps/--nx flags are dead here).
    "miniAMR_intel_bryan": {
        1: ["--npx", "1", "--npy", "1", "--npz", "1", "--max_time", "250"],
        2: ["--npx", "2", "--npy", "1", "--npz", "1", "--max_time", "250"],
        4: ["--npx", "2", "--npy", "2", "--npz", "1", "--max_time", "250"]},
    # sar_pss weak: backprojection work ~ (p+1)^2 (pulse/sample counts fixed),
    # so (p+1)^2 tracks the node count from the strong base P2: 9 -> 16
    # (nearest integer rung to the ideal 18, -11%) -> 36 (exact).
    "sar_pss": {1: _sar_pss_args(2), 2: _sar_pss_args(3), 4: _sar_pss_args(5)},
}

for _c in _DERIVED_AXES:
    if _c.name in _WEAK_AXIS:
        _c.weak_args = _derive_weak(_c, *_WEAK_AXIS[_c.name])
    elif _c.name in _EXPLICIT_WEAK:
        _c.weak_args = _EXPLICIT_WEAK[_c.name]
    elif _c.name in ("stream_org", "stream_sa"):
        # NTIMES knob argv-ized in the app (default 1000 == the compiled value).
        _c.weak_args = {1: ["1000"], 2: ["2000"], 4: ["4000"]}
    elif _c.name in CHOLESKY_WEAK_MATS:
        _c.weak_args = {g: ["--ds", str(n), "--ts", "100", "--fi", path]
                        for g, (n, path) in CHOLESKY_WEAK_MATS[_c.name].items()}
    elif _c.name.startswith("stencil1D_"):
        # Timestep chain scales wall linearly and the weak-efficiency figure
        # assumes nominal 2x/4x work, so use exact doublings (snap-free).
        _t = int(_c.fix_args[2])
        _c.weak_args = {1: list(_c.fix_args),
                        2: _axis_set(_c.fix_args, 2, _t * 2),
                        4: _axis_set(_c.fix_args, 2, _t * 4)}


# Strong-scaling args for _single apps whose broadcast fix_args would run a
# trivial default problem.  Same argv at every node count (constant total work).
_EXPLICIT_STRONG = {
    # miniAMR_intel_bryan default fix_args ([]) is a 1-block problem -- no
    # strong-scaling signal.  --npx*--npy*--npz sets the root-block count and
    # --max_time the timestep bound.  The app anti-scales across nodes
    # (cross-node halo/refine comm dominates: the same 16-block problem is
    # ~0.5 s at 1n_sc but ~60 s at 4n_sc), so the size is bounded by the widest
    # geometry's wall budget, not 1n parallelism.  16 blocks x 800 timesteps
    # keeps 4n_sc well within the per-cell budget while still exercising real
    # multinode communication.
    "miniAMR_intel_bryan": {
        1: ["--npx", "4", "--npy", "2", "--npz", "2", "--max_time", "800"],
        2: ["--npx", "4", "--npy", "2", "--npz", "2", "--max_time", "800"],
        4: ["--npx", "4", "--npy", "2", "--npz", "2", "--max_time", "800"]},
}

for _c in _DERIVED_AXES:
    if _c.name in _EXPLICIT_STRONG:
        _c.strong_args = _EXPLICIT_STRONG[_c.name]


# Unified table: every app is one PerfCase.  There is NO "scalable vs
# single-only" distinction and NO per-app run gate: with the no-hint
# round-robin placement policy every app runs every experiment ("single" and
# "strong" run all of BENCHES, strong broadcasting fix_args where no designed
# strong table exists; "weak" runs every app that has a weak axis, the
# fixed-workload apps being exempt since their weak would duplicate strong).
# The ONLY perf-matrix exclusion is the user-curated PERF_EXCLUDED set below.
# The correctness harness is deliberately NOT curated: it runs everything.
BENCHES: list = _DESIGNED_AXES + _DERIVED_AXES

# User-curated perf-matrix exclusions (fixture micros, demos, redundant
# variants).  Excluded from every perf experiment and from the figures; the
# correctness harness still runs ALL of them -- correctness is never curated.
PERF_EXCLUDED: set = {
    "basicIO", "printf", "testlibs", "xeonNumaSize",
    "globalsum_cgShim", "globalsum_cgNoShim", "globalsum_pcg",
    "cache_offset", "dbctrl", "task_priorities", "multigen", "multigen_2",
    "highbw", "prodcon", "dbcreate_matrix",
    # 220-line OCR demo (adaptive piecewise-linear fit of a hardcoded curve):
    # no reference implementation, no suite provenance; the irregular-task-tree
    # class is represented by the classic task benchmarks instead.
    "curvefit",
    # stencil1D event-idiom micros (user cull): no affinity -> rank0 funnel,
    # single-node by construction; superseded for perf purposes by the true
    # 1-D Stencil1D_intel_chandra (PROBLEM_TYPE=1 build).  The correctness
    # harness keeps them as event-wiring functional tests.
    "stencil1D_channel", "stencil1D_guid", "stencil1D_guidPI",
    "stencil1D_once", "stencil1D_oncePI", "stencil1D_sticky",
    "stencil1D_stickyLG",
    # STREAM family Lean(2) cull: four variants = one kernel chain on a 2x2
    # (DB-lifetime x placement) grid; stream_dist (affinity, persistent) and
    # stream (dispersed, churn) cover the distinct corners.  org/sa remain
    # correctness regression guards (relerr scalar, ONCE-race, db-cache-free).
    "stream_org", "stream_sa",
    # LCS twin cull: shared and distributed_ST are the same linear-space
    # rolling-buffer algorithm (only the S/T string storage differs); shared
    # is the coherence-clean canonical and keeps the perf slot.  ST remains a
    # correctness regression guard (labeled-DB remote metadata-clone path).
    "LCS_distributed_ST",
    # CoMD family cull to {sdsc2, intel-chandra-tiled}: sdsc is the same SDSC
    # lineage superseded by the async sdsc2 (cost: EAM + sync-barrier idiom);
    # intel-chandra is strictly dominated by intel-chandra-tiled (full CoMD
    # 1.1, LJ+EAM, channel halo, clean on all nine runtimes).  Both remain
    # correctness functional tests.
    "CoMD_sdsc", "CoMD_intel_chandra",
    # hpcg reduction-algorithm trio cut to its two endpoints: base (app-managed
    # labeled-GUID/CHANNEL reduction tree; the perf strong+weak core) and
    # Eager_Collective (runtime-native COLLECTIVE_EVT allreduce).  _Eager is the
    # same tree topology as base with EAGER-DB buffer reuse only — a midpoint,
    # kept as a correctness functional test.
    "hpcg_intel_Eager",
    # superseded by their _dist rewrites (per-rank sub-DB decomposition + affinity
    # hints); the single-DB originals stay as correctness functional tests.
    "fft", "quicksort",
    # miniAMR cut to {intel, intel_chandra} (no-hint-lineage numeric-checksum
    # anchor + hinted SPMD exemplar with the only real load balancer): bryan is
    # dominated by same-lineage chandra (dummy compute, bool scalar, no LB,
    # comm-bound blowup at scale).  Remains a correctness functional test.
    "miniAMR_intel_bryan",
    # reduction drivers are library test harnesses (README: "test driver for
    # the reduction library"), i.e. collective microbenchmarks, not app
    # benchmarks; the reduction library itself is exercised in situ by hpcg.
    # Both stay as correctness functional tests.
    "reduction_intel", "reduction_intel_chandra",
    # placement analysis verdict: no movable heavy RW block (all-RO thin halos),
    # and the dominant costs (single hot broadcast DB read by ~1M tiles, serial
    # rank-0 spawn of ~1M EDTs, ~3M consume-once transient DBs) are structural
    # app properties out of hint reach; MN cells are wall-clock sinks.  Stays a
    # correctness functional test; a _dist rewrite would be a separate mission.
    "smithwaterman",
    # RSBench pair cut to sharedDB (user decision, data-flow grounds): the
    # per-nuclide-DB original materialises each table lookup as a 3-EDT chain
    # with up to ~1e3-dep terminal EDTs (~1e8 dep wirings, serial chunk
    # barriers) -> ~100x MN throughput collapse; the only MN path is a driver
    # rewrite strictly inferior to the existing sharedDB execution shape.
    # Remains a correctness functional test.
    "RSBench_intel",
    # XSBench pair: the upstream sharedDB form is single-PD (its thread EDTs
    # all pin to the caller PD), and generation-bracket table re-fetch makes
    # even a PD-spread form comm-bound at MN; XSBench_dist (per-node table
    # replica, true SPMD) holds the perf slot.  Remains a correctness
    # functional test in upstream-pristine form.
    "XSBench_intel_sharedDB",
    # The non-sharedDB intel form measures the same dead end from the other
    # side: no affinity hints, so round-robin EDT placement turns every table
    # lookup into a remote fetch against the single home-resident table --
    # uniform order-of-magnitude anti-scaling at MN with zero protocol
    # differentiation.  Remains a correctness functional test.
    "XSBench_intel",
    # fork-storm stress fixture, dropped after a calibration attempt: the
    # variant has no population cap (target_active/max_num_blocks parsed but
    # unused) and never coarsens, so a genuine multi-root storm is unbounded
    # and bimodal (OOM risk); the only reproducible config is a single-root
    # serial spawn chain, which is not a concurrent-storm benchmark.
    "miniAMR_forkbomb",
}


# ---------------------------------------------------------------------------
# Machine profiles & concurrency-floor validation.
# Full rules: docs/research/2026-07-11-edt-concurrency-parameter-rules.md
#
# The floor is about the INSTANTANEOUS width of the parallel section (EDTs
# simultaneously runnable between barriers), never about total EDT counts:
# a per-level BSP grid, a DAG wavefront, or the live-chain count — not the
# sum over the run.  Width is an APP parameter (OCR: the app expresses its
# own concurrency); the machine profile only says what value to tune it to.
# ---------------------------------------------------------------------------
MACHINES = {
    "cbgpu02":  {"cores_per_node": 12, "max_nodes": 4},
    # Future target — args seeded in the rules doc, calibrate at bring-up.
    "junction": {"cores_per_node": 64, "max_nodes": 32},
}
TASK_FLOOR_MULT = 4   # task-parallel: width >= 4x total cores


def _w_pos(i):
    return lambda a: int(a[i])


def _w_flag(f):
    return lambda a: int(a[a.index(f) + 1])


def _w_flags(*fs):
    def w(a):
        v = 1
        for f in fs:
            v *= int(a[a.index(f) + 1])
        return v
    return w


# name -> (model, width_of(strong-args) | None).  Models:
#   spmd    : persistent per-slot chains — width == cores_per_node*max_nodes
#             (0.8-1.5x tolerance: divisibility/oversub slack)
#   spmd_pn : per-NODE worker knob — value == the per-node worker budget
#   task    : work-stealing surplus — width >= TASK_FLOOR_MULT x total
#   bw      : bandwidth kernel — width == total EXACTLY; raising it multiplies
#             coherence objects (measured regression), never raise
# width None: not statically derivable from argv (volume/recursion/file
# driven) — validated empirically instead (counter audit), not checked here.
CASE_MODEL = {
    "graph500":            ("task", lambda a: int(a[2]) * int(a[3])),  # R*C per BFS level
    "CoMD_sdsc2":          ("task", None),                             # boxes from volume
    "hpcg_intel":          ("spmd", lambda a: int(a[0]) * int(a[1]) * int(a[2])),
    "hpcg_intel_Eager":    ("spmd", lambda a: int(a[0]) * int(a[1]) * int(a[2])),
    "hpcg_intel_Eager_Collective":
                           ("spmd", lambda a: int(a[0]) * int(a[1]) * int(a[2])),
    "nekbone":             ("spmd", lambda a: int(a[0]) * int(a[1]) * int(a[2])),
    "RSBench_intel_sharedDB": ("spmd", _w_flag("-t")),
    "XSBench_dist":        ("spmd_pn", _w_flag("-t")),
    "Stencil2D_intel_chandra": ("spmd", _w_pos(1)),
    "Stencil1D_intel_chandra": ("spmd", _w_pos(1)),
    "Stencil2D_intel_channelEVTs": ("spmd", _w_pos(1)),
    "hpgmg":               ("task", _w_pos(1)),                        # fine-level boxes
    "stream_dist":         ("bw", _w_pos(1)),
    "stream":              ("bw", _w_pos(1)),
    "quicksort_dist":      ("task", lambda a: min(int(a[2]), int(a[3]))),  # narrower phase
    "fft_dist":            ("task", _w_pos(1)),                        # tiles per phase
    "p2p":                 ("spmd", _w_pos(0)),
    "tempest":             ("task", lambda a: 6 * int(a[0]) * int(a[0])),  # self-throttling patches
    "CoMD_intel_chandra_tiled": ("spmd", _w_flags("-i", "-j", "-k")),
    "CoMD_intel_chandra":  ("task", None),
    "miniAMR_intel":       ("task", _w_flags("--npx", "--npy", "--npz")),
    "miniAMR_intel_chandra":
        ("spmd", lambda a: (_w_flags("--npx", "--npy", "--npz")(a)
                            * _w_flags("--init_x", "--init_y", "--init_z")(a))),
    "npb_cg":              ("task", None),   # width = na/blk, na from the class table
    "LCS_shared":          ("task", None),   # anti-diagonal wavefront, quadrant-serialized
    "LCS_all_db_distributed": ("task", None),
    "LCS_distributed_ST":  ("task", None),
    "cholesky":            ("task", lambda a: (int(a[a.index("--ds") + 1])
                                               // int(a[a.index("--ts") + 1])) ** 2 // 2),
    "cholesky_blas":       ("task", lambda a: (int(a[a.index("--ds") + 1])
                                               // int(a[a.index("--ts") + 1])) ** 2 // 2),
    "sar_pss":             ("task", None),   # (Ix/blk)^2 from the Parameter file
}


def check_concurrency_floors(machine: str = "cbgpu02", quiet: bool = False) -> list:
    """Warn for every active strong-axis case whose declared parallel-section
    width violates its model's floor on `machine`; returns the violations.
    Recursion-rich apps (width None) are covered by the counter audit, not
    this static check."""
    prof = MACHINES[machine]
    total = prof["cores_per_node"] * prof["max_nodes"]
    geo = prof["max_nodes"] if prof["max_nodes"] in (1, 2, 4) else 4
    bad = []
    for case in benches_for("strong"):
        model, wf = CASE_MODEL.get(case.name, ("task", None))
        if wf is None:
            continue
        try:
            w = wf([str(x) for x in select_perf_args(case, "strong", geo)])
        except (ValueError, IndexError):
            continue
        ok = ((model == "task" and w >= TASK_FLOOR_MULT * total) or
              (model == "spmd" and 0.8 * total <= w <= 1.5 * total) or
              (model == "spmd_pn" and prof["cores_per_node"] - 2 <= w
               <= prof["cores_per_node"]) or
              (model == "bw" and w == total))
        if not ok:
            bad.append((case.name, model, w, total))
    if bad and not quiet:
        for name, model, w, total in bad:
            print(f"[perf] CONCURRENCY-FLOOR WARNING: {name} ({model}) "
                  f"parallel-section width {w} vs {total} total cores on "
                  f"{machine}", file=sys.stderr)
    return bad


def benches_for(experiment: str, only=None) -> list:
    """Apps to run for an experiment.  The only exclusion is the user-curated
    PERF_EXCLUDED set (perf-matrix curation; correctness runs everything).
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
    table = [b for b in table if b.name not in PERF_EXCLUDED]
    return [b for b in table if b.name in only] if only else table


def stage_single_fixtures() -> None:
    """Input files the single-suite cases read.  The NxN diagonally-dominant
    SPD matrix gives cholesky a real decomposition workload (calibrated to
    ~3 s at 48 workers); basicIO's 10-u64 file matches the correctness harness
    fixture."""
    mats = {n: path for tab in CHOLESKY_WEAK_MATS.values() for _, (n, path) in tab.items()}
    for n, path in sorted(mats.items()):
        p = Path(path)
        if not p.exists():
            with open(p, "w") as f:
                for r in range(n):
                    row = ["1.0"] * n
                    row[r] = f"{n + 1}.0"
                    f.write(" ".join(row) + "\n")
    b = Path(BASIC_IO_DAT)
    if not b.exists():
        b.write_text("\n".join(str(i) for i in range(10)) + "\n")
    # basicIO weak inputs: exact line-count truncations of the vendor file
    # (the app requires size == file line count exactly).
    for path, lines in ((BASIC_IO_250K, 250000), (BASIC_IO_500K, 500000)):
        if not Path(path).exists():
            with open(BASIC_IO_1M) as src, open(path, "w") as dst:
                for _, line in zip(range(lines), src):
                    dst.write(line)
    # smithwaterman weak inputs: seeded random ACGT pairs at ~sqrt(2)x / 2x the
    # large pair's string length => ~2x / 4x the O(len1*len2) work.  Score
    # files start at 0 and are refreshed by the campaign bootstrap run; the
    # app's VERIFY is print-only, so a stale score never affects rc or the
    # "score:" marker line.
    import random
    for (s1, s2, sc), ln in ((SW_XL, 142000), (SW_XXL, 202000)):
        for i, pth in enumerate((s1, s2)):
            if not Path(pth).exists():
                rng = random.Random(ln + i)
                Path(pth).write_text(
                    "".join(rng.choice("ACGT") for _ in range(ln)) + "\n")
        if not Path(sc).exists():
            Path(sc).write_text("0\n")


# ---------------------------------------------------------------------------
# Hang-detection FSM + retry wrapper (Task 6).
# ---------------------------------------------------------------------------

# Geometries (ranks) the "_sc" scalability series exercises.  1n_sc is a
# single-rank direct exec (no launcher); 2/4/8n_sc use the same launcher as
# the corresponding capacity geometry (arts self-fork / mpirun) but point at
# the dedicated cbgpu02/{n}_sc.cfg pair (see configs/local/cbgpu02,
# configs/mpi/cbgpu02) instead of the capacity-matrix cfgs.
_SC_GEOS = {"1n_sc": 1, "2n_sc": 2, "4n_sc": 4}


def classify_run(rc: int, marker_seen: bool, proc_alive: bool,
                 wall: float = 0.0, budget: float = 0.0) -> str:
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
    if budget > 0 and wall >= 0.9 * budget:
        # Budget kill that outlived SIGTERM: the wall-timeout wrapper escalates
        # to SIGKILL (rc==-SIGKILL) when graceful shutdown itself hangs.  The
        # run consumed the whole budget without producing a result, so it is
        # the same "slower than the budget" measurement class as rc==124 —
        # retrying burns iters*retries*budget with no usable data.
        return "TIMEOUT"
    return "COMPUTE_FAIL"


def finalize_status(status: str, e2e_ns, wall: float = 0.0,
                    budget: float = 0.0) -> str:
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
    retry, then the cell stops) rather than retried like COMPUTE_FAIL.

    One demotion exception: a budget-killed run whose completion marker had
    already printed (apps whose marker line recurs per timestep match long
    before completion) is a "slower than the budget" measurement, not a
    transient — demote it to TIMEOUT, not COMPUTE_FAIL, or the retry path
    burns iters x retries x budget on a structurally-too-slow cell.

    An e2e captured by such a run is equally suspect: the [E2E] marker also
    prints during SIGTERM teardown, so a budget-killed run reports e2e ~=
    budget.  A genuine post-result hang's e2e is the (shorter) compute span
    -- an e2e that itself consumed >= 0.9x the budget IS the timeout, never a
    valid measurement."""
    if status in ("OK", "SHUTDOWN_HANG"):
        if e2e_ns is not None:
            if budget > 0 and (e2e_ns / 1e9) >= 0.9 * budget:
                return "TIMEOUT"
            return "OK"
        if budget > 0 and wall >= 0.9 * budget:
            return "TIMEOUT"
        return "COMPUTE_FAIL"
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

            status = classify_run(r.rc, marker_seen, proc_alive,
                                  wall=r.wall, budget=runner.timeout)
            if status == "TIMEOUT" and r.rc != 124:
                # Budget SIGKILL (TERM-surviving teardown): stray ranks may
                # outlive the direct child — reap by exe, same as rc==124.
                runner._reap_exe(exe_path)

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
            status = finalize_status(status, e2e_ns,
                                     wall=r.wall, budget=runner.timeout)

            if status != "OK" and r.log_path:
                # The per-(bench,runtime) log is reused by every iteration;
                # keep a copy of each failing attempt for post-hoc triage.
                try:
                    shutil.copyfile(r.log_path,
                                    f"{r.log_path}.{status}_iter{i}_try{attempt}")
                except OSError:
                    pass

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
                # Only a completed run's e2e is a measurement: the runtime also
                # prints an e2e marker during signal-driven teardown, so a
                # budget-killed run carries a bogus "wall-sized" value that
                # must not reach downstream aggregation.
                "e2e_ns": e2e_ns if status == "OK" else None,
                "rc": r.rc,
                "wall": round(r.wall, 3),
                "status": status if status != "COMPUTE_FAIL" else "FAIL",
                "scalar": parse_marker_scalar(r.stdout, case.completion_marker),
                "metrics_dir": metrics_dir,
                **extra,
            })
            break

    return results


# ---------------------------------------------------------------------------
# Eligibility + emitters (Task 7).
# ---------------------------------------------------------------------------

# WRF_VAL contract-ineligibility mirrors the correctness harness's per-case
# wrf_val_skip declarations (single source of truth): an app whose wiring relies
# on RW/CONST admission exclusion is outside WRF_VAL's DB-WRF contract, so its
# wrf_val_wt cells measure undefined behavior (wrong values, hangs, or crashes from
# clobbered carrier DBs) — never protocol performance.  Exact-name matches
# only: a differently-wired sibling bench (e.g. a _dist rewrite) is judged on
# its own wiring.
import contextlib as _contextlib
import io as _io
with _contextlib.redirect_stdout(_io.StringIO()):  # its import-time banner
    import correctness_harness as _correctness
WRF_RCU_INELIGIBLE: frozenset = frozenset(
    c.name for c in _correctness.CASES if c.wrf_val_skip)


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
    if rt.key == "wrf_val_wt" and case.name in WRF_RCU_INELIGIBLE:
        return False
    return True


def write_results_csv(rows: list, path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in RESULT_COLS})


# ---------------------------------------------------------------------------
# Post-hoc output validation (correctness triage, non-gating).
# ---------------------------------------------------------------------------

def _scalar_eq(a: str, b: str, rel_tol: float = 1e-4) -> bool:
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if fa == fb:
        return True
    denom = max(abs(fa), abs(fb))
    return denom > 0 and abs(fa - fb) / denom <= rel_tol


def write_validation(csv_path, out_txt, out_csv) -> str:
    """Cross-runtime scalar triage over a results.csv: per (bench, config),
    cluster the OK rows' marker-line scalars and flag minority values, plus
    list every non-OK cell — so a wrong answer, hang, or crash is at least
    LOGGED and attributable to its cell log, not silently averaged away.

    Verdicts per (bench, config, runtime):
      OK           scalar agrees with the modal cluster
      DIVERGENT    stable scalar that disagrees with the modal cluster
      UNSTABLE     the runtime's own iterations disagree (marker line is not a
                   pure result for this app, or a real nondeterminism) —
                   excluded from the cross-runtime vote
      NO_SCALAR    OK run but no number on the marker line
      TIMEOUT/FAIL propagated from status (any iteration)
    Non-gating: the campaign proceeds; this is the triage record."""
    by_cell = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["bench"], r["config"])
            by_cell.setdefault(key, {}).setdefault(r["runtime"], []).append(r)

    rows_out, bad_lines = [], []
    for (bench, cfg), by_rt in sorted(by_cell.items()):
        stable = {}  # runtime -> representative scalar
        verdicts = {}
        for rt, rows in sorted(by_rt.items()):
            statuses = {r["status"] for r in rows}
            if statuses - {"OK"}:
                verdicts[rt] = "/".join(sorted(statuses - {"OK"}))
                continue
            scalars = [r.get("scalar") or "" for r in rows]
            if not any(scalars):
                verdicts[rt] = "NO_SCALAR"
                continue
            rep = scalars[0]
            if all(_scalar_eq(rep, s) for s in scalars[1:]):
                stable[rt] = rep
            else:
                verdicts[rt] = "UNSTABLE"
        # modal cluster among stable runtimes
        clusters = []  # list of [representative, [runtimes]]
        for rt, val in stable.items():
            for c in clusters:
                if _scalar_eq(c[0], val):
                    c[1].append(rt)
                    break
            else:
                clusters.append([val, [rt]])
        clusters.sort(key=lambda c: -len(c[1]))
        consensus = clusters[0][0] if clusters else ""
        for rt, val in stable.items():
            verdicts[rt] = "OK" if _scalar_eq(consensus, val) else "DIVERGENT"
        for rt, v in sorted(verdicts.items()):
            rows_out.append({"bench": bench, "config": cfg, "runtime": rt,
                             "verdict": v, "scalar": stable.get(rt, ""),
                             "consensus": consensus})
            if v != "OK":
                bad_lines.append(f"  {bench} x {cfg} x {rt}: {v}"
                                 + (f" (scalar={stable.get(rt)} vs consensus="
                                    f"{consensus})" if v == "DIVERGENT" else ""))

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bench", "config", "runtime",
                                          "verdict", "scalar", "consensus"])
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    n_bad = len(bad_lines)
    summary = (f"validation: {len(rows_out)} cells, {n_bad} non-OK\n"
               + ("\n".join(bad_lines) + "\n" if bad_lines else ""))
    Path(out_txt).write_text(summary)
    return summary


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
               "rc", "wall", "status", "scalar", "kernel2_ns", "mteps"]


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
    p.add_argument("--build-dir", default="build_release_ocr_val_wb",
                   help="Build directory containing apps and configs")
    p.add_argument("--target", default="cbgpu02", choices=["cbgpu02"],
                   help="Machine geometry (drives config subdir + node counts, "
                        "and the concurrency-floor profile)")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated bench names to restrict to")
    p.add_argument("--node", type=str, default="",
                   help="Restrict to a single node-config, e.g. --node 1n")
    p.add_argument("--experiment", type=str, default="strong,weak",
                   help="Comma-separated experiments to run: strong,weak,single")
    p.add_argument("--runtimes", type=str, default="",
                   help="Comma-separated runtime keys to restrict to "
                        "(e.g. ocr_val_wb,xsocr,ocrvx); default all")
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
    p.add_argument("--fixargs-override", type=str, default="",
                   help="JSON file mapping bench name -> fix_args list; "
                        "replaces the table's single-experiment workload per "
                        "app (calibration artifact — strong/weak args are "
                        "untouched). Unknown names are an error.")
    args = p.parse_args()

    check_concurrency_floors(machine=args.target)

    if args.fixargs_override:
        ov = json.loads(Path(args.fixargs_override).read_text())
        by_name = {c.name: c for c in BENCHES}
        unknown = set(ov) - set(by_name)
        if unknown:
            p.error(f"--fixargs-override unknown benches: {sorted(unknown)}")
        for name, fa in ov.items():
            by_name[name].fix_args = [str(x) for x in fa]
        print(f"[perf] fix_args overridden for {len(ov)} benches "
              f"from {args.fixargs_override}", flush=True)

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
                            "status": it["status"], "scalar": it.get("scalar"),
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

    # Correctness triage over everything recorded in this results.csv
    # (including resumed rows): wrong answers / hangs / crashes are logged per
    # cell, never silently absorbed into the perf aggregates.
    summary = write_validation(csv_path, logdir / "validation.txt",
                               logdir / "validation.csv")
    print(f"[perf] {summary}", flush=True)


if __name__ == "__main__":
    main()
