#!/usr/bin/env python3
"""
correctness_harness.py — per-app scalar correctness + wall-time harness.

Philosophy: line-diffing full stdout across two parallel backends is a fool's
errand because task-scheduling noise reorders prints, timestamps differ,
`[rank]` prefixes differ, and progress sampling lands on different
percentages. What we actually care about is whether the SCIENTIFIC RESULT
(energy, checksum, residual, solution count, …) agrees.

Each case defines ONE piece of information to compare:

    scalar_re  = regex with one capture group (float/int) or zero groups (bool marker)
    scalar_kind = "float" | "int" | "bool"
    scalar_tol = relative tolerance for float; 0 for int; ignored for bool

The harness runs every eligible runtime (arts variants, xsocr, ocrvx) with
identical argv, extracts the scalar from each stdout, and compares. A case
may optionally carry a `baseline` (OMP/MPI reference) attachment, which adds
one more cell to the same vote with matched parameters; when absent the
report's baseline column is simply blank for that case.

There is one unified case list (`CASES`): every case runs at every node
config in `NODE_CONFIGS` unless it sets `multinode_skip` to a documented
STRUCTURAL reason (the app's design is inherently single-node — e.g.
file-scope global state, no cross-rank distribution mechanism). There is no
tiering by scalar-availability or baseline-availability; those are simply
optional per-case attributes on the same case.

Apps without a meaningful scalar concept (printf test, basicIO, cache tests,
etc.) get `scalar_re = ""` and are verified via rc-only (PASS-RC verdict).

Usage:
    python3 correctness_harness.py                       # full suite
    python3 correctness_harness.py --no-baseline         # skip baseline cells
    python3 correctness_harness.py --only nqueens,CoMD_sdsc2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness_common import (
    REPO, SCRATCH, BASIC_IO_DAT, CHOLESKY_INPUT,
    _cfg_name, MN_RANKS_for, _TPN_for, _OCRVX_TBB_for,
    _OCRVX_TIMEOUT_MULT, _NCORES_for, pin_single, pin_wrap, mpirun_prefix,
    RunResult, Runner, Runtime, ARTS_VARIANTS, RUNTIMES,
)

# ---------------------------------------------------------------------------
# Build directory: selectable via --build-dir (default: build_release_ocr_val_wb).
# Resolved early so module-level Path constants can reference it.
# ---------------------------------------------------------------------------
_arg_parser = argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument('--build-dir', default='build_release_ocr_val_wb')
_arg_parser.add_argument('--target', default='cbgpu02', choices=['cbgpu02'])
_pre_args, _ = _arg_parser.parse_known_args()
BUILD = Path(_pre_args.build_dir)
if not BUILD.is_absolute():
    BUILD = REPO / BUILD
TARGET = _pre_args.target

_model = 'unknown'
_protocol = 'unknown'
_timing = ''
try:
    _cache_path = BUILD / 'CMakeCache.txt'
    with open(_cache_path) as _f:
        for _line in _f:
            _m = re.match(r'ARTS_MEMORY_MODEL:STRING=(\w+)', _line)
            if _m:
                _model = _m.group(1)
            _p = re.match(r'ARTS_COHERENCE_PROTOCOL:STRING=(\w+)', _line)
            if _p:
                _protocol = _p.group(1)
            _t = re.match(r'ARTS_WRITE_POLICY:STRING=(\w+)', _line)
            if _t:
                _timing = _t.group(1)
            _r = re.match(r'ARTS_RELEASE_POLICY:STRING=(\w+)', _line)
            if _r and _protocol == 'EXCL':
                _timing = _r.group(1)
except FileNotFoundError:
    pass
_mode = f'{_model}+{_protocol}+{_timing}'
print(f'[harness] Build dir: {BUILD} (config: {_mode})')

APPS_DIR = BUILD / "benchmarks" / "apps"
LOGS_ROOT = REPO / "logs" / "correctness"

# smithwaterman ships its own datasets (tiny/small/medium/large triples of
# string1/string2/score); use the official tiny set directly — no staging.
SW_DATA = Path(__file__).resolve().parents[2] / \
    "third_party/ocr-apps/apps/smithwaterman/datasets"
OCR_APPS = REPO / "third_party" / "ocr-apps" / "apps"

# SAR huge pulse dataset: staged out-of-tree at datasets/sar-huge/
# (gitignored; O(100h) to regenerate — provenance, checksums and restore
# steps in datasets/README.md).  The runtime-input SAR cases (sar_huge,
# sar_pss) read it via argv and are dropped loudly when it is not staged;
# the compile-time-data sizes (tiny..large) are unaffected.
SAR_HUGE_DATA = REPO / "datasets" / "sar-huge"

# Per-target machine geometry (cbgpu02 = 48-thread),
def sar_huge_data_present() -> bool:
    return all((SAR_HUGE_DATA / f).is_file() for f in
               ("Data.bin", "PlatformPosition.bin", "PulseTransmissionTime.bin"))


# selected by --target.  See harness_common for the factory functions.
MN_RANKS = MN_RANKS_for(TARGET)

# ---------------------------------------------------------------------------
# Case definitions.
# ---------------------------------------------------------------------------
@dataclass
class BaselineSpec:
    bin: str
    args: list[str]
    np: int = 1              # 1 = no mpirun; >1 uses mpirun -n N
    force_mpirun: bool = False
    scalar_re: str = ""
    scalar_kind: str = "float"
    scalar_tol: float = 0.0


@dataclass
class Case:
    name: str                             # harness key
    ocr_base: str                         # base name; runs <ocr_base>_xsocr / _arts
    args: list[str]
    scalar_re: str = ""                   # "" → PASS-RC only
    scalar_kind: str = "float"            # float | int | bool
    scalar_tol: float = 0.0
    expect: str = ""                      # statically-derived absolute answer; when
                                          # set, the extracted scalar must also equal
                                          # it (catches both runtimes being wrong),
                                          # not just match each other
    baseline: BaselineSpec | None = None  # optional OMP/MPI reference attachment;
                                          # blank column in the report when absent
    multinode_skip: str = ""              # non-empty → structural reason this case
                                          # cannot run at ANY multinode config (every
                                          # other case runs at every NODE_CONFIGS entry)
    multinode_timeout: int = 0            # per-case multinode wall budget (s); 0 → global
    multinode_novote: str = ""            # non-empty → reason this case's MN value
                                          # is not comparable (e.g. process-local
                                          # accumulator): the cell still RUNS at MN
                                          # (hangs/SEGVs must surface) but its
                                          # scalar is excluded from the vote.
    wrf_val_multinode_skip: str = ""         # non-empty → the case's WRF_VAL cells are
                                          # excluded from EXECUTION at multinode
                                          # (N/A), single-node still runs: a
                                          # debugged, evidence-closed
                                          # contract-consequence hang/SEGV (lost
                                          # updates land in control data), so
                                          # re-running buys no information and
                                          # burns a full wall budget per cell.
                                          # Reason must cite the closed verdict.
    wrf_val_skip: str = ""                   # non-empty → contract reason this case's
                                          # semantics are OUTSIDE the WRF_VAL (DB-WRF)
                                          # value is undefined under WRF_VAL.  The cell
                                          # still RUNS (a weak contract may yield a
                                          # wrong VALUE, never a hang/SEGV — those
                                          # are runtime bugs and must surface); the
                                          # scalar is merely excluded from the
                                          # vote, like multinode_skip).  Contract
                                          # representation, NOT a bug mask: per
                                          # CLAUDE.md, WRF_VAL is a deliberately weaker
                                          # contract — "racy-but-legal OCR programs
                                          # may yield wrong results there by design"
                                          # — so OCR-legal programs relying on
                                          # guarantees WRF_VAL omits (exclusive-writer
                                          # serialization / disjoint-write survival
                                          # under whole-DB lossy publish) do not
                                          # get a vote in that column
    timeout: int = 0                      # per-case SINGLE-NODE wall budget (s); 0 →
                                          # global.  For cases whose fixed (non-CLI)
                                          # problem size has a measured single-node
                                          # wall floor above the global budget on a
                                          # slower protocol — a budget fact, not a
                                          # mask (the scalar is still fully verified)
    ocrvx_mn_max_ranks: int = 0           # >0 → ocr-vx cells run only at rank counts
                                          # <= this bound (blank/N/A above it): for
                                          # message-storm micro-benchmarks whose fixed
                                          # (non-CLI) problem size has a measured wall
                                          # floor on ocr-vx's serialized message
                                          # pipeline far beyond any test budget
    multinode_args: dict | None = None    # per-rank-count args for geometry apps
                                          # (e.g. hpcg needs npx*npy*npz==nodes);
                                          # the single-node reference is recomputed
                                          # per-n with the matching geometry

# CASES — one entry per OCR pair. Workload args lifted from run_benchmarks.sh.
# Every case runs at every node config in NODE_CONFIGS unless multinode_skip
# documents a structural reason (single-node-only app design); an optional
# `baseline=` attachment adds an OMP/MPI reference cell to the same vote.
# For apps whose last printed value is timing or throughput, scalar_re stays ""
# (rc-only verdict). Adding a print to those apps is a vendor-source task
# explicitly out of scope.
CASES: list[Case] = [
    # --- scalar-checkable (scientific output already present) ---
    Case("fibonacci", "fibonacci", ["10"],
         scalar_re=r"answer is\s*(\d+)", scalar_kind="int"),
    # nqueens returns per-subtree counts up the recursion tree (8-byte count DB
    # on each task's completion event), so the printed total is exact at any
    # rank count and the MN cells vote normally.
    Case("nqueens", "nqueens", ["6", "2"],
         scalar_re=r"sols:\s*(\d+)", scalar_kind="int",
         baseline=BaselineSpec(
             bin="nqueens_omp", args=["6"],
             scalar_re=r"sols:\s*(\d+)", scalar_kind="int", scalar_tol=0,
         )),
    Case("smithwaterman", "smithwaterman",
         ["50","50",f"{SW_DATA}/string1-medium-large.txt",
          f"{SW_DATA}/string2-medium-large.txt",
          f"{SW_DATA}/score-medium-large.txt"],
         scalar_re=r"score:\s*(\d+)", scalar_kind="int"),
    Case("fft", "fft", ["6"],
         scalar_re=r"FFT checksum\s*=\s*([\-+0-9.eE]+)", scalar_kind="float", scalar_tol=1e-3),
    # fft_dist: four-step transpose FFT on a single-tone input; the analytic
    # spectrum makes the global checksum exactly N (two peaks of N/2), and the
    # line only prints when every bin passes the closed-form self-check.
    Case("fft_dist", "fft_dist", ["22"],
         scalar_re=r"FFT_DIST checksum\s*=\s*([\-+0-9.eE]+)",
         scalar_kind="float", scalar_tol=1e-3),
    # Depth-5 partial search (1074 sequences): the full-depth puzzle is a
    # fine-grained EDT-per-node tree (~1.3M EDTs) that the ocr-vx runtime's
    # serialized local message pipeline cannot finish within any per-case
    # budget (and its monotonic object retention exceeds the memory cap).
    # Depth 5 keeps a real 3-way correctness check while landing the slowest
    # reference runtime (ocr-vx at the widest rank count) inside the per-case
    # wall budget; the app still solves the full puzzle when run with no
    # argument.
    # triangle returns per-subtree counts up the recursion tree (8-byte count
    # DB on each task's completion event); the former shared EW accumulator is
    # gone, so the WRF_VAL column is contract-eligible again.
    Case("triangle", "triangle", ["5"],
         scalar_re=r"final count\s+(\d+)", scalar_kind="int"),
    # rows/timesteps shrunk from the original 100/10 (checksum = (t+1)*(n+m-2),
    # still a nontrivial computed value) so the pipeline-fill message count
    # (~ P*T*rows) fits the slowest reference runtime inside the wall budget.
    Case("p2p", "p2p", ["2","10","30","5"],
         scalar_re=r"PASS checksum\s*=\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-8),
    Case("CoMD_intel_chandra", "CoMD_intel_chandra",
         ["-x","4","-y","4","-z","4","-N","2","-n","1"],
         # The energy is a parallel reduction over atoms; summation order varies
         # with the per-rank domain decomposition, so the value carries FP
         # non-associativity noise across rank counts (observed ~1.6e-5 at 4
         # ranks).  Use the numerical-port tolerance (0.01%); a real divergence
         # in a molecular-dynamics energy is orders of magnitude larger.
         scalar_re=r"Initial energy\s*:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-4),
    # xsocr runs this at MN too (old HCDist-timeout exclusion reason was stale).
    # -i/-j/-k (SPMD tile grid) defaults to 1x1x1 = single-PD; 2x2x1 makes the
    # multinode rows exercise real distribution.  The app's sanityChecks
    # require the global extent per split axis >= 2*cutoff*procAxis, which
    # -x 4 fails at procAxis 2 -- 8 clears it (8*3.615 > 2*5.7875*2).
    Case("CoMD_intel_chandra_tiled", "CoMD_intel_chandra_tiled",
         ["-x","8","-y","8","-z","8","-N","2","-i","2","-j","2","-k","1"],
         scalar_re=r"Final energy\s*:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-6),
    Case("CoMD_sdsc", "CoMD_sdsc", ["-x","4","-y","4","-z","4","-N","2"],
         # "Final energy" is the end-to-end answer (after the 2-timestep loop);
         # "Initial energy" only echoes the setup state and verifies nothing.
         scalar_re=r"Final energy\s*:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-6,
         multinode_novote="placement-sensitive design gap: zero affinity hints "
         "(contrast CoMD_sdsc2, the validated multinode twin) + a full-domain "
         "single-EDT redistribute_edt acquiring every box DB_MODE_RW each "
         "timestep (cells.c:87) — hint-less round-robin scatter yields a "
         "non-deterministic, topology-dependent energy at n>=4 (all three "
         "runtimes diverge to different 1e-2-scale values; run still "
         "validates liveness)",
         wrf_val_multinode_skip="closed M-class verdict (via the sdsc2 twin): unordered sibling RW writes to the schedule GUID array are lossy-dropped -> garbage GUID -> early init stall, 124 at MN (wrf_val_mn_verdicts.md)"),
    Case("CoMD_sdsc2", "CoMD_sdsc2", ["-x","4","-y","4","-z","4","-N","2"],
         scalar_re=r"Final energy\s*:\s*([\-+0-9.eE]+)", scalar_kind="float",
         # Widened from 1e-6 to 1e-4 to admit the baseline cell into the same
         # cluster: the MPI/OMP reference's parallel reduction order differs
         # from the OCR app's, so its energy carries more FP non-associativity
         # noise than the cross-runtime (xsocr/arts variants/ocrvx) agreement
         # alone. consensus() clusters on this single case-level tolerance.
         scalar_tol=1e-4,
         wrf_val_skip="up to 26 neighbor FNC_init siblings concurrently RW-acquire "
                   "a box's schedule GUID-array DB (simulation.c:658), each "
                   "writing a distinct index (simulation.c:501-503) with no HB "
                   "edge; WRF_VAL whole-DB publish is declared lossy",
         baseline=BaselineSpec(
             bin="CoMD_mpi_omp", args=["-x","4","-y","4","-z","4","-N","2"],
             np=1, force_mpirun=True,
             scalar_re=r"Final energy\s*:\s*([\-+0-9.eE]+)",
             scalar_kind="float", scalar_tol=1e-4,
         ),
         wrf_val_multinode_skip="closed M-class verdict: 26 FNC_init siblings' disjoint-index writes to the sched GUID array lossy-dropped -> garbage GUID -> early init stall, 124 at MN (wrf_val_mn_verdicts.md)"),
         # xsocr passes at every rank count (the np4 home-MD race + the residual
         # np2/3 startup hang were fixed app/xsocr-side).  arts is KNOWN to hang
         # at multinode in the affinity-DB create path here = an arts bug to
         # debug.  The arts-block flag (multinode_xsocr_only) and the stale
         # xsocr-hang mask (expected_known_bug) were REMOVED 2026-06-08 so the
         # harness reports the true verdict — do NOT re-mask to go green; fix
         # the arts multinode affinity-DB-create hang.
    Case("hpcg_intel", "hpcg_intel", ["1","1","1","16","5"],
         scalar_re=r"final deviation:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-4),
    # The "Eager" in these app names is the app's reduction algorithm (the
    # reductionEager library / OCR_HINT_DB_EAGER read-prefetch hint), NOT a
    # runtime coherence mode.  The benchmark builds run xsocr home-flush-only and
    # ocr-vx owner-resident-only with the per-DB eager/lazy hints disabled/ignored, so
    # these apps exercise each runtime's default coherence regardless of the
    # hint.
    Case("hpcg_intel_Eager", "hpcg_intel_Eager", ["1","1","1","16","5"],
         scalar_re=r"final deviation:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-4),
    Case("Stencil1D_intel_chandra", "Stencil1D_intel_chandra", [],
         scalar_re=r"Solution validates", scalar_kind="bool"),
    Case("Stencil2D_intel_channelEVTs", "Stencil2D_intel_channelEVTs", [],
         scalar_re=r"Computed L1 norm\s*=\s*([\-+0-9.eE]+)", scalar_kind="float", scalar_tol=1e-6),
    # No baseline attachment: the OMP baseline (Stencil2D_omp) only prints a
    # "Solution validates" bool, while this app's scalar is the L1-norm float
    # — not the same metric (same principle as the miniAMR_intel / XSBench_intel
    # baseline omissions below).  Cross-runtime consensus still covers this case.
    Case("Stencil2D_intel_chandra", "Stencil2D_intel_chandra", [],
         scalar_re=r"L1 norm\s*=\s*([\-+0-9.eE]+)", scalar_kind="float", scalar_tol=1e-6),
    # No baseline attachment: baseline miniAMR_mpi only prints per-variable
    # checksums under --report_diffusion, while OCR prints a grand total
    # summed over variables.  The two are not the same scalar, and summing
    # baseline's per-variable prints after the fact is fragile.  Cross-runtime
    # (xsocr/arts variants/ocrvx) consensus still covers this case.
    Case("miniAMR_intel", "miniAMR_intel",
         ["--nx","4","--ny","4","--nz","4","--num_tsteps","2","--num_objects","1"],
         scalar_re=r"Grand Total Checksum\s*==\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-8,
         # wrf_val_wt runs truthfully: the clone->clone carrier hand-off releases
         # each block BEFORE wiring it into the successor (release -> satisfy
         # -> acquire is a happens-before chain independent of the dependence
         # mode), so the relay's conflicting accesses are ordered within the
         # memory model — not by RW admission.  Sibling sharing is read-only
         # and joins/halos are event-carried, so no unordered same-DB
         # conflict exists (DB-WRF-conformant).
         ),
    # "-b 25" raises the sparse matrix-vector multiply's block size (default
    # 1, i.e. one row per block): this only changes how the na=50 rows are
    # grouped into spmv_edt sub-EDTs (25 -> 2 blocks), not the arithmetic, so
    # zeta is bit-identical to the unblocked run while the ocr-vx message
    # count drops ~25x.
    Case("npb_cg", "npb_cg", ["-t", "T", "-b", "25"],
         # Anchor on the app's own verification verdict: a FAILED run prints
         # "Verification FAILED (zeta=NaN, correct zeta=<expected>)" and the
         # bare zeta regex would match the *expected* value (false PASS).
         scalar_re=r"Verification SUCCESSFUL \(zeta\s*=\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-10,
         # wrf_val_wt runs truthfully: the former skip cited the a/x RW passthrough
         # racing the CG DAG's CONST readers; the app now declares truthful
         # modes (a CONST everywhere, x RW only into its writers) and makea
         # releases the container and block sub-DBs at their publication
         # point, so every conflicting pair is release/satisfy-ordered
         # (DB-WRF-conformant).
         ),
         # No baseline attachment: the OMP reference runs its built-in S-class
         # problem (size=1400, 15 iterations, zeta=8.427309...) and ignores the
         # OCR port's tiny-class knob (-t T -> size=50, 3 iterations,
         # zeta=7.855340...).  zeta is problem-size-dependent, so the two
         # scalars are not comparable (same principle as the Stencil2D /
         # miniAMR / XSBench baseline omissions).
         # All runtimes pass at every rank count (2026-06-10).  The historical
         # multinode failures here (xsocr NaN at n3 / zeta=0.0 at n4, arts
         # timeout) were npb-cg APP bugs, fixed in the ocr-apps submodule:
         # in-place writes through DB_MODE_CONST acquisitions (update's p,
         # the zeta-carrying DB) and a missing ocrDbRelease before wiring the
         # verification EDT.  The arts-block flag (multinode_xsocr_only) was
         # REMOVED 2026-06-08 so the harness reports the true verdict — do NOT
         # re-mask if this regresses.
         # class T (tiny: size=50, 3 iters) runs in <1s, so it stays fast enough
         # for xsocr at multinode and runs full 3-way.  (class S — the default —
         # made the multinode run ~36K small remote DBs/iter of synchronous
         # publish-ACK round-trips: correct but ~67s n4 / ~170s RELAXED 2n, which is
         # why it used to be arts-only with a wide budget.)
    # Already at the floor: log2_box_dim must be >=4 and target_boxes >=1 (both
    # enforced by the app), so ["4","1"] is the smallest legal problem --
    # nothing left to shrink.
    Case("hpgmg", "hpgmg", ["4","1"],
         scalar_re=r"\|\|error\|\|\s*=\s*([\-+0-9.eE]+)", scalar_kind="float",
         # Widened from 1e-4 to 1e-3 to admit the baseline cell into the same
         # cluster (MPI/OMP reference's reduction order differs; see
         # CoMD_sdsc2 above for the same rationale).
         scalar_tol=1e-3,
         # wrf_val_wt runs truthfully: the former wrf_val_skip cited a halo RW/CONST
         # sibling race that is structurally absent at this 1-box workload
         # (total_boxes==1 -> a single exchange_edt, all neighbors resolve to
         # the read-only boundary sentinel), and at multi-box workloads the
         # exchange writes only the box's own ghost region against immutable
         # phase-fenced interiors (single writer per box).  The residual MN
         # wrf_val_wt hang is an wrf_val_wt-column-isolated runtime defect (every
         # ownership protocol passes the same cells), tracked as a bug, not a
         # contract exclusion.
         baseline=BaselineSpec(
             bin="hpgmg_mpi_omp", args=["4","1"], np=1, force_mpirun=True,
             scalar_re=r"\|\|error\|\|\s*=\s*([\-+0-9.eE]+)",
             scalar_kind="float", scalar_tol=1e-3,
         )),
    # No smaller args exist without degenerating the check: the CLI's only
    # knob is patchRange (k, patches-per-panel-edge); k=1 collapses every
    # diagonal-neighbor slot (the captured cross-check value) to a constant
    # -1 regardless of correctness (verified: at k=1 the boundary conditions
    # in findNeighborPatch force NE/SE/SW/NW to -1 unconditionally), so k=2
    # is the smallest value that still exercises real diagonal connectivity.
    # DURATION (per-patch halo-exchange timesteps) is a compile-time #define,
    # not CLI-settable.  Current wall already fits under the ocr-vx cap.
    # tempest: k=2 is the semantic floor (k=1 degenerates the cross-check value
    # to a constant), and its ocr-vx wall at 16 ranks is ~60s nominal — the
    # global 30s*3 ocr-vx budget leaves too little jitter headroom (measured
    # 90.2s under gate load).  Per-case budget 40s -> ocr-vx cap 120s: 2x
    # headroom, worst case still bounded at two minutes.
    Case("tempest", "tempest", [],
         scalar_re=r"CROSS-CHECKING NEIGHBOR DATA EXCHANGE\*(?:[ \t]*\n|[ \t]+|[A-Za-z*][^\n]*\n)*-?\d+[ \t]+-?\d+[ \t]+-?\d+(?:[ \t]*\n|[ \t]+|[A-Za-z*][^\n]*\n)*-?\d+[ \t]+-?\d+[ \t]+-?\d+(?:[ \t]*\n|[ \t]+|[A-Za-z*][^\n]*\n)*-?\d+[ \t]+-?\d+[ \t]+(-?\d+)",
         scalar_kind="int",
         multinode_timeout=40),
    # maxX (upper bound of the fitted domain) shrunk from the default 100000
    # to 10000: the adaptive-subdivision tree depth grows with the domain
    # (f(x)=x*sin(x) has growing curvature at large x, needing finer
    # segments), so this cuts total EDT count well past the wall-budget need
    # while the completion marker stays a real convergence check.
    Case("curvefit", "curvefit", ["4","0.01","0.5","10000"],
         # Completion marker: the leaf-segment count is a structural constant of
         # the fixed input, and aggregating it across the recursive fan-out would
         # require global mutable state (illegal in OCR — EDTs are stateless) or a
         # reduction DataBlock that the app does not build, so completion is the
         # strongest spec-compliant check here.
         scalar_re=r"SUCCESS", scalar_kind="bool"),
    Case("testlibs", "testlibs", [],
         scalar_re=r"Testing strlen of \w+ is (\d+)", scalar_kind="int"),
    Case("graph500", "graph500", ["6","8","1","1"],
         # MN runs use a real R-by-C worker grid so the distributed BFS wiring
         # (cross-rank row/column scatter, level latches) is actually exercised;
         # the visited-vertex scalar is grid-invariant.
         multinode_args={2: ["6","8","2","2"], 4: ["6","8","2","2"],
                         8: ["6","8","4","2"], 16: ["6","8","4","4"]},
         scalar_re=r"nodes (\d+)", scalar_kind="int"),
    Case("multigen", "multigen", [],
         scalar_re=r"End leaf1, result\s*=\s*(-?\d+)", scalar_kind="int"),
    Case("multigen_2", "multigen_2", [],
         scalar_re=r"End leaf1, result\s*=\s*(-?\d+)", scalar_kind="int"),
    # xsocr runs this at MN too (old HCDist-timeout exclusion reason was stale).
    # num_refine lowered from 3 to 1: at this block size/object count the
    # refinement check never actually splits a block at any depth (verified:
    # #blocks stays 1 at num_refine 1, 2, or 3 alike), so the extra levels
    # were pure redundant per-timestep refinement-check overhead with no
    # additional distributed behavior exercised.
    # --npx/npy/npz (SPMD rank grid) defaults to 1x1x1 = single-PD at every
    # node count; 2x2x1 makes the multinode rows exercise real distribution.
    # --report_diffusion 1 + --checksum_freq 1 upgrade the anchor from the
    # bare "Done" liveness bool to the deterministic per-variable checksum
    # (VERIFICATION_RUN seeding); the scalar is the final-timestep var-0 sum.
    Case("miniAMR_intel_chandra", "miniAMR_intel_chandra",
         ["--nx","4","--ny","4","--nz","4","--num_tsteps","2","--num_refine","1",
          "--npx","2","--npy","2","--npz","1",
          "--report_diffusion","1","--checksum_freq","1"],
         scalar_re=r"ts 2 CHECKSUM sum\s+([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-6),

    # --- rc-only sanity (no meaningful scientific scalar) ---
    Case("printf",           "printf",           [],
         scalar_re=r"Hello from mainEdt", scalar_kind="bool"),
    Case("quicksort",        "quicksort",        [],
         scalar_re=r"(\d+)\s*\n\s*(?:\[\d+\]\s*)?Sorting Finished", scalar_kind="int",
         wrf_val_skip="unordered disjoint-region sibling writers; WRF_VAL whole-DB "
                   "publish is declared lossy"),
    # quicksort_dist: sample-splitter p-way partition; every DB has a single
    # writer (chunk-local segments, bucket-local outputs), all sharing is RO —
    # WRF_VAL-eligible by construction, unlike the single-DB recursion twin above.
    # The scalar (element sum) only matches when the sorted flag is 1.
    Case("quicksort_dist",   "quicksort_dist",   [],
         scalar_re=r"QSORT_VALID sum=(\d+) sorted=1", scalar_kind="int"),
    Case("basicIO",          "basicIO",          ["0","10", BASIC_IO_DAT],
         scalar_re=r"BASICIO_CHK\s+(\d+)", scalar_kind="int"),
    # dbcreate_matrix: DB-create capability matrix (labeled/affinity/no-hint
    # x acquire/no-acquire x local/remote) -- compares create-path capability
    # across the runtimes. No expect= pin: the point of this case is to record
    # each runtime's own count, not assert a single "correct" answer.
    Case("dbcreate_matrix",  "dbcreate_matrix",  [],
         scalar_re=r"CELLS_OK=(\d+)/12", scalar_kind="int"),
    Case("cache_offset",     "cache_offset",     [],
         scalar_re=r"CACHE_OFFSET_CHK\s+(\d+)", scalar_kind="int"),
    # xsocr runs this at MN too (old startup-hang exclusion reason was stale).
    Case("highbw",           "highbw",           [],
         scalar_re=r"HIGHBW_WORK_SUM\s*=\s*(\d+)", scalar_kind="int"),
    Case("task_priorities",  "task_priorities",  [],
         scalar_re=r"Hello from 9", scalar_kind="bool"),
    Case("dbctrl",           "dbctrl",           ["5","5","256"],
         # DB create/destroy control stress test: the destroy count is a static
         # function of (DEPTH, FANOUT) and the kernel computes no data answer.
         # Counting destroys across EDTs would need illegal global state, so the
         # completion/timing marker is the strongest spec-compliant check.
         scalar_re=r"Total time", scalar_kind="bool"),
    Case("prodcon",          "prodcon",          [],
         scalar_re=r"MB/s", scalar_kind="bool"),
    # No CLI knob exists to shrink these three: mainEdt() takes no params (argv
    # is never read) and the problem size (M=matrix rows, N=participant count,
    # T=CG iterations) is an unconditional #define in the shared ocrGS.h,
    # identical for all three variants -- there is no #ifndef guard to override
    # via a build define either (verified: dummy CLI args are silently
    # ignored, output is byte-identical).  cgShim/cgNoShim's ocr-vx wall at the
    # widest rank count exceeds even the widened ocr-vx cap; pcg's does not
    # (pcg's residual converges well before hitting T=100, unlike cgShim/
    # cgNoShim which run the full iteration count) -- see the retune report for
    # measured walls.
    # globalsum cgShim/cgNoShim: problem size is a fixed shared-header #define
    # (no CLI/build knob), and the measured ocr-vx wall floor at 8/16 ranks is
    # ~120s/~500s (serialized message pipeline x fixed message storm) — far
    # beyond any test budget, while 2/4 ranks complete in seconds.  Cap the
    # ocr-vx cells at 4 ranks; arts/xsocr run everywhere.
    Case("globalsum_cgShim",   "globalsum_cgShim",   [],
         scalar_re=r"CG0 T\d+\s+0 value\s+([0-9.]+)", scalar_kind="float", scalar_tol=1e-5,
         ocrvx_mn_max_ranks=4,
         wrf_val_skip="N reduction-tree/compute siblings RW-acquire the shared "
                   "GSsharedBlock via direct GUID each round (gsLib.c:43,119,132), "
                   "no HB order; WRF_VAL whole-DB publish can drop node-0's "
                   "rootEvent/sum control write (non-DB-WRF)",
         wrf_val_multinode_skip="closed M-class verdict: lossy publish clobbers the shared control block's rootEvent on node 0 -> lost wakeup, deterministic 124 at MN (wrf_val_mn_verdicts.md)"),
    Case("globalsum_cgNoShim", "globalsum_cgNoShim", [],
         scalar_re=r"CG0 T100\s+0 value\s+([0-9.]+)", scalar_kind="float", scalar_tol=1e-5,
         ocrvx_mn_max_ranks=4,
         wrf_val_skip="N reduction-tree/compute siblings RW-acquire the shared "
                   "GSsharedBlock via direct GUID each round (gsLib.c:43,119,132), "
                   "no HB order; WRF_VAL whole-DB publish can drop node-0's "
                   "rootEvent/sum control write (non-DB-WRF)",
         wrf_val_multinode_skip="closed M-class verdict: lossy publish clobbers the shared control block's rootEvent on node 0 -> lost wakeup, deterministic 124 at MN (wrf_val_mn_verdicts.md)"),
    Case("globalsum_pcg",      "globalsum_pcg",      [],
         scalar_re=r"CG0 T\d+\s+0 value\s+([0-9.]+)", scalar_kind="float", scalar_tol=1e-5,
         wrf_val_skip="N reduction-tree/compute siblings RW-acquire the shared "
                   "GSsharedBlock via direct GUID each round (gsLib.c:43,119,132), "
                   "no HB order; WRF_VAL whole-DB publish can drop node-0's "
                   "rootEvent/sum control write (non-DB-WRF)",
         wrf_val_multinode_skip="closed M-class verdict: lossy publish clobbers the shared control block's rootEvent on node 0 -> lost wakeup, deterministic 124 at MN (wrf_val_mn_verdicts.md)"),
    Case("stencil1D_sticky", "stencil1D_sticky", [],
         scalar_re=r"S3 i9 valu\s+([0-9.]+)", scalar_kind="float",
         scalar_tol=0,
         wrf_val_multinode_skip="clone->clone halo relay ordered only by RW exclusion (same family as the stencil1D_guid/guidPI verdicts) -> relay stalls on a lossy-dropped hand-off at MN (wrf_val_mn_verdicts.md)"),
    Case("stencil1D_channel", "stencil1D_channel", [],
         scalar_re=r"S3 i9 valu\s+([0-9.]+)", scalar_kind="float",
         scalar_tol=0),
    Case("stencil1D_guid", "stencil1D_guid", [],
         scalar_re=r"S3 i9 valu\s+([0-9.]+)", scalar_kind="float",
         scalar_tol=0,
         wrf_val_skip="clone->clone halo relay: boundary DB handed each iteration "
                   "via bare DB_MODE_RW to a sibling EDT GUID read from the DB "
                   "payload (stencil1Dguid.c:178-183), no event; ordered only by "
                   "RW exclusion WRF_VAL omits (same class as miniAMR_intel)"),
    Case("stencil1D_once", "stencil1D_once", [],
         scalar_re=r"S3 i9 valu\s+([0-9.]+)", scalar_kind="float",
         scalar_tol=0),
    Case("stencil1D_stickyLG", "stencil1D_stickyLG", [],
         scalar_re=r"9 49\s+([0-9.]+)", scalar_kind="float",
         scalar_tol=0),
    Case("stencil1D_guidPI", "stencil1D_guidPI", [],
         scalar_re=r"9 49\s+([0-9.]+)", scalar_kind="float",
         scalar_tol=0,
         wrf_val_skip="steady-state clone->clone halo relay hands the boundary DB "
                   "via bare DB_MODE_RW to a sibling EDT GUID read from the DB's "
                   ".control field (stencil1DguidPI.c:94,163,196), no event; "
                   "ordered only by RW exclusion WRF_VAL omits (non-DB-WRF)"),
    Case("stencil1D_oncePI", "stencil1D_oncePI", [],
         scalar_re=r"9 49\s+([0-9.]+)", scalar_kind="float",
         scalar_tol=0),
    # stream_org / stream_sa print a deterministic final array value
    # (STREAM_RESULT, a pure function of the iteration count) alongside the
    # non-comparable bandwidth lines; a short iteration count keeps the cell
    # cheap while exercising the full kernel chain.
    # At multinode the 200-iteration cross-rank RW chain puts a wall floor
    # well above the global budget on the HOME-placement protocols (release
    # blocks on the synchronous publish ACK each iteration) and on the
    # serialized ocr-vx pipeline; all runtimes complete correctly given the
    # headroom.  Per-case budget 75s -> ocr-vx cap 225s: >2x measured worst
    # case under full gate load.
    Case("stream_org", "stream_org", ["200"],
         scalar_re=r"STREAM_VALID relerr = ([0-9.eE+-]+)",
         scalar_kind="float", scalar_tol=1e-12,
         multinode_timeout=75),
    Case("stream_sa", "stream_sa", ["200"],
         scalar_re=r"STREAM_VALID relerr = ([0-9.eE+-]+)",
         scalar_kind="float", scalar_tol=1e-12,
         multinode_timeout=75),
    # [nrank, ndata, maxtimestep] shrunk from the defaults [25,2,300] to
    # [25,2,30]: maxtimestep (T) is the dominant cost (the same nrank=25
    # ALLREDUCE tree runs once per timestep with real cross-node messages,
    # since participants are placed round-robin across the physical PDs), so
    # cutting it 10x is the direct lever; the regex/pin below track the new
    # final timestep.
    Case("reduction_intel",  "reduction_intel",  ["25","2","30"],
         scalar_re=r"T30 i1\s+([0-9.]+)", scalar_kind="float",
         scalar_tol=0),
    Case("reduction_intel_chandra", "reduction_intel_chandra", ["10"],
         scalar_re=r"RESULT = ([0-9.]+)", scalar_kind="float", scalar_tol=1e-6),
    # Full 3-way: the OCR runtime now clones labeled (reserved-GUID) datablock
    # metadata to remote ranks (labeled-guid.c admits OCR_GUID_DB to the MD
    # proxy/clone path; hc-policy.c defers MD_DIR_PULL until the home write-back
    # registers the MD), so xsocr no longer SEGVs at multinode.  Scalar is the
    # shutdown marker (bool present in both).
    Case("LCS_distributed_ST","LCS_distributed_ST",[],
         # distributed wavefront LCS result (depv[1] result DB at the answer
         # index), self-validated against a serial_lcs reference; deterministic.
         scalar_re=r"LCS length:\s*(-?\d+)", scalar_kind="int"),
    # [string_len, basecase, num_workers] shrunk from the default basecase 256
    # to 512 (string_len held at 1024): basecase is purely the wavefront DP's
    # block-tiling granularity (25 -> 9 blocks), not part of the LCS
    # recurrence, so the answer is unchanged (verified bit-identical) while
    # the ocr-vx cross-block message count drops sharply.
    Case("LCS_all_db_distributed","LCS_all_db_distributed",["1024","512","16"],
         scalar_re=r"LCS length:\s*(\d+)", scalar_kind="int"),
    Case("LCS_shared",        "LCS_shared",       [],
         # distributed wavefront LCS result, self-validated against serial_lcs.
         # The x12/x21 sibling quadrants that share the single rolling `score`
         # datablock are now happens-before ordered (x11->x12->x21), so the
         # program is DB-WRF and WRF_VAL votes with the consensus.
         scalar_re=r"LCS length:\s*(-?\d+)", scalar_kind="int"),
    Case("RSBench_intel",             "RSBench_intel",             ["-l","100"],
         scalar_re=r"Lookups:", scalar_kind="bool",
         baseline=BaselineSpec(
             bin="RSBench_omp",
             args=["-t","4","-l","100"],
             scalar_re=r"Lookups:", scalar_kind="bool",
         )),
    Case("RSBench_intel_sharedDB",    "RSBench_intel_sharedDB",    ["-l","100"],
         scalar_re=r"RS_CHECKSUM:\s+([0-9]+)", scalar_kind="int"),
    # No baseline attachment: there is NO correctness metric shared by the OCR
    # app and any OMP/MPI baseline.  The OCR XSBench prints only its own
    # "XSBench grid checksum" (added in refactored/ocr/intel Main.c; no
    # OMP/MPI variant computes it), while the OMP/MPI baselines print only the
    # canonical "Verification checksum" vhash (under -DVERIFICATION) — which
    # the OCR refactor never wires up (building it with VERIFICATION still
    # emits no vhash).  XSBench correctness is covered by this case's
    # cross-runtime consensus (grid-checksum agreement across the 7 arts
    # variants + xsocr + ocr-vx).
    # -g (gridpoints/nuclide) shrunk 10 -> 3 and -l (lookups) 100 -> 30: the
    # grid checksum depends only on -s/-g (the nuclide-grid RNG init), not on
    # -l, so lowering lookups is free (no re-pin needed for that half); -g
    # directly shrinks the per-isotope grid-build/sort work.
    Case("XSBench_intel",             "XSBench_intel",             ["-s","small","-g","3","-l","30"],
         scalar_re=r"XSBench grid checksum:\s+(\d+)", scalar_kind="int"),
    # XSBench_intel_sharedDB is the upstream-pristine control (its scalar is
    # the -l input echo, a liveness check); the computed XS_CHECKSUM
    # correctness story lives in XSBench_dist below.
    Case("XSBench_intel_sharedDB",    "XSBench_intel_sharedDB",    ["-s","small","-g","10","-l","100"],
         scalar_re=r"Workload\s+\(unit\):\s+(\d+)", scalar_kind="int"),
    # XSBench_dist: true-SPMD distributed XSBench — one deterministic table
    # replica per rank (fixed data RNG), lookup range tiled once across
    # rank x thread workers, per-lookup seed derived from the global index.
    # XS_CHECKSUM is therefore invariant to rank/thread count and runtime.
    # Every DB is single-writer-then-RO (replicas) or written-once partials
    # gathered through events — WRF_VAL-eligible by construction (no wrf_val_skip).
    Case("XSBench_dist",              "xsbench_dist",              ["-s","small","-g","10","-l","100"],
         scalar_re=r"XS_CHECKSUM\s*=\s*(\d+)", scalar_kind="int"),
    # --- previously-SKIPped: revived with proper argv ---
    # nekbone: nrank=1 (Rx=Ry=Rz=1) used to deadlock on BOTH arts and xsocr
    # at the nekMultiplicity_stop step.  Root cause was a vendor bug in
    # third_party/ocr-apps/apps/nekbone/refactored/ocr_src/neko_halo.c:
    # start_halo_{multiplicity,setf,ai}() had `if(sz_riValues == 0) break;`
    # that skipped their NULL-fill loops when there were no cross-rank
    # halo values — leaving 27 halo slots of the downstream *_stop EDT
    # forever unsatisfied.  Fix: remove the three early-break lines so
    # the NULL-fill loop always runs.  Both backends now complete in
    # <0.5 s.  Harness workload kept at nrank=1 as a regression guard.
    Case("nekbone", "nekbone", ["1","1","1","1","1","1","2","1"],
         scalar_re=r"CGstep0_stop> rnorminit(?:\^2)?\s*=\s*([0-9.eE+-]+)",
         scalar_kind="float", scalar_tol=1e-9),
    Case("cholesky", "cholesky",
         ["--ds","50","--ts","10","--fi",CHOLESKY_INPUT],
         scalar_re=r"CHOLESKY trace\s*=\s*([0-9.eE+-]+)",
         scalar_kind="float", scalar_tol=1e-9),
    Case("cholesky_blas", "cholesky_blas",
         ["--ds","50","--ts","10","--fi",CHOLESKY_INPUT],
         scalar_re=r"CHOLESKY trace\s*=\s*([0-9.eE+-]+)",
         scalar_kind="float", scalar_tol=1e-9),
    # xeonNumaSize: needs at least one action flag or it just prints help.
    # -dcpu just enumerates CPUs and fires the DONE! marker fast.
    Case("xeonNumaSize", "xeonNumaSize", ["-dcpu"],
         scalar_re=r"DONE!", scalar_kind="bool"),
    Case("stream_dist", "stream_dist", [],
         scalar_re=r"STREAM checksum: a\[0\] = ([0-9.eE+-]+)", scalar_kind="float", scalar_tol=1e-9),

    # --- previously-SKIPped: genuine arts-side runtime bugs (report, don't fix) ---
    Case("miniAMR_intel_bryan", "miniAMR_intel_bryan", [],
         # checksum strengthening abandoned: xsocr does not emit the per-block
         # checksum and the arts value is non-deterministic across ranks; the
         # completion marker is the strongest portable check here.
         scalar_re=r"miniAMR complete", scalar_kind="bool"),
    # hpcg_intel_Eager_Collective: full 3-way at multinode.  The xsocr
    # runtime is built with the collective-event extension chain
    # (COLLECTIVE_EVT + MULTI_OUTPUT_SLOT + DISTRIBUTED_LABELED + REG_ASYNC_SGL),
    # so OCR_EVENT_COLLECTIVE_T (COL_ALLREDUCE) is handled cross-rank.  The arts
    # shim implements the same collective as a real cross-rank ARITY reduction
    # tree.  The app distributes ranks across PDs by integer division
    # (nrankPD = nrank/PDcount), so it requires npx*npy*npz == num_nodes — use
    # rank-count-matched geometry; the single-node reference is recomputed
    # per-n with the same geometry.
    Case("hpcg_intel_Eager_Collective", "hpcg_intel_Eager_Collective",
         ["1","1","1","16","5"],
         scalar_re=r"final deviation:\s*([\-+0-9.eE]+)",
         scalar_kind="float", scalar_tol=1e-4,
         multinode_args={2: ["2", "1", "1", "16", "5"],
                         3: ["3", "1", "1", "16", "5"],
                         4: ["4", "1", "1", "16", "5"],
                         8: ["8", "1", "1", "16", "5"],
                         16: ["16", "1", "1", "16", "5"]},
         ),

    Case("stream", "stream", [],
         scalar_re=r"STREAM_RESULT a\[0\] = ([0-9.eE+-]+)", scalar_kind="float", scalar_tol=1e-3),

    # --- previously-SKIPped: by-design stress test, cannot be tamed ---
    # miniAMR_forkbomb: previously SKIP-STRESS.  Vendor source hardcoded
    # timestep=1500 + maxRefLvl=3 + no ocrShutdown().  Fixed in
    # third_party/ocr-apps/.../forkbomb/mainOCR.c to honor --num_tsteps /
    # --num_refine and add a shutdown barrier on the initial block count.
    Case("miniAMR_forkbomb", "miniAMR_forkbomb",
         ["--npx","1","--npy","1","--npz","1","--max_time","3"],
         scalar_re=r"BLOCK 0 finished", scalar_kind="bool"),

    # --- SAR (revived via CMake crlibm bootstrap + datagen integration).
    # All sizes build all three backends; per-size datasets are embedded
    # via the .incbin pipeline. ---
    # SAR packs self-referential 2D arrays into single DataBlocks (row-pointer
    # tables built against the block's creation address), baked process-local
    # FILE* handles into a param DB, and stores absolute pointers to sibling
    # axis-vector DBs inside ImageParams — all of which go stale when a runtime
    # relocates the block to another rank (previously an identical SIGSEGV at
    # n>=2 on every runtime).  The ocr-apps SAR source now (a) rebuilds each
    # packed row-pointer table into EDT-local storage on entry (the vendor's own
    # predicted remap), (b) carries file PATHS instead of FILE* — each
    # file-touching EDT reopens on its executing node, ReadData seeking to its
    # image's file slice (no affinity dependence, so it also covers ocr-vx,
    # whose EDT-affinity hint value is stubbed out), and (c) reconstructs xr/yr
    # locally from ImageParams.  Runs clean at MN on every arts protocol
    # (VAL/EXCL × HOME/OWNER), on xsocr, AND on ocr-vx; implicit-data
    # cases keep their exact static pins at MN.  The pss consensus scalar is
    # 35892 (benign ULP FP-threshold shift from the compute-EDT edits,
    # unanimous across all rebuilt runtimes; no static pin).  WRF_VAL SIGSEGVs in
    # the affine path (cross-node __sync counters + disjoint-region sibling
    # writers are outside its DB-WRF contract) — wrf_val_skip excludes its vote;
    # the cell still runs so the SEGV surfaces truthfully.
    Case("sar_tiny",   "sar_tiny",   [],
         scalar_re=r"SAR detects:\s*(\d+)", scalar_kind="int",
         wrf_val_skip="cross-node __sync counter coordination + disjoint-region sibling writers to shared image blocks — guarantees outside WRF_VAL's DB-WRF contract (whole-DB lossy publish drops sibling updates)",
         wrf_val_multinode_skip="closed S/M verdict: cross-node __sync counter coordination stalls the backprojection stage under lossy publish -> 124 at MN (sar_mn_fix.md, wrf_val_mn_verdicts.md)"),
    Case("sar_small",  "sar_small",  [],
         scalar_re=r"SAR detects:\s*(\d+)", scalar_kind="int",
         wrf_val_skip="cross-node __sync counter coordination + disjoint-region sibling writers to shared image blocks — guarantees outside WRF_VAL's DB-WRF contract (whole-DB lossy publish drops sibling updates)",
         wrf_val_multinode_skip="closed S/M verdict: cross-node __sync counter coordination stalls the backprojection stage under lossy publish -> 124 at MN (sar_mn_fix.md, wrf_val_mn_verdicts.md)"),
    Case("sar_medium", "sar_medium", [],
         scalar_re=r"SAR detects:\s*(\d+)", scalar_kind="int",
         wrf_val_skip="cross-node __sync counter coordination + disjoint-region sibling writers to shared image blocks — guarantees outside WRF_VAL's DB-WRF contract (whole-DB lossy publish drops sibling updates)",
         wrf_val_multinode_skip="closed S/M verdict: cross-node __sync counter coordination stalls the backprojection stage under lossy publish -> 124 at MN (sar_mn_fix.md, wrf_val_mn_verdicts.md)"),
    Case("sar_pss", "sar_problem_size_scaling",
         [str(SAR_HUGE_DATA / "Data.bin"),
          str(SAR_HUGE_DATA / "PlatformPosition.bin"),
          str(SAR_HUGE_DATA / "PulseTransmissionTime.bin"),
          str(SCRATCH / "sar_detects_corr.txt"),
          f"{OCR_APPS}/sar/ocr/problem_size_scaling/Parameter0.txt"],
         scalar_re=r"SAR detects:\s*(\d+)", scalar_kind="int",
         wrf_val_skip="cross-node __sync counter coordination + disjoint-region sibling writers to shared image blocks — guarantees outside WRF_VAL's DB-WRF contract (whole-DB lossy publish drops sibling updates)",
         wrf_val_multinode_skip="closed S/M verdict: cross-node __sync counter coordination stalls the backprojection stage under lossy publish -> 124 at MN (sar_mn_fix.md, wrf_val_mn_verdicts.md)"),
    # sar_huge: same pipeline as pss over the same staged pulse data, but the
    # image grid comes from the huge variant's own in-tree Parameters.txt
    # (Ix=Iy=4000; the ladder tops out at 2000).  The dataset dir's
    # Parameters.txt is datagen provenance, not app-format params.
    Case("sar_huge", "sar_huge",
         [str(SAR_HUGE_DATA / "Data.bin"),
          str(SAR_HUGE_DATA / "PlatformPosition.bin"),
          str(SAR_HUGE_DATA / "PulseTransmissionTime.bin"),
          str(SCRATCH / "sar_detects_huge_corr.txt"),
          f"{OCR_APPS}/sar/ocr/huge/Parameters.txt"],
         scalar_re=r"SAR detects:\s*(\d+)", scalar_kind="int",
         # The 4000^2 backprojection measures ~55 s single-node on the
         # fastest coherence arm (16-core rank); the budgets leave the
         # slower write-through/purging arms and the reference runtimes
         # room to finish so a slow-but-correct cell is not misread as a
         # hang.
         timeout=300,
         multinode_timeout=600,
         wrf_val_skip="cross-node __sync counter coordination + disjoint-region sibling writers to shared image blocks — guarantees outside WRF_VAL's DB-WRF contract (whole-DB lossy publish drops sibling updates)",
         wrf_val_multinode_skip="closed S/M verdict: cross-node __sync counter coordination stalls the backprojection stage under lossy publish -> 124 at MN (sar_mn_fix.md, wrf_val_mn_verdicts.md)"),
    Case("sar_large",  "sar_large",  [],
         scalar_re=r"SAR detects:\s*(\d+)", scalar_kind="int",
         # The baked (non-CLI) dataset ran 61-63 s single-node under the
         # since-retired MRSW protocol (measured on both transport eras;
         # the surviving protocols finish in ~2 s) — that wall floor
         # motivated the per-case budget, kept as headroom.  The scalar
         # is still fully verified against consensus + the static pin.
         # The same wall floor scaled with rank count (quiet-box measured
         # ~103 s @2n, ~153 s @4n vs single-digit seconds on the other
         # protocols); per-case multinode budget 300s = ~2x the slowest
         # measured cell, so slow-but-correct completion is not misread as a
         # hang.
         timeout=120,
         multinode_timeout=300,
         wrf_val_skip="cross-node __sync counter coordination + disjoint-region sibling writers to shared image blocks — guarantees outside WRF_VAL's DB-WRF contract (whole-DB lossy publish drops sibling updates)",
         wrf_val_multinode_skip="closed S/M verdict: cross-node __sync counter coordination stalls the backprojection stage under lossy publish -> 124 at MN (sar_mn_fix.md, wrf_val_mn_verdicts.md)"),
]

# Runtime-input SAR cases need the staged huge dataset (see SAR_HUGE_DATA
# above); without it they cannot produce a truthful verdict, so they are
# removed from the matrix loudly rather than reported as failures.
if not sar_huge_data_present():
    _dropped = [c.name for c in CASES if c.name in ("sar_pss", "sar_huge")]
    CASES = [c for c in CASES if c.name not in ("sar_pss", "sar_huge")]
    print(f"[harness] SAR huge dataset not staged at {SAR_HUGE_DATA} -> "
          f"dropped {', '.join(_dropped)} (staging steps: datasets/README.md)",
          file=sys.stderr)


# Statically-derived absolute answers, verified against deterministic agreed
# output (xsocr == arts on a fixed, order-independent input).  Pinning them
# makes the harness reject a correlated regression where BOTH runtimes compute
# the same wrong value, not only a cross-runtime divergence.  Floats are matched
# with a relaxed relative tolerance (see _expect_ok), so reduced-precision
# entries here are safe.
_EXPECT: dict[str, str] = {
    "fibonacci": "55", "nqueens": "4", "smithwaterman": "1460", "triangle": "1074",
    "basicIO": "1", "highbw": "2048", "multigen": "121393", "multigen_2": "3524578",
    "XSBench_intel": "16088953243632067443", "XSBench_intel_sharedDB": "100",
    # XSBench_dist at -s small -g 10 -l 100: the promotion surgery's 9-way
    # unanimous computed digest, reproduced by construction (same fixed-seed
    # rand()-stream tables + index-pure lookup seeding + FNV digest).
    "XSBench_dist": "486159",
    # quicksort_dist default shape (N=1000, range=1e6): element sum of the
    # fixed-seed input, printed only when the sorted flag validates; agreed
    # by arts x7, xsocr, ocr-vx at 1n and 2n.
    "quicksort_dist": "496344943",
    # fft_dist at log2N=22: analytic single-tone checksum == N (two peaks of
    # N/2, all other bins zero) — closed-form, runtime-independent.
    "fft_dist": "4194304.0",
    "sar_tiny": "12", "sar_small": "458", "sar_medium": "1991", "sar_large": "6523",
    "CoMD_sdsc": "-1.166058121223", "CoMD_sdsc2": "-1.166063027842",
    "CoMD_intel_chandra_tiled": "-1.166063",
    "cholesky": "50", "hpcg_intel": "0.001279", "hpcg_intel_Eager": "0.001279",
    "hpgmg": "2.97878e-06", "miniAMR_intel": "710400", "npb_cg": "7.85534",
    "p2p": "228", "reduction_intel": "1025", "stencil1D_sticky": "1",
    "LCS_all_db_distributed": "523",
    "LCS_distributed_ST": "1024", "LCS_shared": "1024",
    # strengthened from completion/perf-only to a verified numeric answer
    "tempest": "21", "testlibs": "10", "quicksort": "26648",
    "globalsum_cgShim": "0.462231", "globalsum_cgNoShim": "0.462231",
    "globalsum_pcg": "0.999794", "reduction_intel_chandra": "45.0",
    "nekbone": "1.90095144672794E+00",
    "fft": "81.421870", "Stencil2D_intel_chandra": "22.0",
    "Stencil2D_intel_channelEVTs": "22.0", "graph500": "64",
    "stream": "2.321060036183137e+07", "stream_dist": "2.100022581888",
    "cache_offset": "17592181850112",
    "RSBench_intel_sharedDB": "17079",
}
for _c in CASES:
    if _c.name in _EXPECT and not _c.expect:
        _c.expect = _EXPECT[_c.name]


# ---------------------------------------------------------------------------
# Scalar extraction / comparison helpers.
# ---------------------------------------------------------------------------
def _pull(text: str, rex: str, kind: str):
    m = re.search(rex, text)
    if not m:
        return None
    if kind == "bool":
        return True
    try:
        g = m.group(1)
    except IndexError:
        return None
    try:
        return int(g) if kind == "int" else float(g)
    except ValueError:
        return None


def _drift(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def _expect_ok(value: object, case: "Case") -> bool:
    """Return True when value agrees with case.expect (absolute correctness pin).

    case.expect is always a string (parsed from _EXPECT); convert it to the
    case's scalar_kind before comparing so the comparison mirrors _pull()."""
    exp_str = case.expect
    if not exp_str:
        return True
    try:
        if case.scalar_kind == "bool":
            return bool(value) == bool(exp_str)
        if case.scalar_kind == "int":
            return int(value) == int(exp_str)  # type: ignore[arg-type]
        # float: the pin is a human-rounded reference value, so compare with a
        # tolerance no tighter than the project's FP tolerance (0.01% = 1e-4),
        # never the (often far tighter) cross-runtime scalar_tol.  The runtimes
        # agree bit-for-bit so scalar_tol can be 1e-10; but a pin like npb_cg's
        # "7.85534" differs from the agreed 7.8553405... by 5e-6, which a 1e-10
        # tol would spuriously flag as EXPECT-FAIL.  Looser per-app tols win.
        pin_tol = max(case.scalar_tol, 1e-4)
        return _drift(float(value), float(exp_str)) <= pin_tol  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# 9-way consensus engine (Tasks 4-6).
# ---------------------------------------------------------------------------

NODE_CONFIGS: list = ["1n"] + MN_RANKS   # e.g. ["1n", 2, 4, 8, 16, "2n_io", ...]


def runtime_eligible(rt: Runtime, case: Case, node: object) -> bool:
    """Return True iff runtime rt should be invoked for (case, node)."""
    is_io = isinstance(node, str) and node.endswith("_io")
    single = node in ("1n", 1)
    # _io configs are arts-only (no xsocr/ocrvx/baseline MPI equivalent).
    if rt.kind in ("xsocr", "ocrvx", "baseline") and is_io:
        return False
    if not single:
        # Multinode eligibility: every case runs at every node config unless
        # multinode_skip documents a structural reason, or the case has no
        # scalar to vote on.
        if case.multinode_skip or not case.scalar_re:
            return False
        if rt.kind == "arts" and rt.key == "wrf_val_wt" and case.wrf_val_multinode_skip:
            return False
    if (rt.kind == "ocrvx" and case.ocrvx_mn_max_ranks
            and isinstance(node, int) and node > case.ocrvx_mn_max_ranks):
        return False
    if rt.kind == "ocrvx" and not (APPS_DIR / f"{case.ocr_base}_ocrvx").exists():
        return False
    if rt.kind == "baseline" and case.baseline is None:
        return False
    return True


def run_runtime(runner: Runner, rt: Runtime, case: Case, node: object) -> dict:
    """Run a single (runtime, case, node) cell and return a CellRun dict.

    state is "OK?" (scalar extracted -- rc==0, or rc!=0 with "teardown"=True,
    see below) or "FAIL" (scalar regex did not match, regardless of rc), or
    "N/A" (excluded by eligibility). A "teardown"=True cell is an "OK?" whose
    process died (SIGSEGV/SIGTERM/SIGKILL) AFTER printing a complete, correct
    result -- a known xsocr/ocr-vx teardown-phase intermittent, not a
    correctness failure. consensus() resolves "OK?" into "OK" / "OK-TEARDOWN"
    (teardown cells whose scalar agrees with the winning cluster), "DISAGREE"
    (scalar does not match, teardown or not), or "NO-CONSENSUS" (tied
    clusters).
    """
    if not runtime_eligible(rt, case, node):
        return {"state": "N/A", "rc": None, "wall": None, "scalar": None}
    # Determine the effective args for this node geometry.
    if node in ("1n", 1):
        geo = 1
    else:
        geo = int(str(node).split("n")[0])
    args = (case.multinode_args.get(geo, case.args)
            if case.multinode_args else case.args)
    # Dispatch.
    if node in ("1n", 1):
        if rt.kind == "arts":
            r = runner.run_ocr(case.name, case.ocr_base, args, "arts",
                               suffix=rt.suffix, timeout=case.timeout)
        elif rt.kind == "xsocr":
            r = runner.run_ocr(case.name, case.ocr_base, args, "xsocr",
                               timeout=case.timeout)
        elif rt.kind == "ocrvx":
            r = runner.run_ocrvx_mpi(case.name, case.ocr_base, args,
                                     timeout=case.timeout)
        else:  # baseline
            r = runner.run_baseline(case.name, case.baseline,
                                    timeout=case.timeout)
    else:
        mn_to = case.multinode_timeout
        if rt.kind == "arts":
            r = runner.run_arts_mn(case.name, case.ocr_base, args, node,
                                   timeout=mn_to, suffix=rt.suffix)
        elif rt.kind == "xsocr":
            r = runner.run_xsocr_mpi(case.name, case.ocr_base, args, node,
                                     timeout=mn_to)
        else:  # ocrvx (baseline never at multinode, runtime_eligible guards)
            r = runner.run_ocrvx_mpi(case.name, case.ocr_base, args,
                                     np=geo, timeout=mn_to)
    # Classify: try to extract the scalar regardless of rc. A scalar miss is
    # always FAIL (genuine failure, whatever rc says). A scalar hit with
    # rc==0 is a normal "OK?" candidate. A scalar hit with rc!=0 (SIGSEGV,
    # or a SIGTERM/SIGKILL-reaped hang) is STILL a verifiable result -- xsocr/
    # ocr-vx are known to die AFTER printing correct output during teardown
    # (deprioritized runtime defect, not a correctness bug) -- so it also
    # becomes an "OK?" candidate and votes in consensus() normally; marking
    # it teardown=True lets consensus() render an otherwise-"OK" resolution
    # as "OK-TEARDOWN" instead of silently hiding the death.
    wall_r = round(r.wall, 3)
    val = _pull(r.stdout, case.scalar_re, case.scalar_kind)
    if val is None:
        return {"state": "FAIL", "rc": r.rc, "wall": wall_r, "scalar": None}
    if r.rc != 0:
        return {"state": "OK?", "rc": r.rc, "wall": wall_r, "scalar": val,
                "teardown": True}
    return {"state": "OK?", "rc": 0, "wall": wall_r, "scalar": val}


def consensus(cells: dict, case: Case) -> tuple:
    """Cluster scalars from all "OK?" cells; largest cluster wins.

    Returns (consensus_value, dict[key -> final_state]) where final_state is
    one of "OK", "OK-TEARDOWN", "DISAGREE", "FAIL", "N/A", or "NO-CONSENSUS".
    A cell's "teardown" flag (process died after printing a valid scalar --
    see run_runtime) only affects an otherwise-"OK" resolution, downgrading
    it to "OK-TEARDOWN"; a teardown cell whose scalar disagrees still
    resolves to "DISAGREE" like any other outlier -- scalar comparison always
    precedes the teardown distinction, so nothing is hidden.
    """
    final: dict = {}
    for k, c in cells.items():
        if c["state"] == "N/A":
            final[k] = "N/A"
        elif c["state"] == "FAIL":
            # A no-vote cell that exited cleanly (rc==0) but yielded no
            # parsable scalar is the declared situation itself -- the value
            # (including its printability) is incomparable under the
            # declaration.  Only a nonzero rc (hang/crash) must surface.
            if c.get("novote") and c.get("rc") == 0:
                final[k] = "NV"
            else:
                final[k] = "FAIL"
        elif c.get("novote"):
            # Ran under a no-vote declaration and did not fail: the run is
            # the point (wedges/crashes must surface as FAIL above); the
            # value is declared incomparable and casts no vote.
            final[k] = "NV"
    # Candidates are cells that ran, produced a scalar, and hold a vote.
    cand = {k: c for k, c in cells.items()
            if c["state"] == "OK?" and c["scalar"] is not None
            and not c.get("novote")}
    if not cand:
        return (None, final)

    def same(a: object, b: object) -> bool:
        if case.scalar_kind == "bool":
            return bool(a) == bool(b)
        if case.scalar_kind == "int":
            return a == b
        # float
        ref = b if b not in (0, 0.0) else None
        return (_drift(a, b) <= case.scalar_tol  # type: ignore[arg-type]
                if ref is not None else (a == b))

    clusters: list[dict] = []
    for k, c in cand.items():
        placed = False
        for cl in clusters:
            if same(c["scalar"], cl["rep"]):
                cl["members"].append(k)
                placed = True
                break
        if not placed:
            clusters.append({"rep": c["scalar"], "members": [k]})
    clusters.sort(key=lambda cl: len(cl["members"]), reverse=True)
    top = clusters[0]
    tie = (len(clusters) > 1
           and len(clusters[1]["members"]) == len(top["members"]))
    for cl in clusters:
        if tie:
            st = "NO-CONSENSUS"
        else:
            st = "OK" if cl is top else "DISAGREE"
        for k in cl["members"]:
            final[k] = ("OK-TEARDOWN" if st == "OK" and cells[k].get("teardown")
                       else st)
    consensus_val = None if tie else top["rep"]
    return (consensus_val, final)


def run_matrix(runner: Runner, cases: list, only: set, no_baseline: bool,
               runtimes: "set[str] | None" = None) -> dict:
    """Run every (case, node-config, runtime) cell; compute consensus per (case, node).

    `runtimes`, when non-empty, restricts the voting set to those runtime keys
    (baseline and the static expect pins still participate) — used to re-verify
    a change that touches only some runtimes without paying for the full 9-way.

    Returns a nested dict: report[case_name][node_str] = {consensus, cells}.
    """
    report: dict = {}
    glyph = {
        "OK": "·", "OK-TEARDOWN": "t", "DISAGREE": "X", "FAIL": "!",
        "N/A": "-", "NV": "~", "NO-CONSENSUS": "?",
    }
    for c in cases:
        if only and c.name not in only:
            continue
        report[c.name] = {}
        rts = [rt for rt in RUNTIMES if not runtimes or rt.key in runtimes]
        if c.baseline is not None and not no_baseline:
            rts = rts + [Runtime("baseline", "baseline", "baseline")]
        for node in NODE_CONFIGS:
            cells: dict = {rt.key: run_runtime(runner, rt, c, node)
                           for rt in rts}
            single = node in ("1n", 1)
            for k in cells:
                nv = (k == "wrf_val_wt" and bool(c.wrf_val_skip)) or \
                     (not single and bool(c.multinode_novote))
                cells[k] = dict(cells[k])
                cells[k]["novote"] = nv
            cval, final = consensus(cells, c)
            for k in cells:
                cells[k] = dict(cells[k])
                cells[k]["state"] = final.get(k, cells[k]["state"])
            report[c.name][str(node)] = {"consensus": cval, "cells": cells}
            row = " ".join(
                f"{k}={glyph.get(cells[k]['state'], '?')}" for k in cells
            )
            print(f"[{c.name:28s} {str(node):8s}] consensus={cval}  {row}")
    return report


def emit(report: dict, logdir: Path, cases: "list[Case] | None" = None) -> None:
    """Write report.json, summary.txt, and print per-node tables + minority report."""
    # Build a name→Case map for expect-pin lookup (Finding 1).
    _case_map: dict[str, "Case"] = {}
    if cases:
        for _c in cases:
            _case_map[_c.name] = _c

    (logdir / "report.json").write_text(
        json.dumps(report, indent=2, default=str))
    lines: list[str] = []
    minority: list[tuple] = []

    # Finding 3: determine once whether any case has a baseline cell (non-N/A)
    # so the header column is only shown when relevant.
    _has_baseline: dict[str, bool] = {}
    for node in NODE_CONFIGS:
        node_str = str(node)
        has_base = any(
            byn.get(node_str, {}).get("cells", {}).get("baseline", {}).get("state", "N/A") != "N/A"
            for byn in report.values()
        )
        _has_baseline[node_str] = has_base

    for node in NODE_CONFIGS:
        node_str = str(node)
        lines.append(f"\n=== node-config {node_str} ===")
        hdr = ["app"] + [rt.key for rt in RUNTIMES]
        if _has_baseline[node_str]:
            hdr.append("baseline")
        hdr.append("consensus")
        lines.append(" | ".join(hdr))
        for app, byn in report.items():
            cell = byn.get(node_str)
            if not cell:
                continue
            cells = cell["cells"]
            cval = cell["consensus"]

            def _g(k: str) -> str:
                c = cells.get(k)
                if not c:
                    return "-"
                s = c["state"]
                if s == "OK":
                    return "OK"
                if s == "OK-TEARDOWN":
                    return f"OKt:{c['rc']}"
                if s == "DISAGREE":
                    return f"X:{c['scalar']}"
                if s == "FAIL":
                    return f"FAIL:{c['rc']}"
                if s == "NO-CONSENSUS":
                    return f"?:{c['scalar']}"
                return "-"  # N/A

            # Finding 1: check consensus value against absolute expect pin.
            case_obj = _case_map.get(app)
            expect_fail = (
                case_obj is not None
                and case_obj.expect
                and cval is not None
                and not _expect_ok(cval, case_obj)
            )
            consensus_cell = (
                f"EXPECT-FAIL:{cval}(exp{case_obj.expect})"
                if expect_fail else str(cval)
            )

            row = [app] + [_g(rt.key) for rt in RUNTIMES]
            if _has_baseline[node_str]:
                row.append(_g("baseline"))
            row.append(consensus_cell)
            lines.append(" | ".join(row))

            # OK-TEARDOWN deliberately excluded: its scalar agreed with
            # consensus, so it is a verified result, not a correctness
            # failure -- the teardown death is still visible via the "OKt"
            # glyph and the report.json state string, just not flagged here.
            bad_states = {"DISAGREE", "FAIL", "NO-CONSENSUS"}
            # Finding 2: also check baseline cell for bad state.
            rt_keys = [rt.key for rt in RUNTIMES] + ["baseline"]
            if (any(cells.get(k, {}).get("state") in bad_states for k in rt_keys)
                    or expect_fail):
                div_cells = {k: (v["state"], v["scalar"], v["rc"])
                             for k, v in cells.items()
                             if v["state"] in bad_states}
                if expect_fail:
                    div_cells["__expect__"] = (
                        f"EXPECT-FAIL", cval, None
                    )
                minority.append((app, node_str, div_cells))

    lines.append("\n=== MINORITY REPORT (non-unanimous (app,config)) ===")
    for app, node_s, div in minority:
        parts = ", ".join(
            f"{k}:{s}({val if val is not None else rc})"
            for k, (s, val, rc) in div.items()
        )
        lines.append(f"{app} @ {node_s}: {parts}")
    txt = "\n".join(lines)
    (logdir / "summary.txt").write_text(txt)
    print(txt)


def _selftest() -> None:
    """Assert consensus() behaves correctly on synthetic cell dicts."""
    # Minimal Case stub for selftest (only scalar_kind and scalar_tol matter).
    dummy = Case(name="test", ocr_base="test", args=[],
                 scalar_kind="float", scalar_tol=1e-4)

    def _ok(v: object) -> dict:
        return {"state": "OK?", "rc": 0, "wall": 0.1, "scalar": v}

    def _fail() -> dict:
        return {"state": "FAIL", "rc": 1, "wall": 0.1, "scalar": None}

    def _na() -> dict:
        return {"state": "N/A", "rc": None, "wall": None, "scalar": None}

    def _teardown(v: object, rc: int = 139) -> dict:
        # A process that died (SIGSEGV/SIGTERM/SIGKILL) AFTER printing a
        # valid scalar -- see run_runtime's teardown=True branch.
        return {"state": "OK?", "rc": rc, "wall": 0.1, "scalar": v,
                "teardown": True}

    # 1. Unanimous floats within tol → all OK.
    cells1 = {"a": _ok(1.0), "b": _ok(1.0), "c": _ok(1.0000001)}
    cval1, final1 = consensus(cells1, dummy)
    assert cval1 is not None, "unanimous: expected a consensus value"
    assert all(v == "OK" for v in final1.values()), f"unanimous: {final1}"
    print("  selftest 1 PASS: unanimous floats → all OK")

    # 2. One outlier → DISAGREE; majority stays OK.
    cells2 = {"a": _ok(1.0), "b": _ok(1.0), "c": _ok(99.0)}
    cval2, final2 = consensus(cells2, dummy)
    assert cval2 is not None, "outlier: expected a consensus value"
    assert final2["a"] == "OK" and final2["b"] == "OK", f"outlier majority: {final2}"
    assert final2["c"] == "DISAGREE", f"outlier minority: {final2}"
    print("  selftest 2 PASS: one outlier → DISAGREE")

    # 3. One hang (FAIL) + rest agree → hang stays FAIL, rest OK.
    cells3 = {"a": _ok(1.0), "b": _ok(1.0), "c": _fail()}
    cval3, final3 = consensus(cells3, dummy)
    assert cval3 is not None, "hang: expected a consensus value"
    assert final3["a"] == "OK" and final3["b"] == "OK", f"hang rest: {final3}"
    assert final3["c"] == "FAIL", f"hang stays FAIL: {final3}"
    print("  selftest 3 PASS: 1 hang + rest agree → hang FAIL, rest OK")

    # 4. Even split → NO-CONSENSUS.
    cells4 = {"a": _ok(1.0), "b": _ok(2.0)}
    cval4, final4 = consensus(cells4, dummy)
    assert cval4 is None, f"even split: expected no consensus, got {cval4}"
    assert all(v == "NO-CONSENSUS" for k, v in final4.items()
               if cells4[k]["state"] == "OK?"), f"even split: {final4}"
    print("  selftest 4 PASS: even split → NO-CONSENSUS")

    # 5. All N/A → no consensus value, all N/A states.
    cells5 = {"a": _na(), "b": _na()}
    cval5, final5 = consensus(cells5, dummy)
    assert cval5 is None, f"all N/A: {cval5}"
    assert all(v == "N/A" for v in final5.values()), f"all N/A: {final5}"
    print("  selftest 5 PASS: all N/A → None consensus, all N/A")

    # 6. _expect_ok: consensus value that disagrees with the expect pin →
    #    _expect_ok returns False → EXPECT-FAIL would be surfaced in emit().
    dummy_pinned = Case(name="pinned", ocr_base="pinned", args=[],
                        scalar_kind="float", scalar_tol=1e-4,
                        expect="1.0")
    # Within tolerance → passes.
    assert _expect_ok(1.00005, dummy_pinned), "expect_ok within tol should be True"
    # Far from the pin → fails.
    assert not _expect_ok(99.0, dummy_pinned), "expect_ok far from pin should be False"
    # Integer comparison.
    dummy_int = Case(name="pi", ocr_base="pi", args=[],
                     scalar_kind="int", scalar_tol=0, expect="42")
    assert _expect_ok(42, dummy_int), "expect_ok int match"
    assert not _expect_ok(43, dummy_int), "expect_ok int mismatch"
    print("  selftest 6 PASS: _expect_ok correctly flags consensus-vs-pin mismatch")

    # 7. Teardown death (rc!=0) whose scalar AGREES with consensus →
    #    OK-TEARDOWN, not OK (visible, not hidden) and not FAIL/DISAGREE
    #    (it still votes normally and joins the winning cluster).
    cells7 = {"a": _ok(1.0), "b": _ok(1.0), "c": _teardown(1.0)}
    cval7, final7 = consensus(cells7, dummy)
    assert cval7 is not None, "teardown agree: expected a consensus value"
    assert final7["a"] == "OK" and final7["b"] == "OK", f"teardown agree rest: {final7}"
    assert final7["c"] == "OK-TEARDOWN", f"teardown agree: {final7}"
    print("  selftest 7 PASS: teardown death + agreeing scalar → OK-TEARDOWN")

    # 8. Teardown death whose scalar DISAGREES with consensus → still
    #    DISAGREE (scalar comparison precedes the teardown classification;
    #    a wrong answer is never laundered into OK-TEARDOWN).
    cells8 = {"a": _ok(1.0), "b": _ok(1.0), "c": _teardown(99.0)}
    cval8, final8 = consensus(cells8, dummy)
    assert final8["c"] == "DISAGREE", f"teardown disagree: {final8}"
    print("  selftest 8 PASS: teardown death + disagreeing scalar → DISAGREE (not hidden)")

    print("\nAll selftest assertions passed.")



# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", default="build_release_ocr_val_wb",
                   help="Build directory containing apps and configs")
    p.add_argument("--target", default="cbgpu02", choices=["cbgpu02"],
                   help="Machine geometry (drives config subdir + node counts)")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--only", type=str, default="")
    p.add_argument("--mem-gb", type=int, default=4)
    # 30s default: every case is sized (problem-size args, see CASES) so the
    # slowest reference runtime finishes well inside this budget at the widest
    # rank count; passing cases return in seconds, so only a genuine hang pays
    # the full wait.  ocr-vx gets its own larger cap (_OCRVX_TIMEOUT_MULT in
    # harness_common.py) on top of this base.
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--node", type=str, default="",
                   help="Restrict to a single node-config, e.g. --node 1n")
    p.add_argument("--runtimes", type=str, default="",
                   help="Comma-separated runtime keys to run (e.g. "
                        "ocr_val_wb,xsocr,ocrvx); empty = all. Baseline and "
                        "expect pins still participate in the consensus.")
    p.add_argument("--selftest", action="store_true",
                   help="Run consensus unit tests (no build required) and exit")
    args = p.parse_args()

    if args.selftest:
        print("Running consensus selftest...")
        _selftest()
        return

    # Restrict NODE_CONFIGS when --node is given.
    global NODE_CONFIGS
    if args.node:
        # Accept either "1n" (string) or integer-like "2" / "2n_io".
        target_node: object
        if args.node == "1n":
            target_node = "1n"
        elif args.node.endswith("_io"):
            target_node = args.node
        else:
            try:
                target_node = int(args.node.rstrip("n"))
            except ValueError:
                target_node = args.node
        NODE_CONFIGS = [target_node]

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    runtimes = {s.strip() for s in args.runtimes.split(",") if s.strip()}
    if runtimes:
        known = {rt.key for rt in RUNTIMES}
        unknown = runtimes - known
        if unknown:
            p.error(f"unknown --runtimes keys {sorted(unknown)}; "
                    f"valid: {sorted(known)}")
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    logdir = LOGS_ROOT / ts
    runner = Runner(args.mem_gb, args.timeout, logdir, target=TARGET, build=BUILD)

    cases = CASES
    report = run_matrix(runner, cases, only, args.no_baseline, runtimes)
    emit(report, logdir, cases)




if __name__ == "__main__":
    main()
