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

The harness runs xsocr and arts with identical argv, extracts the scalar
from each stdout, and compares. Tier B adds a baseline run with matched
parameters and its own scalar_re; all three must agree.

Apps without a meaningful scalar concept (printf test, basicIO, cache tests,
etc.) get `scalar_re = ""` and are verified via rc-only (PASS-RC verdict).

Usage:
    python3 correctness_harness.py                       # full suite
    python3 correctness_harness.py --no-baseline         # Tier A only
    python3 correctness_harness.py --only nqueens,CoMD_sdsc2_B
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time

shlex_quote = shlex.quote
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Build directory: selectable via --build-dir (default: build_release_mrnew_lazy).
# Resolved early so module-level Path constants can reference it.
# ---------------------------------------------------------------------------
_arg_parser = argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument('--build-dir', default='build_release_mrnew_lazy')
_arg_parser.add_argument('--target', default='laptop', choices=['laptop', 'server'])
_pre_args, _ = _arg_parser.parse_known_args()
BUILD = Path(_pre_args.build_dir)
if not BUILD.is_absolute():
    BUILD = REPO / BUILD
TARGET = _pre_args.target

_protocol = 'unknown'
_timing = ''
try:
    _cache_path = BUILD / 'CMakeCache.txt'
    with open(_cache_path) as _f:
        for _line in _f:
            _p = re.match(r'ARTS_COHERENCE_PROTOCOL:STRING=(\w+)', _line)
            if _p:
                _protocol = _p.group(1)
            _t = re.match(r'ARTS_PROTOCOL_TIMING:STRING=(\w+)', _line)
            if _t:
                _timing = _t.group(1)
except FileNotFoundError:
    pass
_mode = 'MRMW' if _protocol == 'MRMW' else f'{_protocol}+{_timing}'
print(f'[harness] Build dir: {BUILD} (protocol: {_mode})')

APPS_DIR = BUILD / "benchmarks" / "apps"
BASE_DIR = BUILD / "benchmarks" / "baseline"
LOGS_ROOT = REPO / "benchmarks" / "scripts" / "logs" / "correctness"
ARTS_CFG = REPO / "configs" / "local" / TARGET / "1n.cfg"
XSOCR_CFG = REPO / "configs" / "mpi" / TARGET / "1n.cfg"
OCRVX_APPS_DIR = APPS_DIR   # ocr-vx binaries live in the same apps/ dir

# smithwaterman ships its own datasets (tiny/small/medium/large triples of
# string1/string2/score); use the official tiny set directly — no staging.
SW_DATA = Path(__file__).resolve().parents[2] / \
    "third_party/ocr-apps/apps/smithwaterman/datasets"
BASIC_IO_DAT = "/tmp/arts_basicIO_test.dat"
CHOLESKY_INPUT = "/tmp/arts_cholesky_input.mat"


# ---------------------------------------------------------------------------
# Per-target machine geometry (laptop = 14-thread budget, server = 48-thread),
# selected by --target.  Drives the config subdir, the Tier-M node counts, the
# per-rank thread budget (for taskset pinning), and the ocr-vx TBB width.  Every
# config is sized so node_count * per-node-threads == the machine's core count.
# ---------------------------------------------------------------------------
def _cfg_name(n) -> str:
    return f"{n}n.cfg" if isinstance(n, int) else f"{n}.cfg"


# Multinode node counts exercised in Tier M.  The *_io entries are arts-only:
# xsocr/ocr-vx have a single comm worker (no sender/receiver split), so their
# N-node total already equals the plain N-node config — no separate IO variant.
_MN_NODE_COUNTS = {
    'laptop': [2, 3, 4, "2n_io"],
    'server': [2, 4, 8, 16, "2n_io", "4n_io", "8n_io"],
}
MN_RANKS = _MN_NODE_COUNTS[TARGET]

# Threads per rank (= per-node total thread budget) used to taskset-pin each
# mpirun rank to a disjoint core block, mirroring arts's per-rank pu_offset.
_TPN = {
    'laptop': {2: 7, 3: 4, 4: 3},
    'server': {2: 24, 4: 12, 8: 6, 16: 3},
}[TARGET]

# ocr-vx TBB worker count per node count = full per-rank core budget, matching
# the original runtime's default (tbb::info::default_concurrency() = all cores
# the process sees).  We previously reserved one slot for the runtime's blocking
# shutdown-barrier task (budget-1), but the original gives ocr-vx the full
# budget, so match it.  Budget >= 3 at every node count here, so the P=1
# shutdown deadlock cannot occur.
_OCRVX_TBB = {
    'laptop': {1: 14, 2: 7, 3: 4, 4: 3},
    'server': {1: 48, 2: 24, 4: 12, 8: 6, 16: 3},
}[TARGET]

# ocr-vx's coherence protocol is structurally far slower than arts/xsocr, so a
# correctness run that those two finish well within budget can still time out on
# ocr-vx while it is *still making progress* (not hung).  Give ocr-vx a larger
# wall budget so a genuine result is collected; fast cases are unaffected (they
# return long before the ceiling).
_OCRVX_TIMEOUT_MULT = 8

# Physical cores used on this machine target (= single-node thread budget).
# The reference runtimes (xsocr/ocr-vx/baseline) have no internal core pinning,
# so a single-node run would float its threads across ALL logical CPUs (incl.
# the HT siblings above core count).  Confine them to cores 0..N-1 via taskset
# so they occupy the same cores arts self-pins to.
_NCORES = {'laptop': 14, 'server': 48}[TARGET]
_PIN_SINGLE = f"taskset -c 0-{_NCORES - 1}"


def _pin_wrap(tpn: int) -> str:
    """Wrap an mpirun-launched command so each rank taskset-pins itself to a
    disjoint block of `tpn` cores by PMI_RANK, mirroring arts's per-rank core
    placement so the reference runtimes occupy the same cores in a localhost
    multinode simulation."""
    return ("bash -c 'r=${PMI_RANK:-0}; s=$((r*%d)); e=$((s+%d-1)); "
            "exec taskset -c $s-$e \"$@\"' _" % (tpn, tpn))


# ---------------------------------------------------------------------------
# Portable mpirun launcher.  OpenMPI requires --oversubscribe to place more
# ranks than detected slots; MPICH (Hydra) rejects that flag outright and
# oversubscribes by default.  Probe the active mpirun once so the launcher
# line works under either implementation.
# ---------------------------------------------------------------------------
def _detect_mpirun_oversubscribe() -> str:
    try:
        out = subprocess.run(["mpirun", "--version"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    return "--oversubscribe" if ("Open MPI" in out or "OpenRTE" in out) else ""


MPIRUN_OVERSUB = _detect_mpirun_oversubscribe()


def mpirun_prefix(np: int) -> str:
    """`mpirun [--oversubscribe] -n <np>` — oversubscribe flag only when the
    active launcher is OpenMPI."""
    parts = ["mpirun"]
    if MPIRUN_OVERSUB:
        parts.append(MPIRUN_OVERSUB)
    parts += ["-n", str(np)]
    return " ".join(parts)


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
    skip: str = ""                        # non-empty → SKIP(reason)
    stress_skip: str = ""                 # non-empty → SKIP-STRESS(reason)
    arts_only: bool = False               # True → never invoke xsocr
    expected_known_bug: str = ""          # FAIL → KNOWN-BUG(reason) when set
    baseline: BaselineSpec | None = None
    multinode: bool = False               # True → also run Tier M (arts + xsocr at N ranks)
    multinode_skip: str = ""              # non-empty → skip multinode with reason
    multinode_arts_only: bool = False     # Tier M runs arts only (xsocr blocked at
                                          # multinode); Tier A still compares 3-way
    multinode_xsocr_only: bool = False    # Tier M runs xsocr only (arts blocked at
                                          # multinode, e.g. a transient runtime-refactor
                                          # casualty); Tier A still compares 3-way
    multinode_timeout: int = 0            # per-case Tier-M wall budget (s); 0 → global
    multinode_args: dict | None = None    # per-rank-count args for geometry apps
                                          # (e.g. hpcg needs npx*npy*npz==nodes);
                                          # the single-node reference is recomputed
                                          # per-n with the matching geometry
    ocrvx_skip: str = ""                  # non-empty → skip ocrvx run for this case
                                          # (binary exists but known runtime gap)

# Tier A — one entry per OCR pair. Workload args lifted from run_benchmarks.sh.
# For apps whose last printed value is timing or throughput, scalar_re stays ""
# (rc-only verdict). Adding a print to those apps is a vendor-source task
# explicitly out of scope.
TIER_A: list[Case] = [
    # --- scalar-checkable (scientific output already present) ---
    Case("fibonacci", "fibonacci", ["10"],
         scalar_re=r"answer is\s*(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("nqueens", "nqueens", ["6", "2"],
         scalar_re=r"sols:\s*(\d+)", scalar_kind="int",
         multinode_skip="single-node design: process-local solutions counter + global template GUIDs"),
    Case("smithwaterman", "smithwaterman",
         ["50","50",f"{SW_DATA}/string1-medium-large.txt",
          f"{SW_DATA}/string2-medium-large.txt",
          f"{SW_DATA}/score-medium-large.txt"],
         scalar_re=r"score:\s*(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("fft", "fft", ["6"],
         scalar_re=r"FFT checksum\s*=\s*([\-+0-9.eE]+)", scalar_kind="float", scalar_tol=1e-3,
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    # Depth-6 partial search (5072 sequences): the full-depth puzzle is a
    # fine-grained EDT-per-node tree (~1.3M EDTs) that the ocr-vx runtime's
    # serialized local message pipeline cannot finish within any per-case
    # budget (and its monotonic object retention exceeds the memory cap).
    # Depth 6 keeps a real 3-way correctness check; the app still solves the
    # full puzzle when run with no argument.
    Case("triangle", "triangle", ["6"],
         scalar_re=r"final count\s+(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("p2p", "p2p", ["2","10","100","10"],
         scalar_re=r"PASS checksum\s*=\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-8,
         multinode=True),
    Case("CoMD_intel_chandra", "CoMD_intel_chandra",
         ["-x","4","-y","4","-z","4","-N","2","-n","1"],
         # The energy is a parallel reduction over atoms; summation order varies
         # with the per-rank domain decomposition, so the value carries FP
         # non-associativity noise across rank counts (observed ~1.6e-5 at 4
         # ranks).  Use the numerical-port tolerance (0.01%); a real divergence
         # in a molecular-dynamics energy is orders of magnitude larger.
         scalar_re=r"Initial energy\s*:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-4,
         multinode=True),
    Case("CoMD_intel_chandra_tiled", "CoMD_intel_chandra_tiled",
         ["-x","4","-y","4","-z","4","-N","2"],
         scalar_re=r"Final energy\s*:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-6,
         multinode=True,  # 3-way: xsocr runs it at MN (old HCDist-timeout reason was stale)
         ),
    Case("CoMD_sdsc", "CoMD_sdsc", ["-x","4","-y","4","-z","4","-N","2"],
         # "Final energy" is the end-to-end answer (after the 2-timestep loop);
         # "Initial energy" only echoes the setup state and verifies nothing.
         scalar_re=r"Final energy\s*:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-6,
         multinode_skip="no EDT affinity hints (sdsc variant); runs caller-rank only"),
    Case("CoMD_sdsc2", "CoMD_sdsc2", ["-x","4","-y","4","-z","4","-N","2"],
         scalar_re=r"Final energy\s*:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-6,
         multinode=True),
         # xsocr passes at every rank count (the np4 home-MD race + the residual
         # np2/3 startup hang were fixed app/xsocr-side).  arts is KNOWN to hang
         # at multinode in the affinity-DB create path here = an arts bug to
         # debug.  The arts-block flag (multinode_xsocr_only) and the stale
         # xsocr-hang mask (expected_known_bug) were REMOVED 2026-06-08 so the
         # harness reports the true verdict — do NOT re-mask to go green; fix
         # the arts multinode affinity-DB-create hang.
    Case("hpcg_intel", "hpcg_intel", ["1","1","1","16","5"],
         scalar_re=r"final deviation:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-4,
         multinode=True),
    # The "Eager" in these app names is the app's reduction algorithm (the
    # reductionEager library / OCR_HINT_DB_EAGER read-prefetch hint), NOT a
    # runtime coherence mode.  The benchmark builds run xsocr eager-only and
    # ocr-vx lazy-only with the per-DB eager/lazy hints disabled/ignored, so
    # these apps exercise each runtime's default coherence regardless of the
    # hint.
    Case("hpcg_intel_Eager", "hpcg_intel_Eager", ["1","1","1","16","5"],
         scalar_re=r"final deviation:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-4,
         multinode=True),
    Case("Stencil1D_intel_chandra", "Stencil1D_intel_chandra", [],
         scalar_re=r"Solution validates", scalar_kind="bool",
         multinode=True),
    Case("Stencil2D_intel_channelEVTs", "Stencil2D_intel_channelEVTs", [],
         scalar_re=r"Computed L1 norm\s*=\s*([\-+0-9.eE]+)", scalar_kind="float", scalar_tol=1e-6,
         multinode=True,
         ),
    Case("Stencil2D_intel_chandra", "Stencil2D_intel_chandra", [],
         scalar_re=r"L1 norm\s*=\s*([\-+0-9.eE]+)", scalar_kind="float", scalar_tol=1e-6,
         multinode=True),
    Case("miniAMR_intel", "miniAMR_intel",
         ["--nx","4","--ny","4","--nz","4","--num_tsteps","2","--num_objects","1"],
         scalar_re=r"Grand Total Checksum\s*==\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-8,
         multinode_skip="no EDT affinity hints (intel variant); runs caller-rank only"),
    Case("npb_cg", "npb_cg", ["-t", "T"],
         # Anchor on the app's own verification verdict: a FAILED run prints
         # "Verification FAILED (zeta=NaN, correct zeta=<expected>)" and the
         # bare zeta regex would match the *expected* value (false PASS).
         scalar_re=r"Verification SUCCESSFUL \(zeta\s*=\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-10,
         multinode=True),
         # All runtimes pass at every rank count (2026-06-10).  The historical
         # multinode failures here (xsocr NaN at n3 / zeta=0.0 at n4, arts
         # timeout) were npb-cg APP bugs, fixed in the ocr-apps submodule:
         # in-place writes through DB_MODE_CONST acquisitions (update's p,
         # the zeta-carrying DB) and a missing ocrDbRelease before wiring the
         # verification EDT.  The arts-block flag (multinode_xsocr_only) was
         # REMOVED 2026-06-08 so Tier M reports the true verdict — do NOT
         # re-mask if this regresses.
         # class T (tiny: size=50, 3 iters) runs in <1s, so it stays fast enough
         # for xsocr at multinode and runs full 3-way.  (class S — the default —
         # made the multinode run ~36K small remote DBs/iter of synchronous
         # writeback-ACK round-trips: correct but ~67s n4 / ~170s RELAXED 2n, which is
         # why it used to be arts-only with a wide budget.)
    Case("hpgmg", "hpgmg", ["4","1"],
         scalar_re=r"\|\|error\|\|\s*=\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-4,
         multinode=True),
    Case("tempest", "tempest", [],
         scalar_re=r"CROSS-CHECKING NEIGHBOR DATA EXCHANGE\*(?:[ \t]*\n|[ \t]+|[A-Za-z*][^\n]*\n)*-?\d+[ \t]+-?\d+[ \t]+-?\d+(?:[ \t]*\n|[ \t]+|[A-Za-z*][^\n]*\n)*-?\d+[ \t]+-?\d+[ \t]+-?\d+(?:[ \t]*\n|[ \t]+|[A-Za-z*][^\n]*\n)*-?\d+[ \t]+-?\d+[ \t]+(-?\d+)",
         scalar_kind="int",
         multinode=True),
    Case("curvefit", "curvefit", [],
         # Completion marker: the leaf-segment count is a structural constant of
         # the fixed input, and aggregating it across the recursive fan-out would
         # require global mutable state (illegal in OCR — EDTs are stateless) or a
         # reduction DataBlock that the app does not build, so completion is the
         # strongest spec-compliant check here.
         scalar_re=r"SUCCESS", scalar_kind="bool",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("testlibs", "testlibs", [],
         scalar_re=r"Testing strlen of \w+ is (\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("graph500", "graph500", ["6","8","1","1"],
         scalar_re=r"nodes (\d+)", scalar_kind="int",
         multinode=True),
    Case("multigen", "multigen", [],
         scalar_re=r"End leaf1, result\s*=\s*(-?\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("multigen_2", "multigen_2", [],
         scalar_re=r"End leaf1, result\s*=\s*(-?\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("miniAMR_intel_chandra", "miniAMR_intel_chandra",
         ["--nx","4","--ny","4","--nz","4","--num_tsteps","2","--num_refine","3"],
         scalar_re=r"Done", scalar_kind="bool",
         multinode=True,  # 3-way: xsocr runs it at MN (old HCDist-timeout reason was stale)
         ),

    # --- rc-only sanity (no meaningful scientific scalar) ---
    Case("printf",           "printf",           [],
         scalar_re=r"Hello from mainEdt", scalar_kind="bool",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("quicksort",        "quicksort",        [],
         scalar_re=r"(\d+)\s*\n\s*(?:\[\d+\]\s*)?Sorting Finished", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("basicIO",          "basicIO",          ["0","10", BASIC_IO_DAT],
         scalar_re=r"BASICIO_CHK\s+(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("cache_offset",     "cache_offset",     [],
         scalar_re=r"CACHE_OFFSET_CHK\s+(\d+)", scalar_kind="int",
         multinode_skip="single-node design: file-scope dbGuids/dbPtrs arrays"),
    Case("highbw",           "highbw",           [],
         scalar_re=r"HIGHBW_WORK_SUM\s*=\s*(\d+)", scalar_kind="int",
         multinode=True),  # 3-way: xsocr runs it at MN (old startup-hang reason was stale)
    Case("task_priorities",  "task_priorities",  [],
         scalar_re=r"Hello from 9", scalar_kind="bool",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("dbctrl",           "dbctrl",           ["5","5","256"],
         # DB create/destroy control stress test: the destroy count is a static
         # function of (DEPTH, FANOUT) and the kernel computes no data answer.
         # Counting destroys across EDTs would need illegal global state, so the
         # completion/timing marker is the strongest spec-compliant check.
         scalar_re=r"Total time", scalar_kind="bool",
         multinode_skip="single-node design: file-scope template GUIDs + evtMap array"),
    Case("prodcon",          "prodcon",          [],
         scalar_re=r"MB/s", scalar_kind="bool",
         multinode_skip="single-node design: file-scope mapProdGuid/mapConsGuid"),
    Case("globalsum_cgShim",   "globalsum_cgShim",   [],
         scalar_re=r"CG0 T\d+\s+0 value\s+([0-9.]+)", scalar_kind="float", scalar_tol=1e-5,
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("globalsum_cgNoShim", "globalsum_cgNoShim", [],
         scalar_re=r"CG0 T100\s+0 value\s+([0-9.]+)", scalar_kind="float", scalar_tol=1e-5,
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("globalsum_pcg",      "globalsum_pcg",      [],
         scalar_re=r"CG0 T\d+\s+0 value\s+([0-9.]+)", scalar_kind="float", scalar_tol=1e-5,
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("stencil1D_sticky", "stencil1D_sticky", [],
         scalar_re=r"S3 i9 valu\s+([0-9.]+)", scalar_kind="float",
         scalar_tol=0,
         multinode=True),
    Case("reduction_intel",  "reduction_intel",  [],
         scalar_re=r"T300 i1\s+([0-9.]+)", scalar_kind="float",
         scalar_tol=0,
         multinode=True),
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
         scalar_re=r"LCS length:\s*(-?\d+)", scalar_kind="int",
         multinode=True),
    Case("LCS_all_db_distributed","LCS_all_db_distributed",[],
         scalar_re=r"LCS length:\s*(\d+)", scalar_kind="int",
         multinode=True),
    Case("LCS_shared",        "LCS_shared",       [],
         # distributed wavefront LCS result, self-validated against serial_lcs.
         scalar_re=r"LCS length:\s*(-?\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("RSBench_intel",             "RSBench_intel",             ["-l","100"],
         scalar_re=r"Lookups:", scalar_kind="bool",
         multinode_skip="no EDT affinity hints (intel variant); runs caller-rank only"),
    Case("RSBench_intel_sharedDB",    "RSBench_intel_sharedDB",    ["-l","100"],
         scalar_re=r"RS_CHECKSUM:\s+([0-9]+)", scalar_kind="int",
         multinode=True,
         ),
    Case("XSBench_intel",             "XSBench_intel",             ["-s","small","-g","10","-l","100"],
         scalar_re=r"XSBench grid checksum:\s+(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("XSBench_intel_sharedDB",    "XSBench_intel_sharedDB",    ["-s","small","-g","10","-l","100"],
         scalar_re=r"Workload\s+\(unit\):\s+(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only",
         ),
    Case("uts", "uts", ["-g","1","-t","1","-a","2","-d","7","-b","7","-r","220"],
         # T2L-family geometric tree at gen_mx=7 (4667 nodes, depth 36): same
         # search character as the built-in gen_mx=10 sample (39881 nodes) but
         # sized so the serialized ocr-vx message layer finishes well inside
         # the per-case wall budget.
         scalar_re=r"UTS Tree size\s*=\s*(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),

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
         scalar_kind="float", scalar_tol=1e-9,
         multinode=True),
    Case("cholesky", "cholesky",
         ["--ds","50","--ts","10","--fi",CHOLESKY_INPUT],
         scalar_re=r"CHOLESKY trace\s*=\s*([0-9.eE+-]+)",
         scalar_kind="float", scalar_tol=1e-9,
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    # xeonNumaSize: needs at least one action flag or it just prints help.
    # -dcpu just enumerates CPUs and fires the DONE! marker fast.
    Case("xeonNumaSize", "xeonNumaSize", ["-dcpu"],
         scalar_re=r"DONE!", scalar_kind="bool",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("stream_dist", "stream_dist", [],
         scalar_re=r"STREAM checksum: a\[0\] = ([0-9.eE+-]+)", scalar_kind="float", scalar_tol=1e-9,
         multinode=True),

    # --- previously-SKIPped: genuine arts-side runtime bugs (report, don't fix) ---
    Case("miniAMR_intel_bryan", "miniAMR_intel_bryan", [],
         # checksum strengthening abandoned: xsocr does not emit the per-block
         # checksum and the arts value is non-deterministic across ranks; the
         # completion marker is the strongest portable check here.
         scalar_re=r"miniAMR complete", scalar_kind="bool",
         multinode=True,
         ),
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
         multinode=True,
         multinode_args={2: ["2", "1", "1", "16", "5"],
                         3: ["3", "1", "1", "16", "5"],
                         4: ["4", "1", "1", "16", "5"],
                         8: ["8", "1", "1", "16", "5"],
                         16: ["16", "1", "1", "16", "5"]},
         ),

    Case("stream", "stream", [],
         scalar_re=r"STREAM_RESULT a\[0\] = ([0-9.eE+-]+)", scalar_kind="float", scalar_tol=1e-3,
         multinode_skip="single-node only; stream_dist is the multinode variant"),

    # --- previously-SKIPped: by-design stress test, cannot be tamed ---
    # miniAMR_forkbomb: previously SKIP-STRESS.  Vendor source hardcoded
    # timestep=1500 + maxRefLvl=3 + no ocrShutdown().  Fixed in
    # third_party/ocr-apps/.../forkbomb/mainOCR.c to honor --num_tsteps /
    # --num_refine and add a shutdown barrier on the initial block count.
    Case("miniAMR_forkbomb", "miniAMR_forkbomb",
         ["--nx","1","--ny","1","--nz","1","--num_tsteps","3","--num_refine","0"],
         scalar_re=r"BLOCK 0 finished", scalar_kind="bool",
         multinode_skip="single-node design: file-scope volatile statics for shutdown barrier"),

    # --- SAR (revived via CMake crlibm bootstrap + datagen integration).
    # All sizes build all three backends; per-size datasets are embedded
    # via the .incbin pipeline.  Single-node only (no EDT affinity hints). ---
    Case("sar_tiny",   "sar_tiny",   [],
         scalar_re=r"SAR detects:\s*(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("sar_small",  "sar_small",  [],
         scalar_re=r"SAR detects:\s*(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("sar_medium", "sar_medium", [],
         scalar_re=r"SAR detects:\s*(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("sar_large",  "sar_large",  [],
         scalar_re=r"SAR detects:\s*(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
]

# Tier B — 3-way (baseline ↔ xsocr ↔ arts) scalar comparison.
TIER_B: list[Case] = [
    Case(
        name="nqueens_B", ocr_base="nqueens", args=["6","2"],
        scalar_re=r"sols:\s*(\d+)", scalar_kind="int", scalar_tol=0,
        baseline=BaselineSpec(
            bin="nqueens_omp", args=["6"],
            scalar_re=r"sols:\s*(\d+)", scalar_kind="int", scalar_tol=0,
        ),
    ),
    Case(
        name="CoMD_sdsc2_B", ocr_base="CoMD_sdsc2",
        args=["-x","4","-y","4","-z","4","-N","2"],
        scalar_re=r"Final energy\s*:\s*([\-+0-9.eE]+)",
        scalar_kind="float", scalar_tol=1e-4,
        baseline=BaselineSpec(
            bin="CoMD_mpi_omp", args=["-x","4","-y","4","-z","4","-N","2"],
            np=1, force_mpirun=True,
            scalar_re=r"Final energy\s*:\s*([\-+0-9.eE]+)",
            scalar_kind="float", scalar_tol=1e-4,
        ),
    ),
    Case(
        name="Stencil2D_B", ocr_base="Stencil2D_intel_chandra", args=[],
        scalar_re=r"Solution validates", scalar_kind="bool",
        baseline=BaselineSpec(
            bin="Stencil2D_omp", args=["4","5","64"],
            scalar_re=r"Solution validates", scalar_kind="bool",
        ),
    ),
    Case(
        name="hpgmg_B", ocr_base="hpgmg", args=["4","1"],
        scalar_re=r"\|\|error\|\|\s*=\s*([\-+0-9.eE]+)",
        scalar_kind="float", scalar_tol=1e-3,
        baseline=BaselineSpec(
            bin="hpgmg_mpi_omp", args=["4","1"], np=1, force_mpirun=True,
            scalar_re=r"\|\|error\|\|\s*=\s*([\-+0-9.eE]+)",
            scalar_kind="float", scalar_tol=1e-3,
        ),
    ),
    Case(
        name="npb_cg_B", ocr_base="npb_cg", args=[],
        scalar_re=r"zeta\s*=\s*([\-+0-9.eE]+)",
        scalar_kind="float", scalar_tol=1e-10,
        baseline=BaselineSpec(
            bin="npb_cg_omp", args=[],
            scalar_re=r"zeta\s*=\s*([\-+0-9.eE]+)",
            scalar_kind="float", scalar_tol=1e-10,
        ),
    ),
    # miniAMR_intel_B removed from Tier B: baseline miniAMR_mpi only prints
    # per-variable checksums under --report_diffusion, while OCR prints a
    # grand total summed over variables.  The two are not the same scalar,
    # and summing baseline's per-variable prints after the fact is fragile.
    # xsocr↔arts comparison still runs in Tier A.
    # XSBench_B removed from Tier B: there is NO correctness metric shared by
    # the OCR app and any OMP/MPI baseline.  The OCR XSBench prints only its
    # own "XSBench grid checksum" (added in refactored/ocr/intel Main.c; no
    # OMP/MPI variant computes it), while the OMP/MPI baselines print only the
    # canonical "Verification checksum" vhash (under -DVERIFICATION) — which the
    # OCR refactor never wires up (building it with VERIFICATION still emits no
    # vhash).  The old comparison matched "Workload (unit)" (the OCR-side echo
    # of the -l input = 100) against the baseline's "Lookups" (also the -l
    # input) — i.e. the input parameter, not a computed result.  Aligning on a
    # real metric would require app-source surgery (port the distributed
    # verification hash into the OCR app, or add the grid checksum to a
    # baseline).  XSBench correctness is covered by XSBench_intel in Tier A
    # (grid-checksum consensus across the 7 arts variants + xsocr + ocr-vx).
    Case(
        name="RSBench_B", ocr_base="RSBench_intel",
        args=["-l","100"],
        scalar_re=r"Lookups:", scalar_kind="bool",
        baseline=BaselineSpec(
            bin="RSBench_omp",
            args=["-t","4","-l","100"],
            scalar_re=r"Lookups:", scalar_kind="bool",
        ),
    ),
]


# Statically-derived absolute answers, verified against deterministic agreed
# output (xsocr == arts on a fixed, order-independent input).  Pinning them
# makes the harness reject a correlated regression where BOTH runtimes compute
# the same wrong value, not only a cross-runtime divergence.  Floats are matched
# with a relaxed relative tolerance (see _expect_ok), so reduced-precision
# entries here are safe.
_EXPECT: dict[str, str] = {
    "fibonacci": "55", "nqueens": "4", "smithwaterman": "1460", "triangle": "5072",
    "basicIO": "1", "highbw": "2048", "multigen": "121393", "multigen_2": "3524578",
    "uts": "4667", "XSBench_intel": "10725709712928718927", "XSBench_intel_sharedDB": "100",
    "sar_tiny": "12", "sar_small": "458", "sar_medium": "1991", "sar_large": "6523",
    "CoMD_sdsc": "-1.166058121223", "CoMD_sdsc2": "-1.166063027842",
    "CoMD_intel_chandra_tiled": "-1.166063",
    "cholesky": "50", "hpcg_intel": "0.001279", "hpcg_intel_Eager": "0.001279",
    "hpgmg": "2.97878e-06", "miniAMR_intel": "710400", "npb_cg": "7.85534",
    "p2p": "1188", "reduction_intel": "7775", "stencil1D_sticky": "1",
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
for _c in TIER_A + TIER_B:
    if _c.name in _EXPECT and not _c.expect:
        _c.expect = _EXPECT[_c.name]


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    rc: int
    wall: float
    stdout: str
    log_path: str


class Runner:
    def __init__(self, mem_gb: int, timeout: int, logdir: Path):
        self.mem_kb = mem_gb * 1024 * 1024
        self.mem_gb = mem_gb
        self.timeout = timeout
        self.logdir = logdir
        self.logdir.mkdir(parents=True, exist_ok=True)
        # Detect systemd-run user-scope availability for hard RSS cap.
        # ulimit -v alone caps virtual address space, which mmap-heavy apps
        # can exceed by inflating RSS without the kernel killing them; this
        # has previously taken the host to OOM. cgroup memory.max via
        # systemd-run is a kernel-enforced RSS cap.
        # On a large-memory target, skip memory capping entirely: the box has
        # ample RAM, and ulimit -v breaks mmap-reserving allocators (mimalloc
        # reserves virtual address space far above its actual RSS, so a virtual
        # cap rejects allocations the host could easily satisfy).
        self.cap_memory = (TARGET != 'server')
        self.cgroup_ok = False
        if self.cap_memory:
            try:
                r = subprocess.run(
                    ["systemd-run", "--user", "--scope", "--quiet",
                     "-p", "MemoryMax=64M", "--", "/bin/true"],
                    capture_output=True, timeout=5)
                self.cgroup_ok = (r.returncode == 0)
            except (OSError, subprocess.TimeoutExpired):
                self.cgroup_ok = False
            if not self.cgroup_ok:
                print("WARNING: systemd-run --user not available; "
                      "falling back to ulimit -v only (RSS not capped).")
        self.mem_prefix = (f"ulimit -v {self.mem_kb} && "
                           if self.cap_memory else "")

        # Stage configs + fixtures once.
        shutil.copy2(ARTS_CFG, APPS_DIR / "arts.cfg")
        shutil.copy2(ARTS_CFG, BASE_DIR / "arts.cfg")
        if not Path(BASIC_IO_DAT).exists():
            # 10 deterministic u64 values for basicIO (offset=0, XOR=0^1^...^9=1)
            Path(BASIC_IO_DAT).write_text(
                "\n".join(str(i) for i in range(10)) + "\n")
        # Stage a 50x50 identity matrix for cholesky (SPD, trivial decomp).
        if not Path(CHOLESKY_INPUT).exists():
            with open(CHOLESKY_INPUT, "w") as f:
                for r in range(50):
                    f.write(" ".join("1.0" if c == r else "0.0"
                                     for c in range(50)) + "\n")

    def _run(self, cmd: str, env: dict[str, str], logfile: Path,
             wall_timeout: int = 0) -> RunResult:
        # Pristine-start isolation: reap leftover ranks / MPI launchers and
        # remove leaked MPI/UCX /dev/shm segments so no prior case can pollute
        # this one (the suite is sequential, so nothing live is touched).
        self._cleanup()
        t0 = time.time()
        if self.cgroup_ok:
            # Wrap inner shell in a transient user scope with cgroup memory.max.
            # MemoryMax kills the entire scope (and all descendants, including
            # ARTS workers/SSH spawns) when the cgroup RSS exceeds the cap,
            # protecting the host from harness-induced OOM. MemoryHigh adds a
            # softer throttle threshold.
            cap = f"{self.mem_gb}G"
            soft = f"{max(1, self.mem_gb * 3 // 4)}G"
            wrapped = (
                f"systemd-run --user --scope --quiet "
                f"-p MemoryMax={cap} -p MemoryHigh={soft} "
                f"-- /bin/bash -c {shlex_quote(cmd)}"
            )
        else:
            wrapped = cmd
        # Stream stdout/stderr directly to the per-case log file.  Earlier
        # versions used capture_output=True which made Python buffer the
        # full child stdout in RAM; ARTS verbose builds can emit tens of MB
        # per run, and across 58 cases that bloats the Python harness to
        # multi-GB RSS and starves the host (swap=0 environment).  File
        # redirection caps Python memory; we read the (small) log back
        # afterwards.
        try:
            with open(logfile, "wb") as outf:
                proc = subprocess.run(
                    wrapped, shell=True, executable="/bin/bash",
                    stdout=outf, stderr=subprocess.STDOUT,
                    env=env, timeout=(wall_timeout or self.timeout) + 10,
                )
            rc = proc.returncode
            # Read back only the tail (max 256 KB) so scalar regex extraction
            # works without re-loading huge logs into Python memory.
            try:
                size = logfile.stat().st_size
                with open(logfile, "rb") as f:
                    if size > 262144:
                        f.seek(size - 262144)
                    out = f.read().decode("utf-8", errors="replace")
            except OSError:
                out = ""
        except subprocess.TimeoutExpired:
            rc = 124
            try:
                with open(logfile, "rb") as f:
                    out = f.read().decode("utf-8", errors="replace")
            except OSError:
                out = ""
            out += "\n[HARNESS: subprocess wall-timeout]"
        wall = time.time() - t0
        return RunResult(rc=rc, wall=wall, stdout=out, log_path=str(logfile))

    def run_ocr(self, case_name: str, bin_name: str, args: list[str], backend: str,
               suffix: str = "") -> RunResult:
        # When backend=="arts" and suffix is given, exec <bin_name>_arts_<suffix>.
        if backend == "arts" and suffix:
            binary = f"{bin_name}_arts_{suffix}"
            log_tag = f"arts_{suffix}"
        else:
            binary = f"{bin_name}_{backend}"
            log_tag = backend
        logfile = self.logdir / f"{case_name}.{log_tag}.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        if backend == "xsocr":
            env["OCR_CONFIG"] = str(XSOCR_CFG)
        # arts self-pins (hwloc); the xsocr reference needs taskset to occupy
        # the same cores instead of floating across all logical CPUs.
        pin = f"{_PIN_SINGLE} " if backend == "xsocr" else ""
        cmd = (
            f"cd {APPS_DIR} && {self.mem_prefix}"
            f"timeout -k 1 {self.timeout} {pin}./{binary} " + " ".join(args)
        )
        result = self._run(cmd, env, logfile)
        self._reap_exe(APPS_DIR / binary)
        return result

    # --- Multinode runners (Tier M) ---

    # Per-target multinode config paths, generated from MN_RANKS.  The *_io
    # variants (sender/receiver-heavy IO-forwarding stress) are arts-only — no
    # xsocr/MPI equivalent, so they are absent from the xsocr map.
    _ARTS_MN_CFGS = {n: f"local/{TARGET}/{_cfg_name(n)}" for n in MN_RANKS}
    _XSOCR_MN_CFGS = {n: f"mpi/{TARGET}/{_cfg_name(n)}"
                      for n in MN_RANKS if isinstance(n, int)}

    # Per-run unique TCP port base for arts multinode runs.  A straggler from
    # the previous case (a rank still releasing its listen socket, or an orphan
    # from a timed-out run) would poison every later case that binds the same
    # range; the runtime lets the environment override any config key, so each
    # run gets its own range.
    #
    # The base MUST stay BELOW the kernel ephemeral range (32768-60999): a base
    # inside it randomly collides with the source port the kernel assigns to an
    # outgoing connection (the launcher's inter-rank connects), and bind() then
    # fails instantly (exit 255).  A monotonic counter climbs into that range
    # over a long matrix, so cycle within a fixed sub-ephemeral window instead.
    # Runs are serialized (one MN runner at a time), so a base is free for reuse
    # long before the cycle returns to it.  Each run needs nodes*port_count
    # consecutive ports (<= 16 nodes * 2 _io ports = 32 ports); _PORT_STEP
    # exceeds that span so adjacent bases never overlap, and _PORT_BASE_HI keeps
    # the whole span clear of 32768.
    _PORT_BASE_LO = 20000
    _PORT_BASE_HI = 32700
    _PORT_STEP = 48
    _arts_port_ctr = itertools.count(0)

    @staticmethod
    def _reap_exe(exe_path: Path) -> None:
        """SIGKILL every process whose executable is exe_path.

        Matching on /proc/<pid>/exe is the only reliable way to reap ARTS
        rank processes: comm is truncated to 15 chars (pkill -x misses) and
        pattern matching the command line (pkill -f) can match the caller
        itself."""
        target = str(exe_path)
        for pid_dir in Path("/proc").glob("[0-9]*"):
            try:
                if os.readlink(pid_dir / "exe") == target:
                    os.kill(int(pid_dir.name), signal.SIGKILL)
            except OSError:
                continue

    @staticmethod
    def _cleanup() -> None:
        """Pristine-start isolation, run before EVERY test execution.

        The suite is sequential, but the host carries cross-test state that the
        per-run _reap_exe (one binary, after the fact) does not clear:
          1. Orphaned benchmark ranks or MPI launchers/proxies left by a
             SIGKILLed / timed-out / crashed prior run, still holding CPU,
             ports, or memory.
          2. MPI/UCX shared-memory segments leaked into /dev/shm: when an
             MPI/UCX rank is SIGKILLed its cleanup is skipped, so the segment
             persists.  These accumulate across a long suite and can starve a
             later run (the "roaming" reference-runtime timeout symptom).
        Reaping is by /proc/<pid>/exe (comm is truncated to 15 chars so pkill
        -x misses; pkill -f can self-match the harness).  Only benchmark
        executables (under APPS_DIR / BASE_DIR), known MPI launchers, and
        MPI/UCX-owned /dev/shm patterns are touched — never unrelated state."""
        apps, base = str(APPS_DIR), str(BASE_DIR)
        launchers = {"mpirun", "mpiexec", "mpiexec.hydra",
                     "hydra_pmi_proxy", "hydra_bstrap_proxy"}
        self_pid = os.getpid()
        for pid_dir in Path("/proc").glob("[0-9]*"):
            try:
                pid = int(pid_dir.name)
                if pid == self_pid:
                    continue
                exe = os.readlink(pid_dir / "exe")
            except (OSError, ValueError):
                continue
            if (exe.startswith(apps) or exe.startswith(base)
                    or os.path.basename(exe) in launchers):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    continue
        shm = Path("/dev/shm")
        for pat in ("sm_segment.*", "ucx_shm_*", "*ompi*",
                    "vader_*", "hydra_*", "psm*"):
            for f in shm.glob(pat):
                try:
                    f.unlink()
                except OSError:
                    continue

    def run_arts_mn(self, case_name: str, bin_name: str, args: list[str],
                    nodes: int, timeout: int = 0, suffix: str = "") -> RunResult:
        """Run arts at N nodes (self-fork launcher via cfg).

        When suffix is set, exec <bin_name>_arts_<suffix> instead of <bin_name>_arts.
        """
        to = timeout or self.timeout
        cfg_name = self._ARTS_MN_CFGS[nodes]
        cfg_src = REPO / "configs" / cfg_name
        shutil.copy2(cfg_src, APPS_DIR / "arts.cfg")  # arts reads ./arts.cfg
        log_tag = f"arts_{suffix}_mn{nodes}" if suffix else f"arts_mn{nodes}"
        logfile = self.logdir / f"{case_name}.{log_tag}.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        base = self._PORT_BASE_LO + (
            next(self._arts_port_ctr) * self._PORT_STEP
        ) % (self._PORT_BASE_HI - self._PORT_BASE_LO)
        # The override must carry the same port COUNT as the cfg it replaces:
        # the count doubles as the per-node parallel-connection count
        # (port_count = default_ports_count), so a mismatch breaks the
        # startup handshake.  The *_io variants use two ports (2 sender / 2
        # receiver threads); every other local cfg uses one.
        if isinstance(nodes, str):  # *_io variant
            env["default_ports"] = f"[{base}-{base + 1}]"
        else:
            env["default_ports"] = str(base)
        arts_bin = f"{bin_name}_arts_{suffix}" if suffix else f"{bin_name}_arts"
        cmd = (
            f"cd {APPS_DIR} && {self.mem_prefix}"
            f"timeout -k 1 {to} ./{arts_bin} " + " ".join(args)
        )
        result = self._run(cmd, env, logfile, wall_timeout=to)
        # A timed-out run can leave rank processes behind (a hung rank can
        # survive SIGTERM); reap them so they cannot interfere with later
        # cases or hold CPU.
        self._reap_exe(APPS_DIR / arts_bin)
        # Restore single-node cfg for subsequent single-node runs
        shutil.copy2(ARTS_CFG, APPS_DIR / "arts.cfg")
        return result

    def run_xsocr_mpi(self, case_name: str, bin_name: str, args: list[str],
                      np: int, timeout: int = 0) -> RunResult:
        """Run xsocr at N MPI ranks (mpirun launcher)."""
        to = timeout or self.timeout
        logfile = self.logdir / f"{case_name}.xsocr_mpi{np}.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        xsocr_cfg = REPO / "configs" / self._XSOCR_MN_CFGS[np]
        cmd = (
            f"cd {APPS_DIR} && {self.mem_prefix}"
            f"timeout -k 1 {to} {mpirun_prefix(np)} {_pin_wrap(_TPN[np])} "
            f"./{bin_name}_xsocr -ocr:cfg {xsocr_cfg} "
            + " ".join(args)
        )
        result = self._run(cmd, env, logfile, wall_timeout=to)
        self._reap_exe(APPS_DIR / f"{bin_name}_xsocr")
        return result

    # ocr-vx TBB compute parallelism per node count (target-selected, see
    # module-level _OCRVX_TBB).  Sized so the *active* thread budget matches the
    # arts/xsocr configs: the runtime's shutdown barrier is a TBB task that
    # blocks (zero CPU) permanently occupying one parallelism slot, so effective
    # compute width is P-1 and P=1 deadlocks outright — hence P = per-node
    # budget - 1.
    _OCRVX_TBB_THREADS = _OCRVX_TBB

    def run_ocrvx_mpi(self, case_name: str, bin_name: str, args: list[str],
                      np: int = 1, timeout: int = 0) -> RunResult:
        """Run ocrvx binary; np > 1 uses mpirun (ocr-vx MPI transport)."""
        to = (timeout or self.timeout) * _OCRVX_TIMEOUT_MULT
        suffix = f"_ocrvx_mpi{np}" if np > 1 else "_ocrvx"
        logfile = self.logdir / f"{case_name}{suffix}.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        env["OCRVX_NUM_THREADS"] = str(self._OCRVX_TBB_THREADS[np])
        if np > 1:
            launcher = f"{mpirun_prefix(np)} {_pin_wrap(_TPN[np])} ./{bin_name}_ocrvx"
        else:
            launcher = f"{_PIN_SINGLE} ./{bin_name}_ocrvx"
        cmd = (
            f"cd {APPS_DIR} && {self.mem_prefix}"
            f"timeout -k 1 {to} {launcher} " + " ".join(args)
        )
        result = self._run(cmd, env, logfile, wall_timeout=to)
        self._reap_exe(APPS_DIR / f"{bin_name}_ocrvx")
        return result

    def run_baseline(self, case_name: str, spec: BaselineSpec) -> RunResult:
        logfile = self.logdir / f"{case_name}.baseline.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        if spec.np > 1 or spec.force_mpirun:
            # Multinode: per-rank disjoint blocks; single-rank force_mpirun:
            # confine the one rank to cores 0..N-1 like the other references.
            pin = f"{_pin_wrap(_TPN[spec.np])} " if spec.np in _TPN else f"{_PIN_SINGLE} "
            launcher = f"{mpirun_prefix(spec.np)} {pin}./{spec.bin}"
        else:
            launcher = f"{_PIN_SINGLE} ./{spec.bin}"
        cmd = (
            f"cd {BASE_DIR} && {self.mem_prefix}"
            f"timeout {self.timeout} {launcher} " + " ".join(spec.args)
        )
        return self._run(cmd, env, logfile)


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

@dataclass(frozen=True)
class Runtime:
    key: str          # column id, e.g. "mrnew_lazy", "xsocr", "ocrvx", "baseline"
    label: str        # display
    kind: str         # "arts" | "xsocr" | "ocrvx" | "baseline"
    suffix: str = ""  # arts variant binary suffix (empty for non-arts)

ARTS_VARIANTS = [
    "mrnew_eager", "mrnew_lazy",
    "mrsw_eager", "mrsw_lazy",
    "mrmw",
    "lock_eager", "lock_lazy",
]
RUNTIMES: list[Runtime] = (
    [Runtime(s, f"arts_{s}", "arts", s) for s in ARTS_VARIANTS]
    + [Runtime("xsocr", "xsocr", "xsocr"),
       Runtime("ocrvx", "ocrvx", "ocrvx")]
)

NODE_CONFIGS: list = ["1n"] + MN_RANKS   # e.g. ["1n", 2, 4, 8, 16, "2n_io", ...]


def runtime_eligible(rt: Runtime, case: Case, node: object) -> bool:
    """Return True iff runtime rt should be invoked for (case, node)."""
    is_io = isinstance(node, str) and node.endswith("_io")
    single = node in ("1n", 1)
    # _io configs are arts-only (no xsocr/ocrvx/baseline MPI equivalent).
    if rt.kind in ("xsocr", "ocrvx", "baseline") and is_io:
        return False
    if not single:
        # Multinode eligibility.
        if case.multinode_skip or not case.multinode or not case.scalar_re:
            return False
        if rt.kind == "xsocr" and (case.arts_only or case.multinode_arts_only):
            return False
        if rt.kind == "ocrvx" and case.ocrvx_skip:
            return False
    if rt.kind == "ocrvx" and not (APPS_DIR / f"{case.ocr_base}_ocrvx").exists():
        return False
    if rt.kind == "baseline" and case.baseline is None:
        return False
    return True


def run_runtime(runner: Runner, rt: Runtime, case: Case, node: object) -> dict:
    """Run a single (runtime, case, node) cell and return a CellRun dict.

    state is "OK?" (ran and scalar extracted), "FAIL" (non-zero rc or scalar
    miss), or "N/A" (excluded by eligibility).  Consensus in Task 5 resolves
    "OK?" into "OK", "DISAGREE", or "NO-CONSENSUS".
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
                               suffix=rt.suffix)
        elif rt.kind == "xsocr":
            r = runner.run_ocr(case.name, case.ocr_base, args, "xsocr")
        elif rt.kind == "ocrvx":
            r = runner.run_ocrvx_mpi(case.name, case.ocr_base, args)
        else:  # baseline
            r = runner.run_baseline(case.name, case.baseline)
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
    # Classify.
    wall_r = round(r.wall, 3)
    if r.rc == 124:
        return {"state": "FAIL", "rc": 124, "wall": wall_r, "scalar": None}
    if r.rc != 0:
        return {"state": "FAIL", "rc": r.rc, "wall": wall_r, "scalar": None}
    val = _pull(r.stdout, case.scalar_re, case.scalar_kind)
    if val is None:
        return {"state": "FAIL", "rc": 0, "wall": wall_r, "scalar": None}
    return {"state": "OK?", "rc": 0, "wall": wall_r, "scalar": val}


def consensus(cells: dict, case: Case) -> tuple:
    """Cluster scalars from all "OK?" cells; largest cluster wins.

    Returns (consensus_value, dict[key -> final_state]) where final_state is
    one of "OK", "DISAGREE", "FAIL", "N/A", or "NO-CONSENSUS".
    """
    final: dict = {}
    for k, c in cells.items():
        if c["state"] == "N/A":
            final[k] = "N/A"
        elif c["state"] == "FAIL":
            final[k] = "FAIL"
    # Candidates are cells that ran and produced a scalar.
    cand = {k: c for k, c in cells.items()
            if c["state"] == "OK?" and c["scalar"] is not None}
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
            final[k] = st
    consensus_val = None if tie else top["rep"]
    return (consensus_val, final)


def run_matrix(runner: Runner, cases: list, only: set, no_baseline: bool) -> dict:
    """Run every (case, node-config, runtime) cell; compute consensus per (case, node).

    Returns a nested dict: report[case_name][node_str] = {consensus, cells}.
    """
    report: dict = {}
    glyph = {
        "OK": "·", "DISAGREE": "X", "FAIL": "!",
        "N/A": "-", "NO-CONSENSUS": "?",
    }
    for c in cases:
        if only and c.name not in only:
            continue
        report[c.name] = {}
        rts = list(RUNTIMES)
        if c.baseline is not None and not no_baseline:
            rts = rts + [Runtime("baseline", "baseline", "baseline")]
        for node in NODE_CONFIGS:
            cells: dict = {rt.key: run_runtime(runner, rt, c, node)
                           for rt in rts}
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

    print("\nAll selftest assertions passed.")



# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", default="build_release_mrnew_lazy",
                   help="Build directory containing apps and configs")
    p.add_argument("--target", default="laptop", choices=["laptop", "server"],
                   help="Machine geometry: laptop (14-thread) or server (48-thread)")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--only", type=str, default="")
    p.add_argument("--mem-gb", type=int, default=4)
    # 90s default: MRSW serializes writes (single-writer), so heavy single-node
    # apps like sar_large legitimately run longer than a 60s budget; passing
    # cases still return fast, so only genuine hangs wait the full budget.
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--node", type=str, default="",
                   help="Restrict to a single node-config, e.g. --node 1n")
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
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    logdir = LOGS_ROOT / ts
    runner = Runner(args.mem_gb, args.timeout, logdir)

    cases = TIER_A + TIER_B
    report = run_matrix(runner, cases, only, args.no_baseline)
    emit(report, logdir, cases)




if __name__ == "__main__":
    main()
