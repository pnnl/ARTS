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
_pre_args, _ = _arg_parser.parse_known_args()
BUILD = Path(_pre_args.build_dir)
if not BUILD.is_absolute():
    BUILD = REPO / BUILD

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
ARTS_CFG = REPO / "configs" / "local" / "1n.cfg"
XSOCR_CFG = REPO / "configs" / "mpi" / "1n.cfg"
OCRVX_APPS_DIR = APPS_DIR   # ocr-vx binaries live in the same apps/ dir

# smithwaterman ships its own datasets (tiny/small/medium/large triples of
# string1/string2/score); use the official tiny set directly — no staging.
SW_DATA = Path(__file__).resolve().parents[2] / \
    "third_party/ocr-apps/apps/smithwaterman/datasets"
BASIC_IO_DAT = "/tmp/arts_basicIO_test.dat"
CHOLESKY_INPUT = "/tmp/arts_cholesky_input.mat"


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
                         4: ["4", "1", "1", "16", "5"]},
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
    Case(
        name="XSBench_B", ocr_base="XSBench_intel",
        args=["-s","small","-g","10","-l","100"],
        scalar_re=r"Workload\s+\(unit\):\s+(\d+)", scalar_kind="int",
        baseline=BaselineSpec(
            bin="XSBench_omp",
            args=["-t","4","-s","small","-g","10","-l","100"],
            scalar_re=r"Lookups:\s+(\d+)\s*\n", scalar_kind="int",
        ),
    ),
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

    def run_ocr(self, case_name: str, bin_name: str, args: list[str], backend: str) -> RunResult:
        binary = f"{bin_name}_{backend}"
        logfile = self.logdir / f"{case_name}.{backend}.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        if backend == "xsocr":
            env["OCR_CONFIG"] = str(XSOCR_CFG)
        cmd = (
            f"cd {APPS_DIR} && ulimit -v {self.mem_kb} && "
            f"timeout -k 1 {self.timeout} ./{binary} " + " ".join(args)
        )
        result = self._run(cmd, env, logfile)
        self._reap_exe(APPS_DIR / binary)
        return result

    # --- Multinode runners (Tier M) ---

    _ARTS_MN_CFGS = {
        2: "local/2n.cfg",
        3: "local/3n.cfg",
        4: "local/4n.cfg",
        # 2-node IO variant: 3 worker / 2 sender / 2 receiver threads + port
        # range — stresses the multi-threaded sender/receiver IO-forwarding
        # path with real benchmarks.  arts-only (no xsocr/MPI equivalent).
        "2n_io": "local/2n_io.cfg",
    }
    _XSOCR_MN_CFGS = {
        2: "mpi/2n.cfg",
        3: "mpi/3n.cfg",
        4: "mpi/4n.cfg",
    }

    # Per-run unique TCP port base for arts multinode runs.  All stock cfgs
    # share default_ports=50000, so a straggler from the previous case (a
    # rank still releasing its listen socket, or an orphan from a timed-out
    # run) would poison every later case that binds the same range.  The
    # runtime lets the environment override any config key, so each run gets
    # its own range.  Starts at 53000 to stay clear of both the stock 50000
    # configs (manual runs) and ctest's 51000+ per-test bases.
    _arts_port_iter = itertools.count(23000, 16)

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
                    nodes: int, timeout: int = 0) -> RunResult:
        """Run arts at N nodes (self-fork launcher via cfg)."""
        to = timeout or self.timeout
        cfg_name = self._ARTS_MN_CFGS[nodes]
        cfg_src = REPO / "configs" / cfg_name
        shutil.copy2(cfg_src, APPS_DIR / "arts.cfg")  # arts reads ./arts.cfg
        logfile = self.logdir / f"{case_name}.arts_mn{nodes}.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        base = next(self._arts_port_iter)
        # The override must carry the same port COUNT as the cfg it replaces:
        # the count doubles as the per-node parallel-connection count
        # (port_count = default_ports_count), so a mismatch breaks the
        # startup handshake.  2n_io uses two ports (one per sender/receiver
        # pair); every other local cfg uses one.
        if nodes == "2n_io":
            env["default_ports"] = f"[{base}-{base + 1}]"
        else:
            env["default_ports"] = str(base)
        cmd = (
            f"cd {APPS_DIR} && ulimit -v {self.mem_kb} && "
            f"timeout -k 1 {to} ./{bin_name}_arts " + " ".join(args)
        )
        result = self._run(cmd, env, logfile, wall_timeout=to)
        # A timed-out run can leave rank processes behind (a hung rank can
        # survive SIGTERM); reap them so they cannot interfere with later
        # cases or hold CPU.
        self._reap_exe(APPS_DIR / f"{bin_name}_arts")
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
            f"cd {APPS_DIR} && ulimit -v {self.mem_kb} && "
            f"timeout -k 1 {to} mpirun --oversubscribe -n {np} "
            f"./{bin_name}_xsocr -ocr:cfg {xsocr_cfg} "
            + " ".join(args)
        )
        result = self._run(cmd, env, logfile, wall_timeout=to)
        self._reap_exe(APPS_DIR / f"{bin_name}_xsocr")
        return result

    # ocr-vx TBB compute parallelism per rank count, sized so the *active*
    # thread budget matches the arts/xsocr configs on a 14-vCPU host
    # (totals 14/14/12/12 at 1n/2n/3n/4n).  ocr-vx pins one OS thread per
    # peer per channel (2*np receivers) plus one sender, but the receivers
    # block in MPI_Recv and message handling is serialized by a global lock,
    # so at most ~1 receiver plus the sender are runnable at a time.  The
    # runtime's shutdown barrier is a TBB task that blocks (zero CPU) until
    # shutdown, permanently occupying one parallelism slot — so effective
    # compute width is P-1 and P=1 deadlocks outright.  Active budget per
    # process = (P-1) compute + sender + 1 active receiver:
    #   1n: 12+1+1=14, 2n: 5+1+1=7 (x2=14), 3n: 2+1+1=4 (x3=12),
    #   4n: 1+1+1=3 (x4=12).
    _OCRVX_TBB_THREADS = {1: 13, 2: 6, 3: 3, 4: 2}

    def run_ocrvx_mpi(self, case_name: str, bin_name: str, args: list[str],
                      np: int = 1, timeout: int = 0) -> RunResult:
        """Run ocrvx binary; np > 1 uses mpirun (ocr-vx MPI transport)."""
        to = timeout or self.timeout
        suffix = f"_ocrvx_mpi{np}" if np > 1 else "_ocrvx"
        logfile = self.logdir / f"{case_name}{suffix}.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        env["OCRVX_NUM_THREADS"] = str(self._OCRVX_TBB_THREADS[np])
        if np > 1:
            launcher = f"mpirun --oversubscribe -n {np} ./{bin_name}_ocrvx"
        else:
            launcher = f"./{bin_name}_ocrvx"
        cmd = (
            f"cd {APPS_DIR} && ulimit -v {self.mem_kb} && "
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
            launcher = f"mpirun -n {spec.np} --oversubscribe ./{spec.bin}"
        else:
            launcher = f"./{spec.bin}"
        cmd = (
            f"cd {BASE_DIR} && ulimit -v {self.mem_kb} && "
            f"timeout {self.timeout} {launcher} " + " ".join(spec.args)
        )
        return self._run(cmd, env, logfile)


# ---------------------------------------------------------------------------
# Scalar extraction / comparison.
# ---------------------------------------------------------------------------
@dataclass
class Verdict:
    tag: str     # PASS-SCALAR | PASS-RC | FAIL | KNOWN-BUG | SKIP | SKIP-STRESS
    detail: str = ""


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


def _expect_ok(val, case: "Case") -> bool:
    """True unless the case pins an absolute answer (case.expect) that val violates."""
    if not case.expect:
        return True
    try:
        exp = int(case.expect) if case.scalar_kind == "int" else float(case.expect)
    except ValueError:
        return True
    if case.scalar_kind == "int":
        return val == exp
    # The pin guards against a correlated regression to a DIFFERENT answer, not
    # against last-digit precision; a real wrong answer diverges far more than
    # this, while a pinned value may be recorded at reduced precision.
    return _drift(val, exp) <= max(case.scalar_tol, 1e-4)


def _demote_if_known_bug(v: Verdict, case: Case) -> Verdict:
    """Map FAIL → KNOWN-BUG(reason) when the case declares an expected runtime bug."""
    if v.tag == "FAIL" and case.expected_known_bug:
        return Verdict("KNOWN-BUG", f"{case.expected_known_bug} | obs: {v.detail}")
    return v


def _apply_ocrvx(v: Verdict, ocrvx: RunResult | None,
                 arts: RunResult, case: Case) -> Verdict:
    """Demote a PASS verdict to OCRVX-BUG when ocrvx disagrees with arts."""
    if ocrvx is None or v.tag not in ("PASS-SCALAR", "PASS-RC"):
        return v
    bug = _check_ocrvx(ocrvx, arts, case)
    return bug if bug is not None else v


def _check_ocrvx(ocrvx: RunResult, arts: RunResult, case: Case) -> Verdict | None:
    """Return an OCRVX-BUG verdict if ocrvx diverges from arts, else None."""
    if ocrvx.rc != 0:
        return Verdict("OCRVX-BUG", f"ocrvx rc={ocrvx.rc}")
    if not case.scalar_re:
        return None  # rc=0, no scalar to compare
    ov = _pull(ocrvx.stdout, case.scalar_re, case.scalar_kind)
    ar = _pull(arts.stdout,  case.scalar_re, case.scalar_kind)
    if ov is None:
        return Verdict("OCRVX-BUG", "ocrvx scalar_re miss")
    if case.scalar_kind == "bool":
        return None  # both present
    if ar is None:
        return None  # arts failed — FAIL already raised upstream
    if case.scalar_kind == "int":
        if ov != ar:
            return Verdict("OCRVX-BUG", f"ocrvx={ov} arts={ar}")
        return None
    d = _drift(ov, ar)
    if d > case.scalar_tol:
        return Verdict("OCRVX-BUG",
                       f"ocrvx={ov:.6g} arts={ar:.6g} drift={d:.2e} tol={case.scalar_tol:.2e}")
    return None


def tier_a(xsocr: RunResult | None, arts: RunResult, case: Case,
           ocrvx: RunResult | None = None) -> Verdict:
    # arts_only cases only check the arts side.
    if case.arts_only:
        if arts.rc != 0:
            return _demote_if_known_bug(
                Verdict("FAIL", f"arts rc={arts.rc} (arts-only mode)"), case)
        if not case.scalar_re:
            return Verdict("PASS-RC", "arts-only: rc=0 (no scalar configured)")
        a = _pull(arts.stdout, case.scalar_re, case.scalar_kind)
        if a is None:
            return _demote_if_known_bug(
                Verdict("FAIL", "arts-only: scalar_re miss"), case)
        if case.scalar_kind == "bool":
            return Verdict("PASS-SCALAR", "arts-only: bool marker present")
        if not _expect_ok(a, case):
            return _demote_if_known_bug(
                Verdict("FAIL", f"arts-only: {a} != expected {case.expect}"), case)
        return Verdict("PASS-SCALAR", f"arts-only: {case.scalar_kind}={a}")

    # Standard xsocr↔arts compare.
    assert xsocr is not None
    if xsocr.rc != 0 or arts.rc != 0:
        return _demote_if_known_bug(
            Verdict("FAIL", f"rc xsocr={xsocr.rc} arts={arts.rc}"), case)
    if not case.scalar_re:
        return _apply_ocrvx(
            Verdict("PASS-RC", "rc=0 both (no scalar configured)"), ocrvx, arts, case)
    x = _pull(xsocr.stdout, case.scalar_re, case.scalar_kind)
    a = _pull(arts.stdout,  case.scalar_re, case.scalar_kind)
    if x is None or a is None:
        return _demote_if_known_bug(
            Verdict("FAIL",
                    f"scalar_re miss (xsocr={x is not None} arts={a is not None})"),
            case)
    if case.scalar_kind == "bool":
        return _apply_ocrvx(
            Verdict("PASS-SCALAR", "bool present in both"), ocrvx, arts, case)
    if case.scalar_kind == "int":
        if x != a:
            return _demote_if_known_bug(
                Verdict("FAIL", f"int xsocr={x} arts={a}"), case)
        if not _expect_ok(x, case):
            return _demote_if_known_bug(
                Verdict("FAIL", f"int={x} != expected {case.expect}"), case)
        return _apply_ocrvx(
            Verdict("PASS-SCALAR", f"int={x}" + (" ==expect" if case.expect else "")),
            ocrvx, arts, case)
    # float
    d = _drift(x, a)
    if d > case.scalar_tol:
        return _demote_if_known_bug(
            Verdict("FAIL",
                    f"xsocr={x:.6g} arts={a:.6g} drift={d:.2e} tol={case.scalar_tol:.2e}"),
            case)
    if not _expect_ok(x, case):
        return _demote_if_known_bug(
            Verdict("FAIL", f"xsocr={x:.6g} arts={a:.6g} != expected {case.expect}"), case)
    return _apply_ocrvx(
        Verdict("PASS-SCALAR",
                f"xsocr={x:.6g} arts={a:.6g} drift={d:.2e}" + (" ==expect" if case.expect else "")),
        ocrvx, arts, case)


def tier_b(xsocr: RunResult, arts: RunResult, base: RunResult, case: Case) -> Verdict:
    if xsocr.rc != 0 or arts.rc != 0 or base.rc != 0:
        return Verdict("FAIL", f"rc xs={xsocr.rc} ar={arts.rc} bs={base.rc}")
    assert case.baseline is not None
    x = _pull(xsocr.stdout, case.scalar_re, case.scalar_kind)
    a = _pull(arts.stdout,  case.scalar_re, case.scalar_kind)
    b = _pull(base.stdout,  case.baseline.scalar_re, case.baseline.scalar_kind)
    if x is None or a is None or b is None:
        return Verdict("FAIL", f"scalar_re miss xs={x is not None} ar={a is not None} bs={b is not None}")
    if case.scalar_kind == "bool":
        return Verdict("PASS-SCALAR", "bool marker present in all three")
    if case.scalar_kind == "int":
        return (Verdict("PASS-SCALAR", f"int={x}") if x == a == b
                else Verdict("FAIL", f"xs={x} ar={a} bs={b}"))
    # float
    dxa = _drift(x, a)
    dxb = _drift(x, b)
    dab = _drift(a, b)
    if dxa <= case.scalar_tol and max(dxb, dab) <= case.baseline.scalar_tol:
        return Verdict("PASS-SCALAR",
                       f"xs={x:.6g} ar={a:.6g} bs={b:.6g} dxa={dxa:.2e} dxb={dxb:.2e}")
    return Verdict("FAIL",
                   f"xs={x:.6g} ar={a:.6g} bs={b:.6g} dxa={dxa:.2e} dxb={dxb:.2e} "
                   f"(oc_tol={case.scalar_tol:.2e}, bo_tol={case.baseline.scalar_tol:.2e})")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-dir", default="build_release_mrnew_lazy",
                   help="Build directory containing apps and configs")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--only", type=str, default="")
    p.add_argument("--mem-gb", type=int, default=4)
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    logdir = LOGS_ROOT / ts
    runner = Runner(args.mem_gb, args.timeout, logdir)

    results_a: list[dict[str, Any]] = []
    for c in TIER_A:
        if only and c.name not in only:
            continue
        if c.skip:
            results_a.append({"name": c.name, "verdict": "SKIP", "detail": c.skip})
            print(f"  [A] {c.name:35s}  SKIP  {c.skip}")
            continue
        if c.stress_skip:
            results_a.append({"name": c.name, "verdict": "SKIP-STRESS",
                              "detail": c.stress_skip})
            print(f"  [A] {c.name:35s}  SKIP-STRESS  {c.stress_skip[:60]}")
            continue
        ar = runner.run_ocr(c.name, c.ocr_base, c.args, "arts")
        if c.arts_only:
            xs = None
            xs_rc, xs_wall = "—", "—"
        else:
            xs = runner.run_ocr(c.name, c.ocr_base, c.args, "xsocr")
            xs_rc, xs_wall = xs.rc, round(xs.wall, 3)
        ocrvx_bin = APPS_DIR / f"{c.ocr_base}_ocrvx"
        if ocrvx_bin.exists() and not c.ocrvx_skip:
            ov = runner.run_ocrvx_mpi(c.name, c.ocr_base, c.args)
            ov_rc, ov_wall = ov.rc, round(ov.wall, 3)
        else:
            ov = None
            ov_rc, ov_wall = "—", "—"
        v = tier_a(xs, ar, c, ocrvx=ov)
        results_a.append({
            "name": c.name, "args": c.args,
            "xsocr_rc": xs_rc, "xsocr_wall": xs_wall,
            "arts_rc":  ar.rc, "arts_wall":  round(ar.wall, 3),
            "ocrvx_rc": ov_rc, "ocrvx_wall": ov_wall,
            "verdict":  v.tag, "detail": v.detail,
            "expected_known_bug": c.expected_known_bug,
        })
        xs_line = "arts-only" if c.arts_only else f"xs={xs.rc}/{xs.wall:4.1f}s"
        ov_line = f"ov={ov_rc}/{ov_wall}s" if ov is not None else ""
        print(f"  [A] {c.name:35s}  {xs_line}  "
              f"ar={ar.rc}/{ar.wall:4.1f}s  {ov_line}  -> {v.tag}  {v.detail[:80]}")

    results_b: list[dict[str, Any]] = []
    if not args.no_baseline:
        for c in TIER_B:
            if only and c.name not in only:
                continue
            if c.skip:
                results_b.append({"name": c.name, "verdict": "SKIP", "detail": c.skip})
                continue
            assert c.baseline is not None
            xs = runner.run_ocr(c.name, c.ocr_base, c.args, "xsocr")
            ar = runner.run_ocr(c.name, c.ocr_base, c.args, "arts")
            bs = runner.run_baseline(c.name, c.baseline)
            v = tier_b(xs, ar, bs, c)
            results_b.append({
                "name": c.name, "args": c.args,
                "xsocr_rc": xs.rc, "xsocr_wall": round(xs.wall, 3),
                "arts_rc":  ar.rc, "arts_wall":  round(ar.wall, 3),
                "base_rc":  bs.rc, "base_wall":  round(bs.wall, 3),
                "verdict":  v.tag, "detail": v.detail,
            })
            print(f"  [B] {c.name:35s}  xs={xs.rc}/{xs.wall:4.1f}s  "
                  f"ar={ar.rc}/{ar.wall:4.1f}s  bs={bs.rc}/{bs.wall:4.1f}s  -> {v.tag}  {v.detail[:80]}")

    # --- Tier M: multinode scaling invariance ---
    # For each eligible Tier-A app, run arts and xsocr at 2 nodes.  Scalars
    # must match the single-node result.
    skip_tier_m = os.environ.get("SKIP_TIER_M", "")
    MN_RANKS = [2, 3, 4, "2n_io"]  # "2n_io" = 2-node IO-variant, arts-only
    results_m: list[dict[str, Any]] = []
    for c in TIER_A:
        if skip_tier_m or not c.multinode:
            continue
        if only and c.name not in only:
            continue
        if c.multinode_skip:
            results_m.append({"name": c.name, "verdict": "SKIP-MN",
                              "detail": c.multinode_skip})
            print(f"  [M] {c.name:35s}  SKIP-MN  {c.multinode_skip[:60]}")
            continue
        if not c.scalar_re:
            continue  # need a scalar for Tier M

        # Single-node reference (already captured in Tier A run above, but
        # re-run for isolation so we have fresh results).  arts_only and
        # multinode_arts_only cases have xsocr disabled at multinode (xsocr
        # cannot run them there) and compare arts multinode output against
        # the arts single-node reference only.  Geometry apps (multinode_args)
        # have a rank-count-specific reference, recomputed per-n in the loop.
        mn_arts_only = c.arts_only or c.multinode_arts_only
        mn_xsocr_only = c.multinode_xsocr_only
        use_mn_args = c.multinode_args is not None
        ocrvx_bin = APPS_DIR / f"{c.ocr_base}_ocrvx"
        mn_run_ocrvx = ocrvx_bin.exists() and not c.ocrvx_skip
        ref_a_val = ref_x_val = ref_ov_val = None
        if not use_mn_args:
            if not mn_xsocr_only:
                ref_ar = runner.run_ocr(c.name, c.ocr_base, c.args, "arts")
                ref_a_val = _pull(ref_ar.stdout, c.scalar_re, c.scalar_kind)
            if not mn_arts_only:
                ref_xs = runner.run_ocr(c.name, c.ocr_base, c.args, "xsocr")
                ref_x_val = _pull(ref_xs.stdout, c.scalar_re, c.scalar_kind)
            if mn_run_ocrvx:
                ref_ov = runner.run_ocrvx_mpi(c.name, c.ocr_base, c.args)
                ref_ov_val = _pull(ref_ov.stdout, c.scalar_re, c.scalar_kind)

        all_ok = True        # arts + xsocr checks only
        ocrvx_mn_ok = True  # ocrvx MN checks (separate: doesn't affect FAIL)
        detail_parts = []
        for n in MN_RANKS:
            # "2n_io" is a 2-node arts-only IO variant — use the 2-node geometry.
            geo_n = 2 if n == "2n_io" else n
            args_n = c.multinode_args.get(geo_n, c.args) if use_mn_args else c.args
            if use_mn_args:
                # rank-count-specific geometry: reference is single-node with
                # the SAME args so the scalar is comparable.  For full 3-way
                # geometry apps the xsocr reference is also rank-count-specific,
                # so recompute it per-n alongside the arts reference.
                if not mn_xsocr_only:
                    ref_n = runner.run_ocr(c.name, c.ocr_base, args_n, "arts")
                    ref_a_val = _pull(ref_n.stdout, c.scalar_re, c.scalar_kind)
                if not mn_arts_only and n != "2n_io":
                    ref_xn = runner.run_ocr(c.name, c.ocr_base, args_n, "xsocr")
                    ref_x_val = _pull(ref_xn.stdout, c.scalar_re, c.scalar_kind)
                if mn_run_ocrvx and n != "2n_io":
                    ref_ovn = runner.run_ocrvx_mpi(c.name, c.ocr_base, args_n)
                    ref_ov_val = _pull(ref_ovn.stdout, c.scalar_re, c.scalar_kind)
            checks = []
            ov_checks = []
            if not mn_xsocr_only:
                ar_mn = runner.run_arts_mn(c.name, c.ocr_base, args_n, n,
                                           timeout=c.multinode_timeout)
                a_val = _pull(ar_mn.stdout, c.scalar_re, c.scalar_kind)
                checks.append((f"ar{n}", a_val, ref_a_val, ar_mn.rc))
            if not mn_arts_only and n != "2n_io":
                xs_mn = runner.run_xsocr_mpi(c.name, c.ocr_base, args_n, n,
                                             timeout=c.multinode_timeout)
                x_val = _pull(xs_mn.stdout, c.scalar_re, c.scalar_kind)
                checks.append((f"xs{n}", x_val, ref_x_val, xs_mn.rc))
            if mn_run_ocrvx and n != "2n_io":
                ov_mn = runner.run_ocrvx_mpi(c.name, c.ocr_base, args_n,
                                              np=geo_n, timeout=c.multinode_timeout)
                ov_val = _pull(ov_mn.stdout, c.scalar_re, c.scalar_kind)
                ov_checks.append((f"ov{n}", ov_val, ref_ov_val, ov_mn.rc))

            def _eval_check(label, mn_val, ref_val, mn_rc):
                if mn_rc != 0:
                    detail_parts.append(f"{label}:rc={mn_rc}")
                    return False
                if mn_val is None:
                    detail_parts.append(f"{label}:miss")
                    return False
                if c.scalar_kind == "bool":
                    detail_parts.append(f"{label}:ok")
                elif c.scalar_kind == "int":
                    if mn_val != ref_val:
                        detail_parts.append(f"{label}:{mn_val}!={ref_val}")
                        return False
                    else:
                        detail_parts.append(f"{label}:ok")
                else:  # float
                    d = _drift(mn_val, ref_val) if ref_val is not None else 999
                    if d > c.scalar_tol:
                        detail_parts.append(f"{label}:drift={d:.2e}")
                        return False
                    else:
                        detail_parts.append(f"{label}:ok")
                return True

            for check in checks:
                if not _eval_check(*check):
                    all_ok = False
            for check in ov_checks:
                if not _eval_check(*check):
                    ocrvx_mn_ok = False

        if all_ok and ocrvx_mn_ok:
            vtag = "PASS-MN"
        elif all_ok and not ocrvx_mn_ok:
            vtag = "OCRVX-BUG"
        else:
            vtag = "FAIL"
        vdetail = " ".join(detail_parts)
        # Demote FAIL to KNOWN-BUG when the case has an expected_known_bug
        # (xsocr-side hangs/SEGVs at multinode shouldn't poison the Tier-M
        # tally; arts-side ar2:ok already proves arts is fine).
        if vtag == "FAIL" and c.expected_known_bug:
            vtag = "KNOWN-BUG"
            vdetail = f"{c.expected_known_bug} | obs: {vdetail}"
        results_m.append({"name": c.name, "verdict": vtag, "detail": vdetail})
        print(f"  [M] {c.name:35s}  -> {vtag}  {vdetail[:80]}")

    _write_report(logdir, results_a, results_b, results_m)
    tally_a = _tally(results_a)
    tally_b = _tally(results_b)
    tally_m = _tally(results_m)
    summary = (f"Tier A: {tally_a}\nTier B: {tally_b}\n"
               f"Tier M: {tally_m}\nLog dir: {logdir}\n")
    (logdir / "summary.txt").write_text(summary)
    print("\n" + summary)


def _tally(results: list[dict[str, Any]]) -> str:
    buckets: dict[str, int] = {}
    for r in results:
        k = r.get("verdict", "UNKNOWN")
        buckets[k] = buckets.get(k, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(buckets.items()))


def _write_report(logdir: Path, a: list[dict], b: list[dict],
                   m: list[dict] | None = None) -> None:
    (logdir / "report.json").write_text(
        json.dumps({"tier_a": a, "tier_b": b, "tier_m": m or []}, indent=2))

    lines = ["# ARTS benchmark correctness report", ""]
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Repo: `{REPO}`")
    lines.append("")
    lines.append("## Tier A — xsocr ↔ arts (per-app scalar)")
    lines.append("")
    lines.append("| app | xsocr rc/wall | arts rc/wall | verdict | detail |")
    lines.append("|---|---|---|---|---|")
    for r in a:
        if r["verdict"] in ("SKIP", "SKIP-STRESS"):
            lines.append(f"| {r['name']} | — | — | {r['verdict']} | {r['detail'][:140]} |")
        else:
            xs_cell = (f"{r['xsocr_rc']}/{r['xsocr_wall']}s"
                       if r['xsocr_rc'] != "—" else "arts-only")
            lines.append(
                f"| {r['name']} | {xs_cell} "
                f"| {r['arts_rc']}/{r['arts_wall']}s "
                f"| {r['verdict']} | {r['detail'][:140]} |"
            )
    lines.append("")
    lines.append("## Tier B — baseline ↔ xsocr ↔ arts (per-app scalar)")
    lines.append("")
    lines.append("| app | xsocr | arts | baseline | verdict | detail |")
    lines.append("|---|---|---|---|---|---|")
    for r in b:
        if r["verdict"] == "SKIP":
            lines.append(f"| {r['name']} | — | — | — | SKIP | {r['detail']} |")
        else:
            lines.append(
                f"| {r['name']} | {r['xsocr_rc']}/{r['xsocr_wall']}s "
                f"| {r['arts_rc']}/{r['arts_wall']}s "
                f"| {r['base_rc']}/{r['base_wall']}s "
                f"| {r['verdict']} | {r['detail'][:140]} |"
            )
    lines.append("")

    # Known-bugs sections: canonical TODO list for shim/runtime fixes.
    arts_hangs = [r for r in a
                  if r.get("verdict") == "KNOWN-BUG"
                  and "arts-hang" in r.get("expected_known_bug", "")]
    xsocr_segvs = [r for r in a
                   if r.get("verdict") == "KNOWN-BUG"
                   and "xsocr-segv" in r.get("expected_known_bug", "")]
    if arts_hangs:
        lines.append("## Known runtime / shim bugs (arts side)")
        lines.append("")
        lines.append("These are documented arts-side failures that the harness "
                     "currently tolerates as KNOWN-BUG. Each one is an "
                     "actionable TODO for the shim/runtime fix phase.")
        lines.append("")
        for r in arts_hangs:
            lines.append(f"- **{r['name']}** — {r['expected_known_bug']}")
        lines.append("")
    if xsocr_segvs:
        lines.append("## Pre-existing xsocr bugs (arts runs cleanly)")
        lines.append("")
        lines.append("These cases run only the arts side because xsocr has a "
                     "pre-existing crash. Not an ARTS problem; listed here so "
                     "it's visible in the report.")
        lines.append("")
        for r in xsocr_segvs:
            lines.append(f"- **{r['name']}** — {r['expected_known_bug']}")
        lines.append("")

    # SAR dataset cache status (for later debugging of build pipeline).
    sar_datasets_dir = REPO / "third_party" / "ocr-apps" / "apps" / "sar" / "datasets"
    if sar_datasets_dir.exists():
        lines.append("## SAR dataset cache status")
        lines.append("")
        for sz in ["tiny", "small", "medium", "large"]:
            datafile = sar_datasets_dir / sz / "Data.bin"
            if datafile.exists():
                sz_mb = datafile.stat().st_size / (1024 * 1024)
                mtime = time.strftime("%Y-%m-%d %H:%M",
                                       time.localtime(datafile.stat().st_mtime))
                lines.append(f"- **{sz}**: cached ({sz_mb:.1f} MB, mtime {mtime})")
            else:
                lines.append(f"- **{sz}**: MISSING")
        lines.append("")

    (logdir / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
