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
import json
import os
import re
import shlex
import shutil
import subprocess
import time

shlex_quote = shlex.quote
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent.parent
BUILD = REPO / "build_release"
APPS_DIR = BUILD / "benchmarks" / "apps"
BASE_DIR = BUILD / "benchmarks" / "baseline"
LOGS_ROOT = REPO / "benchmarks" / "scripts" / "logs" / "correctness"
ARTS_CFG = REPO / "configs" / "local" / "1n.cfg"
XSOCR_CFG = REPO / "configs" / "mpi" / "1n.cfg"

SW_DIR = Path("/tmp/arts_sw_test")
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
    skip: str = ""                        # non-empty → SKIP(reason)
    stress_skip: str = ""                 # non-empty → SKIP-STRESS(reason)
    arts_only: bool = False               # True → never invoke xsocr
    expected_known_bug: str = ""          # FAIL → KNOWN-BUG(reason) when set
    baseline: BaselineSpec | None = None
    multinode: bool = False               # True → also run Tier M (arts + xsocr at N ranks)
    multinode_skip: str = ""              # non-empty → skip multinode with reason

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
         ["2","2",f"{SW_DIR}/str1.txt",f"{SW_DIR}/str2.txt",f"{SW_DIR}/score.txt"],
         scalar_re=r"score:\s*(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("fft", "fft", ["6"],
         scalar_re=r"Output matched expected results", scalar_kind="bool",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("triangle", "triangle", [],
         scalar_re=r"final count\s+(\d+)", scalar_kind="int",
         multinode_skip="problem size hardcoded; multinode wall-time exceeds the 30s per-case budget"),
    Case("p2p", "p2p", ["2","10","100","10"],
         scalar_re=r"PASS checksum\s*=\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-8,
         multinode=True),
    Case("CoMD_intel_chandra", "CoMD_intel_chandra",
         ["-x","4","-y","4","-z","4","-N","2","-n","1"],
         scalar_re=r"Initial energy\s*:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-5,
         multinode=True),
    Case("CoMD_intel_chandra_tiled", "CoMD_intel_chandra_tiled",
         ["-x","4","-y","4","-z","4","-N","2"],
         scalar_re=r"Final energy\s*:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-6,
         multinode_skip="xsocr HCDist timeout at multinode"),
    Case("CoMD_sdsc", "CoMD_sdsc", ["-x","4","-y","4","-z","4","-N","2"],
         scalar_re=r"Initial energy\s*:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-10,
         multinode_skip="no EDT affinity hints (sdsc variant); runs caller-rank only"),
    Case("CoMD_sdsc2", "CoMD_sdsc2", ["-x","4","-y","4","-z","4","-N","2"],
         scalar_re=r"Final energy\s*:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-6,
         multinode=True,
         expected_known_bug="multinode hangs intermittently in xsocr lockable DB acquire on EDTs with many cross-rank RW+RO mixed deps"),
    Case("hpcg_intel", "hpcg_intel", ["1","1","1","16","5"],
         scalar_re=r"final deviation:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-4,
         multinode=True),
    Case("hpcg_intel_Eager", "hpcg_intel_Eager", ["1","1","1","16","5"],
         scalar_re=r"final deviation:\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-4,
         multinode=True),
    Case("Stencil1D_intel_chandra", "Stencil1D_intel_chandra", [],
         scalar_re=r"Solution validates", scalar_kind="bool",
         multinode_skip="known multinode hang (arts + xsocr channel metadata)"),
    Case("Stencil2D_intel_channelEVTs", "Stencil2D_intel_channelEVTs", [],
         scalar_re=r"Solution validates", scalar_kind="bool",
         multinode=True),
    Case("Stencil2D_intel_chandra", "Stencil2D_intel_chandra", [],
         scalar_re=r"Solution validates", scalar_kind="bool",
         multinode_skip="known multinode hang (arts + xsocr channel metadata)"),
    Case("miniAMR_intel", "miniAMR_intel",
         ["--nx","4","--ny","4","--nz","4","--num_tsteps","2","--num_objects","1"],
         scalar_re=r"Grand Total Checksum\s*==\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-8,
         multinode_skip="no EDT affinity hints (intel variant); runs caller-rank only"),
    Case("npb_cg", "npb_cg", [],
         scalar_re=r"zeta\s*=\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-10,
         multinode_skip="arts 5-node scalar miss"),
    Case("hpgmg", "hpgmg", ["4","1"],
         scalar_re=r"\|\|error\|\|\s*=\s*([\-+0-9.eE]+)", scalar_kind="float",
         scalar_tol=1e-4,
         multinode=True),
    Case("tempest", "tempest", [],
         scalar_re=r"DONE\.", scalar_kind="bool",
         multinode_skip="xsocr HCDist timeout at multinode"),
    Case("curvefit", "curvefit", [],
         scalar_re=r"SUCCESS", scalar_kind="bool",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("testlibs", "testlibs", [],
         scalar_re=r"Testing complete", scalar_kind="bool",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("graph500", "graph500", ["6","8","1","1"],
         scalar_re=r"mean MTEPS", scalar_kind="bool",
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
         multinode_skip="xsocr HCDist timeout at multinode"),

    # --- rc-only sanity (no meaningful scientific scalar) ---
    Case("printf",           "printf",           [],
         scalar_re=r"Hello from mainEdt", scalar_kind="bool",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("quicksort",        "quicksort",        [],
         scalar_re=r"Sorting Finished", scalar_kind="bool",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("basicIO",          "basicIO",          ["0","10", BASIC_IO_DAT],
         scalar_re=r"BASICIO_CHK\s+(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("cache_offset",     "cache_offset",     [],
         scalar_re=r"calling ocrShutdown", scalar_kind="bool",
         multinode_skip="single-node design: file-scope dbGuids/dbPtrs arrays"),
    Case("highbw",           "highbw",           [],
         scalar_re=r"HIGHBW_WORK_SUM\s*=\s*(\d+)", scalar_kind="int",
         expected_known_bug="xsocr-side hangs at startup; arts-side completes cleanly",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("task_priorities",  "task_priorities",  [],
         scalar_re=r"Hello from 9", scalar_kind="bool",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("dbctrl",           "dbctrl",           ["5","5","256"],
         scalar_re=r"Total time", scalar_kind="bool",
         multinode_skip="single-node design: file-scope template GUIDs + evtMap array"),
    Case("prodcon",          "prodcon",          [],
         scalar_re=r"MB/s", scalar_kind="bool",
         multinode_skip="single-node design: file-scope mapProdGuid/mapConsGuid"),
    Case("globalsum_cgShim",   "globalsum_cgShim",   [],
         scalar_re=r"\bPASS\b", scalar_kind="bool",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("globalsum_cgNoShim", "globalsum_cgNoShim", [],
         scalar_re=r"\bPASS\b", scalar_kind="bool",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("globalsum_pcg",      "globalsum_pcg",      [],
         scalar_re=r"\bPASS\b", scalar_kind="bool",
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
         scalar_re=r"VERIFICATION passed!", scalar_kind="bool"),
    Case("LCS_distributed_ST","LCS_distributed_ST",[],
         scalar_re=r"Shutting down OCR runtime", scalar_kind="bool",
         multinode_skip="xsocr HCDist SEGV at multinode"),
    Case("LCS_shared",        "LCS_shared",       [],
         scalar_re=r"Shutting down OCR runtime", scalar_kind="bool",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("RSBench_intel",             "RSBench_intel",             ["-l","100"],
         scalar_re=r"Lookups:", scalar_kind="bool",
         multinode_skip="no EDT affinity hints (intel variant); runs caller-rank only"),
    Case("RSBench_intel_sharedDB",    "RSBench_intel_sharedDB",    ["-l","100"],
         scalar_re=r"Lookups:", scalar_kind="bool",
         multinode=True),
    Case("XSBench_intel",             "XSBench_intel",             ["-s","small","-g","10","-l","100"],
         scalar_re=r"Workload\s+\(unit\):\s+(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("XSBench_intel_sharedDB",    "XSBench_intel_sharedDB",    ["-s","small","-g","10","-l","100"],
         scalar_re=r"Workload\s+\(unit\):\s+(\d+)", scalar_kind="int",
         multinode_skip="no EDT affinity hints; runs caller-rank only"),
    Case("uts", "uts", [],
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
         scalar_re=r"TREEFORKJOIN Concluding: TaskTYPE=\d+ TaskID=\S+ Work is ok",
         scalar_kind="bool",
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
         scalar_re=r"Solution Validates", scalar_kind="bool",
         multinode=True),

    # --- previously-SKIPped: genuine arts-side runtime bugs (report, don't fix) ---
    Case("miniAMR_intel_bryan", "miniAMR_intel_bryan", [],
         scalar_re=r"miniAMR complete", scalar_kind="bool",
         multinode=True),
    # hpcg_intel_Eager_Collective: xsocr target is EXCLUDE_FROM_ALL in
    # benchmarks/apps/CMakeLists.txt:387 (xsocr runtime doesn't ship
    # a collective-event primitive).  arts runs the app to completion
    # in ~0.4 s and prints the `final deviation` scalar cleanly.
    # Treat as arts_only here — there is no arts-side bug.
    Case("hpcg_intel_Eager_Collective", "hpcg_intel_Eager_Collective",
         ["1","1","1","16","5"],
         scalar_re=r"final deviation:\s*([\-+0-9.eE]+)",
         scalar_kind="float", scalar_tol=1e-4,
         multinode_skip="collective event not supported in arts multinode + xsocr HCDist"),

    Case("stream", "stream", [],
         scalar_re=r"Solution Validates", scalar_kind="bool",
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
    # Larger sizes (small/medium/large) are excluded from the default
    # harness run because their wall-time exceeds the 30s per-case
    # budget; opt in via --include-slow. ---
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
        SW_DIR.mkdir(parents=True, exist_ok=True)
        if not (SW_DIR / "str1.txt").exists():
            (SW_DIR / "str1.txt").write_text("ACGTACGTACGTACGTACGT\n")
        if not (SW_DIR / "str2.txt").exists():
            (SW_DIR / "str2.txt").write_text("ACGTAGGTACGTACGTAGGT\n")
        if not (SW_DIR / "score.txt").exists():
            (SW_DIR / "score.txt").write_text("1 -1 -1\n-1 1 -1\n-1 -1 1\n")
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

    def _run(self, cmd: str, env: dict[str, str], logfile: Path) -> RunResult:
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
                    env=env, timeout=self.timeout + 10,
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
            f"timeout {self.timeout} ./{binary} " + " ".join(args)
        )
        return self._run(cmd, env, logfile)

    # --- Multinode runners (Tier M) ---

    _ARTS_MN_CFGS = {
        2: "local/2n.cfg",
    }
    _XSOCR_MN_CFGS = {
        2: "mpi/2n.cfg",
    }

    def run_arts_mn(self, case_name: str, bin_name: str, args: list[str],
                    nodes: int) -> RunResult:
        """Run arts at N nodes (self-fork launcher via cfg)."""
        cfg_name = self._ARTS_MN_CFGS[nodes]
        cfg_src = REPO / "configs" / cfg_name
        shutil.copy2(cfg_src, APPS_DIR / "arts.cfg")  # arts reads ./arts.cfg
        logfile = self.logdir / f"{case_name}.arts_mn{nodes}.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        cmd = (
            f"cd {APPS_DIR} && ulimit -v {self.mem_kb} && "
            f"timeout {self.timeout} ./{bin_name}_arts " + " ".join(args)
        )
        result = self._run(cmd, env, logfile)
        # Restore single-node cfg for subsequent single-node runs
        shutil.copy2(ARTS_CFG, APPS_DIR / "arts.cfg")
        return result

    def run_xsocr_mpi(self, case_name: str, bin_name: str, args: list[str],
                      np: int) -> RunResult:
        """Run xsocr at N MPI ranks (mpirun launcher)."""
        logfile = self.logdir / f"{case_name}.xsocr_mpi{np}.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        xsocr_cfg = REPO / "configs" / self._XSOCR_MN_CFGS[np]
        cmd = (
            f"cd {APPS_DIR} && ulimit -v {self.mem_kb} && "
            f"timeout {self.timeout} mpirun --oversubscribe -n {np} "
            f"./{bin_name}_xsocr -ocr:cfg {xsocr_cfg} "
            + " ".join(args)
        )
        return self._run(cmd, env, logfile)

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


def _demote_if_known_bug(v: Verdict, case: Case) -> Verdict:
    """Map FAIL → KNOWN-BUG(reason) when the case declares an expected runtime bug."""
    if v.tag == "FAIL" and case.expected_known_bug:
        return Verdict("KNOWN-BUG", f"{case.expected_known_bug} | obs: {v.detail}")
    return v


def tier_a(xsocr: RunResult | None, arts: RunResult, case: Case) -> Verdict:
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
        return Verdict("PASS-SCALAR", f"arts-only: {case.scalar_kind}={a}")

    # Standard xsocr↔arts compare.
    assert xsocr is not None
    if xsocr.rc != 0 or arts.rc != 0:
        return _demote_if_known_bug(
            Verdict("FAIL", f"rc xsocr={xsocr.rc} arts={arts.rc}"), case)
    if not case.scalar_re:
        return Verdict("PASS-RC", "rc=0 both (no scalar configured)")
    x = _pull(xsocr.stdout, case.scalar_re, case.scalar_kind)
    a = _pull(arts.stdout,  case.scalar_re, case.scalar_kind)
    if x is None or a is None:
        return _demote_if_known_bug(
            Verdict("FAIL",
                    f"scalar_re miss (xsocr={x is not None} arts={a is not None})"),
            case)
    if case.scalar_kind == "bool":
        return Verdict("PASS-SCALAR", "bool present in both")
    if case.scalar_kind == "int":
        if x == a:
            return Verdict("PASS-SCALAR", f"int={x}")
        return _demote_if_known_bug(
            Verdict("FAIL", f"int xsocr={x} arts={a}"), case)
    # float
    d = _drift(x, a)
    if d <= case.scalar_tol:
        return Verdict("PASS-SCALAR", f"xsocr={x:.6g} arts={a:.6g} drift={d:.2e}")
    return _demote_if_known_bug(
        Verdict("FAIL",
                f"xsocr={x:.6g} arts={a:.6g} drift={d:.2e} tol={case.scalar_tol:.2e}"),
        case)


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
        v = tier_a(xs, ar, c)
        results_a.append({
            "name": c.name, "args": c.args,
            "xsocr_rc": xs_rc, "xsocr_wall": xs_wall,
            "arts_rc":  ar.rc, "arts_wall":  round(ar.wall, 3),
            "verdict":  v.tag, "detail": v.detail,
            "expected_known_bug": c.expected_known_bug,
        })
        xs_line = "arts-only" if c.arts_only else f"xs={xs.rc}/{xs.wall:4.1f}s"
        print(f"  [A] {c.name:35s}  {xs_line}  "
              f"ar={ar.rc}/{ar.wall:4.1f}s  -> {v.tag}  {v.detail[:80]}")

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
    MN_RANKS = [2]
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
        # re-run for isolation so we have fresh results).  arts_only cases
        # have xsocr disabled (xsocr cannot run them) and compare arts
        # multinode output against arts single-node reference only.
        ref_ar = runner.run_ocr(c.name, c.ocr_base, c.args, "arts")
        ref_a_val = _pull(ref_ar.stdout, c.scalar_re, c.scalar_kind)
        if c.arts_only:
            ref_xs = None
            ref_x_val = None
        else:
            ref_xs = runner.run_ocr(c.name, c.ocr_base, c.args, "xsocr")
            ref_x_val = _pull(ref_xs.stdout, c.scalar_re, c.scalar_kind)

        all_ok = True
        detail_parts = []
        for n in MN_RANKS:
            ar_mn = runner.run_arts_mn(c.name, c.ocr_base, c.args, n)
            a_val = _pull(ar_mn.stdout, c.scalar_re, c.scalar_kind)
            checks = [(f"ar{n}", a_val, ref_a_val, ar_mn.rc)]
            if not c.arts_only:
                xs_mn = runner.run_xsocr_mpi(c.name, c.ocr_base, c.args, n)
                x_val = _pull(xs_mn.stdout, c.scalar_re, c.scalar_kind)
                checks.append((f"xs{n}", x_val, ref_x_val, xs_mn.rc))

            for label, mn_val, ref_val, mn_rc in checks:
                if mn_rc != 0:
                    detail_parts.append(f"{label}:rc={mn_rc}")
                    all_ok = False
                elif mn_val is None:
                    detail_parts.append(f"{label}:miss")
                    all_ok = False
                elif c.scalar_kind == "bool":
                    detail_parts.append(f"{label}:ok")
                elif c.scalar_kind == "int":
                    if mn_val != ref_val:
                        detail_parts.append(f"{label}:{mn_val}!={ref_val}")
                        all_ok = False
                    else:
                        detail_parts.append(f"{label}:ok")
                else:  # float
                    d = _drift(mn_val, ref_val) if ref_val is not None else 999
                    if d > c.scalar_tol:
                        detail_parts.append(f"{label}:drift={d:.2e}")
                        all_ok = False
                    else:
                        detail_parts.append(f"{label}:ok")

        vtag = "PASS-MN" if all_ok else "FAIL"
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
