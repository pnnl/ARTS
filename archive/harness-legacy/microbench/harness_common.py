#!/usr/bin/env python3
"""
harness_common.py — shared infrastructure for the correctness and
performance harnesses: machine geometry, the launcher-pinning helpers, and
the `Runner` class that drives ocr/arts/xsocr/ocrvx/baseline subprocesses
(hwloc-aware core pinning, cleanup/reap-by-exe-path).

Nothing here is correctness-specific (no scalar extraction, no Case
definitions) — that lives in correctness_harness.py and any future perf
harness that imports this module.
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time

shlex_quote = shlex.quote
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# The repo-local toolchain libs (install/lib: hwloc, numa) back the MPI stack
# the reference runtimes link against.  DT_RUNPATH does not propagate to a
# shared library's own dependencies, so every child process needs the path on
# LD_LIBRARY_PATH — injected here so the harness works identically however it
# was launched (interactive shell, detached, cron).
_TOOLCHAIN_LIB = str(REPO / "install" / "lib")
if _TOOLCHAIN_LIB not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    _prev = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = _TOOLCHAIN_LIB + (":" + _prev if _prev else "")
# Same launch-mode independence for executables: the repo-local toolchain bin
# and the user-level MPI launcher (mpirun) live outside the default
# non-interactive PATH.
for _d in (str(Path.home() / ".local" / "bin"), str(REPO / "install" / "bin")):
    if _d not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = _d + ":" + os.environ.get("PATH", "")

# Everything an app run generates is confined to one gitignored scratch
# directory: every runner chdirs here before exec, so relative-path outputs
# (default output files, debug dumps) can never litter the repo root or the
# build tree, and staged fixtures live beside them.  Deliberately NOT /tmp:
# that is tmpfs (RAM) on the target hosts, and app outputs reach hundreds of
# MB.  Contents are disposable and overwritten run-to-run.
SCRATCH = REPO / "scratch"
SCRATCH.mkdir(exist_ok=True)

BASIC_IO_DAT = str(SCRATCH / "basicIO_test.dat")
CHOLESKY_INPUT = str(SCRATCH / "cholesky_input.mat")


def _cfg_name(n) -> str:
    return f"{n}n_sc.cfg" if isinstance(n, int) else f"{n}.cfg"


def BUILD_for(build_dir: str) -> Path:
    """Resolve a --build-dir argument (relative or absolute) to an absolute
    Path under REPO."""
    build = Path(build_dir)
    if not build.is_absolute():
        build = REPO / build
    return build


def APPS_DIR_for(build: Path) -> Path:
    return build / "benchmarks" / "apps"


# ---------------------------------------------------------------------------
# Per-target machine geometry (cbgpu02 = 48-core/96HT; bentley = 2x EPYC 7742,
# 128-core/256HT, 8 NUMA nodes x 16 physical cores -- the 16-thread rank budget
# aligns each rank to one NUMA node, and OS CPUs 0-127 are the per-core first
# hyperthreads, so contiguous 16-blocks are SMT-clean by construction).
# Drives the config subdir, the multinode node counts, the per-rank thread budget
# (for taskset pinning), and the ocr-vx TBB width.  Every config is sized so
# node_count * per-node-threads == the machine's core count.
# ---------------------------------------------------------------------------

# Multinode node counts every case runs at unless it documents a structural
# exclusion (see correctness_harness.py's Case.multinode_skip).  The *_io
# entries are arts-only:
# xsocr/ocr-vx have a single comm worker (no worker/progress split), so their
# N-node total already equals the plain N-node config — no separate IO variant.
_MN_NODE_COUNTS = {
    'cbgpu02': [2, 4],
    'bentley': [2, 4, 8],
}


def MN_RANKS_for(target: str) -> list:
    return _MN_NODE_COUNTS[target]


# Threads per rank (= per-node total thread budget) used to taskset-pin each
# mpirun rank to a disjoint core block, mirroring arts's per-rank pu_offset.
_TPN = {
    'cbgpu02': {2: 12, 4: 12},
    'bentley': {2: 16, 4: 16, 8: 16},
}


def _TPN_for(target: str) -> dict:
    return _TPN[target]


# ocr-vx TBB worker count per node count = full per-rank core budget, matching
# the original runtime's default (tbb::info::default_concurrency() = all cores
# the process sees).  We previously reserved one slot for the runtime's blocking
# shutdown-barrier task (budget-1), but the original gives ocr-vx the full
# budget, so match it.  Budget >= 3 at every node count here, so the P=1
# shutdown deadlock cannot occur.
_OCRVX_TBB = {
    'cbgpu02': {1: 48, 2: 12, 4: 12},
    'bentley': {1: 16, 2: 16, 4: 16, 8: 16},
}


def _OCRVX_TBB_for(target: str) -> dict:
    return _OCRVX_TBB[target]


# ocr-vx's coherence protocol is structurally far slower than arts/xsocr, so a
# correctness run that those two finish well within budget can still time out on
# ocr-vx while it is *still making progress* (not hung).  Give ocr-vx a larger
# wall budget so a genuine result is collected; fast cases are unaffected (they
# return long before the ceiling).  Cases are sized so the slowest reference
# runtime (ocr-vx at the widest rank count) finishes in well under a minute, so
# the multiplier only needs to cover ocr-vx's serialized message pipeline with
# headroom for variance -- a genuine hang still costs at most base*mult
# seconds, not the old 12-minute ceiling.
_OCRVX_TIMEOUT_MULT = 3

# Physical cores used on this machine target (= single-node thread budget).
# The reference runtimes (xsocr/ocr-vx/baseline) have no internal core pinning,
# so a single-node run would float its threads across ALL logical CPUs (incl.
# the HT siblings above core count).  Confine them to cores 0..N-1 via taskset
# so they occupy the same cores arts self-pins to.
_NCORES = {'cbgpu02': 48, 'bentley': 16}


def _NCORES_for(target: str) -> int:
    return _NCORES[target]


def pin_single(target: str) -> str:
    return f"taskset -c 0-{_NCORES_for(target) - 1}"


def pin_wrap(tpn: int) -> str:
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


@dataclass
class RunResult:
    rc: int
    wall: float
    stdout: str
    log_path: str


class Runner:
    def __init__(self, mem_gb: int, timeout: int, logdir: Path,
                 target: str, build: Path):
        self.target = target
        self.build = build
        self.mem_kb = mem_gb * 1024 * 1024
        self.mem_gb = mem_gb
        self.timeout = timeout
        self.logdir = logdir
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.apps_dir = build / "benchmarks" / "apps"
        self.base_dir = build / "benchmarks" / "baseline"
        self.arts_cfg = REPO / "configs" / "local" / target / "1n.cfg"
        self.xsocr_cfg = REPO / "configs" / "mpi" / target / "1n.cfg"

        # Per-target multinode config paths, generated from MN_RANKS.  The
        # *_io variants (progress-thread-heavy IO-forwarding stress) are
        # arts-only — no xsocr/MPI equivalent, so they are absent from the
        # xsocr map.
        mn_ranks = MN_RANKS_for(target)
        self._arts_mn_cfgs = {n: f"local/{target}/{_cfg_name(n)}" for n in mn_ranks}
        self._xsocr_mn_cfgs = {n: f"mpi/{target}/{_cfg_name(n)}"
                               for n in mn_ranks if isinstance(n, int)}

        self._tpn = _TPN_for(target)
        self._pin_single = pin_single(target)
        # ocr-vx TBB compute parallelism per node count (target-selected).
        # Sized so the *active* thread budget matches the arts/xsocr configs:
        # the runtime's shutdown barrier is a TBB task that blocks (zero CPU)
        # permanently occupying one parallelism slot, so effective compute
        # width is P-1 and P=1 deadlocks outright — hence P = per-node budget
        # - 1.
        self._ocrvx_tbb_threads = _OCRVX_TBB_for(target)

        # Memory capping (ulimit -v / cgroup MemoryMax) was retired entirely
        # (2026-07-14): virtual-address caps break mmap-reserving allocators
        # and this host never needs them. Runaway protection = per-case
        # timeout + reap-by-exe, not memory caps.

        # Stage fixtures once (arts config selection is per-run via the
        # ARTS_CONFIG env var — the runtime reads it before the ./arts.cfg
        # cwd fallback, so no config files are copied anywhere).
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

    def run_ocr(self, case_name: str, bin_name: str, args: list[str],
                backend: str,
               suffix: str = "", cfg_path: Path | None = None,
               width: int | None = None,
               extra_env: dict | None = None, timeout: int = 0) -> RunResult:
        # When backend=="arts" and suffix is given, exec <bin_name>_arts_<suffix>.
        # cfg_path overrides the capacity-1n default cfg this backend reads
        # (e.g. a scalability-series "_sc" cfg) -- callers outside the
        # capacity matrix (perf harness) use this; correctness never sets it.
        # extra_env layers caller-supplied env vars on top (e.g. an arts cfg
        # key override such as counter_folder -- every cfg key is overridable
        # by an identically-named env var, see config.c's per-variable
        # getenv() check).
        # timeout: per-case single-node wall budget (s); 0 -> the global
        # default (mirrors run_arts_mn's contract on the multinode side).
        if backend == "arts" and suffix:
            binary = f"{bin_name}_arts_{suffix}"
            log_tag = f"arts_{suffix}"
        else:
            binary = f"{bin_name}_{backend}"
            log_tag = backend
        logfile = self.logdir / f"{case_name}.{log_tag}.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        if extra_env:
            env.update(extra_env)
        if backend == "arts":
            env["ARTS_CONFIG"] = str(cfg_path or self.arts_cfg)
        if backend == "xsocr":
            env["OCR_CONFIG"] = str(cfg_path or self.xsocr_cfg)
        # Defense in depth: every backend gets an OS-level cpuset matching the
        # run's thread width, even the self-pinning ones — a stray unpinned
        # thread (aux/launcher) must not escape the measured core block.
        w = width if width is not None else _NCORES_for(self.target)
        pin = f"taskset -c 0-{w - 1} "
        to = timeout or self.timeout
        cmd = (
            f"cd {SCRATCH} && "
            f"timeout -k 1 {to} {pin}{self.apps_dir / binary} " + " ".join(args)
        )
        result = self._run(cmd, env, logfile, wall_timeout=to)
        self._reap_exe(self.apps_dir / binary)
        return result

    # --- Multinode runners ---

    # Per-run unique TCP port base for arts multinode runs.  A straggler from
    # the previous case (a rank still releasing its listen socket, or an orphan
    # from a timed-out run) would poison every later case that binds the same
    # range; the runtime lets the environment override any config key, so each
    # run gets its own range.
    #
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

    def _cleanup(self) -> None:
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
        executables (under apps_dir / base_dir), known MPI launchers, and
        MPI/UCX-owned /dev/shm patterns are touched — never unrelated state."""
        apps, base = str(self.apps_dir), str(self.base_dir)
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
                    nodes: int, timeout: int = 0, suffix: str = "",
                    cfg_path: Path | None = None,
                    extra_env: dict | None = None) -> RunResult:
        """Run arts at N nodes (self-fork launcher via cfg).

        When suffix is set, exec <bin_name>_arts_<suffix> instead of <bin_name>_arts.
        cfg_path overrides the cfg looked up from `nodes` via self._arts_mn_cfgs
        (which only maps the capacity MN_RANKS int/io keys) -- callers outside
        the capacity matrix (e.g. a "_sc" scalability-series node) must pass
        the absolute cfg path explicitly; correctness never sets it.
        extra_env layers caller-supplied env vars on top (e.g. counter_folder).
        """
        to = timeout or self.timeout
        cfg_src = cfg_path or (REPO / "configs" / self._arts_mn_cfgs[nodes])
        log_tag = f"arts_{suffix}_mn{nodes}" if suffix else f"arts_mn{nodes}"
        logfile = self.logdir / f"{case_name}.{log_tag}.log"
        env = os.environ.copy()
        env["ARTS_CONFIG"] = str(cfg_src)
        env["OMP_NUM_THREADS"] = "4"
        if extra_env:
            env.update(extra_env)
        # No port override: a local multinode run claims its own block (the
        # runtime probes and slides before spawning the peers), so nothing here
        # has to keep runs apart by handing out bases.
        arts_bin = f"{bin_name}_arts_{suffix}" if suffix else f"{bin_name}_arts"
        cmd = (
            f"cd {SCRATCH} && "
            f"timeout -k 1 {to} {self.apps_dir / arts_bin} " + " ".join(args)
        )
        result = self._run(cmd, env, logfile, wall_timeout=to)
        # A timed-out run can leave rank processes behind (a hung rank can
        # survive SIGTERM); reap them so they cannot interfere with later
        # cases or hold CPU.
        self._reap_exe(self.apps_dir / arts_bin)
        return result

    def run_xsocr_mpi(self, case_name: str, bin_name: str, args: list[str],
                      np: int, timeout: int = 0, cfg_path: Path | None = None,
                      tpn: int | None = None) -> RunResult:
        """Run xsocr at N MPI ranks (mpirun launcher).

        cfg_path/tpn override the cfg/per-rank-core-count looked up from `np`
        via self._xsocr_mn_cfgs/self._tpn (which only map the capacity
        MN_RANKS int keys) -- callers outside the capacity matrix (e.g. a
        "_sc" scalability-series node) must pass both explicitly;
        correctness never sets them.
        """
        to = timeout or self.timeout
        logfile = self.logdir / f"{case_name}.xsocr_mpi{np}.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        xsocr_cfg = cfg_path or (REPO / "configs" / self._xsocr_mn_cfgs[np])
        cmd = (
            f"cd {SCRATCH} && "
            f"timeout -k 1 {to} {mpirun_prefix(np)} {pin_wrap(tpn or self._tpn[np])} "
            f"{self.apps_dir / f'{bin_name}_xsocr'} -ocr:cfg {xsocr_cfg} "
            + " ".join(args)
        )
        result = self._run(cmd, env, logfile, wall_timeout=to)
        self._reap_exe(self.apps_dir / f"{bin_name}_xsocr")
        return result

    def run_ocrvx_mpi(self, case_name: str, bin_name: str, args: list[str],
                      np: int = 1, timeout: int = 0,
                      tpn: int | None = None, tbb: int | None = None) -> RunResult:
        """Run ocrvx binary; np > 1 uses mpirun (ocr-vx MPI transport).

        tpn/tbb override the per-rank taskset block and OCRVX_NUM_THREADS looked
        up from the capacity tables (_tpn[np] / _ocrvx_tbb_threads[np]); the
        scalability (_sc) family must pass both so ocr-vx gets the same 4-worker
        / 6-core-per-rank budget as arts/xsocr instead of the capacity budget."""
        to = (timeout or self.timeout) * _OCRVX_TIMEOUT_MULT
        suffix = f"_ocrvx_mpi{np}" if np > 1 else "_ocrvx"
        logfile = self.logdir / f"{case_name}{suffix}.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        env["OCRVX_NUM_THREADS"] = str(tbb if tbb is not None
                                        else self._ocrvx_tbb_threads[np])
        ocrvx_bin = self.apps_dir / f"{bin_name}_ocrvx"
        if np > 1:
            block = tpn if tpn is not None else self._tpn[np]
            launcher = f"{mpirun_prefix(np)} {pin_wrap(block)} {ocrvx_bin}"
        else:
            # No native pinning: the taskset IS placement.  Confine to exactly
            # as many cores as threads, or a reduced-width run floats over the
            # whole machine and measures a wider-cache configuration than its
            # peers.
            width = tbb if tbb is not None else _NCORES_for(self.target)
            launcher = f"taskset -c 0-{width - 1} {ocrvx_bin}"
        cmd = (
            f"cd {SCRATCH} && "
            f"timeout -k 1 {to} {launcher} " + " ".join(args)
        )
        result = self._run(cmd, env, logfile, wall_timeout=to)
        self._reap_exe(self.apps_dir / f"{bin_name}_ocrvx")
        return result

    def run_baseline(self, case_name: str, spec, timeout: int = 0) -> RunResult:
        # timeout: per-case single-node wall budget (s); 0 -> the global
        # default (mirrors run_ocr's/run_ocrvx_mpi's contract).
        to = timeout or self.timeout
        logfile = self.logdir / f"{case_name}.baseline.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        if spec.np > 1 or spec.force_mpirun:
            # Multinode: per-rank disjoint blocks; single-rank force_mpirun:
            # confine the one rank to cores 0..N-1 like the other references.
            pin = (f"{pin_wrap(self._tpn[spec.np])} " if spec.np in self._tpn
                   else f"{self._pin_single} ")
            launcher = f"{mpirun_prefix(spec.np)} {pin}{self.base_dir / spec.bin}"
        else:
            launcher = f"{self._pin_single} {self.base_dir / spec.bin}"
        cmd = (
            f"cd {SCRATCH} && "
            f"timeout -k 1 {to} {launcher} " + " ".join(spec.args)
        )
        return self._run(cmd, env, logfile, wall_timeout=to)


@dataclass(frozen=True)
class Runtime:
    key: str          # column id, e.g. "ocr_val_wb", "xsocr", "ocrvx", "baseline"
    label: str        # display
    kind: str         # "arts" | "xsocr" | "ocrvx" | "baseline"
    suffix: str = ""  # arts variant binary suffix (empty for non-arts)


ARTS_VARIANTS = [
    "ocr_val_wt", "ocr_val_wb",
    "wrf_val_wt",
    "ocr_inv_wt", "ocr_inv_wb",
    "ocr_excl_purge", "ocr_excl_retain",
]
RUNTIMES: list[Runtime] = (
    [Runtime(s, f"arts_{s}", "arts", s) for s in ARTS_VARIANTS]
    + [Runtime("xsocr", "xsocr", "xsocr"),
       Runtime("ocrvx", "ocrvx", "ocrvx")]
)
