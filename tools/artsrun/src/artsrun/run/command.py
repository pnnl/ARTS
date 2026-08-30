"""Build the command line a cell runs under, per runtime.

Three runtimes launch three ways: ARTS spawns or remote-launches its own
ranks from the configuration it is handed, while both references are MPI
programs.  Every reference rank, on every launcher, runs inside the same
explicit CPU envelope (tools/artsrun/envelope.sh): the profile-width block of
per-core first threads, verified on the compute node itself and applied as
the rank's affinity mask before exec.  Per-core placement INSIDE the envelope
is each runtime's own mechanism — ARTS pins itself, xsocr binds through its
own configuration, ocr-vx deliberately does not pin — so the envelope is what
keeps the measured hardware identical across runtimes, and the launcher's own
affinity machinery is disabled outright (mpirun -bind-to none here, srun
--cpu-bind=none in the slurm script) so the wrapper is the only affinity
actor.

Launch models: local colocates ranks in disjoint rank-indexed blocks; ssh
distributes one rank per listed host through mpirun's ssh bootstrap; slurm
cells get their srun step prefix at the job-script level (run/slurm.py), so
build_command returns the bare wrapped rank command there.
"""

from __future__ import annotations

import functools
import shlex
import subprocess

from artsrun.model.plane import RuntimeKind
from artsrun.model.profile import Launcher, Profile
from artsrun.paths import envelope_script
from artsrun.run.types import Cell


class MpiProbeError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def _mpi_probe() -> str | None:
    """Returns "openmpi", "mpich", or None when there is no answering mpirun.

    Every mpirun-flavored decision (bootstrap, host list, binding, placement
    flags) goes through this one probe so they never see different answers.
    It runs on the host building the command, not the compute nodes the job
    will land on; that stands in fine because a cluster keeps one MPI build
    on $PATH for both the submitting host and the allocation it submits
    into, so the flavor here is the flavor there too.
    """
    try:
        out = subprocess.run(
            ["mpirun", "--version"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return "openmpi" if ("Open MPI" in out or "OpenRTE" in out) else "mpich"


def mpi_flavor() -> str:
    """The probed flavor, fatal when mpirun is absent.

    A silent default would pick one flavor's flag spellings and fail far
    from the cause — or not fail at all where the flags happen to overlap.
    """
    flavor = _mpi_probe()
    if flavor is None:
        raise MpiProbeError(
            "mpirun --version did not answer: ssh/local reference cells "
            "launch through mpirun, so a working one must be on PATH"
        )
    return flavor


def mpirun(np: int, *, launcher: Launcher, one_per_node: bool = False,
           hosts: list[str] | None = None) -> list[str]:
    """An mpirun prefix for reference ranks (local and ssh launchers only —
    slurm cells launch through srun in the job script instead)."""
    flavor = mpi_flavor()
    mpich = flavor == "mpich"
    parts = ["mpirun"]
    if launcher is Launcher.LOCAL and not mpich:
        # OpenMPI needs the flag to place more ranks than slots on one host.
        # Remote launchers never colocate — there it would only license the
        # packing the envelope's local-rank guard forbids.
        parts.append("--oversubscribe")
    if launcher is Launcher.SSH:
        # Both flavors silently switch to a slurm bootstrap when SLURM_*
        # variables are visible and then ignore the host list; ssh is forced
        # so placement follows the profile, not an environment leftover.
        parts += ["-launcher", "ssh"] if mpich else ["--mca", "plm", "rsh"]
    if hosts:
        parts += ["-hosts" if mpich else "--host", ",".join(hosts)]
    # The envelope wrapper is the only affinity actor; OpenMPI's default
    # binding is even np-dependent (core at small np, socket above), the
    # classic shape that passes a 2-rank smoke test and breaks at 8.
    parts += ["-bind-to" if mpich else "--bind-to", "none"]
    if one_per_node:
        # Neither implementation defaults to one rank per node: left alone,
        # both fill the first host to its slot count before moving on.
        parts += ["-ppn", "1"] if mpich else ["--map-by", "ppr:1:node"]
    return parts + ["-n", str(np)]


def envelope_wrap(cell: Cell, profile: Profile) -> list[str]:
    """The envelope prefix of one reference rank.

    mode=rank shifts the block by the launcher-reported rank index (local
    colocated ranks); mode=fixed is the one-rank-per-host contract every
    remote launcher runs under, and a single local rank trivially satisfies.
    """
    script = envelope_script()
    if not script.is_file():
        raise FileNotFoundError(f"envelope wrapper missing: {script}")
    colocated = profile.launcher is Launcher.LOCAL
    mode = "rank" if (colocated and cell.nodes > 1) else "fixed"
    return ["bash", str(script), str(profile.threads_per_node),
            str(cell.nodes), mode, "--"]


def build_command(cell: Cell, profile: Profile) -> list[str]:
    """The argv of one cell, without the timeout wrapper."""
    kind = cell.entry.kind
    binary = str(cell.binary)

    if kind is RuntimeKind.ARTS:
        # The runtime reads its geometry from the configuration and spawns or
        # remote-launches its own ranks (under slurm, the job script's srun
        # starts its one process per node).
        return [binary, *cell.args]

    wrap = envelope_wrap(cell, profile)
    tail = [binary]
    if kind is RuntimeKind.XSOCR and cell.cfg:
        tail += ["-ocr:cfg", str(cell.cfg)]
    tail += list(cell.args)

    launcher = profile.launcher
    if launcher is Launcher.SLURM:
        # srun launches the ranks; the prefix is part of the job script
        # (run/slurm.py), which wraps this argv for every kind identically.
        return [*wrap, *tail]

    remote = launcher is Launcher.SSH
    hosts = profile.hosts[:cell.nodes] if remote else None
    if kind is RuntimeKind.XSOCR or cell.nodes > 1:
        return [*mpirun(cell.nodes, launcher=launcher, one_per_node=remote,
                        hosts=hosts), *wrap, *tail]
    # A single ocr-vx rank needs no launcher at all (MPI singleton init).
    return [*wrap, *tail]


def with_post_verify(argv: list[str], cell: Cell) -> list[str]:
    """Chain the application's declared verifier behind the run.

    `&&` keeps both exit statuses honest: the binary's failure propagates
    untouched, and a verifier failure fails the cell the same way.  The
    shell runs in the cell's working directory, where the run's output
    files land."""
    hook = cell.app.post_verify
    if not hook:
        return argv
    return ["sh", "-c", f"{render(argv)} && {{ {hook} ; }}"]


def build_env(cell: Cell, profile: Profile) -> dict[str, str]:
    env = dict(cell.env)
    # All three runtimes carry the same env-gated end-to-end stamp — rank 0
    # prints "[E2E] <ns>" spanning application start to shutdown recognition
    # (runtime init and teardown excluded on both ends) — so every cell asks
    # for it and the log parse turns it into the cell's measured time.
    env["ARTS_E2E_MARKER"] = "1"
    if cell.entry.kind is RuntimeKind.ARTS and cell.cfg:
        env["ARTS_CONFIG"] = str(cell.cfg)
    elif cell.entry.kind is RuntimeKind.OCRVX:
        # The TBB compute-thread cap is this runtime's counterpart of a worker
        # count, so it tracks the profile's worker width.
        env["OCRVX_NUM_THREADS"] = str(profile.workers)
    if cell.entry.kind is not RuntimeKind.ARTS:
        # MVAPICH2 applies its own per-rank core affinity at MPI_Init by
        # default, which squeezes a threaded runtime onto one core; the
        # envelope is the only affinity actor, so it is disabled.  Other MPIs
        # ignore the variable.
        env["MV2_ENABLE_AFFINITY"] = "0"
    return env


def with_timeout(argv: list[str], seconds: int) -> list[str]:
    """Plain SIGTERM is a graceful shutdown the runtime may itself be stuck in;
    -k escalates to SIGKILL so a wedged run cannot outlive its budget."""
    return ["timeout", "-k", "1", str(seconds), *argv]


def render(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)
