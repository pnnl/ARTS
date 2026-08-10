"""Build the command line a cell runs under, per runtime.

Three runtimes launch three ways: ARTS spawns its own ranks from the
configuration it is handed, while both references are MPI programs.  On a
machine where the "nodes" are really rank blocks of one host, the references
are taskset-confined to the same disjoint core blocks ARTS pins itself to —
otherwise a reference run floats over the whole machine and is measured on a
wider configuration than its peers.
"""

from __future__ import annotations

import functools
import shlex
import subprocess

from artsrun.model.plane import RuntimeKind
from artsrun.model.profile import Launcher, Profile
from artsrun.run.types import Cell


@functools.lru_cache(maxsize=1)
def _oversubscribe_flag() -> str:
    """OpenMPI needs a flag to place more ranks than slots; MPICH rejects it."""
    try:
        out = subprocess.run(
            ["mpirun", "--version"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    return "--oversubscribe" if ("Open MPI" in out or "OpenRTE" in out) else ""


def mpirun(np: int) -> list[str]:
    parts = ["mpirun"]
    flag = _oversubscribe_flag()
    if flag:
        parts.append(flag)
    return parts + ["-n", str(np)]


def _rank_pin(threads: int) -> list[str]:
    """Pin each MPI rank to its own contiguous core block, by rank index."""
    script = (
        f"r=${{PMI_RANK:-${{OMPI_COMM_WORLD_RANK:-0}}}}; "
        f"s=$((r*{threads})); e=$((s+{threads}-1)); "
        f'exec taskset -c $s-$e "$@"'
    )
    return ["bash", "-c", script, "_"]


def build_command(cell: Cell, profile: Profile) -> list[str]:
    """The argv of one cell, without the timeout wrapper."""
    kind = cell.entry.kind
    binary = str(cell.binary)

    if kind is RuntimeKind.ARTS:
        # The runtime reads its geometry from the configuration and spawns or
        # ssh-launches its own ranks.
        return [binary, *cell.args]

    colocated = profile.launcher is Launcher.LOCAL
    pin = _rank_pin(profile.threads_per_node) if colocated and cell.nodes > 1 else []

    if kind is RuntimeKind.XSOCR:
        cmd = [*mpirun(cell.nodes), *pin, binary]
        if cell.cfg:
            cmd += ["-ocr:cfg", str(cell.cfg)]
        return cmd + list(cell.args)

    # ocr-vx has no internal pinning: a taskset block is its placement.
    if cell.nodes > 1:
        return [*mpirun(cell.nodes), *pin, binary, *cell.args]
    if colocated:
        width = profile.threads_per_node
        return ["taskset", "-c", f"0-{width - 1}", binary, *cell.args]
    return [binary, *cell.args]


def build_env(cell: Cell, profile: Profile) -> dict[str, str]:
    env = dict(cell.env)
    if cell.entry.kind is RuntimeKind.ARTS and cell.cfg:
        env["ARTS_CONFIG"] = str(cell.cfg)
    elif cell.entry.kind is RuntimeKind.OCRVX:
        # The TBB compute-thread cap is this runtime's counterpart of a worker
        # count, so it tracks the profile's worker width.
        env["OCRVX_NUM_THREADS"] = str(profile.workers)
    return env


def with_timeout(argv: list[str], seconds: int) -> list[str]:
    """Plain SIGTERM is a graceful shutdown the runtime may itself be stuck in;
    -k escalates to SIGKILL so a wedged run cannot outlive its budget."""
    return ["timeout", "-k", "1", str(seconds), *argv]


def render(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)
