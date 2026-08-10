"""Run cells on this machine, one at a time.

A local campaign's ranks share one host's cores, so there is nothing to
overlap: the next cell starts when the last one has been reaped.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from artsrun.model.profile import Profile
from artsrun.paths import scratch_dir
from artsrun.run.command import build_command, build_env, render, with_timeout
from artsrun.run.types import Cell, CellResult, Status

TIMEOUT_RC = 124


def reap(binary: Path) -> int:
    """Kill leftovers of one executable.

    Matched by /proc/<pid>/exe, never by name: comm is truncated at 15
    characters and a name match can also hit the caller.
    """
    killed = 0
    target = binary.resolve()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.joinpath("exe").resolve() != target:
                continue
            os.kill(int(entry.name), 9)
            killed += 1
        except (OSError, PermissionError):
            continue
    return killed


class LocalBackend:
    """Foreground execution; `submit` returns only when the cell is done."""

    def __init__(self, profile: Profile, log_dir: Path):
        self.profile = profile
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.capacity = 1

    def cost(self, cell: Cell) -> int:
        return 1

    def submit(self, cell: Cell) -> CellResult:
        argv = with_timeout(build_command(cell, self.profile), cell.timeout_s)
        env = os.environ.copy()
        env.update(build_env(cell, self.profile))
        log_path = self.log_dir / cell.log_name
        cwd = scratch_dir()
        cwd.mkdir(parents=True, exist_ok=True)

        started = time.monotonic()
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"$ {render(argv)}\n")
            log.flush()
            proc = subprocess.run(
                argv, cwd=cwd, env=env, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        wall = time.monotonic() - started

        # A timed-out run can leave ranks behind: a wedged rank survives the
        # graceful path, and a survivor holds cores and ports for whatever runs
        # next.
        reap(cell.binary)

        status = Status.TIMEOUT if proc.returncode == TIMEOUT_RC else (
            Status.OK if proc.returncode == 0 else Status.FAIL
        )
        return CellResult(
            cell=cell, status=status, rc=proc.returncode,
            wall_s=wall, log_path=log_path,
        )

    def poll(self, result: CellResult) -> CellResult:
        return result

    def shutdown(self) -> None:
        return
