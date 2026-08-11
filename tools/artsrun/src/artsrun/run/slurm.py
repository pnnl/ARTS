"""Submit each cell as its own node-exclusive Slurm job.

There is no long-lived allocation: a cell gets exactly the nodes it needs and
nothing else runs on them, so a runtime only ever observes the nodes of its own
job.  Concurrency comes from the scheduler's node budget, not from packing
several runs into one allocation.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from artsrun.model.profile import Profile
from artsrun.paths import scratch_dir
from artsrun.run.command import build_command, build_env, render
from artsrun.run.types import Cell, CellResult, Status

_JOB_ID = re.compile(r"(\d+)")

# Slurm job states that mean the job is no longer running.
_FINAL = {
    "COMPLETED": Status.OK,
    "FAILED": Status.FAIL,
    "CANCELLED": Status.FAIL,
    "NODE_FAIL": Status.FAIL,
    "OUT_OF_MEMORY": Status.FAIL,
    "PREEMPTED": Status.FAIL,
    "TIMEOUT": Status.TIMEOUT,
    "DEADLINE": Status.TIMEOUT,
}


class SlurmError(RuntimeError):
    pass


def job_script(cell: Cell, profile: Profile) -> str:
    """The batch script one cell runs as, byte for byte."""
    argv = build_command(cell, profile)
    env = build_env(cell, profile)
    exports = "\n".join(f"export {k}={v}" for k, v in sorted(env.items()))
    return (
        "#!/bin/bash\n"
        f"cd {scratch_dir()}\n"
        f"{exports}\n"
        # srun carries the step onto the job's own nodes; the timeout is
        # kept inside the job as well so a wedged run dies on its own
        # budget instead of the queue's.
        f"exec timeout -k 1 {cell.timeout_s} srun "
        f"--ntasks-per-node=1 -N {cell.nodes} {render(argv)}\n"
    )


def sbatch_argv(cell: Cell, profile: Profile, log_path: Path) -> list[str]:
    """The submission command one cell goes out under."""
    settings = profile.slurm
    cmd = [
        "sbatch",
        "--parsable",
        "--exclusive",
        f"--nodes={cell.nodes}",
        "--ntasks-per-node=1",
        f"--job-name=arts-{cell.app.name}-{cell.entry.key}-{cell.nodes}n",
        f"--output={log_path}",
        f"--time={max(2, (cell.timeout_s + 59) // 60 + 2)}",
    ]
    # The job already owns the node; this only keeps the step's affinity
    # from being narrowed to a single core where a cgroup task plugin
    # would otherwise do that, so it follows the run's own width.
    cmd.append(f"--cpus-per-task={profile.threads_per_node}")
    if settings.partition:
        cmd.append(f"--partition={settings.partition}")
    if settings.account:
        cmd.append(f"--account={settings.account}")
    if settings.qos:
        cmd.append(f"--qos={settings.qos}")
    return cmd + list(settings.extra_sbatch)


class SlurmBackend:
    def __init__(self, profile: Profile, log_dir: Path):
        if profile.slurm is None:
            raise SlurmError("profile has no slurm section")
        self.profile = profile
        self.settings = profile.slurm
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.capacity = self.settings.budget
        self._jobs: dict[str, CellResult] = {}
        self._submitted_at: dict[str, float] = {}

    def cost(self, cell: Cell) -> int:
        return cell.nodes

    # -- submission --------------------------------------------------------
    def submit(self, cell: Cell) -> CellResult:
        log_path = self.log_dir / cell.log_name
        proc = subprocess.run(
            sbatch_argv(cell, self.profile, log_path),
            input=job_script(cell, self.profile),
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            return CellResult(
                cell=cell, status=Status.FAIL, rc=proc.returncode,
                log_path=log_path, note=f"sbatch failed: {proc.stderr.strip()}",
            )
        m = _JOB_ID.search(proc.stdout)
        if not m:
            return CellResult(
                cell=cell, status=Status.FAIL, rc=1, log_path=log_path,
                note=f"could not read a job id from: {proc.stdout.strip()}",
            )
        job_id = m.group(1)
        result = CellResult(
            cell=cell, status=Status.SUBMITTED, log_path=log_path, note=f"job {job_id}"
        )
        self._jobs[job_id] = result
        self._submitted_at[job_id] = time.monotonic()
        result.extra["job_id"] = job_id
        return result

    # -- polling -----------------------------------------------------------
    def _state(self, job_id: str) -> tuple[str, int]:
        """(state, exit code).  squeue answers while queued or running; once the
        job leaves the queue only the accounting database still knows it."""
        proc = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            capture_output=True, text=True, check=False,
        )
        state = proc.stdout.strip().splitlines()
        if state and state[0].strip():
            return state[0].strip(), 0
        proc = subprocess.run(
            ["sacct", "-n", "-P", "-X", "-j", job_id, "-o", "State,ExitCode"],
            capture_output=True, text=True, check=False,
        )
        line = proc.stdout.strip().splitlines()
        if not line:
            return "UNKNOWN", 0
        parts = line[0].split("|")
        state = parts[0].split()[0] if parts and parts[0] else "UNKNOWN"
        rc = 0
        if len(parts) > 1 and ":" in parts[1]:
            rc = int(parts[1].split(":")[0] or 0)
        return state, rc

    def poll(self, result: CellResult) -> CellResult:
        job_id = result.extra.get("job_id")
        if not job_id:
            return result
        state, rc = self._state(job_id)
        if state in ("PENDING", "CONFIGURING"):
            result.status = Status.SUBMITTED
            return result
        if state in ("RUNNING", "COMPLETING", "RESIZING", "SUSPENDED"):
            result.status = Status.RUNNING
            return result
        result.status = _FINAL.get(state, Status.FAIL)
        result.rc = rc
        result.wall_s = time.monotonic() - self._submitted_at.get(job_id, time.monotonic())
        if state == "UNKNOWN":
            result.note = "job left the queue with no accounting record"
        self._jobs.pop(job_id, None)
        return result

    def shutdown(self) -> None:
        """Cancel anything still queued so an interrupted campaign leaves no
        jobs behind."""
        for job_id in list(self._jobs):
            subprocess.run(["scancel", job_id], capture_output=True, check=False)
        self._jobs.clear()
