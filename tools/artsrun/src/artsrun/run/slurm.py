"""Submit each cell as its own node-exclusive Slurm job, all at once.

There is no long-lived allocation: a cell gets exactly the nodes it needs and
nothing else runs on them, so a runtime only ever observes the nodes of its own
job.  Every job is submitted up front — scheduling is Slurm's whole purpose,
so no admission budget stands between the campaign and the queue — and each
job writes its own outcome marker on the shared filesystem as it ends.  The
submitting process is thereby optional: a login node may die with the queue
full, and any later look at the run directory reconstructs what happened
from the markers alone.
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


def srun_build_prefix(profile: Profile) -> list[str]:
    """Run build work inside a small job of its own.

    On a cluster the login node is not where a configure and a full build
    belong — and a compute node is also the environment the artifacts will
    run in.  The job is deliberately narrow (one task, a few cpus, nothing
    exclusive) so it slots into whatever gap the scheduler has.
    """
    settings = profile.slurm
    cmd = [
        "srun", "-n", "1",
        f"--cpus-per-task={settings.build_cpus}",
        "--job-name=arts-build",
    ]
    partition = settings.build_partition or settings.partition
    if partition:
        cmd.append(f"--partition={partition}")
    if settings.account:
        cmd.append(f"--account={settings.account}")
    if settings.qos:
        cmd.append(f"--qos={settings.qos}")
    return cmd


def alive_jobs(job_ids: list[str]) -> set[str]:
    """The subset squeue still knows — queued or running."""
    if not job_ids:
        return set()
    proc = subprocess.run(
        ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i"],
        capture_output=True, text=True, check=False,
    )
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def marker_path(log_dir: Path, cell: Cell) -> Path:
    """Where a cell's job records its own outcome."""
    return log_dir / f"{cell.slug}.rc"


def read_marker(path: Path) -> tuple[int, float] | None:
    """(exit status, wall seconds) a job recorded for itself, if it has."""
    try:
        parts = path.read_text().split()
        rc, start, end = int(parts[0]), float(parts[1]), float(parts[2])
    except (OSError, IndexError, ValueError):
        return None
    return rc, max(0.0, end - start)


def job_script(cell: Cell, profile: Profile, marker: Path) -> str:
    """The batch script one cell runs as, byte for byte.

    The job records its own outcome — exit status and the run's own start
    and end stamps — because the process that submitted it may be gone by
    the time it ends: the marker on the shared filesystem is what any later
    look at the run reconstructs the result from.  The write lands under a
    temporary name first, so a reader never sees half a marker.
    """
    argv = build_command(cell, profile)
    env = build_env(cell, profile)
    exports = "\n".join(f"export {k}={v}" for k, v in sorted(env.items()))
    return (
        "#!/bin/bash\n"
        f"cd {scratch_dir()}\n"
        f"{exports}\n"
        's=$(date +%s.%N)\n'
        # srun carries the step onto the job's own nodes; the timeout is
        # kept inside the job as well so a wedged run dies on its own
        # budget instead of the queue's.
        f"timeout -k 1 {cell.timeout_s} srun "
        f"--ntasks-per-node=1 -N {cell.nodes} {render(argv)}\n"
        "rc=$?\n"
        f'echo "$rc $s $(date +%s.%N)" > {marker}.tmp\n'
        f"mv {marker}.tmp {marker}\n"
        "exit $rc\n"
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
    # Scheduling a full queue is Slurm's job, not this process's: every cell
    # is admitted immediately, and nothing here meters the cluster.
    UNBOUNDED = 1 << 30

    def __init__(self, profile: Profile, log_dir: Path):
        if profile.slurm is None:
            raise SlurmError("profile has no slurm section")
        self.profile = profile
        self.settings = profile.slurm
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.capacity = self.UNBOUNDED
        self._jobs: dict[str, CellResult] = {}
        self._submitted_at: dict[str, float] = {}
        self._running_at: dict[str, float] = {}

    def cost(self, cell: Cell) -> int:
        return cell.nodes

    # -- submission --------------------------------------------------------
    def submit(self, cell: Cell) -> CellResult:
        log_path = self.log_dir / cell.log_name
        proc = subprocess.run(
            sbatch_argv(cell, self.profile, log_path),
            input=job_script(cell, self.profile,
                             marker_path(self.log_dir, cell)),
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
        # The job's own marker is the authority on how it ended — written by
        # the job, so it exists whether or not anyone was watching, and it
        # carries the run's own wall rather than the queue's.
        marker = read_marker(marker_path(self.log_dir, result.cell))
        if marker is not None:
            rc, wall = marker
            result.rc = rc
            result.wall_s = wall
            # timeout(1) reports 124 for a TERM expiry and 137 when the -k
            # escalation had to kill.
            result.status = (Status.TIMEOUT if rc in (124, 137)
                             else Status.OK if rc == 0 else Status.FAIL)
            self._jobs.pop(job_id, None)
            self._running_at.pop(job_id, None)
            return result
        state, rc = self._state(job_id)
        if state in ("PENDING", "CONFIGURING"):
            result.status = Status.SUBMITTED
            return result
        if state in ("RUNNING", "RESIZING", "SUSPENDED"):
            self._running_at.setdefault(job_id, time.monotonic())
            result.status = Status.RUNNING
            return result
        if state == "COMPLETING":
            # The program is over; the scheduler is tearing the job down.
            self._running_at.setdefault(job_id, time.monotonic())
            result.status = Status.ENDING
            return result
        result.status = _FINAL.get(state, Status.FAIL)
        result.rc = rc
        # Final in the accounting but no marker: the script never reached
        # its last lines (a node failure, an external scancel).  Queue wait
        # is not runtime, so the wall runs from the moment the job was seen
        # running — submission only when it never was.
        now = time.monotonic()
        started = self._running_at.pop(
            job_id, self._submitted_at.get(job_id, now))
        result.wall_s = now - started
        if state == "UNKNOWN":
            result.note = "job left the queue with no accounting record"
        self._jobs.pop(job_id, None)
        return result

    def abort(self) -> None:
        """An explicit stop cancels what is still out — and only an explicit
        stop: someone asked for the campaign to end, queue included."""
        for job_id in list(self._jobs):
            subprocess.run(["scancel", job_id], capture_output=True, check=False)
        self._jobs.clear()

    def shutdown(self) -> None:
        """Leave the queue alone.

        The submitted jobs do not belong to this process's lifetime: each
        writes its own outcome marker, so a campaign whose submitter died —
        or merely finished — is completed by Slurm and reconstructed from
        the run directory."""
