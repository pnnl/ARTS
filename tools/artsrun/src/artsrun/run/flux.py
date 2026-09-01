"""Submit each cell as its own node-exclusive Flux job, all at once.

The same contract as the Slurm backend: no long-lived allocation, every job
submitted up front, and each job writes its own outcome marker on the shared
filesystem as it ends — the submitting process is optional, and any later
look at the run directory reconstructs what happened from the markers alone.

Flux differences that shape the code: job ids are F58 strings, not numbers
(forced to ASCII at submission and stored verbatim; queries print both f58
and decimal so a stored id matches either); a job's roster/rank environment
is what the ARTS runtime configures itself from, so the batch script's inner
`flux run` uses the proven `-N n -n n -c W` shape (per-resource and per-task
options cannot be combined); and the shell's affinity plugins are disabled on
every run line so the envelope wrapper (references) and the runtime's own
pinning (arts) stay the only affinity actors.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

from artsrun.model.plane import RuntimeKind
from artsrun.model.profile import FluxSettings, Profile
from artsrun.paths import scratch_dir
from artsrun.run.command import build_command, build_env, render, with_post_verify
from artsrun.run.markers import marker_path, read_marker
from artsrun.run.types import Cell, CellResult, Status

# A job that has left the queue (state INACTIVE) carries exactly one result.
_RESULT = {
    "COMPLETED": Status.OK,
    "FAILED": Status.FAIL,
    "CANCELED": Status.FAIL,
    "TIMEOUT": Status.TIMEOUT,
}

# States on the way to and through execution.  An unrecognized state is NOT
# final: a merely-queued cell must never be recorded as failed, so the poll
# leaves it unchanged and asks again.
_ACTIVE = {
    "NEW": Status.SUBMITTED,
    "DEPEND": Status.SUBMITTED,
    "PRIORITY": Status.SUBMITTED,
    "SCHED": Status.SUBMITTED,
    "RUN": Status.RUNNING,
    "CLEANUP": Status.ENDING,
}


class FluxError(RuntimeError):
    pass


def _flux_env() -> dict[str, str]:
    """Every flux invocation forces ASCII F58 ids, so the id captured at
    submission and the ids later queries print compare as equal strings."""
    env = dict(os.environ)
    env["FLUX_F58_FORCE_ASCII"] = "1"
    return env


def _run_opts(settings: FluxSettings) -> list[str]:
    """The shell options every inner `flux run` line carries.

    cpu/gpu-affinity are disabled so the envelope wrapper and the runtime's
    own pinning are the only affinity actors; mpibind (which replaces both
    where the site loads it) is disabled by name only where the profile says
    the plugin exists — an instance without it may reject the unknown name.
    """
    opts = ["-o", "cpu-affinity=off", "-o", "gpu-affinity=off"]
    if settings.mpibind:
        opts += ["-o", "mpibind=off"]
    return opts


def flux_build_prefix(profile: Profile) -> list[str]:
    """Run build work inside a small job of its own.

    Same reasoning as the Slurm build prefix: the login node is not where a
    build belongs, and a compute node is the environment the artifacts will
    run in.  One narrow non-exclusive job; `flux run` blocks until it ends,
    which is exactly how the caller treats the srun equivalent.
    """
    settings = profile.flux
    cmd = [
        "flux", "run", "-n1",
        f"-c{settings.build_cpus}",
        "--job-name=arts-build",
    ]
    if settings.build_time:
        cmd += ["-t", settings.build_time]
    queue = settings.build_queue or settings.queue
    if queue:
        cmd.append(f"--queue={queue}")
    if settings.bank:
        cmd.append(f"--bank={settings.bank}")
    return cmd


def flux_available() -> None:
    """One up-front probe so a resume aborts loudly when flux itself cannot
    answer, instead of reading an unreachable broker as an empty queue and
    resubmitting every still-queued job."""
    try:
        proc = subprocess.run(
            ["flux", "jobs", "-n", "--count=1"],
            capture_output=True, text=True, check=False, env=_flux_env(),
        )
    except OSError as exc:
        raise FluxError(f"flux is not runnable: {exc}") from exc
    if proc.returncode != 0:
        raise FluxError(
            f"flux cannot reach a broker: {proc.stderr.strip()}")


def alive_jobs(job_ids: list[str]) -> set[str]:
    """The subset the job manager still holds as active.

    One query per id, deliberately: an id the job manager has purged makes a
    combined query error, and error semantics for mixed lists vary by
    version — while an empty answer here un-filters the resume path into
    resubmitting every still-queued exclusive job.  A missing row simply
    means "not alive"; broker unavailability is excluded up front.
    """
    if not job_ids:
        return set()
    flux_available()
    alive: set[str] = set()
    for job_id in job_ids:
        proc = subprocess.run(
            ["flux", "jobs", "-n", "-o", "{id.f58}|{id.dec}|{state}", job_id],
            capture_output=True, text=True, check=False, env=_flux_env(),
        )
        if proc.returncode != 0:
            continue
        for line in proc.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 3 or job_id not in (parts[0], parts[1]):
                continue
            if parts[2] and parts[2] != "INACTIVE":
                alive.add(job_id)
    return alive


def script_path(log_dir: Path, cell: Cell) -> Path:
    """Where a cell's batch script lives — next to its log and marker, so
    the on-machine debugging artifact survives the submitter."""
    return log_dir / f"{cell.slug}.sh"


def _launch(cell: Cell, profile: Profile) -> str:
    """The line that actually starts the cell's ranks, once, on the job's
    own nodes.

    `flux run` is the step launcher for every kind: it spawns arts's one
    process per node (the runtime reads rank and roster from the task
    environment), and the site's PMI plugin is how the reference MPI ranks
    form their world.  The shape is `-N n -n n -c W`: per-resource options
    (--tasks-per-node) cannot be combined with -c, and -N alone distributes
    the n tasks one per node.  -c is on EVERY line, arts included, as the
    backstop — if the affinity off-switch were ever rejected or ignored,
    the default per-task set would be the right width instead of one core.
    Never add --label here: it prefixes every output line with the rank id,
    which silently breaks the line-anchored [E2E] stamp parse.

    A declared post-verify hook wraps OUTSIDE the flux run line, so it runs
    once on the script's node rather than once per rank.
    """
    settings = profile.flux
    width = profile.threads_per_node
    prefix = ["flux", "run", "-N", str(cell.nodes), "-n", str(cell.nodes),
              f"-c{width}", *_run_opts(settings)]
    if cell.entry.kind is not RuntimeKind.ARTS and settings.pmi:
        prefix += ["-o", f"pmi={settings.pmi}"]
    prefix += list(settings.extra_run)
    argv = with_post_verify([*prefix, *build_command(cell, profile)], cell)
    return render(argv)


def job_script(cell: Cell, profile: Profile, marker: Path) -> str:
    """The batch script one cell runs as, byte for byte — the same shape as
    the Slurm one: the job records its own outcome under a temporary name
    first, so a reader never sees half a marker."""
    launch = _launch(cell, profile)
    env = build_env(cell, profile)
    exports = "\n".join(f"export {k}={shlex.quote(str(v))}"
                        for k, v in sorted(env.items()))
    return (
        "#!/bin/bash\n"
        f"cd {scratch_dir()}\n"
        f"{exports}\n"
        f"printf '%s\\n' {shlex.quote('$ ' + launch)}\n"
        's=$(date +%s.%N)\n'
        # The timeout stays inside the job so a wedged run dies on its own
        # budget instead of the queue's.
        f"timeout -k 1 {cell.timeout_s} {launch}\n"
        "rc=$?\n"
        f'echo "$rc $s $(date +%s.%N)" > {marker}.tmp\n'
        f"mv {marker}.tmp {marker}\n"
        "exit $rc\n"
    )


def batch_argv(cell: Cell, profile: Profile, log_path: Path,
               script: Path) -> list[str]:
    """The submission command one cell goes out under.

    Only --output is named: flux routes stderr to the stdout destination
    when --error is unset (documented, and verified), which is where the
    envelope's failure lines and the runtime's own diagnostics must land
    for the checker to see them.  The time limit leaves the in-job timeout
    two minutes of room, so a wedge still writes its marker.
    """
    settings = profile.flux
    cmd = [
        "flux", "batch",
        "--exclusive",
        f"--nodes={cell.nodes}",
        f"--job-name=arts-{cell.app.name}-{cell.entry.key}-{cell.nodes}n",
        f"--output={log_path}",
        "-t", f"{cell.timeout_s + 120}s",
    ]
    if settings.queue:
        cmd.append(f"--queue={settings.queue}")
    if settings.bank:
        cmd.append(f"--bank={settings.bank}")
    return cmd + list(settings.extra_batch) + [str(script)]


class FluxBackend:
    # Scheduling a full queue is Flux's job, not this process's: every cell
    # is admitted immediately, and nothing here meters the cluster.
    UNBOUNDED = 1 << 30

    def __init__(self, profile: Profile, log_dir: Path):
        if profile.flux is None:
            raise FluxError("profile has no flux section")
        self.profile = profile
        self.settings = profile.flux
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
        script = script_path(self.log_dir, cell)
        text = job_script(cell, self.profile,
                          marker_path(self.log_dir, cell))
        # Write-once by contract: an already-submitted cell reads this path
        # when it starts, so a resume must not swap the script under it.
        # The mismatch fails THIS cell rather than raising — an exception
        # here would escape the scheduler and kill the rest of a
        # partially-completed campaign.
        if script.exists() and script.read_text() != text:
            return CellResult(
                cell=cell, status=Status.FAIL, rc=1, log_path=log_path,
                note=f"cell script changed since this campaign started: "
                     f"{script}; start a new campaign instead of resuming",
            )
        script.write_text(text)
        proc = subprocess.run(
            batch_argv(cell, self.profile, log_path, script),
            capture_output=True, text=True, check=False, env=_flux_env(),
        )
        if proc.returncode != 0:
            return CellResult(
                cell=cell, status=Status.FAIL, rc=proc.returncode,
                log_path=log_path,
                note=f"flux batch failed: {proc.stderr.strip()}",
            )
        job_id = proc.stdout.strip().split()[-1] if proc.stdout.split() else ""
        if not job_id:
            return CellResult(
                cell=cell, status=Status.FAIL, rc=1, log_path=log_path,
                note=f"could not read a job id from: {proc.stdout.strip()}",
            )
        result = CellResult(
            cell=cell, status=Status.SUBMITTED, log_path=log_path,
            note=f"job {job_id}",
        )
        self._jobs[job_id] = result
        self._submitted_at[job_id] = time.monotonic()
        result.extra["job_id"] = job_id
        return result

    # -- polling -----------------------------------------------------------
    def _state(self, job_id: str) -> tuple[str, str, str] | None:
        """(state, result, returncode) as flux renders them, or None when
        the job manager no longer knows the id.  Fields are empty strings
        while the job is active; the pipe separator is load-bearing for
        exactly that reason."""
        proc = subprocess.run(
            ["flux", "jobs", "-n", "-o",
             "{id.f58}|{id.dec}|{state}|{result}|{returncode}", job_id],
            capture_output=True, text=True, check=False, env=_flux_env(),
        )
        if proc.returncode != 0:
            return None
        for line in proc.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 5 and job_id in (parts[0], parts[1]):
                return parts[2], parts[3], parts[4]
        return None

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
        row = self._state(job_id)
        if row is not None:
            state, job_result, returncode = row
            status = _ACTIVE.get(state)
            if status is not None:
                if status in (Status.RUNNING, Status.ENDING):
                    self._running_at.setdefault(job_id, time.monotonic())
                result.status = status
                return result
            if state == "INACTIVE":
                result.status = _RESULT.get(job_result, Status.FAIL)
                try:
                    result.rc = int(returncode)
                except ValueError:
                    pass
                now = time.monotonic()
                started = self._running_at.pop(
                    job_id, self._submitted_at.get(job_id, now))
                result.wall_s = now - started
                # Final in the job manager but no marker: the script never
                # reached its last lines (node failure, external cancel).
                result.note = (f"job {job_id} ended "
                               f"{job_result or 'without a result'} "
                               f"with no marker")
                self._jobs.pop(job_id, None)
                return result
            # An unrecognized non-empty state is not a verdict; ask again.
            return result
        # No row at all: purged from the job manager, or never real.  Same
        # contract as a job that left Slurm's accounting unseen.
        result.status = Status.FAIL
        result.note = "job left the queue with no marker and no job record"
        now = time.monotonic()
        started = self._running_at.pop(
            job_id, self._submitted_at.get(job_id, now))
        result.wall_s = now - started
        self._jobs.pop(job_id, None)
        return result

    def abort(self) -> None:
        """An explicit stop cancels what is still out — and only an explicit
        stop: someone asked for the campaign to end, queue included."""
        for job_id in list(self._jobs):
            subprocess.run(["flux", "cancel", job_id],
                           capture_output=True, check=False, env=_flux_env())
        self._jobs.clear()

    def shutdown(self) -> None:
        """Leave the queue alone: the submitted jobs do not belong to this
        process's lifetime — each writes its own outcome marker, so a
        campaign whose submitter died or finished is completed by Flux and
        reconstructed from the run directory."""
