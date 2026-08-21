from __future__ import annotations

from pathlib import Path

from artsrun.model.benchset import ResolvedApp
from artsrun.model.catalog import AppClass, Version
from artsrun.model.plane import RuntimeKind, SelectionEntry
from artsrun.run.scheduler import Scheduler, WallCache, order
from artsrun.run.types import Cell, CellResult, Status


def _cell(nodes: int, name: str = "app") -> Cell:
    app = ResolvedApp(
        name=name, version=Version.BASE, binary=name, cls=AppClass.TASK,
        marker="X", scalar_re="X",
    )
    entry = SelectionEntry(
        key="arts_val_wb", label="arts_val_wb", kind=RuntimeKind.ARTS,
        cell="VAL/RETAIN/WB", variant="ocr_val_wb",
    )
    return Cell(entry=entry, app=app, nodes=nodes, repeat=1,
                binary=Path("/nonexistent"), args=[], timeout_s=1)


class FakeBackend:
    """Records the concurrency it was asked to sustain."""

    def __init__(self, capacity: int, wall: float = 1.0):
        self.capacity = capacity
        self.wall = wall
        self.running: list[CellResult] = []
        self.peak = 0
        self.order: list[int] = []

    def cost(self, cell: Cell) -> int:
        return cell.nodes

    def submit(self, cell: Cell) -> CellResult:
        result = CellResult(cell=cell, status=Status.SUBMITTED)
        self.running.append(result)
        self.order.append(cell.nodes)
        self.peak = max(self.peak, sum(r.cell.nodes for r in self.running))
        return result

    def poll(self, result: CellResult) -> CellResult:
        # Everything completes on its first poll.
        if result in self.running:
            self.running.remove(result)
        result.status = Status.OK
        result.wall_s = self.wall
        return result

    def shutdown(self) -> None:
        return


def _cache(tmp_path: Path) -> WallCache:
    return WallCache(tmp_path / "wall.json")


def test_the_budget_is_never_exceeded(tmp_path):
    backend = FakeBackend(capacity=8)
    cells = [_cell(n) for n in (4, 4, 2, 1, 1, 8)]
    sched = Scheduler(backend, cells, _cache(tmp_path), poll_interval_s=0)
    results = sched.run()
    assert len(results) == len(cells)
    assert backend.peak <= 8


def test_a_budget_of_one_serializes(tmp_path):
    backend = FakeBackend(capacity=1)
    sched = Scheduler(backend, [_cell(1) for _ in range(5)], _cache(tmp_path),
                      poll_interval_s=0)
    sched.run()
    assert backend.peak == 1


def test_widest_cells_go_first_when_nothing_is_known(tmp_path):
    backend = FakeBackend(capacity=32)
    cells = [_cell(1), _cell(16), _cell(4), _cell(32), _cell(2)]
    Scheduler(backend, cells, _cache(tmp_path), poll_interval_s=0).run()
    assert backend.order == sorted(backend.order, reverse=True)


def test_a_cell_wider_than_the_budget_is_reported_not_hung(tmp_path):
    backend = FakeBackend(capacity=4)
    results = Scheduler(backend, [_cell(8), _cell(2)], _cache(tmp_path),
                        poll_interval_s=0).run()
    skipped = [r for r in results if r.status is Status.SKIPPED]
    assert len(skipped) == 1
    assert "budget" in skipped[0].note


def test_observed_walls_reorder_by_node_seconds(tmp_path):
    cache = _cache(tmp_path)
    wide_but_quick = _cell(8, "quick")
    narrow_but_slow = _cell(2, "slow")
    cache.put(wide_but_quick, 1.0)     # 8 node-seconds
    cache.put(narrow_but_slow, 100.0)  # 200 node-seconds
    ordered = order([wide_but_quick, narrow_but_slow], cache)
    assert ordered[0].app.name == "slow"


def test_walls_are_recorded_for_the_next_campaign(tmp_path):
    backend = FakeBackend(capacity=4, wall=7.5)
    cache = _cache(tmp_path)
    cells = [_cell(2)]
    Scheduler(backend, cells, cache, poll_interval_s=0).run()
    assert cache.get(cells[0]) == 7.5
    assert cache.path.is_file()


def test_a_slurm_job_asks_for_the_width_it_will_use(tmp_path):
    # The job owns the node outright; --cpus-per-task only keeps a cgroup task
    # plugin from narrowing the step to one core, so it follows workers+progress.
    from artsrun.model.profile import Profile
    from artsrun.run.slurm import job_script, sbatch_argv

    profile = Profile.model_validate({
        "name": "t", "launcher": "slurm", "nodes": [1, 2],
        "workers": 63, "progress": 1, "ports": [25000],
        "slurm": {},
    })
    cell = _cell(2)
    assert profile.threads_per_node == 64
    argv = sbatch_argv(cell, profile, tmp_path / "cell.log")
    assert "--cpus-per-task=64" in argv
    script = job_script(cell, profile, tmp_path / "cell.rc")
    assert "--ntasks-per-node=1" in script
    assert f"-N {cell.nodes}" in script
    # The job records its own outcome: the submitter may be gone by then.
    assert str(tmp_path / "cell.rc") in script
    assert ".tmp" in script  # written aside, then moved — never half a marker


def test_a_synchronous_backend_announces_a_start_through_notify(tmp_path):
    # submit() blocks for the whole cell, so the only road a start
    # announcement has runs through the backend; the scheduler hangs its own
    # event callback there.
    events = []

    class Blocking(FakeBackend):
        notify = staticmethod(lambda kind, result: None)

        def submit(self, cell):
            self.notify("started", CellResult(cell=cell, status=Status.RUNNING))
            return CellResult(cell=cell, status=Status.OK, wall_s=0.01)

    backend = Blocking(capacity=1)
    sched = Scheduler(backend, [_cell(1)], _cache(tmp_path),
                      on_event=lambda kind, r: events.append(kind),
                      poll_interval_s=0)
    sched.run()
    assert events == ["started", "finished"]


def test_a_queued_job_reports_the_moment_it_starts_running(tmp_path):
    # An asynchronous backend answers polls; the submitted->running edge is
    # the scheduler's to notice, and it must be reported once, not per poll.
    events = []

    class Queued(FakeBackend):
        def __init__(self):
            super().__init__(capacity=1)
            self.polls = 0

        def poll(self, result):
            self.polls += 1
            result.status = (Status.SUBMITTED if self.polls == 1 else
                             Status.RUNNING if self.polls <= 3 else Status.OK)
            result.wall_s = 0.01
            return result

    sched = Scheduler(Queued(), [_cell(1)], _cache(tmp_path),
                      on_event=lambda kind, r: events.append(kind),
                      poll_interval_s=0)
    sched.run()
    assert events == ["submitted", "running", "finished"]


def test_a_draining_job_is_reported_as_ending_not_running(tmp_path):
    # COMPLETING means the program is over and the scheduler is tearing the
    # job down; the edge is reported once so a watcher can stop its clock.
    events = []

    class Draining(FakeBackend):
        def __init__(self):
            super().__init__(capacity=1)
            self.polls = 0

        def poll(self, result):
            self.polls += 1
            result.status = (Status.RUNNING if self.polls == 1 else
                             Status.ENDING if self.polls <= 3 else Status.OK)
            result.wall_s = 0.01
            return result

    sched = Scheduler(Draining(), [_cell(1)], _cache(tmp_path),
                      on_event=lambda kind, r: events.append(kind),
                      poll_interval_s=0)
    sched.run()
    assert events == ["submitted", "running", "ending", "finished"]


def test_a_marker_is_the_authority_on_how_a_job_ended(tmp_path):
    # The job records its own outcome; the poll believes the marker without
    # asking the queue, so a dead controller cannot hide a finished cell.
    from artsrun.model.profile import Profile
    from artsrun.run.slurm import SlurmBackend

    profile = Profile.model_validate({
        "name": "t", "launcher": "slurm", "nodes": [1, 2],
        "workers": 2, "progress": 1, "ports": [25000], "slurm": {},
    })
    backend = SlurmBackend(profile, tmp_path)
    cell = _cell(2)
    (tmp_path / f"{cell.slug}.rc").write_text("124 10.0 70.5\n")
    result = CellResult(cell=cell, status=Status.SUBMITTED,
                        extra={"job_id": "9"})
    updated = backend.poll(result)
    assert updated.status is Status.TIMEOUT
    assert updated.rc == 124
    assert updated.wall_s == 60.5


def test_resume_never_resubmits_a_job_the_queue_still_holds(tmp_path, monkeypatch):
    import json

    from artsrun import campaign as campaign_mod

    cells = [_cell(1, "a"), _cell(1, "b")]
    rows = [
        {"t": 1.0, "event": "submitted", "cell": cells[0].key,
         "status": "submitted", "extra": {"job_id": "11"}},
        {"t": 1.0, "event": "submitted", "cell": cells[1].key,
         "status": "submitted", "extra": {"job_id": "12"}},
    ]
    (tmp_path / "track.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr("artsrun.run.slurm.alive_jobs",
                        lambda ids: {"12"})
    still = campaign_mod.queued_cells(tmp_path, cells)
    assert still == {cells[1].key}


def test_recorded_results_collect_markers_the_track_never_saw(tmp_path):
    # The continuation and the after-the-fact report both read this: a job
    # that finished with no process watching left only its marker behind.
    from artsrun.campaign import recorded_results

    cell = _cell(1)
    (tmp_path / "cells").mkdir()
    (tmp_path / "cells" / f"{cell.slug}.rc").write_text("0 5.0 8.5\n")
    out = recorded_results(tmp_path, [cell])
    assert len(out) == 1
    assert out[0].status is Status.OK
    assert out[0].wall_s == 3.5


def test_build_work_on_slurm_rides_a_small_job_not_a_node():
    # The login node is not where a configure and a full build belong, but a
    # compile is not a measurement either: it asks for a few cpus it can get
    # anywhere in the queue, never an exclusive node.
    from artsrun.model.profile import Profile
    from artsrun.run.slurm import srun_build_prefix

    profile = Profile.model_validate({
        "name": "t", "launcher": "slurm", "nodes": [1, 2],
        "workers": 63, "progress": 1, "ports": [25000],
        "slurm": {"partition": "pbatch", "build_partition": "pdebug"},
    })
    prefix = srun_build_prefix(profile)
    assert prefix[:3] == ["srun", "-n", "1"]
    assert "--exclusive" not in prefix
    assert "--cpus-per-task=8" in prefix        # the default build slot
    assert "--partition=pdebug" in prefix       # build queues elsewhere

    wider = profile.model_copy(deep=True)
    wider.slurm.build_cpus = 16
    assert "--cpus-per-task=16" in srun_build_prefix(wider)


def test_local_submit_announces_pid_and_logs_the_command(tmp_path):
    import shutil

    from artsrun.model.profile import Profile
    from artsrun.run.local import LocalBackend

    profile = Profile.model_validate({
        "name": "t", "launcher": "local", "nodes": [1],
        "workers": 1, "progress": 0,
    })
    binary = tmp_path / "echo_copy"
    shutil.copy("/bin/echo", binary)
    binary.chmod(0o755)
    cell = Cell(
        entry=_cell(1).entry, app=_cell(1).app, nodes=1, repeat=1,
        binary=binary, args=["hello"], timeout_s=5,
    )
    backend = LocalBackend(profile, tmp_path / "cells")
    events = []
    backend.notify = lambda kind, result: events.append((kind, result))
    result = backend.submit(cell)
    assert result.status is Status.OK
    assert [kind for kind, _ in events] == ["started"]
    assert events[0][1].extra.get("pid", "").isdigit()
    log = (tmp_path / "cells" / cell.log_name).read_text()
    assert log.startswith("$ ")
    assert "hello" in log


def test_a_stop_is_noticed_between_cells_not_after_the_last_one():
    # A synchronous backend returns from submit() only when the cell is done,
    # so a whole campaign passes inside one admission sweep; a stop checked
    # only around that sweep would arrive after everything had run.
    import threading

    from artsrun.run.scheduler import Scheduler, WallCache
    from artsrun.run.types import CellResult, Status

    stop = threading.Event()

    class Sequential:
        capacity = 1

        def __init__(self):
            self.ran = 0

        def cost(self, cell):
            return 1

        def submit(self, cell):
            self.ran += 1
            if self.ran == 2:
                stop.set()          # asked to stop while this one runs
            return CellResult(cell=cell, status=Status.OK, wall_s=0.01)

        def poll(self, result):
            return result

        def shutdown(self):
            pass

        def abort(self):
            pass

    cells = [_cell(nodes=1, name=f"a{i}") for i in range(6)]
    backend = Sequential()
    sched = Scheduler(backend, cells, WallCache(Path("/nonexistent")),
                      stop=stop, poll_interval_s=0)
    done = sched.run()
    assert backend.ran == 2, "the sweep kept going after the stop"
    assert len(done) == 2
    assert sched.stopped and sched.unreached == 4
