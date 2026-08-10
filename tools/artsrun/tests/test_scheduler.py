from __future__ import annotations

from pathlib import Path

from artsrun.model.benchset import ResolvedApp
from artsrun.model.catalog import AppClass, Version
from artsrun.model.plane import RuntimeKind, SelectionEntry
from artsrun.run.scheduler import Scheduler, WallCache, order
from artsrun.run.types import Cell, CellResult, Status


def _cell(nodes: int, name: str = "app") -> Cell:
    app = ResolvedApp(
        name=name, version=Version.ASBORN, binary=name, cls=AppClass.TASK,
        marker="X", scalar_re="X",
    )
    entry = SelectionEntry(
        key="ocr_val_wb", label="v", kind=RuntimeKind.ARTS,
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
    from artsrun.run.slurm import SlurmBackend

    profile = Profile.model_validate({
        "name": "t", "launcher": "slurm", "nodes": [1, 2],
        "workers": 63, "progress": 1, "ports": [25000],
        "slurm": {"budget": 2},
    })
    backend = SlurmBackend(profile, tmp_path)
    cell = _cell(2)
    # Build the argv the way submit() does, without submitting.
    assert profile.threads_per_node == 64
    script = backend._script(cell)
    assert "--ntasks-per-node=1" in script
    assert f"-N {cell.nodes}" in script
