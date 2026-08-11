"""Admission control over a node budget.

Cells are rigid parallel tasks: each occupies its whole node count for its
whole duration.  Where several share a budget the queue is ordered widest
first -- list scheduling's usual defence against a wide cell arriving last and
running alone -- and by node-seconds once a campaign has observed wall times,
which packs the same budget more tightly.

A budget of one node makes the campaign strictly serial, so no separate serial
mode exists; it also makes that ordering pointless, and there the queue runs
narrowest first instead so the cheapest evidence arrives first.

A backend backed by a real scheduler declares an unbounded capacity: metering
Slurm's queue from here would only duplicate, badly, the thing Slurm is for.
Admission then degenerates to submitting everything at once, and this loop's
remaining job is to watch the outcomes come back.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from artsrun.run.types import Cell, CellResult, Status


class WallCache:
    """Observed wall times, keyed by what determines them."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, float] = {}
        if path.is_file():
            try:
                self.data = json.loads(path.read_text())
            except (OSError, ValueError):
                self.data = {}

    @staticmethod
    def key(cell: Cell) -> str:
        return f"{cell.app.key}@{cell.nodes}n/{cell.entry.key}"

    def get(self, cell: Cell) -> float | None:
        return self.data.get(self.key(cell))

    def put(self, cell: Cell, wall: float) -> None:
        self.data[self.key(cell)] = round(wall, 3)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        except OSError:
            pass


def order(cells: Iterable[Cell], cache: WallCache, *, capacity: int = 0) -> list[Cell]:
    """Widest first; by node-seconds where a prior run measured them.

    A backend that runs one cell at a time has nothing to pack, so ordering
    there decides only what a watcher sees first — and the cheapest cells are
    the ones worth seeing first, since a mistake shows up in them soonest.
    Narrowest first in that case.

    Where cells do share a budget, the two keys have to be in the same unit.
    A cell nobody has timed is estimated from the cells somebody has, rather
    than standing in with its node count: a bare count cannot outrank any
    real node-second figure, so one measured narrow cell would otherwise
    displace every unmeasured wide one — the exact ordering this defends
    against.
    """
    cells = list(cells)
    if capacity == 1:
        return sorted(cells, key=lambda c: (c.nodes, c.key))

    measured = sorted(cache.get(c) for c in cells if cache.get(c))
    typical = measured[len(measured) // 2] if measured else 1.0

    def weight(c: Cell) -> tuple[float, int, str]:
        area = c.nodes * (cache.get(c) or typical)
        return (-area, -c.nodes, c.key)

    return sorted(cells, key=weight)


class Scheduler:
    def __init__(
        self,
        backend,
        cells: list[Cell],
        cache: WallCache,
        *,
        on_event: Callable[[str, CellResult], None] | None = None,
        poll_interval_s: float = 5.0,
        stop: "threading.Event | None" = None,
    ):
        self.backend = backend
        # Asked to stop from another thread: nothing is torn down here, the
        # loop simply stops admitting and lets what is running end.
        self.stop = stop or threading.Event()
        self.cache = cache
        self.pending = order(cells, cache,
                             capacity=getattr(backend, "capacity", 0))
        self.in_flight: list[CellResult] = []
        self.done: list[CellResult] = []
        self.on_event = on_event or (lambda kind, result: None)
        self.poll_interval_s = poll_interval_s
        # A synchronous backend is mid-submit when a cell starts, so the only
        # road its start announcement has runs through the backend itself.
        if hasattr(backend, "notify"):
            backend.notify = self.on_event

    @property
    def used(self) -> int:
        return sum(self.backend.cost(r.cell) for r in self.in_flight)

    def _admit(self) -> bool:
        """Submit whatever fits in the remaining budget; True if any went out."""
        moved = False
        for cell in list(self.pending):
            # A backend that runs a cell synchronously returns from submit()
            # only when that cell is done, so this loop is where a whole
            # campaign passes; checking only in run() would notice a stop
            # after the last cell rather than before the next one.
            if self.stopped:
                break
            cost = self.backend.cost(cell)
            if cost > self.backend.capacity:
                # A cell wider than the whole budget can never be admitted.
                self.pending.remove(cell)
                result = CellResult(
                    cell=cell,
                    status=Status.SKIPPED,
                    note=(
                        f"needs {cost} nodes but the budget is "
                        f"{self.backend.capacity}"
                    ),
                )
                self.done.append(result)
                self.on_event("skipped", result)
                moved = True
                continue
            if self.used + cost > self.backend.capacity:
                continue
            self.pending.remove(cell)
            result = self.backend.submit(cell)
            moved = True
            if result.ran or result.status is Status.SKIPPED:
                self._finish(result)
            else:
                self.in_flight.append(result)
                self.on_event("submitted", result)
        return moved

    def _finish(self, result: CellResult) -> None:
        if result.wall_s:
            self.cache.put(result.cell, result.wall_s)
        self.done.append(result)
        self.on_event("finished", result)

    def _reap(self) -> bool:
        finished = False
        for result in list(self.in_flight):
            was = result.status
            updated = self.backend.poll(result)
            if updated.ran or updated.status is Status.SKIPPED:
                self.in_flight.remove(result)
                self._finish(updated)
                finished = True
            elif (updated.status is not was
                  and updated.status in (Status.RUNNING, Status.ENDING)):
                self.on_event(
                    "running" if updated.status is Status.RUNNING
                    else "ending", updated)
        return finished

    @property
    def stopped(self) -> bool:
        return self.stop.is_set()

    def run(self) -> list[CellResult]:
        """Admit and reap until the queue drains, or until asked to stop.

        A stop leaves the cells it never reached alone rather than recording
        them as anything: they were not measured, and a run that ends early is
        described by what it did measure.  Whatever is already in flight is
        told to end -- a cell that has to be interrupted is worth nothing, and
        waiting for one is what a stop is asking not to do.
        """
        try:
            while (self.pending or self.in_flight) and not self.stopped:
                self._admit()
                if not self.in_flight:
                    continue
                if not self._reap():
                    time.sleep(self.poll_interval_s)
        finally:
            self.cache.save()
            self.backend.shutdown()
        return self.done

    @property
    def unreached(self) -> int:
        """Cells a stop left in the queue."""
        return len(self.pending)
