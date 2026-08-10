"""Admission control over a node budget.

Cells are rigid parallel tasks: each occupies its whole node count for its
whole duration.  The queue is ordered widest-first, which is list scheduling's
usual defence against a wide cell arriving last and running alone; once a
campaign has observed wall times, the order switches to largest node-seconds
first, which packs the same budget more tightly.

A budget of one node makes the campaign strictly serial, so no separate serial
mode exists.
"""

from __future__ import annotations

import json
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


def order(cells: Iterable[Cell], cache: WallCache) -> list[Cell]:
    """Widest first; by node-seconds where a prior run measured them."""

    def weight(c: Cell) -> tuple[float, int, str]:
        wall = cache.get(c)
        area = c.nodes * wall if wall else float(c.nodes)
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
    ):
        self.backend = backend
        self.cache = cache
        self.pending = order(cells, cache)
        self.in_flight: list[CellResult] = []
        self.done: list[CellResult] = []
        self.on_event = on_event or (lambda kind, result: None)
        self.poll_interval_s = poll_interval_s

    @property
    def used(self) -> int:
        return sum(self.backend.cost(r.cell) for r in self.in_flight)

    def _admit(self) -> bool:
        """Submit whatever fits in the remaining budget; True if any went out."""
        moved = False
        for cell in list(self.pending):
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
            updated = self.backend.poll(result)
            if updated.ran or updated.status is Status.SKIPPED:
                self.in_flight.remove(result)
                self._finish(updated)
                finished = True
        return finished

    def run(self) -> list[CellResult]:
        try:
            while self.pending or self.in_flight:
                self._admit()
                if not self.in_flight:
                    continue
                if not self._reap():
                    time.sleep(self.poll_interval_s)
        finally:
            self.cache.save()
            self.backend.shutdown()
        return self.done
