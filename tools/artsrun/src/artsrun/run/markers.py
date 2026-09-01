"""The cell outcome marker: one file a job writes about itself.

Scheduler backends submit fire-and-forget, so the process that submitted a
job may be gone by the time it ends.  Each job therefore records its own
outcome — exit status plus its own start and end stamps — on the shared
filesystem, and everything that later looks at the run (poll, watch, report,
resume) reads these files as the authority.  The path contract lives here,
launcher-neutral, so the writer and every reader agree by construction.
"""

from __future__ import annotations

from pathlib import Path

from artsrun.run.types import Cell


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
