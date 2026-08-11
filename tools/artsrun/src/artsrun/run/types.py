"""What a campaign runs, and what comes back."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from artsrun.model.benchset import ResolvedApp
from artsrun.model.plane import SelectionEntry


class Status(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    # The program is over but the scheduler is still tearing the job down
    # (Slurm's COMPLETING); nothing is computing, so no clock should run.
    ENDING = "ending"
    OK = "ok"
    FAIL = "fail"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


def modern_key(key: str) -> str:
    """A recorded cell key, under today's version names.

    Track files carry the version by its name at the time of writing, and the
    optimized version was once named "hinted"; a reader matching old records
    against freshly expanded cells has to speak one language.
    """
    return key.replace(":hinted@", ":optimized@")


@dataclass(frozen=True)
class Cell:
    """One measured run: a configuration, an application version, a geometry."""

    entry: SelectionEntry
    app: ResolvedApp
    nodes: int
    repeat: int
    binary: Path
    args: list[str]
    timeout_s: int
    cfg: Path | None = None
    env: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.app.key}@{self.nodes}n/{self.entry.key}#{self.repeat}"

    @property
    def slug(self) -> str:
        """Filesystem-safe cell identity, unique within a campaign."""
        return (
            f"{self.app.name}.{self.app.version.value}."
            f"{self.entry.key}.{self.nodes}n.r{self.repeat}"
        )

    @property
    def log_name(self) -> str:
        return f"{self.slug}.log"


@dataclass
class CellResult:
    cell: Cell
    status: Status
    rc: int = 0
    wall_s: float = 0.0
    log_path: Path | None = None
    scalar: str | None = None
    extra: dict[str, str] = field(default_factory=dict)
    note: str = ""

    @property
    def ran(self) -> bool:
        return self.status in (Status.OK, Status.FAIL, Status.TIMEOUT)


@dataclass(frozen=True)
class Skipped:
    """A cell that was never eligible, with the structural reason why."""

    entry_key: str
    app_key: str
    nodes: int
    reason: str
