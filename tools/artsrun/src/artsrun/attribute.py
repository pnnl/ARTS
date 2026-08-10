"""Read a run's per-task tables and say which task moved the data.

The global counters say an application acquires remotely; these say which of
its tasks did.  That is the granularity a placement hint is written at, so it
is the granularity this report answers in.

A task's identity on the wire is its function address, which the tables key on
and this module resolves against the binary that produced them.  The runtime's
own bootstrap task carries no identity and so cannot appear; whatever it
accounts for is reported as a residual rather than folded into a task's row.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_NM_LINE = re.compile(r"^([0-9a-fA-F]+)\s+[tTwWiI]\s+(\S.*)$")


def symbolize(binary: Path) -> dict[int, str]:
    """Map text addresses to function names.

    Returns an empty map when the binary carries no symbols, which downgrades
    the report to raw addresses rather than failing it.
    """
    try:
        out = subprocess.run(
            ["nm", "-C", "--defined-only", str(binary)],
            capture_output=True, text=True, check=False,
        ).stdout
    except OSError:
        return {}
    table: dict[int, str] = {}
    for line in out.splitlines():
        m = _NM_LINE.match(line)
        if m:
            table.setdefault(int(m.group(1), 16), m.group(2))
    return table


@dataclass
class Task:
    """One task kind's share of the run."""

    arts_id: int
    name: str
    acquires: int = 0
    remote: int = 0
    bytes: int = 0
    runs: int = 0
    exec_ns: int = 0

    @property
    def local_hit_pct(self) -> float:
        return 100.0 * (self.acquires - self.remote) / self.acquires if self.acquires else 0.0

    @property
    def mean_exec_ns(self) -> float:
        return self.exec_ns / self.runs if self.runs else 0.0


@dataclass
class Attribution:
    """Per-task rows plus what the global counters say they should add to."""

    tasks: list[Task] = field(default_factory=list)
    global_acquires: int = 0
    global_remote: int = 0
    collisions: int = 0
    ranks: int = 0

    @property
    def attributed_acquires(self) -> int:
        return sum(t.acquires for t in self.tasks)

    @property
    def attributed_remote(self) -> int:
        return sum(t.remote for t in self.tasks)

    @property
    def residual_acquires(self) -> int:
        """Acquires no task claimed — unlabelled work, not an error by itself."""
        return max(0, self.global_acquires - self.attributed_acquires)

    @property
    def residual_remote(self) -> int:
        return max(0, self.global_remote - self.attributed_remote)

    def ranked(self) -> list[Task]:
        """Tasks worst-first: the one to hint is the one at the top."""
        return sorted(self.tasks, key=lambda t: (-t.remote, -t.bytes, t.name))


def _counter_value(blob: dict, name: str) -> int:
    body = blob.get("counters", blob)
    v = body.get(name, 0)
    return int(v.get("value", 0)) if isinstance(v, dict) else int(v)


def read(counters_dir: Path, binary: Path | None = None) -> Attribution:
    """Aggregate every rank's tables in one counter directory."""
    sym = symbolize(binary) if binary else {}
    acc: dict[int, Task] = {}
    out = Attribution()

    def task(arts_id: int) -> Task:
        if arts_id not in acc:
            acc[arts_id] = Task(arts_id, sym.get(arts_id, f"0x{arts_id:x}"))
        return acc[arts_id]

    for path in sorted(counters_dir.glob("object_n*.json")):
        blob = json.loads(path.read_text())
        out.ranks += 1
        out.collisions += int(blob.get("edt_collisions", 0)) + int(
            blob.get("db_collisions", 0)
        )
        for e in blob.get("db_objects", []):
            t = task(int(e["arts_id"]))
            t.acquires += int(e["count"])
            t.remote += int(e["cache_misses"])
            t.bytes += int(e.get("bytes", 0))
        for e in blob.get("edt_objects", []):
            t = task(int(e["arts_id"]))
            t.runs += int(e["count"])
            t.exec_ns += int(e["exec_ns"])

    for path in sorted(counters_dir.glob("n*.json")):
        if path.name.startswith("object_"):
            continue
        blob = json.loads(path.read_text())
        out.global_acquires += _counter_value(blob, "NUM_DB_ACQUIRE_LOCAL_HIT")
        out.global_acquires += _counter_value(blob, "NUM_DB_ACQUIRE_REMOTE")
        out.global_remote += _counter_value(blob, "NUM_DB_ACQUIRE_REMOTE")

    out.tasks = list(acc.values())
    return out


def cells(run_dir: Path) -> list[Path]:
    """Counter directories a campaign left behind, one per measured cell."""
    root = run_dir / "counters"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and any(p.glob("*.json")))
