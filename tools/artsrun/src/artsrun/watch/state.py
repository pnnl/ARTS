"""Rebuild a campaign's state from its run directory, as it happens.

The disk is the source of truth: the manifest says what was asked, the track
file says what has happened so far, and the per-cell logs carry the output.
This reader owns no campaign machinery, so it can watch a run started in this
process, a detached one, or one long finished — and a run recorded before the
manifest existed still replays from its track alone, just without commands or
verdicts.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from artsrun.check import Group, Verdict, apply_to, close, vote
from artsrun.run.manifest import Manifest
from artsrun.run.types import Cell, CellResult, Status, modern_key

_KEY = re.compile(r"^(?P<app>[^@]+)@(?P<nodes>\d+)n/(?P<entry>[^#]+)#(?P<rep>\d+)$")

_LIVE = (Status.OK, Status.FAIL, Status.TIMEOUT)


@dataclass
class CellView:
    """One cell as a watcher sees it: identity, invocation, progress."""

    key: str
    cell: Cell | None = None
    app_name: str = ""
    version: str = ""
    entry_key: str = ""
    kind: str = ""
    family: str = ""
    release: str = ""
    write: str = ""
    nodes: int = 0
    repeat: int = 0
    command: str = ""
    script: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    cfg: str = ""
    log_path: Path | None = None
    timeout_s: int = 0

    status: Status = Status.PENDING
    rc: int | None = None
    wall_s: float = 0.0
    e2e_s: float | None = None
    scalar: str | None = None
    note: str = ""
    extra: dict[str, str] = field(default_factory=dict)
    started_at: float | None = None
    ending_at: float | None = None
    finished_at: float | None = None
    verdict: Verdict | None = None
    consensus: str | None = None

    @property
    def elapsed_s(self) -> float:
        """Wall time so far: the recorded figure once there is one, the clock
        against the start while the cell is still out — and frozen where the
        drain began once nothing is computing any more."""
        if self.wall_s:
            return self.wall_s
        if self.started_at is None:
            return 0.0
        if self.status in (Status.SUBMITTED, Status.RUNNING):
            return time.time() - self.started_at
        if self.status is Status.ENDING:
            return max(0.0, (self.ending_at or self.started_at)
                       - self.started_at)
        return 0.0

    @property
    def active(self) -> bool:
        return self.status in (Status.SUBMITTED, Status.RUNNING,
                               Status.ENDING)


def _from_manifest(manifest: Manifest, cell: Cell) -> CellView:
    parts = (cell.entry.cell or "").split("/")
    family, release, write = (parts + ["", "", ""])[:3]
    meta = manifest.commands.get(cell.key, {})
    log = meta.get("log", "")
    return CellView(
        key=cell.key,
        cell=cell,
        app_name=cell.app.name,
        version=cell.app.version.value,
        entry_key=cell.entry.key,
        kind=cell.entry.kind.value,
        family=family,
        release=release,
        write=write,
        nodes=cell.nodes,
        repeat=cell.repeat,
        command=meta.get("command", ""),
        script=meta.get("script"),
        env=dict(cell.env),
        cfg=str(cell.cfg) if cell.cfg else "",
        log_path=Path(log) if log else None,
        timeout_s=cell.timeout_s,
    )


def _stub(key: str, log_dir: Path) -> CellView:
    """A cell known only from its track events (a run with no manifest)."""
    view = CellView(key=key)
    m = _KEY.match(key)
    if m:
        app = m.group("app")
        view.app_name, _, view.version = app.partition(":")
        view.entry_key = m.group("entry")
        view.nodes = int(m.group("nodes"))
        view.repeat = int(m.group("rep"))
        slug = (f"{view.app_name}.{view.version}.{view.entry_key}"
                f".{view.nodes}n.r{view.repeat}")
        # Named even before it exists: the log appears when the cell starts,
        # and a tail over a missing file simply has nothing to say yet.
        view.log_path = log_dir / f"{slug}.log"
    return view


class RunState:
    """The whole campaign, replayed from disk and refreshable in place."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.views: dict[str, CellView] = {}
        self.order: list[str] = []
        self.groups: list[Group] = []
        self.skipped = []
        self.launcher = ""
        self.cwd = ""
        self.counters_cfg: Path | None = None
        self.manifest: Manifest | None = None
        self._track_pos = 0
        self._marker_sweep_at = 0.0
        self.refresh()

    @property
    def track_path(self) -> Path:
        return self.run_dir / "track.jsonl"

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "cells"

    # -- loading -----------------------------------------------------------
    def _load_manifest(self) -> bool:
        """The manifest appears only after the build phase, so an attached
        watcher keeps asking until it does."""
        if self.manifest is not None:
            return False
        manifest = Manifest.load(self.run_dir)
        if manifest is None:
            return False
        self.manifest = manifest
        self.launcher = manifest.launcher
        self.cwd = manifest.cwd
        self.counters_cfg = manifest.counters_cfg
        self.skipped = manifest.skipped
        for cell in manifest.cells:
            if cell.key not in self.views:
                self.views[cell.key] = _from_manifest(manifest, cell)
                self.order.append(cell.key)
            else:
                # Track events got here first (the stub carries the progress);
                # graft the identity and invocation onto what they built.
                seen = self.views[cell.key]
                fresh = _from_manifest(manifest, cell)
                for name in ("cell", "app_name", "version", "entry_key",
                             "kind", "family", "release", "write", "nodes",
                             "repeat", "command", "script", "env", "cfg",
                             "log_path", "timeout_s"):
                    setattr(seen, name, getattr(fresh, name))
        return True

    def _new_events(self) -> list[dict]:
        """Track lines appended since the last look.

        Byte offsets, and only lines a newline has sealed: the writer flushes
        line-wise, but a reader polling mid-write must not parse half a row.
        """
        try:
            with self.track_path.open("rb") as fh:
                fh.seek(self._track_pos)
                data = fh.read()
        except OSError:
            return []
        last_nl = data.rfind(b"\n")
        if last_nl < 0:
            return []
        self._track_pos += last_nl + 1
        events = []
        for raw in data[:last_nl].split(b"\n"):
            try:
                events.append(json.loads(raw.decode("utf-8", errors="replace")))
            except ValueError:
                continue
        return events

    def _apply(self, row: dict) -> None:
        key = modern_key(row.get("cell", ""))
        if not key:
            return
        view = self.views.get(key)
        if view is None:
            view = _stub(key, self.log_dir)
            self.views[key] = view
            self.order.append(key)
        event = row.get("event")
        when = float(row.get("t", 0.0)) or None
        view.extra.update(row.get("extra", {}))
        view.note = row.get("note", "") or view.note
        if event == "submitted":
            view.status = Status.SUBMITTED
            view.started_at = when
        elif event == "started":
            view.status = Status.RUNNING
            view.started_at = when
        elif event == "running":
            view.status = Status.RUNNING
            view.started_at = view.started_at or when
        elif event == "ending":
            view.status = Status.ENDING
            view.started_at = view.started_at or when
            view.ending_at = when
        elif event == "finished":
            try:
                view.status = Status(row.get("status", ""))
            except ValueError:
                view.status = Status.FAIL
            view.rc = int(row.get("rc", 0))
            view.wall_s = float(row.get("wall_s", 0.0))
            raw_e2e = row.get("e2e_s")
            view.e2e_s = float(raw_e2e) if raw_e2e is not None else None
            view.scalar = row.get("scalar")
            view.finished_at = when
        elif event == "skipped":
            view.status = Status.SKIPPED
            view.verdict = Verdict.NA

    # -- consensus ---------------------------------------------------------
    def _revote(self) -> None:
        """Vote over what has finished so far, and mark each cell against its
        group's consensus.

        The group verdicts are entry-keyed, so repeats of one entry would
        shadow each other there; each row is judged against the consensus
        directly instead, so a repeat that strayed is the row that turns red.
        """
        pairs: list[tuple[CellResult, CellView]] = []
        for key in self.order:
            view = self.views[key]
            if view.cell is None or view.status not in _LIVE:
                continue
            result = CellResult(
                cell=view.cell, status=view.status, rc=view.rc or 0,
                wall_s=view.wall_s, scalar=view.scalar, note=view.note,
            )
            pairs.append((result, view))
        self.groups = vote([r for r, _ in pairs])
        by_group = {(g.app_key, g.nodes): g for g in self.groups}
        for result, view in pairs:
            group = by_group.get((result.cell.app.key, result.cell.nodes))
            view.consensus = group.consensus if group else None
            view.verdict = _cell_verdict(result, group)

    def _reconcile_markers(self) -> bool:
        """Outcomes the jobs recorded for themselves, folded in for cells
        nobody was watching when they ended.

        A fire-and-forget campaign's submitter may be long gone by the time
        a job finishes; its marker on the shared filesystem is the finish
        line, and reading it here is what makes a reattached view whole.
        Swept at a gentle pace — every outstanding cell costs one stat on a
        filesystem that may be networked.
        """
        now = time.time()
        if now - self._marker_sweep_at < 5.0:
            return False
        self._marker_sweep_at = now
        from artsrun.run.markers import marker_path, read_marker

        changed = False
        for key in self.order:
            view = self.views[key]
            if view.cell is None or not view.active:
                continue
            marker = read_marker(marker_path(self.log_dir, view.cell))
            if marker is None:
                continue
            rc, wall = marker
            status = (Status.TIMEOUT if rc in (124, 137)
                      else Status.OK if rc == 0 else Status.FAIL)
            result = CellResult(cell=view.cell, status=status, rc=rc,
                                wall_s=wall, log_path=view.log_path)
            apply_to(result)
            view.status = result.status
            view.rc = rc
            view.wall_s = wall
            view.e2e_s = result.e2e_s
            view.scalar = result.scalar
            view.note = result.note or view.note
            changed = True
        return changed

    # -- the one entry point ------------------------------------------------
    def refresh(self) -> bool:
        """Fold in whatever the disk has gained; True if anything changed."""
        changed = self._load_manifest()
        events = self._new_events()
        for row in events:
            self._apply(row)
        reconciled = self._reconcile_markers()
        if reconciled or any(row.get("event") in ("finished", "skipped")
                             for row in events):
            self._revote()
        return changed or reconciled or bool(events)

    # -- summaries ---------------------------------------------------------
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for view in self.views.values():
            out[view.status.value] = out.get(view.status.value, 0) + 1
        return out

    @property
    def broken_groups(self) -> list[Group]:
        return [
            g for g in self.groups
            if any(v in (Verdict.DISAGREE, Verdict.EXPECT_FAIL, Verdict.FAIL)
                   for v in g.verdicts.values())
        ]

    @property
    def finished(self) -> bool:
        return bool(self.views) and all(
            not v.active and v.status is not Status.PENDING
            for v in self.views.values()
        )

    def scaling(self) -> tuple[list[int], list[dict]]:
        """Best observed wall per node count, one row per (app, version, entry).

        Only completed cells measure anything, and repeats collapse to their
        best observation — the same reading the written report takes.
        """
        node_counts = sorted({v.nodes for v in self.views.values() if v.nodes})
        rows: dict[tuple, dict[int, float]] = {}
        for key in self.order:
            view = self.views[key]
            if view.status is not Status.OK or not view.wall_s:
                continue
            walls = rows.setdefault(
                (view.app_name, view.version, view.entry_key), {}
            )
            walls[view.nodes] = min(
                walls.get(view.nodes, view.wall_s), view.wall_s
            )
        out = [
            {"app": app, "version": version, "entry": entry, "walls": walls}
            for (app, version, entry), walls in rows.items()
        ]
        return node_counts, out


def _cell_verdict(result: CellResult, group: Group | None) -> Verdict:
    if result.status is not Status.OK or result.scalar is None:
        return Verdict.FAIL
    if group is None or group.consensus is None:
        return Verdict.FAIL
    app = result.cell.app
    if not close(result.scalar, group.consensus, app.scalar_kind, app.tolerance):
        return Verdict.DISAGREE
    if any(v is Verdict.EXPECT_FAIL for v in group.verdicts.values()):
        return Verdict.EXPECT_FAIL
    return Verdict.OK


class LogTail:
    """Incremental reader of one growing log file.

    The first look at an already-large file starts near the end — a viewer
    opening mid-run wants the present, not a megabyte of history — and a file
    that shrank is read again from the top, since that is a new file wearing
    an old name.
    """

    def __init__(self, path: Path, tail_bytes: int = 256 * 1024):
        self.path = Path(path)
        self.tail_bytes = tail_bytes
        self._pos: int | None = None

    def read_new(self) -> str:
        try:
            size = self.path.stat().st_size
        except OSError:
            return ""
        skipped = ""
        if self._pos is None:
            if size > self.tail_bytes:
                self._pos = size - self.tail_bytes
                skipped = f"… [{self._pos} earlier bytes not shown]\n"
            else:
                self._pos = 0
        elif size < self._pos:
            self._pos = 0
        if size == self._pos:
            return ""
        try:
            with self.path.open("rb") as fh:
                fh.seek(self._pos)
                data = fh.read()
        except OSError:
            return ""
        self._pos += len(data)
        return skipped + data.decode("utf-8", errors="replace")


def latest_run_dir(root: Path) -> Path | None:
    """The newest directory that holds a campaign, by its stamp name."""
    if not root.is_dir():
        return None
    for d in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        if (d / "track.jsonl").is_file() or (d / "selection.yaml").is_file():
            return d
    return None
