"""Run a whole campaign: build once, then measure cell by cell.

Every campaign writes a track file as it goes, so a detached run can be
inspected or resumed after the session that started it is gone.  The disk is
the source of truth; the console is a convenience.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from artsrun import check, report
from artsrun.build import (
    BuildPlan, build, check_build_dir, configure_counters, counter_mismatch,
    plan_targets,
)
from artsrun.model.benchset import Benchset
from artsrun.model.catalog import Catalog
from artsrun.model.counters import Counterset
from artsrun.model.plane import Plane
from artsrun.model.profile import Launcher, Profile
from artsrun.model.selection import Selection
from artsrun.paths import default_build_dir, logs_root, wall_cache_path
from artsrun.render import write_configs, write_counter_config
from artsrun.run.plan import expand
from artsrun.run.scheduler import Scheduler, WallCache
from artsrun.run.types import CellResult, Skipped, Status


@dataclass
class Campaign:
    selection: Selection
    plane: Plane
    catalog: Catalog
    benchset: Benchset
    profile: Profile
    build_dir: Path
    run_dir: Path
    counterset: Counterset | None = None
    # Set from another thread to end the campaign after whatever is running.
    stop_requested: threading.Event = field(default_factory=threading.Event)

    def request_stop(self) -> None:
        """Ask the campaign to end without waiting for the rest of the queue.

        The backend is told as well, because a local cell blocks for its whole
        run and would otherwise hold the campaign open until its own timeout.
        """
        self.stop_requested.set()
        backend = getattr(self, "_backend", None)
        if backend is not None and hasattr(backend, "abort"):
            backend.abort()

    @classmethod
    def prepare(
        cls,
        selection: Selection,
        plane: Plane,
        catalog: Catalog,
        benchset: Benchset,
        profile: Profile,
        *,
        counterset: Counterset | None = None,
        build_dir: Path | None = None,
        run_dir: Path | None = None,
    ) -> "Campaign":
        selection.validate_against(plane, catalog, profile)
        bd = build_dir or (
            Path(selection.build_dir) if selection.build_dir else default_build_dir()
        )
        # Cells run from the scratch directory, so every path they carry has to
        # survive the change of working directory.
        bd = bd.expanduser().resolve()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(
            selection=selection,
            plane=plane,
            catalog=catalog,
            benchset=benchset,
            profile=profile,
            build_dir=bd,
            run_dir=run_dir or (logs_root() / stamp),
            counterset=counterset,
        )

    # -- phases ------------------------------------------------------------
    def build_plan(self, *, on_line=None) -> BuildPlan:
        check_build_dir(self.build_dir)
        if self.counterset and self.counterset.enabled:
            # Written next to the run so the file the build was configured
            # against is the one the results can be read back through.
            wanted = write_counter_config(self.counterset, self.run_dir / "cfg")
            differing = counter_mismatch(self.build_dir, self.counterset)
            if differing:
                configure_counters(self.build_dir, wanted, on_line=on_line)
        return plan_targets(
            self.selection, self.plane, self.catalog, self.benchset, self.build_dir
        )

    def apps_dir(self) -> Path:
        return self.build_dir / "benchmarks" / "apps"

    def counters_dir(self, cell) -> Path:
        return self.run_dir / "counters" / cell.slug

    def cells(self):
        counters = self.counterset
        on = bool(counters and counters.enabled)
        configs = {
            n: write_configs(self.profile, n, self.run_dir / "cfg")
            for n in self.selection.node_counts
        }
        # Counters name their output directory in the configuration, so a
        # measured campaign gives each cell one of its own; without counters
        # the cells share the node count's configuration.
        cell_cfg = None
        if on:
            from artsrun.render import write_arts_cfg

            def cell_cfg(cell):
                # The runtime creates the last path component only, so the
                # campaign's own layout has to exist before the cell runs.
                out = self.counters_dir(cell)
                out.parent.mkdir(parents=True, exist_ok=True)
                return write_arts_cfg(
                    self.profile, cell.nodes,
                    self.run_dir / "cfg" / f"arts_{cell.slug}.cfg",
                    counter_folder=str(out),
                    capture_interval=counters.capture_interval,
                )

        return expand(
            self.selection, self.plane, self.catalog, self.benchset,
            self.profile, self.apps_dir(), configs, cell_cfg=cell_cfg,
        )

    def backend(self):
        log_dir = self.run_dir / "cells"
        if self.profile.launcher is Launcher.SLURM:
            from artsrun.run.slurm import SlurmBackend

            return SlurmBackend(self.profile, log_dir)
        from artsrun.run.local import LocalBackend

        return LocalBackend(self.profile, log_dir)

    # -- execution ---------------------------------------------------------
    def run(self, *, on_line=None, resume: bool = False,
            retry_failed: bool = True) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        say = on_line or (lambda _msg: None)

        (self.run_dir / "selection.yaml").write_text(
            json.dumps(self.selection.model_dump(mode="json"), indent=2)
        )

        plan = self.build_plan(on_line=say)
        say(f"building {len(plan.targets)} targets in {self.build_dir}")
        build(plan, on_line=lambda line: say(line))

        cells, skipped = self.cells()
        # A continuation keeps what the interrupted run measured and runs the
        # rest into the same directory, so one report describes both halves.
        carried: list[CellResult] = []
        if resume:
            earlier = recorded_results(self.run_dir, cells)
            keep = [
                r for r in earlier
                if r.status is Status.OK or not retry_failed
            ]
            carried = keep
            done = {r.cell.key for r in keep}
            cells = [c for c in cells if c.key not in done]
            again = len(earlier) - len(keep)
            say(f"continuing: {len(keep)} cells already measured"
                + (f", {again} being retried" if again else ""))

        say(f"{len(cells)} cells to run, {len(skipped)} structurally ineligible")

        track = (self.run_dir / "track.jsonl").open("a", encoding="utf-8")

        def on_event(kind: str, result: CellResult) -> None:
            track.write(json.dumps({
                "t": time.time(),
                "event": kind,
                "cell": result.cell.key,
                "status": result.status.value,
                "rc": result.rc,
                "wall_s": round(result.wall_s, 3),
                "note": result.note,
            }) + "\n")
            track.flush()
            if kind == "finished":
                say(f"[{result.status.value}] {result.cell.key} "
                    f"({result.wall_s:.1f}s)")

        backend = self.backend()
        self._backend = backend
        poll = (
            self.profile.slurm.poll_interval_s
            if self.profile.slurm else 5.0
        )
        scheduler = Scheduler(
            backend, cells, WallCache(wall_cache_path()),
            on_event=on_event, poll_interval_s=poll,
            stop=self.stop_requested,
        )
        try:
            results = scheduler.run()
        finally:
            track.close()
        if scheduler.stopped:
            say(f"stopped: {len(results)} cells measured, "
                f"{scheduler.unreached} never started")

        results = carried + results
        for r in results:
            check.apply_to(r)
        groups = check.vote([r for r in results if r.status is not Status.SKIPPED])

        report.write_results_csv(results, self.run_dir / "results.csv")
        report.write_report_json(
            results, groups, skipped, self.selection, self.run_dir / "report.json"
        )
        summary = report.write_summary(
            results, groups, skipped, self.selection, self.plane,
            self.run_dir / "summary.txt",
        )
        return {
            "run_dir": self.run_dir,
            "results": results,
            "groups": groups,
            "skipped": skipped,
            "summary": summary,
        }


def recorded_results(run_dir: Path, cells: list) -> list[CellResult]:
    """What a previous run of this campaign measured, rebuilt from its track.

    A continuation has to report on both halves: consensus is a vote across
    the configurations that ran one application, so a report covering only the
    cells run after the interruption would be voting with half a ballot.  The
    track file says what finished and the per-cell logs are still there, so
    each earlier cell comes back as the result it was -- scalar included, since
    that is extracted from the log rather than remembered.
    """
    track = run_dir / "track.jsonl"
    if not track.is_file():
        return []
    by_key = {c.key: c for c in cells}
    rows: dict[str, dict] = {}
    for line in track.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("event") == "finished" and row.get("cell") in by_key:
            rows[row["cell"]] = row  # a later attempt supersedes an earlier one
    out = []
    for key, row in rows.items():
        cell = by_key[key]
        log = run_dir / "cells" / cell.log_name
        try:
            status = Status(row.get("status", ""))
        except ValueError:
            continue
        out.append(CellResult(
            cell=cell, status=status, rc=int(row.get("rc", 0)),
            wall_s=float(row.get("wall_s", 0.0)),
            log_path=log if log.is_file() else None,
            note=row.get("note", ""),
        ))
    return out


def summarize_skips(skipped: list[Skipped]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in skipped:
        counts[s.reason] = counts.get(s.reason, 0) + 1
    return counts


@dataclass(frozen=True)
class PastRun:
    """One campaign on disk, and how far it got."""

    run_id: str
    run_dir: Path
    measured: int
    ok: int
    total: int

    @property
    def finished(self) -> bool:
        """Whether it reached the end, as opposed to being stopped or dying."""
        return (self.run_dir / "summary.txt").is_file() and not self.remaining

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.measured)

    @property
    def label(self) -> str:
        state = "complete" if self.finished else f"{self.remaining} left"
        return f"{self.run_id}  ({self.measured}/{self.total} cells, {state})"


def past_runs(limit: int = 40) -> list[PastRun]:
    """Campaigns that recorded a selection, newest first.

    A run is resumable when its selection is on disk; how far it got comes
    from the track file, which is written as the campaign goes rather than at
    the end -- so a run that was stopped, or whose session died, is described
    as accurately as one that finished.
    """
    root = logs_root()
    if not root.is_dir():
        return []
    out: list[PastRun] = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        saved = d / "selection.yaml"
        if not saved.is_file():
            continue
        try:
            selection = Selection.model_validate(json.loads(saved.read_text()))
        except (OSError, ValueError):
            continue
        measured, ok = 0, 0
        track = d / "track.jsonl"
        if track.is_file():
            seen: dict[str, str] = {}
            for line in track.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("event") == "finished":
                    seen[row.get("cell", "")] = row.get("status", "")
            measured = len(seen)
            ok = sum(1 for v in seen.values() if v == Status.OK.value)
        out.append(PastRun(d.name, d, measured, ok, selection.cell_count))
        if len(out) >= limit:
            break
    return out


def load_past(run_id: str) -> tuple[Selection, Path]:
    """The selection a past run recorded, and the directory to continue in."""
    run_dir = logs_root() / run_id
    saved = run_dir / "selection.yaml"
    if not saved.is_file():
        raise FileNotFoundError(f"{run_dir} has no selection.yaml to continue from")
    return Selection.model_validate(json.loads(saved.read_text())), run_dir
