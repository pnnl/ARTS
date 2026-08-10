"""Run a whole campaign: build once, then measure cell by cell.

Every campaign writes a track file as it goes, so a detached run can be
inspected or resumed after the session that started it is gone.  The disk is
the source of truth; the console is a convenience.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from artsrun import check, report
from artsrun.build import (
    BuildPlan, build, check_build_dir, check_counter_config, plan_targets,
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
    def build_plan(self) -> BuildPlan:
        check_build_dir(self.build_dir)
        if self.counterset and self.counterset.enabled:
            # Written next to the run so the file the build was configured
            # against is the one the results can be read back through.
            wanted = write_counter_config(self.counterset, self.run_dir / "cfg")
            check_counter_config(self.build_dir, wanted)
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
    def run(self, *, on_line=None, resume_from: Path | None = None) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        say = on_line or (lambda _msg: None)

        (self.run_dir / "selection.yaml").write_text(
            json.dumps(self.selection.model_dump(mode="json"), indent=2)
        )

        plan = self.build_plan()
        say(f"building {len(plan.targets)} targets in {self.build_dir}")
        build(plan, on_line=lambda line: say(line))

        cells, skipped = self.cells()
        done_keys = _completed_keys(resume_from) if resume_from else set()
        if done_keys:
            before = len(cells)
            cells = [c for c in cells if c.key not in done_keys]
            say(f"resuming: {before - len(cells)} cells already complete")

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
        poll = (
            self.profile.slurm.poll_interval_s
            if self.profile.slurm else 5.0
        )
        scheduler = Scheduler(
            backend, cells, WallCache(wall_cache_path()),
            on_event=on_event, poll_interval_s=poll,
        )
        try:
            results = scheduler.run()
        finally:
            track.close()

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


def _completed_keys(run_dir: Path) -> set[str]:
    """Cells a previous run finished, read from its track file."""
    track = run_dir / "track.jsonl"
    if not track.is_file():
        return set()
    keys = set()
    for line in track.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("event") == "finished" and row.get("status") == Status.OK.value:
            keys.add(row["cell"])
    return keys


def summarize_skips(skipped: list[Skipped]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in skipped:
        counts[s.reason] = counts.get(s.reason, 0) + 1
    return counts
