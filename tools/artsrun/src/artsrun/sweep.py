"""Run a sweep: one probe, an argument matrix, every arm, fixed node count.

Deliberately NOT built on the campaign's selection surfaces: `expand()` and
`plan_targets()` resolve names against the plane and the benchset, and a sweep
runs argument points that exist nowhere but its own spec, on arms (the
ablation twins) the plane deliberately does not offer.  The sweep builds its
cells and its target list by hand and reuses everything downstream — the
scheduler, the backends, the track file, extraction and voting, the report
writers, resume — unchanged.

Cell order is the experiment's own: repeat-block major, and inside a block the
arm order rotates per (repeat, point), so machine drift averages over arms
(the A/B interleaving discipline).  The scheduler is told to keep it.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from artsrun import check, report
from artsrun.build import (BuildError, BuildPlan, available_targets, build,
                           configure_counters, counter_mismatch,
                           ensure_build_dir)
from artsrun.model.catalog import Catalog, Version
from artsrun.model.counters import Counterset
from artsrun.model.profile import Launcher, Profile
from artsrun.model.selection import Selection
from artsrun.model.sweep import SweepSpec
from artsrun.paths import default_build_dir, logs_root, wall_cache_path
from artsrun.render import config_for, write_arts_cfg, write_configs, \
    write_counter_config
from artsrun.run.manifest import write_manifest
from artsrun.run.scheduler import Scheduler, WallCache
from artsrun.run.types import Cell, CellResult, Status

SPEC_FILE = "sweep.json"


@dataclass
class SweepCampaign:
    spec: SweepSpec
    catalog: Catalog
    profile: Profile
    build_dir: Path
    run_dir: Path
    counterset: Counterset | None = None
    stop_requested: threading.Event = field(default_factory=threading.Event)

    def request_stop(self) -> None:
        self.stop_requested.set()
        backend = getattr(self, "_backend", None)
        if backend is not None and hasattr(backend, "abort"):
            backend.abort()

    @classmethod
    def prepare(cls, spec: SweepSpec, catalog: Catalog, profile: Profile, *,
                counterset: Counterset | None = None,
                build_dir: Path | None = None,
                run_dir: Path | None = None) -> "SweepCampaign":
        spec.validate_against(catalog, profile)
        bd = (build_dir or default_build_dir()).expanduser().resolve()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(
            spec=spec, catalog=catalog, profile=profile, build_dir=bd,
            run_dir=run_dir or (logs_root() / f"{stamp}-sweep-{spec.name}"),
            counterset=counterset,
        )

    # -- what runs ---------------------------------------------------------
    def entries(self):
        return [SweepSpec.arm_entry(a) for a in self.spec.arms]

    def apps_dir(self) -> Path:
        return self.build_dir / "benchmarks" / "apps"

    def counters_dir(self, cell) -> Path:
        return self.run_dir / "counters" / cell.slug

    def cells(self) -> list[Cell]:
        spec = self.spec
        entries = self.entries()
        node_counts = sorted({spec.nodes,
                              *(p.nodes for p in spec.points if p.nodes)})
        configs = {n: write_configs(self.profile, n, self.run_dir / "cfg")
                   for n in node_counts}
        counters_on = bool(self.counterset and self.counterset.enabled)
        apps = {p.name: spec.resolved_app(self.catalog, p)
                for p in spec.points}

        cells: list[Cell] = []
        for rep in range(1, spec.repeats + 1):
            for pi, point in enumerate(spec.points):
                app = apps[point.name]
                nodes = point.nodes or spec.nodes
                # Rotate the arm order per (repeat, point): over the whole
                # sweep every arm occupies every position in the block
                # equally often, which is what neutralizes slow drift.
                shift = (rep - 1 + pi) % len(entries)
                for entry in entries[shift:] + entries[:shift]:
                    cell = Cell(
                        entry=entry,
                        app=app,
                        nodes=nodes,
                        repeat=rep,
                        binary=self.apps_dir()
                        / entry.binary(app.binary, hinted=False),
                        args=app.args_for(nodes),
                        timeout_s=app.timeout_for(nodes)
                        or self.profile.cell_timeout_s,
                        cfg=config_for(entry.kind, configs[nodes]),
                    )
                    if counters_on:
                        from dataclasses import replace

                        out = self.counters_dir(cell)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        cfg = write_arts_cfg(
                            self.profile, cell.nodes,
                            self.run_dir / "cfg" / f"arts_{cell.slug}.cfg",
                            counter_folder=str(out),
                            capture_interval=self.counterset.capture_interval,
                        )
                        cell = replace(cell, cfg=cfg)
                    cells.append(cell)
        return cells

    # -- build -------------------------------------------------------------
    def build_plan(self, *, on_line=None, bootstrap: bool = False) -> BuildPlan:
        prefix = self._build_prefix()
        ensure_build_dir(self.build_dir, bootstrap=bootstrap, on_line=on_line,
                         prefix=prefix)
        self.counters_cfg = None
        if self.counterset:
            wanted = write_counter_config(self.counterset, self.run_dir / "cfg")
            if self.counterset.enabled:
                self.counters_cfg = wanted
            if counter_mismatch(self.build_dir, self.counterset):
                configure_counters(self.build_dir, wanted, on_line=on_line,
                                   prefix=prefix)
        row = self.catalog.apps[self.spec.app]
        targets = sorted({
            e.binary(row.binary, hinted=False) for e in self.entries()
        })
        have = available_targets(self.build_dir)
        missing = [t for t in targets if t not in have] if have else []
        return BuildPlan(build_dir=self.build_dir, targets=targets,
                         missing=missing)

    def _build_prefix(self) -> list[str]:
        if self.profile.launcher is Launcher.SLURM:
            from artsrun.run.slurm import srun_build_prefix

            return srun_build_prefix(self.profile)
        return []

    def backend(self):
        log_dir = self.run_dir / "cells"
        if self.profile.launcher is Launcher.SLURM:
            from artsrun.run.slurm import SlurmBackend

            return SlurmBackend(self.profile, log_dir)
        from artsrun.run.local import LocalBackend

        return LocalBackend(self.profile, log_dir)

    def _selection(self) -> Selection:
        """A faithful Selection for the report writers; never validated
        against the plane (ablation arms are not on it)."""
        return Selection.model_construct(
            profile=self.profile.name,
            benchset=f"sweep:{self.spec.name}",
            entries=[e.key for e in self.entries()],
            apps={f"{self.spec.app}+{p.name}": [Version.BASE]
                  for p in self.spec.points},
            node_counts=[self.spec.nodes],
            repeats=self.spec.repeats,
            build_dir=str(self.build_dir),
        )

    # -- execution ---------------------------------------------------------
    def run(self, *, on_line=None, resume: bool = False,
            retry_failed: bool = True) -> dict:
        say = on_line or (lambda _msg: None)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / SPEC_FILE).write_text(
            json.dumps(self.spec.model_dump(mode="json"), indent=2))
        (self.run_dir / "selection.yaml").write_text(
            json.dumps(self._selection().model_dump(mode="json"), indent=2))

        plan = self.build_plan(on_line=say, bootstrap=True)
        if not plan.ok:
            raise BuildError(
                "the build tree has no target for: " + ", ".join(plan.missing))
        say(f"building {len(plan.targets)} targets in {self.build_dir}")
        prefix = self._build_prefix()
        jobs = self.profile.slurm.build_cpus if prefix else None
        build(plan, on_line=say, prefix=prefix, jobs=jobs)

        cells = self.cells()
        write_manifest(self.run_dir, cells, [], self.profile, self.build_dir,
                       counters_cfg=getattr(self, "counters_cfg", None))

        carried: list[CellResult] = []
        if resume:
            from artsrun.campaign import queued_cells, recorded_results

            earlier = recorded_results(self.run_dir, cells)
            keep = [r for r in earlier
                    if r.status is Status.OK or not retry_failed]
            carried = keep
            done = {r.cell.key for r in keep}
            cells = [c for c in cells if c.key not in done]
            if self.profile.launcher is Launcher.SLURM:
                still_out = queued_cells(self.run_dir, cells)
                if still_out:
                    cells = [c for c in cells if c.key not in still_out]
                    say(f"{len(still_out)} cells still in the queue — "
                        "left to finish on their own")
            again = len(earlier) - len(keep)
            say(f"continuing: {len(keep)} cells already measured"
                + (f", {again} being retried" if again else ""))
        say(f"{len(cells)} cells to run")

        track = (self.run_dir / "track.jsonl").open("a", encoding="utf-8")

        def on_event(kind: str, result: CellResult) -> None:
            if kind == "finished":
                check.apply_to(result)
            row = {
                "t": time.time(), "event": kind, "cell": result.cell.key,
                "status": result.status.value, "rc": result.rc,
                "wall_s": round(result.wall_s, 3), "note": result.note,
            }
            if result.e2e_s is not None:
                row["e2e_s"] = round(result.e2e_s, 3)
            if result.scalar is not None:
                row["scalar"] = result.scalar
            if result.extra:
                row["extra"] = dict(result.extra)
            track.write(json.dumps(row) + "\n")
            track.flush()
            if kind == "finished":
                say(f"[{result.status.value}] {result.cell.key} "
                    f"({result.wall_s:.1f}s)")

        backend = self.backend()
        self._backend = backend
        poll = (self.profile.slurm.poll_interval_s
                if self.profile.slurm else 5.0)
        scheduler = Scheduler(
            backend, cells, WallCache(wall_cache_path()),
            on_event=on_event, poll_interval_s=poll,
            stop=self.stop_requested, keep_order=True,
        )
        try:
            results = scheduler.run()
        finally:
            track.close()
        if scheduler.stopped:
            say(f"stopped: {len(results)} cells measured, "
                f"{scheduler.unreached} never started")

        results = carried + results
        for r in carried:
            check.apply_to(r)
        groups = check.vote(
            [r for r in results if r.status is not Status.SKIPPED])
        selection = self._selection()
        report.write_results_csv(results, self.run_dir / "results.csv")
        report.write_report_json(results, groups, [], selection,
                                 self.run_dir / "report.json")
        from artsrun.model.plane import load_plane

        summary = report.write_summary(results, groups, [], selection,
                                       load_plane(),
                                       self.run_dir / "summary.txt")
        return {"run_dir": self.run_dir, "results": results,
                "groups": groups, "summary": summary}


def load_past_sweep(run_id: str) -> tuple[SweepSpec, Path]:
    run_dir = logs_root() / run_id
    saved = run_dir / SPEC_FILE
    if not saved.is_file():
        raise FileNotFoundError(f"{run_dir} has no {SPEC_FILE} to continue from")
    return SweepSpec.model_validate(json.loads(saved.read_text())), run_dir
