"""Expand a selection into the cells it runs, and record what it cannot.

A cell that is dropped is dropped for a structural reason — the application
cannot run multinode, or a reference runtime cannot run it at all.  The reason
is carried through to the report; nothing is dropped silently, and nothing is
dropped because a run once failed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from artsrun.model.benchset import Benchset, ResolvedApp
from artsrun.model.catalog import Catalog
from artsrun.model.plane import Plane, RuntimeKind, SelectionEntry
from artsrun.model.profile import Profile
from artsrun.model.selection import Selection
from artsrun.run.types import Cell, Skipped


def _missing_inputs(app: ResolvedApp) -> list[str]:
    """Declared inputs the application needs that this machine does not have.

    Cheap deterministic fixtures are synthesized on the spot (fixtures.stage);
    the rest are too expensive to regenerate on every host and are staged onto
    it instead.  A cell whose input is absent has nothing to measure, and
    running it would report a failure of the runtime for what is a property of
    the machine — so it is dropped the way a structurally ineligible cell is.

    The paths are declared rather than inferred from the arguments: an
    application is handed the paths it writes as well as the ones it reads,
    and only the catalog knows which is which.
    """
    from artsrun.fixtures import stage

    return stage(app.fixtures)


def _ineligible(entry: SelectionEntry, app: ResolvedApp, nodes: int) -> str | None:
    # Belt over the selection surfaces' braces: a replayed selection.yaml can
    # predate the catalog marking an application unsupported.
    if app.unsupported:
        return f"application is unsupported: {app.unsupported}"
    if nodes > 1 and app.multinode_skip:
        return f"application cannot run multinode: {app.multinode_skip}"
    if entry.kind is RuntimeKind.OCRVX and app.ocrvx_skip:
        return "application uses OCR extensions this reference does not implement"
    absent = _missing_inputs(app)
    if absent:
        return "input not staged on this machine: " + ", ".join(absent)
    return None


def expand(
    selection: Selection,
    plane: Plane,
    catalog: Catalog,
    benchset: Benchset,
    profile: Profile,
    apps_dir: Path,
    configs: dict[int, dict[str, Path]],
    cell_cfg: Callable[[Cell], Path] | None = None,
) -> tuple[list[Cell], list[Skipped]]:
    """Expand into cells.

    `cell_cfg`, when given, replaces an ARTS cell's shared configuration with
    one of its own.  Counters are the reason it exists: their output directory
    is a configuration key, so cells sharing a configuration would write over
    each other's counters.
    """
    from artsrun.render import config_for

    resolved = {a.key: a for a in benchset.resolve(catalog)}
    entries = [plane.entry(k) for k in selection.entries]

    cells: list[Cell] = []
    skipped: list[Skipped] = []

    for name, versions in selection.apps.items():
        for version in versions:
            app = resolved.get(f"{name}:{version.value}")
            if app is None:
                skipped.append(
                    Skipped("*", f"{name}:{version.value}", 0,
                            "not enabled in the benchset")
                )
                continue
            for nodes in selection.node_counts:
                for entry in entries:
                    why = _ineligible(entry, app, nodes)
                    if why:
                        skipped.append(Skipped(entry.key, app.key, nodes, why))
                        continue
                    for repeat in range(1, selection.repeats + 1):
                        cell = Cell(
                            entry=entry,
                            app=app,
                            nodes=nodes,
                            repeat=repeat,
                            binary=apps_dir / entry.binary(app.binary, hinted=False),
                            args=app.args_for(nodes),
                            timeout_s=app.timeout_for(nodes) or profile.cell_timeout_s,
                            cfg=config_for(entry.kind, configs[nodes]),
                        )
                        if cell_cfg is not None and entry.kind is RuntimeKind.ARTS:
                            cell = replace(cell, cfg=cell_cfg(cell))
                        cells.append(cell)
    return cells, skipped
