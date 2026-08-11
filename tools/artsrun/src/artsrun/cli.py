"""Command line for the experiment driver.

Every selection can be made on the command line as well as on the screens, so
a campaign is scriptable and a saved selection replays exactly.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from artsrun import store
from artsrun.model.catalog import Version, load_catalog
from artsrun.model.plane import load_plane
from artsrun.model.selection import Selection
from artsrun.paths import default_build_dir, logs_root, repo_root

app = typer.Typer(
    add_completion=False,
    help="Select, build and run ARTS experiments.",
    no_args_is_help=False,
)
profile_app = typer.Typer(help="Machine and node settings.")
benchset_app = typer.Typer(help="Application rosters and calibration.")
counterset_app = typer.Typer(help="Counter sets compiled into a build.")
config_app = typer.Typer(help="Rendered runtime configurations.")
app.add_typer(profile_app, name="profile")
app.add_typer(benchset_app, name="benchset")
app.add_typer(counterset_app, name="counterset")
app.add_typer(config_app, name="config")

console = Console()


def _fail(message: str) -> None:
    console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(1)


def _split(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_apps(spec: str | None, catalog, benchset) -> dict[str, list[Version]]:
    """`--apps quicksort,nqueens:optimized,fft:asborn+restructured` -> versions."""
    if not spec:
        return {
            a.name: benchset.versions_for(a)
            for a in catalog.rows
            if benchset.is_enabled(a)
        }
    out: dict[str, list[Version]] = {}
    for item in _split(spec):
        name, _, versions = item.partition(":")
        if name not in catalog.apps:
            _fail(f"unknown application: {name}")
        if versions:
            try:
                picked = [Version(v) for v in versions.split("+")]
            except ValueError as exc:
                _fail(str(exc))
            available = catalog.apps[name].own_versions
            bad = [v for v in picked if v not in available]
            if bad:
                _fail(
                    f"{name} has no {', '.join(v.value for v in bad)} version "
                    f"(has {', '.join(v.value for v in available)})"
                )
        else:
            picked = benchset.versions_for(catalog.apps[name])
        out.setdefault(name, []).extend(v for v in picked if v not in out.get(name, []))
    return out


def _open_editor(path: Path) -> None:
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    subprocess.run([*shlex.split(editor), str(path)], check=False)


def _revalidate_profile(name: str) -> None:
    """A file edited by hand is only useful if it still loads."""
    try:
        store.load_profile(name)
    except Exception as exc:
        console.print(f"[red]invalid after editing:[/red] {exc}")
        raise typer.Exit(1)
    console.print("[green]valid[/green]")


def _load(profile_name: str, benchset_name: str | None):
    plane = load_plane()
    catalog = load_catalog()
    try:
        profile = store.load_profile(profile_name)
    except store.NotFound as exc:
        _fail(str(exc))
    if benchset_name:
        try:
            benchset = store.load_benchset(benchset_name)
        except store.NotFound as exc:
            _fail(str(exc))
    else:
        benchset = store.default_benchset()
    return plane, catalog, profile, benchset


# --------------------------------------------------------------------------
@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    from artsrun.tui.app import run_tui

    raise typer.Exit(run_tui())


@app.command("plane")
def show_plane() -> None:
    """Print the configuration plane and what each position offers."""
    plane = load_plane()
    table = Table(title="Coherence configuration plane", expand=False)
    table.add_column("family", style="bold")
    for release, write in plane.columns():
        table.add_column(f"{plane.write_label(write)}\n{plane.release_label(release)}",
                         justify="center")
    for family in plane.families:
        row = [plane.family_labels.get(family, family.value)]
        for release, write in plane.columns():
            cell = plane.cell(family, release, write)
            if not cell.buildable:
                row.append("[dim]····[/dim]")
            else:
                names = " | ".join(e.label for e in plane.entries_of(cell))
                row.append(names)
        table.add_row(*row)
    console.print(table)
    console.print(f"\n{len(plane.entries)} selectable entries: "
                  f"{', '.join(plane.entry_keys)}")
    for cell in plane.cells:
        if not cell.buildable:
            console.print(f"[dim]{cell.key}: {cell.reason}[/dim]")
            break


@app.command("apps")
def show_apps(
    benchset: str = typer.Option(None, "--benchset", "-b"),
    enabled_only: bool = typer.Option(False, "--enabled"),
) -> None:
    """Print the application catalog with its versions."""
    catalog = load_catalog()
    bs = store.load_benchset(benchset) if benchset else store.default_benchset()
    table = Table(title="Applications", expand=False)
    table.add_column("app", no_wrap=True)
    table.add_column("kind")
    table.add_column("class")
    # The binary usually repeats the name; showing it only when it differs
    # keeps the version columns from being squeezed out of the table.
    table.add_column("binary", no_wrap=True)
    for column in ("as-born", "optimized", "restructured"):
        table.add_column(column, justify="center", no_wrap=True)
    table.add_column("args", max_width=36, overflow="ellipsis")

    from artsrun.model.catalog import Kind

    first = True
    for kind in (Kind.APPLICATION, Kind.MICROBENCH):
        rows = [a for a in catalog.rows_of(kind)
                if not enabled_only or bs.is_enabled(a)]
        if not rows:
            continue
        if not first:
            table.add_section()
        first = False
        for entry in rows:
            on = bs.is_enabled(entry)
            versions = entry.own_versions
            mark = lambda v: "[green]✓[/green]" if v in versions else "[dim]·[/dim]"  # noqa: E731
            name = entry.name if on else f"[dim]{entry.name}[/dim]"
            binary = entry.binary if entry.binary != entry.name else "[dim]·[/dim]"
            label = ("app" if kind is Kind.APPLICATION
                     else "[dim]micro[/dim]")
            table.add_row(
                name, label, entry.cls.value, binary,
                mark(Version.ASBORN), mark(Version.OPTIMIZED),
                mark(Version.RESTRUCTURED),
                " ".join(entry.args) or "[dim]none[/dim]",
            )
    console.print(table)
    console.print(
        "[dim]as-born = the application as published, its own hints included "
        "· optimized = structure untouched, EDT/DB placement hints added or "
        "changed as far as hints alone can carry it · restructured = its "
        "decomposition redesigned as a separate target[/dim]"
    )
    console.print(
        "[dim]app = a benchmark with a provenance, what a result is claimed "
        "about · micro = one runtime mechanism, or a fixture with no workload"
        "[/dim]"
    )


@app.command("run")
def run_cmd(
    # Not required: --resume and --from carry the profile the campaign was
    # selected against, and taking a different one would silently
    # re-measure against another machine.
    profile: str = typer.Option(None, "--profile", "-p"),
    benchset: str = typer.Option(None, "--benchset", "-b"),
    counters: str = typer.Option(None, "--counters", "-c",
                                 help="counter set to compile in"),
    entries: str = typer.Option(None, "--entries", "-e",
                                help="comma-separated plane entries (default: all)"),
    apps: str = typer.Option(None, "--apps", "-a",
                             help="app[:version[+version]] list (default: benchset)"),
    nodes: str = typer.Option(None, "--nodes", "-n",
                              help="comma-separated node counts (default: profile)"),
    repeats: int = typer.Option(None, "--repeats"),
    build_dir: Path = typer.Option(None, "--build-dir"),
    from_file: Path = typer.Option(None, "--from", help="replay a saved selection"),
    resume: str = typer.Option(None, "--resume", help="run id to continue"),
    retry_failed: bool = typer.Option(
        True, "--retry-failed/--keep-failed",
        help="rerun cells the earlier run failed (default), or leave them as they are",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    detach: bool = typer.Option(False, "--detach"),
    plain: bool = typer.Option(
        False, "--plain",
        help="line output instead of the live table (the default when "
             "stdout is not a terminal)",
    ),
) -> None:
    """Build everything selected, then run it cell by cell."""
    from artsrun.campaign import Campaign

    run_dir = None
    if resume:
        # Continuing means continuing THAT campaign: its selection is what was
        # measured against, and its directory is where the halves meet.
        from artsrun.campaign import load_past

        try:
            selection, run_dir = load_past(resume)
        except (FileNotFoundError, ValueError) as exc:
            _fail(str(exc))
        plane, catalog, prof, bs = _load(selection.profile, selection.benchset)
    elif from_file:
        import json

        selection = Selection.model_validate(json.loads(from_file.read_text()))
        plane, catalog, prof, bs = _load(selection.profile, selection.benchset)
    else:
        if not profile:
            _fail("--profile is required unless --resume or --from names a run")
        plane, catalog, prof, bs = _load(profile, benchset)
        keys = _split(entries) or plane.entry_keys
        unknown = [k for k in keys if k not in plane.entry_keys]
        if unknown:
            _fail(f"unknown plane entries: {', '.join(unknown)}")
        node_counts = [int(n) for n in _split(nodes)] or list(prof.nodes)
        selection = Selection(
            profile=prof.name,
            benchset=bs.name,
            entries=keys,
            apps=_parse_apps(apps, catalog, bs),
            node_counts=node_counts,
            repeats=repeats or prof.repeats,
            build_dir=str(build_dir) if build_dir else None,
        )

    try:
        selection.validate_against(plane, catalog, prof)
    except ValueError as exc:
        _fail(str(exc))

    try:
        cset = (store.load_counterset(counters) if counters
                else store.default_counterset())
    except store.NotFound as exc:
        _fail(str(exc))
    campaign = Campaign.prepare(
        selection, plane, catalog, bs, prof, counterset=cset,
        build_dir=build_dir, run_dir=run_dir,
    )

    if dry_run:
        _dry_run(campaign, selection)
        return

    if detach:
        _detach(sys.argv)
        return

    if plain or not sys.stdout.isatty():
        result = campaign.run(
            on_line=lambda line: console.print(line, highlight=False,
                                               markup=False),
            resume=bool(resume), retry_failed=retry_failed,
        )
        console.print(result["summary"])
        console.print(f"\nrun directory: {result['run_dir']}")
        return

    from artsrun.tui.runview import WatchApp

    watch = WatchApp(
        campaign.run_dir, campaign=campaign,
        runner=lambda on_line: campaign.run(
            on_line=on_line, resume=bool(resume), retry_failed=retry_failed,
            announce_cells=False,
        ),
    )
    watch.run()
    if watch.running:
        # The view was detached; the campaign belongs to this process and
        # keeps going, its lines now landing on the plain console.
        console.print(
            f"view detached; campaign continues — reattach with: "
            f"artsrun watch {campaign.run_dir.name}"
        )
        watch.line_sink = lambda line: console.print(
            line, highlight=False, markup=False)
        watch.join()
    if watch.error is not None:
        _fail(str(watch.error))
    if watch.outcome is not None:
        # The screen already showed every result and the live consensus;
        # the console keeps only the pointer to the written record.
        from artsrun.check import minority_report
        from artsrun.run.types import Status

        results = watch.outcome["results"]
        ok = sum(1 for r in results if r.status is Status.OK)
        minority = minority_report(watch.outcome["groups"])
        verdict = (f"{len(minority)} group(s) disagree" if minority
                   else "consensus clean")
        console.print(f"finished: {ok}/{len(results)} ok · {verdict}")
    console.print(f"run directory: {campaign.run_dir}")


def _dry_run(campaign, selection: Selection) -> None:
    console.print(f"[bold]{selection.cell_count} cells[/bold] "
                  f"= {len(selection.entries)} entries "
                  f"x {sum(len(v) for v in selection.apps.values())} app-versions "
                  f"x {len(selection.node_counts)} node counts "
                  f"x {selection.repeats} repeats")
    try:
        plan = campaign.build_plan()
        console.print(f"build: {len(plan.targets)} targets in {campaign.build_dir}")
        if plan.missing:
            console.print(f"[red]missing targets:[/red] {', '.join(plan.missing[:10])}")
    except Exception as exc:  # build tree may not exist yet
        console.print(f"[yellow]build check:[/yellow] {exc}")

    cells, skipped = campaign.cells()
    from artsrun.run.scheduler import WallCache, order
    from artsrun.paths import wall_cache_path

    ordered = order(cells, WallCache(wall_cache_path()),
                    capacity=campaign.backend().capacity)
    table = Table(title="Submission order (first 20)", expand=False)
    for column in ("#", "app", "version", "entry", "nodes", "timeout"):
        table.add_column(column)
    for i, cell in enumerate(ordered[:20], 1):
        table.add_row(str(i), cell.app.name, cell.app.version.value,
                      cell.entry.key, str(cell.nodes), f"{cell.timeout_s}s")
    console.print(table)
    console.print(f"{len(cells)} cells to run, {len(skipped)} structurally ineligible")
    if skipped:
        from artsrun.campaign import summarize_skips

        for reason, count in sorted(summarize_skips(skipped).items(),
                                    key=lambda kv: -kv[1]):
            console.print(f"  [dim]{count:4d}[/dim] {reason}")


def _detach(argv: list[str]) -> None:
    """Re-exec without --detach under setsid, so the campaign outlives the
    session that started it."""
    stamp = logs_root() / "detached"
    stamp.mkdir(parents=True, exist_ok=True)
    log = stamp / "campaign.log"
    inner = [a for a in argv if a != "--detach"]
    if inner and not inner[0].endswith("artsrun"):
        inner = [sys.executable, "-m", "artsrun", *inner[1:]]
    cmd = f"setsid nohup {shlex.join(inner)} </dev/null >>{shlex.quote(str(log))} 2>&1 &"
    subprocess.run(["bash", "-c", cmd], check=False, cwd=repo_root())
    console.print(f"detached; log: {log}")
    console.print("[dim]watch it live with: artsrun watch[/dim]")


@app.command("watch")
def watch_cmd(
    run: str = typer.Argument(None, help="run id (default: latest)"),
) -> None:
    """Attach a live view to a campaign — running, detached, or finished."""
    root = logs_root()
    if run:
        run_dir = root / run
        if not run_dir.is_dir():
            _fail(f"no run directory {run_dir}")
    else:
        from artsrun.watch.state import latest_run_dir

        run_dir = latest_run_dir(root)
        if run_dir is None:
            _fail(f"no runs under {root}")
    from artsrun.tui.runview import WatchApp

    WatchApp(run_dir).run()


@app.command("report")
def report_cmd(run: str = typer.Argument(None, help="run id (default: latest)")) -> None:
    """Print a campaign's summary, rebuilding it from the run directory
    when the campaign outlived its submitter."""
    root = logs_root()
    if run:
        run_dir = root / run
        if not run_dir.is_dir():
            _fail(f"no run directory {run_dir}")
    else:
        candidates = sorted(p for p in root.glob("*")
                            if (p / "selection.yaml").is_file())
        if not candidates:
            _fail(f"no runs under {root}")
        run_dir = candidates[-1]

    summary = run_dir / "summary.txt"
    track = run_dir / "track.jsonl"
    # The summary written at a campaign's end is definitive only while
    # nothing happened after it; fire-and-forget jobs finish on their own
    # schedule, so a stale or absent summary is recomputed from the disk.
    stale = (not summary.is_file()
             or (track.is_file()
                 and track.stat().st_mtime > summary.stat().st_mtime))
    if stale:
        from artsrun.campaign import reconcile

        outcome = reconcile(run_dir)
        if outcome is None and not summary.is_file():
            _fail(f"{run_dir} has no manifest to reconcile from and no "
                  "summary")
        if outcome is not None:
            console.print(outcome["summary"])
            console.print(f"\nrun directory: {run_dir}")
            return
    console.print(summary.read_text())
    console.print(f"\nrun directory: {run_dir}")


@app.command("runs")
def list_runs(limit: int = typer.Option(20, "--limit")) -> None:
    """Past campaigns, and how far each got."""
    from artsrun.campaign import past_runs

    rows = past_runs(limit)
    if not rows:
        console.print(f"no runs under {logs_root()}")
        return
    table = Table(title="Campaigns", expand=False)
    table.add_column("run", no_wrap=True)
    table.add_column("measured", justify="right")
    table.add_column("ok", justify="right")
    table.add_column("of", justify="right")
    table.add_column("state")
    for r in rows:
        table.add_row(
            r.run_id, f"{r.measured}", f"{r.ok}", f"{r.total}",
            "[green]complete[/green]" if r.finished
            else f"[yellow]{r.remaining} left[/yellow]",
        )
    console.print(table)
    console.print("\n[dim]continue one with: artsrun run --resume <run>[/dim]")


def _latest_run(run: str | None) -> Path:
    root = logs_root()
    if run:
        run_dir = root / run
        if not run_dir.is_dir():
            _fail(f"no run {run} under {root}")
        return run_dir
    candidates = sorted(p for p in root.glob("*") if p.is_dir())
    if not candidates:
        _fail(f"no runs under {root}")
    return candidates[-1]


def _binary_of_cell(run_dir: Path, slug: str) -> Path | None:
    """Rebuild the exact executable a cell ran.

    Names must come from the binary that produced the addresses: every
    configuration links its own executable, and resolving against a sibling
    yields names that look plausible and are wrong.
    """
    parts = slug.split(".")
    if len(parts) < 3:
        return None
    name, version, entry_key = parts[0], parts[1], parts[2]
    try:
        saved = json.loads((run_dir / "selection.yaml").read_text())
        selection = Selection.model_validate(saved)
        catalog = load_catalog()
        benchset = (
            store.load_benchset(selection.benchset)
            if selection.benchset
            else store.default_benchset()
        )
        resolved = {a.key: a for a in benchset.resolve(catalog)}
        app_row = resolved.get(f"{name}:{version}")
        entry = load_plane().entry(entry_key)
        if app_row is None or entry is None:
            return None
        build_dir = (
            Path(selection.build_dir) if selection.build_dir else default_build_dir()
        )
        candidate = (
            build_dir.expanduser().resolve()
            / "benchmarks" / "apps"
            / entry.binary(app_row.binary, optimized=False)
        )
    except Exception:
        return None
    return candidate if candidate.is_file() else None


@app.command("attribute")
def attribute_cmd(
    run: str = typer.Argument(None, help="run id (default: latest)"),
    cell: str = typer.Option(None, "--cell", help="cell slug (default: list them)"),
    binary: Path = typer.Option(None, "--binary", help="binary to resolve names against"),
    top: int = typer.Option(15, "--top", help="rows to show"),
) -> None:
    """Say which task kind moved the data, for one measured cell.

    Needs a campaign run with a counter set that enables the OBJ counters
    (`attribution` is the one built for this); other sets leave no per-task
    tables and the cell is reported as having none.
    """
    from artsrun import attribute as attr

    run_dir = _latest_run(run)
    found = attr.cells(run_dir)
    if not found:
        _fail(
            f"{run_dir} has no per-cell counters — rerun with "
            "`artsrun run -c attribution`"
        )
    if not cell:
        console.print(f"[bold]cells with counters in {run_dir.name}[/bold]")
        for p in found:
            console.print(f"  {p.name}")
        console.print("\nchoose one with --cell")
        return
    match = [p for p in found if p.name == cell]
    if not match:
        match = [p for p in found if cell in p.name]
    if not match:
        _fail(f"no cell matching {cell!r}; run without --cell to list them")
    target = match[0]

    binp = binary or _binary_of_cell(run_dir, target.name)

    a = attr.read(target, binp)
    if not a.tasks:
        _fail(
            f"{target.name} has no per-task tables — its run used a counter set "
            "without the OBJ counters"
        )

    table = Table(title=f"{target.name}   ({a.ranks} ranks)")
    table.add_column("task", style="bold")
    table.add_column("acquires", justify="right")
    table.add_column("remote", justify="right")
    table.add_column("local hit", justify="right")
    table.add_column("bytes", justify="right")
    table.add_column("runs", justify="right")
    for t in a.ranked()[:top]:
        table.add_row(
            t.name, f"{t.acquires:,}", f"{t.remote:,}",
            f"{t.local_hit_pct:.1f}%", f"{t.bytes:,}", f"{t.runs:,}",
        )
    console.print(table)

    if binp is None:
        console.print("[yellow]no binary found — names shown as addresses[/yellow]")
    if a.global_acquires:
        console.print(
            f"attributed {a.attributed_acquires:,} of {a.global_acquires:,} acquires "
            f"({a.attributed_remote:,} of {a.global_remote:,} remote)"
        )
        if a.residual_acquires:
            console.print(
                f"[yellow]{a.residual_acquires:,} acquires "
                f"({a.residual_remote:,} remote) carried no task identity[/yellow]"
            )
    if a.collisions:
        console.print(
            f"[yellow]{a.collisions:,} table collisions — some rows merge "
            f"distinct tasks[/yellow]"
        )


# --- profile --------------------------------------------------------------
@profile_app.command("list")
def profile_list() -> None:
    names = store.list_profiles()
    if not names:
        console.print(
            f"[dim]no profiles in {store.profiles_dir()} — create one with: "
            "artsrun profile new <name>[/dim]"
        )
        return
    for name in names:
        try:
            p = store.load_profile(name)
            console.print(f"{name:16s} {p.launcher.value:6s} nodes={p.nodes} "
                          f"{p.workers}w+{p.progress}p")
        except Exception as exc:
            console.print(f"{name:16s} [red]invalid[/red]: {exc}")


@profile_app.command("show")
def profile_show(name: str) -> None:
    console.print(store.profile_path(name).read_text())


@profile_app.command("new")
def profile_new(
    name: str,
    copy_from: str = typer.Option(None, "--from", help="start from this profile"),
    edit: bool = typer.Option(True, "--edit/--no-edit"),
) -> None:
    """Create a profile, optionally copied from an existing one."""
    from artsrun.model.profile import Profile
    from artsrun.tui import form

    if name in store.list_profiles():
        _fail(f"profile '{name}' already exists")
    if copy_from:
        base = store.load_profile(copy_from).model_dump(mode="json")
        base["name"] = name
        profile = Profile.model_validate(base)
    else:
        profile = Profile.model_validate(
            form.values_to_profile_data(name, form.blank_values())
        )
    path = store.save_profile(profile)
    console.print(f"wrote {path}")
    if edit:
        _open_editor(path)
        _revalidate_profile(name)


@profile_app.command("edit")
def profile_edit(name: str) -> None:
    """Open a profile in $EDITOR, then re-validate it."""
    path = store.profile_path(name)
    if not path.is_file():
        _fail(f"no profile '{name}'")
    _open_editor(path)
    _revalidate_profile(name)


@profile_app.command("set")
def profile_set(
    name: str,
    assignments: list[str] = typer.Argument(
        ..., help="key=value, e.g. workers=31 launcher=slurm slurm.partition=pbatch"
    ),
) -> None:
    """Change fields of a profile without opening an editor."""
    from artsrun.model.profile import Profile
    from artsrun.tui import form

    try:
        profile = store.load_profile(name)
    except store.NotFound as exc:
        _fail(str(exc))
    values = form.profile_to_values(profile)
    for assignment in assignments:
        key, sep, value = assignment.partition("=")
        if not sep:
            _fail(f"expected key=value, got '{assignment}'")
        key = key.strip()
        if key not in form.BY_KEY:
            _fail(f"unknown field '{key}'. Known: "
                  f"{', '.join(sorted(form.BY_KEY))}")
        spec = form.BY_KEY[key]
        values[key] = (
            value.strip().lower() in ("1", "true", "yes", "on")
            if spec.kind == "bool" else value.strip()
        )
    try:
        updated = Profile.model_validate(
            form.values_to_profile_data(name, values)
        )
    except Exception as exc:
        _fail(form.first_problem(exc))
    path = store.save_profile(updated)
    console.print(f"updated {path}")
    profile_show(name)


@profile_app.command("fields")
def profile_fields() -> None:
    """List the fields a profile has, with their meaning."""
    from artsrun.tui import form

    table = Table(title="Profile fields", expand=False)
    for column in ("field", "type", "section", "meaning"):
        table.add_column(column)
    for spec in form.PROFILE_FIELDS:
        table.add_row(spec.key, spec.kind, spec.section,
                      spec.help or "[dim]—[/dim]")
    console.print(table)


@profile_app.command("validate")
def profile_validate(name: str = typer.Argument(None)) -> None:
    names = [name] if name else store.list_profiles()
    bad = 0
    for n in names:
        try:
            store.load_profile(n)
            console.print(f"[green]ok[/green]    {n}")
        except Exception as exc:
            bad += 1
            console.print(f"[red]error[/red] {n}: {exc}")
    raise typer.Exit(1 if bad else 0)


# --- benchset -------------------------------------------------------------
@app.command("counters")
def show_counters(
    counterset: str = typer.Option(None, "--set", "-s"),
    enabled_only: bool = typer.Option(False, "--enabled"),
) -> None:
    """Print the counters the runtime defines, and what a set turns on."""
    from artsrun.model.counters import load_counter_catalog

    catalog = load_counter_catalog()
    cset = (store.load_counterset(counterset) if counterset
            else store.default_counterset())
    table = Table(title=f"Counters ({cset.name})", expand=False)
    table.add_column("counter", no_wrap=True)
    table.add_column("group")
    table.add_column("unit")
    table.add_column("mode", justify="center")
    table.add_column("level", justify="center")
    table.add_column("reduce", justify="center")
    table.add_column("meaning", max_width=52, overflow="ellipsis")
    for group, infos in catalog.groups():
        shown = [i for i in infos
                 if not enabled_only or cset.setting(i.name).enabled]
        if not shown:
            continue
        for info in shown:
            s = cset.setting(info.name)
            on = s.enabled
            name = info.name if on else f"[dim]{info.name}[/dim]"
            mode = (f"[green]{s.mode.value}[/green]" if on
                    else "[dim]OFF[/dim]")
            table.add_row(name, group, info.unit.value, mode,
                          s.level.value if on else "[dim]·[/dim]",
                          s.reduce.value if on else "[dim]·[/dim]",
                          info.help)
    console.print(table)
    console.print(
        f"[dim]{len(cset.enabled)} of {len(catalog.names)} compiled in · "
        f"periodic capture every {cset.capture_interval} EDT executions · "
        f"the selection is a build-time decision[/dim]"
    )


@counterset_app.command("list")
def counterset_list() -> None:
    for name in store.list_countersets():
        cs = store.load_counterset(name)
        console.print(f"{name:16s} {len(cs.enabled):>2} counters  "
                      f"interval={cs.capture_interval}")


@counterset_app.command("show")
def counterset_show(name: str) -> None:
    console.print(store.counterset_path(name).read_text())


@counterset_app.command("render")
def counterset_render(name: str) -> None:
    """Print the counter file a build would be configured against."""
    from artsrun.render import render_counters

    console.print(render_counters(store.load_counterset(name)),
                  highlight=False, markup=False)


@benchset_app.command("list")
def benchset_list() -> None:
    for name in store.list_benchsets() or []:
        bs = store.load_benchset(name)
        on = sum(1 for e in bs.apps.values() if e.enabled)
        console.print(f"{name:16s} {on} enabled")


@benchset_app.command("new")
def benchset_new(
    name: str,
    copy_from: str = typer.Option(None, "--from"),
    edit: bool = typer.Option(True, "--edit/--no-edit"),
) -> None:
    """Create a benchmark set, optionally copied from an existing one."""
    from artsrun.model.benchset import Benchset

    if name in store.list_benchsets():
        _fail(f"benchset '{name}' already exists")
    if copy_from:
        data = store.load_benchset(copy_from).model_dump(mode="json")
        data["name"] = name
        benchset = Benchset.model_validate(data)
    else:
        # An empty roster defers to the catalog's own defaults, which is the
        # useful starting point for narrowing rather than for building up.
        benchset = Benchset(name=name, description="")
    path = store.save_benchset(benchset)
    console.print(f"wrote {path}")
    if edit:
        _open_editor(path)
        try:
            store.load_benchset(name)
        except Exception as exc:
            _fail(f"invalid after editing: {exc}")
        console.print("[green]valid[/green]")


@benchset_app.command("edit")
def benchset_edit(name: str) -> None:
    """Open a benchmark set in $EDITOR, then re-validate it."""
    path = store.benchset_path(name)
    if not path.is_file():
        _fail(f"no benchset '{name}'")
    _open_editor(path)
    try:
        store.load_benchset(name).resolve(load_catalog())
    except Exception as exc:
        _fail(f"invalid after editing: {exc}")
    console.print("[green]valid[/green]")


@benchset_app.command("set")
def benchset_set(
    name: str,
    app_name: str = typer.Argument(..., metavar="APP"),
    args: str = typer.Option(None, "--args", help="override the calibration"),
    versions: str = typer.Option(None, "--versions",
                                 help="asborn,optimized,restructured"),
    enabled: bool = typer.Option(None, "--enable/--disable"),
) -> None:
    """Change one application's entry in a benchmark set."""
    from artsrun.model.benchset import BenchsetEntry

    catalog = load_catalog()
    if app_name not in catalog.apps:
        _fail(f"unknown application: {app_name}")
    try:
        benchset = store.load_benchset(name)
    except store.NotFound as exc:
        _fail(str(exc))
    entry = benchset.apps.get(app_name) or BenchsetEntry()
    if args is not None:
        entry.args = args.split() or None
    if versions is not None:
        try:
            entry.versions = [Version(v) for v in _split(versions)]
        except ValueError as exc:
            _fail(str(exc))
    if enabled is not None:
        entry.enabled = enabled
    benchset.apps[app_name] = entry
    try:
        benchset.resolve(catalog)
    except ValueError as exc:
        _fail(str(exc))
    path = store.save_benchset(benchset)
    console.print(f"updated {path}")


@benchset_app.command("show")
def benchset_show(name: str) -> None:
    catalog = load_catalog()
    bs = store.load_benchset(name)
    table = Table(title=f"benchset {name}", expand=False)
    for column in ("app", "versions", "args", "source"):
        table.add_column(column)
    for resolved in bs.resolve(catalog):
        table.add_row(
            resolved.name, resolved.version.value,
            " ".join(resolved.args) or "[dim]none[/dim]",
            "override" if resolved.args_overridden else "[dim]catalog[/dim]",
        )
    console.print(table)


# --- config ---------------------------------------------------------------
@config_app.command("render")
def config_render(
    profile: str = typer.Option(..., "--profile", "-p"),
    nodes: int = typer.Option(..., "--nodes", "-n"),
    runtime: str = typer.Option("arts", "--runtime", help="arts | ocr"),
) -> None:
    """Print a rendered runtime configuration."""
    from artsrun.render import render_arts, render_ocr

    prof = store.load_profile(profile)
    text = render_arts(prof, nodes) if runtime == "arts" else render_ocr(prof, nodes)
    console.print(text, highlight=False, markup=False)


def main() -> None:
    os.environ.setdefault("ARTS_REPO", str(repo_root()))
    app()


if __name__ == "__main__":
    main()
