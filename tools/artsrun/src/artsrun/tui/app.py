"""The selection screens.

Three surfaces, one campaign: the configuration plane, the node profile and
the application roster.  Everything is selected by default, because the common
case is a full sweep and the interesting cases are subtractions from it.
"""

from __future__ import annotations

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import (
    Button, Footer, Header, RichLog, Select, TabbedContent, TabPane,
)

from artsrun import store
from artsrun.model.catalog import load_catalog
from artsrun.model.plane import load_plane
from artsrun.model.selection import Selection
from artsrun.tui.panels import (
    BenchsetPanel, CounterPanel, LauncherChanged, NodeListChanged,
    PlanePanel, ProfilePanel, RunPanel,
)


class ArtsRunApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "artsrun"
    SUB_TITLE = "experiment driver"

    BINDINGS = [
        ("1", "show('tab-plane')", "coherence"),
        ("2", "show('tab-profile')", "profile"),
        ("3", "show('tab-bench')", "apps"),
        ("4", "show('tab-counters')", "counters"),
        ("5", "show('tab-run')", "run tab"),
        ("a", "toggle_all", "all / none"),
        ("r", "run", "run"),
        ("d", "dry_run", "dry run"),
        ("s", "stop", "stop"),
        ("c", "continue", "continue"),
        ("q", "quit", "quit"),
    ]

    def __init__(self, profile: str | None = None, benchset: str | None = None):
        super().__init__()
        self.plane = load_plane()
        self.catalog = load_catalog()
        profiles = store.list_profiles()
        if not profiles:
            raise SystemExit(
                f"no profiles in {store.profiles_dir()}; write one first"
            )
        self.profile = store.load_profile(profile or profiles[0])
        benchsets = store.list_benchsets()
        self.benchset = (
            store.load_benchset(benchset or benchsets[0])
            if benchsets else store.default_benchset()
        )
        sets = store.list_countersets()
        self.counterset = (store.load_counterset(sets[0]) if sets
                           else store.default_counterset())
        self._campaign_running = False
        self._campaign = None

    # -- layout ------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="tab-plane"):
            with TabPane("Coherence", id="tab-plane"):
                yield PlanePanel(self.plane, id="plane")
            with TabPane("Profile", id="tab-profile"):
                yield ProfilePanel(self.profile, id="profile")
            with TabPane("Applications", id="tab-bench"):
                yield BenchsetPanel(self.catalog, self.benchset, id="bench")
            with TabPane("Counters", id="tab-counters"):
                yield CounterPanel(self.counterset, id="counters")
            with TabPane("Run", id="tab-run"):
                with Vertical():
                    yield RunPanel(id="run")
                    yield RichLog(id="run-log", wrap=False, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_size()

    # -- selection ---------------------------------------------------------
    def _missing(self) -> list[str]:
        """Which surfaces are empty, in the words the footer uses."""
        plane_panel = self.query_one("#plane", PlanePanel)
        profile_panel = self.query_one("#profile", ProfilePanel)
        bench_panel = self.query_one("#bench", BenchsetPanel)
        return [
            name for name, ok in
            (("a configuration", plane_panel.selected()),
             ("an application", bench_panel.selected()),
             ("a node count", profile_panel.node_counts()))
            if not ok
        ]

    def build_selection(self, *, announce: bool = False) -> Selection | None:
        """The campaign the screens currently describe, or None if incomplete.

        `announce` is for the moment someone asks to run: clearing a surface is
        a normal editing step, and every checkbox on it reports its own change,
        so warning during editing would raise one notification per box.
        """
        plane_panel = self.query_one("#plane", PlanePanel)
        profile_panel = self.query_one("#profile", ProfilePanel)
        bench_panel = self.query_one("#bench", BenchsetPanel)

        entries = plane_panel.selected()
        apps = bench_panel.selected()
        nodes = profile_panel.node_counts()
        if not entries or not apps or not nodes:
            if announce:
                self.notify(f"select {', '.join(self._missing())}",
                            severity="warning")
            return None
        return Selection(
            profile=profile_panel.profile.name,
            benchset=self.benchset.name,
            entries=entries,
            apps=apps,
            node_counts=sorted(nodes),
            repeats=profile_panel.profile.repeats,
        )

    def _refresh_size(self) -> None:
        try:
            selection = self.build_selection()
        except Exception:
            return
        label = self.query_one("#run-size")
        if selection is None:
            label.update(f"[yellow]select {', '.join(self._missing())}[/yellow]")
            return
        label.update(
            f"{selection.cell_count} cells = {len(selection.entries)} configurations "
            f"x {sum(len(v) for v in selection.apps.values())} app-versions "
            f"x {len(selection.node_counts)} node counts "
            f"x {selection.repeats} repeats"
        )

    # -- events ------------------------------------------------------------
    # A Select announces its initial value while the screen is still being
    # built, so both handlers ignore a change to what is already loaded.
    @on(Select.Changed, "#profile-select")
    def _profile_changed(self, event: Select.Changed) -> None:
        if event.value and str(event.value) != self.profile.name:
            self.query_one("#profile", ProfilePanel).reload(str(event.value))
            self.profile = self.query_one("#profile", ProfilePanel).profile
            self._refresh_size()

    @on(Button.Pressed, "#run-button")
    def _run_pressed(self) -> None:
        self.action_run()

    @on(Button.Pressed, "#dry-button")
    def _dry_pressed(self) -> None:
        self.action_dry_run()

    @on(Button.Pressed, "#stop-button")
    def _stop_pressed(self) -> None:
        self.action_stop()

    @on(Select.Changed, "#resume-select")
    def _resume_picked(self, event: Select.Changed) -> None:
        self.query_one("#resume-button", Button).disabled = not event.value

    @on(Button.Pressed, "#resume-button")
    def _resume_pressed(self) -> None:
        self.action_continue()

    # -- profile editing ---------------------------------------------------
    @on(Button.Pressed, "#profile-save")
    def _profile_save(self) -> None:
        self._save_profile(as_new=False)

    @on(Button.Pressed, "#profile-saveas")
    def _profile_save_as(self) -> None:
        self._save_profile(as_new=True)

    @on(Button.Pressed, "#profile-revert")
    def _profile_revert(self) -> None:
        self.query_one("#profile", ProfilePanel).revert()
        self._refresh_size()

    def _save_profile(self, *, as_new: bool) -> None:
        panel = self.query_one("#profile", ProfilePanel)
        if panel.save(as_new=as_new):
            self.profile = panel.profile
            self._refresh_size()

    # -- benchset editing --------------------------------------------------
    @on(Select.Changed, "#bench-select")
    def _benchset_changed(self, event: Select.Changed) -> None:
        if event.value and str(event.value) != self.benchset.name:
            self._reload_benchset(str(event.value))

    # -- counter set editing -----------------------------------------------
    @on(Select.Changed, "#counter-select")
    def _counterset_changed(self, event: Select.Changed) -> None:
        if event.value and str(event.value) != self.counterset.name:
            self._reload_counterset(str(event.value))

    @on(Button.Pressed, "#counter-save")
    def _counter_save(self) -> None:
        self._save_counterset(as_new=False)

    @on(Button.Pressed, "#counter-saveas")
    def _counter_save_as(self) -> None:
        self._save_counterset(as_new=True)

    def _save_counterset(self, *, as_new: bool) -> None:
        panel = self.query_one("#counters", CounterPanel)
        saved = panel.save(as_new=as_new)
        if saved:
            self.counterset = saved

    def _reload_counterset(self, name: str) -> None:
        panel = self.query_one("#counters", CounterPanel)
        panel.reload(name)
        self.counterset = panel.counterset

    @on(Button.Pressed, "#bench-save")
    def _bench_save(self) -> None:
        self._save_benchset(as_new=False)

    @on(Button.Pressed, "#bench-saveas")
    def _bench_save_as(self) -> None:
        self._save_benchset(as_new=True)

    @on(Button.Pressed, "#bench-revert")
    def _bench_revert(self) -> None:
        self._reload_benchset(self.benchset.name)

    def _save_benchset(self, *, as_new: bool) -> None:
        panel = self.query_one("#bench", BenchsetPanel)
        saved = panel.save(as_new=as_new)
        if saved:
            self.benchset = saved
            self._refresh_size()

    def _reload_benchset(self, name: str) -> None:
        panel = self.query_one("#bench", BenchsetPanel)
        panel.reload(name)
        self.benchset = panel.benchset
        self._refresh_size()

    def on_checkbox_changed(self) -> None:
        self._refresh_size()

    @on(NodeListChanged)
    def _nodes_changed(self) -> None:
        self._refresh_size()

    @on(LauncherChanged)
    def _launcher_changed(self) -> None:
        self._refresh_size()

    # -- actions -----------------------------------------------------------
    def action_show(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab

    def action_toggle_all(self) -> None:
        """One control for both directions, on whichever surface is showing."""
        active = self.query_one(TabbedContent).active
        panel = {
            "tab-plane": ("#plane", PlanePanel),
            "tab-profile": ("#profile", ProfilePanel),
            "tab-bench": ("#bench", BenchsetPanel),
            "tab-counters": ("#counters", CounterPanel),
        }.get(active)
        if panel is None:
            return
        self.query_one(panel[0], panel[1]).toggle_all()
        self._refresh_size()

    def action_dry_run(self) -> None:
        selection = self.build_selection(announce=True)
        if selection is None:
            return
        self.query_one(TabbedContent).active = "tab-run"
        self._dry_run(selection)

    def action_continue(self) -> None:
        """Carry a past run on, measuring against the selection it recorded.

        The screens are left alone: a continuation belongs to the campaign it
        continues, and taking the current selection instead would append cells
        of one experiment to the results of another.
        """
        if self._campaign_running:
            self.notify("a campaign is already running", severity="warning")
            return
        picked = self.query_one("#resume-select", Select).value
        if not picked:
            self.notify("choose a run to continue", severity="warning")
            return
        from artsrun.campaign import load_past

        try:
            selection, run_dir = load_past(str(picked))
        except (OSError, ValueError) as exc:
            self.notify(str(exc), severity="error")
            return
        self.query_one(TabbedContent).active = "tab-run"
        self._log(f"continuing {run_dir.name}")
        self._set_running(True)
        self._run_campaign(selection, run_dir=run_dir)

    def action_stop(self) -> None:
        campaign = getattr(self, "_campaign", None)
        if not self._campaign_running or campaign is None:
            self.notify("nothing is running", severity="warning")
            return
        self._log("stopping — the cell in progress is being ended")
        campaign.request_stop()

    def _set_running(self, running: bool) -> None:
        """Show the stop control exactly while it has something to stop."""
        self._campaign_running = running
        try:
            self.query_one("#stop-button").display = running
            if not running:
                # A run that just finished is no longer resumable, and one
                # that was stopped now is.
                self.query_one("#run", RunPanel).refresh_runs()
        except Exception:
            pass

    def action_run(self) -> None:
        if self._campaign_running:
            self.notify("a campaign is already running", severity="warning")
            return
        selection = self.build_selection(announce=True)
        if selection is None:
            return
        self.query_one(TabbedContent).active = "tab-run"
        self._set_running(True)
        self._run_campaign(selection)

    # -- workers -----------------------------------------------------------
    def _log(self, line: str) -> None:
        """Write one line to the run log, from a worker or from the screen.

        The campaign writes from its worker thread and the controls write from
        the screen's own, and the hand-off the first needs is an error for the
        second.
        """
        write = self.query_one("#run-log", RichLog).write
        try:
            self.call_from_thread(write, line)
        except RuntimeError:
            write(line)

    @work(thread=True)
    def _dry_run(self, selection: Selection) -> None:
        from artsrun.campaign import Campaign
        from artsrun.paths import wall_cache_path
        from artsrun.run.scheduler import WallCache, order

        log = self.query_one("#run-log", RichLog)
        profile = self.query_one("#profile", ProfilePanel).effective_profile()
        counters = self.query_one("#counters", CounterPanel).effective_counterset()
        campaign = Campaign.prepare(
            selection, self.plane, self.catalog, self.benchset, profile,
            counterset=counters,
        )
        try:
            plan = campaign.build_plan()
            self._log(f"build: {len(plan.targets)} targets in {campaign.build_dir}")
            if plan.missing:
                self._log(f"MISSING TARGETS: {', '.join(plan.missing[:8])}")
        except Exception as exc:
            self._log(f"build check: {exc}")
        cells, skipped = campaign.cells()
        ordered = order(cells, WallCache(wall_cache_path()),
                        capacity=campaign.backend().capacity)
        self._log(f"{len(cells)} cells, {len(skipped)} structurally ineligible")
        for cell in ordered[:20]:
            self._log(f"  {cell.nodes:>3}n  {cell.app.key:<28} {cell.entry.key}")
        if len(ordered) > 20:
            self._log(f"  … {len(ordered) - 20} more")
        self.call_from_thread(log.write, "")

    @work(thread=True)
    def _run_campaign(self, selection: Selection, run_dir=None) -> None:
        from artsrun.campaign import Campaign

        # Everything the worker does sits inside the guard: a campaign that
        # cannot even be prepared is the case a caller most needs told about,
        # and an exception escaping a worker thread is only written to the
        # framework's own log -- the screen would show nothing at all, and the
        # in-progress flag would never be lowered.
        try:
            profile = self.query_one("#profile", ProfilePanel).effective_profile()
            counters = self.query_one("#counters", CounterPanel).effective_counterset()
            campaign = Campaign.prepare(
                selection, self.plane, self.catalog, self.benchset, profile,
                counterset=counters, run_dir=run_dir,
            )
            self._campaign = campaign
            result = campaign.run(on_line=self._log, resume=run_dir is not None)
            self._log("")
            for line in result["summary"].splitlines():
                self._log(line)
            self._log(f"run directory: {result['run_dir']}")
        except Exception as exc:
            # Several of these carry the command that resolves them, over more
            # than one line; collapsing them to one would cut it off.
            self._log("campaign failed:")
            for line in str(exc).splitlines():
                self._log(f"  {line}")
        finally:
            self._campaign = None
            self.call_from_thread(self._set_running, False)


def run_tui(profile: str | None = None, benchset: str | None = None) -> int:
    ArtsRunApp(profile=profile, benchset=benchset).run()
    return 0
