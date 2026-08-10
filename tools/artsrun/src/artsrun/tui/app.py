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
        self._running = False

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

    def action_run(self) -> None:
        if self._running:
            self.notify("a campaign is already running", severity="warning")
            return
        selection = self.build_selection(announce=True)
        if selection is None:
            self._running = False
            return
        self.query_one(TabbedContent).active = "tab-run"
        self._running = True
        self._run_campaign(selection)

    # -- workers -----------------------------------------------------------
    def _log(self, line: str) -> None:
        self.call_from_thread(self.query_one("#run-log", RichLog).write, line)

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
        ordered = order(cells, WallCache(wall_cache_path()))
        self._log(f"{len(cells)} cells, {len(skipped)} structurally ineligible")
        for cell in ordered[:20]:
            self._log(f"  {cell.nodes:>3}n  {cell.app.key:<28} {cell.entry.key}")
        if len(ordered) > 20:
            self._log(f"  … {len(ordered) - 20} more")
        self.call_from_thread(log.write, "")

    @work(thread=True)
    def _run_campaign(self, selection: Selection) -> None:
        from artsrun.campaign import Campaign

        profile = self.query_one("#profile", ProfilePanel).effective_profile()
        counters = self.query_one("#counters", CounterPanel).effective_counterset()
        campaign = Campaign.prepare(
            selection, self.plane, self.catalog, self.benchset, profile,
            counterset=counters,
        )
        try:
            result = campaign.run(on_line=self._log)
            self._log("")
            for line in result["summary"].splitlines():
                self._log(line)
            self._log(f"run directory: {result['run_dir']}")
        except Exception as exc:
            self._log(f"campaign failed: {exc}")
        finally:
            self._running = False


def run_tui(profile: str | None = None, benchset: str | None = None) -> int:
    ArtsRunApp(profile=profile, benchset=benchset).run()
    return 0
