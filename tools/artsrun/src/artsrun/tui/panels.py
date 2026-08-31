"""The three selection surfaces."""

from __future__ import annotations

from pydantic import ValidationError
from textual import on
from textual.app import ComposeResult
from textual.message import Message
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, Select, Static, Switch

from artsrun import store
from artsrun.model.benchset import Benchset, BenchsetEntry
from artsrun.model.catalog import Catalog, Kind, Version
from artsrun.model.counters import (
    Counterset, CounterSetting, Level, Mode, Reduce,
    load_counter_catalog,
)
from artsrun.model.plane import Plane
from artsrun.model.profile import Profile
from artsrun.tui import form
from artsrun.tui.widgets import NodeDelete, Toggle, toggle_all


class NodeListChanged(Message):
    """The node counts changed, so the campaign size did too."""


class LauncherChanged(Message):
    """A different launcher applies, so different sections do."""


class PlanePanel(Vertical):
    """The 4x4 grid.

    Every grid position is drawn, including the five that carry no
    configuration: seeing why EXCL has no WT column, and no family has a
    WB×PURGE cell, is the point of showing the axes as axes.
    """

    def __init__(self, plane: Plane, **kwargs):
        super().__init__(**kwargs)
        self.plane = plane

    def compose(self) -> ComposeResult:
        yield Static("[b]Coherence configuration[/b]  [dim]which protocol "
                     "arms to measure · space toggles · a = all/none[/dim]",
                     classes="panel-head")
        # Two header rows rather than one compound name: the write policy
        # spans its pair of release policies, so the grid reads as two axes
        # instead of four labels.
        top = Horizontal(classes="plane-row")
        top.styles.height = 1
        with top:
            yield Label("", classes="plane-family")
            for write in self.plane.writes:
                yield Label(self.plane.write_label(write),
                            classes="plane-write-head")
        sub = Horizontal(classes="plane-row")
        sub.styles.height = 1
        with sub:
            yield Label("", classes="plane-family")
            for release, _ in self.plane.columns():
                yield Label(self.plane.release_label(release),
                            classes="plane-col-head")
        for family in self.plane.families:
            with Horizontal(classes="plane-row"):
                yield Label(
                    self.plane.family_labels.get(family, family.value),
                    classes="plane-family",
                )
                for release, write in self.plane.columns():
                    cell = self.plane.cell(family, release, write)
                    if not cell.buildable:
                        blank = Static("····", classes="plane-cell blank")
                        blank.tooltip = cell.reason
                        yield blank
                    else:
                        with Horizontal(classes="plane-cell"):
                            for entry in self.plane.entries_of(cell):
                                yield Toggle(
                                    entry.label if entry.is_reference else "ARTS",
                                    entry.key,
                                    classes="entry-toggle",
                                )

    @property
    def toggles(self) -> list[Toggle]:
        return list(self.query(Toggle))

    def selected(self) -> list[str]:
        return [t.ident for t in self.toggles if t.value]

    def toggle_all(self) -> None:
        toggle_all(self.toggles)


class ProfilePanel(VerticalScroll):
    """The machine settings, editable.

    The checked node counts ARE the profile's node list: saving writes them,
    and not saving leaves them applying to this campaign only.
    """

    def __init__(self, profile: Profile, **kwargs):
        super().__init__(**kwargs)
        self.profile = profile
        self.dirty = False
        self.last_status = ""

    # -- layout ------------------------------------------------------------
    def compose(self) -> ComposeResult:
        names = store.list_profiles()
        if self.profile.name not in names:
            # The bootstrap profile of a machine with none saved yet: it is
            # on the screen but not on disk, and the dropdown must say so
            # rather than refuse a value it does not know.
            names = [self.profile.name, *names]
        yield Static("[b]Node profile[/b]  [dim]edit and save, or start a new "
                     "one from this[/dim]", classes="panel-head")
        with Horizontal(id="profile-bar"):
            yield Select([(n, n) for n in names], value=self.profile.name,
                         id="profile-select", allow_blank=False)
            yield Input(value=self.profile.name, placeholder="name",
                        id="profile-name")
            yield Button("Save", id="profile-save", variant="primary")
            yield Button("Save as new", id="profile-saveas")
            yield Button("Revert", id="profile-revert")
        yield Static("", id="profile-status")

        values = form.profile_to_values(self.profile)
        for section, title in form.SECTIONS:
            specs = [f for f in form.PROFILE_FIELDS if f.section == section]
            if not specs and section != "run":
                continue
            with Vertical(id=f"section-{section}", classes="form-block"):
                yield Static(f"[b]{title}[/b]", classes="form-section")
                for spec in specs:
                    with Horizontal(classes="form-row"):
                        yield Label(spec.label, classes="form-label")
                        yield self._widget(spec, values[spec.key])
                        yield Label(spec.help, classes="form-help")
                if section == "run":
                    yield from self._node_editor()

    def _node_editor(self) -> ComposeResult:
        """The node counts, as the list itself.

        Laid out as the other rows are — label, control, explanation — so it
        reads as one more setting rather than as an attachment.
        """
        with Horizontal(classes="form-row node-form-row"):
            yield Label("node counts", classes="form-label")
            with Horizontal(id="node-row"):
                for n in sorted(self.profile.nodes):
                    yield from self._node_item(n)
                yield Input(placeholder="add", id="node-add",
                            classes="node-add")
                yield Button("+", id="node-add-btn", classes="node-add-btn")
        # The list is as wide as it is long, so its explanation cannot sit
        # beside it and still line up with the others; it goes underneath, at
        # the column the other explanations start in.
        with Horizontal(classes="form-row node-help-row"):
            yield Label("", classes="node-help-gutter")
            yield Label("checked ones run; the list is what Save writes",
                        classes="form-help node-help")

    def _node_item(self, n: int) -> ComposeResult:
        with Horizontal(classes="node-item"):
            yield Toggle(str(n), str(n), value=True, classes="node-toggle")
            yield NodeDelete(n)

    def node_list(self) -> list[int]:
        """Every node count on the screen, checked or not."""
        return sorted(int(t.ident)
                      for t in self.query(".node-toggle").results(Toggle))

    def _widget(self, spec: form.FieldSpec, value):
        wid = f"f-{spec.key.replace('.', '-')}"
        if spec.kind == "bool":
            return Switch(value=bool(value), id=wid, classes="form-switch")
        if spec.kind == "choice":
            chosen = str(value) or spec.none_choice or spec.choices[0]
            return Select([(c, c) for c in spec.choices], value=chosen,
                          id=wid, allow_blank=False, classes="form-select")
        # The example is a placeholder rather than a default: an empty optional
        # field means "unset", and prefilling it would silently set it.
        return Input(value=str(value), placeholder=spec.example, id=wid,
                     classes="form-input")

    # -- reading and writing the form --------------------------------------
    def _values(self) -> dict:
        out: dict = {"nodes": self.node_list()}
        for spec in form.PROFILE_FIELDS:
            wid = f"#f-{spec.key.replace('.', '-')}"
            widget = self.query_one(wid)
            if spec.kind == "bool":
                out[spec.key] = widget.value
            elif spec.kind == "choice":
                out[spec.key] = str(widget.value)
            else:
                out[spec.key] = widget.value
        return out

    def _fill(self, values: dict) -> None:
        for spec in form.PROFILE_FIELDS:
            widget = self.query_one(f"#f-{spec.key.replace('.', '-')}")
            widget.value = values[spec.key] if spec.kind == "bool" else str(
                values[spec.key] or ""
            )

    def status(self, message: str, *, error: bool = False) -> None:
        self.last_status = message
        style = "red" if error else "green"
        self.query_one("#profile-status", Static).update(
            f"[{style}]{message}[/{style}]" if message else ""
        )

    # -- launcher-dependent parts -----------------------------------------
    @on(Select.Changed, "#f-launcher")
    def _launcher_changed(self, event: Select.Changed) -> None:
        event.stop()
        self.apply_launcher()
        self.post_message(LauncherChanged())

    def apply_launcher(self) -> None:
        """Show only what the selected launcher decides.

        Ports are the clearest case: a local run's ranks share a machine, so
        they cannot share a port either — the spawning rank finds its own
        block and naming one here is a configuration error.
        """
        launcher = str(self.query_one("#f-launcher", Select).value)
        for section, owner in form.LAUNCHER_SECTIONS.items():
            block = self.query_one(f"#section-{section}", Vertical)
            block.display = launcher == owner
        # The connection count applies to every launcher; only the port
        # numbers are the remote launchers' business.
        ports = self.query_one("#f-ports", Input)
        ports.disabled = launcher == "local"
        ports.placeholder = (
            "claimed automatically" if launcher == "local"
            else form.BY_KEY["ports"].example
        )
        if launcher == "local":
            ports.value = ""

    def edited_profile(self, name: str) -> Profile:
        """Validate the form as a profile, or raise with a readable reason."""
        data = form.values_to_profile_data(name, self._values())
        return Profile.model_validate(data)

    def effective_profile(self) -> Profile:
        """What this campaign runs under, saved or not.

        A campaign uses the screen as it stands, so narrowing the node counts
        for one run does not require saving the profile first.
        """
        try:
            return self.edited_profile(self.profile.name)
        except Exception:
            return self.profile

    def save(self, *, as_new: bool) -> Profile | None:
        name = self.query_one("#profile-name", Input).value.strip()
        if not name:
            self.status("name is required", error=True)
            return None
        if as_new and name in store.list_profiles():
            self.status(f"'{name}' already exists — pick another name",
                        error=True)
            return None
        try:
            profile = self.edited_profile(name)
        except (ValidationError, ValueError) as exc:
            self.status(form.first_problem(exc), error=True)
            return None
        path = store.save_profile(profile)
        self.profile = profile
        self.dirty = False
        self._refresh_choices(name)
        self._refresh_nodes()
        self.status(f"saved {path}")
        return profile

    def revert(self) -> None:
        try:
            self.profile = store.load_profile(self.profile.name)
        except store.NotFound:
            self.status("nothing saved to revert to", error=True)
            return
        self.query_one("#profile-name", Input).value = self.profile.name
        self._fill(form.profile_to_values(self.profile))
        self._refresh_nodes()
        self.apply_launcher()
        self.dirty = False
        self.status("reverted")

    def on_mount(self) -> None:
        self.apply_launcher()
        if self.profile.name not in store.list_profiles():
            self.status("not saved yet — adjust this machine's settings and "
                        "Save to write its first profile")

    def reload(self, name: str) -> None:
        self.profile = store.load_profile(name)
        self.query_one("#profile-name", Input).value = name
        self._fill(form.profile_to_values(self.profile))
        self._refresh_nodes()
        self.apply_launcher()
        self.dirty = False
        self.status("")

    def _refresh_choices(self, selected: str) -> None:
        select = self.query_one("#profile-select", Select)
        select.set_options([(n, n) for n in store.list_profiles()])
        select.value = selected

    def _refresh_nodes(self) -> None:
        """Rebuild the items, leaving the add box where it is."""
        row = self.query_one("#node-row", Horizontal)
        for item in list(row.query(".node-item").results(Horizontal)):
            item.remove()
        box = row.query_one("#node-add", Input)
        for n in sorted(self.profile.nodes):
            item = Horizontal(classes="node-item")
            row.mount(item, before=box)
            item.mount(Toggle(str(n), str(n), value=True,
                              classes="node-toggle"))
            item.mount(NodeDelete(n))

    def node_counts(self) -> list[int]:
        return sorted(int(t.ident)
                      for t in self.query(".node-toggle").results(Toggle)
                      if t.value)

    # -- node list editing -------------------------------------------------
    @on(Button.Pressed, ".node-del")
    def _remove_node(self, event: Button.Pressed) -> None:
        event.stop()
        if len(self.node_list()) <= 1:
            self.status("a profile needs at least one node count", error=True)
            return
        n = event.button.node
        for item in self.query(".node-item").results(Horizontal):
            if int(item.query_one(Toggle).ident) == n:
                item.remove()
                break
        self.status("")
        self.post_message(NodeListChanged())

    @on(Button.Pressed, "#node-add-btn")
    def _add_node_button(self, event: Button.Pressed) -> None:
        event.stop()
        self._add_node()

    @on(Input.Submitted, "#node-add")
    def _add_node_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._add_node()

    def _add_node(self) -> None:
        box = self.query_one("#node-add", Input)
        text = box.value.strip()
        if not text:
            return
        try:
            n = int(text)
        except ValueError:
            self.status(f"'{text}' is not a node count", error=True)
            return
        if n < 1:
            self.status("a node count is at least 1", error=True)
            return
        if n in self.node_list():
            self.status(f"{n} is already listed", error=True)
            return
        row = self.query_one("#node-row", Horizontal)
        item = Horizontal(classes="node-item")
        row.mount(item, before=box)
        item.mount(Toggle(str(n), str(n), value=True, classes="node-toggle"))
        item.mount(NodeDelete(n))
        box.value = ""
        self.status("")
        self.post_message(NodeListChanged())

    def toggle_all(self) -> None:
        toggle_all(list(self.query(".node-toggle").results(Toggle)))




class BenchsetPanel(VerticalScroll):
    """Applications as rows, versions as columns.

    A version an application does not have is drawn as an absence rather than
    an unchecked box, so the three columns stay readable as a claim about the
    application and not about the selection.
    """

    def __init__(self, catalog: Catalog, benchset: Benchset, **kwargs):
        super().__init__(**kwargs)
        self.catalog = catalog
        self.benchset = benchset
        self.last_status = ""

    def compose(self) -> ComposeResult:
        yield Static(
            "[b]Applications[/b]  [dim]a = all/none · arguments are editable; "
            "an empty box means the catalog's own calibration[/dim]",
            classes="panel-head",
        )
        names = store.list_benchsets()
        if self.benchset.name not in names:
            # A machine with no saved benchsets runs on the catalog's own
            # defaults; the dropdown carries that unsaved name rather than
            # refuse a value it does not know.
            names = [self.benchset.name, *names]
        with Horizontal(id="bench-bar"):
            yield Select([(n, n) for n in names],
                         value=self.benchset.name, id="bench-select",
                         allow_blank=False)
            yield Input(value=self.benchset.name, placeholder="name",
                        id="bench-name-input")
            yield Button("Save", id="bench-save", variant="primary")
            yield Button("Save as new", id="bench-saveas")
            yield Button("Revert", id="bench-revert")
        yield Static("", id="bench-status")

        with Horizontal(classes="bench-row head"):
            yield Label("", classes="bench-name")
            for column in ("base", "hinted", "restructured"):
                yield Label(column, classes="bench-col-head")
            yield Label("arguments", classes="bench-args-head")

        # Three groups, applications first: a result is claimed about an
        # application; a microbenchmark is a characterization probe that
        # sweeps run; a toy exercises one mechanism and belongs in a
        # regression suite rather than in any measurement.
        for kind, title in (
            (Kind.APP, "Applications"),
            (Kind.MICROBENCH, "Microbenchmarks (characterization probes)"),
            (Kind.TOY, "Toys and fixtures"),
        ):
            rows = self.catalog.rows_of(kind)
            if not rows:
                continue
            yield Static(f"[b]{title}[/b]  [dim]{len(rows)}[/dim]",
                         classes="bench-group")
            yield from self._app_rows(rows)

    def _app_rows(self, rows) -> ComposeResult:
        for app in rows:
            enabled = self.benchset.is_enabled(app)
            picked = self.benchset.versions_for(app) if enabled else []
            with Horizontal(classes="bench-row"):
                # The name opens the application's structural document —
                # what the parameters mean and how they size the task graph.
                yield Label(
                    f"[@click=app.show_app_doc({app.name!r})]{app.name}[/]",
                    classes="bench-name",
                )
                if app.unsupported:
                    # The row stays visible — the document and the reason are
                    # the point — but its boxes cannot be taken: the program
                    # needs semantics the runtime does not implement.
                    for version in (Version.BASE, Version.HINTED,
                                    Version.RESTRUCTURED):
                        if version not in app.own_versions:
                            yield Static("·", classes="bench-cell blank")
                        else:
                            with Horizontal(classes="bench-cell"):
                                box = Toggle("", f"{app.name}:{version.value}",
                                             value=False,
                                             classes="app-toggle unsupported")
                                box.disabled = True
                                yield box
                    box = Input(value="", placeholder=app.unsupported,
                                id=f"a-{app.name}", classes="bench-args")
                    box.disabled = True
                    yield box
                    continue
                for version in (Version.BASE, Version.HINTED, Version.RESTRUCTURED):
                    if version not in app.own_versions:
                        yield Static("·", classes="bench-cell blank")
                    else:
                        # The box sizes to itself and the cell does the
                        # aligning: a label-less checkbox stretched to the
                        # column width leaves dead space that still takes
                        # clicks.
                        with Horizontal(classes="bench-cell"):
                            yield Toggle(
                                "", f"{app.name}:{version.value}",
                                value=version in picked,
                                classes="app-toggle",
                            )
                entry = self.benchset.apps.get(app.name)
                override = entry.args if entry and entry.args is not None else None
                yield Input(
                    value=" ".join(override) if override else "",
                    placeholder=" ".join(app.args) or "no arguments",
                    id=f"a-{app.name}", classes="bench-args",
                )
            # A rewrite is a separate target with its own command line and its
            # own calibration, so it cannot share the row's box.  It takes a
            # line of its own carrying nothing but that command line: whether
            # it runs is already the row's `restructured` box.
            if app.restructured_as:
                rewrite = self.catalog.apps[app.restructured_as]
                sub = self.benchset.apps.get(rewrite.name)
                sub_args = sub.args if sub and sub.args is not None else None
                with Horizontal(classes="bench-row"):
                    yield Label(f"[dim]  └ {rewrite.name}[/dim]",
                                classes="bench-name")
                    for _ in range(3):
                        yield Static("", classes="bench-cell blank")
                    yield Input(
                        value=" ".join(sub_args) if sub_args else "",
                        placeholder=" ".join(rewrite.args) or "no arguments",
                        id=f"a-{rewrite.name}", classes="bench-args",
                    )

    @property
    def toggles(self) -> list[Toggle]:
        # Unsupported rows draw a box but never a choice: the shared
        # all/none control and the selection reader skip them alike.
        return [t for t in self.query(".app-toggle").results(Toggle)
                if not t.disabled]

    def selected(self) -> dict[str, list[Version]]:
        out: dict[str, list[Version]] = {}
        for t in self.toggles:
            if not t.value:
                continue
            name, _, version = t.ident.partition(":")
            out.setdefault(name, []).append(Version(version))
        return out

    def toggle_all(self) -> None:
        toggle_all(self.toggles)

    # -- editing -----------------------------------------------------------
    def reload(self, name: str) -> None:
        """Load another roster into the rows already on screen.

        The rows come from the catalog, so a benchset changes which boxes are
        ticked and what the argument fields hold, never which rows exist.
        """
        try:
            self.benchset = store.load_benchset(name)
        except store.NotFound:
            self.status("nothing saved to reload", error=True)
            return
        self.query_one("#bench-name-input", Input).value = self.benchset.name
        for app in self.catalog.rows:
            enabled = self.benchset.is_enabled(app)
            picked = self.benchset.versions_for(app) if enabled else []
            for version in app.own_versions:
                ident = f"{app.name}:{version.value}"
                for toggle in self.toggles:
                    if toggle.ident == ident:
                        toggle.value = version in picked
                        break
            entry = self.benchset.apps.get(app.name)
            override = entry.args if entry and entry.args is not None else None
            self.query_one(f"#a-{app.name}", Input).value = (
                " ".join(override) if override else ""
            )
            if app.restructured_as:
                sub = self.benchset.apps.get(app.restructured_as)
                sub_args = sub.args if sub and sub.args is not None else None
                self.query_one(f"#a-{app.restructured_as}", Input).value = (
                    " ".join(sub_args) if sub_args else ""
                )
        self.status("")

    def status(self, message: str, *, error: bool = False) -> None:
        self.last_status = message
        style = "red" if error else "green"
        self.query_one("#bench-status", Static).update(
            f"[{style}]{message}[/{style}]" if message else ""
        )

    def edited_benchset(self, name: str) -> Benchset:
        """The roster the screen currently describes.

        An application with no version checked is written as disabled rather
        than dropped, so a saved set states its own roster instead of leaving
        membership to the catalog's defaults.
        """
        picked = self.selected()
        apps: dict[str, BenchsetEntry] = {}
        for app in self.catalog.rows:
            versions = picked.get(app.name, [])
            box = self.query_one(f"#a-{app.name}", Input).value.strip()
            args = box.split() if box else None
            apps[app.name] = BenchsetEntry(
                enabled=bool(versions),
                versions=versions or None,
                args=args,
            )
            # The rewrite's line is addressed by the rewrite's own name, which
            # is where `resolve` looks for it.  It carries arguments only —
            # membership belongs to the row it appears under — so an empty box
            # writes no entry at all and the catalog's calibration stands.
            if app.restructured_as:
                sub_box = self.query_one(
                    f"#a-{app.restructured_as}", Input).value.strip()
                if sub_box:
                    apps[app.restructured_as] = BenchsetEntry(
                        args=sub_box.split())
        return Benchset(name=name, description=self.benchset.description,
                        apps=apps)

    def save(self, *, as_new: bool) -> Benchset | None:
        name = self.query_one("#bench-name-input", Input).value.strip()
        if not name:
            self.status("name is required", error=True)
            return None
        if as_new and name in store.list_benchsets():
            self.status(f"'{name}' already exists — pick another name",
                        error=True)
            return None
        try:
            benchset = self.edited_benchset(name)
            benchset.resolve(self.catalog)          # surfaces bad versions
        except ValueError as exc:
            self.status(str(exc), error=True)
            return None
        path = store.save_benchset(benchset)
        self.benchset = benchset
        select = self.query_one("#bench-select", Select)
        select.set_options([(n, n) for n in store.list_benchsets()])
        select.value = name
        self.status(f"saved {path}")
        return benchset


class RunPanel(Vertical):
    """Start a campaign, or carry an earlier one on.

    A continuation is a separate control rather than a mode of the first,
    because the two differ in what they measure against: one starts from the
    screens, the other from what a past run already recorded.
    """

    def compose(self) -> ComposeResult:
        from artsrun.campaign import past_runs

        # One row for every control: each further fixed row here is a row the
        # live table underneath does not get.
        unfinished = [r for r in past_runs() if r.remaining]
        with Horizontal(id="run-controls"):
            yield Button("Build and run", id="run-button", variant="primary")
            yield Button("Dry run", id="dry-button")
            # Shown only while something is running: a stop with nothing to
            # stop invites a press that reports an error for no reason.
            stop = Button("Stop", id="stop-button", variant="error")
            stop.display = False
            yield stop
            yield Select(
                [(r.label, r.run_id) for r in unfinished],
                prompt="continue a past run…", id="resume-select",
                allow_blank=True,
            )
            yield Button("Continue", id="resume-button", disabled=True)
            yield Static("", id="run-size")

    def refresh_runs(self) -> None:
        """Re-read what is resumable, after a campaign changes the answer."""
        from artsrun.campaign import past_runs

        select = self.query_one("#resume-select", Select)
        select.set_options(
            (r.label, r.run_id) for r in past_runs() if r.remaining
        )
        self.query_one("#resume-button", Button).disabled = True


class CounterPanel(VerticalScroll):
    """Which counters the build compiles in, and how each is reduced.

    Grouped as the runtime's own declaration list groups them.  Level is the
    interesting column: CLUSTER still writes the per-rank value, so a
    distribution and a total come from the same run.
    """

    def __init__(self, counterset: Counterset, **kwargs):
        super().__init__(**kwargs)
        self.counterset = counterset
        self.catalog = load_counter_catalog()
        self.last_status = ""

    def compose(self) -> ComposeResult:
        yield Static(
            "[b]Counters[/b]  [dim]compiled into the build — changing the set "
            "needs a reconfigure and a full rebuild · a = all/none[/dim]",
            classes="panel-head",
        )
        names = store.list_countersets()
        if self.counterset.name not in names:
            # The set in use may not be on disk — a fresh checkout runs on
            # the built-in "none" — and the dropdown carries that name
            # rather than refuse a value it does not know.
            names = [self.counterset.name, *names]
        with Horizontal(id="counter-bar"):
            yield Select([(n, n) for n in names],
                         value=self.counterset.name,
                         id="counter-select", allow_blank=True)
            yield Input(value=self.counterset.name, placeholder="name",
                        id="counter-name")
            yield Button("Save", id="counter-save", variant="primary")
            yield Button("Save as new", id="counter-saveas")
        yield Static("", id="counter-status")

        with Horizontal(classes="form-row"):
            yield Label("sample every", classes="form-label")
            yield Input(value=str(self.counterset.capture_interval),
                        placeholder="100", id="counter-interval",
                        classes="form-input")
            yield Label("EDT executions between periodic captures; ONCE "
                        "counters ignore it", classes="form-help")

        with Horizontal(classes="bench-row head"):
            yield Label("counter", classes="counter-name")
            yield Label("mode", classes="counter-col")
            yield Label("level", classes="counter-col")
            yield Label("reduce", classes="counter-col")
            yield Label("", classes="counter-help-head")

        for group, infos in self.catalog.groups():
            yield Static(f"[b]{group}[/b]", classes="bench-group")
            for info in infos:
                setting = self.counterset.setting(info.name)
                with Horizontal(classes="bench-row"):
                    yield Label(info.name, classes="counter-name")
                    yield Select([(m.value, m.value) for m in Mode],
                                 value=setting.mode.value, allow_blank=False,
                                 id=f"cm-{info.name}", classes="counter-select")
                    yield Select([(v.value, v.value) for v in Level],
                                 value=setting.level.value, allow_blank=False,
                                 id=f"cl-{info.name}", classes="counter-select")
                    yield Select([(r.value, r.value) for r in Reduce],
                                 value=setting.reduce.value, allow_blank=False,
                                 id=f"cr-{info.name}", classes="counter-select")
                    yield Label(f"{info.help} [{info.unit.value}]",
                                classes="counter-help")

    # -- reading -----------------------------------------------------------
    def edited_counterset(self, name: str) -> Counterset:
        interval = self.query_one("#counter-interval", Input).value.strip()
        counters = {}
        for info in self.catalog.counters.values():
            mode = Mode(str(self.query_one(f"#cm-{info.name}", Select).value))
            if mode is Mode.OFF:
                continue
            counters[info.name] = CounterSetting(
                mode=mode,
                level=Level(str(self.query_one(f"#cl-{info.name}", Select).value)),
                reduce=Reduce(str(self.query_one(f"#cr-{info.name}", Select).value)),
            )
        return Counterset(
            name=name,
            description=self.counterset.description,
            capture_interval=int(interval) if interval.isdigit() else 100,
            folder=self.counterset.folder,
            counters=counters,
        )

    def effective_counterset(self) -> Counterset:
        try:
            return self.edited_counterset(self.counterset.name)
        except Exception:
            return self.counterset

    def reload(self, name: str) -> None:
        """Load another set into the rows already on screen.

        The rows come from the runtime's counter list, so they never change;
        only the values do.  Rebuilding the panel instead would try to mount a
        second widget under the same id before the first has been removed.
        """
        self.counterset = store.load_counterset(name)
        self.query_one("#counter-name", Input).value = self.counterset.name
        self.query_one("#counter-interval", Input).value = str(
            self.counterset.capture_interval
        )
        for info in self.catalog.counters.values():
            setting = self.counterset.setting(info.name)
            self.query_one(f"#cm-{info.name}", Select).value = setting.mode.value
            self.query_one(f"#cl-{info.name}", Select).value = setting.level.value
            self.query_one(f"#cr-{info.name}", Select).value = setting.reduce.value
        self.status("")

    def status(self, message: str, *, error: bool = False) -> None:
        self.last_status = message
        style = "red" if error else "green"
        self.query_one("#counter-status", Static).update(
            f"[{style}]{message}[/{style}]" if message else ""
        )

    def save(self, *, as_new: bool) -> Counterset | None:
        name = self.query_one("#counter-name", Input).value.strip()
        if not name:
            self.status("name is required", error=True)
            return None
        if as_new and name in store.list_countersets():
            self.status(f"'{name}' already exists — pick another name",
                        error=True)
            return None
        try:
            counterset = self.edited_counterset(name)
        except (ValidationError, ValueError) as exc:
            self.status(form.first_problem(exc), error=True)
            return None
        path = store.save_counterset(counterset)
        self.counterset = counterset
        select = self.query_one("#counter-select", Select)
        select.set_options([(n, n) for n in store.list_countersets()])
        select.value = name
        self.status(f"saved {path} — {len(counterset.enabled)} counters on")
        return counterset

    @property
    def mode_selects(self) -> list:
        return [self.query_one(f"#cm-{n}", Select) for n in self.catalog.names]

    def toggle_all(self) -> None:
        """All on means PERIODIC at the level each already carries."""
        anything = any(str(s.value) != Mode.OFF.value for s in self.mode_selects)
        target = Mode.OFF.value if anything else Mode.PERIODIC.value
        for select in self.mode_selects:
            select.value = target
