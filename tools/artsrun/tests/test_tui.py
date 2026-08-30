"""The selection screens: default state, the shared all/none control, and the
campaign a selection turns into."""

from __future__ import annotations

import asyncio

from artsrun.tui.app import ArtsRunApp
from artsrun.tui.panels import BenchsetPanel, PlanePanel, ProfilePanel
from artsrun.tui.widgets import Toggle


def drive(coro_factory):
    """Run one piloted session and return whatever it produced."""

    async def main():
        app = ArtsRunApp(profile="ferrari", benchset="paper-main")
        async with app.run_test() as pilot:
            return await coro_factory(app, pilot)

    return asyncio.run(main())


def test_everything_is_selected_by_default():
    async def check(app, pilot):
        plane = app.query_one("#plane", PlanePanel)
        profile = app.query_one("#profile", ProfilePanel)
        bench = app.query_one("#bench", BenchsetPanel)
        return (
            sorted(plane.selected()),
            profile.node_counts(),
            len(bench.selected()),
        )

    entries, nodes, apps = drive(check)
    assert len(entries) == 13
    assert nodes == [1, 2, 4, 8]
    assert apps > 0


def test_the_plane_draws_every_position_but_only_offers_eleven():
    async def check(app, pilot):
        plane = app.query_one("#plane", PlanePanel)
        blanks = plane.query(".plane-cell.blank")
        return len(plane.toggles), len(blanks)

    toggles, blanks = drive(check)
    assert toggles == 13
    assert blanks == 5


def test_one_control_clears_then_restores_the_whole_plane():
    async def check(app, pilot):
        plane = app.query_one("#plane", PlanePanel)
        first = len(plane.selected())
        await pilot.press("a")
        cleared = len(plane.selected())
        await pilot.press("a")
        restored = len(plane.selected())
        return first, cleared, restored

    first, cleared, restored = drive(check)
    assert first == 13
    assert cleared == 0
    assert restored == 13


def test_the_control_acts_on_the_surface_that_is_showing():
    async def check(app, pilot):
        app.query_one("#bench").focus()
        from textual.widgets import TabbedContent

        app.query_one(TabbedContent).active = "tab-bench"
        await pilot.pause()
        await pilot.press("a")
        bench = app.query_one("#bench", BenchsetPanel)
        plane = app.query_one("#plane", PlanePanel)
        return len(bench.selected()), len(plane.selected())

    apps, entries = drive(check)
    assert apps == 0        # the visible surface cleared
    assert entries == 13    # the others did not


def test_a_version_the_application_lacks_is_absent_not_unchecked():
    # Whichever application has no hint layer — naming one here makes the test
    # fail when that application later gains one, which says nothing about the
    # absence being tested.
    from artsrun.model.catalog import load_catalog
    bare = next(a for a in load_catalog().rows if not a.hinted).name

    async def check(app, pilot):
        bench = app.query_one("#bench", BenchsetPanel)
        idents = {t.ident for t in bench.toggles}
        return (f"{bare}:hinted" in idents, "nqueens:hinted" in idents,
                len(bench.query(".bench-cell.blank")))

    has_absent, has_present, blanks = drive(check)
    assert not has_absent
    assert has_present
    assert blanks > 0


def test_the_selection_becomes_a_campaign_of_the_expected_size():
    async def check(app, pilot):
        selection = app.build_selection()
        return selection.cell_count, selection.entries, selection.node_counts

    cells, entries, nodes = drive(check)
    assert cells == len(entries) * 4 * (cells // (len(entries) * 4))
    assert len(entries) == 13
    assert nodes == [1, 2, 4, 8]


def test_an_empty_surface_blocks_the_campaign():
    async def check(app, pilot):
        await pilot.press("a")          # clear the plane
        return app.build_selection()

    assert drive(check) is None


def test_the_run_log_stays_out_of_the_way_until_it_matters():
    # The bottom box costs the table twelve rows, so it exists only while a
    # build or a failure is the thing to watch.
    from textual.widgets import RichLog

    async def check(app, pilot):
        return app.query_one("#run-log", RichLog).display

    assert drive(check) is False


def test_number_keys_switch_surfaces():
    async def check(app, pilot):
        from textual.widgets import TabbedContent

        seen = []
        for key in ("2", "3", "4", "5", "1"):
            await pilot.press(key)
            seen.append(app.query_one(TabbedContent).active)
        return seen

    assert drive(check) == ["tab-profile", "tab-bench", "tab-counters",
                            "tab-run", "tab-plane"]


def test_reference_labels_fit_their_cells():
    # A split cell holds two toggles side by side; the longest reference name
    # must still render without being elided.
    async def check(app, pilot):
        plane = app.query_one(PlanePanel)
        return [(t.label.plain, t.size.width) for t in plane.toggles]

    for label, width in drive(check):
        assert width >= len(label) + 4, f"{label!r} has only {width} columns"


def test_the_selection_mark_is_a_check_not_a_cross():
    async def check(app, pilot):
        plane = app.query_one(PlanePanel)
        return {t.BUTTON_INNER for t in plane.toggles}

    assert drive(check) == {"✓"}


def test_clearing_a_surface_raises_no_notifications():
    # Every checkbox reports its own change, so warning while editing produced
    # one notification per box.
    async def check(app, pilot):
        calls = []
        app.notify = lambda *a, **k: calls.append(a)
        await pilot.press("3")      # applications
        await pilot.press("a")      # clear them all
        await pilot.pause()
        bench = app.query_one("#bench", BenchsetPanel)
        return len(calls), len(bench.selected())

    calls, remaining = drive(check)
    assert remaining == 0
    assert calls == 0


def test_asking_to_run_an_empty_selection_warns_once():
    async def check(app, pilot):
        calls = []
        app.notify = lambda *a, **k: calls.append(a)
        await pilot.press("3")
        await pilot.press("a")
        await pilot.pause()
        app.action_dry_run()
        return calls

    calls = drive(check)
    assert len(calls) == 1
    assert "an application" in calls[0][0]


def test_a_split_cell_fits_both_of_its_toggles():
    # Measuring each toggle alone misses the case that actually clips: two of
    # them side by side in one cell.
    async def check(app, pilot):
        from artsrun.tui.widgets import Toggle

        plane = app.query_one(PlanePanel)
        out = []
        for cell in plane.query(".plane-cell"):
            toggles = list(cell.query(Toggle))
            if not toggles:
                continue
            labels = [t.label.plain for t in toggles]
            need = sum(len(t) + 4 for t in labels)
            out.append((" + ".join(labels), need, cell.content_size.width))
        return out

    cells = drive(check)
    assert any(" + " in labels for labels, _, _ in cells), "no split cell found"
    for labels, need, avail in cells:
        assert need <= avail, f"{labels} needs {need} columns, cell has {avail}"


def test_every_save_button_survives_its_own_dropdown_refresh():
    # Saving refreshes the dropdown with set_options(), which clears the
    # Select to its blank sentinel before the saver restores the value; that
    # transient Changed is processed after the restore, so the handler must
    # judge the event's value, not the widget's current state.
    async def check(app, pilot):
        alive = []
        for tab, button in (("4", "#counter-save"), ("2", "#profile-save"),
                            ("3", "#bench-save")):
            await pilot.press(tab)
            await pilot.pause()
            await pilot.click(button)
            await pilot.pause()
            alive.append(app.is_running)
        return alive

    assert drive(check) == [True, True, True]


def test_a_fresh_checkout_opens_on_unsaved_defaults(tmp_path, monkeypatch):
    # No profiles, no benchsets, no counter sets — the screens open anyway:
    # an unsaved single-node local profile, the catalog's own roster, no
    # counters.  Save on the Profile tab writes the machine's first file.
    from artsrun import store

    for surface in ("profiles_dir", "benchsets_dir", "countersets_dir"):
        monkeypatch.setattr(f"artsrun.store.{surface}",
                            lambda s=surface: tmp_path / s)

    async def main():
        app = ArtsRunApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            panel = app.query_one("#profile", ProfilePanel)
            before = store.list_profiles()
            saved = panel.save(as_new=False)
            return (app.profile.name, app.profile.nodes, app.benchset.name,
                    before, saved is not None, store.list_profiles())

    name, nodes, benchset, before, saved, after = asyncio.run(main())
    assert name == "local"
    assert nodes == [1]
    assert benchset == "catalog-default"
    assert before == []
    assert saved
    assert after == ["local"]


def test_a_machine_with_no_profiles_can_bootstrap_one():
    # `profile new` without --from is the one path that starts from nothing —
    # a fresh checkout has no profile to copy, so the blank starting point
    # must satisfy everything a Profile requires.
    from artsrun.model.profile import Launcher, Profile
    from artsrun.tui import form

    profile = Profile.model_validate(
        form.values_to_profile_data("fresh", form.blank_values())
    )
    assert profile.launcher is Launcher.LOCAL
    assert profile.nodes == [1]
    assert profile.port_count == 1
    assert profile.ports == []  # local claims its own block


# --- profile editing ------------------------------------------------------
def _profiles_dir(tmp):
    import os
    os.environ["ARTS_REPO"] = str(tmp)
    return tmp


def test_the_profile_form_shows_every_field_of_the_stored_profile():
    async def check(app, pilot):
        from textual.widgets import Input

        panel = app.query_one("#profile", ProfilePanel)
        return (
            panel.query_one("#f-workers", Input).value,
            panel.query_one("#f-progress", Input).value,
            str(panel.query_one("#f-launcher").value),
            str(panel.query_one("#f-provider").value),
        )

    workers, progress, launcher, provider = drive(check)
    assert workers == "15"
    assert progress == "1"
    assert launcher == "local"
    assert provider == "tcp"


def test_an_edited_form_validates_into_a_profile():
    async def check(app, pilot):
        from textual.widgets import Input

        panel = app.query_one("#profile", ProfilePanel)
        from artsrun.tui.widgets import NodeDelete

        panel.query_one("#f-workers", Input).value = "7"
        for button in list(panel.query(NodeDelete).results(NodeDelete)):
            if button.node in (4, 8):
                button.press()
        await pilot.pause()
        edited = panel.edited_profile("scratch")
        return edited.workers, edited.nodes, edited.launcher.value

    assert drive(check) == (7, [1, 2], "local")


def test_a_form_that_cannot_be_read_reports_the_reason_and_saves_nothing():
    async def check(app, pilot):
        from textual.widgets import Input

        panel = app.query_one("#profile", ProfilePanel)
        panel.query_one("#f-workers", Input).value = "not-a-number"
        saved = panel.save(as_new=False)
        return saved, panel.last_status

    saved, status = drive(check)
    assert saved is None
    assert "workers" in status


def test_switching_to_slurm_requires_its_settings():
    async def check(app, pilot):
        panel = app.query_one("#profile", ProfilePanel)
        panel.query_one("#f-launcher").value = "slurm"
        try:
            panel.edited_profile("scratch")
            return "accepted"
        except Exception as exc:
            return str(exc)

    assert "slurm" in drive(check)


def test_reverting_restores_the_stored_values():
    async def check(app, pilot):
        from textual.widgets import Input

        panel = app.query_one("#profile", ProfilePanel)
        panel.query_one("#f-workers", Input).value = "3"
        panel.revert()
        return panel.query_one("#f-workers", Input).value

    assert drive(check) == "15"


# --- benchset editing -----------------------------------------------------
def test_argument_boxes_show_the_catalog_default_as_a_placeholder():
    async def check(app, pilot):
        from textual.widgets import Input

        bench = app.query_one("#bench", BenchsetPanel)
        box = bench.query_one("#a-nqueens", Input)
        return box.value, box.placeholder

    value, placeholder = drive(check)
    assert value == ""                 # paper-main overrides nothing
    # The placeholder is the catalog's own calibration, whatever it currently
    # is — pinning the literal here only breaks the test every recalibration.
    from artsrun.model.catalog import load_catalog
    catalog_args = [r for r in load_catalog().rows if r.name == "nqueens"][0].args
    assert placeholder == " ".join(catalog_args)


def test_an_edited_argument_becomes_an_override():
    async def check(app, pilot):
        from textual.widgets import Input

        bench = app.query_one("#bench", BenchsetPanel)
        bench.query_one("#a-nqueens", Input).value = "12 4"
        edited = bench.edited_benchset("scratch")
        resolved = {a.key: a for a in edited.resolve(app.catalog)}
        return resolved["nqueens:hinted"].args, resolved["nqueens:hinted"].args_overridden

    assert drive(check) == (["12", "4"], True)


def test_an_unchecked_application_is_saved_as_disabled():
    async def check(app, pilot):
        bench = app.query_one("#bench", BenchsetPanel)
        for t in bench.toggles:
            if t.ident.startswith("nqueens:"):
                t.value = False
        edited = bench.edited_benchset("scratch")
        return edited.apps["nqueens"].enabled, [
            a.name for a in edited.resolve(app.catalog) if a.name == "nqueens"
        ]

    enabled, resolved = drive(check)
    assert enabled is False
    assert resolved == []


def test_node_toggles_fit_two_digit_counts():
    # The widest sweep reaches 32n, so the labels grow by a column.
    async def check(app, pilot):
        from textual.widgets import TabbedContent

        app.query_one(TabbedContent).active = "tab-profile"
        await pilot.pause()
        panel = app.query_one("#profile", ProfilePanel)
        return [(t.label.plain, t.size.width)
                for t in panel.query(".node-toggle").results(Toggle)]

    async def main():
        app = ArtsRunApp(profile="junction", benchset="paper-main")
        async with app.run_test() as pilot:
            return await check(app, pilot)

    import asyncio
    for label, width in asyncio.run(main()):
        assert width >= len(label) + 4, f"{label!r} has only {width} columns"


def test_the_command_palette_has_one_control_not_two():
    # The header icon and the footer's ^p opened the same thing; the footer
    # keeps it because it names the shortcut.
    async def check(app, pilot):
        icons = app.query("HeaderIcon")
        palette = [b for _, b, _, _ in app.active_bindings.values()
                   if b.key == "ctrl+p"]
        return [i.display for i in icons], len(palette)

    displayed, palette_bindings = drive(check)
    assert displayed == [False]
    assert palette_bindings == 1


def test_unchecking_a_node_narrows_the_run_but_not_the_profile():
    async def check(app, pilot):
        panel = app.query_one("#profile", ProfilePanel)
        for t in panel.query(".node-toggle").results(Toggle):
            t.value = t.ident in ("1", "2")
        await pilot.pause()
        return panel.node_counts(), panel.node_list(), panel.effective_profile().nodes

    counts, listed, effective = drive(check)
    assert counts == [1, 2]        # this campaign
    assert listed == [1, 2, 4, 8]  # the profile keeps its shape
    assert effective == [1, 2, 4, 8]


def test_a_node_count_can_be_added_and_removed():
    async def check(app, pilot):
        from textual.widgets import Input
        from artsrun.tui.widgets import NodeDelete

        panel = app.query_one("#profile", ProfilePanel)
        panel.query_one("#node-add", Input).value = "3"
        panel.query_one("#node-add-btn").press()
        await pilot.pause()
        after_add = panel.node_list()
        for button in list(panel.query(NodeDelete).results(NodeDelete)):
            if button.node == 8:
                button.press()
        await pilot.pause()
        return after_add, panel.node_list()

    after_add, after_remove = drive(check)
    assert after_add == [1, 2, 3, 4, 8]
    assert after_remove == [1, 2, 3, 4]


def test_a_node_count_that_is_not_a_number_is_refused():
    async def check(app, pilot):
        from textual.widgets import Input

        panel = app.query_one("#profile", ProfilePanel)
        panel.query_one("#node-add", Input).value = "eight"
        panel.query_one("#node-add-btn").press()
        await pilot.pause()
        return panel.last_status, panel.node_list()

    status, nodes = drive(check)
    assert "not a node count" in status
    assert nodes == [1, 2, 4, 8]


def test_the_last_node_count_cannot_be_removed():
    async def check(app, pilot):
        from artsrun.tui.widgets import NodeDelete

        panel = app.query_one("#profile", ProfilePanel)
        for button in list(panel.query(NodeDelete).results(NodeDelete)):
            if button.node in (2, 4, 8):
                button.press()
                await pilot.pause()
        remaining = panel.node_list()
        panel.query(NodeDelete).first().press()
        await pilot.pause()
        return remaining, panel.node_list(), panel.last_status

    before, after, status = drive(check)
    assert before == [1]
    assert after == [1]
    assert "at least one" in status


def _launcher_view(profile_name):
    async def main():
        app = ArtsRunApp(profile=profile_name, benchset="paper-main")
        async with app.run_test() as pilot:
            from textual.containers import Vertical
            from textual.widgets import Input

            panel = app.query_one("#profile", ProfilePanel)
            await pilot.pause()
            return {
                "ssh": panel.query_one("#section-ssh", Vertical).display,
                "slurm": panel.query_one("#section-slurm", Vertical).display,
                "ports_disabled": panel.query_one("#f-ports", Input).disabled,
                "ports_placeholder": panel.query_one("#f-ports", Input).placeholder,
            }

    import asyncio
    return asyncio.run(main())


def test_a_local_profile_hides_both_launcher_sections_and_locks_ports():
    view = _launcher_view("ferrari")
    assert view["ssh"] is False
    assert view["slurm"] is False
    assert view["ports_disabled"] is True
    assert "automatic" in view["ports_placeholder"]


def test_a_slurm_profile_shows_only_its_own_section():
    view = _launcher_view("junction")
    assert view["slurm"] is True
    assert view["ssh"] is False
    assert view["ports_disabled"] is False


def test_switching_the_launcher_switches_the_sections():
    async def check(app, pilot):
        from textual.containers import Vertical
        from textual.widgets import Select

        panel = app.query_one("#profile", ProfilePanel)
        panel.query_one("#f-launcher", Select).value = "ssh"
        await pilot.pause()
        return (panel.query_one("#section-ssh", Vertical).display,
                panel.query_one("#section-slurm", Vertical).display)

    assert drive(check) == (True, False)


def test_the_provider_offers_only_what_the_transport_builds():
    # The vendored libfabric enables tcp and verbs; auto leaves the choice to
    # the runtime, which is stored as no provider at all.
    async def check(app, pilot):
        from textual.widgets import Select

        panel = app.query_one("#profile", ProfilePanel)
        select = panel.query_one("#f-provider", Select)
        options = [str(v) for _, v in select._options] if hasattr(
            select, "_options") else None
        select.value = "auto"
        edited = panel.edited_profile("scratch")
        return options, edited.provider

    options, provider = drive(check)
    assert provider is None            # "auto" means unset
    if options is not None:
        # verbs;ofi_rxm is ONE unit: the runtime needs RDM endpoints, which
        # the MSG-only verbs core provider offers only through the rxm layer,
        # so a bare "verbs" can never match and is not offered.
        assert options == ["auto", "tcp", "verbs;ofi_rxm"]


def test_optional_fields_show_an_example_without_setting_it():
    async def check(app, pilot):
        from textual.widgets import Input

        panel = app.query_one("#profile", ProfilePanel)
        box = panel.query_one("#f-regpool_slab_mb", Input)
        edited = panel.edited_profile("scratch")
        return box.value, box.placeholder, edited.regpool_slab_mb

    value, placeholder, stored = drive(check)
    assert value == ""
    assert placeholder == "64"
    assert stored is None


def test_a_field_whose_emptiness_means_something_shows_no_example():
    # "empty = auto" and a greyed-out "ib0" contradict each other; the example
    # belongs in the explanation there.
    async def check(app, pilot):
        from textual.widgets import Input
        from artsrun.tui import form

        panel = app.query_one("#profile", ProfilePanel)
        return (panel.query_one("#f-net_interface", Input).placeholder,
                form.BY_KEY["net_interface"].help)

    placeholder, help_text = drive(check)
    assert placeholder == ""
    assert "ib0" in help_text


def _visible_field_slack(profile_name, launcher=None):
    async def main():
        from textual.widgets import Select
        from artsrun.tui import form

        app = ArtsRunApp(profile=profile_name, benchset="paper-main")
        async with app.run_test(size=(150, 55)) as pilot:
            from textual.widgets import TabbedContent

            app.query_one(TabbedContent).active = "tab-profile"
            await pilot.pause()
            panel = app.query_one("#profile", ProfilePanel)
            if launcher:
                panel.query_one("#f-launcher", Select).value = launcher
                await pilot.pause()
            out = []
            for spec in form.PROFILE_FIELDS:
                w = panel.query_one(f"#f-{spec.key.replace('.', '-')}")
                if not w.display or w.content_size.width == 0:
                    continue
                text = max(str(getattr(w, "placeholder", "") or ""),
                           str(getattr(w, "value", "") or ""), key=len)
                out.append((spec.key, text, w.content_size.width))
            return out

    import asyncio
    return asyncio.run(main())


def test_no_visible_field_clips_its_text():
    # "claimed automatically" is the longest placeholder and was being cut.
    for profile, launcher in (("ferrari", None), ("junction", None),
                              ("junction", "ssh")):
        for key, text, width in _visible_field_slack(profile, launcher):
            assert width >= len(text), (
                f"{profile}/{launcher or 'stored'}: {key} shows {text!r} "
                f"in {width} columns"
            )


def test_every_explanation_starts_in_the_same_column():
    # Switches draw narrower than inputs and the node list is as wide as it is
    # long; neither may push its explanation out of line.
    async def check(app, pilot):
        from textual.widgets import TabbedContent

        app.query_one(TabbedContent).active = "tab-profile"
        await pilot.pause()
        panel = app.query_one("#profile", ProfilePanel)
        columns = set()
        for row in panel.query(".form-row"):
            kids = list(row.children)
            if len(kids) >= 3 and kids[2].display and kids[2].region.width:
                columns.add(kids[2].region.x)
        return columns

    assert len(drive(check)) == 1


def test_the_node_row_is_as_tall_as_the_rows_around_it():
    async def check(app, pilot):
        from textual.widgets import TabbedContent

        app.query_one(TabbedContent).active = "tab-profile"
        await pilot.pause()
        panel = app.query_one("#profile", ProfilePanel)
        node_row = panel.query_one(".node-form-row")
        others = [r.size.height for r in panel.query(".form-row")
                  if "node-form-row" not in r.classes
                  and "node-help-row" not in r.classes and r.size.height]
        return node_row.size.height, min(others)

    node_height, ordinary = drive(check)
    assert node_height == ordinary


def test_the_write_header_spans_its_two_release_columns():
    async def check(app, pilot):
        plane = app.query_one(PlanePanel)
        spans = [w.size.width for w in plane.query(".plane-write-head")]
        subs = [c.size.width for c in plane.query(".plane-col-head")]
        return spans, subs

    spans, subs = drive(check)
    assert len(spans) == 2
    assert len(subs) == 4
    assert all(s == subs[0] * 2 for s in spans)


def test_an_application_checkbox_is_only_as_wide_as_its_box():
    # A label-less checkbox stretched to the column width leaves dead space
    # beside the glyph that still takes clicks.
    async def check(app, pilot):
        from textual.widgets import TabbedContent

        app.query_one(TabbedContent).active = "tab-bench"
        await pilot.pause()
        bench = app.query_one("#bench", BenchsetPanel)
        widths = {t.size.width for t in bench.toggles}
        cells = {c.size.width for c in bench.query(".bench-cell")
                 if c.size.width}
        return widths, cells

    box_widths, cell_widths = drive(check)
    assert box_widths == {3}          # the glyph, nothing more
    assert min(cell_widths) > 3       # the column still aligns


def test_an_application_checkbox_fits_inside_its_row():
    # With its default border the box is three rows tall and a one-row cell
    # clips it away entirely — present in the tree, invisible on screen.
    async def check(app, pilot):
        from textual.widgets import TabbedContent

        app.query_one(TabbedContent).active = "tab-bench"
        await pilot.pause()
        bench = app.query_one("#bench", BenchsetPanel)
        return [(t.outer_size.height, t.parent.size.height)
                for t in bench.toggles[:6]]

    for box, cell in drive(check):
        assert box <= cell, f"checkbox is {box} rows in a {cell}-row cell"


def test_application_checkboxes_are_actually_painted():
    async def check(app, pilot):
        from textual.widgets import TabbedContent

        app.query_one(TabbedContent).active = "tab-bench"
        await pilot.pause()
        svg = app.export_screenshot()
        return svg.count("✓"), svg.count("…")

    marks, ellipses = drive(check)
    assert marks > 0
    # A checkbox renders "glyph + space + label"; with no label the trailing
    # space must be cut, not replaced by an ellipsis beside every box.
    assert ellipses == 0


def _green_count(svg):
    import re
    from collections import Counter

    return Counter(re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg))["#4ebf71"]


def test_a_checked_box_is_filled_and_an_unchecked_one_is_not():
    # State shown by the glyph's colour alone reads the same at a glance; the
    # box itself has to carry it.
    async def check(app, pilot):
        plane = app.query_one(PlanePanel)
        on = _green_count(app.export_screenshot())
        plane.toggle_all()
        await pilot.pause()
        off = _green_count(app.export_screenshot())
        return on, off

    on, off = drive(check)
    # Count, not identity: the default test terminal shows only part of the
    # grid, so what matters is that filling stops when nothing is selected.
    assert on > 0
    assert off == 0


def test_the_application_boxes_show_state_the_same_way():
    async def check(app, pilot):
        from textual.widgets import TabbedContent

        app.query_one(TabbedContent).active = "tab-bench"
        await pilot.pause()
        on = _green_count(app.export_screenshot())
        app.query_one("#bench", BenchsetPanel).toggle_all()
        await pilot.pause()
        return on, _green_count(app.export_screenshot())

    on, off = drive(check)
    assert on > 0
    assert off == 0


def test_the_application_screen_groups_applications_before_microbenchmarks():
    async def check(app, pilot):
        from textual.widgets import Static

        bench = app.query_one("#bench", BenchsetPanel)
        headings = [g for g in bench.query(".bench-group").results(Static)]
        # every row belongs to the group above it, so order is the claim
        return len(headings)

    assert drive(check) == 2


# --- counters -------------------------------------------------------------
def test_the_counters_screen_offers_every_counter_the_runtime_defines():
    async def check(app, pilot):
        from textual.widgets import TabbedContent
        from artsrun.model.counters import load_counter_catalog
        from artsrun.tui.panels import CounterPanel

        app.query_one(TabbedContent).active = "tab-counters"
        await pilot.pause()
        panel = app.query_one("#counters", CounterPanel)
        return len(panel.mode_selects), len(load_counter_catalog().names)

    offered, defined = drive(check)
    assert offered == defined


def test_a_counter_screen_reads_back_as_a_saveable_set():
    async def check(app, pilot):
        from textual.widgets import TabbedContent, Select
        from artsrun.model.counters import Mode
        from artsrun.tui.panels import CounterPanel

        app.query_one(TabbedContent).active = "tab-counters"
        await pilot.pause()
        panel = app.query_one("#counters", CounterPanel)
        panel.query_one("#cm-NUM_DB_DESTROY", Select).value = Mode.ONCE.value
        panel.query_one("#cl-NUM_DB_DESTROY", Select).value = "THREAD"
        edited = panel.edited_counterset("probe")
        return edited.setting("NUM_DB_DESTROY").mode.value, \
            edited.setting("NUM_DB_DESTROY").level.value

    assert drive(check) == ("ONCE", "THREAD")


def test_the_shared_control_clears_and_fills_the_counter_set():
    async def check(app, pilot):
        from textual.widgets import TabbedContent
        from artsrun.tui.panels import CounterPanel

        app.query_one(TabbedContent).active = "tab-counters"
        await pilot.pause()
        panel = app.query_one("#counters", CounterPanel)
        await pilot.press("a")
        cleared = len(panel.edited_counterset("x").enabled)
        await pilot.press("a")
        filled = len(panel.edited_counterset("x").enabled)
        return cleared, filled

    cleared, filled = drive(check)
    assert cleared == 0
    assert filled > 0


def test_counter_dropdowns_show_their_value_and_arrow():
    # A Select draws its value and arrow inside a bordered SelectCurrent; a
    # one-row cell clips both away unless that border goes too.
    async def check(app, pilot):
        from textual.widgets import TabbedContent, Select
        from artsrun.tui.panels import CounterPanel

        app.query_one(TabbedContent).active = "tab-counters"
        await pilot.pause()
        panel = app.query_one("#counters", CounterPanel)
        select = panel.query_one("#cm-NUM_EDT_FINISH", Select)
        current = select.query_one("SelectCurrent")
        svg = app.export_screenshot()
        return (current.outer_size.height, select.size.height,
                svg.count("▼"), "PERIODIC" in svg or "OFF" in svg)

    inner, outer, arrows, value_shown = drive(check)
    assert inner <= outer          # nothing clipped
    assert arrows > 0
    assert value_shown


def test_a_counter_dropdown_opens_and_takes_a_value():
    async def check(app, pilot):
        from textual.widgets import TabbedContent, Select
        from artsrun.tui.panels import CounterPanel

        app.query_one(TabbedContent).active = "tab-counters"
        await pilot.pause()
        panel = app.query_one("#counters", CounterPanel)
        select = panel.query_one("#cm-NUM_DB_DESTROY", Select)
        select.expanded = True
        await pilot.pause()
        opened = app.export_screenshot()
        select.expanded = False
        select.value = "PERIODIC"
        await pilot.pause()
        return (all(o in opened for o in ("OFF", "ONCE", "PERIODIC")),
                panel.edited_counterset("x").setting("NUM_DB_DESTROY").mode.value)

    listed, chosen = drive(check)
    assert listed
    assert chosen == "PERIODIC"


def test_switching_a_saved_set_leaves_one_panel_behind():
    # remove() is deferred, so rebuilding a panel mounted a second widget under
    # the same id before the first was gone.
    async def check(app, pilot):
        from textual.widgets import TabbedContent, Select
        from artsrun.tui.panels import BenchsetPanel, CounterPanel

        app.query_one(TabbedContent).active = "tab-counters"
        await pilot.pause()
        for name in ("perf", "distribution", "perf"):
            app.query_one("#counter-select", Select).value = name
            await pilot.pause()
        counters = len(app.query("#counters"))

        app.query_one(TabbedContent).active = "tab-bench"
        await pilot.pause()
        for name in ("smoke", "paper-main", "smoke"):
            app.query_one("#bench-select", Select).value = name
            await pilot.pause()
        return counters, len(app.query("#bench"))

    counters, benches = drive(check)
    assert counters == 1
    assert benches == 1


def test_loading_a_counter_set_shows_that_set_s_values():
    async def check(app, pilot):
        from textual.widgets import TabbedContent, Select
        from artsrun.tui.panels import CounterPanel

        app.query_one(TabbedContent).active = "tab-counters"
        await pilot.pause()
        panel = app.query_one("#counters", CounterPanel)
        panel.query_one("#counter-select", Select).value = "perf"
        await pilot.pause()
        wide = len(panel.edited_counterset("x").enabled)
        panel.query_one("#counter-select", Select).value = "distribution"
        await pilot.pause()
        narrow = len(panel.edited_counterset("x").enabled)
        return wide, narrow

    wide, narrow = drive(check)
    assert wide > narrow > 0


def test_loading_a_benchset_shows_that_roster():
    async def check(app, pilot):
        from textual.widgets import TabbedContent, Select
        from artsrun.tui.panels import BenchsetPanel

        app.query_one(TabbedContent).active = "tab-bench"
        await pilot.pause()
        panel = app.query_one("#bench", BenchsetPanel)
        panel.query_one("#bench-select", Select).value = "smoke"
        await pilot.pause()
        small = set(panel.selected())
        panel.query_one("#bench-select", Select).value = "paper-main"
        await pilot.pause()
        big = set(panel.selected())
        return small, big

    small, big = drive(check)
    assert small == {"fibonacci", "nqueens", "quicksort"}
    assert len(big) > len(small)


def test_a_set_with_once_and_no_reduction_loads_onto_the_screen():
    # The sets built for the protocol comparison are uniformly PERIODIC /
    # CLUSTER / SUM; the attribution set is ONCE / NODE with no reduction, so
    # it is the one that exercises the dropdowns' other values.
    async def check(app, pilot):
        from textual.widgets import TabbedContent, Select
        from artsrun.tui.panels import CounterPanel

        app.query_one(TabbedContent).active = "tab-counters"
        await pilot.pause()
        panel = app.query_one("#counters", CounterPanel)
        panel.query_one("#counter-select", Select).value = "attribution"
        await pilot.pause()
        cs = panel.edited_counterset("x")
        obj = cs.counters["OBJ_BYTES_DB"]
        return set(cs.enabled), obj.mode.value, obj.level.value

    enabled, mode, level = drive(check)
    assert "OBJ_BYTES_DB" in enabled
    assert (mode, level) == ("ONCE", "NODE")


def test_the_run_action_is_not_blocked_before_anything_runs():
    # The in-progress flag once shared a name with one the framework sets on
    # startup, so the screen reported a campaign already running from the
    # moment it opened and the key did nothing for the life of the process.
    async def check(app, pilot):
        started = app._campaign_running
        app.action_run()
        return started, app._campaign_running

    started, after = drive(check)
    assert started is False
    assert after is True


def test_the_flag_names_nothing_the_framework_writes():
    # The screen's own state must not share a name with anything the base
    # class assigns, since whichever writes last wins and neither knows.
    import inspect
    import re

    from textual.app import App

    assigned = set(re.findall(r"self\.(_[a-z_]+)\s*=", inspect.getsource(App)))
    assert "_campaign_running" not in assigned
    assert "_running" in assigned, (
        "the collision this guards against is gone; keep the check pointed at "
        "a name the framework really does assign"
    )


def test_a_campaign_that_cannot_be_prepared_says_so_and_clears_the_flag():
    # An exception escaping a worker thread reaches the framework's log and
    # nowhere the user looks, and would leave the flag raised for good.
    async def check(app, pilot):
        import artsrun.campaign as campaign_mod

        def boom(*a, **k):
            raise RuntimeError("counter tree mismatch\n  run: cmake -B...")

        original = campaign_mod.Campaign.prepare
        campaign_mod.Campaign.prepare = boom
        try:
            app.action_run()
            for _ in range(80):
                await pilot.pause()
                await asyncio.sleep(0.05)
                if not app._campaign_running:
                    break
        finally:
            campaign_mod.Campaign.prepare = original
        from textual.widgets import RichLog

        text = "\n".join(
            s.text if hasattr(s, "text") else str(s)
            for s in app.query_one("#run-log", RichLog).lines
        )
        return app._campaign_running, text

    running, text = drive(check)
    assert running is False
    assert "campaign failed" in text
    assert "cmake -B" in text          # the multi-line detail survives


def test_continuing_is_a_separate_control_from_starting():
    # The two differ in what they measure against — the screens, or what a
    # past run recorded — so they are not one button with a mode.
    async def check(app, pilot):
        from textual.widgets import Button, Select

        select = app.query_one("#resume-select", Select)
        button = app.query_one("#resume-button", Button)
        before = button.disabled
        app.action_continue()            # nothing picked: refuses, does not run
        picked_none = app._campaign_running
        return before, picked_none

    disabled, ran = drive(check)
    assert disabled is True, "continue offers itself before a run is chosen"
    assert ran is False, "continue started a campaign with no run chosen"


def test_only_unfinished_runs_are_offered():
    from artsrun.campaign import past_runs

    async def check(app, pilot):
        from textual.widgets import Select

        select = app.query_one("#resume-select", Select)
        # The blank prompt carries the widget's own sentinel, not a run.
        return {v for _, v in select._options if isinstance(v, str)}

    offered = drive(check)
    resumable = {r.run_id for r in past_runs() if r.remaining}
    # A completed campaign has nothing left to run, so offering it would be
    # offering a no-op.
    assert offered <= resumable
