"""The application documents: one page per application, openable from the
roster and from the command line, falling back to catalog facts."""

from __future__ import annotations

import asyncio

from artsrun.data import load_doc
from artsrun.docs import document, facts_markdown
from artsrun.model.catalog import Kind, load_catalog
from artsrun.tui.app import ArtsRunApp
from artsrun.tui.docview import AppDocScreen

# Every document tells the same structural story in the same order, so a
# reader who knows one page knows them all.
REQUIRED_SECTIONS = [
    "## Overview",
    "## Parameters",
    "## Structure",
    "## Wiring",
    "## Flow",
    "## Placement (base)",
    "## Sizing",
]


def drive(coro_factory):
    async def main():
        app = ArtsRunApp(profile="ferrari", benchset="paper-main")
        async with app.run_test() as pilot:
            return await coro_factory(app, pilot)

    return asyncio.run(main())


def test_a_written_document_loads_and_a_missing_one_is_none():
    assert load_doc("fibonacci")
    assert load_doc("no_such_application") is None


def test_every_application_row_has_a_document():
    """A result is claimed about an application; every application row the
    roster offers must open to a real structural document, not the fallback."""
    catalog = load_catalog()
    missing = [a.name for a in catalog.rows_of(Kind.APP)
               if load_doc(a.name) is None]
    assert not missing, f"application rows without a document: {missing}"


def test_every_written_document_tells_the_whole_story():
    catalog = load_catalog()
    for name in catalog.apps:
        text = load_doc(name)
        if text is None:
            continue
        assert text.startswith(f"# {name}\n"), f"{name}: title is not the entry name"
        for section in REQUIRED_SECTIONS:
            assert f"\n{section}" in text, f"{name}: missing '{section}'"


def test_an_entry_without_a_document_still_renders_its_facts():
    catalog = load_catalog()
    probe = next(a for a in catalog.apps.values() if a.kind is Kind.ATTACK)
    text = document(probe)
    assert probe.name in text
    assert probe.binary in facts_markdown(probe)


def test_clicking_an_application_name_opens_its_document():
    async def check(app, pilot):
        app.query_one("#bench")
        await app.run_action("show_app_doc('fibonacci')")
        await pilot.pause()
        opened = isinstance(app.screen, AppDocScreen)
        name = app.screen.entry.name if opened else None
        await pilot.press("escape")
        await pilot.pause()
        closed = not isinstance(app.screen, AppDocScreen)
        return opened, name, closed

    opened, name, closed = drive(check)
    assert opened and name == "fibonacci" and closed


def test_an_unknown_name_from_a_stale_label_is_ignored():
    async def check(app, pilot):
        await app.run_action("show_app_doc('no_such_application')")
        await pilot.pause()
        return isinstance(app.screen, AppDocScreen)

    assert drive(check) is False


def test_an_unsupported_row_shows_its_boxes_but_takes_no_selection():
    """The row stays visible — the document and the reason are the point —
    but no benchset, click, or all/none sweep can turn it on."""
    async def check(app, pilot):
        bench = app.query_one("#bench")
        from artsrun.tui.widgets import Toggle

        dead = [t for t in bench.query(".app-toggle").results(Toggle)
                if t.disabled]
        names = {t.ident.partition(":")[0] for t in dead}
        selected_before = set(bench.selected())
        await pilot.press("a")
        await pilot.press("a")
        selected_after = set(bench.selected())
        return names, selected_before | selected_after

    names, ever_selected = drive(check)
    assert "CoMD_intel_chandra" in names and "CoMD_sdsc2" in names
    assert not ever_selected & names

    catalog = load_catalog()
    for name in ("CoMD_intel_chandra", "CoMD_sdsc2"):
        assert catalog.apps[name].unsupported
        # No benchset can override an unsupported row.
        from artsrun import store

        assert not store.default_benchset().is_enabled(catalog.apps[name])


def test_clicking_a_roster_name_opens_that_row_s_document():
    async def check(app, pilot):
        await pilot.press("3")
        await pilot.pause()
        # The first .bench-name is the header row's empty label; the first
        # roster row follows it.  The link spans only the name's own
        # characters and the label is padded, so aim at its first column
        # rather than its centre.
        target = app.query(".bench-name").nodes[1]
        await pilot.click(target, offset=(1, 0))
        await pilot.pause()
        opened = isinstance(app.screen, AppDocScreen)
        return opened, app.screen.entry.name if opened else None

    opened, name = drive(check)
    assert opened
    catalog = load_catalog()
    assert name in catalog.apps
