"""The live view over a run directory: the table, the detail pane, the tail."""

from __future__ import annotations

import asyncio

from rich.text import Text
from textual.widgets import DataTable, RichLog, Static

from artsrun.tui.runview import CellDetail, RunView, WatchApp
from test_watch import _cell, _finished, _track, _write


def drive(run_dir, coro):
    async def main():
        app = WatchApp(run_dir)
        async with app.run_test(size=(160, 48)) as pilot:
            await pilot.pause()
            return await coro(app, pilot)

    return asyncio.run(main())


def _plain(value) -> str:
    return value.plain if isinstance(value, Text) else str(value)


def test_the_table_holds_one_row_per_cell_with_its_axes(tmp_path):
    cells = [_cell("arts_val_wb"), _cell("arts_excl_retain", nodes=2)]
    _write(tmp_path, cells)
    _track(tmp_path, [_finished(cells[0], "1.0")])

    async def check(app, pilot):
        table = app.query_one("#watch-table", DataTable)
        return table.row_count, [_plain(v) for v in table.get_row(cells[0].key)]

    count, row = drive(tmp_path, check)
    assert count == 2
    _, app_name, version, runtime, family, write, release, n, rep, \
        status, wall, scalar, verdict = row
    assert (app_name, version, runtime) == ("app", "asborn", "arts_val_wb")
    assert (family, write, release) == ("VAL", "WB", "RETAIN")
    assert (n, rep) == ("1", "1")
    assert status == "ok"
    assert wall == "2.5s"
    assert scalar == "1.0"
    assert verdict == "OK"


def test_new_events_grow_and_recolor_the_table_live(tmp_path):
    cells = [_cell("arts_val_wb"), _cell("arts_excl_retain")]
    _write(tmp_path, cells)

    async def check(app, pilot):
        view = app.query_one(RunView)
        table = app.query_one("#watch-table", DataTable)
        before = _plain(table.get_row(cells[1].key)[9])
        _track(tmp_path, [_finished(cells[0], "1.0"),
                          _finished(cells[1], "2.0")])
        view._tick()
        await pilot.pause()
        after = _plain(table.get_row(cells[1].key)[9])
        verdict = _plain(table.get_row(cells[1].key)[12])
        summary = str(app.query_one("#watch-summary", Static).content)
        return before, after, verdict, summary

    before, after, verdict, summary = drive(tmp_path, check)
    assert before == "pending"
    assert after == "ok"
    assert verdict == "DIFF"
    assert "broken" in summary


def test_a_single_click_opens_the_cell_under_the_mouse(tmp_path):
    cells = [_cell("arts_val_wb"), _cell("arts_excl_retain")]
    _write(tmp_path, cells)

    async def check(app, pilot):
        detail = app.query_one(CellDetail)
        # One click on a row the cursor is NOT on (the cursor starts on the
        # first row; the second data row sits at y=2 under the header).
        await pilot.click("#watch-table", offset=(10, 2))
        await pilot.pause()
        first = (detail.display, detail.key)
        # A second click on the same row closes it again.
        await pilot.click("#watch-table", offset=(10, 2))
        await pilot.pause()
        second = detail.display
        return first, second

    (opened, key), closed = drive(tmp_path, check)
    assert opened
    assert key == cells[1].key
    assert not closed


def test_an_open_pane_follows_the_cursor(tmp_path):
    cells = [_cell("arts_val_wb"), _cell("arts_excl_retain")]
    _write(tmp_path, cells)

    async def check(app, pilot):
        table = app.query_one("#watch-table", DataTable)
        table.focus()
        await pilot.press("enter")     # open on the first row
        await pilot.pause()
        detail = app.query_one(CellDetail)
        before = detail.key
        await pilot.press("down")      # cursor moves; the pane follows
        await pilot.pause()
        return before, detail.key, detail.display

    before, after, still_open = drive(tmp_path, check)
    assert before == cells[0].key
    assert after == cells[1].key
    assert still_open


def test_the_buttons_open_the_configurations_behind_the_cell(tmp_path):
    from textual.widgets import Button

    cfg = tmp_path / "cfg" / "arts_1n.cfg"
    cfg.parent.mkdir()
    cfg.write_text("workers = 2\nprogress = 1\n")
    counters = tmp_path / "cfg" / "counters_perf.cfg"
    counters.write_text("EDT_RUN = PERIODIC NODE SUM\n")
    _write(tmp_path, [_cell("arts_val_wb", cfg=cfg)], counters_cfg=counters)

    async def check(app, pilot):
        table = app.query_one("#watch-table", DataTable)
        table.focus()
        await pilot.press("enter")
        await pilot.pause()
        detail = app.query_one(CellDetail)
        out = []
        for btn in ("#detail-btn-cfg", "#detail-btn-counters"):
            await pilot.click(f"CellDetail {btn}")
            await pilot.pause()
            box = detail.query_one("#detail-cfg-box")
            out.append((box.display,
                        str(detail.query_one("#detail-cfg", Static).content)))
        # Pressing the open one again folds it away.  (The wait outlasts the
        # button's active-effect window, which swallows a same-instant click.)
        await pilot.pause(0.4)
        await pilot.click("CellDetail #detail-btn-counters")
        await pilot.pause()
        out.append(detail.query_one("#detail-cfg-box").display)
        return out

    (cfg_open, cfg_text), (ctr_open, ctr_text), folded = drive(tmp_path, check)
    assert cfg_open and "workers = 2" in cfg_text
    assert ctr_open and "EDT_RUN = PERIODIC NODE SUM" in ctr_text
    assert "counters_perf.cfg" in ctr_text
    assert not folded


def test_the_scaling_button_swaps_in_place_and_back(tmp_path):
    # Scaling belongs to the run it was measured in: the button on the summary
    # row swaps the table for the reading, and the same button swaps it back.
    from textual.widgets import Button

    from artsrun.tui.runview import ScalingView

    cells = [_cell("arts_val_wb"), _cell("arts_val_wb", nodes=2)]
    _write(tmp_path, cells)
    _track(tmp_path, [_finished(cells[0], "1.0"), _finished(cells[1], "1.0")])

    async def check(app, pilot):
        scaling = app.query_one(ScalingView)
        table = app.query_one("#watch-table", DataTable)
        button = app.query_one("#watch-scaling-btn", Button)
        await pilot.click("#watch-scaling-btn")
        await pilot.pause()
        swapped = (scaling.display, table.display, str(button.label))
        await pilot.pause(0.4)
        await pilot.click("#watch-scaling-btn")
        await pilot.pause()
        return swapped, (scaling.display, table.display, str(button.label))

    swapped, back = drive(tmp_path, check)
    assert swapped == (True, False, "table")
    assert back == (False, True, "scaling")


def test_g_swaps_the_table_for_the_scaling_reading(tmp_path):
    from artsrun.tui.runview import ScalingView

    cells = [_cell("arts_val_wb"), _cell("arts_val_wb", nodes=2)]
    _write(tmp_path, cells)
    _track(tmp_path, [
        {"t": 1.0, "event": "finished", "cell": cells[0].key, "status": "ok",
         "rc": 0, "wall_s": 10.0, "note": "", "scalar": "1.0"},
        {"t": 2.0, "event": "finished", "cell": cells[1].key, "status": "ok",
         "rc": 0, "wall_s": 8.0, "note": "", "scalar": "1.0"},
    ])

    async def check(app, pilot):
        table = app.query_one("#watch-table", DataTable)
        table.focus()
        await pilot.press("g")
        await pilot.pause()
        scaling = app.query_one(ScalingView)
        shown = (scaling.display, not table.display)
        grid = scaling.query_one(DataTable)
        row = [_plain(v) for v in grid.get_row_at(0)]
        await pilot.press("escape")
        await pilot.pause()
        return shown, row, (scaling.display, table.display)

    (scaling_on, table_off), row, (scaling_after, table_after) = \
        drive(tmp_path, check)
    assert scaling_on and table_off
    app_name, version, entry, one, two = row
    assert (app_name, version, entry) == ("app", "asborn", "arts_val_wb")
    assert one == "10.00s"
    assert two == "8.00s ×1.25"  # speedup vs the 1n base
    assert scaling_after is False and table_after is True


def test_selecting_a_row_opens_the_invocation_and_its_log(tmp_path):
    cells = [_cell("arts_val_wb")]
    _write(tmp_path, cells)
    log_dir = tmp_path / "cells"
    log_dir.mkdir()
    (log_dir / cells[0].log_name).write_text("$ the command\nhello from rank 0\n")

    async def check(app, pilot):
        table = app.query_one("#watch-table", DataTable)
        table.focus()
        await pilot.press("enter")
        await pilot.pause()
        detail = app.query_one(CellDetail)
        opened = detail.display
        cmd = str(detail.query_one("#detail-cmd", Static).content)
        lines = len(detail.query_one("#detail-log", RichLog).lines)
        await pilot.press("escape")
        await pilot.pause()
        return opened, cmd, lines, detail.display

    opened, cmd, lines, still_open = drive(tmp_path, check)
    assert opened
    assert "timeout -k 1 60 /opt/bin/app 3" in cmd
    assert "stdin: /dev/null" in cmd
    assert lines >= 2
    assert not still_open
