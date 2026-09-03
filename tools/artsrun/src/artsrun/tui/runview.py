"""The live campaign view: a table of every cell, and the cell under the glass.

Everything shown here is read back from the run directory — the manifest for
what was asked, the track for what has happened, the per-cell log for what a
run is saying right now.  The view never touches the campaign machinery, so
the same screen watches an in-process run, a detached one, or a finished one.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, DataTable, Footer, Header, RichLog, Static

from artsrun.check import Verdict
from artsrun.run.types import Status
from artsrun.watch.state import CellView, LogTail, RunState

_STATUS_STYLE = {
    Status.PENDING: ("pending", "dim"),
    Status.SUBMITTED: ("queued", "yellow"),
    Status.RUNNING: ("running", "bold yellow"),
    Status.ENDING: ("ending", "dim yellow"),
    Status.OK: ("ok", "green"),
    Status.FAIL: ("fail", "bold red"),
    Status.TIMEOUT: ("timeout", "bold red"),
    Status.SKIPPED: ("skipped", "dim"),
}

_VERDICT_STYLE = {
    Verdict.OK: ("OK", "green"),
    Verdict.DISAGREE: ("DIFF", "bold white on red"),
    Verdict.FAIL: ("FAIL", "red"),
    Verdict.EXPECT_FAIL: ("EXP!", "bold yellow"),
    Verdict.NA: ("--", "dim"),
    Verdict.LONE: ("LONE", "magenta"),
}

# Width None auto-sizes to content (identity columns never shrink back).
_COLUMNS = (
    ("#", 4), ("app", None), ("version", 12), ("runtime", 16), ("family", 8),
    ("write", 5), ("release", 7), ("n", 3), ("rep", 3), ("status", 8),
    ("wall", 8), ("scalar", 18), ("verdict", 6),
)


def _fmt_wall(view: CellView) -> Text:
    seconds = view.elapsed_s
    if not seconds and view.status in (Status.PENDING, Status.SKIPPED):
        return Text("")
    if seconds >= 3600:
        text = f"{int(seconds) // 3600}:{int(seconds) % 3600 // 60:02d}:{int(seconds) % 60:02d}"
    elif seconds >= 60:
        text = f"{int(seconds) // 60}:{int(seconds) % 60:04.1f}"
    else:
        text = f"{seconds:.1f}s"
    if view.status is Status.ENDING:
        # Frozen where the drain began — nothing is computing any more.
        return Text(text, style="dim yellow")
    if view.active:
        return Text(text, style="yellow")
    return Text(text)


def _fmt_scalar(view: CellView) -> Text:
    value = view.scalar or ""
    if len(value) > 17:
        value = value[:16] + "…"
    if view.verdict is Verdict.DISAGREE:
        return Text(value, style="red")
    return Text(value)


def _row_cells(index: int, view: CellView) -> list[Text]:
    status, status_style = _STATUS_STYLE.get(view.status, ("?", ""))
    verdict, verdict_style = ("", "")
    if view.verdict is not None:
        verdict, verdict_style = _VERDICT_STYLE[view.verdict]
    dim = "dim" if view.status is Status.PENDING else ""
    return [
        Text(str(index), style="dim"),
        Text(view.app_name, style=dim or "bold"),
        Text(view.version, style=dim),
        Text(view.entry_key, style=dim),
        Text(view.family, style=dim),
        Text(view.write, style=dim),
        Text(view.release, style=dim),
        Text(str(view.nodes or ""), style=dim),
        Text(str(view.repeat or ""), style=dim),
        Text(status, style=status_style),
        _fmt_wall(view),
        _fmt_scalar(view),
        Text(verdict, style=verdict_style),
    ]


def _snapshot(view: CellView) -> tuple:
    """What a row shows; a row is repainted only when this changes.  Active
    cells carry a live clock, so they change on every look by construction."""
    return (
        view.status, view.verdict, view.scalar, view.note,
        round(view.elapsed_s, 1), view.command != "",
    )


class CellDetail(Vertical):
    """One cell opened up: the exact invocation, its state, and its output.

    The invocation block scrolls within a bounded height, so a long Slurm
    script can never push the live log below the fold — following the output
    is what this pane is mostly for.
    """

    DEFAULT_CSS = """
    CellDetail {
        height: 60%;
        border-top: heavy $primary;
        padding: 0 1;
    }
    CellDetail #detail-head { height: auto; }
    CellDetail #detail-buttons {
        height: auto;
        margin-top: 1;
    }
    CellDetail #detail-buttons Button {
        height: 1;
        border: none;
        min-width: 18;
        margin-right: 2;
        background: $boost;
        color: $text;
    }
    CellDetail #detail-cmd-box {
        height: auto;
        max-height: 40%;
        background: $boost;
        margin: 1 0;
    }
    CellDetail #detail-cmd {
        height: auto;
        padding: 0 1;
    }
    CellDetail #detail-cfg-box {
        height: auto;
        max-height: 40%;
        background: $boost;
        margin-bottom: 1;
    }
    CellDetail #detail-cfg {
        height: auto;
        padding: 0 1;
    }
    CellDetail #detail-log {
        height: 1fr;
        min-height: 6;
        background: $surface;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.view: CellView | None = None
        self._tail: LogTail | None = None
        self.counters_cfg: Path | None = None
        self._cfg_showing: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="detail-head")
        with Horizontal(id="detail-buttons"):
            yield Button("run config", id="detail-btn-cfg")
            yield Button("counter config", id="detail-btn-counters")
        with VerticalScroll(id="detail-cmd-box"):
            yield Static("", id="detail-cmd")
        box = VerticalScroll(id="detail-cfg-box")
        box.display = False
        with box:
            yield Static("", id="detail-cfg")
        yield RichLog(id="detail-log", wrap=False, markup=False,
                      auto_scroll=True, max_lines=4000)

    @property
    def key(self) -> str | None:
        return self.view.key if self.view else None

    def show(self, view: CellView, counters_cfg: Path | None = None) -> None:
        self.view = view
        self.counters_cfg = counters_cfg
        self._tail = LogTail(view.log_path) if view.log_path else None
        log = self.query_one("#detail-log", RichLog)
        log.clear()
        if self._tail is None:
            log.write("(no log yet)")
        self._cfg_showing = None
        self.query_one("#detail-cfg-box").display = False
        self.query_one("#detail-btn-cfg", Button).display = bool(view.cfg)
        self.query_one("#detail-btn-counters", Button).display = (
            counters_cfg is not None
        )
        self.refresh_view()
        self.tick()

    # -- the configuration files behind the cell ---------------------------
    @on(Button.Pressed, "#detail-btn-cfg")
    def _cfg_pressed(self) -> None:
        self._toggle_cfg("run")

    @on(Button.Pressed, "#detail-btn-counters")
    def _counters_pressed(self) -> None:
        self._toggle_cfg("counters")

    def _toggle_cfg(self, which: str) -> None:
        box = self.query_one("#detail-cfg-box")
        if self._cfg_showing == which:
            self._cfg_showing = None
            box.display = False
            return
        path = (Path(self.view.cfg) if which == "run" and self.view
                else self.counters_cfg)
        if path is None:
            return
        body = Text()
        body.append(f"# {path}\n", style="dim")
        try:
            body.append(path.read_text(errors="replace").rstrip("\n"))
        except OSError:
            body.append("(file not found)", style="italic")
        self.query_one("#detail-cfg", Static).update(body)
        self._cfg_showing = which
        box.display = True

    def refresh_view(self) -> None:
        view = self.view
        if view is None:
            return
        head = Text()
        head.append(view.key, style="bold")
        status, style = _STATUS_STYLE.get(view.status, ("?", ""))
        head.append("   ")
        head.append(status, style=style)
        if view.rc is not None:
            rc = f"rc={view.rc}"
            if view.rc == 124:
                rc += " (timeout)"
            head.append(f"   {rc}", style="red" if view.rc else "green")
        if view.elapsed_s:
            head.append(f"   {view.elapsed_s:.1f}s")
        if view.scalar is not None:
            head.append(f"   scalar={view.scalar}")
        if view.verdict is not None:
            text, style = _VERDICT_STYLE[view.verdict]
            head.append("   ")
            head.append(text, style=style)
            if view.consensus is not None:
                head.append(f" (consensus {view.consensus})", style="dim")
        for name, value in sorted(view.extra.items()):
            head.append(f"   {name}={value}", style="dim")
        if view.note:
            head.append(f"\n{view.note}", style="italic")
        self.query_one("#detail-head", Static).update(head)

        cmd = Text()
        cmd.append("$ ", style="dim")
        cmd.append(view.command or "(command unknown — no manifest)")
        if view.script:
            cmd.append("\n\n")
            cmd.append("script:\n", style="dim")
            cmd.append(view.script.rstrip("\n"))
        for name, value in sorted(view.env.items()):
            cmd.append(f"\n{name}={value}", style="dim")
        extras = []
        if view.timeout_s:
            extras.append(f"timeout {view.timeout_s}s")
        extras.append("stdin: /dev/null · stderr merged into stdout")
        if view.log_path:
            extras.append(f"log: {view.log_path}")
        cmd.append("\n" + " · ".join(extras), style="dim")
        self.query_one("#detail-cmd", Static).update(cmd)

    def tick(self) -> None:
        if self._tail is None and self.view and self.view.log_path:
            # The log appears when the cell starts; pick it up then.
            self._tail = LogTail(self.view.log_path)
        if self._tail is None:
            return
        chunk = self._tail.read_new()
        if not chunk:
            return
        log = self.query_one("#detail-log", RichLog)
        for line in chunk.splitlines():
            # Runtime logs carry ANSI colour; parsed, not shown as escapes.
            log.write(Text.from_ansi(line))


class ScalingView(Vertical):
    """Strong scaling, read live: best wall per node count for every
    (application, version, configuration) that has completed cells.

    Each wider cell carries its speedup against the row's smallest measured
    node count, coloured by parallel efficiency — an anti-scaling row turns
    red the moment its wider run comes back slower.
    """

    DEFAULT_CSS = """
    ScalingView #scaling-note {
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }
    ScalingView #scaling-table { height: 1fr; }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "best wall over repeats · ×speedup vs the row's smallest measured "
            "node count · colour = parallel efficiency "
            "(green ≥ 75%, yellow ≥ 45%, red below)",
            id="scaling-note",
        )
        yield DataTable(id="scaling-table", cursor_type="row",
                        zebra_stripes=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fingerprint: tuple | None = None

    @staticmethod
    def _cell(walls: dict[int, float], n: int) -> Text:
        wall = walls.get(n)
        if wall is None:
            return Text("–", style="dim")
        base = min(walls)
        text = Text(f"{wall:.2f}s")
        if n == base or len(walls) < 2:
            return text
        speedup = walls[base] / wall
        efficiency = speedup / (n / base)
        style = ("green" if efficiency >= 0.75
                 else "yellow" if efficiency >= 0.45 else "red")
        text.append(f" ×{speedup:.2f}", style=style)
        return text

    def update_from(self, state: RunState) -> None:
        node_counts, rows = state.scaling()
        fingerprint = tuple(
            (r["app"], r["version"], r["entry"], tuple(sorted(r["walls"].items())))
            for r in rows
        )
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint
        table = self.query_one("#scaling-table", DataTable)
        table.clear(columns=True)
        table.add_column("app")
        table.add_column("version", width=12)
        table.add_column("configuration", width=16)
        for n in node_counts:
            table.add_column(f"{n}n", width=14)
        last = None
        for row in rows:
            group = (row["app"], row["version"])
            first = group != last
            last = group
            table.add_row(
                Text(row["app"], style="bold") if first else Text(""),
                Text(row["version"]) if first else Text(""),
                Text(row["entry"]),
                *(self._cell(row["walls"], n) for n in node_counts),
            )


class RunView(Vertical):
    """The campaign table, one row per cell, growing and recoloring live."""

    DEFAULT_CSS = """
    RunView #watch-topbar {
        height: auto;
        background: $panel;
    }
    RunView #watch-summary {
        width: 1fr;
        height: auto;
        padding: 0 1;
        background: $panel;
    }
    RunView #watch-scaling-btn {
        height: 1;
        border: none;
        min-width: 11;
        background: $boost;
        color: $text;
    }
    RunView #watch-table { height: 1fr; }
    """

    BINDINGS = [
        ("escape", "close_detail", "close"),
        ("g", "toggle_scaling", "scaling"),
    ]

    class ManifestLoaded(Message):
        """The run's manifest appeared: the table has taken over from the
        build log as the thing worth watching."""

    def __init__(self, poll_s: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.state: RunState | None = None
        self.poll_s = poll_s
        self._snapshots: dict[str, tuple] = {}
        self._known: set[str] = set()
        self._clicked_at = 0.0
        self._announced = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="watch-topbar"):
            yield Static("waiting for a run…", id="watch-summary")
            # Scaling belongs to the run it was measured in, so its reading
            # lives right here: one press swaps the table for it, the next
            # swaps back.
            yield Button("scaling", id="watch-scaling-btn")
        table = DataTable(id="watch-table", cursor_type="row",
                          zebra_stripes=True)
        yield table
        scaling = ScalingView(id="watch-scaling")
        scaling.display = False
        yield scaling
        detail = CellDetail(id="watch-detail")
        detail.display = False
        yield detail

    def on_mount(self) -> None:
        self._ensure_columns()
        self.set_interval(self.poll_s, self._tick)

    def _ensure_columns(self) -> None:
        # Reached from both mount and attach, whichever comes first: mount
        # order between this widget and whoever attaches it is not a contract.
        table = self.query_one("#watch-table", DataTable)
        if table.columns:
            return
        for name, width in _COLUMNS:
            table.add_column(name, key=name, width=width)

    # -- attachment ----------------------------------------------------------
    def attach(self, run_dir: Path) -> None:
        """Point the view at a run directory (from any thread's request)."""
        self._ensure_columns()
        self.state = RunState(run_dir)
        self._snapshots.clear()
        self._known.clear()
        self._announced = False
        table = self.query_one("#watch-table", DataTable)
        table.clear()
        detail = self.query_one(CellDetail)
        detail.display = False
        detail.view = None
        self._paint()

    # -- painting ------------------------------------------------------------
    def _tick(self) -> None:
        if self.state is None:
            return
        self.state.refresh()
        if self.state.manifest is not None and not self._announced:
            self._announced = True
            self.post_message(self.ManifestLoaded())
        self._paint()
        scaling = self.query_one(ScalingView)
        if scaling.display:
            scaling.update_from(self.state)
        detail = self.query_one(CellDetail)
        if detail.display and detail.view is not None:
            detail.refresh_view()
            detail.tick()

    def _paint(self) -> None:
        state = self.state
        if state is None:
            return
        table = self.query_one("#watch-table", DataTable)
        for index, key in enumerate(state.order, 1):
            view = state.views[key]
            snap = _snapshot(view)
            if key not in self._known:
                table.add_row(*_row_cells(index, view), key=key)
                self._known.add(key)
                self._snapshots[key] = snap
            elif self._snapshots.get(key) != snap:
                cells = _row_cells(index, view)
                for (name, _w), value in zip(_COLUMNS, cells):
                    table.update_cell(key, name, value)
                self._snapshots[key] = snap
        self._summary()

    def _summary(self) -> None:
        state = self.state
        if state is None:
            return
        counts = state.counts()
        text = Text()
        text.append(state.run_dir.name, style="bold")
        text.append(f" · {state.launcher or '?'}", style="dim")
        for status in (Status.PENDING, Status.SUBMITTED, Status.RUNNING,
                       Status.ENDING, Status.OK, Status.FAIL, Status.TIMEOUT,
                       Status.SKIPPED):
            n = counts.get(status.value, 0)
            if not n:
                continue
            label, style = _STATUS_STYLE[status]
            text.append(f"   {n} ")
            text.append(label, style=style)
        broken = state.broken_groups
        if broken:
            text.append(f"   consensus: {len(broken)} group(s) broken",
                        style="bold red")
            worst = ", ".join(f"{g.app_key}@{g.nodes}n" for g in broken[:3])
            more = len(broken) - 3
            text.append(f" ({worst}{f', +{more}' if more > 0 else ''})",
                        style="red")
        elif state.groups:
            text.append("   consensus: clean", style="green")
        if state.finished:
            text.append("   — finished", style="bold")
        if not self.query_one(CellDetail).display and state.order:
            text.append("   · click a row: command, configs, live output",
                        style="dim")
        self.query_one("#watch-summary", Static).update(text)

    # -- interaction ---------------------------------------------------------
    # The table consumes clicks whole: the first click on a row only moves
    # the cursor (a highlight), and a selection is posted only for a click on
    # the row the cursor is already on.  A single click has to open the cell,
    # so the mouse press is remembered here — it does bubble — and the
    # highlight it causes is treated as the selection it was meant as.
    def on_mouse_up(self, event: events.MouseUp) -> None:
        self._clicked_at = time.monotonic()

    def _open(self, key: str) -> None:
        view = self.state.views.get(key) if self.state else None
        if view is None:
            return
        detail = self.query_one(CellDetail)
        detail.display = True
        detail.show(view, self.state.counters_cfg)

    def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        if self.state is None or event.row_key is None:
            return
        key = event.row_key.value
        detail = self.query_one(CellDetail)
        if detail.display:
            # Open pane follows the cursor, keys and mouse alike.
            if detail.key != key:
                self._open(key)
        elif time.monotonic() - self._clicked_at < 0.5:
            self._open(key)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if self.state is None or event.row_key is None:
            return
        key = event.row_key.value
        detail = self.query_one(CellDetail)
        if detail.display and detail.key == key:
            detail.display = False
            return
        self._open(key)

    def action_close_detail(self) -> None:
        if self.query_one(ScalingView).display:
            self.action_toggle_scaling()
            return
        self.query_one(CellDetail).display = False

    @on(Button.Pressed, "#watch-scaling-btn")
    def _scaling_pressed(self) -> None:
        self.action_toggle_scaling()

    def action_toggle_scaling(self) -> None:
        """Swap the cell table for the strong-scaling reading and back.

        Focus moves with the swap: the hidden table cannot keep it, and the
        bindings that fold the view back live on this widget's subtree.
        """
        scaling = self.query_one(ScalingView)
        table = self.query_one("#watch-table", DataTable)
        button = self.query_one("#watch-scaling-btn", Button)
        if scaling.display:
            scaling.display = False
            table.display = True
            button.label = "scaling"
            table.focus()
            return
        if self.state is None:
            return
        self.query_one(CellDetail).display = False
        scaling.update_from(self.state)
        table.display = False
        scaling.display = True
        button.label = "table"
        self.query_one("#scaling-table", DataTable).focus()


class WatchApp(App):
    """A live view over one run directory, optionally driving the campaign.

    With only a run directory it is a pure watcher — of a detached campaign,
    another process's, or a finished one.  Handed a campaign as well, it runs
    it on a plain daemon thread while the view reads the same disk everybody
    else would; closing the view then merely detaches it, because the thread
    belongs to the process, not to the screen.
    """

    TITLE = "artsrun"

    BINDINGS = [
        ("q", "quit", "quit"),
        ("d", "detach", "detach view"),
        ("s", "stop", "stop campaign"),
        ("l", "toggle_log", "campaign log"),
    ]

    DEFAULT_CSS = """
    WatchApp #campaign-log {
        height: 8;
        border-top: solid $panel;
        display: none;
    }
    """

    def __init__(self, run_dir: Path, campaign=None, runner=None):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.campaign = campaign
        self.runner = runner
        self.sub_title = str(self.run_dir.name)
        self.outcome: dict | None = None
        self.error: Exception | None = None
        # Where campaign lines go once the screen is gone; the caller that
        # keeps waiting for the thread points this at its own console.
        self.line_sink = None
        self._thread: threading.Thread | None = None
        self._log_pinned = False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def compose(self) -> ComposeResult:
        yield Header()
        yield RunView(id="watch-view")
        yield RichLog(id="campaign-log", wrap=False, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(RunView).attach(self.run_dir)
        if self.runner is not None:
            self.query_one("#campaign-log", RichLog).display = True
            self._thread = threading.Thread(target=self._drive, daemon=True)
            self._thread.start()

    def on_run_view_manifest_loaded(self) -> None:
        """Build output mattered while there was nothing else to watch; once
        the table takes over the log folds away (l brings it back)."""
        if not self._log_pinned:
            self.query_one("#campaign-log", RichLog).display = False

    def _log(self, line: str) -> None:
        sink = self.line_sink
        if sink is not None:
            sink(line)
            return
        write = self.query_one("#campaign-log", RichLog).write
        try:
            self.call_from_thread(write, line)
        except RuntimeError:
            write(line)

    def _drive(self) -> None:
        try:
            self.outcome = self.runner(self._log)
        except Exception as exc:
            self.error = exc
            try:
                # A folded log must not swallow the one thing that matters.
                self.call_from_thread(
                    lambda: setattr(
                        self.query_one("#campaign-log", RichLog),
                        "display", True,
                    )
                )
            except RuntimeError:
                pass
            self._log("campaign failed:")
            for line in str(exc).splitlines():
                self._log(f"  {line}")

    def action_quit(self) -> None:
        if self.running:
            self.notify(
                "campaign is running — s stops it, d detaches the view "
                "(reattach later with artsrun watch)",
                severity="warning",
            )
            return
        self.exit()

    def action_detach(self) -> None:
        self.exit()

    def action_stop(self) -> None:
        if self.campaign is None:
            self.notify("view only — nothing to stop", severity="warning")
            return
        self._log("stopping — the cell in progress is being ended")
        self.campaign.request_stop()

    def action_toggle_log(self) -> None:
        log = self.query_one("#campaign-log", RichLog)
        log.display = not log.display
        self._log_pinned = log.display
