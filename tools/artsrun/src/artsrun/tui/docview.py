"""The application document screen.

One application's structural story — its parameters, how they size the task
graph, how the graph is wired, and how to pick arguments for a machine — shown
over the roster without leaving it.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static

from artsrun.docs import document
from artsrun.model.catalog import AppEntry


class AppDocScreen(ModalScreen):
    """The document over the screen that named it; escape puts it back."""

    BINDINGS = [
        ("escape", "app.pop_screen", "close"),
        ("q", "app.pop_screen", "close"),
    ]

    def __init__(self, entry: AppEntry, **kwargs):
        super().__init__(**kwargs)
        self.entry = entry

    def compose(self) -> ComposeResult:
        text = document(self.entry)
        with Vertical(id="doc-box"):
            yield Static(
                f"[b]{self.entry.name}[/b]  "
                f"[dim]esc closes · arrows and page keys scroll[/dim]",
                id="doc-head",
            )
            with VerticalScroll(id="doc-body"):
                yield Markdown(text)

    def on_mount(self) -> None:
        self.query_one("#doc-body").focus()
