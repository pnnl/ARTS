"""Shared selection behaviour for the three surfaces."""

from __future__ import annotations

from textual.widgets import Button, Checkbox


class Toggle(Checkbox):
    """A checkbox that remembers what it stands for.

    The mark is a check rather than the default cross: on a grid whose empty
    positions already mean "not available", a cross reads as exclusion instead
    of selection.
    """

    BUTTON_INNER = "✓"

    def __init__(self, label: str, ident: str, *, value: bool = True, **kwargs):
        super().__init__(label, value=value, **kwargs)
        self.ident = ident


def toggle_all(boxes: list[Checkbox]) -> bool:
    """One control for both directions: anything on clears, nothing on fills.

    Returns the state everything was set to.
    """
    target = not any(b.value for b in boxes)
    for box in boxes:
        box.value = target
    return target


class NodeDelete(Button):
    """Removes one node count from the list.

    It carries the value rather than encoding it in an id: the list is rebuilt
    in place, and a recycled id collides with the widget still being removed.
    """

    def __init__(self, node: int, **kwargs):
        super().__init__("✕", classes="node-del", **kwargs)
        self.node = node
