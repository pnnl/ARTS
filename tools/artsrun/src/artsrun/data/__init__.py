"""Packaged catalog data (values that ship with the tool)."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from ruamel.yaml import YAML

_yaml = YAML(typ="safe")


def load_data(name: str) -> Any:
    """Load a packaged YAML catalog by file name."""
    with files(__package__).joinpath(name).open("r", encoding="utf-8") as fh:
        return _yaml.load(fh)
