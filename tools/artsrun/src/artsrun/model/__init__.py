"""Typed models for the three selection surfaces and the campaign they form."""

from artsrun.model.benchset import Benchset, BenchsetEntry, ResolvedApp
from artsrun.model.catalog import (
    AppClass, AppEntry, Catalog, Kind, Version, load_catalog,
)
from artsrun.model.plane import (
    Family,
    Plane,
    PlaneCell,
    Release,
    RuntimeKind,
    SelectionEntry,
    Write,
    load_plane,
)
from artsrun.model.profile import FluxSettings, Launcher, Profile, SlurmSettings
from artsrun.model.selection import Selection

__all__ = [
    "AppClass",
    "AppEntry",
    "Benchset",
    "BenchsetEntry",
    "Catalog",
    "Family",
    "FluxSettings",
    "Kind",
    "Launcher",
    "Plane",
    "PlaneCell",
    "Profile",
    "Release",
    "ResolvedApp",
    "RuntimeKind",
    "Selection",
    "SelectionEntry",
    "SlurmSettings",
    "Version",
    "Write",
    "load_catalog",
    "load_plane",
]
