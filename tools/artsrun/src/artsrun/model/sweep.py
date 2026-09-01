"""A sweep: one attack probe, many argument points, at one node count.

A campaign compares applications across the plane at their calibrated
arguments; a sweep holds ONE probe and moves its knobs across a matrix.  The
spec is data (experiments/sweeps/*.yaml, untracked like profiles), and every
point expands to a virtual application row so the ordinary cell machinery —
track file, consensus, resume, detach — carries the sweep unchanged.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator

from artsrun.model.catalog import Catalog, Kind, Version
from artsrun.model.benchset import ResolvedApp
from artsrun.model.plane import (RuntimeKind, SelectionEntry, load_plane,
                                 modern_entry_key)

_POINT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class SweepPoint(BaseModel):
    """One argument point: a name (rides in the cell identity) and the knobs
    it changes from the sweep's base."""

    name: str
    set: dict[str, int | float | str] = Field(default_factory=dict)
    # Which figure cut this point belongs to (purely a grouping label the
    # figure tool reads; the runner ignores it).
    group: str = ""
    # Node-count override: the one workload dimension that is not an argv
    # knob.  A node-scaling cut varies it per point; everything else in the
    # sweep (arms, interleave, report) is node-count-agnostic.
    nodes: int | None = None


class SweepSpec(BaseModel):
    name: str
    description: str | None = None
    app: str
    nodes: int
    arms: list[str] = Field(min_length=1)
    repeats: int = 5
    arg_order: list[str] = Field(min_length=1)
    base: dict[str, int | float | str] = Field(default_factory=dict)
    points: list[SweepPoint] = Field(min_length=1)
    # 0 = the catalog's / profile's budget decides.
    timeout: int = 0

    @model_validator(mode="after")
    def _check(self) -> "SweepSpec":
        seen: set[str] = set()
        for p in self.points:
            if not _POINT_NAME.match(p.name):
                raise ValueError(f"point name {p.name!r} is not slug-safe")
            if p.name in seen:
                raise ValueError(f"duplicate point name {p.name!r}")
            seen.add(p.name)
            unknown = [k for k in p.set if k not in self.arg_order]
            if unknown:
                raise ValueError(
                    f"point {p.name}: knobs {unknown} not in arg_order")
        unknown = [k for k in self.base if k not in self.arg_order]
        if unknown:
            raise ValueError(f"base knobs {unknown} not in arg_order")
        return self

    # -- arms -------------------------------------------------------------
    @staticmethod
    def arm_suffix(arm: str) -> str:
        """The build suffix an arm name denotes.

        Arms are addressed by selection-entry key (arts_val_wb) or directly
        by build suffix (ocr_val_wb_nocomb) — the latter is what lets a
        sweep run the off-plane ablation twins the plane deliberately does
        not offer.
        """
        arm = modern_entry_key(arm)
        if arm.startswith("arts_"):
            return "ocr_" + arm.removeprefix("arts_")
        return arm

    @staticmethod
    def arm_entry(arm: str) -> SelectionEntry:
        """A SelectionEntry for the arm — the plane's own where it has one,
        a synthesized twin (cell "ablation") where it does not."""
        key = modern_entry_key(arm)
        if key.startswith("ocr_"):
            key = "arts_" + key.removeprefix("ocr_")
        plane = load_plane()
        try:
            return plane.entry(key)
        except KeyError:
            return SelectionEntry(
                key=key, label=key, kind=RuntimeKind.ARTS, cell="ablation",
                variant=SweepSpec.arm_suffix(arm))

    # -- points -----------------------------------------------------------
    def argv_for(self, point: SweepPoint) -> list[str]:
        knobs = dict(self.base)
        knobs.update(point.set)
        missing = [k for k in self.arg_order if k not in knobs]
        if missing:
            raise ValueError(f"point {point.name}: no value for {missing}")
        return [str(knobs[k]) for k in self.arg_order]

    def knobs_for(self, point: SweepPoint) -> dict[str, int | float | str]:
        knobs = dict(self.base)
        knobs.update(point.set)
        return knobs

    def resolved_app(self, catalog: Catalog, point: SweepPoint) -> ResolvedApp:
        row = catalog.apps[self.app]
        return ResolvedApp(
            name=f"{self.app}+{point.name}",
            version=Version.BASE,
            binary=row.binary,
            cls=row.cls,
            marker=row.marker,
            scalar_re=row.result_re,
            scalar_kind=row.scalar_kind,
            tolerance=row.tolerance,
            extra_scalars=row.extra_scalars,
            args=self.argv_for(point),
            timeout=self.timeout or row.timeout,
            multinode_timeout=self.timeout or row.multinode_timeout,
            fixtures=row.fixtures,
            args_overridden=True,
        )

    def validate_against(self, catalog: Catalog, profile) -> None:
        if self.app not in catalog.apps:
            raise ValueError(f"unknown probe: {self.app}")
        row = catalog.apps[self.app]
        if row.kind is Kind.TOY:
            raise ValueError(
                f"{self.app} is a toy, not an attack probe or application — "
                f"a sweep runs characterization probes and applications whose "
                f"campaign varies arguments with geometry (weak scaling)")
        for n in {self.nodes, *(p.nodes for p in self.points if p.nodes)}:
            if n not in profile.nodes:
                raise ValueError(
                    f"node count {n} is not in profile "
                    f"'{profile.name}' sweep {profile.nodes}")
        for arm in self.arms:
            entry = self.arm_entry(arm)
            if entry.kind is not RuntimeKind.ARTS:
                raise ValueError(f"arm {arm}: sweeps run ARTS builds only")
