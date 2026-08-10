"""A campaign selection: the three surfaces, resolved and serializable.

A saved selection replays a campaign exactly, so it is written next to every
run's results.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from artsrun.model.benchset import Benchset
from artsrun.model.catalog import Catalog, Version
from artsrun.model.plane import Plane
from artsrun.model.profile import Profile


class Selection(BaseModel):
    profile: str
    benchset: str
    entries: list[str] = Field(min_length=1)
    apps: dict[str, list[Version]] = Field(min_length=1)
    node_counts: list[int] = Field(min_length=1)
    repeats: int = 1
    build_dir: str | None = None

    def validate_against(
        self, plane: Plane, catalog: Catalog, profile: Profile
    ) -> None:
        unknown = [k for k in self.entries if k not in plane.entry_keys]
        if unknown:
            raise ValueError(f"unknown plane entries: {', '.join(unknown)}")
        for name, versions in self.apps.items():
            if name not in catalog.apps:
                raise ValueError(f"unknown application: {name}")
            available = catalog.apps[name].own_versions
            bad = [v for v in versions if v not in available]
            if bad:
                raise ValueError(
                    f"{name}: no {', '.join(v.value for v in bad)} version"
                )
        off_sweep = [n for n in self.node_counts if n not in profile.nodes]
        if off_sweep:
            raise ValueError(
                f"node counts {off_sweep} are not in profile "
                f"'{profile.name}' sweep {profile.nodes}"
            )

    @classmethod
    def everything(
        cls,
        plane: Plane,
        catalog: Catalog,
        benchset: Benchset,
        profile: Profile,
        *,
        build_dir: str | None = None,
    ) -> "Selection":
        """The default state of the selection screens: all of it."""
        apps = {
            app.name: benchset.versions_for(app)
            for app in catalog.rows
            if benchset.is_enabled(app)
        }
        return cls(
            profile=profile.name,
            benchset=benchset.name,
            entries=plane.entry_keys,
            apps=apps,
            node_counts=list(profile.nodes),
            repeats=profile.repeats,
            build_dir=build_dir,
        )

    @property
    def cell_count(self) -> int:
        versions = sum(len(v) for v in self.apps.values())
        return len(self.entries) * versions * len(self.node_counts) * self.repeats
