"""A campaign selection: the three surfaces, resolved and serializable.

A saved selection replays a campaign exactly, so it is written next to every
run's results.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from artsrun.model.benchset import Benchset
from artsrun.model.catalog import Catalog, Version
from artsrun.model.plane import Plane, modern_entry_key
from artsrun.model.profile import Launcher, Profile


def _first_sibling_cpu_count() -> int | None:
    """How many per-core first SMT threads this host exposes.

    The envelope confines ranks to per-core first threads, so this — not the
    logical CPU count — is the budget colocated local blocks must fit in.
    None when the topology is unreadable; the envelope wrapper re-checks the
    same property per cell either way.
    """
    root = Path("/sys/devices/system/cpu")
    try:
        cpus = [p for p in root.iterdir() if re.fullmatch(r"cpu\d+", p.name)]
    except OSError:
        return None
    count = 0
    for p in cpus:
        f = p / "topology" / "thread_siblings_list"
        if not f.is_file():
            f = p / "topology" / "core_cpus_list"
        try:
            first = re.split(r"[,-]", f.read_text().strip(), maxsplit=1)[0]
        except OSError:
            continue
        if first == p.name[3:]:
            count += 1
    return count or None


class Selection(BaseModel):
    profile: str
    benchset: str
    entries: list[str] = Field(min_length=1)
    apps: dict[str, list[Version]] = Field(min_length=1)

    @field_validator("entries")
    @classmethod
    def _modern_entries(cls, entries: list[str]) -> list[str]:
        """Replayed selections may carry pre-promotion comb-suffixed keys;
        map them to today's names.  Deduplicated in order, because an old
        selection could name a val entry under both spellings."""
        out: list[str] = []
        for key in (modern_entry_key(k) for k in entries):
            if key not in out:
                out.append(key)
        return out
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
        if any(plane.entry(k).is_reference for k in self.entries):
            self._check_reference_geometry(profile)

    def _check_reference_geometry(self, profile: Profile) -> None:
        """What a reference cell needs from the profile.

        These are selection-time errors rather than render-time ones on
        purpose: the ocr configuration is rendered for every campaign, even
        one that selects no reference, and an arts-only campaign must not be
        refused over a geometry only the references cannot express.
        """
        width = profile.threads_per_node
        if not profile.pin:
            raise ValueError(
                "reference entries require pin: true — an unpinned arts run "
                "floats over the whole machine while the references stay "
                "confined to their envelope, so the two would be measured on "
                "different hardware"
            )
        if width < 2:
            raise ValueError(
                f"reference entries require workers+progress >= 2: at width "
                f"{width} the xsocr configuration degenerates to a "
                f"communication worker with no compute workers"
            )
        if width > 255:
            raise ValueError(
                f"reference entries require workers+progress <= 255: the "
                f"runtime parses its per-core binding policy fields through "
                f"a u8 (width {width} would wrap)"
            )
        if profile.launcher is Launcher.LOCAL:
            nodes = max(self.node_counts)
            if nodes <= 1:
                return
            if nodes > 255:
                raise ValueError(
                    f"colocated reference ranks are capped at 255 ({nodes} "
                    f"requested): the runtime's block policy parses the rank "
                    f"count through a u8"
                )
            if nodes * width > 1024:
                raise ValueError(
                    f"colocated reference blocks reach cpu {nodes * width - 1}, "
                    f"past CPU_SETSIZE (1024): binding there is a silent no-op "
                    f"in glibc"
                )
            cores = _first_sibling_cpu_count()
            if cores is not None and nodes * width > cores:
                raise ValueError(
                    f"colocated reference blocks need {nodes * width} per-core "
                    f"first threads but this host has {cores}"
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
