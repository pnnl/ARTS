"""The coherence-configuration plane and the entries selectable on it."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, Field

from artsrun.data import load_data


class Family(StrEnum):
    EXCL = "EXCL"
    INV = "INV"
    VAL = "VAL"
    VAL_COMB = "VAL_COMB"


class Release(StrEnum):
    PURGE = "PURGE"
    RETAIN = "RETAIN"


class Write(StrEnum):
    WT = "WT"
    WB = "WB"


class RuntimeKind(StrEnum):
    ARTS = "arts"
    XSOCR = "xsocr"
    OCRVX = "ocrvx"


def _arts_key(variant: str) -> str:
    """The name an ARTS configuration is selected and reported by.

    Its build suffix leads with the memory model (`ocr_`), which the plane
    holds fixed, so carrying that into the name distinguishes nothing and
    misreads as the reference runtime of the same name.
    """
    return "arts_" + variant.removeprefix("ocr_")


class SelectionEntry(BaseModel):
    """One selectable runtime configuration.

    `key` is the stable identifier used on the command line, in saved
    selections and in result rows: an ARTS entry is named by the
    configuration it selects, a reference entry by the runtime.
    """

    key: str
    label: str
    kind: RuntimeKind
    cell: str
    variant: str | None = None

    @property
    def is_reference(self) -> bool:
        return self.kind is not RuntimeKind.ARTS

    def binary(self, app_binary: str, *, hinted: bool) -> str:
        """Executable name this entry runs a given application under."""
        stem = f"{app_binary}_opt" if hinted else app_binary
        if self.kind is RuntimeKind.ARTS:
            return f"{stem}_arts_{self.variant}"
        return f"{stem}_{self.kind.value}"


class PlaneCell(BaseModel):
    family: Family
    release: Release
    write: Write
    variant: str | None = None
    reference: str | None = None
    reason: str | None = None

    @property
    def key(self) -> str:
        return f"{self.family}/{self.release}/{self.write}"

    @property
    def buildable(self) -> bool:
        return self.variant is not None


class Plane(BaseModel):
    families: list[Family]
    releases: list[Release]
    writes: list[Write]
    family_labels: dict[Family, str] = Field(default_factory=dict)
    write_labels: dict[Write, str] = Field(default_factory=dict)
    release_labels: dict[Release, str] = Field(default_factory=dict)
    cells: list[PlaneCell]
    entries: list[SelectionEntry]

    # -- lookup ----------------------------------------------------------
    def cell(self, family: Family, release: Release, write: Write) -> PlaneCell:
        for c in self.cells:
            if c.family is family and c.release is release and c.write is write:
                return c
        raise KeyError(f"{family}/{release}/{write}")

    def entry(self, key: str) -> SelectionEntry:
        for e in self.entries:
            if e.key == key:
                return e
        raise KeyError(key)

    def entries_of(self, cell: PlaneCell) -> list[SelectionEntry]:
        return [e for e in self.entries if e.cell == cell.key]

    @property
    def entry_keys(self) -> list[str]:
        return [e.key for e in self.entries]

    def columns(self) -> list[tuple[Release, Write]]:
        """Column order: the write policy groups, the release policy divides.

        Writing is the coarser decision — it says where the bytes live — so it
        forms the outer pair and the release policy splits each half.
        """
        return [(r, w) for w in self.writes for r in self.releases]

    def write_label(self, write: Write) -> str:
        return self.write_labels.get(write, write.value)

    def release_label(self, release: Release) -> str:
        return self.release_labels.get(release, release.value)


def _reason(family: Family, release: Release, write: Write, texts: dict) -> str:
    """Why this grid position carries no configuration.

    The rules mirror the build's own refusals, so a disabled position always
    reports the same ground the build would.
    """
    if family is Family.EXCL and write is Write.WT:
        return texts["excl_needs_wb"]
    return texts["needs_retain"]


@lru_cache(maxsize=1)
def load_plane() -> Plane:
    raw = load_data("protocols.yaml")
    families = [Family(f) for f in raw["families"]]
    releases = [Release(r) for r in raw["releases"]]
    writes = [Write(w) for w in raw["writes"]]

    defined = {
        (Family(c["family"]), Release(c["release"]), Write(c["write"])): c
        for c in raw["cells"]
    }
    refs = raw.get("references", {})

    cells: list[PlaneCell] = []
    entries: list[SelectionEntry] = []
    for family in families:
        for release, write in [(r, w) for r in releases for w in writes]:
            spec = defined.get((family, release, write))
            if spec is None:
                cells.append(
                    PlaneCell(
                        family=family,
                        release=release,
                        write=write,
                        reason=_reason(family, release, write, raw["unbuildable"]),
                    )
                )
                continue
            cell = PlaneCell(
                family=family,
                release=release,
                write=write,
                variant=spec["variant"],
                reference=spec.get("reference"),
            )
            cells.append(cell)
            entries.append(
                SelectionEntry(
                    # The build suffix leads with the memory model, which this
                    # plane fixes, so as a name it says nothing and reads as
                    # the wrong runtime beside the two references.  The entry
                    # names the runtime it selects; the suffix stays the
                    # build's own.
                    key=_arts_key(cell.variant),
                    label=_arts_key(cell.variant),
                    kind=RuntimeKind.ARTS,
                    cell=cell.key,
                    variant=cell.variant,
                )
            )
            if cell.reference:
                ref = refs[cell.reference]
                entries.append(
                    SelectionEntry(
                        key=cell.reference,
                        label=ref["label"],
                        kind=RuntimeKind(ref["kind"]),
                        cell=cell.key,
                    )
                )

    return Plane(
        families=families,
        releases=releases,
        writes=writes,
        family_labels={Family(k): v for k, v in raw.get("family_labels", {}).items()},
        write_labels={Write(k): v for k, v in raw.get("write_labels", {}).items()},
        release_labels={Release(k): v
                        for k, v in raw.get("release_labels", {}).items()},
        cells=cells,
        entries=entries,
    )
