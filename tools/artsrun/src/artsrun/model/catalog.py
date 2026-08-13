"""The application catalog: what each benchmark structurally is.

Structural facts (binary name, completion marker, which versions exist, which
node counts it can run) ship with the tool.  Everything a campaign changes —
which apps run, with which arguments — lives in a benchset instead.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, Field, model_validator

from artsrun.data import load_data
from artsrun.paths import repo_root


class AppClass(StrEnum):
    TASK = "task"
    SPMD = "spmd"
    BW = "bw"


class Kind(StrEnum):
    """What the program is, as opposed to whether a campaign runs it.

    The test is whether the thing is cited by name: an application is one a
    field refers to as such — graph500, HPCG, CoMD, XSBench, STREAM — so a
    result about it means something to a reader who never saw this catalog.
    A microbenchmark is named after the operation it performs (a global sum, a
    reduction driver, a data-block create loop) or after nothing at all; it
    exercises one runtime mechanism and belongs in a regression suite rather
    than in a comparison.

    A variant of a cited application stays an application even when it
    duplicates a sibling — that is a reason to leave it off a campaign, which
    `default_enabled` says, not a reason to reclassify it.
    """

    APPLICATION = "application"
    MICROBENCH = "microbench"


class Version(StrEnum):
    """One program's answers to the same problem, in increasing order of how
    much of it was rewritten.

    ASBORN is the application as published — including whatever hints its
    authors already gave it, which is why "hinted" was the wrong name for the
    next step.  OPTIMIZED changes no code structure: it only adds or changes
    placement hints on EDTs and DBs, statically optimized as far as hints
    alone can carry the program.  RESTRUCTURED redesigns the decomposition
    itself, as a separate target shown in the row of the application it
    re-implements.
    """

    ASBORN = "asborn"
    OPTIMIZED = "optimized"
    RESTRUCTURED = "restructured"

    @classmethod
    def _missing_(cls, value):
        # The optimized version was recorded as "hinted" before the rename;
        # selections and benchsets written under that name still resolve.
        if value == "hinted":
            return cls.OPTIMIZED
        return None


class ScalarKind(StrEnum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"


class AppEntry(BaseModel):
    name: str
    binary: str
    cls: AppClass = Field(alias="class")
    kind: Kind = Kind.APPLICATION
    # Where the program came from, in one line: the suite that cites it,
    # or what it was written to exercise.
    provenance: str | None = None

    # `marker` proves the run reached its result; `scalar_re` reads that
    # result.  They are usually the same line, and default to each other.
    marker: str
    scalar_re: str | None = None
    scalar_kind: ScalarKind = ScalarKind.FLOAT
    # A pinned answer is only an answer for the workload it was derived from,
    # so it is carried with those arguments and applied only when the campaign
    # runs exactly them.
    expect: str | None = None
    expect_args: list[str] = Field(default_factory=list)
    tolerance: float = 0.0
    extra_scalars: dict[str, str] = Field(default_factory=dict)

    args: list[str] = Field(default_factory=list)
    args_by_nodes: dict[int, list[str]] = Field(default_factory=dict)

    optimized: bool = False
    restructured_as: str | None = None
    restructured_from: str | None = None

    default_enabled: bool = True
    # An application whose correctness depends on semantics the runtime
    # deliberately does not implement.  It stays a visible row — the doc and
    # the reason are the point — but it cannot be selected or run.
    unsupported: str | None = None
    multinode_skip: str | None = None
    ocrvx_skip: bool = False
    fixtures: list[str] = Field(default_factory=list)
    timeout: int = 0
    multinode_timeout: int = 0

    model_config = {"populate_by_name": True}

    @property
    def result_re(self) -> str:
        return self.scalar_re or self.marker

    def timeout_for(self, nodes: int) -> int:
        """Per-cell wall budget, or 0 to use the profile's."""
        if nodes > 1 and self.multinode_timeout:
            return self.multinode_timeout
        return self.timeout

    def args_for(self, nodes: int) -> list[str]:
        return self.args_by_nodes.get(nodes, self.args)

    @property
    def own_versions(self) -> list[Version]:
        """Versions this entry offers as a row of its own.

        A restructured version is a separate application target, but it
        belongs in the row of the application it re-implements: the three
        versions are one program's answers to the same problem, in increasing
        order of how much of it was rewritten.
        """
        v = [Version.ASBORN]
        if self.optimized:
            v.append(Version.OPTIMIZED)
        if self.restructured_as:
            v.append(Version.RESTRUCTURED)
        return v


class Catalog(BaseModel):
    apps: dict[str, AppEntry]

    @model_validator(mode="after")
    def _check_twins(self) -> "Catalog":
        for name, app in self.apps.items():
            if app.restructured_as and app.restructured_as not in self.apps:
                raise ValueError(
                    f"{name}: restructured_as '{app.restructured_as}' not in catalog")
            if app.restructured_from and app.restructured_from not in self.apps:
                raise ValueError(
                    f"{name}: restructured_from '{app.restructured_from}' not in catalog")
        return self

    @property
    def rows(self) -> list[AppEntry]:
        """Apps that head a row: every app that is not another app's rewrite."""
        return [a for a in self.apps.values() if a.restructured_from is None]

    def rows_of(self, kind: Kind) -> list[AppEntry]:
        return sorted((a for a in self.rows if a.kind is kind),
                      key=lambda a: a.name.lower())

    def resolve(self, name: str, version: Version) -> tuple[AppEntry, str]:
        """The catalog entry that supplies the arguments, and the binary stem.

        The `dist` version runs a different target than the row it appears in,
        and that target carries its own calibration.
        """
        app = self.apps[name]
        if version is Version.RESTRUCTURED:
            if not app.restructured_as:
                raise KeyError(f"{name} has no restructured version")
            other = self.apps[app.restructured_as]
            return other, other.binary
        if version is Version.OPTIMIZED:
            if not app.optimized:
                raise KeyError(f"{name} has no optimized version: its source "
                               "carries no hint layer")
            return app, f"{app.binary}_opt"
        return app, app.binary


def _expand(value, root: str):
    """Resolve `{repo}` in argument lists against this checkout.

    An application that reads an input file is given an absolute path, since
    a cell runs from the scratch directory rather than the checkout.  The
    catalog is committed, so it cannot carry one machine's absolute path.
    """
    if isinstance(value, str):
        return value.replace("{repo}", root)
    if isinstance(value, list):
        return [_expand(v, root) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, root) for k, v in value.items()}
    return value


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    raw = load_data("apps.yaml")
    root = str(repo_root())
    apps = {}
    for name, spec in raw["apps"].items():
        spec = dict(spec)
        spec.setdefault("name", name)
        spec.setdefault("binary", name)
        apps[name] = AppEntry.model_validate(_expand(spec, root))
    return Catalog(apps=apps)
