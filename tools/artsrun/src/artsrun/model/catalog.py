"""The application catalog: what each benchmark structurally is.

Structural facts (binary name, completion marker, which versions exist, which
node counts it can run) ship with the tool.  Everything a campaign changes —
which apps run, with which arguments — lives in a benchset instead.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import AliasChoices, BaseModel, Field, model_validator

from artsrun.data import load_data
from artsrun.paths import repo_root


class AppClass(StrEnum):
    TASK = "task"
    SPMD = "spmd"
    MW = "mw"


class Kind(StrEnum):
    """What the program is, as opposed to whether a campaign runs it.

    APP: cited by name — graph500, HPCG, CoMD, XSBench, STREAM — so a result
    about it means something to a reader who never saw this catalog.  A
    variant of a cited application stays an app even when it duplicates a
    sibling — that is a reason to leave it off a campaign, which
    `default_enabled` says, not a reason to reclassify it.

    TOY: a fixture, library driver, or idiom study — named after the
    operation it performs (a global sum, a reduction driver, a data-block
    create loop) or after nothing at all; it exercises one runtime mechanism
    and belongs in a regression suite rather than in any measurement.

    ATTACK: an adversarial characterization probe, written FOR measurement —
    its knobs move a designed workload across the coherence design space
    (read/write mix, sharers, handoff shapes) to force each arm's best and
    worst case by construction.  Toys check that a mechanism works; attacks
    measure what a mechanism costs when the workload is built against it.
    """

    APP = "app"
    TOY = "toy"
    ATTACK = "attack"

    @classmethod
    def _missing_(cls, value):
        # Pre-split spelling: "application" named today's APP.  The probe
        # group was spelled "microbench" before its rename to ATTACK; the
        # still-older use of that word for today's TOY group predates the
        # split and is not what the alias restores.
        if value == "application":
            return cls.APP
        if value == "microbench":
            return cls.ATTACK
        return None


class Version(StrEnum):
    """Which build of an application a cell runs.

    BASE is the application as published, plus the disclosed
    structure-preserving conformance adaptations every version and runtime
    shares — including whatever hints its authors shipped.  HINTED changes
    no code structure: it only adds or changes placement hints on EDTs and
    DBs, as far as hints alone can carry the program.  RESTRUCTURED
    redesigns the task/data decomposition and is registered as its own
    target.
    """

    BASE = "base"
    HINTED = "hinted"
    RESTRUCTURED = "restructured"

    @classmethod
    def _missing_(cls, value):
        # The tiers were recorded as "asborn"/"optimized" before the rename;
        # accept the old names so recorded selections stay replayable.
        if value == "asborn":
            return cls.BASE
        if value == "optimized":
            return cls.HINTED
        return None

class ScalarKind(StrEnum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"


class AppEntry(BaseModel):
    name: str
    binary: str
    cls: AppClass = Field(alias="class")
    kind: Kind = Kind.APP
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

    # "optimized" is the field's pre-rename spelling, accepted on input
    # so entries renamed in later bundles still parse meanwhile.
    hinted: bool = Field(default=False, validation_alias=AliasChoices("hinted", "optimized"))
    restructured_as: str | None = None
    restructured_from: str | None = None

    # The version tiers an HPX port mirrors, one target per tier named from
    # the row's binary — absent means no port.  A port is one source under a
    # placement guard, exactly as the OCR row builds `<app>` and
    # `<app>_hinted`; a rewrite has no mirror because its base tier IS a
    # shared mutable object no coherence-free model can express.
    hpx: list[Version] = Field(default_factory=list)
    # Removed field, declared only so a leftover value fails loudly.
    hpx_tier: str | None = None

    # Optional post-run verifier: a shell command run in the cell's working
    # directory after the binary exits 0 (chained with &&, so its exit status
    # fails the cell through the ordinary rc plumbing).  {repo} expands like
    # every other catalog path.  Declarative on purpose: each application
    # registers a command, not code.
    post_verify: str | None = None

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
        v = [Version.BASE]
        if self.hinted:
            v.append(Version.HINTED)
        if self.restructured_as:
            v.append(Version.RESTRUCTURED)
        return v

    def hpx_target(self, version: Version) -> str | None:
        """The port's build target for a tier, or None where it mirrors none.

        The runtime tag is the last token, as for every reference binary
        (`<stem>_xsocr`, `<stem>_ocrvx`), and the tier marker sits on the
        application stem.
        """
        if version not in self.hpx:
            return None
        stem = f"{self.binary}_hinted" if version is Version.HINTED else self.binary
        return f"{stem}_hpx"

    @model_validator(mode="after")
    def _check_hpx(self) -> "AppEntry":
        if self.hpx_tier is not None:
            raise ValueError(
                f"{self.name}: `hpx_tier` was replaced by `hpx: [<version>, ...]`"
                " — a port mirrors a list of tiers, one target each; note that"
                " `<binary>_hpx` now names the BASE tier and the hinted port is"
                " `<binary>_hinted_hpx`")
        if Version.RESTRUCTURED in self.hpx:
            raise ValueError(
                f"{self.name}: hpx names restructured — a rewrite has no HPX"
                " mirror (its base tier is a shared mutable object)")
        bad = [v for v in self.hpx if v not in self.own_versions]
        if bad:
            raise ValueError(
                f"{self.name}: hpx names {', '.join(v.value for v in bad)}, "
                f"which this row does not offer "
                f"({', '.join(v.value for v in self.own_versions)})")
        return self


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
        if version is Version.HINTED:
            if not app.hinted:
                raise KeyError(f"{name} has no hinted version: its source "
                               "carries no hint layer")
            return app, f"{app.binary}_hinted"
        return app, app.binary


def expand_repo(value, root: str):
    """Resolve `{repo}` in argument lists against this checkout.

    An application that reads an input file is given an absolute path, since
    a cell runs from the scratch directory rather than the checkout.  The
    catalog is committed, so it cannot carry one machine's absolute path.
    """
    if isinstance(value, str):
        return value.replace("{repo}", root)
    if isinstance(value, list):
        return [expand_repo(v, root) for v in value]
    if isinstance(value, dict):
        return {k: expand_repo(v, root) for k, v in value.items()}
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
        apps[name] = AppEntry.model_validate(expand_repo(spec, root))
    return Catalog(apps=apps)
