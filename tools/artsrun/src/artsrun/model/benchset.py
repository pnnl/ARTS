"""A benchmark set: which applications a campaign runs, in which versions,
with which arguments.

The catalog says what an application is; a benchset says what this campaign
does with it.  Anything omitted falls through to the catalog, so a benchset
only has to carry its deltas.
"""

from __future__ import annotations

import sys

from pydantic import BaseModel, Field

from artsrun.model.catalog import (AppClass, AppEntry, Catalog, ScalarKind,
                                   Version, expand_repo)
from artsrun.paths import repo_root


class BenchsetEntry(BaseModel):
    enabled: bool | None = None
    args: list[str] | None = None
    args_by_nodes: dict[int, list[str]] | None = None
    versions: list[Version] | None = None


class ResolvedApp(BaseModel):
    """One selectable (application, version) pair with its arguments settled."""

    name: str
    version: Version
    binary: str
    cls: AppClass
    marker: str
    scalar_re: str
    scalar_kind: ScalarKind = ScalarKind.FLOAT
    expect: str | None = None
    expect_args: list[str] = Field(default_factory=list)
    tolerance: float = 0.0
    extra_scalars: dict[str, str] = Field(default_factory=dict)
    args: list[str] = Field(default_factory=list)
    args_by_nodes: dict[int, list[str]] = Field(default_factory=dict)
    unsupported: str | None = None
    post_verify: str | None = None
    multinode_skip: str | None = None
    ocrvx_skip: bool = False
    fixtures: list[str] = Field(default_factory=list)
    timeout: int = 0
    multinode_timeout: int = 0
    args_overridden: bool = False

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version.value}"

    def args_for(self, nodes: int) -> list[str]:
        return self.args_by_nodes.get(nodes, self.args)

    def timeout_for(self, nodes: int) -> int:
        if nodes > 1 and self.multinode_timeout:
            return self.multinode_timeout
        return self.timeout


class Benchset(BaseModel):
    name: str
    description: str | None = None
    apps: dict[str, BenchsetEntry] = Field(default_factory=dict)

    def is_enabled(self, app: AppEntry) -> bool:
        """Whether this campaign runs the application at all.

        Naming applications defines the roster: a benchset that lists any
        application runs only the ones it lists, so a short file is a short
        campaign rather than the whole catalog with three entries annotated.
        A benchset that lists none defers to the catalog's own defaults.
        """
        if app.unsupported:
            # Not a roster choice: the application needs semantics the
            # runtime does not implement, so no benchset can turn it on.
            return False
        entry = self.apps.get(app.name)
        if entry is not None and entry.enabled is not None:
            return entry.enabled
        if entry is not None:
            return True
        if self.apps:
            return False
        return app.default_enabled

    def versions_for(self, app: AppEntry) -> list[Version]:
        entry = self.apps.get(app.name)
        available = app.own_versions
        if entry is None or entry.versions is None:
            return available
        # A version the application does not offer is dropped and said out
        # loud.  Raising instead would be the wrong trade: a roster records
        # what a campaign wants to measure, and a version can be withdrawn
        # from the catalog while that intent stays correct -- so the roster
        # keeps naming it and picks it up again when the catalog restores it,
        # rather than every roster needing an edit in the meantime.
        unknown = [v for v in entry.versions if v not in available]
        if unknown:
            print(
                f"artsrun: {app.name} has no "
                f"{', '.join(v.value for v in unknown)} version "
                f"(offers {', '.join(v.value for v in available)}) — not run",
                file=sys.stderr,
            )
        return [v for v in entry.versions if v in available]

    def resolve(self, catalog: Catalog) -> list[ResolvedApp]:
        """Expand to the (application, version) pairs this set runs."""
        out: list[ResolvedApp] = []
        for app in catalog.rows:
            if not self.is_enabled(app):
                continue
            for version in self.versions_for(app):
                source, binary = catalog.resolve(app.name, version)
                entry = self.apps.get(app.name)
                args = source.args
                args_by_nodes = source.args_by_nodes
                overridden = False
                # An override applies to the row, so it follows the version
                # into the rewrite only when the rewrite shares the CLI.
                if entry is not None and version is not Version.RESTRUCTURED:
                    # A roster's override goes through the same `{repo}`
                    # resolution the catalog's own arguments get: the catalog
                    # is committed and cannot carry one machine's absolute
                    # path, and neither can a roster.  Without this an
                    # override naming an input file reaches the application
                    # as the literal "{repo}/..." and it fails on open.
                    root = str(repo_root())
                    if entry.args is not None:
                        args, overridden = expand_repo(entry.args, root), True
                    if entry.args_by_nodes is not None:
                        args_by_nodes = expand_repo(entry.args_by_nodes, root)
                        overridden = True
                out.append(
                    ResolvedApp(
                        name=app.name,
                        version=version,
                        binary=binary,
                        cls=source.cls,
                        marker=source.marker,
                        scalar_re=source.result_re,
                        scalar_kind=source.scalar_kind,
                        expect=source.expect,
                        expect_args=source.expect_args,
                        tolerance=source.tolerance,
                        extra_scalars=source.extra_scalars,
                        args=args,
                        args_by_nodes=args_by_nodes,
                        unsupported=app.unsupported,
                        post_verify=source.post_verify,
                        multinode_skip=source.multinode_skip,
                        ocrvx_skip=source.ocrvx_skip,
                        fixtures=source.fixtures,
                        timeout=source.timeout,
                        multinode_timeout=source.multinode_timeout,
                        args_overridden=overridden,
                    )
                )
        return out
