"""Read each cell's result scalar and vote a consensus across configurations.

Three rules make the verdict trustworthy. A run whose completion marker never
appeared fails regardless of its exit status, because a wedged run that printed
part of an answer must not pass. A process reaped by timeout is judged on what
its log already carries, not its exit code alone: one whose log shows both the
completion marker and the runtime's own end-to-end stamp measured itself
before the reap, so it is OK with a note rather than a failure — a hung
teardown after a finished, measured run is not the same defect as a run that
never finished; a reap missing either one stays a failure. And the only cells
excluded from a vote are structurally ineligible ones — never a cell that
merely disagreed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum

from artsrun.model.catalog import ScalarKind
from artsrun.run.types import CellResult, Status


class Verdict(StrEnum):
    OK = "OK"
    DISAGREE = "DISAGREE"
    FAIL = "FAIL"
    NA = "N/A"
    EXPECT_FAIL = "EXPECT-FAIL"


def extract(text: str, marker: str, scalar_re: str) -> tuple[bool, str | None]:
    """(completed, scalar).

    A regex with a capture group yields that group; one without is a pure
    completion assertion whose "value" is the fact that it matched.
    """
    completed = re.search(marker, text) is not None
    if not completed:
        return False, None
    m = re.search(scalar_re, text)
    if m is None:
        return True, None
    if m.groups():
        return True, m.group(1)
    return True, "MATCHED"


_E2E_RE = re.compile(r"^\[E2E\]\s+(\d+)\s*$", re.M)

# One line per violated check, printed by the CPU-envelope wrapper every
# reference rank runs under.  The line, not the exit code, is the signal: a
# partial rank failure can surface to the launcher as a timeout (the
# surviving ranks block in MPI until the budget fires), so the verdict must
# not depend on what the process tree happened to exit with.
_ENVELOPE_RE = re.compile(r"^ARTSRUN-ENVELOPE-FAIL: (.*)$", re.M)


def extract_e2e(text: str) -> float | None:
    """Seconds spanned by the runtime's own end-to-end marker, if printed.

    The marker is rank 0's span from application start to shutdown
    recognition, so it excludes runtime init and teardown by construction.
    The last match wins: anything an application itself echoes earlier
    cannot shadow the runtime's stamp at exit.
    """
    matches = _E2E_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1]) / 1e9


def extract_extra(text: str, patterns: dict[str, str]) -> dict[str, str]:
    out = {}
    for name, pattern in patterns.items():
        m = re.search(pattern, text)
        if m and m.groups():
            out[name] = m.group(1)
    return out


def apply_to(result: CellResult) -> CellResult:
    """Fill in a finished cell's scalar, and settle a run whose exit status
    alone does not say whether it measured anything."""
    if result.log_path is None or not result.log_path.is_file():
        if result.status is Status.OK:
            result.status = Status.FAIL
            result.note = result.note or "no log"
        return result
    text = result.log_path.read_text(errors="replace")
    completed, scalar = extract(text, result.cell.app.marker, result.cell.app.scalar_re)
    result.scalar = scalar
    result.e2e_s = extract_e2e(text)
    result.extra.update(extract_extra(text, result.cell.app.extra_scalars))
    # Both demotions run BEFORE the timeout-leniency promotion below: a cell
    # whose rank 0 printed marker and stamp before another rank's envelope
    # died must not be promoted to OK by that branch.
    envelope = _ENVELOPE_RE.search(text)
    if envelope:
        result.status = Status.FAIL
        result.note = f"envelope: {envelope.group(1)}"
        return result
    stamps = len(_E2E_RE.findall(text))
    if stamps > 1:
        # Exactly one rank (rank 0) prints the stamp.  N stamps means N
        # independent worlds: a launch whose process manager never formed
        # the MPI world degrades every rank to a size-1 singleton that
        # solves the whole problem alone and "succeeds" — with the correct
        # scalar, so not even the consensus vote can catch it.
        result.status = Status.FAIL
        result.note = (f"world-size mismatch: {stamps} [E2E] stamps against "
                       f"a rank-0-only contract (PMI missing / singleton "
                       f"MPI init suspected)")
        return result
    if result.status is Status.OK and not completed:
        result.status = Status.FAIL
        result.note = "exited cleanly but never printed its completion marker"
    elif result.status is Status.TIMEOUT and completed and result.e2e_s is not None:
        # The process was reaped by timeout, but its log already carries both
        # the completion marker and the runtime's own end-to-end stamp — the
        # application finished and the run measured itself; only the
        # teardown afterward hung.  A reap missing either one stays a
        # timeout: nothing says the run measured itself completely.
        result.status = Status.OK
        result.teardown_hang = True
        result.note = "reaped by timeout after a completed, measured run (teardown hang)"
    return result


def _as_number(value: str, kind: ScalarKind) -> float | None:
    if kind is ScalarKind.BOOL:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def close(a: str, b: str, kind: ScalarKind, tol: float) -> bool:
    x, y = _as_number(a, kind), _as_number(b, kind)
    if x is None or y is None:
        return a == b
    if tol <= 0:
        # Floating-point results still need a floor: identical computations
        # reorder across configurations.
        tol = 1e-9 if kind is ScalarKind.INT else 1e-4
    return math.isclose(x, y, rel_tol=tol, abs_tol=tol)


@dataclass
class Group:
    """All results for one (application, version, node count)."""

    app_key: str
    nodes: int
    results: list[CellResult] = field(default_factory=list)
    consensus: str | None = None
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    # Keyed the same as verdicts: whether the entry's voting cell was a
    # teardown-hang leniency case, for a table to tag rather than hide.
    teardown_hang: dict[str, bool] = field(default_factory=dict)

    @property
    def unanimous(self) -> bool:
        return all(v is not Verdict.DISAGREE for v in self.verdicts.values())


def vote(results: list[CellResult]) -> list[Group]:
    """Cluster each group's scalars; the largest cluster is the consensus."""
    groups: dict[tuple[str, int], Group] = {}
    for r in results:
        key = (r.cell.app.key, r.cell.nodes)
        groups.setdefault(key, Group(app_key=key[0], nodes=key[1])).results.append(r)

    out = []
    for group in groups.values():
        app = group.results[0].cell.app
        voters = [
            r for r in group.results
            if r.status is Status.OK and r.scalar is not None
        ]
        clusters: list[list[CellResult]] = []
        for r in voters:
            for cluster in clusters:
                if close(cluster[0].scalar, r.scalar, app.scalar_kind, app.tolerance):
                    cluster.append(r)
                    break
            else:
                clusters.append([r])
        clusters.sort(key=len, reverse=True)
        if clusters:
            group.consensus = clusters[0][0].scalar
            majority = {id(r) for r in clusters[0]}
        else:
            majority = set()

        for r in group.results:
            if r.status is Status.SKIPPED:
                group.verdicts[r.cell.entry.key] = Verdict.NA
            elif r.status is not Status.OK or r.scalar is None:
                group.verdicts[r.cell.entry.key] = Verdict.FAIL
            elif id(r) in majority:
                group.verdicts[r.cell.entry.key] = Verdict.OK
            else:
                group.verdicts[r.cell.entry.key] = Verdict.DISAGREE
            group.teardown_hang[r.cell.entry.key] = r.teardown_hang

        # A pinned answer catches the case where every configuration agrees on
        # the same wrong value — but it only answers for the workload it was
        # derived from, so a campaign running other arguments has no pin.
        args = group.results[0].cell.args
        pinned = bool(app.expect) and list(app.expect_args) == list(args)
        if pinned and group.consensus is not None:
            if not close(app.expect, group.consensus, app.scalar_kind, app.tolerance):
                for key, verdict in group.verdicts.items():
                    if verdict is Verdict.OK:
                        group.verdicts[key] = Verdict.EXPECT_FAIL
        out.append(group)

    out.sort(key=lambda g: (g.app_key, g.nodes))
    return out


def minority_report(groups: list[Group]) -> list[Group]:
    return [g for g in groups if not g.unanimous or
            any(v is Verdict.EXPECT_FAIL for v in g.verdicts.values())]
