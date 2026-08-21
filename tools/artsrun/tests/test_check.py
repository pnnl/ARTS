from __future__ import annotations

from pathlib import Path

from artsrun.check import Verdict, extract, minority_report, vote
from artsrun.model.benchset import ResolvedApp
from artsrun.model.catalog import AppClass, ScalarKind, Version
from artsrun.model.plane import RuntimeKind, SelectionEntry
from artsrun.run.types import Cell, CellResult, Status


def _app(**over) -> ResolvedApp:
    base = dict(
        name="app", version=Version.BASE, binary="app", cls=AppClass.TASK,
        marker=r"RESULT", scalar_re=r"RESULT\s*=\s*([\d.]+)",
        scalar_kind=ScalarKind.FLOAT, args=["1"],
    )
    base.update(over)
    return ResolvedApp(**base)


def _cell(entry_key: str, app: ResolvedApp, nodes: int = 1) -> Cell:
    entry = SelectionEntry(
        key=entry_key, label=entry_key, kind=RuntimeKind.ARTS,
        cell="VAL/RETAIN/WB", variant=entry_key,
    )
    return Cell(
        entry=entry, app=app, nodes=nodes, repeat=1,
        binary=Path("/nonexistent"), args=app.args, timeout_s=10,
    )


def _result(entry_key: str, scalar: str | None, app: ResolvedApp,
            status: Status = Status.OK) -> CellResult:
    r = CellResult(cell=_cell(entry_key, app), status=status)
    r.scalar = scalar
    return r


# --- extraction -----------------------------------------------------------
def test_a_run_that_never_printed_its_marker_did_not_complete():
    completed, scalar = extract("starting up\n", r"RESULT", r"RESULT\s*=\s*(\d+)")
    assert not completed and scalar is None


def test_a_capture_group_yields_the_value():
    completed, scalar = extract("RESULT = 42\n", r"RESULT", r"RESULT\s*=\s*(\d+)")
    assert completed and scalar == "42"


def test_a_pattern_without_a_group_is_a_completion_assertion():
    completed, scalar = extract("PASSED\n", r"PASSED", r"PASSED")
    assert completed and scalar == "MATCHED"


# --- voting ---------------------------------------------------------------
def test_e2e_marker_is_parsed_in_seconds():
    from artsrun.check import extract_e2e
    assert extract_e2e("noise\n[E2E] 1500000000\n") == 1.5


def test_e2e_last_marker_wins_and_absence_is_none():
    from artsrun.check import extract_e2e
    assert extract_e2e("[E2E] 1000000000\n[E2E] 2000000000\n") == 2.0
    assert extract_e2e("no marker here\n") is None


def test_e2e_marker_must_stand_alone_on_its_line():
    from artsrun.check import extract_e2e
    assert extract_e2e("app says [E2E] 5 things\n") is None


def test_unanimous_results_all_pass():
    app = _app()
    groups = vote([_result(k, "1.0", app) for k in ("a", "b", "c")])
    assert len(groups) == 1
    assert groups[0].consensus == "1.0"
    assert set(groups[0].verdicts.values()) == {Verdict.OK}
    assert not minority_report(groups)


def test_a_lone_dissenter_loses_the_vote():
    app = _app()
    results = [_result("a", "1.0", app), _result("b", "1.0", app),
               _result("c", "2.0", app)]
    group = vote(results)[0]
    assert group.consensus == "1.0"
    assert group.verdicts["c"] is Verdict.DISAGREE
    assert minority_report(vote(results))


def test_a_failed_cell_does_not_vote():
    app = _app()
    results = [_result("a", "1.0", app), _result("b", None, app, Status.TIMEOUT)]
    group = vote(results)[0]
    assert group.verdicts["b"] is Verdict.FAIL
    assert group.consensus == "1.0"


def test_an_ineligible_cell_is_not_a_failure():
    app = _app()
    results = [_result("a", "1.0", app), _result("b", None, app, Status.SKIPPED)]
    group = vote(results)[0]
    assert group.verdicts["b"] is Verdict.NA


def test_floating_point_results_cluster_within_tolerance():
    app = _app(tolerance=1e-3)
    group = vote([_result("a", "1.0000", app), _result("b", "1.0005", app)])[0]
    assert set(group.verdicts.values()) == {Verdict.OK}


def test_agreeing_on_the_wrong_value_still_fails_against_a_pin():
    app = _app(expect="9.0", expect_args=["1"])
    group = vote([_result("a", "1.0", app), _result("b", "1.0", app)])[0]
    assert set(group.verdicts.values()) == {Verdict.EXPECT_FAIL}


def test_a_pin_does_not_apply_to_a_different_workload():
    # The pinned answer was derived for other arguments, so a campaign that
    # changed the workload has no pin — not a failing one.
    app = _app(expect="9.0", expect_args=["999"])
    group = vote([_result("a", "1.0", app), _result("b", "1.0", app)])[0]
    assert set(group.verdicts.values()) == {Verdict.OK}


def test_node_counts_are_voted_separately():
    app = _app()
    r1 = _result("a", "1.0", app)
    r2 = CellResult(cell=_cell("a", app, nodes=2), status=Status.OK)
    r2.scalar = "2.0"
    groups = vote([r1, r2])
    assert [g.nodes for g in groups] == [1, 2]
    assert all(g.unanimous for g in groups)
