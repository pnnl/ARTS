"""The run-directory reader: manifest round-trip, track replay, live voting."""

from __future__ import annotations

import json
from pathlib import Path

from artsrun.check import Verdict
from artsrun.model.benchset import ResolvedApp
from artsrun.model.catalog import AppClass, ScalarKind, Version
from artsrun.model.plane import RuntimeKind, SelectionEntry
from artsrun.model.profile import Profile
from artsrun.run.manifest import Manifest, write_manifest
from artsrun.run.types import Cell, Skipped, Status
from artsrun.watch.state import LogTail, RunState

ENTRIES = {
    "arts_val_wb": "VAL/RETAIN/WB",
    "arts_excl_retain": "EXCL/RETAIN/WB",
}


def _profile(launcher: str = "local") -> Profile:
    data = {
        "name": "t", "launcher": launcher, "nodes": [1, 2],
        "workers": 2, "progress": 1,
    }
    if launcher == "slurm":
        data["ports"] = [25000]
        data["slurm"] = {"budget": 2}
    return Profile.model_validate(data)


def _cell(entry_key: str, nodes: int = 1, repeat: int = 1,
          cfg: Path | None = None) -> Cell:
    app = ResolvedApp(
        name="app", version=Version.ASBORN, binary="app", cls=AppClass.TASK,
        marker=r"RESULT", scalar_re=r"RESULT\s*=\s*([\d.]+)",
        scalar_kind=ScalarKind.FLOAT,
    )
    entry = SelectionEntry(
        key=entry_key, label=entry_key, kind=RuntimeKind.ARTS,
        cell=ENTRIES[entry_key], variant=entry_key,
    )
    return Cell(entry=entry, app=app, nodes=nodes, repeat=repeat,
                binary=Path("/opt/bin/app"), args=["3"], timeout_s=60,
                cfg=cfg)


def _write(run_dir: Path, cells: list[Cell], *, launcher: str = "local",
           skipped: list[Skipped] | None = None,
           counters_cfg: Path | None = None) -> None:
    write_manifest(run_dir, cells, skipped or [], _profile(launcher),
                   Path("/opt/build"), counters_cfg=counters_cfg)


def _track(run_dir: Path, rows: list[dict]) -> None:
    with (run_dir / "track.jsonl").open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _finished(cell: Cell, scalar: str, status: str = "ok") -> dict:
    return {"t": 1.0, "event": "finished", "cell": cell.key, "status": status,
            "rc": 0 if status == "ok" else 1, "wall_s": 2.5, "note": "",
            "scalar": scalar}


# --- manifest -------------------------------------------------------------


def test_the_manifest_round_trips_cells_and_their_invocations(tmp_path):
    cells = [_cell("arts_val_wb"), _cell("arts_excl_retain", nodes=2)]
    _write(tmp_path, cells,
           skipped=[Skipped("arts_val_wb", "other:asborn", 2, "why not")])
    manifest = Manifest.load(tmp_path)
    assert manifest is not None
    assert [c.key for c in manifest.cells] == [c.key for c in cells]
    back = manifest.cells[0]
    assert back.app.scalar_re == cells[0].app.scalar_re
    assert back.entry.cell == "VAL/RETAIN/WB"
    meta = manifest.commands[back.key]
    assert meta["command"].startswith("timeout -k 1 60 /opt/bin/app")
    assert meta["script"] is None
    assert meta["log"].endswith(f"cells/{cells[0].log_name}")
    assert manifest.skipped[0].reason == "why not"


def test_the_manifest_names_the_configurations_behind_a_cell(tmp_path):
    cfg = tmp_path / "cfg" / "arts_1n.cfg"
    counters = tmp_path / "cfg" / "counters_perf.cfg"
    _write(tmp_path, [_cell("arts_val_wb", cfg=cfg)], counters_cfg=counters)
    manifest = Manifest.load(tmp_path)
    assert manifest.cells[0].cfg == cfg
    assert manifest.counters_cfg == counters

    state = RunState(tmp_path)
    assert state.counters_cfg == counters
    assert state.views[manifest.cells[0].key].cfg == str(cfg)


def test_a_slurm_manifest_carries_both_layers_of_the_command(tmp_path):
    _write(tmp_path, [_cell("arts_val_wb", nodes=2)], launcher="slurm")
    manifest = Manifest.load(tmp_path)
    meta = manifest.commands[next(iter(manifest.commands))]
    assert meta["command"].startswith("sbatch")
    assert "--nodes=2" in meta["command"]
    assert "srun" in meta["script"]
    assert "timeout -k 1 60" in meta["script"]


# --- replay ---------------------------------------------------------------


def test_every_cell_starts_pending_from_the_manifest_alone(tmp_path):
    cells = [_cell("arts_val_wb"), _cell("arts_excl_retain")]
    _write(tmp_path, cells)
    state = RunState(tmp_path)
    assert len(state.order) == 2
    assert all(v.status is Status.PENDING for v in state.views.values())
    assert state.views[cells[0].key].family == "VAL"
    assert state.views[cells[1].key].family == "EXCL"


def test_events_walk_a_cell_through_its_life(tmp_path):
    cell = _cell("arts_val_wb")
    _write(tmp_path, [cell])
    state = RunState(tmp_path)

    _track(tmp_path, [{"t": 5.0, "event": "started", "cell": cell.key,
                       "status": "running", "rc": 0, "wall_s": 0,
                       "note": "", "extra": {"pid": "42"}}])
    assert state.refresh()
    view = state.views[cell.key]
    assert view.status is Status.RUNNING
    assert view.extra["pid"] == "42"
    assert view.started_at == 5.0

    _track(tmp_path, [_finished(cell, "1.5")])
    assert state.refresh()
    assert view.status is Status.OK
    assert view.scalar == "1.5"
    assert view.wall_s == 2.5
    assert state.finished


def test_consensus_recolors_as_results_arrive(tmp_path):
    agree = _cell("arts_val_wb")
    dissent = _cell("arts_excl_retain")
    _write(tmp_path, [agree, dissent])
    state = RunState(tmp_path)

    _track(tmp_path, [_finished(agree, "1.0")])
    state.refresh()
    assert state.views[agree.key].verdict is Verdict.OK
    assert not state.broken_groups

    _track(tmp_path, [_finished(dissent, "2.0")])
    state.refresh()
    assert state.views[agree.key].verdict is Verdict.OK
    assert state.views[dissent.key].verdict is Verdict.DISAGREE
    assert state.views[dissent.key].consensus == "1.0"
    assert len(state.broken_groups) == 1


def test_a_failure_is_a_failed_verdict_not_a_vote(tmp_path):
    ok = _cell("arts_val_wb")
    bad = _cell("arts_excl_retain")
    _write(tmp_path, [ok, bad])
    state = RunState(tmp_path)
    _track(tmp_path, [_finished(ok, "1.0"),
                      {"t": 2.0, "event": "finished", "cell": bad.key,
                       "status": "timeout", "rc": 124, "wall_s": 60.0,
                       "note": ""}])
    state.refresh()
    assert state.views[bad.key].status is Status.TIMEOUT
    assert state.views[bad.key].verdict is Verdict.FAIL
    assert state.views[ok.key].verdict is Verdict.OK


def test_a_partial_track_line_waits_for_its_newline(tmp_path):
    first = _cell("arts_val_wb")
    second = _cell("arts_excl_retain")
    _write(tmp_path, [first, second])
    state = RunState(tmp_path)

    whole = json.dumps(_finished(first, "1.0")) + "\n"
    half = json.dumps(_finished(second, "1.0"))
    cut = len(half) // 2
    with (tmp_path / "track.jsonl").open("ab") as fh:
        fh.write(whole.encode() + half[:cut].encode())
    state.refresh()
    assert state.views[first.key].status is Status.OK
    assert state.views[second.key].status is Status.PENDING

    with (tmp_path / "track.jsonl").open("ab") as fh:
        fh.write(half[cut:].encode() + b"\n")
    state.refresh()
    assert state.views[second.key].status is Status.OK


def test_a_run_without_a_manifest_replays_from_its_track_alone(tmp_path):
    _track(tmp_path, [{"t": 1.0, "event": "finished",
                       "cell": "graph500:asborn@2n/arts_inv_wb#3",
                       "status": "ok", "rc": 0, "wall_s": 26.5, "note": ""}])
    state = RunState(tmp_path)
    view = state.views["graph500:asborn@2n/arts_inv_wb#3"]
    assert view.app_name == "graph500"
    assert view.version == "asborn"
    assert view.entry_key == "arts_inv_wb"
    assert view.nodes == 2
    assert view.repeat == 3
    assert view.status is Status.OK
    assert view.verdict is None  # no metadata to vote with
    assert view.log_path is not None
    assert view.log_path.name == "graph500.asborn.arts_inv_wb.2n.r3.log"


def test_the_manifest_may_arrive_after_the_track(tmp_path):
    # A watcher can attach during the build phase, when the track of an
    # earlier continuation exists but the manifest is yet to be written.
    cell = _cell("arts_val_wb")
    _track(tmp_path, [_finished(cell, "1.0")])
    state = RunState(tmp_path)
    assert state.views[cell.key].command == ""

    _write(tmp_path, [cell, _cell("arts_excl_retain")])
    assert state.refresh()
    view = state.views[cell.key]
    assert view.status is Status.OK  # progress kept
    assert view.command.startswith("timeout")  # identity grafted on
    assert len(state.order) == 2


def test_scaling_takes_the_best_wall_of_completed_cells_only(tmp_path):
    one = _cell("arts_val_wb")
    one_again = _cell("arts_val_wb", repeat=2)
    two = _cell("arts_val_wb", nodes=2)
    other = _cell("arts_excl_retain", nodes=2)
    _write(tmp_path, [one, one_again, two, other])
    state = RunState(tmp_path)
    _track(tmp_path, [
        {"t": 1.0, "event": "finished", "cell": one.key, "status": "ok",
         "rc": 0, "wall_s": 10.0, "note": "", "scalar": "1.0"},
        {"t": 2.0, "event": "finished", "cell": one_again.key, "status": "ok",
         "rc": 0, "wall_s": 8.0, "note": "", "scalar": "1.0"},
        {"t": 3.0, "event": "finished", "cell": two.key, "status": "ok",
         "rc": 0, "wall_s": 5.0, "note": "", "scalar": "1.0"},
        {"t": 4.0, "event": "finished", "cell": other.key, "status": "timeout",
         "rc": 124, "wall_s": 60.0, "note": ""},
    ])
    state.refresh()
    node_counts, rows = state.scaling()
    assert node_counts == [1, 2]
    assert len(rows) == 1  # the timed-out entry measured nothing
    assert rows[0]["entry"] == "arts_val_wb"
    assert rows[0]["walls"] == {1: 8.0, 2: 5.0}  # repeats collapse to best


def test_a_draining_cell_freezes_its_clock_where_the_run_ended(tmp_path):
    # Slurm's COMPLETING means nothing is computing any more; the elapsed
    # figure stops at the drain's start instead of counting teardown time.
    cell = _cell("arts_val_wb")
    _write(tmp_path, [cell])
    state = RunState(tmp_path)
    _track(tmp_path, [
        {"t": 100.0, "event": "running", "cell": cell.key,
         "status": "running", "rc": 0, "wall_s": 0, "note": ""},
        {"t": 160.0, "event": "ending", "cell": cell.key,
         "status": "ending", "rc": 0, "wall_s": 0, "note": ""},
    ])
    state.refresh()
    view = state.views[cell.key]
    assert view.status is Status.ENDING
    assert view.elapsed_s == 60.0  # frozen, no matter when we look
    assert view.active
    assert not state.finished


def test_a_pre_rename_track_still_lands_on_its_cell(tmp_path):
    # The optimized version was once named "hinted": its value in a saved
    # manifest parses through the alias, and an old track key still lands on
    # the freshly keyed cell instead of spawning a stub twin.
    from artsrun.model.catalog import Version

    cell = _cell("arts_val_wb")
    cell = Cell(entry=cell.entry,
                app=cell.app.model_copy(update={"version": Version.OPTIMIZED}),
                nodes=1, repeat=1, binary=cell.binary, args=cell.args,
                timeout_s=cell.timeout_s)
    assert cell.key == "app:optimized@1n/arts_val_wb#1"
    _write(tmp_path, [cell])
    state = RunState(tmp_path)
    _track(tmp_path, [{"t": 1.0, "event": "finished",
                       "cell": "app:hinted@1n/arts_val_wb#1", "status": "ok",
                       "rc": 0, "wall_s": 2.0, "note": "", "scalar": "1.0"}])
    state.refresh()
    assert len(state.order) == 1
    assert state.views[cell.key].status is Status.OK


# --- log tailing ----------------------------------------------------------


def test_the_tail_reads_only_what_is_new(tmp_path):
    log = tmp_path / "cell.log"
    log.write_text("one\n")
    tail = LogTail(log)
    assert tail.read_new() == "one\n"
    assert tail.read_new() == ""
    with log.open("a") as fh:
        fh.write("two\n")
    assert tail.read_new() == "two\n"


def test_a_shrunken_log_is_a_new_file_wearing_an_old_name(tmp_path):
    log = tmp_path / "cell.log"
    log.write_text("a long first life\n")
    tail = LogTail(log)
    tail.read_new()
    log.write_text("rebirth\n")
    assert tail.read_new() == "rebirth\n"


def test_a_late_attach_starts_near_the_end_of_a_big_log(tmp_path):
    log = tmp_path / "cell.log"
    log.write_text("x" * 100 + "\ntail line\n")
    tail = LogTail(log, tail_bytes=16)
    first = tail.read_new()
    assert "earlier bytes not shown" in first
    assert first.endswith("tail line\n")
    assert "x" * 50 not in first
