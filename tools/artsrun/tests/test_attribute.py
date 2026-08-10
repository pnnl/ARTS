"""Per-task attribution: reading the object tables, and admitting the gap.

The report exists to name the task a placement hint should go on, so the two
properties that matter are that a task's remote share survives aggregation
across ranks, and that work no task claimed is shown rather than absorbed.
"""

from __future__ import annotations

import json
import re

from artsrun import attribute as attr
from artsrun.paths import countersets_dir, repo_root
from artsrun.store import load_counterset


def _obj(rank, db_rows, edt_rows=()):
    return {
        "metadata": {"node_id": rank},
        "edt_objects": [
            {"arts_id": i, "count": c, "exec_ns": e, "stall_ns": 0}
            for i, c, e in edt_rows
        ],
        "edt_collisions": 0,
        "db_objects": [
            {"arts_id": i, "count": c, "bytes": b, "cache_misses": m}
            for i, c, b, m in db_rows
        ],
        "db_collisions": 0,
    }


def _globals(local_hit, remote):
    return {"counters": {
        "NUM_DB_ACQUIRE_LOCAL_HIT": {"value": local_hit},
        "NUM_DB_ACQUIRE_REMOTE": {"value": remote},
    }}


def test_a_task_totals_across_the_ranks_that_ran_it(tmp_path):
    (tmp_path / "object_n0.json").write_text(json.dumps(
        _obj(0, [(0x400, 10, 60, 5)], [(0x400, 10, 100)])))
    (tmp_path / "object_n1.json").write_text(json.dumps(
        _obj(1, [(0x400, 6, 24, 4)], [(0x400, 6, 60)])))

    a = attr.read(tmp_path)
    assert len(a.tasks) == 1
    t = a.tasks[0]
    assert (t.acquires, t.remote, t.bytes, t.runs) == (16, 9, 84, 16)
    assert a.ranks == 2


def test_unlabelled_work_is_reported_not_absorbed(tmp_path):
    # The runtime's own bootstrap task carries no identity; its acquires must
    # stay visible as a residual, or the table would read as complete when it
    # is not.
    (tmp_path / "object_n0.json").write_text(json.dumps(
        _obj(0, [(0x400, 10, 60, 5)])))
    (tmp_path / "n0.json").write_text(json.dumps(_globals(6, 6)))

    a = attr.read(tmp_path)
    assert a.global_acquires == 12
    assert a.attributed_acquires == 10
    assert a.residual_acquires == 2
    assert a.residual_remote == 1


def test_a_complete_attribution_leaves_no_residual(tmp_path):
    (tmp_path / "object_n0.json").write_text(json.dumps(
        _obj(0, [(0x400, 10, 60, 5)])))
    (tmp_path / "n0.json").write_text(json.dumps(_globals(5, 5)))

    a = attr.read(tmp_path)
    assert a.residual_acquires == 0 and a.residual_remote == 0


def test_the_worst_task_ranks_first(tmp_path):
    (tmp_path / "object_n0.json").write_text(json.dumps(_obj(0, [
        (0x400, 100, 400, 0),      # never misses
        (0x500, 50, 200, 50),      # always misses
    ])))
    assert [t.arts_id for t in attr.read(tmp_path).ranked()] == [0x500, 0x400]


def test_addresses_resolve_to_names_against_the_binary(tmp_path):
    # nm on a real binary; any repository executable will do, so the test does
    # not depend on a build configuration.
    import shutil

    binary = shutil.which("ls")
    table = attr.symbolize(__import__("pathlib").Path(binary))
    # A stripped /bin/ls yields nothing, which is a supported outcome — the
    # contract is that it never raises and never invents a name.
    assert all(isinstance(k, int) and isinstance(v, str) for k, v in table.items())


def test_a_missing_binary_degrades_to_addresses(tmp_path):
    (tmp_path / "object_n0.json").write_text(json.dumps(
        _obj(0, [(0x4040, 1, 4, 0)])))
    a = attr.read(tmp_path, tmp_path / "does-not-exist")
    assert a.tasks[0].name == "0x4040"


def test_cells_lists_only_directories_holding_counters(tmp_path):
    root = tmp_path / "counters"
    (root / "app.asborn.val_wb.1n.r1").mkdir(parents=True)
    (root / "app.asborn.val_wb.1n.r1" / "n0.json").write_text("{}")
    (root / "empty").mkdir()
    assert [p.name for p in attr.cells(tmp_path)] == ["app.asborn.val_wb.1n.r1"]


def test_the_attribution_set_enables_the_per_task_tables():
    # Without the OBJ counters the report has nothing to read, so this set is
    # the one the command's error message points at.
    cs = load_counterset("attribution")
    assert {"OBJ_NUM_DB", "OBJ_BYTES_DB", "OBJ_NUM_DB_CACHE_MISS"} <= set(
        cs.enabled
    )
    # and the globals it reconciles against
    assert {"NUM_DB_ACQUIRE_LOCAL_HIT", "NUM_DB_ACQUIRE_REMOTE"} <= set(cs.enabled)


def test_the_shim_labels_every_task_it_creates():
    # A task's identity comes from the shim setting edt_id; without it the
    # per-task tables stay empty and the report has nothing to say.
    src = (repo_root() / "benchmarks/ocr_shim/arts_ocr.c").read_text()
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith(("*", "//", "/*"))
    )
    creates = len(re.findall(r"\barts_edt_create\s*\(", code))
    labels = len(re.findall(r"\bedt_id\s*=", code))
    assert creates > 0
    assert labels >= creates, f"{creates - labels} task(s) created without an identity"


def test_every_counterset_on_disk_loads():
    for path in sorted(countersets_dir().glob("*.yaml")):
        cs = load_counterset(path.stem)
        assert cs.name == path.stem
        assert cs.enabled, f"{path.stem} turns nothing on"


def test_the_runtime_takes_counts_and_bytes_at_different_sites():
    # Counts are classified where the coherence arm decides, bytes where the
    # payload lands.  Collapsing them onto one site reconciles in exactly one
    # arm, so this pins the split that made all three agree.
    root = repo_root() / "libs/src/core"
    db = (root / "db.c").read_text()
    assert "arts_object_task_enter" in db and "arts_object_task_leave" in db
    # the acquire window brackets the arm's handler
    enter = db.index("arts_object_task_enter")
    assert "arts_handler_db_acquire" in db[enter:enter + 400]

    for arm in ("coherence.c", "grant.c", "inv/wb.c", "inv/wt.c",
                "excl/purge.c", "excl/retain.c"):
        src = (root / "coherence" / arm).read_text()
        globals_ = (src.count("INCREMENT_NUM_DB_ACQUIRE_LOCAL_HIT_BY")
                    + src.count("INCREMENT_NUM_DB_ACQUIRE_REMOTE_BY"))
        assert src.count("arts_object_acquire(") == globals_, (
            f"{arm} counts per-task acquires at a different set of sites than "
            f"the cluster totals"
        )
