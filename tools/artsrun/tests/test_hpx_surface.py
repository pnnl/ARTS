"""The cross-model HPX entry: where its cells come from and where they don't.

The port is one program mirroring one version tier, so exactly one row per
application carries it; every other surface — binary path, build target,
launch shape — follows from that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from artsrun.build import plan_targets
from artsrun.model.benchset import Benchset
from artsrun.model.catalog import Version, load_catalog
from artsrun.model.plane import RuntimeKind, load_plane
from artsrun.model.profile import Profile
from artsrun.model.selection import Selection
from artsrun.run.command import build_command
from artsrun.run.plan import expand


def _profile() -> Profile:
    return Profile.model_validate({
        "name": "t", "launcher": "local", "nodes": [1, 2],
        "workers": 15, "progress": 1,
    })


def _selection(entries: list[str]) -> Selection:
    return Selection(
        profile="t", benchset="paper-main", entries=entries,
        apps={"nqueens": [Version.BASE, Version.HINTED]},
        node_counts=[1, 2], repeats=1,
    )


def test_the_port_binds_to_the_tier_it_mirrors():
    catalog = load_catalog()
    resolved = {a.key: a for a in Benchset(name="t").resolve(catalog)}
    assert resolved["nqueens:hinted"].hpx_binary == "nqueens_hpx"
    assert resolved["nqueens:base"].hpx_binary is None


def test_hpx_cells_exist_only_for_the_mirrored_tier(tmp_path):
    plane = load_plane()
    catalog = load_catalog()
    cells, skipped = expand(
        _selection(["hpx"]), plane, catalog, Benchset(name="t"), _profile(),
        tmp_path / "apps", {1: {}, 2: {}},
    )
    assert {(c.app.key, c.nodes) for c in cells} == {
        ("nqueens:hinted", 1), ("nqueens:hinted", 2),
    }
    # The binary is the port's own target beside the OCR apps, never a
    # version-stem derivation.
    assert all(c.binary == tmp_path / "hpx" / "nqueens_hpx" for c in cells)
    assert all(c.cfg is None for c in cells)
    reasons = {s.reason for s in skipped if s.app_key == "nqueens:base"}
    assert reasons == {"application has no HPX port at this version tier"}


def test_the_build_plan_wants_the_port_target_once(tmp_path):
    plane = load_plane()
    catalog = load_catalog()
    plan = plan_targets(
        _selection(["hpx", "arts_val_wb"]), plane, catalog,
        Benchset(name="t"), tmp_path,
    )
    assert plan.targets.count("nqueens_hpx") == 1
    assert "nqueens_arts_ocr_val_wb" in plan.targets
    assert "nqueens_hinted_arts_ocr_val_wb" in plan.targets
    assert not any("hinted_hpx" in t for t in plan.targets)


def test_an_hpx_cell_launches_like_a_reference(monkeypatch, tmp_path):
    monkeypatch.setattr("artsrun.run.command._mpi_probe", lambda: "mpich")
    plane = load_plane()
    catalog = load_catalog()
    cells, _ = expand(
        _selection(["hpx"]), plane, catalog, Benchset(name="t"), _profile(),
        tmp_path / "apps", {1: {}, 2: {}},
    )
    by_nodes = {c.nodes: c for c in cells}
    two = build_command(by_nodes[2], _profile())
    assert two[:3] == ["mpirun", "-bind-to", "none"]
    assert "rank" in two            # colocated ranks get shifted blocks
    assert str(by_nodes[2].binary) in two
    # A single locality is an MPI singleton: no launcher, envelope only.
    one = build_command(by_nodes[1], _profile())
    assert one[0] == "bash" and one[1].endswith("envelope.sh")
    assert "mpirun" not in one


def test_hpx_never_derives_a_binary_from_a_version_stem():
    plane = load_plane()
    with pytest.raises(ValueError):
        plane.entry("hpx").binary("nqueens", hinted=True)
