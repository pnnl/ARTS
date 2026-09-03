"""The cross-model HPX entry: where its cells come from and where they don't.

A port is one source mirroring a LIST of version tiers, one target per tier
named by the row's binary; every other surface — binary path, build target,
launch shape, ineligibility reason — follows from that list.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from artsrun.build import plan_targets
from artsrun.model.benchset import Benchset
from artsrun.model.catalog import AppEntry, Version, load_catalog
from artsrun.model.plane import RuntimeKind, load_plane
from artsrun.model.profile import Profile
from artsrun.model.selection import Selection
from artsrun.model.sweep import SweepSpec
from artsrun.run.command import build_command
from artsrun.run.plan import expand


def _profile() -> Profile:
    return Profile.model_validate({
        "name": "t", "launcher": "local", "nodes": [1, 2],
        "workers": 15, "progress": 1,
    })


def _selection(entries: list[str], versions=None) -> Selection:
    return Selection(
        profile="t", benchset="paper-main", entries=entries,
        apps={"nqueens": versions or [Version.BASE, Version.HINTED]},
        node_counts=[1, 2], repeats=1,
    )


def _entry(**over) -> AppEntry:
    base = {"name": "x", "binary": "x", "class": "task", "marker": "M"}
    base.update(over)
    return AppEntry.model_validate(base)


# --- the catalog field ------------------------------------------------------
def test_hpx_target_names_one_binary_per_mirrored_tier():
    e = _entry(hinted=True, hpx=["base", "hinted"])
    assert e.hpx_target(Version.BASE) == "x_hpx"
    assert e.hpx_target(Version.HINTED) == "x_hinted_hpx"
    assert e.hpx_target(Version.RESTRUCTURED) is None


def test_a_port_may_mirror_only_tiers_the_row_offers():
    with pytest.raises(ValidationError, match="does not offer"):
        _entry(hpx=["hinted"])          # hinted: false


def test_a_rewrite_has_no_hpx_mirror():
    with pytest.raises(ValidationError, match="restructured"):
        _entry(restructured_as="y", hpx=["restructured"])


def test_the_old_single_tier_field_is_a_loud_error():
    with pytest.raises(ValidationError, match="hpx_tier"):
        _entry(hinted=True, hpx_tier="hinted")


def test_no_port_is_an_empty_list():
    assert _entry().hpx == []
    assert _entry().hpx_target(Version.BASE) is None


# --- resolution -------------------------------------------------------------
def test_the_port_binds_to_every_tier_it_mirrors():
    catalog = load_catalog()
    resolved = {a.key: a for a in Benchset(name="t").resolve(catalog)}
    row = catalog.apps["nqueens"]
    for version in row.hpx:
        assert resolved[f"nqueens:{version.value}"].hpx_binary == row.hpx_target(version)
    for version in set(row.own_versions) - set(row.hpx):
        assert resolved[f"nqueens:{version.value}"].hpx_binary is None
    assert resolved["nqueens:base"].hpx_versions == list(row.hpx)


def test_hpx_cells_exist_only_for_mirrored_tiers(tmp_path):
    plane = load_plane()
    catalog = load_catalog()
    row = catalog.apps["nqueens"]
    cells, skipped = expand(
        _selection(["hpx"]), plane, catalog, Benchset(name="t"), _profile(),
        tmp_path / "apps", {1: {}, 2: {}},
    )
    assert {(c.app.key, c.nodes) for c in cells} == {
        (f"nqueens:{v.value}", n) for v in row.hpx for n in (1, 2)
    }
    for c in cells:
        assert c.binary == tmp_path / "hpx" / row.hpx_target(c.app.version)
        assert c.cfg is None
    unmirrored = [v for v in row.own_versions if v not in row.hpx]
    for v in unmirrored:
        reasons = {s.reason for s in skipped if s.app_key == f"nqueens:{v.value}"}
        assert reasons == {
            f"the HPX port does not mirror the {v.value} tier "
            f"(it mirrors {', '.join(x.value for x in row.hpx)})"
        }


def test_an_application_without_a_port_says_so(tmp_path):
    plane = load_plane()
    catalog = load_catalog()
    sel = Selection(profile="t", benchset="paper-main", entries=["hpx"],
                    apps={"fft": [Version.BASE]}, node_counts=[1], repeats=1)
    cells, skipped = expand(sel, plane, catalog, Benchset(name="t"),
                            _profile(), tmp_path / "apps", {1: {}})
    assert not cells
    assert {s.reason for s in skipped} == {
        "no HPX port exists for this application"}


def test_expansion_records_a_skip_once_however_many_repeats(tmp_path):
    plane = load_plane()
    catalog = load_catalog()
    sel = Selection(profile="t", benchset="paper-main", entries=["hpx"],
                    apps={"fft": [Version.BASE]}, node_counts=[1], repeats=3)
    cells, skipped = expand(sel, plane, catalog, Benchset(name="t"),
                            _profile(), tmp_path / "apps", {1: {}})
    assert not cells and len(skipped) == 1


def test_the_build_plan_wants_one_target_per_mirrored_tier(tmp_path):
    plane = load_plane()
    catalog = load_catalog()
    row = catalog.apps["nqueens"]
    plan = plan_targets(
        _selection(["hpx", "arts_val_wb"]), plane, catalog,
        Benchset(name="t"), tmp_path,
    )
    for version in row.hpx:
        assert plan.targets.count(row.hpx_target(version)) == 1
    assert "nqueens_arts_ocr_val_wb" in plan.targets
    assert "nqueens_hinted_arts_ocr_val_wb" in plan.targets


def test_an_hpx_cell_launches_like_a_reference(monkeypatch, tmp_path):
    monkeypatch.setattr("artsrun.run.command._mpi_probe", lambda: "mpich")
    plane = load_plane()
    catalog = load_catalog()
    cells, _ = expand(
        _selection(["hpx"], [Version.HINTED]), plane, catalog,
        Benchset(name="t"), _profile(), tmp_path / "apps", {1: {}, 2: {}},
    )
    by_nodes = {c.nodes: c for c in cells}
    two = build_command(by_nodes[2], _profile())
    assert two[:3] == ["mpirun", "-bind-to", "none"]
    assert "rank" in two
    assert str(by_nodes[2].binary) in two
    one = build_command(by_nodes[1], _profile())
    assert one[0] == "bash" and one[1].endswith("envelope.sh")
    assert "mpirun" not in one


def test_hpx_never_derives_a_binary_from_a_version_stem():
    plane = load_plane()
    with pytest.raises(ValueError):
        plane.entry("hpx").binary("nqueens", hinted=True)


def test_cells_carry_the_cpu_block_they_were_granted(tmp_path):
    plane = load_plane()
    catalog = load_catalog()
    # A real "arts"/"ocr" pair, not the {} the pure-HPX tests above get away
    # with: mixing in an ARTS entry proves cpu_width is set once at the
    # shared Cell(...) site, not only for the HPX arm.
    configs = {n: {"arts": tmp_path / f"arts_{n}.cfg", "ocr": tmp_path / f"ocr_{n}.cfg"}
               for n in (1, 2)}
    cells, _ = expand(_selection(["hpx", "arts_val_wb"]), plane, catalog,
                      Benchset(name="t"), _profile(), tmp_path / "apps",
                      configs)
    assert cells and all(c.cpu_width == 16 for c in cells)


def _hpx_cell(tmp_path):
    plane = load_plane()
    catalog = load_catalog()
    cells, _ = expand(_selection(["hpx"], [Version.HINTED]), plane, catalog,
                      Benchset(name="t"), _profile(), tmp_path / "apps",
                      {1: {}, 2: {}})
    return cells[0]


def test_colocated_hpx_ranks_get_the_posix_shared_memory_transport(tmp_path):
    from artsrun.run.command import build_env

    assert build_env(_hpx_cell(tmp_path), _profile())["UCX_TLS"] == "^sysv"


@pytest.mark.parametrize("launcher, extra", [
    ("slurm", {"slurm": {"partition": "p"}}),
    ("flux", {"flux": {"queue": "q"}}),
    ("ssh", {"ssh": {"budget": 2, "hosts": ["n01", "n02"]}}),
])
def test_a_scheduled_hpx_cell_keeps_the_site_mpi_defaults(tmp_path, launcher,
                                                          extra):
    from artsrun.run.command import build_env

    remote = Profile.model_validate({
        "name": "t", "launcher": launcher, "nodes": [1, 2],
        "workers": 15, "progress": 1, "ports": [20000], **extra,
    })
    assert "UCX_TLS" not in build_env(_hpx_cell(tmp_path), remote)


def test_an_arts_cell_never_carries_the_transport_override(tmp_path):
    from artsrun.run.command import build_env

    plane = load_plane()
    catalog = load_catalog()
    configs = {n: {"arts": tmp_path / f"arts_{n}.cfg"} for n in (1, 2)}
    cells, _ = expand(_selection(["arts_val_wb"], [Version.HINTED]), plane,
                      catalog, Benchset(name="t"), _profile(),
                      tmp_path / "apps", configs)
    assert "UCX_TLS" not in build_env(cells[0], _profile())


def test_the_struct_marker_is_forwarded_only_when_set(monkeypatch):
    from artsrun.run.command import build_env
    plane = load_plane()
    catalog = load_catalog()
    cells, _ = expand(_selection(["hpx"], [Version.HINTED]), plane, catalog,
                      Benchset(name="t"), _profile(), Path("/apps"), {1: {}, 2: {}})
    monkeypatch.delenv("ARTS_STRUCT_MARKER", raising=False)
    assert "ARTS_STRUCT_MARKER" not in build_env(cells[0], _profile())
    monkeypatch.setenv("ARTS_STRUCT_MARKER", "1")
    assert build_env(cells[0], _profile())["ARTS_STRUCT_MARKER"] == "1"


# --- sweeps ------------------------------------------------------------------
def test_sweep_refuses_hpx_arm():
    catalog = load_catalog()
    spec = SweepSpec.model_validate({
        "name": "t", "app": "nqueens", "nodes": 1, "arms": ["hpx"],
        "arg_order": ["n"], "points": [{"name": "a", "set": {"n": 1}}],
    })
    with pytest.raises(ValueError, match="sweeps run ARTS builds only"):
        spec.validate_against(catalog, _profile())
