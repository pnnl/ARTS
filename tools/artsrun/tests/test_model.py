from __future__ import annotations

import pytest
from pydantic import ValidationError

from artsrun.model import (
    Benchset,
    BenchsetEntry,
    Family,
    Kind,
    Profile,
    Release,
    RuntimeKind,
    Selection,
    Version,
    Write,
    load_catalog,
    load_plane,
)


# --- plane ----------------------------------------------------------------
def test_plane_has_sixteen_positions_and_eight_configurations():
    plane = load_plane()
    assert len(plane.cells) == 16
    assert sum(c.buildable for c in plane.cells) == 8


def test_every_unbuildable_position_states_a_reason():
    plane = load_plane()
    for cell in plane.cells:
        if not cell.buildable:
            assert cell.reason
            if cell.family is Family.EXCL:
                assert cell.write is Write.WT
            else:
                assert cell.release is Release.PURGE


def test_two_configurations_offer_a_reference_and_the_rest_do_not():
    plane = load_plane()
    assert len(plane.entries) == 10
    refs = [e for e in plane.entries if e.is_reference]
    assert {e.key for e in refs} == {"xsocr", "ocrvx"}
    assert plane.entry("xsocr").cell == "EXCL/PURGE/WB"
    assert plane.entry("ocrvx").cell == "INV/RETAIN/WB"


def test_wrf_is_not_selectable():
    plane = load_plane()
    assert not any("wrf" in key for key in plane.entry_keys)


def test_binary_names_follow_the_build_convention():
    plane = load_plane()
    arts = plane.entry("arts_val_wb")
    assert arts.binary("nqueens", hinted=False) == "nqueens_arts_ocr_val_wb"
    assert arts.binary("nqueens", hinted=True) == "nqueens_hinted_arts_ocr_val_wb"
    assert plane.entry("xsocr").binary("nqueens", hinted=False) == "nqueens_xsocr"
    assert plane.entry("ocrvx").kind is RuntimeKind.OCRVX


# --- catalog --------------------------------------------------------------
def test_catalog_rows_exclude_rewrites():
    catalog = load_catalog()
    rewrites = {a.name for a in catalog.apps.values() if a.restructured_from}
    assert rewrites
    assert not rewrites & {a.name for a in catalog.rows}


def test_a_restructured_version_resolves_to_the_rewrite_target():
    # Whichever application offers one — naming a specific application here
    # makes the test fail when that application's rewrite is held back, which
    # says nothing about the resolution being tested.
    catalog = load_catalog()
    named = next(a for a in catalog.rows if a.restructured_as)
    source, stem = catalog.resolve(named.name, Version.RESTRUCTURED)
    assert source.name == named.restructured_as
    assert stem == catalog.apps[named.restructured_as].binary


def test_optimized_version_resolves_to_the_opt_target():
    catalog = load_catalog()
    _, stem = catalog.resolve("nqueens", Version.HINTED)
    assert stem.endswith("_hinted")


def test_optimized_is_refused_where_the_source_has_no_hint_layer():
    # Whichever application has no layer — naming one here makes the test fail
    # when that application later gains one, which says nothing about the
    # refusal being tested.
    catalog = load_catalog()
    bare = next(a for a in catalog.rows if not a.hinted)
    with pytest.raises(KeyError):
        catalog.resolve(bare.name, Version.HINTED)


def test_a_real_run_configures_a_missing_build_tree(tmp_path, monkeypatch):
    # The experiment tree is fully determined (Release, benchmarks on), so a
    # missing one is a first run, not an error to hand back to the user.
    from artsrun import build as build_mod

    calls = {}
    tree = tmp_path / "bt"

    class FakeProc:
        stdout = iter(())

        def wait(self):
            return 0

    def fake_popen(cmd, **_kw):
        calls["cmd"] = cmd
        tree.mkdir(parents=True, exist_ok=True)
        (tree / "build.ninja").write_text("")
        return FakeProc()

    monkeypatch.setattr(build_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(build_mod.shutil, "which", lambda _n: "/usr/bin/cmake")
    build_mod.ensure_build_dir(tree, bootstrap=True)
    assert "-GNinja" in calls["cmd"]
    assert f"-B{tree}" in calls["cmd"]
    assert "-DCMAKE_BUILD_TYPE=Release" in calls["cmd"]


def test_a_dry_run_configures_nothing(tmp_path):
    from artsrun import build as build_mod

    with pytest.raises(build_mod.BuildError, match="real run configures"):
        build_mod.ensure_build_dir(tmp_path / "bt", bootstrap=False)
    assert not (tmp_path / "bt").exists()


def test_the_old_hinted_name_still_parses_as_optimized():
    # The version was recorded as "hinted" before the rename; selections and
    # benchsets written under that name must replay unchanged.
    assert Version("hinted") is Version.HINTED
    selection = Selection.model_validate({
        "profile": "p", "benchset": "b", "entries": ["arts_val_wb"],
        "apps": {"nqueens": ["hinted"]}, "node_counts": [1],
    })
    assert selection.apps["nqueens"] == [Version.HINTED]
    bench = Benchset.model_validate(
        {"name": "b", "apps": {"nqueens": {"versions": ["hinted"]}}})
    assert bench.apps["nqueens"].versions == [Version.HINTED]


# --- profile --------------------------------------------------------------
def _local(**over):
    base = dict(
        name="t", launcher="local", nodes=[1, 2], workers=3, progress=1,
    )
    base.update(over)
    return base


def test_local_profile_rejects_ports():
    with pytest.raises(ValidationError, match="ports must not be set"):
        Profile.model_validate(_local(ports=[25000]))


def test_a_geometry_wider_than_the_machine_is_not_the_tool_s_call():
    # ARTS does not oversubscribe and says so at startup; the profile only
    # records the shape asked for.
    profile = Profile.model_validate(_local(nodes=[1, 2, 4, 8], workers=63))
    assert profile.threads_per_node == 64


def test_remote_profile_requires_ports():
    with pytest.raises(ValidationError, match="ports is required"):
        Profile.model_validate(
            _local(launcher="ssh", hosts=["a", "b"])
        )


def test_slurm_needs_no_budget_and_defaults_its_build_slot():
    # Scheduling the queue is Slurm's whole purpose: every cell is submitted
    # up front, so a profile carries no admission budget — only where the
    # cells and the build work go, and how wide the build slot is.
    profile = Profile.model_validate(_local(
        launcher="slurm", nodes=[1, 2, 4, 8], ports=[25000],
        slurm={"partition": "pbatch", "build_partition": "pdebug"},
    ))
    assert profile.slurm.partition == "pbatch"
    assert profile.slurm.build_partition == "pdebug"
    assert profile.slurm.build_cpus == 8


# --- benchset -------------------------------------------------------------
def test_benchset_falls_through_to_the_catalog():
    catalog = load_catalog()
    resolved = {a.key: a for a in Benchset(name="empty").resolve(catalog)}
    assert resolved["nqueens:base"].args == catalog.apps["nqueens"].args
    assert not resolved["nqueens:base"].args_overridden


def test_benchset_override_marks_the_argument_source():
    catalog = load_catalog()
    bs = Benchset(name="o", apps={"nqueens": BenchsetEntry(args=["8", "2"])})
    resolved = {a.key: a for a in bs.resolve(catalog)}
    assert resolved["nqueens:base"].args == ["8", "2"]
    assert resolved["nqueens:base"].args_overridden


def test_benchset_disable_removes_every_version():
    catalog = load_catalog()
    bs = Benchset(name="o", apps={"nqueens": BenchsetEntry(enabled=False)})
    assert not [a for a in bs.resolve(catalog) if a.name == "nqueens"]


def test_a_version_an_application_lacks_is_dropped_and_said_out_loud(capsys):
    # A roster keeps naming a version the catalog has withdrawn, so it comes
    # back on its own when the catalog restores it; what must not happen is
    # the campaign running as though it had measured it.
    catalog = load_catalog()
    bare = next(a for a in catalog.rows if not a.hinted)
    bs = Benchset(name="o", apps={bare.name: BenchsetEntry(
        versions=[Version.BASE, Version.HINTED])})
    got = bs.resolve(catalog)
    assert [a.key for a in got] == [f"{bare.name}:base"]
    assert "no hinted version" in capsys.readouterr().err


# --- selection ------------------------------------------------------------
def test_selection_rejects_a_node_count_outside_the_profile_sweep():
    plane, catalog = load_plane(), load_catalog()
    profile = Profile.model_validate(_local())
    sel = Selection(
        profile="t", benchset="b", entries=["arts_val_wb"],
        apps={"nqueens": [Version.BASE]}, node_counts=[8],
    )
    with pytest.raises(ValueError, match="not in profile"):
        sel.validate_against(plane, catalog, profile)


def test_cell_count_is_the_product_of_the_three_surfaces():
    sel = Selection(
        profile="t", benchset="b", entries=["arts_val_wb", "xsocr"],
        apps={"nqueens": [Version.BASE, Version.HINTED]},
        node_counts=[1, 2], repeats=3,
    )
    assert sel.cell_count == 2 * 2 * 2 * 3


def test_a_benchset_that_names_applications_defines_the_roster():
    # A short benchset is a short campaign, not the whole catalog with three
    # entries annotated.
    catalog = load_catalog()
    bs = Benchset(name="small", apps={"nqueens": BenchsetEntry()})
    names = {a.name for a in bs.resolve(catalog)}
    assert names == {"nqueens"}


def test_an_empty_benchset_defers_to_the_catalog_defaults():
    catalog = load_catalog()
    names = {a.name for a in Benchset(name="empty").resolve(catalog)}
    assert names == {a.name for a in catalog.rows if a.default_enabled}


# --- ports and hosts ------------------------------------------------------
def test_a_local_run_states_a_connection_count_but_no_ports():
    profile = Profile.model_validate(_local(port_count=4))
    assert profile.port_count == 4
    assert profile.ports == []


def test_a_remote_port_list_must_match_the_connection_count():
    with pytest.raises(ValidationError, match="must name exactly that many"):
        Profile.model_validate(_local(
            launcher="slurm", ports=[25000], port_count=2,
            slurm={},
        ))


def test_a_matching_port_list_is_accepted():
    profile = Profile.model_validate(_local(
        launcher="slurm", ports=[25000, 25001], port_count=2,
        slurm={},
    ))
    assert len(profile.ports) == profile.port_count


def test_ssh_hosts_must_number_exactly_the_node_budget():
    with pytest.raises(ValidationError, match="name one host per node"):
        Profile.model_validate(_local(
            launcher="ssh", ports=[25000],
            ssh={"budget": 4, "hosts": ["n01", "n02"]},
        ))


def test_ssh_budget_below_the_widest_cell_is_refused():
    with pytest.raises(ValidationError, match="nowhere to run"):
        Profile.model_validate(_local(
            launcher="ssh", nodes=[1, 4], ports=[25000],
            ssh={"budget": 2, "hosts": ["n01", "n02"]},
        ))


def test_a_consistent_ssh_profile_is_accepted():
    profile = Profile.model_validate(_local(
        launcher="ssh", nodes=[1, 2, 4], ports=[25000],
        ssh={"budget": 4, "hosts": ["n01", "n02", "n03", "n04"]},
    ))
    assert profile.hosts == ["n01", "n02", "n03", "n04"]


def test_columns_group_by_write_policy_then_split_by_release():
    plane = load_plane()
    labelled = [(plane.write_label(w), plane.release_label(r))
                for r, w in plane.columns()]
    assert labelled == [
        ("Write Through", "Purge"), ("Write Through", "Retain"),
        ("Write Back", "Purge"), ("Write Back", "Retain"),
    ]


def test_the_write_policy_is_spelled_out():
    plane = load_plane()
    assert plane.write_label(Write.WT) == "Write Through"
    assert plane.write_label(Write.WB) == "Write Back"


def test_the_policies_read_as_words_and_the_families_as_acronyms():
    plane = load_plane()
    assert plane.release_label(Release.PURGE) == "Purge"
    assert plane.release_label(Release.RETAIN) == "Retain"
    assert plane.family_labels[Family.EXCL] == "EXCL"


# --- application vs microbenchmark ---------------------------------------
def test_the_catalog_separates_applications_from_microbenchmarks():
    catalog = load_catalog()
    apps = catalog.rows_of(Kind.APPLICATION)
    micro = catalog.rows_of(Kind.MICROBENCH)
    assert apps and micro
    assert len(apps) + len(micro) == len(catalog.rows)


def test_no_microbenchmark_is_enabled_by_default():
    # A microbenchmark is a regression check, not something a comparison is
    # claimed over, so a fresh campaign does not pick one up.
    catalog = load_catalog()
    assert not [a for a in catalog.rows_of(Kind.MICROBENCH) if a.default_enabled]


def test_the_suite_core_is_made_of_applications():
    catalog = load_catalog()
    for name in ("graph500", "hpcg_intel", "CoMD_sdsc2", "quicksort",
                 "nekbone", "hpgmg", "npb_cg", "cholesky_blas"):
        assert catalog.apps[name].kind is Kind.APPLICATION, name


def test_fixtures_and_library_drivers_are_microbenchmarks():
    # Upstream files these under kernels/ or examples/, or their own README
    # calls them a driver for one library.
    catalog = load_catalog()
    for name in ("printf", "testlibs", "basicIO", "highbw", "prodcon",
                 "dbctrl", "reduction_intel", "xeonNumaSize"):
        assert catalog.apps[name].kind is Kind.MICROBENCH, name


def test_an_idiom_study_series_is_a_mechanism_probe():
    # The david stencil1D set shares a directory with the PRK Stencil ports
    # but neither the kernel nor the provenance: its own README presents it
    # as a comparison of event-passing styles on a hand-written solver, and
    # its default arguments were tuned to CI execution time, not to a
    # problem anyone cites — that is exercising one runtime mechanism, not
    # a variant of the published benchmark.
    catalog = load_catalog()
    for name in ("stencil1D_once", "stencil1D_oncePI", "stencil1D_sticky",
                 "stencil1D_stickyLG", "stencil1D_guid", "stencil1D_guidPI",
                 "stencil1D_channel"):
        entry = catalog.apps[name]
        assert entry.kind is Kind.MICROBENCH, name
        assert not entry.default_enabled, name


def test_operations_named_after_themselves_are_microbenchmarks():
    # "globalsum" is the name of a sum, not of a benchmark anyone cites.
    catalog = load_catalog()
    for name in ("globalsum_cgShim", "globalsum_cgNoShim", "globalsum_pcg",
                 "curvefit", "dbcreate_matrix"):
        assert catalog.apps[name].kind is Kind.MICROBENCH, name


def test_cited_benchmarks_are_applications():
    catalog = load_catalog()
    for name in ("graph500", "hpcg_intel", "CoMD_sdsc2", "XSBench_intel",
                 "stream", "nekbone", "smithwaterman", "npb_cg"):
        assert catalog.apps[name].kind is Kind.APPLICATION, name


def test_every_catalog_entry_states_where_it_came_from():
    catalog = load_catalog()
    missing = [a.name for a in catalog.apps.values() if not a.provenance]
    assert not missing, f"no provenance for: {', '.join(sorted(missing))}"


def test_sar_is_one_application_and_its_restructured_tier():
    # sar_tiny/small/medium/large were rungs of one program's shipped
    # parameter ladder, carried as if they were different applications.  A
    # rung is an argument, so the roster keeps the one row and gives it the
    # restructured tier instead.
    catalog = load_catalog()
    for rung in ("sar_tiny", "sar_small", "sar_medium", "sar_large"):
        assert rung not in catalog.apps, f"{rung} is a rung, not an application"
    assert "sar_pss" in catalog.apps
    assert catalog.apps["sar_dist"].restructured_from == "sar_pss"
    # Both rows answer for themselves: each runs a rung of its own, so each
    # carries its own pinned answer.
    for name in ("sar_pss", "sar_dist"):
        assert catalog.apps[name].expect, f"{name} has no pinned answer"


# --- continuing a run -----------------------------------------------------
def _track(run_dir, rows):
    import json as _json

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "track.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in rows) + "\n"
    )


def test_a_continuation_carries_the_earlier_results_into_the_report(tmp_path):
    # Consensus is a vote across the configurations that ran one application,
    # so a report covering only the cells run after the interruption would be
    # voting with half a ballot.
    from artsrun.campaign import recorded_results
    from artsrun.run.types import Status

    class FakeCell:
        def __init__(self, key):
            self.key = key
            self.log_name = f"{key}.log"
            self.slug = key

    cells = [FakeCell("a"), FakeCell("b"), FakeCell("c")]
    _track(tmp_path, [
        {"event": "finished", "cell": "a", "status": "ok", "rc": 0, "wall_s": 1.5},
        {"event": "finished", "cell": "b", "status": "fail", "rc": 1, "wall_s": 0.5},
        {"event": "submitted", "cell": "c", "status": "submitted"},
    ])
    got = {r.cell.key: r for r in recorded_results(tmp_path, cells)}
    assert set(got) == {"a", "b"}          # "c" never finished
    assert got["a"].status is Status.OK and got["a"].wall_s == 1.5
    assert got["b"].status is Status.FAIL


def test_a_later_attempt_supersedes_the_earlier_one(tmp_path):
    from artsrun.campaign import recorded_results
    from artsrun.run.types import Status

    class FakeCell:
        def __init__(self, key):
            self.key = key
            self.log_name = f"{key}.log"
            self.slug = key

    _track(tmp_path, [
        {"event": "finished", "cell": "a", "status": "fail", "rc": 1},
        {"event": "finished", "cell": "a", "status": "ok", "rc": 0},
    ])
    got = recorded_results(tmp_path, [FakeCell("a")])
    assert len(got) == 1 and got[0].status is Status.OK


def test_a_run_with_no_selection_is_not_offered_for_continuing(tmp_path, monkeypatch):
    # Without it there is nothing to measure against, so it is not a campaign
    # anyone can carry on.
    import artsrun.campaign as mod

    monkeypatch.setattr(mod, "logs_root", lambda: tmp_path)
    (tmp_path / "20260101-000000").mkdir()
    _track(tmp_path / "20260101-000000", [
        {"event": "finished", "cell": "a", "status": "ok"},
    ])
    assert mod.past_runs() == []
