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
    arts = plane.entry("ocr_val_wb")
    assert arts.binary("nqueens", hinted=False) == "nqueens_arts_ocr_val_wb"
    assert arts.binary("nqueens", hinted=True) == "nqueens_opt_arts_ocr_val_wb"
    assert plane.entry("xsocr").binary("nqueens", hinted=False) == "nqueens_xsocr"
    assert plane.entry("ocrvx").kind is RuntimeKind.OCRVX


# --- catalog --------------------------------------------------------------
def test_catalog_rows_exclude_rewrites():
    catalog = load_catalog()
    rewrites = {a.name for a in catalog.apps.values() if a.restructured_from}
    assert rewrites
    assert not rewrites & {a.name for a in catalog.rows}


def test_dist_version_resolves_to_the_rewrite_target():
    catalog = load_catalog()
    source, stem = catalog.resolve("quicksort", Version.RESTRUCTURED)
    assert stem == catalog.apps["quicksort_dist"].binary
    assert source.name == "quicksort_dist"


def test_hinted_version_resolves_to_the_opt_target():
    catalog = load_catalog()
    _, stem = catalog.resolve("nqueens", Version.HINTED)
    assert stem.endswith("_opt")


def test_hinted_is_refused_where_the_source_has_no_hint_layer():
    catalog = load_catalog()
    assert not catalog.apps["graph500"].hinted
    with pytest.raises(KeyError):
        catalog.resolve("graph500", Version.HINTED)


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


def test_slurm_budget_below_the_widest_cell_is_refused():
    # A cell occupies its whole node count at once, so a smaller budget never
    # admits the widest one — it would sit in the queue forever.
    with pytest.raises(ValidationError, match="below the widest node count"):
        Profile.model_validate(_local(
            launcher="slurm", nodes=[4, 8], ports=[25000],
            slurm={"budget": 4},
        ))


def test_a_budget_equal_to_the_widest_cell_is_enough():
    profile = Profile.model_validate(_local(
        launcher="slurm", nodes=[1, 2, 4, 8], ports=[25000],
        slurm={"budget": 8},
    ))
    assert profile.slurm.budget == profile.max_nodes


# --- benchset -------------------------------------------------------------
def test_benchset_falls_through_to_the_catalog():
    catalog = load_catalog()
    resolved = {a.key: a for a in Benchset(name="empty").resolve(catalog)}
    assert resolved["nqueens:asborn"].args == catalog.apps["nqueens"].args
    assert not resolved["nqueens:asborn"].args_overridden


def test_benchset_override_marks_the_argument_source():
    catalog = load_catalog()
    bs = Benchset(name="o", apps={"nqueens": BenchsetEntry(args=["8", "2"])})
    resolved = {a.key: a for a in bs.resolve(catalog)}
    assert resolved["nqueens:asborn"].args == ["8", "2"]
    assert resolved["nqueens:asborn"].args_overridden


def test_benchset_disable_removes_every_version():
    catalog = load_catalog()
    bs = Benchset(name="o", apps={"nqueens": BenchsetEntry(enabled=False)})
    assert not [a for a in bs.resolve(catalog) if a.name == "nqueens"]


def test_asking_for_a_version_an_application_lacks_is_an_error():
    catalog = load_catalog()
    bs = Benchset(name="o", apps={"graph500": BenchsetEntry(versions=[Version.HINTED])})
    with pytest.raises(ValueError, match="only has"):
        bs.resolve(catalog)


# --- selection ------------------------------------------------------------
def test_selection_rejects_a_node_count_outside_the_profile_sweep():
    plane, catalog = load_plane(), load_catalog()
    profile = Profile.model_validate(_local())
    sel = Selection(
        profile="t", benchset="b", entries=["ocr_val_wb"],
        apps={"nqueens": [Version.ASBORN]}, node_counts=[8],
    )
    with pytest.raises(ValueError, match="not in profile"):
        sel.validate_against(plane, catalog, profile)


def test_cell_count_is_the_product_of_the_three_surfaces():
    sel = Selection(
        profile="t", benchset="b", entries=["ocr_val_wb", "xsocr"],
        apps={"nqueens": [Version.ASBORN, Version.HINTED]},
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
            slurm={"budget": 2},
        ))


def test_a_matching_port_list_is_accepted():
    profile = Profile.model_validate(_local(
        launcher="slurm", ports=[25000, 25001], port_count=2,
        slurm={"budget": 2},
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
                 "nekbone", "hpgmg", "npb_cg", "cholesky"):
        assert catalog.apps[name].kind is Kind.APPLICATION, name


def test_fixtures_and_library_drivers_are_microbenchmarks():
    # Upstream files these under kernels/ or examples/, or their own README
    # calls them a driver for one library.
    catalog = load_catalog()
    for name in ("printf", "testlibs", "basicIO", "highbw", "prodcon",
                 "dbctrl", "reduction_intel", "xeonNumaSize"):
        assert catalog.apps[name].kind is Kind.MICROBENCH, name


def test_a_variant_of_a_published_application_is_still_an_application():
    # Being a duplicate of a sibling is a reason to leave it off a campaign,
    # not a reason to call it a mechanism probe.
    catalog = load_catalog()
    for name in ("stencil1D_sticky", "stream_org", "miniAMR_forkbomb"):
        entry = catalog.apps[name]
        assert entry.kind is Kind.APPLICATION, name
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


def test_the_whole_sar_size_ladder_is_present():
    # The catalog was seeded from the perf list, which carried only the
    # runtime-input variant; the compiled-in sizes live in the suite too.
    catalog = load_catalog()
    for name in ("sar_tiny", "sar_small", "sar_medium", "sar_large", "sar_pss"):
        assert name in catalog.apps, name
    # The compiled-in sizes have a fixed answer; the runtime-input variant
    # takes its problem size from the arguments, so it has none.
    for name in ("sar_tiny", "sar_small", "sar_medium", "sar_large"):
        assert catalog.apps[name].expect, f"{name} has no pinned answer"
