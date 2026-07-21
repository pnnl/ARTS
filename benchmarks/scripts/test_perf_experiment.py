import performance_harness as ph


def _case():
    return ph.PerfCase(
        "x", "x", r"done",
        fix_args=["F"],
        strong_args={1: ["s1"], 2: ["s2"], 4: ["s4"], 8: ["s8"]},
        weak_args={1: ["w1"], 2: ["w2"], 4: ["w4"], 8: ["w8"]},
    )


def test_fix_returns_fixed_list_regardless_of_geo():
    c = _case()
    assert ph.select_perf_args(c, "fix", 1) == ["F"]
    assert ph.select_perf_args(c, "fix", 8) == ["F"]


def test_strong_keys_by_geo():
    assert ph.select_perf_args(_case(), "strong", 4) == ["s4"]


def test_weak_keys_by_geo():
    assert ph.select_perf_args(_case(), "weak", 8) == ["w8"]


def test_constant_list_strong_args_is_broadcast():
    # Category-B apps store a single list for strong (constant across nodes).
    c = ph.PerfCase("c", "c", r"d", fix_args=["F"],
                    strong_args=["S"], weak_args={1: ["w1"]})
    assert ph.select_perf_args(c, "strong", 2) == ["S"]
    assert ph.select_perf_args(c, "strong", 8) == ["S"]


def test_experiments_table_configs():
    assert ph.EXPERIMENTS["single"] == ["1n", "1n_sc"]
    assert ph.EXPERIMENTS["strong"] == ph.EXPERIMENTS["weak"]
    assert ph.EXPERIMENTS["strong"] == ["1n_sc", "2n_sc", "4n_sc"]


# --- unified BENCHES table + benches_for filtering ---

def test_benches_unified_and_named_uniquely():
    # Every app is one PerfCase; no duplicate names across the merged table.
    names = [b.name for b in ph.BENCHES]
    assert len(names) == len(set(names))
    assert isinstance(ph.BENCHES, list) and len(ph.BENCHES) > 0


def test_every_bench_has_a_scaling_decision():
    # Each app EITHER carries a weak series OR is a documented _WEAK_EXEMPT
    # entry -- never neither, never both.  (strong needs no decision: it
    # broadcasts fix_args wherever no designed table exists;
    # pure analysis metadata.)
    for b in ph.BENCHES:
        has_weak = b.weak_args is not None
        exempt = b.name in ph._WEAK_EXEMPT
        assert has_weak != exempt, b.name


def test_single_and_strong_run_every_app():
    # No ANALYSIS skip field gates a run; the only run gate is the curated
    # TOY_BENCHES exclusion (runtime-fixture micros are not benchmarks).
    expected = [b for b in ph.BENCHES if b.name not in ph.PERF_EXCLUDED]
    assert ph.benches_for("single") == expected
    assert ph.benches_for("strong") == expected


def test_weak_runs_every_app_except_documented():
    # weak runs EVERY app that has any weak series (designed dict, derived
    # axis, argv-ized repeat knob, or dataset ladder); the only allowed
    # exemptions are the documented _WEAK_EXEMPT entries.
    weak = ph.benches_for("weak")
    assert all(b.weak_args is not None for b in weak)
    names = {b.name for b in weak}
    exempt = ({b.name for b in ph.BENCHES} - names) - ph.PERF_EXCLUDED
    assert exempt == set(ph._WEAK_EXEMPT) == set()
    assert {"graph500", "hpcg_intel", "stream_dist"} <= names          # core
    assert {"fibonacci", "nqueens", "quicksort_dist"} <= names  # derived
    assert {"fft_dist", "triangle", "npb_cg", "sar_pss"} <= names


def test_non_core_app_broadcasts_fix_args_in_strong():
    # An app with no strong_args broadcasts its fix_args to every geo -- a fixed
    # workload = constant-total-work (strong) semantics.
    fib = next(b for b in ph.BENCHES if b.name == "fibonacci")
    assert fib.strong_args is None
    assert ph.select_perf_args(fib, "strong", 1) == fib.fix_args
    assert ph.select_perf_args(fib, "strong", 4) == fib.fix_args


def test_weak_derivation_scales_the_work_axis():
    # linear knob x2/x4 at geo 2/4; exponential/log knob +1/+2.
    red = next(b for b in ph.BENCHES if b.name == "reduction_intel")
    assert red.weak_args[1] == red.fix_args                       # 1n_sc == base
    assert ph.select_perf_args(red, "weak", 2)[2] == "40000"      # 20000 x2
    assert ph.select_perf_args(red, "weak", 4)[2] == "80000"      # 20000 x4
    fib = next(b for b in ph.BENCHES if b.name == "fibonacci")
    assert ph.select_perf_args(fib, "weak", 2) == ["34"]          # 33 +1
    assert ph.select_perf_args(fib, "weak", 4) == ["35"]          # 33 +2


def test_benches_for_only_filter():
    got = ph.benches_for("single", {"fibonacci", "hpcg_intel"})
    assert {b.name for b in got} == {"fibonacci", "hpcg_intel"}
    # Non-core names are NOT dropped from strong now (every app runs).
    assert {b.name for b in ph.benches_for("strong", {"fibonacci", "hpcg_intel"})} \
        == {"fibonacci", "hpcg_intel"}


def test_scaling_apps_use_sc_geo_keys():
    # Apps that DO carry designed strong/weak dicts key them by _sc geo (1,2,4).
    for b in [x for x in ph.BENCHES if isinstance(x.strong_args, dict)
              and isinstance(x.weak_args, dict)]:
        for d in (b.strong_args, b.weak_args):
            assert isinstance(d, dict) and set(d) == {1, 2, 4}, b.name
