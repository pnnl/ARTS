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
    assert ph.EXPERIMENTS["fix"][0] == "1n"
    assert ph.EXPERIMENTS["strong"] == ph.EXPERIMENTS["weak"]
    assert "8n_sc" in ph.EXPERIMENTS["strong"]
    assert "16" in [str(x) for x in ph.EXPERIMENTS["fix"]]
