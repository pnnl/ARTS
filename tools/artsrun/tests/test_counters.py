"""Counter selection: what the runtime defines, what a set turns on, and the
file a build is configured against."""

from __future__ import annotations

import pathlib
import re

import pytest
from pydantic import ValidationError

from artsrun.model.counters import (
    Counterset, CounterSetting, Level, Mode, Reduce, load_counter_catalog,
)
from artsrun.paths import repo_root
from artsrun.render import render_counters


def test_the_catalog_mirrors_the_runtime_declaration_list():
    # counter.h's X-macro list is the authority; a name only the driver knows
    # would be silently ignored by the build's parser.
    header = (repo_root() /
              "libs/include/internal/arts/counter/counter.h").read_text()
    body = header.split("ARTS_COUNTER_LIST", 1)[1].split("// Generate enum")[0]
    declared = set(re.findall(r"X\(([A-Z0-9_]+)\)", body))
    assert set(load_counter_catalog().names) == declared


def test_every_counter_states_a_group_and_a_unit():
    for info in load_counter_catalog().counters.values():
        assert info.group
        assert info.unit


def test_a_set_may_not_name_a_counter_the_runtime_lacks():
    with pytest.raises(ValidationError, match="not counters this runtime"):
        Counterset(name="x", counters={"NUM_INVENTED": CounterSetting()})


def test_an_empty_set_compiles_nothing_in():
    assert Counterset(name="none").enabled == []


def test_the_rendered_file_lists_every_counter_including_the_off_ones():
    # The build's parser leaves an unmentioned counter at its own default, so
    # a set states what it turns off as plainly as what it turns on.
    cset = Counterset(name="one", counters={
        "NUM_EDT_FINISH": CounterSetting(mode=Mode.PERIODIC,
                                         level=Level.CLUSTER),
    })
    lines = [l for l in render_counters(cset).splitlines()
             if l and not l.startswith("#")]
    assert len(lines) == len(load_counter_catalog().names)
    assert "NUM_EDT_FINISH=PERIODIC,CLUSTER,SUM" in lines
    assert "NUM_EDT_CREATE=OFF" in lines


def test_the_rendered_syntax_is_what_the_build_parses():
    # Mirrors the regex in libs/include/internal/arts/counter/CMakeLists.txt.
    pattern = re.compile(
        r"^([A-Za-z0-9_]+)[ \t]*=[ \t]*(OFF|ONCE|PERIODIC)"
        r"(,[ \t]*(THREAD|NODE|CLUSTER))?(,[ \t]*(SUM|MAX|MIN|MASTER))?$"
    )
    cset = Counterset(name="mixed", counters={
        "TIME_EDT_EXEC": CounterSetting(mode=Mode.ONCE, level=Level.THREAD,
                                        reduce=Reduce.MAX),
        "NUM_EDT_FINISH": CounterSetting(mode=Mode.PERIODIC,
                                         level=Level.CLUSTER),
    })
    for line in render_counters(cset).splitlines():
        if line and not line.startswith("#"):
            assert pattern.match(line), line


def test_the_shipped_sets_load_and_enable_something():
    from artsrun import store

    for name in store.list_countersets():
        cset = store.load_counterset(name)
        assert cset.enabled, f"{name} turns nothing on"
        assert cset.capture_interval >= 1


def _tree_with_counters(build_dir, settings):
    """A build tree carrying the header the configure step would generate."""
    header = build_dir / "libs/include/internal/arts/counter/Preamble.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    modes = {"OFF": 0, "ONCE": 1, "PERIODIC": 2}
    levels = {"THREAD": 0, "NODE": 1, "CLUSTER": 2}
    reducers = {"SUM": 0, "MAX": 1, "MIN": 2, "MASTER": 3}
    out = []
    for name, (mode, level, reduce) in settings.items():
        out.append(f"#define ENABLE_{name} {0 if mode == 'OFF' else 1}")
        out.append(f"#define COUNTER_MODE_{name} {modes[mode]}")
        out.append(f"#define COUNTER_LEVEL_{name} {levels[level]}")
        out.append(f"#define COUNTER_REDUCE_{name} {reducers[reduce]}")
    header.write_text("\n".join(out) + "\n")
    return build_dir


def test_what_a_tree_compiled_is_read_from_its_generated_header(tmp_path):
    # Not from the file it was configured against: a set is values that get
    # rendered afresh per campaign, so the file they land in is incidental.
    from artsrun.build import compiled_counters

    build_dir = _tree_with_counters(tmp_path / "build", {
        "NUM_EDT_CREATE": ("PERIODIC", "CLUSTER", "SUM"),
        "NUM_DB_CREATE": ("OFF", "NODE", "SUM"),
    })
    assert compiled_counters(build_dir) == {
        "NUM_EDT_CREATE": ("PERIODIC", "CLUSTER", "SUM"),
        "NUM_DB_CREATE": ("OFF", "NODE", "SUM"),
    }


def test_a_tree_already_carrying_the_set_needs_no_reconfigure(tmp_path):
    from artsrun.build import counter_mismatch
    from artsrun.model.counters import Counterset, CounterSetting, Level, Mode, Reduce

    build_dir = _tree_with_counters(tmp_path / "build", {
        "NUM_EDT_CREATE": ("PERIODIC", "CLUSTER", "SUM"),
    })
    same = Counterset(name="x", counters={"NUM_EDT_CREATE": CounterSetting(
        mode=Mode.PERIODIC, level=Level.CLUSTER, reduce=Reduce.SUM)})
    assert counter_mismatch(build_dir, same) == []


def test_a_tree_carrying_other_counters_is_named_counter_by_counter(tmp_path):
    # What has to change is a counter, not a path, so that is what is reported.
    from artsrun.build import counter_mismatch
    from artsrun.model.counters import Counterset, CounterSetting, Level, Mode

    build_dir = _tree_with_counters(tmp_path / "build", {
        "NUM_EDT_CREATE": ("PERIODIC", "CLUSTER", "SUM"),
        "NUM_DB_CREATE": ("OFF", "NODE", "SUM"),
    })
    other = Counterset(name="x", counters={
        "NUM_DB_CREATE": CounterSetting(mode=Mode.ONCE, level=Level.NODE),
    })
    # NUM_EDT_CREATE is on in the tree and unnamed by the set; NUM_DB_CREATE
    # differs in mode.  Both have to move.
    assert counter_mismatch(build_dir, other) == ["NUM_DB_CREATE", "NUM_EDT_CREATE"]


def test_the_sampling_interval_reaches_the_runtime_configuration():
    # Interval and folder are runtime keys, so they move without a rebuild.
    from artsrun.render import render_arts
    from artsrun.store import load_profile

    text = render_arts(load_profile("bentley"), 2,
                       counter_folder="/runs/c", capture_interval=250)
    assert "counter_capture_interval=250" in text
    assert "counter_folder=/runs/c" in text
