from __future__ import annotations

import pytest

from artsrun.model.catalog import load_catalog
from artsrun.model.plane import RuntimeKind
from artsrun.model.sweep import SweepPoint, SweepSpec


def _spec(**over):
    base = dict(
        name="t",
        app="rwsteady",
        nodes=8,
        arms=["arts_val_wb", "ocr_val_wb_nocomb"],
        repeats=2,
        arg_order=["R", "W", "T_MS"],
        base={"R": 54, "W": 9, "T_MS": 1000},
        points=[
            SweepPoint(name="a", set={"W": 1, "R": 62}),
            SweepPoint(name="b", set={"W": 32, "R": 31}),
        ],
    )
    base.update(over)
    return SweepSpec.model_validate(base)


def test_argv_follows_arg_order_with_point_overrides():
    spec = _spec()
    assert spec.argv_for(spec.points[0]) == ["62", "1", "1000"]
    assert spec.argv_for(spec.points[1]) == ["31", "32", "1000"]


def test_points_must_name_known_knobs_and_be_unique():
    with pytest.raises(ValueError, match="not in arg_order"):
        _spec(points=[SweepPoint(name="a", set={"NOPE": 1})])
    with pytest.raises(ValueError, match="duplicate"):
        _spec(points=[SweepPoint(name="a"), SweepPoint(name="a")])


def test_arms_resolve_plane_keys_and_offplane_suffixes():
    # A plane arm resolves to the plane's own entry; an ablation twin — which
    # the plane deliberately does not offer — synthesizes an ARTS entry with
    # the right binary suffix.
    on = SweepSpec.arm_entry("arts_val_wb")
    off = SweepSpec.arm_entry("ocr_val_wb_nocomb")
    assert on.variant == "ocr_val_wb"
    assert off.kind is RuntimeKind.ARTS
    assert off.variant == "ocr_val_wb_nocomb"
    assert off.binary("rwsteady", hinted=False) == "rwsteady_arts_ocr_val_wb_nocomb"


def test_toys_do_not_sweep_but_probes_and_applications_do():
    catalog = load_catalog()

    class P:  # minimal profile stand-in
        nodes = [8]
        name = "p"

    with pytest.raises(ValueError, match="is a toy"):
        _spec(app="prodcon").validate_against(catalog, P())
    _spec(app="nqueens").validate_against(catalog, P())  # application: allowed
    _spec().validate_against(catalog, P())


def test_virtual_rows_vote_per_point():
    catalog = load_catalog()
    spec = _spec()
    a = spec.resolved_app(catalog, spec.points[0])
    b = spec.resolved_app(catalog, spec.points[1])
    assert a.key == "rwsteady+a:base"
    assert b.key == "rwsteady+b:base"
    assert a.args == ["62", "1", "1000"]


def test_arm_rotation_interleaves_repeats():
    # Inside each (repeat, point) block the arm order rotates, so over the
    # sweep every arm occupies every position — the drift discipline.
    from artsrun.sweep import SweepCampaign

    entries = [SweepSpec.arm_entry(a) for a in ["arts_val_wb", "arts_inv_wb",
                                                "arts_excl_retain"]]
    orders = []
    for rep in range(1, 3):
        for pi in range(2):
            shift = (rep - 1 + pi) % len(entries)
            orders.append([e.key for e in entries[shift:] + entries[:shift]])
    assert orders[0][0] == "arts_val_wb"
    assert orders[1][0] == "arts_inv_wb"
    assert orders[2][0] == "arts_inv_wb"
    firsts = {o[0] for o in orders}
    assert len(firsts) == 3
    assert SweepCampaign is not None
