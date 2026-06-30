import harness_common as hc


def test_geometry_server():
    assert hc.MN_RANKS_for('server') == [2, 4, 8, 16, "2n_io", "4n_io", "8n_io"]
    assert hc._TPN_for('server') == {2: 24, 4: 12, 8: 6, 16: 3}
    assert hc._NCORES_for('server') == 48


def test_pin_wrap_shape():
    w = hc.pin_wrap(6)
    assert "PMI_RANK" in w and "taskset -c" in w


def test_runtimes_nine():
    keys = [r.key for r in hc.RUNTIMES]
    assert keys == ["mrnew_eager", "mrnew_lazy", "mrsw_eager", "mrsw_lazy",
                     "mrmw", "lock_eager", "lock_lazy", "xsocr", "ocrvx"]
