import harness_common as hc


def test_geometry_server():
    assert hc.MN_RANKS_for('cbgpu02') == [2, 4]
    assert hc._TPN_for('cbgpu02') == {2: 12, 4: 12}
    assert hc._NCORES_for('cbgpu02') == 48


def test_pin_wrap_shape():
    w = hc.pin_wrap(6)
    assert "PMI_RANK" in w and "taskset -c" in w


def test_runtimes_nine():
    keys = [r.key for r in hc.RUNTIMES]
    assert keys == ["mrnew_eager", "mrnew_lazy", "mrsw_eager", "mrsw_lazy",
                     "mrmw", "lock_eager", "lock_lazy", "xsocr", "ocrvx"]


import inspect


def test_run_ocrvx_mpi_accepts_sc_overrides():
    sig = inspect.signature(hc.Runner.run_ocrvx_mpi)
    assert "tpn" in sig.parameters and "tbb" in sig.parameters
    assert sig.parameters["tpn"].default is None
    assert sig.parameters["tbb"].default is None
