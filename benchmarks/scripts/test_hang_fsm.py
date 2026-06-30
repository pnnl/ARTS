import performance_harness as ph

def test_classify():
    assert ph.classify_run(rc=0, marker_seen=True, proc_alive=False) == "OK"
    assert ph.classify_run(rc=124, marker_seen=True, proc_alive=True) == "SHUTDOWN_HANG"
    assert ph.classify_run(rc=124, marker_seen=False, proc_alive=True) == "COMPUTE_FAIL"
    assert ph.classify_run(rc=139, marker_seen=False, proc_alive=False) == "COMPUTE_FAIL"
