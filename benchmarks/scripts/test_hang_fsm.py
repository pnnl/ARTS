import performance_harness as ph

def test_classify():
    assert ph.classify_run(rc=0, marker_seen=True, proc_alive=False) == "OK"
    assert ph.classify_run(rc=124, marker_seen=True, proc_alive=True) == "SHUTDOWN_HANG"
    assert ph.classify_run(rc=124, marker_seen=False, proc_alive=True) == "COMPUTE_FAIL"
    assert ph.classify_run(rc=139, marker_seen=False, proc_alive=False) == "COMPUTE_FAIL"
    # Teardown death (SIGSEGV/SIGKILL) AFTER the result printed: marker_seen
    # with a nonzero, non-124 rc and the process already exited (not "still
    # alive") -> the generic "OK" bucket, same as a clean rc==0 exit.
    assert ph.classify_run(rc=139, marker_seen=True, proc_alive=False) == "OK"
    assert ph.classify_run(rc=137, marker_seen=True, proc_alive=False) == "OK"


# --- finalize_status: e2e-marker gating on top of classify_run's verdict ---

def test_finalize_status_valid_despite_teardown_death():
    # (a) scalar+e2e present, rc=139 (SIGSEGV during teardown) -> valid OK.
    status = ph.classify_run(rc=139, marker_seen=True, proc_alive=False)
    assert ph.finalize_status(status, e2e_ns=123456.0) == "OK"


def test_finalize_status_ok_unchanged_for_clean_exit():
    # (b) scalar present, rc=0 -> OK, unaffected by the e2e gate.
    status = ph.classify_run(rc=0, marker_seen=True, proc_alive=False)
    assert ph.finalize_status(status, e2e_ns=999.0) == "OK"


def test_finalize_status_missing_scalar_stays_fail():
    # (c) no scalar at all, rc=139 -> COMPUTE_FAIL regardless of e2e presence
    # (a run missing the completion marker is never valid).
    status = ph.classify_run(rc=139, marker_seen=False, proc_alive=False)
    assert status == "COMPUTE_FAIL"
    assert ph.finalize_status(status, e2e_ns=123456.0) == "COMPUTE_FAIL"


def test_finalize_status_requires_e2e_marker():
    # An "OK"-shaped run (marker seen) whose e2e marker never printed is not
    # a valid measurement -- demoted to COMPUTE_FAIL so the caller retries it
    # (or drops it as FAIL once retries exhaust).
    assert ph.finalize_status("OK", None) == "COMPUTE_FAIL"
    assert ph.finalize_status("SHUTDOWN_HANG", None) == "COMPUTE_FAIL"
    assert ph.finalize_status("SHUTDOWN_HANG", 42.0) == "OK"
    assert ph.finalize_status("COMPUTE_FAIL", None) == "COMPUTE_FAIL"
