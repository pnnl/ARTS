# benchmarks/scripts/test_configs.py
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]

def _kv(p):
    d = {}
    for line in Path(p).read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#") and not line.startswith("["):
            k, v = line.split("=", 1); d[k.strip()] = v.strip()
    return d

def test_sc_geometry():
    g = REPO / "configs/local/cbgpu02"
    a = _kv(g / "1n_sc.cfg"); assert a["worker_threads"] == "12"
    for n in ("2n_sc","4n_sc"):
        c = _kv(g / f"{n}.cfg")
        assert c["worker_threads"] == "11" and c["progress_threads"] == "1"
    assert _kv(g / "4n_sc.cfg")["node_count"] == "4"  # 4*(11+1)=48

def test_perf_counters_cluster_and_interval():
    c = _kv(REPO / "configs/perf_counters.cfg")
    assert c["counter_capture_interval"] == "250"
    # End-to-end time is the "[E2E]" stderr marker, not a counter, so TIME_TOTAL
    # is intentionally absent here. Only the per-rank coherence metrics remain.
    for key in ("NUM_EDT_CREATE","NUM_DB_CREATE","BYTES_REMOTE_SENT",
                "NUM_DB_ACQUIRE_READ","NUM_OWNER_UPDATE_PERFORMED"):
        assert key in c, f"{key} missing"
    assert "TIME_TOTAL" not in c and "TIME_INIT" not in c
