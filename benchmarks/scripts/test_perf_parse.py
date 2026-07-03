import json
import performance_harness as ph


def test_parse_arts_counters(tmp_path):
    # Real counter.c schema: each counter is an object keyed by its name,
    # with a "value" field holding the final (ONCE) or reduced (CLUSTER)
    # uint64 -- NOT a "final" key. n{id}.json holds NODE/CLUSTER counters
    # for that rank; cluster.json holds the master's cross-node reduction.
    (tmp_path / "n0.json").write_text(json.dumps({
        "metadata": {"node_id": 0},
        "counters": {
            "NUM_EDT_CREATE": {"captureMode": "PERIODIC", "captureLevel": "CLUSTER",
                               "value": 100},
            "NUM_DB_CREATE": {"captureMode": "PERIODIC", "captureLevel": "CLUSTER",
                              "value": 10},
        },
    }))
    (tmp_path / "n1.json").write_text(json.dumps({
        "metadata": {"node_id": 1},
        "counters": {
            "NUM_EDT_CREATE": {"captureMode": "PERIODIC", "captureLevel": "CLUSTER",
                               "value": 80},
            "NUM_DB_CREATE": {"captureMode": "PERIODIC", "captureLevel": "CLUSTER",
                              "value": 12},
        },
    }))
    # e2e is no longer a counter (it is the [E2E] stderr marker), so
    # parse_arts_counters returns only per-rank coherence metrics.
    out = ph.parse_arts_counters(tmp_path)
    assert out["per_rank"][0]["NUM_EDT_CREATE"] == 100
    assert out["per_rank"][1]["NUM_EDT_CREATE"] == 80
    assert out["per_rank"][0]["NUM_DB_CREATE"] == 10


def test_parse_arts_counters_missing_files(tmp_path):
    # No n*.json at all -- must not raise.
    out = ph.parse_arts_counters(tmp_path)
    assert out["per_rank"] == {}


def test_parse_e2e_reference():
    assert ph.parse_reference_e2e("foo\n[E2E] 9876543\nbar") == 9876543.0
    assert ph.parse_reference_e2e("no marker") is None


def test_parse_extra_scalars_graph500():
    txt = "blah\n[kernel2 time 3.5]\nmean MTEPS 42.7\nnodes 64\n"
    es = {"kernel2_ns": r"\[kernel2 time ([0-9.eE+-]+)\]",
          "mteps": r"mean MTEPS ([0-9.eE+-]+)"}
    out = ph.parse_extra_scalars(txt, es)
    assert out["kernel2_ns"] == 3.5 and out["mteps"] == 42.7


def test_parse_extra_scalars_absent():
    out = ph.parse_extra_scalars("no markers here", {"k": r"k=([0-9]+)"})
    assert out["k"] is None
