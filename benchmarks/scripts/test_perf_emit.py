import csv
import performance_harness as ph
from harness_common import Runtime


def test_eligibility_io_refs():
    xs = Runtime("xsocr","xsocr","xsocr")
    ar = Runtime("mrnew_lazy","arts_mrnew_lazy","arts","mrnew_lazy")
    case = ph.PerfCase("graph500","graph500",["6","8","1","1"], r"nodes \d+")
    assert ph.perf_runtime_eligible(ar, case, "2n_io") is True
    assert ph.perf_runtime_eligible(xs, case, "2n_io") is False   # refs skip _io
    assert ph.perf_runtime_eligible(xs, case, "4n_sc") is True    # refs run _sc

def test_write_results_csv(tmp_path):
    rows = [{"bench":"graph500","config":"1n","runtime":"xsocr","iter":0,
             "e2e_ns":123,"rc":0,"wall":1.2,"status":"OK"}]
    p = tmp_path / "results.csv"; ph.write_results_csv(rows, p)
    got = list(csv.DictReader(p.open()))
    assert got[0]["runtime"] == "xsocr" and got[0]["e2e_ns"] == "123"
