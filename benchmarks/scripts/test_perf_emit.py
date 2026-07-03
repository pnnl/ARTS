import csv
import performance_harness as ph
from harness_common import Runtime


def test_eligibility_io_refs():
    xs = Runtime("xsocr","xsocr","xsocr")
    ar = Runtime("mrnew_lazy","arts_mrnew_lazy","arts","mrnew_lazy")
    case = ph.PerfCase("graph500","graph500", r"nodes \d+",
                       fix_args=["6","8","1","1"], strong_args={1:["6","8","1","1"]},
                       weak_args={1:["6","8","1","1"]})
    assert ph.perf_runtime_eligible(ar, case, "2n_io") is True
    assert ph.perf_runtime_eligible(xs, case, "2n_io") is False   # refs skip _io
    assert ph.perf_runtime_eligible(xs, case, "4n_sc") is True    # refs run _sc

def test_write_results_csv(tmp_path):
    rows = [{"bench":"graph500","config":"1n","runtime":"xsocr","iter":0,
             "e2e_ns":123,"rc":0,"wall":1.2,"status":"OK"}]
    p = tmp_path / "results.csv"; ph.write_results_csv(rows, p)
    got = list(csv.DictReader(p.open()))
    assert got[0]["runtime"] == "xsocr" and got[0]["e2e_ns"] == "123"


def test_write_results_csv_has_experiment_column(tmp_path):
    rows = [{"experiment":"strong","bench":"graph500","config":"2n_sc",
             "runtime":"mrnew_lazy","iter":0,"e2e_ns":9,"rc":0,"wall":1.0,
             "status":"OK","kernel2_ns":3.5,"mteps":42.7}]
    p = tmp_path / "r.csv"; ph.write_results_csv(rows, p)
    got = list(csv.DictReader(p.open()))
    assert got[0]["experiment"] == "strong"
    assert got[0]["kernel2_ns"] == "3.5" and got[0]["mteps"] == "42.7"


def test_resume_append_and_load_done(tmp_path):
    # Append two cells, reload -> both recognized as done (per-program resume).
    import performance_harness as ph
    p = tmp_path / "results.csv"
    ph._append_result_row(p, {"experiment":"strong","bench":"graph500","config":"1n_sc",
                              "runtime":"mrnew_lazy","iter":0,"e2e_ns":5,"rc":0,"wall":5.0,
                              "status":"OK","kernel2_ns":1.2,"mteps":9.9})
    ph._append_result_row(p, {"experiment":"strong","bench":"hpcg_intel","config":"1n_sc",
                              "runtime":"xsocr","iter":0,"e2e_ns":9,"rc":0,"wall":9.0,
                              "status":"OK"})
    done = ph._load_done_cells(p)
    assert ("strong","graph500","1n_sc","mrnew_lazy") in done
    assert ("strong","hpcg_intel","1n_sc","xsocr") in done
    assert ("strong","graph500","2n_sc","mrnew_lazy") not in done
    # header written once; 2 data rows
    lines = p.read_text().strip().splitlines()
    assert lines[0].startswith("experiment,bench,config,runtime")
    assert len(lines) == 3


def test_load_done_missing_file(tmp_path):
    import performance_harness as ph
    assert ph._load_done_cells(tmp_path / "nope.csv") == set()
