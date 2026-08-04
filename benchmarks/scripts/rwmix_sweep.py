#!/usr/bin/env python3
"""rwmix writer-count sweep across runtimes (M0 pre-measurement).

Runs the rwmix probe (racy single-DB, N gated tasks, W writers uniformly
interleaved) over runtimes x node counts x W values on the local host,
reusing the harness Runner so cfg/port/pinning discipline is identical to
the perf harness (_sc geometry: 12 workers per node).

Progress and results append to a track file on repo disk so the sweep
survives session loss.  Usage:
    python3 benchmarks/scripts/rwmix_sweep.py --build-dir build_release_ocr_val_wb
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_common import Runner, REPO  # noqa: E402

RE_OUT = re.compile(
    r"RWMIX PDS=(\d+) N=(\d+) W=(\d+) E_US=(\d+) BYTES=(\d+) ELAPSED_US=(\d+)")

ARTS_SUFFIXES = ["ocr_val_wt", "ocr_val_wb",
                 "ocr_excl_purge", "ocr_excl_retain"]
RUNTIMES = [("xsocr", None), ("ocrvx", None)] + [("arts", s) for s in ARTS_SUFFIXES]

W_VALUES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 512, 2048]
NODES = [1, 2, 4]
N_TASKS = 2048
E_US = 50
BYTES = 1024
ITERS = 2


def run_cell(runner: Runner, kind: str, suffix, nodes: int, args: list[str],
             timeout: int):
    cfg_base = REPO / "configs"
    name = f"rwmix_{kind}{('_' + suffix) if suffix else ''}_{nodes}n"
    if nodes == 1:
        if kind == "arts":
            return runner.run_ocr(name, "rwmix", args, "arts", suffix=suffix,
                                  cfg_path=cfg_base / "local" / "cbgpu02" / "1n_sc.cfg",
                                  timeout=timeout)
        if kind == "xsocr":
            return runner.run_ocr(name, "rwmix", args, "xsocr", width=12,
                                  cfg_path=cfg_base / "mpi" / "cbgpu02" / "1n_sc.cfg",
                                  timeout=timeout)
        return runner.run_ocrvx_mpi(name, "rwmix", args, tbb=12, timeout=timeout)
    if kind == "arts":
        return runner.run_arts_mn(name, "rwmix", args, nodes, suffix=suffix,
                                  cfg_path=cfg_base / "local" / "cbgpu02" / f"{nodes}n_sc.cfg",
                                  timeout=timeout)
    if kind == "xsocr":
        return runner.run_xsocr_mpi(name, "rwmix", args, nodes,
                                    cfg_path=cfg_base / "mpi" / "cbgpu02" / f"{nodes}n_sc.cfg",
                                    tpn=12, timeout=timeout)
    return runner.run_ocrvx_mpi(name, "rwmix", args, np=nodes, tpn=12, tbb=12,
                                timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build_release_ocr_val_wb")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    build = REPO / args.build_dir
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = REPO / "benchmarks" / "scripts" / "logs" / f"rwmix_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    track = outdir / "track.jsonl"
    runner = Runner(mem_gb=0, timeout=args.timeout, logdir=outdir,
                    target="cbgpu02", build=build)

    total = len(NODES) * len(RUNTIMES) * len(W_VALUES) * ITERS
    done = 0
    for nodes in NODES:
        for kind, suffix in RUNTIMES:
            for w in W_VALUES:
                bench_args = [str(N_TASKS), str(w), str(E_US), str(BYTES)]
                for it in range(ITERS):
                    r = run_cell(runner, kind, suffix, nodes, bench_args,
                                 args.timeout)
                    m = RE_OUT.search(r.stdout or "")
                    rec = {
                        "kind": kind, "suffix": suffix, "nodes": nodes,
                        "w": w, "iter": it, "rc": r.rc,
                        "elapsed_us": int(m.group(6)) if m else None,
                        "pds": int(m.group(1)) if m else None,
                    }
                    with open(track, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    done += 1
                    print(f"[{done}/{total}] {rec}", flush=True)
    print(f"DONE -> {track}")


if __name__ == "__main__":
    main()
