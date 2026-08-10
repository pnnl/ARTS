#!/usr/bin/env python3
"""rwmix node sweep: sharer-count axis at fixed read/write mix (M0 pre-sim).

Nodes 1..24 on one host, ONE worker (+1 progress) per node so
node_count x 2 threads = 48 cores exactly.  Per node count, runs W=0
(pure-reader baseline) and W=N/2 (the 50:50 mix), so per-write cost can
be computed as (T(N/2) - T(0)) / (N/2).  A third slice, W=N (writer-only), isolates
protocol tax from implementation overhead: invalidation has no copies to
chase without readers, so only the MIXED slice should grow with nodes if
the blame lies with the protocol.

Thread budget per rank (2 threads each): arts = 1 worker + 1 progress;
xsocr = 2 workers (comm inline on workers); ocr-vx = 2 TBB threads.
Generated cfgs land in the run's log dir.
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

ARTS_SUFFIXES = ["ocr_excl_purge", "ocr_excl_retain"]
RUNTIMES = [("arts", s) for s in ARTS_SUFFIXES]

NODES = [1, 2, 4, 8, 16, 24]
N_TASKS = 2048
E_US = 50
BYTES = 1024
ITERS = 3


def gen_cfgs(outdir: Path):
    """1-worker arts cfg per node count + one 2-worker xsocr cfg."""
    arts_tpl = (REPO / "configs" / "local" / "cbgpu02" / "2n_sc.cfg").read_text()
    cfgs = {}
    for n in NODES:
        txt = arts_tpl.replace("worker_threads=11", "worker_threads=1")
        txt = txt.replace("node_count=2", f"node_count={n}")
        p = outdir / f"arts_{n}n_1w.cfg"
        p.write_text(txt)
        cfgs[("arts", n)] = p
    xs_tpl = (REPO / "configs" / "mpi" / "cbgpu02" / "2n_sc.cfg").read_text()
    xs = xs_tpl.replace("0-11", "0-1").replace("1-11", "1-1")
    xp = outdir / "xsocr_2w.cfg"
    xp.write_text(xs)
    cfgs["xsocr"] = xp
    return cfgs


def run_cell(runner, cfgs, kind, suffix, nodes, args, timeout):
    name = f"rwmixns_{kind}{('_' + suffix) if suffix else ''}_{nodes}n"
    if kind == "arts":
        if nodes == 1:
            return runner.run_ocr(name, "rwmix", args, "arts", suffix=suffix,
                                  cfg_path=cfgs[("arts", 1)], timeout=timeout)
        return runner.run_arts_mn(name, "rwmix", args, nodes, suffix=suffix,
                                  cfg_path=cfgs[("arts", nodes)], timeout=timeout)
    if kind == "xsocr":
        if nodes == 1:
            return runner.run_ocr(name, "rwmix", args, "xsocr", width=2,
                                  cfg_path=cfgs["xsocr"], timeout=timeout)
        return runner.run_xsocr_mpi(name, "rwmix", args, nodes,
                                    cfg_path=cfgs["xsocr"], tpn=2, timeout=timeout)
    return runner.run_ocrvx_mpi(name, "rwmix", args, np=nodes, tpn=2, tbb=2,
                                timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build_release")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    build = REPO / args.build_dir
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = REPO / "logs" / "micro" / f"excl_contention_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    track = outdir / "track.jsonl"
    cfgs = gen_cfgs(outdir)
    runner = Runner(mem_gb=0, timeout=args.timeout, logdir=outdir,
                    target="cbgpu02", build=build)

    total = len(NODES) * len(RUNTIMES) * 3 * ITERS
    done = 0
    for nodes in NODES:
        # N divisible by 2*nodes so reader/writer counts split exactly evenly
        per_half = max(1, round(N_TASKS / (2 * nodes)))
        n_tasks = 2 * per_half * nodes
        w_values = [0, n_tasks // 2, n_tasks]  # reader-only / 50:50 / writer-only
        for kind, suffix in RUNTIMES:
            for w in w_values:
                bench_args = [str(n_tasks), str(w), str(E_US), str(BYTES)]
                for it in range(ITERS):
                    r = run_cell(runner, cfgs, kind, suffix, nodes, bench_args,
                                 args.timeout)
                    m = RE_OUT.search(r.stdout or "")
                    rec = {"kind": kind, "suffix": suffix, "nodes": nodes,
                           "n": n_tasks, "w": w, "iter": it, "rc": r.rc,
                           "elapsed_us": int(m.group(6)) if m else None,
                           "pds": int(m.group(1)) if m else None}
                    with open(track, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    done += 1
                    print(f"[{done}/{total}] {rec}", flush=True)
    print(f"DONE -> {track}")


if __name__ == "__main__":
    main()
