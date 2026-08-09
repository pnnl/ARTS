#!/usr/bin/env python3
"""rwmix node sweep v2 (user-approved geometry, 2026-07-21).

Sharer-count axis: nodes 1..16, per-node EDT count FIXED at 128
(N = 128 x nodes), writer ratio slices {0%, 50%, 100%}, 3 iterations.

Per-rank budget = 3 cores, every runtime = 2 compute + 1 comm:
  arts : worker_threads=2 + progress_threads=1, runtime per-thread hard pin
  xsocr: workers 0-2 (id 0 = HC_COMM comm worker, 1-2 = HC compute),
         per-rank cfg with absolute `binding = 3r-3r+2` (hard pin) inside
         a matching per-rank taskset block
  ocrvx: OCRVX_NUM_THREADS=2 (TBB compute; comm threads separate) in a
         3-core taskset block (no internal pinning possible)
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_common import Runner, REPO, mpirun_prefix  # noqa: E402

RE_OUT = re.compile(
    r"RWMIX PDS=(\d+) N=(\d+) W=(\d+) E_US=(\d+) BYTES=(\d+) ELAPSED_US=(\d+)")

ARTS_SUFFIXES = ["ocr_val_wt", "ocr_val_wb",
                 "ocr_excl_purge", "ocr_excl_retain"]
RUNTIMES = [("xsocr", None), ("ocrvx", None)] + [("arts", s) for s in ARTS_SUFFIXES]

NODES = list(range(1, 17))
PER_NODE = 128          # EDTs per node (64 per compute worker)
E_US = 50
BYTES = 1024
ITERS = 3
CORES_PER_RANK = 3


def gen_cfgs(outdir: Path):
    cfgs = {}
    arts_tpl = (REPO / "configs" / "local" / "cbgpu02" / "2n_sc.cfg").read_text()
    for n in NODES:
        txt = arts_tpl.replace("worker_threads=11", "worker_threads=2")
        txt = txt.replace("node_count=2", f"node_count={n}")
        p = outdir / f"arts_{n}n_2w.cfg"
        p.write_text(txt)
        cfgs[("arts", n)] = p
    xs_tpl = (REPO / "configs" / "mpi" / "cbgpu02" / "2n_sc.cfg").read_text()
    xs = xs_tpl.replace("0-11", "0-2").replace("1-11", "1-2")
    assert "binding" not in xs, "template already carries a binding key"
    # insert an absolute per-rank binding right after CompPlatformInst0's body
    m = re.search(r"(\[CompPlatformInst0\][^\[]*?stacksize[^\n]*\n)", xs)
    assert m, "CompPlatformInst0 section not found in xsocr template"
    for r in range(max(NODES)):
        s, e = r * CORES_PER_RANK, (r + 1) * CORES_PER_RANK - 1
        txt = xs.replace(m.group(1), m.group(1) + f"    binding\t=\t{s}-{e}\n")
        p = outdir / f"xsocr_rank{r}.cfg"
        p.write_text(txt)
        cfgs[("xsocr", r)] = p
    return cfgs


def xsocr_wrap(outdir: Path, args: list[str]) -> str:
    """Per-rank taskset block + per-rank binding cfg, dispatched by PMI_RANK."""
    c = CORES_PER_RANK
    return (f"bash -c 'r=${{PMI_RANK:-0}}; s=$((r*{c})); e=$((s+{c}-1)); "
            f"exec taskset -c $s-$e ./rwmix_xsocr -ocr:cfg {outdir}/xsocr_rank$r.cfg "
            + " ".join(args) + "' _")


def run_cell(runner, cfgs, outdir, kind, suffix, nodes, args, timeout):
    name = f"rwmixv2_{kind}{('_' + suffix) if suffix else ''}_{nodes}n"
    if kind == "arts":
        if nodes == 1:
            return runner.run_ocr(name, "rwmix", args, "arts", suffix=suffix,
                                  cfg_path=cfgs[("arts", 1)], timeout=timeout)
        return runner.run_arts_mn(name, "rwmix", args, nodes, suffix=suffix,
                                  cfg_path=cfgs[("arts", nodes)], timeout=timeout)
    if kind == "xsocr":
        import os
        logfile = runner.logdir / f"{name}.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        launcher = xsocr_wrap(outdir, args)
        if nodes == 1:
            cmd = f"cd {runner.apps_dir} && timeout -k 1 {timeout} {launcher}"
        else:
            cmd = (f"cd {runner.apps_dir} && timeout -k 1 {timeout} "
                   f"{mpirun_prefix(nodes)} {launcher}")
        result = runner._run(cmd, env, logfile, wall_timeout=timeout)
        runner._reap_exe(runner.apps_dir / "rwmix_xsocr")
        return result
    # ocrvx: 2 TBB compute threads inside a 3-core block
    if nodes == 1:
        import os
        logfile = runner.logdir / f"{name}.log"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"
        env["OCRVX_NUM_THREADS"] = "2"
        cmd = (f"cd {runner.apps_dir} && timeout -k 1 {timeout * 3} "
               f"taskset -c 0-{CORES_PER_RANK - 1} ./rwmix_ocrvx " + " ".join(args))
        result = runner._run(cmd, env, logfile, wall_timeout=timeout * 3)
        runner._reap_exe(runner.apps_dir / "rwmix_ocrvx")
        return result
    return runner.run_ocrvx_mpi(name, "rwmix", args, np=nodes,
                                tpn=CORES_PER_RANK, tbb=2, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build_release")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    build = REPO / args.build_dir
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = REPO / "logs" / "micro" / f"rwmixv2_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    track = outdir / "track.jsonl"
    cfgs = gen_cfgs(outdir)
    runner = Runner(mem_gb=0, timeout=args.timeout, logdir=outdir,
                    target="cbgpu02", build=build)

    total = len(NODES) * len(RUNTIMES) * 3 * ITERS
    done = 0
    for nodes in NODES:
        n_tasks = PER_NODE * nodes
        for kind, suffix in RUNTIMES:
            for w in (0, n_tasks // 2, n_tasks):
                bench_args = [str(n_tasks), str(w), str(E_US), str(BYTES)]
                for it in range(ITERS):
                    r = run_cell(runner, cfgs, outdir, kind, suffix, nodes,
                                 bench_args, args.timeout)
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
