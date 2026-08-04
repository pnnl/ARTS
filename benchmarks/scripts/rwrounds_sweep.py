#!/usr/bin/env python3
"""rwrounds sweep: [1W -> n-1R] rounds, superadditivity test (M0 v4).
xsocr + ocrvx ONLY. Geometry = approved v2 (2 compute + 1 comm, pinned)."""
import json, os, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_common import Runner, REPO, mpirun_prefix
from rwmix_nodesweep2 import gen_cfgs, xsocr_wrap, CORES_PER_RANK

RE_OUT = re.compile(r"RWROUNDS PDS=(\d+) MODE=(\d+) ROUNDS=(\d+) E_US=(\d+) BYTES=(\d+) ELAPSED_US=(\d+)")
NODES = list(range(2, 17))
RMULT, E_US, BYTES, ITERS = 4, 50, 1024, 3

def run(runner, outdir, kind, nodes, mode, timeout):
    name = f"rwrounds_{kind}_{nodes}n_m{mode}"
    logfile = runner.logdir / f"{name}.log"
    env = os.environ.copy(); env["OMP_NUM_THREADS"] = "4"
    args = [str(mode), str(RMULT), str(E_US), str(BYTES)]
    if kind == "xsocr":
        launcher = xsocr_wrap(outdir, args).replace("rwmix_xsocr", "rwrounds_xsocr")
        to = timeout
        cmd = f"cd {runner.apps_dir} && timeout -k 1 {to} {mpirun_prefix(nodes)} {launcher}"
        binname = "rwrounds_xsocr"
    else:
        env["OCRVX_NUM_THREADS"] = "2"
        c = CORES_PER_RANK
        wrap = (f"bash -c 'r=${{PMI_RANK:-0}}; s=$((r*{c})); e=$((s+{c}-1)); "
                f"exec taskset -c $s-$e ./rwrounds_ocrvx " + " ".join(args) + "' _")
        to = timeout * 3
        cmd = f"cd {runner.apps_dir} && timeout -k 1 {to} {mpirun_prefix(nodes)} {wrap}"
        binname = "rwrounds_ocrvx"
    r = runner._run(cmd, env, logfile, wall_timeout=to)
    runner._reap_exe(runner.apps_dir / binname)
    return r

def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = REPO/"benchmarks"/"scripts"/"logs"/f"rwrounds_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    track = outdir/"track.jsonl"
    gen_cfgs(outdir)
    runner = Runner(mem_gb=0, timeout=240, logdir=outdir, target="cbgpu02",
                    build=REPO/"build_release_ocr_val_wb")
    total = len(NODES) * 2 * 3 * ITERS
    done = 0
    for nodes in NODES:
        for kind in ("xsocr", "ocrvx"):
            for mode in (0, 1, 2):
                for it in range(ITERS):
                    r = run(runner, outdir, kind, nodes, mode, 240)
                    m = RE_OUT.search(r.stdout or "")
                    rec = {"kind": kind, "nodes": nodes, "mode": mode,
                           "iter": it, "rc": r.rc,
                           "elapsed_us": int(m.group(6)) if m else None}
                    with open(track, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    done += 1
                    print(f"[{done}/{total}] {rec}", flush=True)
    print(f"DONE -> {track}")

if __name__ == "__main__":
    main()
