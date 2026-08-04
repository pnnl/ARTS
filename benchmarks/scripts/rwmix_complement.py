#!/usr/bin/env python3
"""Complement runner: fill ONLY node-sweep cells missing from the surviving
track files (never re-runs a cell that already has its 2 iterations)."""
import json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_common import Runner, REPO
from rwmix_nodesweep import (RE_OUT, RUNTIMES, NODES, N_TASKS, E_US, BYTES,
                             ITERS, gen_cfgs, run_cell)

SURVIVING = [
    REPO/"benchmarks"/"scripts"/"logs"/"rwmixns_20260721-125446"/"track.jsonl",
    REPO/"benchmarks"/"scripts"/"logs"/"rwmixns_20260721-143717"/"track.jsonl",
]

have = {}
for t in SURVIVING:
    if not t.exists(): continue
    for line in open(t):
        r = json.loads(line)
        k = (r["kind"], r["suffix"], r["nodes"], r["w"])
        have[k] = have.get(k, 0) + 1

ts = time.strftime("%Y%m%d-%H%M%S")
outdir = REPO/"benchmarks"/"scripts"/"logs"/f"rwmixns_comp_{ts}"
outdir.mkdir(parents=True, exist_ok=True)
track = outdir/"track.jsonl"
cfgs = gen_cfgs(outdir)
runner = Runner(mem_gb=0, timeout=180, logdir=outdir, target="cbgpu02",
                build=REPO/"build_release_ocr_val_wb")

todo = []
for nodes in NODES:
    per_half = max(1, round(N_TASKS/(2*nodes)))
    n_tasks = 2*per_half*nodes
    for kind, suffix in RUNTIMES:
        for w in (0, n_tasks//2, n_tasks):
            k = (kind, suffix, nodes, w)
            missing = ITERS - have.get(k, 0)
            for _ in range(max(0, missing)):
                todo.append((kind, suffix, nodes, n_tasks, w))
print(f"missing cells: {len(todo)}", flush=True)
for i, (kind, suffix, nodes, n_tasks, w) in enumerate(todo, 1):
    r = run_cell(runner, cfgs, kind, suffix, nodes,
                 [str(n_tasks), str(w), str(E_US), str(BYTES)], 180)
    m = RE_OUT.search(r.stdout or "")
    rec = {"kind": kind, "suffix": suffix, "nodes": nodes, "n": n_tasks,
           "w": w, "iter": -1, "rc": r.rc,
           "elapsed_us": int(m.group(6)) if m else None,
           "pds": int(m.group(1)) if m else None}
    with open(track, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{i}/{len(todo)}] {rec}", flush=True)
print(f"DONE -> {track}")
