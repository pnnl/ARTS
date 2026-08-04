#!/usr/bin/env python3
"""rwrounds on ARTS Rcu/Lazy (user-requested comparison run)."""
import json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_common import Runner, REPO
from rwmix_nodesweep2 import gen_cfgs

RE_OUT = re.compile(r"RWROUNDS PDS=(\d+) MODE=(\d+) ROUNDS=(\d+) E_US=(\d+) BYTES=(\d+) ELAPSED_US=(\d+)")
ts = time.strftime("%Y%m%d-%H%M%S")
outdir = REPO/"benchmarks"/"scripts"/"logs"/f"rwrounds_arts_{ts}"
outdir.mkdir(parents=True, exist_ok=True)
track = outdir/"track.jsonl"
cfgs = gen_cfgs(outdir)
runner = Runner(mem_gb=0, timeout=240, logdir=outdir, target="cbgpu02",
                build=REPO/"build_release_ocr_val_wb")
done=0; total=15*3*3
for nodes in range(2,17):
    for mode in (0,1,2):
        for it in range(3):
            r = runner.run_arts_mn(f"rwrounds_arts_{nodes}n_m{mode}", "rwrounds",
                                   [str(mode),"4","50","1024"], nodes,
                                   suffix="ocr_val_wb",
                                   cfg_path=cfgs[("arts", nodes)], timeout=240)
            m = RE_OUT.search(r.stdout or "")
            rec = {"kind":"arts_val_wb","nodes":nodes,"mode":mode,"iter":it,
                   "rc":r.rc,"elapsed_us":int(m.group(6)) if m else None}
            with open(track,"a") as f: f.write(json.dumps(rec)+"\n")
            done+=1; print(f"[{done}/{total}] {rec}", flush=True)
print(f"DONE -> {track}")
