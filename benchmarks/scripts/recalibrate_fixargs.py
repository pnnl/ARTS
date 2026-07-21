#!/usr/bin/env python3
"""recalibrate_fixargs.py — derive a --fixargs-override JSON that retargets the
single-experiment workloads so the BASELINE runtime's 1n wall hits a target.

Reads a calibration sweep's results.csv (one runtime, --node 1n), scales each
app's work knob by target/measured according to the knob's growth law, and
writes {bench: fix_args} JSON plus a human table. Apps within the accept band,
apps without a work knob, and apps whose sweep row is unusable are passed
through unchanged (reported).

Growth laws (knob value v, factor f = target/measured):
  lin   v' = v*f          quad  v' = v*sqrt(f)
  log2  v' = v + log2(f)  exp   v' = v + log2(f)
Values snap to human-natural numbers (2 significant digits; ints stay ints).

Usage: recalibrate_fixargs.py <calib_results.csv> <out.json> [--target 10]
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from performance_harness import BENCHES, _WEAK_AXIS  # noqa: E402

# Work-knob axis for the scalable core (fix_args knob; the weak/strong tables
# are hand-shaped per geometry and are NOT touched by recalibration).
_CORE_AXIS = {
    "graph500": (0, "log2"),          # SCALE
    "CoMD_sdsc2": ("-N", "lin"),
    "hpcg_intel": (4, "lin"),         # iters
    "nekbone": (7, "lin"),            # CGcount
    "RSBench_intel_sharedDB": ("-l", "lin"),
    "Stencil2D_intel_chandra": (2, "lin"),  # ntimesteps
    "hpgmg": (1, "lin"),              # target_boxes
    "stream_dist": (2, "lin"),        # ntimes
    # stencil1D septet: timestep count (arg 3) is the linear work knob.
    **{f"stencil1D_{v}": (2, "lin") for v in
       ("sticky", "channel", "guid", "once", "stickyLG", "guidPI", "oncePI")},
}
AXIS = {**_WEAK_AXIS, **_CORE_AXIS}

ACCEPT_LO, ACCEPT_HI = 0.7, 1.6      # measured/target band that needs no change
MIN_FACTOR, MAX_FACTOR = 1 / 64, 64  # clamp insane rescales (bad measurement)


def _snap(x: float) -> int:
    if x < 1:
        return 1
    mag = 10 ** (math.floor(math.log10(x)) - 1)
    return max(1, int(round(x / mag) * mag))


def _knob_index(fix_args: list, sel) -> int:
    if isinstance(sel, int):
        return sel
    return fix_args.index(sel) + 1  # flag: value is the following token


def rescale(fix_args: list, sel, law: str, f: float) -> list:
    out = list(fix_args)
    i = _knob_index(out, sel)
    v = float(out[i])
    if law == "lin":
        nv = _snap(v * f)
    elif law == "quad":
        nv = _snap(v * math.sqrt(f))
    elif law in ("log2", "exp"):
        nv = int(round(v + math.log2(f)))
    else:
        raise ValueError(law)
    out[i] = str(max(1, nv))
    return out


def main():
    csv_path, out_path = sys.argv[1], sys.argv[2]
    target = float(sys.argv[sys.argv.index("--target") + 1]) if "--target" in sys.argv else 10.0

    measured = {}
    for r in csv.DictReader(open(csv_path, newline="")):
        if r["config"] != "1n":
            continue
        bench = r["bench"]
        if r["status"] == "OK" and r["e2e_ns"]:
            measured.setdefault(bench, []).append(float(r["e2e_ns"]) / 1e9)
        elif r["status"] in ("TIMEOUT", "FAIL"):
            measured.setdefault(bench, [])

    by_name = {c.name: c for c in BENCHES}
    override, report = {}, []
    for name, case in by_name.items():
        vals = measured.get(name)
        if vals is None:
            report.append((name, "-", "no sweep row", None))
            continue
        if not vals:
            report.append((name, "-", "sweep TIMEOUT/FAIL — fix by hand", None))
            continue
        m = sorted(vals)[len(vals) // 2]
        if name not in AXIS:
            report.append((name, f"{m:.1f}s", "no work knob (fixed)", None))
            continue
        ratio = m / target
        if ACCEPT_LO <= ratio <= ACCEPT_HI:
            report.append((name, f"{m:.1f}s", "in band", None))
            continue
        f = min(MAX_FACTOR, max(MIN_FACTOR, target / m))
        sel, law = AXIS[name]
        try:
            new_args = rescale(case.fix_args, sel, law, f)
        except (ValueError, IndexError) as e:
            report.append((name, f"{m:.1f}s", f"axis error: {e}", None))
            continue
        if new_args != case.fix_args:
            override[name] = new_args
            report.append((name, f"{m:.1f}s", f"x{f:.2f} ({law})", new_args))
        else:
            report.append((name, f"{m:.1f}s", "snap = unchanged", None))

    Path(out_path).write_text(json.dumps(override, indent=1))
    w = max(len(n) for n, *_ in report)
    for name, m, note, new in sorted(report):
        line = f"{name:<{w}}  {m:>8}  {note}"
        if new:
            line += f"  -> {' '.join(new)}"
        print(line)
    print(f"\n{len(override)} overrides -> {out_path} (target {target}s)")


if __name__ == "__main__":
    main()
