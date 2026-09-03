#!/usr/bin/env python3
"""Compare a port's [STRUCT] line against expected values, tolerance zero.

usage: check_struct.py <log> field=value [field=value ...]
Every named field must appear on the line with exactly that value; fields
on the line but not named are ignored.  Exit 0 on match, 1 otherwise.
"""
import re
import sys


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    text = open(argv[1], errors="replace").read()
    lines = re.findall(r"^\[STRUCT\] (.*)$", text, re.M)
    if len(lines) != 1:
        print(f"expected exactly one [STRUCT] line, found {len(lines)}")
        return 1
    got = {k: int(v) for k, v in (kv.split("=") for kv in lines[0].split())}
    bad = []
    for spec in argv[2:]:
        k, v = spec.split("=")
        if got.get(k) != int(v):
            bad.append(f"{k}: got {got.get(k)}, want {v}")
    print("\n".join(bad) if bad else "STRUCT OK " + lines[0])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
