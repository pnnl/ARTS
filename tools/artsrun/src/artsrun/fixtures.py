"""Fixture staging: declared inputs the tool knows how to synthesize.

Most fixtures are cheap deterministic files (an identity matrix, a counting
sequence); regenerating them on the machine that needs them beats carrying
them around or dropping the cell.  Anything not in the registry stays a
missing input and is reported the usual way — a generated file is written
atomically so a crashed generator never leaves a half-file that later runs
mistake for the fixture.
"""

from __future__ import annotations

from pathlib import Path


def _identity_matrix(path: Path, n: int) -> None:
    # The cholesky ports read a whitespace-separated n×n text matrix; the
    # identity is SPD with a unit factor, so the result trace equals n.
    with open(path, "w") as fh:
        for r in range(n):
            fh.write(" ".join("1.0" if c == r else "0.0"
                              for c in range(n)) + "\n")


def _counting_file(path: Path) -> None:
    # basicIO reads u64 values one per line; 0..9 gives a fixed checksum.
    path.write_text("\n".join(str(i) for i in range(10)) + "\n")


GENERATORS = {
    "cholesky_input.mat": lambda p: _identity_matrix(p, 50),
    "cholesky_perf5k.mat": lambda p: _identity_matrix(p, 5000),
    "cholesky_perf7k5.mat": lambda p: _identity_matrix(p, 7500),
    "basicIO_test.dat": _counting_file,
}


def stage(paths: list[str]) -> list[str]:
    """Generate whatever is missing and known; return what is still missing."""
    left: list[str] = []
    for name in paths:
        path = Path(name)
        if path.exists():
            continue
        generator = GENERATORS.get(path.name)
        if generator is None:
            left.append(name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".staging")
        generator(tmp)
        tmp.replace(path)
    return left
