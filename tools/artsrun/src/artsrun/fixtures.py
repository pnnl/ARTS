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


def _identity_tiles_bin(path: Path, n: int, ts: int) -> None:
    # The tile-stream layout convertData emits and cholesky's --fib reads:
    # the lower-triangular tiles in (i, j<=i) order, each ts*ts doubles,
    # row-major within the tile, host byte order.  The tile size is part of
    # the layout, so it is part of the fixture's NAME — a stream baked for
    # one ts is not a valid input for another.
    import struct

    assert n % ts == 0
    tiles = n // ts
    zero_tile = bytes(8 * ts * ts)
    diag = bytearray(zero_tile)
    one = struct.pack("=d", 1.0)
    for r in range(ts):
        off = 8 * (r * ts + r)
        diag[off:off + 8] = one
    with open(path, "wb") as fh:
        for i in range(tiles):
            for j in range(i + 1):
                fh.write(diag if i == j else zero_tile)


def _acgt_pair(path: Path, which: int, lens=(515000, 517000),
               seed=20260819) -> None:
    # The two alignment inputs are one seeded ACGT stream split at lens[0];
    # regenerating either file alone must not restart the stream, so one
    # call derives both and writes the sibling (atomically) when absent.
    import random

    rng = random.Random(seed)
    both = ["".join(rng.choice("ACGT") for _ in range(n)) for n in lens]
    path.write_text(both[which])
    sibling_name = ["string1-huge.txt", "string2-huge.txt"][1 - which]
    sibling = path.parent / sibling_name
    if not sibling.exists():
        tmp = sibling.with_name(sibling.name + ".staging")
        tmp.write_text(both[1 - which])
        tmp.replace(sibling)


def _counting_file(path: Path) -> None:
    # basicIO reads u64 values one per line; 0..9 gives a fixed checksum.
    path.write_text("\n".join(str(i) for i in range(10)) + "\n")


GENERATORS = {
    "cholesky_input.mat": lambda p: _identity_matrix(p, 50),
    "cholesky_perf5k.mat": lambda p: _identity_matrix(p, 5000),
    "cholesky_perf7k5.mat": lambda p: _identity_matrix(p, 7500),
    "cholesky_perf5k_ts100.bin": lambda p: _identity_tiles_bin(p, 5000, 100),
    "cholesky_perf7k5_ts100.bin": lambda p: _identity_tiles_bin(p, 7500, 100),
    "cholesky_perf28k.mat": lambda p: _identity_matrix(p, 28000),
    "cholesky_perf40k_ts100.bin": lambda p: _identity_tiles_bin(p, 40000, 100),
    "string1-huge.txt": lambda p: _acgt_pair(p, 0),
    "string2-huge.txt": lambda p: _acgt_pair(p, 1),
    # The alignment's expected global score for the pair above, computed by an
    # independent sequential reference of the same DP (border = gap*position,
    # match 2 / transition -2 / transversion -4 / gap -1, no zero clamp).
    "score-huge.txt": lambda p: p.write_text("318128\n"),
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
