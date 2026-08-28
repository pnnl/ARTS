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
    # A row differs from the all-zero row in one field, so the zero row is
    # built once and patched -- at the calibrated sizes this is hundreds of
    # millions of fields and the naive join dominates the staging.
    zero = ["0.0"] * n
    with open(path, "w") as fh:
        for r in range(n):
            zero[r] = "1.0"
            fh.write(" ".join(zero) + "\n")
            zero[r] = "0.0"


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


def _acgt_pair(path: Path, which: int, lens, seed, names) -> None:
    # The two alignment inputs are one seeded ACGT stream split at lens[0];
    # regenerating either file alone must not restart the stream, so one call
    # derives both and writes the sibling (atomically) when absent.
    import random

    rng = random.Random(seed)
    both = ["".join(rng.choice("ACGT") for _ in range(n)) for n in lens]
    path.write_text(both[which])
    sibling = path.parent / names[1 - which]
    if not sibling.exists():
        tmp = sibling.with_name(sibling.name + ".staging")
        tmp.write_text(both[1 - which])
        tmp.replace(sibling)


def _counting_file(path: Path) -> None:
    # basicIO reads u64 values one per line; 0..9 gives a fixed checksum.
    path.write_text("\n".join(str(i) for i in range(10)) + "\n")


_SW_HUGE  = ("string1-huge.txt", "string2-huge.txt")
_SW_CAL   = ("string1-cal.txt", "string2-cal.txt")
_SW_TREND = ("string1-trend.txt", "string2-trend.txt")
_SW_SWD   = ("string1-swd.txt", "string2-swd.txt")

GENERATORS = {
    "cholesky_input.mat": lambda p: _identity_matrix(p, 50),
    "cholesky_perf16700_ts100.bin": lambda p: _identity_tiles_bin(p, 16700, 100),
    "cholesky_perf90000_ts500.bin": lambda p: _identity_tiles_bin(p, 90000, 500),
    # Three alignment pairs, each a seeded ACGT stream split in two.  The
    # scores are the expected global alignment of the pair, each computed by an
    # independent sequential reference of the same DP (border = gap*position,
    # match 2 / transition -2 / transversion -4 / gap -1, no zero clamp).
    "string1-huge.txt":  lambda p: _acgt_pair(p, 0, (515000, 517000), 20260819, _SW_HUGE),
    "string2-huge.txt":  lambda p: _acgt_pair(p, 1, (515000, 517000), 20260819, _SW_HUGE),
    "score-huge.txt":    lambda p: p.write_text("318128\n"),
    "string1-cal.txt":   lambda p: _acgt_pair(p, 0, (140000, 140400), 20260819, _SW_CAL),
    "string2-cal.txt":   lambda p: _acgt_pair(p, 1, (140000, 140400), 20260819, _SW_CAL),
    "score-cal.txt":     lambda p: p.write_text("86360\n"),
    "string1-trend.txt": lambda p: _acgt_pair(p, 0, (70000, 70000), 20260828, _SW_TREND),
    "string2-trend.txt": lambda p: _acgt_pair(p, 1, (70000, 70000), 20260828, _SW_TREND),
    "score-trend.txt":   lambda p: p.write_text("43068\n"),
    "string1-swd.txt":   lambda p: _acgt_pair(p, 0, (800000, 800000), 20260830, _SW_SWD),
    "string2-swd.txt":   lambda p: _acgt_pair(p, 1, (800000, 800000), 20260830, _SW_SWD),
    "score-swd.txt":     lambda p: p.write_text("493680\n"),
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
