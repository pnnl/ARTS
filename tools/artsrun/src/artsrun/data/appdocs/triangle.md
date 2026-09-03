# triangle

*The 14-peg triangular-board solitaire puzzle, solved by recursive game-tree
search — the tree's own return path is the reduction, no shared counter
exists.*
Source: `third_party/ocr-apps/apps/triangle/refactored/ocr/intel/triangle.c`
(~370 lines).

## Overview

Despite the directory name, this is not a graph benchmark: it counts
solutions to the classic triangular peg-solitaire puzzle — the author's
board is the 15-hole 5-row triangle (`BOARDSIZE` = 15, `MOVESIZE` = 36
directed jumps, `BOTTOM` = 13 moves for a full solve), and `rows` (argv[3])
plays the same game on a larger triangle (see Parameters). `triangleTask(nummoves, oldmove, ..., depth)` applies `oldmove`
to a copy of its parent's board, and either (a) the search has reached
`depth` moves — one solution, return 1 — or (b) it enumerates every currently
legal jump (`nlegal`) and spawns one child `triangleTask` per legal move plus
one `sumCountsTask` continuation that waits on all the children's counts,
sums them, and forwards the total upward — a leaf with no legal moves before
`depth` is a dead end and returns 0. The recursion's return path *is* the
reduction tree; there is no shared/global counter. `final count` is the
result scalar; at the default full-depth search the known answer is 29760
solutions (checked in-source against a `PASS`/`FAIL` literal).

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `depth` | number of moves to search; the full puzzle is `BOTTOM` (13) | `BOTTOM` (13) if omitted or outside `[1, BOTTOM]` | ✓ parsed in `mainEdt`, carried via EDT paramv to every `triangleTask` create — multinode-safe |
| `argv[2]` = `rounds` | repeat the full (fresh, independent) search this many times, sequentially | 1 | ✓ parsed in `mainEdt`, carried via paramv through `wrapupTask`/`launch_round` — multinode-safe |
| `argv[3]` = `rows` | board rows: absent = the author's 5-row board driven by the author's hand table; given (5 included) = the jump-table generator drives, holes = rows(rows+1)/2, full solve = holes-2 moves; generator checked against the author's table on the 5-row board | absent (author mode) | ✓ parsed in `mainEdt`, geometry carried via paramv (`holes`, `nmoves`) to every task — multinode-safe |
| `BOARDSIZE` / `MOVESIZE` / `BOTTOM` | the author's-board instances of the geometry (15 holes, 36 jumps, 13-move full solve) | 15 / 36 / 13 | now derived from `rows` when one is given; the `#define`s remain the author-mode values and the generator's oracle |
| `TRIANGLE_SCATTER_LEVELS` | hinted-placement-only: top levels scattered by a deterministic hash of the board bitmask (a pure function of the position — not round-robin, not random) | 3 | ✗ compile-time; read only inside the `OCR_APP_OPTIMIZED_PLACEMENT` guard, dead in base |

## Structure

Branching (`nlegal` per node) is board-state-dependent, so total node counts
have no closed form — only per-node object counts do:

| object | count | size |
|--------|-------|------|
| `triangleTask` | 1 root, plus (data-dependent) `nlegal` children per internal node | — |
| `sumCountsTask` | 1 per internal node | — |
| `newboardDb` | 1 per child edge (`nlegal` per internal node) | 120 B (`BOARDSIZE·8`) |
| `pmovesDb` | 1, created once, shared read-only by the whole tree | 864 B (`MOVESIZE·3·8`) |
| `oldboardDb` / `boardDb` | 1 each, created once by `launch_round` | 120 B each |
| count DB (`returnCount`) | 1 per `triangleTask` node — via its own direct call (terminal/dead) or its `sumCountsTask` (internal) | 8 B |
| ONCE events | 1 "once" broadcast + 1 `childDone` per child, per internal node | — |
| `rootDone` event | 1, created once | — |

Worked example (`depth = 2`, hand-traced from the fixed starting board — see
notes for the full derivation): the root has 2 legal opening moves; each of
its 2 depth-1 children has 4 legal moves of its own (not 2 and 3 — a first
pass under-traced this), for 8 depth-2 grandchildren, all terminal. That's 11
`triangleTask` (1 root + 2 + 8), 3 `sumCountsTask` (root's + its 2 children's),
17 EDTs total (+ `mainEdt`/`realmainTask`/`wrapupTask`), 24 DBs, 14 events. At
the calibrated `args = []` (`depth = BOTTOM = 13`, the full puzzle) the known
solution-leaf count is 29760 — a lower bound on total `triangleTask` nodes,
since internal branch points and dead ends add more; the true total is not
knowable statically.

Counter cross-check: verified (1 node, `depth=1` vs `depth=2`): measured
absolutes EDT 8/18, DB 9/25, EVT 4/14; subtracting the runtime's constant
baseline (+1 EDT, +1 DB, +0 EVT per run) gives app-side EDT 7/17, DB 8/24, EVT
4/14 — exactly the re-traced worked numbers above (`depth=1`'s 7/8/4 already
matched on the first pass; `depth=2`'s 17/24/14 required the retrace).

An HPX port (`benchmarks/hpx/triangle.cpp`) mirrors both tiers as
`triangle_hpx` / `triangle_hinted_hpx`: a node applies its move to the board
its parent pushed (`8·holes` bytes per child, one copy per child — the
roster's one produced-data fan-out), the summer is a spawn (hinted: on the
creating locality; base: blind) that collects its children's counts by key.
At the calibrated `8 1 8`: `tree = nodes + summers = NUM_EDT_CREATE − 4 =
22,725,548` (counted run, 2026-09-02); `boards = nodes − 1`,
`board_bytes = 288 × boards`.

## Wiring

Every `triangleTask` copies its `oldboard` into its own `board`, applies
`oldmove`, and — if internal — creates a per-node `once` STICKY-ONCE event
that broadcasts its own `board` (CONST mode) to every one of its (up to
`nlegal ≤ MOVESIZE = 36`) children at once: this is the app's per-node RO
fan-out, bounded by 36. `pmovesDb`, the 864-byte move table, has no such
bound — it is CONST-read by *every* `triangleTask` node in the whole tree, so
its concurrent-reader count is limited only by how many nodes are runnable at
once, up to the full worker count of the run. Each child also gets a fresh
`newboardDb` (RW, becomes its own `board`) and a `childDone` ONCE event wired
into its parent's `sumCountsTask`. `sumCountsTask` waits on all `nlegal`
`childDone` events (RO), sums, destroys the children's count DBs, and
forwards the total via `returnCount`. `pmovesDb` is therefore the single
dominant fan-in/contention point — small, hot, globally shared, read-only —
the structural reason a per-DB RO-request-combining mitigation has an
outsized effect on this app specifically.

## Flow

The search is depth-first in code structure but not in execution: a node
creates all of its children before any of them runs, so the actual unfolding
is scheduler-driven breadth much like a tree of independent forks. Depth is
bounded by `depth` (≤ `BOTTOM = 13`); width at any instant is the number of
currently-live `triangleTask` nodes, pruned hard in practice since legal-move
count shrinks as the board empties (root always starts with exactly 2 legal
moves). The reduction wave (`sumCountsTask`) follows strictly behind the
search wave. `mainEdt`/`realmainTask`/`launch_round` are a short rank-0
preamble; `rounds > 1` chains independent full searches serially through
`wrapupTask`, so rounds never overlap.

## Placement (base)

`OCR_APP_OPTIMIZED_PLACEMENT` gates `triChildEdtHint`/`triLocalEdtHint`;
base both collapse to `NULL_HINT` on every `triangleTask`/`sumCountsTask`
create. Effective policy: EDT → runtime round-robin (per-creating-rank
counter), DB → home = creating rank. Consequence: a node's `once` broadcast
delivers its own board DB (homed on whichever rank the node itself was
round-robin-placed onto) to children that are themselves scattered onto
arbitrary ranks, so almost every child's `oldboard` read is a remote CONST
acquire; the same is true of `pmovesDb` (homed wherever `launch_round` ran,
effectively rank 0) against a tree scattered across every rank. Locality the
puzzle's tree structure would allow (keeping a subtree together) is never
expressed base — why the app is a strong RO-fan-in coherence stress case,
and why read combining measurably helps it.

## Placement (hinted)

As-born is placement-blind: tree tasks round-robin, each node's board DB homes
with its creating parent, so a child usually reads its board from another rank
and the summers collect counts remotely.

The layer mirrors nqueens: a deterministic per-subtree key, levels <=
`TRIANGLE_SCATTER_LEVELS` (default 3, `#ifndef`-overridable for calibration)
scattered on `mixKey(key) % nranks` — a hash of the position, so placement is
reproducible and independent of creation order — deeper tasks and the summers
pinned to the creating rank — each scattered subtree computes, allocates and
sums on one rank.  Board/return DBs keep `NULL_HINT`: the runtime's
creator-home default is the placement the 2026-07-10 re-pin experiment showed
this one-shot class needs (tri 58s->21s at the time it was an explicit pin).

## Family shape (measured, 15w+1p x 1/2/4/8 nodes, `7 1 7`)

hinted, e2e seconds — VAL alone anti-scales (the whole tree re-validates
the one move-table DB homed at rank 0 on every acquire, the family's
defining read cost), request combining erases exactly that, and INV/EXCL
are structurally immune (covering read / retained copy):

| arm | 1n | 2n | 4n | 8n |
|---|---|---|---|---|
| val_wb_nocomb | 0.4 | 4.9 | 6.7 | 9.4 |
| val_wb | 0.4 | 0.5 | 0.3 | 0.1 |
| inv_wb | 0.5 | 0.3 | 0.2 | 0.1 |
| excl_retain | 0.6 | 0.3 | 0.2 | 0.1 |

base anti-scales on EVERY arm (2n: val_nocomb 24.5 / val 30.3 / inv 73.4 /
excl 39.1) — each tree node's board is a fresh remote datablock, a cold-read
storm no coherence family can serve locally, and INV pays its directory on
top.  At the calibrated size the base multinode cells are therefore
reported as censored points (TIMEOUT, or the OOM kill an unbounded in-flight
backlog produces under VAL), never shrunk to fit.

## Sizing

The CLI is `depth [rounds [rows]]`.  `rows` (absent = the author's 5-row
board, its hand-written jump table driving the run; given — 5 included — a
generator enumerates the board's jumps, and on the 5-row board it must
reproduce the author's table, which is checked at startup) picks the board
and with it the graph family: holes = rows(rows+1)/2, a full game is
holes-2 moves.  `depth` is the size dial inside that board — it truncates
the search, and each +1 multiplies the tree by the board's branching
(measured ~6.4x at rows 6, ~12x at rows 7-8).  `rounds` repeats the whole
search sequentially and stays 1: the graph is the dial, not repetition.

Measured on the Dane-mirror geometry (1 node, 108w+4p, Release, rounds=1):
rows 6 depth 10 = 49.5 s (count 75,516,988); rows 7 depth 9 = 72.0 s
(114,947,436); rows 8 depth 9 = 193.7 s (325,211,332).  The lattice of
feasible points is sparse — one depth step multiplies the tree by the
branching, so each board offers exactly one point under the 300 s cell
ceiling (the next depth measured or extrapolates well past it: rows 6
depth 11 = 318 s, rows 7 depth 10 > 600 s).  The calibrated arguments are
the feasible point NEAREST the ~150 s anchor, which is also the largest
instance the ceiling admits; rows 7 depth 9 stands as the conservative
fallback if the 32-node anti-scaled cell needs more ceiling margin.  Peak RSS is ~4 GB and FLAT across all of these: with the board
destroys and counted events on, live objects track the search frontier, not
the tree's total size.  The author's full puzzle (no rows argument,
depth 13) remains the pinned 29,760-answer correctness case.
