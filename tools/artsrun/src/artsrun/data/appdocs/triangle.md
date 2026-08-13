# triangle

*The 14-peg triangular-board solitaire puzzle, solved by recursive game-tree
search — the tree's own return path is the reduction, no shared counter
exists.*
Source: `third_party/ocr-apps/apps/triangle/refactored/ocr/intel/triangle.c`
(~370 lines).

## Overview

Despite the directory name, this is not a graph benchmark: it counts
solutions to the classic 15-hole triangular peg-solitaire puzzle (`BOARDSIZE`
= 15 holes, `MOVESIZE` = 36 directed jump-moves, `BOTTOM` = 13 moves for a
full solve). `triangleTask(nummoves, oldmove, ..., depth)` applies `oldmove`
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
| `BOARDSIZE` / `MOVESIZE` / `BOTTOM` | board topology constants (15 holes, 36 jump-moves, 13-move full solve) | 15 / 36 / 13 | ✗ `#define`s — these fix the puzzle itself, not a workload-size knob |
| `TRIANGLE_RR_LEVELS` | optimized-placement-only: top levels round-robin distributed on the board's bitmask | 3 | ✗ compile-time; read only inside the `OCR_APP_OPTIMIZED_PLACEMENT` guard, dead in as-born |

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

## Placement (as-born)

`OCR_APP_OPTIMIZED_PLACEMENT` gates `triChildEdtHint`/`triLocalEdtHint`;
as-born both collapse to `NULL_HINT` on every `triangleTask`/`sumCountsTask`
create. Effective policy: EDT → runtime round-robin (per-creating-rank
counter), DB → home = creating rank. Consequence: a node's `once` broadcast
delivers its own board DB (homed on whichever rank the node itself was
round-robin-placed onto) to children that are themselves scattered onto
arbitrary ranks, so almost every child's `oldboard` read is a remote CONST
acquire; the same is true of `pmovesDb` (homed wherever `launch_round` ran,
effectively rank 0) against a tree scattered across every rank. Locality the
puzzle's tree structure would allow (keeping a subtree together) is never
expressed as-born — why the app is a strong RO-fan-in coherence stress case,
and why read combining measurably helps it.

## Sizing

`depth` is the primary dial: below `BOTTOM = 13` it truncates the search (a
strictly smaller, strictly cheaper tree), and only `depth = BOTTOM` reaches
the full, pinned 29760-solution answer. `rounds` repeats the whole (fresh)
search sequentially — useful for extending wall time, not for widening
parallelism (rounds never overlap). Because the tree's total size cannot be
predicted in closed form, sizing against a machine is empirical: `depth =
BOTTOM` (the calibrated `args = []`) is already the largest single-round
workload the puzzle offers and is used unmodified at every node count in the
strong-scaling sweep (1/2/4/8 nodes × 15 workers/node); a smaller `depth`
(e.g. 8-10) would shorten a debug run at the cost of a much smaller, less
representative tree. There is no separate memory concern — every DB in this
app is well under 1 KiB.
