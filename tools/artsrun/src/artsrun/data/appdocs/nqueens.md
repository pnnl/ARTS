# nqueens

*Backtracking N-Queens search over EDTs — every partial board is a task, a
reduction tree sums subtree solution counts.*
Source: `third_party/ocr-apps/apps/nqueens/refactored/ocr/nqueens.c` (~300
lines).

## Overview

Counts the solutions to the N-Queens problem (place N mutually
non-attacking queens on an N×N board) by column/diagonal-pruned
backtracking. Each partial placement is a `findSolutionsEdt`; once the
number of queens already placed exceeds `n - cutoff`, the EDT stops
spawning children and instead finishes its *entire* remaining subtree in
one call via plain sequential recursion (`count_solutions_seq`) — `cutoff`
is exactly the granularity knob between fine-grained EDT parallelism and
coarser in-task compute. Children's counts fold back up through a
`sumSolutionsEdt` continuation per spawning node. The result scalar
(`sols: N`) is checked externally by the harness (catalog `expect`), not
inside the program. Per-EDT work above the cutoff is a handful of
bitmask instructions (no floating point), so — like the cutoff frontier is
made coarser or finer — the program shifts between measuring pure
task/event/DB churn and measuring real backtracking compute.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `n` | board size; asserted `0 < n < 31` | none (required) | ✓ parsed in `mainEdt` via `ocrGetArgv`, carried in `struct nqueens_args`/`shutdown_args` paramv — multinode-safe |
| `argv[2]` = `cutoff` | queens-placed depth at which an EDT switches from spawning children to a single sequential subtree search; asserted `cutoff < n` | none (required) | ✓ same paramv path — multinode-safe |
| `argv[3]` = `rounds` (optional) | repeats the whole search that many times, chained through `shutdownEdt`; only the final round prints/times | 1 | ✓ but ⚠ a value `< 1` is silently coerced back to 1 (no error) |

`NQUEENS_RR_LEVELS` (= 3) is a compile-time constant of the *hinted*
placement layer only (round-robins the top 3 levels of the tree, pins the
rest local); the base build never reads it.

## Structure

Let `d = popcount(cols)` (queens already placed). A `findSolutionsEdt` at
depth `d`: (i) if `d > n − cutoff`, finishes its whole remaining subtree
serially — no further EDTs; (ii) else if `cols` spans all `n` columns, it
is a leaf solution; (iii) else it creates one child `findSolutionsEdt` per
legal column (`available = ~(ldiag|cols|rdiag) & all`) plus one
`sumSolutionsEdt` continuation. There is no closed form for the node
count — which columns survive diagonal pruning at each depth is
board-state-dependent, the same reason N-Queens search trees have no
simple recurrence — but every other quantity reduces to it exactly. Let
`T(n,cutoff)` = total `findSolutionsEdt` invocations and `S(n,cutoff)` =
of those, the spawning ones (creators of a `sumSolutionsEdt`):

| object | count | size |
|--------|-------|------|
| `findSolutionsEdt` | `T(n,cutoff)` | — |
| `sumSolutionsEdt` | `S(n,cutoff)` | — |
| `shutdownEdt` | `rounds` (1 per round) | — |
| `mainEdt` | 1 | — |
| DBs | `T(n,cutoff)` — exactly one 8-byte result block per `findSolutionsEdt`, delivered directly (leaf/dead-end/complete) or through its `sumSolutionsEdt` | 8 bytes each |
| Events | `T(n,cutoff)` ONCE events — one `child_done` per non-root node plus the one `rootDone` | — |
| Templates | `find_template`/`shutdown_template` persist for the whole run; `sum_template` is created and destroyed by every one of the `S(n,cutoff)` spawning calls (pure churn) | — |

Both identities (DBs = Events = `T(n,cutoff)`) hold regardless of `n`/`cutoff`
because every tree node contributes exactly one result and one incoming
edge-or-root-event. Worked numbers (static replay of the recurrence
above, not the compiled binary): `n=6,cutoff=2` (catalog `expect_args`) →
`T=149, S=99`; `n=15,cutoff=8` (calibrated `args`, `rounds=1`) →
`T=8,586,246`, `S=2,461,096`, giving ≈11.05M EDTs and ≈8.59M DBs/events.
Live-set tracks the active frontier, not the total, since results are
destroyed as soon as their consumer reads them.

Counter cross-check: verified (1 node, `6 2` vs `8 3`): NUM_EDT_CREATE
251 → 2591, NUM_DB_CREATE 150 → 1654, NUM_EVENT_CREATE 149 → 1653 —
exactly `T+S+2` / `T` / `T` (app values 250/2590, 149/1653, 149/1653)
plus the runtime's constant +1 EDT/+1 DB/+0 EVT baseline.

## Wiring

- Every dependence in this app is `DB_MODE_RO` (child results into
  `sumSolutionsEdt`, the root count into `shutdownEdt`) — there is no
  `DB_MODE_RW` anywhere. Each 8-byte result DB has exactly one producer and
  one consumer; there is no fan-out and no contention point.
- `findSolutionsEdt` wires a fresh ONCE event to `sumSolutionsEdt`'s next
  slot *before* creating the child that will satisfy it, so a child can
  never fire an unregistered event.
- `shutdownEdt` depends on the root's `rootDone` event (slot 0, RO); on
  `rounds_left > 1` it re-enters `solve_nqueens` instead of shutting down.

## Flow

`mainEdt` (rank 0) creates both templates once, then `solve_nqueens` seeds
one root `findSolutionsEdt` and its `shutdownEdt` per round. The tree
unfolds depth-first in creation order but executes with scheduler-driven
parallelism: width grows with the branching factor down through depth
`n − cutoff`, then hands off to `n − cutoff + 1`-depth EDTs that each run
an independent, embarrassingly-parallel sequential subtree search
(the actual backtracking compute). A completion wave of `sumSolutionsEdt`s
folds counts back up. `shutdownEdt` is the only serial join point; with
`rounds > 1` it re-seeds a fresh, independent (but identically-shaped,
since the search is deterministic) tree for each subsequent round before
the final one prints and calls `ocrShutdown()`.

## Placement (base)

Both hint helpers (`nqPlaceEdtHint`, `nqLocalEdtHint`) return `NULL_HINT`
outside `OCR_APP_OPTIMIZED_PLACEMENT`; there is no base affinity usage
to report. Effective policy:

- **EDTs**: NULL hint → shim passes `ARTS_HINT_ANY_RANK` → runtime
  round-robin. `findSolutionsEdt`, `sumSolutionsEdt` and `shutdownEdt` all
  scatter across ranks with no relation to which subtree they belong to.
- **DBs**: NULL hint → home = creating rank, i.e. wherever the producing
  `findSolutionsEdt`/`sumSolutionsEdt` happened to land.

Consequence: a child rarely executes on the rank that created its 8-byte
argument (arguments travel via paramv, so this costs nothing), but a
`sumSolutionsEdt` or `shutdownEdt` very often acquires its inputs from a
remote rank holding an 8-byte DB — the same fine-grain coherence stress
pattern as the app's sibling recursive tree-of-tasks benchmarks, at
N-Queens's combinatorial (not Fibonacci) growth rate.

## Placement (hinted)

As-born is placement-blind (see above): `findSolutionsEdt` scatter round-robin
with no relation to their subtree, and every 8-byte result DB homes wherever
its producer landed, so the summing side acquires almost everything remotely.

The layer (`nqPlaceEdtHint` in `nqueens.c`) uses the column bitmask as a
distinct per-subtree key and its popcount as the tree level: levels below
`NQUEENS_RR_LEVELS` (default 3, `#ifndef`-overridable for calibration) scatter
round-robin on `mixKey(cols) % nranks`, deeper tasks pin to the creating rank
(`nqLocalEdtHint` likewise pins the sum EDTs), so each scattered subtree — its
spawn tree, its result DBs, and its sums — stays on one rank.  Result DBs keep
`NULL_HINT`: the runtime's creator-home default gives the one-shot 8-byte
blocks the local home the 2026-07-09 A/B (2n 48s->250s without it) showed they
must have.

## Sizing

`n` and `cutoff` move parallelism in different ways: `n` scales total
search size combinatorially; `cutoff` trades EDT-tree depth/width for
per-leaf serial compute (small `cutoff` → deep tree, many tiny EDTs, more
churn; large `cutoff` → shallow tree, fewer but heavier leaf EDTs).
`rounds` only extends wall time linearly (independent repeats), it does
not change per-round parallelism.

- Pick `cutoff` so the number of depth-`(n−cutoff+1)` leaf EDTs
  (the embarrassingly-parallel serial-search frontier) comfortably exceeds
  total workers — that count is not closed-form, but it grows with the
  branching factor at that depth, so raising `cutoff` by 1–2 is normally
  enough headroom. Pick `n` so `T(n,cutoff)` ≫ total workers for the
  scheduling/tree-churn phase.
- The calibrated `n=15, cutoff=8` (no `rounds`) gives ≈11M EDTs total —
  comfortably above 8 nodes × 15 workers = 120 workers with wide margin at
  both the tree-churn and the serial-leaf phases.
- Memory is not the limiting factor (8-byte DBs, destroyed on consumption);
  size for total EDT count and remote-acquire volume, not footprint.
