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
| `argv[4]` = scatter levels (optional) | tree levels (queens placed) scattered across ranks before a subtree pins to the rank it landed on | `NQUEENS_RR_LEVELS` (3) | ✓ parsed in `mainEdt`, carried in `struct nqueens_args` paramv to every task — no global |

**`rounds` is repetition, not refinement**: `shutdownEdt` restarts the
identical search, chained after the previous one finishes, so a round adds
wall time and nothing else.  It stays at 1 and the board size carries the
weight.  The scatter depth used to be a compile-time constant; it is the app's
only parallelism dial and is calibrated by measurement, so it is an argument,
with the `#define` surviving as the default.  The base build reads neither.

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
distinct per-subtree key and its popcount as the tree level: levels below the
scatter depth (argument, default 3, calibrated to **5**) are placed by
`mixKey(cols) % nranks`, deeper tasks pin to the creating rank
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
- Measured at the Dane anchor node (108w+4p, `cutoff 12`, one round):
  14 → 0.01 s, 16 → 0.11, 17 → 0.81, 18 → 6.6, 19 → 62.9, 20 → 624.8.  The
  ladder is a factor of ~8 per step.  `20` is the calibrated size; at the
  calibrated grain its anchor is **~289 s**, not the 624.8 s the port's
  inherited `cutoff 12` produced.
- **The solution count needs 64 bits from `n = 19` on** (4 968 057 848 for 19,
  39 029 188 884 for 20).  The port truncated it to `u32` when printing, so
  every size at or above 19 reported a wrong answer until this cycle.
- **`cutoff` and the scatter depth are one plane, not two knobs.**  `cutoff`
  fixes `max_set = n - cutoff`, so tasks exist at popcounts `0 … max_set+1`
  and the ones at `max_set+1` are the sequential leaves.  A scatter of
  `max_set+1` therefore means "scatter every spawning task, leave the
  sequential leaves where they were created", `max_set+2` means "scatter
  those too", and anything beyond saturates: at `cutoff 16`, `s5` = 295.1 s
  leaves the leaves local, `s6` = 287.2 scatters them, and `s7` = 287.3 is
  the same run again.  The plane at `n = 20` over 8 bentley nodes, E2E seconds:

  | cutoff (max_set) | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
  |---|---|---|---|---|---|---|---|
  | 13 (7) | — | 373.1 | — | 318.3 | — | 321.6 | 540.3 |
  | 14 (6) | — | — | — | — | 305.8 | — | — |
  | 15 (5) | — | 361.1 | 339.3 | 308.9 | **299.8** | 308.8 | — |
  | 16 (4) | 552.3 | 345.6 | 325.3 | 295.1 | **287.2** | 287.3 | — |
  | 17 (3) | 530.1 | 332.2 | 312.4 | **283.6** | — | — | — |
  | 18 (2) | — | — | 304.2 | 301.9 | — | — | — |

  What the plane says is not "scatter as deep as possible": the cost tracks
  the **number of tasks scattered**, and that is exponential in depth (levels
  1…4 hold about 55 k boards, level 7 about 28 M).  Scattering every spawning
  level is right only while that count stays near 1e5 — `c13 s8` scatters
  28 M and costs 540 s, nearly twice the optimum.  Too little scatter is just
  as bad: the `s2` column is 1.8× the optimum because seven ranks sit idle.
- **The fastest cell is not the calibrated one.**  `c17 s5` is 283.6 s and
  `c16 s6` is 287.2 s, but the placement counters at 8 nodes say why the
  slower one is chosen: `c17 s5` leaves 72 417 units (600 per worker here,
  **21 per worker at 32 nodes**) and already shows 1.19× spread in
  `TIME_EDT_EXEC` with 173× spread in steal attempts, while `c16 s6` leaves
  788 725 units, 1.05× and 47×.  The 1.3% is real, not noise — the `s6`/`s7`
  pair of the same row is an accidental repeat of one configuration (scatter
  saturates at `max_set+2`, so both scatter everything) and reproduced to
  287.2 / 287.3 s, putting the run-to-run band under 0.1%.  The 1.3% is
  knowingly paid for 10× the load-balancing headroom at the geometry the
  campaign actually ends at.
- The one-node anchor cannot decide any of this: scatter is a no-op there
  (every affinity resolves to the only rank), so the anchor sees grain only
  and prefers ever-coarser cutoffs — 301.3 / 289.0 / 276.8 s for c15 / c16 /
  c17.  The 8-node plane is what the calibration rests on.
- Memory is not the limiting factor (8-byte DBs, destroyed on consumption);
  size for total EDT count and remote-acquire volume, not footprint.
- **The hinted tier is what the app is for.**  Base at `n=18, cutoff 12`:
  32.7 s at one bentley node, 151.3 at two, 132.7 at four.  Hinted at the same
  size: 32.7 / 18.5 / 10.0 / 5.2 over 1/2/4/8 nodes.  At the trend size with
  the calibrated grain (`18 14 1 5`) the four arms run
  30.8 / 16.8 / 9.8 / 5.1 and sit within **0.5%** of each other — the
  coherence configuration is nearly not a variable for this application, and
  that is itself the result.  The placement counters say why: at the trend
  size 99.8% of acquires are local hits, with `NUM_EDT_FINISH` spread 1.10x
  and `TIME_EDT_EXEC` 1.07x across ranks.
