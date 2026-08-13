# LCS_shared

*Recursive quad-tree wavefront DP over a single shared score matrix —
every leaf task takes an exclusive turn on the one DB, by construction.*
Source: `third_party/ocr-apps/apps/LCS/refactored/ocr/intel-jesmin-lcs_shared_datablocks/lcs.c`
(~560 lines; author Jesmin Jahan Tithi, Intel 2016).

## Overview

Computes a longest-common-subsequence-style alignment score between two
random strings of length `N` by dynamic programming, using a
cache-oblivious recursive decomposition: `recLCSEdt(n)` with `n > base`
splits its `n×n` square region into four `(n/2)×(n/2)` quadrants — `x11`
(top-left, unblocked), `x12`/`x21` (top-right/bottom-left, both gated on
`x11`'s completion, running concurrently with each other), `x22`
(bottom-right, gated on both `x12` and `x21`) — each spawned as its own
`EDT_PROP_FINISH` sub-recursion. Once `n ≤ base`, the call instead spawns a
single `seqLCSEdt` that fills its whole `n×n` block with a serial
antidiagonal sweep. What makes this the **shared** variant: all three DP
inputs — `S`, `T`, and the DP score itself — are each held in exactly
**one** datablock for the entire run. The score DB is not the full `N×N`
matrix; it uses a compact antidiagonal-offset encoding (`idx = N +
(xj+j-xi-i)`, independent of absolute position) so only `O(N)` longs are
ever allocated, with distinct quadrants writing into overlapping index
ranges across time by design — the recursion's finish-EDT ordering is what
makes that reuse safe. Every one of the `seqLCSEdt` leaves — however many
of them the recursion produces — takes its turn on that same one DB.
`shutDownEdt` prints `LCS length: N` and asserts it against a *native*,
single-threaded recomputation of the identical recurrence (`serial_lcs`,
run inside `mainEdt` before any EDT is created) — a self-consistency check
between the parallel and serial versions of this recurrence, not a check
against an external LCS oracle. The DAG shape depends only on `N` and
`base`, never on string content. Per-leaf compute is small (an `O(n²)`
antidiagonal sweep at `n ≈ base`); the program's real stress is DB
contention — see Wiring.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `N` | string length; sizes the whole DAG and every DB payload | 1024 | ✓ `atol` in `mainEdt`, propagated to the recursion through `LCS_task_params.N` in `paramv` — multinode-safe |
| `argv[2]` = `base` | recursion base case: quadrant side length at which the split stops | 256 | ✓ same, via `LCS_task_params.base`; clamped to `N` if larger |
| `argv[3]` = `num_workers` | intended worker count | 16 | ⚠ parsed on rank 0, used only in one `ocrPrintf` — never sets ARTS's actual thread count (that is `ARTS_CONFIG`'s own `workers` key) and is not propagated anywhere else. Passing it prints a one-line warning that it has no effect in this port |

`GAP_PENALTY` (`= 0`) and the `CHECK_RESULTS`/`PRINT` compile-time
switches (both unconditionally `#define`d in this file) are compile-time
only, no argv path.

## Structure

Let `d` be the recursion depth: the smallest `d ≥ 0` with `N` right-shifted
`d` times (`n ← n>>1`, repeated) `≤ base`. Every branch of the quad-tree
reaches the base case at the *same* depth `d` (shift depends only on
level, never on position), so the recursion is a perfect 4-ary tree of
depth `d`:

| object | count | size |
|--------|-------|------|
| `recLCSEdt` (recursive, incl. root) | `(4^(d+1)-1)/3` — one node per tree position, levels `0..d` | — |
| `seqLCSEdt` (leaf DP kernel) | `4^d` — one per level-`d` node | — |
| `mainEdt` / `shutDownEdt` | 1 each | — |
| DBs | **3, fixed** — `S`, `T`, `score`; never depth-dependent | `S`=`T`=`4·(N+1)` B; `score`=`16·(N+1)` B (2 longs per position) |
| Events | `6·4^d − 1` — every `recLCSEdt` create (root + `x11`/`x12`/`x21`/`x22`, count `(4^(d+1)-1)/3`) is `EDT_PROP_FINISH` with a non-NULL output-event argument, so each costs 3 events (1 app `ocrEventCreate` STICKY + 1 runtime finish event + 1 runtime output event); every `seqLCSEdt` create (count `4^d`) is `EDT_PROP_NONE` but still passes a non-NULL output-event argument and is preceded by an app `ocrEventCreate`, so each costs 2 (no finish event); `shutDownEdt`'s create is `EDT_PROP_NONE` with a NULL output event and costs 0 | — |
| EDT templates | `2·4^d + 1`, none ever destroyed (3 per internal node + 1 per leaf + 2 top-level) | — |

Worked numbers at the calibrated `args = [40960, 1024, 48]` (`N=40960,
base=1024`): `40960 → 20480 → 10240 → 5120 → 2560 → 1280 → 640` is 6
shifts to reach `≤ base`, so `d=6`. `recLCSEdt` = 5,461; `seqLCSEdt` =
4,096; total EDTs = 9,559; Events = `6·4⁶ − 1` = 24,575; DBs = 3
(`S`=`T`≈160 KiB, `score`≈640 KiB, unaffected by `d`). Each unit of `d`
(i.e. each doubling of `N/base`) multiplies EDT/event counts by ~4 — far
steeper than Fibonacci's ×1.618 — while DB *count* never changes and DB
*payload* grows only linearly in `N`.

Counter cross-check: verified (1 node, `8 4 1` (`d=1`) vs `16 4 1`
(`d=2`)): ΔNUM_EDT_CREATE = 28, ΔNUM_DB_CREATE = 0, ΔNUM_EVENT_CREATE =
72 — exactly the formulas' deltas (`(4^(d+1)-1)/3 + 4^d` grows 11→39;
DB count is pinned at 3; Events grows `6·4¹−1=23` → `6·4²−1=95`); the
runtime adds a constant baseline of +1 EDT and +1 DB per run (Events'
baseline is +0), giving measured totals 12/4/23 → 40/4/95.

## Wiring

- The root `recLCSEdt` (the one `mainEdt` creates) has a 3-slot template
  wired `S` (RO) / `T` (RO) / `score` (RW), but `recLCSEdt`'s body never
  touches `depv[]` — these slots exist only to gate the root call on the
  three `ocrDbCreate`s finishing; the RW hold on `score` is taken and
  released around that single near-instant call, long before any leaf
  runs.
- Internal (non-root, non-leaf) `recLCSEdt` calls carry **no** DB
  dependence at all — the 1-slot template they use is wired to
  `NULL_GUID`/`DB_MODE_NULL` (`x11`, unblocked) or to a sibling's STICKY
  output event (`x12`/`x21` wait on `x11`'s event; `x22` waits on both
  `x12`'s and `x21`'s). `S`/`T`/`score`'s GUIDs travel only as *values*
  inside `LCS_task_params` (`paramv`) — untouched by the coherence
  machinery until a leaf finally acquires them.
- Each leaf `seqLCSEdt` (3-slot template) is wired `S` (RO, slot 0), `T`
  (RO, slot 1), `score` (RW, slot 2) — this is where all real DB traffic
  happens, once per leaf, `4^d` times total.
- **DB concurrency — the point of this variant.** `S` and `T` are RO
  everywhere, so ARTS lets any number of ready leaves read them
  concurrently: up to `min(rows,cols)` of the `2^d × 2^d` leaf-tile grid
  can hold a live RO acquire at once. `score` is RW everywhere it is
  touched, and ARTS's per-node-exclusive RW guarantee (OCR `RW`/`EW` both
  map to it) allows exactly **one** holder at a time — full stop,
  regardless of whether the leaves currently wanting it (up to `2^d` of
  them ready simultaneously, since `x12`- and `x21`-subtree leaves are
  never ordered against each other by the DAG) touch disjoint index
  ranges of the antidiagonal-collapsed array or not. `score` is the
  program's sole contention point: every one of the `4^d` leaf turns is
  serialized onto it, one at a time, independent of node/worker count.

## Flow

`mainEdt` runs entirely on rank 0: creates the 3 DBs, fills `S`/`T` with
`genRandInput` (native, `O(N)`), and — because `CHECK_RESULTS` is always
defined in this file — recomputes the *same* recurrence natively via
`serial_lcs`, a full `O(N²)` double loop over antidiagonals, single-
threaded, entirely before the first `ocrEdtCreate`. Only then is the root
`recLCSEdt` created (`EDT_PROP_FINISH`), which unfolds the quad-tree:
`x11` first, `x12`/`x21` concurrently once `x11`'s whole subtree
completes, `x22` once both of those complete — recursing to depth `d`,
where `4^d` `seqLCSEdt` leaves each take a turn on `score`. `shutDownEdt`
fires once the root's finish scope closes, prints, asserts against the
native `true_value`, and shuts down. The DAG offers up to `2^d`-wide
concurrency in principle (see Wiring), but the single shared `score` DB
reduces the *effective* critical path to `4^d` sequential turns
regardless of available parallelism.

## Placement (as-born)

No `OCR_APP_OPTIMIZED_PLACEMENT` guard exists anywhere in this file —
every `ocrEdtCreate`/`ocrDbCreate` passes `NULL_HINT`. Effective policy:

- **EDTs**: NULL hint → shim's `ARTS_HINT_ANY_RANK` → runtime round-robin.
  Every `recLCSEdt`/`seqLCSEdt` instance lands on an independently chosen
  rank with no relation to its position in the quad-tree.
- **DBs**: NULL hint → home = creating rank. `S`, `T`, and `score` are all
  created inside `mainEdt`, so all three home at rank 0 for the whole run.

Consequence: since leaves scatter round-robin while all three DBs home at
rank 0, most of the `4^d` sequential `score` turns (and most RO `S`/`T`
reads) are remote round trips to rank 0 — the single-DB serialization
described in Wiring is compounded by network latency on top, and adding
nodes only increases the fraction of turns that cross the network without
relieving the serialization itself (an anti-scaling shape by
construction, in the same spirit as `fibonacci`).

## Sizing

`N` and `base` together set `d = ⌈log₂(N/base)⌉`-ish (exact rule: shifts
of `N` until `≤ base`), which sets **both** the DB *payload* (linear in
`N`) and the total leaf/turn count (`4^d`, i.e. ×4 per unit of `d`).
Because the whole run is bottlenecked by one RW-exclusive `score` DB
(Wiring), **more workers or nodes do not shorten the critical path** —
the run's wall time tracks the `4^d` sequential `score` turns, each
paying an acquire/compute/release round trip that is a remote hop most of
the time under as-born placement. Sizing choices here trade run *length*
against how much of that traffic crosses the network, not against
available concurrency:

- Smaller `base` relative to `N` → larger `d` → many more, cheaper turns,
  more of the run's cost is DB-acquire/coherence overhead rather than
  antidiagonal compute — use this to stress the serialization path
  harder.
- Larger `base` relative to `N` → smaller `d` → fewer, larger turns, each
  doing more real antidiagonal work per acquire — closer to a compute
  benchmark, though still capped at one concurrent `score` holder.
- The calibrated `args = [40960, 1024, 48]` gives `d=6` (4,096 leaf
  turns) — a size picked to keep the run observable (seconds-to-minutes)
  while still exercising many turns; it is not chosen to saturate any
  particular node/worker count, since worker count does not move this
  app's parallelism ceiling.
- Memory is never the limit (DB payload is `O(N)`, tens to hundreds of KB
  even at large `N`); leave it out of the sizing decision.
