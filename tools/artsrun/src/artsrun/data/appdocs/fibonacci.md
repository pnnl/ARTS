# fibonacci

*Recursive Fibonacci over EDTs — every call is a task, every argument a
datablock.*
Source: `third_party/ocr-apps/apps/fibonacci/ocr/fib.c` (~280 lines).

## Overview

Computes `fib(n)` by literal binary recursion: each `fibEdt(n)` with `n >= 2`
spawns `fibEdt(n-1)` and `fibEdt(n-2)` plus a `complete` EDT that sums their
results; `n < 2` is a leaf.  The result scalar (`answer is N`) is checked
inside the program against a natively computed reference before
`ocrShutdown()`.  There is no compute payload — each EDT does a handful of
instructions — so the program measures task creation, scheduling and
fine-grain data movement, not arithmetic.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|------------------|
| `argv[1]` = `n` | recursion input; sizes the whole graph exponentially | 10 (with a usage note) | ✓ parsed in `mainEdt` via `ocrGetArgv`, propagated through the argument DB — multinode-safe |

No other runtime knobs.  `FIB_RR_LEVELS` (= 11) is a compile-time constant of
the *optimized* placement layer only; the as-born build does not read it.

## Structure

With `F` the Fibonacci sequence (`F(1) = F(2) = 1`) and `I(n) = F(n+1) - 1`
the number of internal (`n >= 2`) calls:

| object | count | size |
|--------|-------|------|
| EDTs total | `3·F(n+1)` — one `fibEdt` per call (`2·F(n+1) - 1`), one `complete` per internal call (`I(n)`), plus `mainEdt` and the final checker | — |
| DBs | `2·F(n+1) - 1` — one argument/result block per call | 4 bytes each |
| Events | `2·F(n+1) - 1` ONCE events (TAKES_ARG) | — |
| EDT templates | created and destroyed around every EDT create (pure churn) | — |

Examples: `n = 20` → ~32.8k EDTs; `n = 28` → ~1.54M; `n = 33` (calibrated
args) → ~17.1M EDTs, ~11.4M DBs.  Each +1 on `n` multiplies everything by
~1.618; +5 is ~×11.  Payload memory is negligible (4 B per DB); footprint is
runtime metadata per object, and blocks are destroyed as `complete` EDTs
consume them, so the live set tracks the active frontier rather than the
total.

Counter cross-check: verified (1 node, `n=10` vs `n=12`): ΔNUM_EDT_CREATE
= 432, ΔNUM_DB_CREATE = ΔNUM_EVENT_CREATE = 288, exactly the formulas'
deltas; the runtime adds a constant baseline of +1 EDT and +1 DB per run.

## Wiring

- `fibEdt(n)` (internal) creates: two argument DBs holding `n-1`/`n-2`, two
  child `fibEdt`s (each wired to its argument DB, RO), two ONCE events, and
  one `complete` EDT with three slots — slot 0/1 the child events (RO), slot
  2 its own argument DB (RW, the result carrier).
- A leaf (`n < 2`) satisfies its parent event with its own argument DB — the
  stored value already equals `fib(n)` for `n ∈ {0,1}`.
- `complete` writes `in1 + in2` into slot 2, destroys both child DBs,
  releases slot 2 and satisfies the parent's event with it.
- The root event feeds the final checker (RO), which verifies and shuts down.

Every DB access is dataflow-ordered — create-write → child RO read →
`complete` RW write → parent RO read → destroy.  No DB ever has two
concurrent accessors; there is no sharing fan-out, so all coherence traffic
is pure migration of 4-byte blocks.

## Flow

`mainEdt` first computes the expected answer by native recursion — an
exponential *serial* preamble on the rank-0 worker (~same call count as the
whole DAG, but at nanoseconds per call) — then seeds the root.  The tree then
unfolds: the active frontier grows ~×1.6 per level down to the leaves
(`F(n+1)` of them, i.e. millions at calibrated size), and a completion wave
of `complete` EDTs folds values back up over `n` levels.  Parallelism is
never the constraint; per-task runtime overhead is the entire cost.

## Placement (as-born)

The source passes `NULL_HINT` on every create.  Effective policy:

- **EDTs**: shim passes `ARTS_HINT_ANY_RANK` → runtime round-robin (per-rank
  atomic counter, modulo rank count) — `fibEdt`, `complete` and the checker
  all land on arbitrary ranks.
- **DBs**: NULL hint → home = creating rank (creator/first-touch).

Consequence at multinode: a child EDT rarely lands where its 4-byte argument
DB was created, and a `complete` rarely lands where any of its three DBs
live, so nearly **every dependence edge is a remote acquire of a 4-byte
block**.  The app is a worst-case fine-grain coherence stress by
construction; locality exists in the algorithm (subtrees) but the as-born
program never expresses it.

## Sizing

`n` is the only dial, and it scales *work*, not per-task size:

- Pick `n` so total EDTs (`3·F(n+1)`) ≫ total workers; ~10⁴ EDTs per worker
  keeps every deque busy through the fold-up wave.  1 node × 15 workers:
  `n ≈ 28–30` (1.5–4M EDTs).  8 nodes × 120 workers: `n ≈ 32–34` (6.5–28M).
  The calibrated strong-scaling argument is `33` (~17.1M EDTs), sized so the
  1-node run takes minutes rather than seconds.
- Wall time ≈ total EDTs × per-EDT overhead / (nodes × workers) — but at
  multinode the remote-edge cost above, not worker count, dominates; expect
  anti-scaling in the naive placement and treat the app as a scheduler/
  coherence probe, not a FLOPS benchmark.
- Memory is never the limit; leave it out of the sizing decision.
