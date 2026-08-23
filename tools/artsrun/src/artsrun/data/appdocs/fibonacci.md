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
| `argv[2]` = scatter levels | how deep the spawn scatters across ranks before a subtree pins to the rank it landed on | `FIB_RR_LEVELS` (11) | ✓ parsed in `mainEdt`, carried in each task's paramv — no global, so every rank sees it |

The scatter depth used to be a compile-time constant.  It is the app's only
parallelism dial and is calibrated by measurement, so it is an argument; the
`#define` survives as the default when the argument is absent, and the base
build ignores it either way.

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

## Placement (base)

The source passes `NULL_HINT` on every create.  Effective policy:

- **EDTs**: shim passes `ARTS_HINT_ANY_RANK` → runtime round-robin (per-rank
  atomic counter, modulo rank count) — `fibEdt`, `complete` and the checker
  all land on arbitrary ranks.
- **DBs**: NULL hint → home = creating rank (creator/first-touch).

Consequence at multinode: a child EDT rarely lands where its 4-byte argument
DB was created, and a `complete` rarely lands where any of its three DBs
live, so nearly **every dependence edge is a remote acquire of a 4-byte
block**.  The app is a worst-case fine-grain coherence stress by
construction; locality exists in the algorithm (subtrees) but the base
program never expresses it.

## Placement (hinted)

As-born every create is `NULL_HINT`: EDTs round-robin (each recursion child on
an arbitrary rank), the 4-byte argument DBs home on their creating rank.  The
work itself is negligible per task, so the whole cost of the program is
wherever the tree's edges cross ranks.

The layer (`OCR_APP_OPTIMIZED_PLACEMENT` in `fib.c`) makes the crossing edges a
prefix of the tree: children at level <= the scatter depth (argument, default
11) are placed by the child's deterministic path id; every deeper child pins to
its creating rank, so each scattered subtree runs wire-free below its root.
The `complete` (sum) EDT pins to the creating rank.

**The path id is hashed before the modulus, and that is not cosmetic.**  A path
id is the branch sequence read as a binary number, so a raw `% nranks` aliases
with the tree's own shape — and this tree is asymmetric (`fib(n-1)` dwarfs
`fib(n-2)`), so the alias puts unequal subtrees on the same rank.  Measured at
4 nodes with the placement counters: raw modulus gave 2.62x spread in
`NUM_EDT_FINISH`, **2.92x in `TIME_EDT_EXEC`** (one rank doing 104 s of work
against another's 36 s) and 91x in steal attempts — three ranks spinning for
work while the first drowned; two ranks even landed on identical counts, the
signature of an aliasing key.  Mixing first (the same finalizer used by
`nqueens`) brings those to **1.13x / 1.15x / 15x** and the cell from 7.19 s to
**4.82 s**.  The argument DBs stay `NULL_HINT` deliberately: the
runtime's no-hint DB home is the creator, which is exactly the one-shot
small-DB placement the 2026-07-10 pin experiments showed this class needs.

## Sizing

`n` is the only dial, and it scales *work*, not per-task size:

- Pick `n` so total EDTs (`3·F(n+1)`) ≫ total workers; ~10⁴ EDTs per worker
  keeps every deque busy through the fold-up wave.  Measured at the Dane
  anchor node (108w+4p): 30 → 0.83 s, 33 → 3.6, 36 → 15.7, 38 → 39.9,
  40 → 105.8, **41 → 173.4**.  The calibrated argument is `41`.
- Scatter depth `11` gives 2048 distinct paths — 64 per rank at 32 nodes,
  enough for the asymmetric tree to average out.  It sits on the measured
  plateau: at 4 nodes 9/11/14 give 4.75/4.77/4.89 s and at 8 nodes
  2.26/2.28/2.37; only `6` (6.02 s at 4 nodes) and `16` fall off.
- Memory is never the limit; leave it out of the sizing decision.
- **The hinted tier is what the app is for.**  Base at `n = 38`: 78.9 s at
  one bentley node, and TIMEOUT past 400 s at two and at four — every
  recursion child is a remote spawn.  Hinted at the same size: 78.8 / 43.2 /
  25.0.  At the trend size (`36`, scatter 11) the hinted tier runs
  24.2 / 10.6 / 4.8 / 2.3 s over 1/2/4/8 nodes — **10.5x on 8 nodes**, and
  all four coherence arms land within 3% of each other, because 99.99% of
  its acquires are local hits.
