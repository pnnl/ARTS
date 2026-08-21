# cholesky

*Tiled Cholesky factorization over hand-rolled element loops — POTRF, TRSM,
SYRK and GEMM as four EDT kinds wired into a right-looking `lkji` DAG.*
Source: `third_party/ocr-apps/apps/cholesky/ocr/cholesky.c` (~1035 lines; the
`ocr/` directory is a symlink to `apps/examples/cholesky/`).

## Overview

Factors a symmetric positive-definite matrix `A = L·Lᵀ` by right-looking
tiled Cholesky: the `ds×ds` matrix is split into `t = ds/ts` tiles per
dimension, and the classic `k`-outer loop (`POTRF(k,k)` → `TRSM(k,j)` for
`j>k` → `{GEMM,SYRK}` updates of the trailing `(t-1-k)×(t-1-k)` submatrix)
is unrolled into four EDT kinds. All four kernels are hand-written triple
nested loops over `double` elements (no BLAS) — see `cholesky_blas` for the
CBLAS/LAPACKE port of the identical DAG. The marker line
(`CHOLESKY trace = %.6f`) is the trace of the computed factor `L`, printed
by `wrap_up_task` once every tile is final; the calibrated `expect_args`
matrix is a 50×50 identity (trivially SPD, `L = I`, trace = 50 — matches
the catalog's `expect: '50'`). Because the per-tile kernels are real
`O(ts³)` dense-linear-algebra loops, this is a genuine FLOPS benchmark, but
the wiring pattern (single-writer-per-tile chains, a serial `t`-deep
POTRF/TRSM/SYRK backbone) also makes it a real DAG-scheduling and
data-movement stress at the same time.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `--ds` | matrix size (rows = cols); `t = ds/ts` tiles per dimension | required | ✓ `getopt_long` in `mainEdt`, `atoi` |
| `--ts` | tile size (each tile is `ts×ts` doubles) | required | ✓ same |
| `--fi` | input matrix, whitespace-separated text, read via `fscanf` | none (alternative to `--fib`) | ✓; **no size validation** — see below |
| `--fib` | input matrix as the tile-stream binary (`convertData`'s output: lower-triangular tiles in `(i, j≤i)` order, `ts²` doubles each, host order) | none | ✓; strictly validates file size `== t(t+1)/2·ts²·8` bytes, aborts cleanly on mismatch; streams tiles straight into their datablocks — no whole-matrix host buffer |
| `--ps` | print a status line per kernel EDT to stdout | `0` (off) | ✓ `atoi`; pure stdout verbosity, no DAG effect |
| `--ol` | output selection 0–5 (stdout / text file / binary file / both / binary+timing-CSV) | `2` (binary file `cholesky.out`) | ✓ `atoi`; consumed only by `wrap_up_task`, no DAG-shape effect |

`matrixSize % tileSize != 0` is rejected before anything is built. All four
values are threaded through each kernel's `*PRM_t` struct as EDT paramv —
propagated to wherever the EDT executes, never read from a C global, so
this is multinode-safe by construction.

**`--ds`/`--fi` mismatch**: the text (`--fi`) reader always requests exactly
`ds²` `fscanf` calls and never checks the return value — a file with *fewer*
numbers than `ds²` leaves the tail of the (unzeroed) `malloc`'d matrix as
garbage with no error; a *longer* file silently has its extra numbers
ignored. The binary (`--fib`) reader is the opposite: `fstat` must match
the tile stream's exact byte count, else the run aborts with a clean
message before building any EDTs — and because the tile size shapes the
stream, a `.bin` baked for one `ts` is rejected under another. The catalog
uses `--fi` for the correctness cell and the tile stream for the perf
cells; the standalone `convertData`/`convertOut` tools (built as
`cholesky_convertData`/`cholesky_convertOut`) convert a text matrix into
the stream and the binary result file back into text.

## Structure

Let `t = ds/ts`. Kernel EDT counts fall out of the `k,j,i` loop nest
directly (`C(t,3) = t(t-1)(t-2)/6`, the number of `k<i<j<t` triples):

| object | count | size |
|--------|-------|------|
| EDTs | `2 + t² + C(t,3)` — POTRF `t`, TRSM `t(t-1)/2`, diagonal update (SYRK) `t(t-1)/2`, off-diagonal update (GEMM) `C(t,3)`, plus `mainEdt` + `wrap_up_task` | — |
| DBs (tile payload) | `t² + t` — initial tiles `t(t+1)/2`, POTRF output `t`, TRSM output `t(t-1)/2` (the GEMM/SYRK updates mutate their input tile **in place** and create nothing) | `ts²·8` bytes each |
| DBs (event-guid metadata) | `1 + t + t(t+1)/2` (outer array, per-row arrays, per-tile generation arrays) | negligible (8-byte guids/pointers) |
| Events | `t(t+1)/2 · (t+1)` STICKY events (one 0…`t` generation array per lower-triangular tile) | — |
| EDT templates | 5, created once, never destroyed | — |

Worked numbers at the calibrated `args` (`--ds 5000 --ts 100` → `t=50`):
EDTs = 22,102 (POTRF 50, TRSM 1225, SYRK 1225, GEMM 19,600); DBs = 3876
(2550 payload + 1326 metadata); Events = 65,025. At `expect_args`
(`--ds 50 --ts 10` → `t=5`): EDTs = 37, DBs = 51, Events = 90.

No DB, metadata DB, or event is ever destroyed (`ocrDbDestroy` /
`ocrEventDestroy` do not appear in this file). Every tile position passes
through exactly one "pre-factor" DB generation before POTRF/TRISOLVE
replaces it with a fresh one — that pre-factor DB becomes dead weight the
instant it's replaced, so peak resident tile-payload memory is
`≈ 2·(t(t+1)/2)·ts²·8` bytes, roughly double what `wrap_up_task` actually
needs (`t(t+1)/2` final tiles).

Counter cross-check: verified (1 node, `--ds 50 --ts 25` (`t=2`) vs `--ds 50
--ts 10` (`t=5`), both against `scratch/cholesky_input.mat`): measured
absolutes EDT 7/38, DB 13/52, EVT 9/90; subtracting the runtime's constant
baseline (+1 EDT, +1 DB, +0 EVT per run) gives app-side EDT 6/37, DB 12/51,
EVT 9/90 — exactly `EDT(t)`/`DB(t)`/`Event(t)` above at `t=2` and `t=5`. The
event formula in particular needs no baseline adjustment at all (`Event(t) =
t(t+1)²/2` reproduces the measured absolute directly).

## Wiring

- Wiring runs through STICKY events indexed `[row][col][generation]`
  rather than direct DB→EDT deps, because a tile's next consumer isn't
  created until later in `mainEdt`'s loop. `DB_MODE_RW` always targets the
  kernel's own output-in-progress tile (slot 0); `DB_MODE_RO` always
  targets a finished `L`-factor input.
- POTRF(k): RW `event[k][k][k]` → new DB → satisfies `event[k][k][k+1]`.
  TRISOLVE(k,j): RW `event[j][k][k]` + RO `event[k][k][k+1]` → new DB →
  satisfies `event[j][k][k+1]`. GEMM(k,j,i): RW `event[j][i][k]` + RO
  `event[j][k][k+1]` + RO `event[i][k][k+1]` → **same** DB, mutated in
  place → satisfies `event[j][i][k+1]`. SYRK/diagonal-update(k,j): RW
  `event[j][j][k]` + RO `event[j][k][k+1]` → same DB in place → satisfies
  `event[j][j][k+1]`. `wrap_up_task` RO-reads all `t(t+1)/2` final
  `event[i][j][j+1]`.
- Per-tile ownership is a strict single-writer chain: exactly one RW
  holder exists at a time per tile, handed off generation-by-generation —
  no DB is ever concurrently RW-held by two tasks, so there is no
  write-write race anywhere in this app by construction.
- Read fan-out: at level `k`, both the diagonal factor `L(k,k)` and every
  off-diagonal factor `L(j,k)` (`j>k`) are each read RO by exactly
  `t-1-k` downstream tasks — maximal at `k=0` (`t-1` simultaneous
  readers of a single DB), the natural multinode contention point (see
  Placement).
- `wrap_up_task` is the one many-to-one EDT in the graph — a single sink
  joining all `t(t+1)/2` final tiles.

## Flow

`mainEdt` is a single-threaded, rank-0 synchronous preamble: it stages
every initial tile (`t(t+1)/2` `ocrDbCreate`+`ocrEventSatisfy` pairs),
allocates the whole 3-D event-guid structure, then walks the `k,j,i` loop
issuing every kernel's `ocrEdtCreate`+`ocrAddDependence` calls — `O(t³)`
calls total, dominated by the `C(t,3)` GEMM wiring, all before `mainEdt`
returns. Dependencies resolve incrementally as they're wired, so POTRF(0)
and later kernels can start executing on other workers while `mainEdt` is
still issuing later-`k` wiring — but the wiring itself is single-thread,
single-rank work that grows cubically with `t`.

The outer `k`-loop (`t` iterations) is the true sequential backbone:
POTRF(k) needs the `(k,k)` tile to have absorbed all `k` of its diagonal
updates, each of which needs the previous level's TRISOLVE, which needs
the previous POTRF — a minimum critical path of `~3t` EDT-hops that no
amount of parallelism shortens (**the k-loop's POTRF is serial**). Within
one level, fan-out is real: `t-1-k` independent TRISOLVE tasks, then up to
`(t-1-k)(t-k)/2` mutually-independent GEMM/SYRK tasks (each owns a
distinct output tile). With no phase barrier, a level's tail can overlap
the next level's head once that level's one relevant diagonal update
finishes, so peak instantaneous width across the run is `max_k
(t-1-k)(t-k)/2 = t(t-1)/2` at `k=0`, decaying roughly quadratically as `k`
grows.

## Placement (base)

`choleskyTileHint` (a 2-D block-cyclic, ScaLAPACK-style owner map) is
guarded by `OCR_APP_OPTIMIZED_PLACEMENT` and returns `NULL_HINT` on every
EDT create in this build; every `ocrDbCreate` call also passes `NULL_HINT`
directly. Effective policy:

- **EDTs**: NULL hint → shim's `ARTS_HINT_ANY_RANK` → runtime round-robin
  (per-creating-rank atomic counter) — every POTRF/TRISOLVE/GEMM/SYRK/
  wrap_up instance lands on an arbitrary rank, uncorrelated with the tile
  it touches.
- **DBs**: NULL hint → home = creating rank. All `t(t+1)/2` initial tiles
  and all `1+t+t(t+1)/2` metadata DBs are created synchronously inside
  `mainEdt`, so they all home to rank 0 — every level-0 TRISOLVE/GEMM/SYRK
  instance's first read is a remote acquire back to rank 0 unless it
  happened to round-robin there itself. Later-generation output DBs
  (POTRF's/TRISOLVE's new blocks) home wherever their producing EDT
  executed — round-robin, still uncorrelated with tile adjacency.

Consequence: nearly every dependence edge is a remote acquire; the
algorithm's real locality (a tile's consumers are its own row/column) is
never expressed by placement, and the widest read fan-out (up to `t-1`
simultaneous RO acquires of one DB) lands wherever that single DB's home
happens to be — a worst-case coherence stress layered on top of genuine
FLOPS.

## Placement (hinted)

As-born is placement-blind: every kernel EDT scatters round-robin, so at 2
ranks half of all tile acquires cross the wire (measured 2026-08-19, 2x(15w+1p):
51.0% of 65,026 acquires remote, 0.92 GB payload crossed).

The layer (`choleskyTileHint`) is a 2-D block-cyclic (ScaLAPACK) owner map:
factor the rank count into a near-square P x Q grid, place tile (row,col) on
rank `(row % P) * Q + (col % Q)`, and key every kernel EDT on the coordinate of
the single tile it WRITES — potrf/trsm/gemm each co-locate with their RW
output, so the per-tile ownership acquire settles locally and stays there
across generations.  Two independent mod axes keep the active trailing
submatrix spread over all ranks instead of folding a frontier onto one.  DB
hints are not used (ownership follows the pinned EDTs).  `nranks <= 1` returns
`NULL_HINT` so a single-node run is bit-identical to base.

## Sizing

`ts` sets task grain (compute `~ts³` flops per kernel call via the
element-wise triple loop; DB size `ts²·8` bytes); `ds` (via `t=ds/ts`) sets
DAG depth (`t` serial POTRF steps) and peak width (`t(t-1)/2`). Both
calibrated configurations keep `t` comfortably above the 128-physical-core
machine, so worker count is not the bottleneck — wall time is set by the
`~3t`-deep serial critical path and by per-tile compute (expensive here;
`cholesky_blas` trades that for BLAS3 kernels at the same DAG shape).

- Given `N` nodes × `C` workers (≤128 total), peak width `t(t-1)/2 ≳ N·C`
  already holds at `t ≳ 17` for the largest configured machine (8×15=120),
  so `ts` is really chosen for compute grain and memory, not parallelism.
- 1 node × 15 workers: any `t ≳ 6` (`t(t-1)/2 > 15`) saturates the workers
  at peak width; `--ds 5000 --ts 100` (`t=50`) is comfortably oversized
  for strong scaling up to 8 nodes.
- Memory: peak resident tile payload is `≈2·(t(t+1)/2)·ts²·8` bytes
  (Structure's dead-weight note) — ≈195 MiB at the calibrated `t=50`,
  never the limiting factor on this machine.
- Output byproducts (`cholesky.out`, and `ocr_cholesky_stats.csv` under
  `--ol 5`) land in the process's working directory, not under `--ds`/
  `--ts` control.
