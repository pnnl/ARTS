# cholesky_blas

*The same tiled-Cholesky DAG as `cholesky`, with every kernel body replaced
by a CBLAS/LAPACKE call — nothing about the wiring changes, only the grain
and what's linked underneath.*
Source: `third_party/ocr-apps/apps/cholesky/ocr-mkl/ocr_mkl_cholesky.c`
(~640 lines).

## Overview

Structurally identical to `cholesky` (same reference — read it first): the
same `t = ds/ts`, the same right-looking `k,j,i` loop nest, the same
`event[row][col][generation]` STICKY-event wiring, the same per-tile
single-writer chain, the same POTRF/TRSM/SYRK/GEMM task shapes and
dependency slots. The headline difference is that each kernel body calls a
standard dense-linear-algebra routine — `LAPACKE_dpotrf`, `cblas_dtrsm`,
`cblas_dsyrk`, `cblas_dgemm` — against the vendored `third_party/OpenBLAS`
submodule (`arts::openblas`, built **single-threaded**, `USE_THREAD=0`, so a
BLAS call never spawns its own threads underneath the per-tile EDT —
task-level parallelism stays the only parallelism), instead of `cholesky`'s
hand-rolled triple loop. A second, less obvious structural difference rides
along with it: unlike `cholesky`, this port's POTRF/TRSM kernels mutate
their acquired tile **in place** rather than creating a fresh output DB (see
Wiring), and its initial-tiling loop stages a `temp2D` pointer-array DB per
tile that `cholesky`'s stride-based `memcpy` doesn't need (see Structure) —
two changes that happen to cancel exactly in the DB *count* (not in what the
DBs *are*), so the measured DB totals below are identical to `cholesky`'s at
the same `t`. The "MKL" in the source path/binary name is historical (the
upstream OCR example targeted Intel MKL); this build links OpenBLAS, an
open-source implementation of the same CBLAS/LAPACKE interface, not MKL
itself. The marker/scalar (`CHOLESKY trace = %.6f`, `tolerance: 1e-09`) and
the EDT/Event formulas below are identical to `cholesky.md`; only the
Parameters differences, the DB-plane details, and the grain/compute contrast
are new here.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `--ds` | matrix size; `t = ds/ts` | required | ✓ `getopt_long`, `atoi` |
| `--ts` | tile size | required | ✓ same |
| `--fi` | input matrix, whitespace-separated text | required | ⚠ **crash risk**: `fileNameIn` is declared with no initializer and never NULL-checked before `fopen(fileNameIn, "r")` — omitting `--fi` (while still passing `--ds`/`--ts`) dereferences an uninitialized stack pointer instead of erring cleanly |
| `--fo` | output file name | `ocr_mkl_cholesky.out` | ✗ **dead AND a heap overflow**: `wrap_up_task`'s paramv (`numTiles`/`tileSize`/`outSelLevel` only) carries no filename, and `wrap_up_task` hardcodes `fopen("ocr_mkl_cholesky.out", "w")` — the parsed value is never used. Getting there is also unsafe: `fileNameOut = realloc(fileNameOut, sizeof(optarg))` reallocates to `sizeof(char*)` (8 bytes, the *pointer's* size, not the string length) before `strcpy`ing the full argument into it — any `--fo` value longer than 7 characters overflows the heap allocation |
| `--ol` | output selection 0–5 | `2` (binary file) | ✓ `atoi`, consumed only by `wrap_up_task` |

Unlike `cholesky`, there is **no `--ps`, `--fib`, or `--convert`** here —
this build only reads matrices as whitespace text (`readMatrix`, malloc'd
row-pointer array + `fscanf`, same unchecked-return-value gap as
`cholesky`'s text path: a short file leaves garbage tail elements, a long
one has its extra numbers silently ignored). `matrixSize % tileSize != 0`
is still rejected before anything is built. `ds`/`ts`/`ol` propagate via
each kernel's paramv (`u64 func_args[]`, not a struct — same effect),
multinode-safe by construction.

## Structure

EDT and Event formulas are identical to `cholesky` — DAG shape depends only
on `t`, never on which kernel implementation runs it. The DB *count* is also
numerically identical to `cholesky`'s, but for a different reason (see
below) — an earlier version of this doc claimed cholesky_blas creates
*additional* ephemeral DBs beyond `cholesky`'s total; that was wrong, and is
corrected here against measured counters:

| object | count | size |
|--------|-------|------|
| EDTs | `2 + t² + C(t,3)` (POTRF `t`, TRSM `t(t-1)/2`, diagonal update `t(t-1)/2`, off-diagonal update `C(t,3)`, + `mainEdt` + `wrap_up_task`) | — |
| DBs (initial-tile payload) | `t(t+1)/2` — created once each in `satisfyInitialTiles`. Unlike `cholesky`, POTRF/TRSM never replace a tile with a fresh output DB — they mutate `depv[0]` in place and satisfy the next generation's event with the *same* GUID (`ocr_mkl_cholesky.c:73-74`, `102-103`; see Wiring) — so this is the *only* payload-DB source here | `ts²·8` bytes each |
| DBs (event-guid metadata) | `1 + t + t(t+1)/2` | negligible |
| DBs (`temp2D` pointer-array staging, one per initial tile) | `t(t+1)/2` — created **and destroyed** within `satisfyInitialTiles` (`ocr_mkl_cholesky.c:463,479` — the only `ocrDbDestroy` call in either app); still counted, since `NUM_DB_CREATE` is a pure creation counter unaffected by a later destroy | `ts·8` bytes each |
| **DBs total** | `1 + t + t(t+1)/2 + t(t+1)` — metadata (`1+t+t(t+1)/2`) plus `t(t+1)` split evenly between initial-tile payload and `temp2D` staging (`t(t+1)/2` each). This is **algebraically the same value** as `cholesky`'s `1+t²+2t+t(t+1)/2` (`t(t+1) ≡ t²+t`): `cholesky` spends that `t(t+1)/2`-sized second term on fresh POTRF/TRSM output DBs (`t+t(t-1)/2=t(t+1)/2` of them) that this port doesn't create, and this port spends the identical count on `temp2D` staging that `cholesky` doesn't need — different objects, same count, verified against measured counters below | — |
| Events | `t(t+1)/2 · (t+1)` STICKY events | — |

Worked numbers at the calibrated `args` (`--ds 7500 --ts 100` → `t=75`):
EDTs = 73,152 (POTRF 75, TRSM 2775, SYRK 2775, GEMM 67,525); DBs = 8626
(2926 metadata + 2850 initial-tile payload + 2850 `temp2D` staging) — the
same total `cholesky` would give at `t=75`, not `5700` payload `+2850`
ephemeral on top as an earlier version of this doc claimed; Events =
216,600. Unlike `cholesky`, no kernel-output dead weight accumulates on the
tile-payload plane (POTRF/TRSM never replace a tile with a fresh DB, so
there is no pre-factor generation left dangling): peak resident
tile-payload memory is just the `t(t+1)/2` live tiles × `ts²·8` bytes ≈ 217
MiB at `t=75` — about half of `cholesky`'s doubled figure at the same `t`.

Counter cross-check: verified (1 node, `--ds 50 --ts 25` (`t=2`) vs `--ds 50
--ts 10` (`t=5`), both against `scratch/cholesky_input.mat`): measured
absolutes EDT 7/38, DB 13/52, EVT 9/90 — **identical to `cholesky`'s
measured absolutes at the same `t`**. Subtracting the runtime's constant
baseline (+1 EDT, +1 DB, +0 EVT per run) gives app-side DB 12/51, matching
`1+t+t(t+1)/2+t(t+1)` exactly: `1+2+3+6=12` at `t=2`, `1+5+15+30=51` at
`t=5`.

## Wiring

The event-wiring skeleton is the same as `cholesky` (STICKY-event indexed
`[row][col][generation]`, `DB_MODE_RW` on a kernel's own tile, `DB_MODE_RO`
on its `L`-factor inputs) — see that page's Wiring section for the full
slot-by-slot breakdown. What differs from `cholesky` is which kernels mutate
in place: here **all four** kernels (POTRF, TRSM, SYRK, GEMM) satisfy the
next generation's event with their own acquired `depv[0]` DB, unchanged
(`ocr_mkl_cholesky.c:73-74` POTRF, `:102-103` TRSM, `:128-129` SYRK,
`:155-156` GEMM) — `cholesky`, by contrast, has POTRF and TRISOLVE create a
*fresh* output DB and only GEMM/SYRK mutate in place. One consequence: a
tile's GUID is fixed for its *entire* life once `satisfyInitialTiles`
creates it — there is no later-generation replacement DB at all in this
port (see Placement). The single-writer-per-tile chain and the `t-1-k` RO
fan-out at level `k` (maximal `t-1` at `k=0`) hold identically here.

## Flow

Same `k,j,i`-loop preamble shape and the same `~3t`-deep serial
POTRF→TRISOLVE→SYRK backbone as `cholesky` (**the k-loop's POTRF is still
serial** — LAPACKE_dpotrf is a single dense factorization call per tile,
not itself parallelized). What changes is the constant behind each hop:
`LAPACKE_dpotrf`/`cblas_dtrsm`/`cblas_dsyrk`/`cblas_dgemm` are blocked,
vectorized BLAS3 routines against the same `O(ts³)` flop count `cholesky`'s
element loops compute — materially cheaper per call at the same `ts`, so
the same peak-width formula (`t(t-1)/2` at `k=0`, decaying quadratically)
carries proportionally less wall-clock weight per level here.

## Placement (base)

Same hint story as `cholesky`: `choleskyTileHint`'s 2-D block-cyclic map is
guarded by `OCR_APP_OPTIMIZED_PLACEMENT` and returns `NULL_HINT` here; every
`ocrDbCreate` also passes `NULL_HINT`. EDTs round-robin
(`ARTS_HINT_ANY_RANK`); all initial tiles and metadata DBs home to rank 0
(created synchronously inside `mainEdt`). The DB-placement *consequence*
differs from `cholesky`, though: since POTRF/TRSM never create a
later-generation output DB here (Wiring), every payload tile's home stays
fixed at rank 0 — where it was created — for the tile's entire life, rather
than migrating to wherever a round-robin-placed POTRF/TRISOLVE happened to
execute. Every kernel instance downstream of level 0 (TRSM/SYRK/GEMM at any
`k>0`) still round-robin-places independently of the tile it touches, so
nearly every dependence edge is still a remote acquire — the level-0
factors remain the widest (`t-1`-reader) contention point — but the
specific remote target for a given tile is deterministic (rank 0) rather
than scattered across whichever rank last wrote it.

## Placement (hinted)

As-born is placement-blind: every kernel EDT scatters round-robin, so at 2
ranks half of all tile acquires cross the wire (measured 2026-08-19, 2x(15w+1p):
50.7% of 216,601 acquires remote, 2.94 GB payload crossed).

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

`ts` is the tile edge and sets grain (`~ts³` flops a kernel, `ts²·8` bytes a
block); `t = ds/ts` sets DAG depth (`t` serial POTRF steps) and peak width
(`t(t-1)/2`).

Width comes from the class rule -- four times the largest geometry's 3456
workers, so 13,824 -- which puts `t` at 167, i.e. `ds = 16700` at `ts = 100`.
The identity input is SPD with a unit factor, so the trace is exactly `ds` and
the pin is analytic rather than measured.

This row anti-scales mildly:

| geometry, `--ds 10000 --ts 100` | time |
|---|---|
| 1 node x 15 workers | 9.51 s |
| 2 nodes | 10.65 s |
| 4 nodes | 12.76 s |

so the window is 10-30 s.

At the calibrated arguments the row measures 31.1 s, 31.2 s and 35.2 s on the
three coherence families, holding 7 GB.

**Deliberate deviation from the window.**  The anchor reaches 35.2 s against a
10-30 s window.  Trimming `ds` to 15000 would fit, but that narrows the width to
0.81x of the rule for a timing reason, which is the one move the width rule
forbids; and this row degrades only 1.36x per node doubling, so a slightly
longer anchor costs nothing at 32 nodes -- unlike tempest, where the window is
what keeps the campaign finite.

Most of the gap to `cholesky` is input, not arithmetic.  This port has no binary
input option, so it parses a 1.12 GB text matrix with `fscanf`, and that parse is
about 65% of the run: the same DAG from the tile stream is 12.3 s.  That is the
application's own structure and the base tier's business to exhibit.

The placement layer clears its gate: 5.35 s to 4.53 s at two nodes and 6.01 s to
5.08 s at four, the same 1.18x both times.
