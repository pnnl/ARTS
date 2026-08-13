# sar_tiny

*Synthetic-aperture radar change detection — tiled backprojection, affine
registration, coherent change detection and CFAR, every stage driven through
one whole-image datablock.*
Source: `third_party/ocr-apps/apps/sar/ocr/src/` (11 files, ~4.6k lines).
All five SAR rows share these sources; `sar_tiny` is the build that compiles in
`ocr/tiny/Parameters.h` and links the `tiny` dataset.

## Overview

Forms two SAR images from a pulse stream and reports how many pixels changed
between them.  Per image: `ReadData` fills the pulse-return block `X` (plus
platform positions `Pt`, timestamps `Tp`), `FormImage` copies the previous
image into `refImage` and zeroes `curImage`, and `BackProj` fans out 4
tiles that each accumulate all 132 pulses into their `32×32` pixels.  After
the second image, `Affine` registers current against reference (2-D
correlation at `N²` control points → 6-parameter least-squares warp → tiled
resampling), `CCD` builds a normalized correlation map, `CFAR` declares a
detection wherever the cell under test is less correlated than its local
clutter ring, and `post_CFAR` prints `SAR detects: <Nd>` — the catalog marker
and scalar — and writes the detects to `Detects.txt` in the working directory.
Provenance: the Georgia Tech Research Institute Streaming Sensor Challenge
Problem reference, ported to OCR through the `RAG_*` macro layer.  `Nd` is a
deterministic function of the dataset (every accumulation that could reorder
is tile-private or an exact integer sum carried in `double`), so the catalog's
`expect: '12'` is a hard equality.  Arithmetic is real — 0.54 M complex
MACs of backprojection per image — but structurally this is a handful of
whole-image blocks read by thousands of tile tasks: it stresses broadcast of
large read-only blocks and exclusive re-acquisition of one image block per
tile.

## Parameters

`sar_tiny` takes **no command-line arguments**; the catalog's `args: []` is
literal.  `mainEdt`'s argv-parsing block sits inside
`#ifndef RAG_IMPLICIT_INPUTS` (`main.c:156-208`) and this target is built
*with* it: the dataset is linked in as `.rodata`, the detects path is the
compiled `argv_4` default, and every tunable is compile-time.

| knob | meaning | value here | CLI reachability |
|------|---------|------------|------------------|
| `argv[1..4]` | data / platform / pulse-time / detects paths | — | ⚠ parsed only in the runtime-input build (`sar_pss`); silently ignored here |
| `RAG_Ix`, `RAG_Iy` | image pixels | 64 × 64 | ✗ `ocr/tiny/Parameters.h` |
| `RAG_P1`, `RAG_S1` | pulses per image; complex samples per pulse | 132, 128 | ✗ same; must match the linked dataset |
| `RAG_Sx`, `RAG_Sy` | spotlight subimage | 64 × 64, so `TF = Ix/Sx = 1` | ✗ same; `TF > 1` aborts ("not yet supported") |
| `RAG_Nc` | affine control-point budget | 8 → `N = ⌊√Nc⌋ = 2` per axis | ✗ same |
| `RAG_Sc`, `RAG_Rc`, `RAG_Tc` | correlation window / search radius / accept threshold | 7, 8, 0.7 | ✗ same |
| `RAG_Ncor`; `RAG_Ncfar`, `RAG_Nguard`, `RAG_Tcfar` | CCD window; CFAR window / guard cells / percentile | 5; 25, 17, 90 | ✗ same |
| `RAG_NumberImages` | images to process | 2 — **only 2 works**; any other value is a run-time error in `main_body_edt` | ✗ same |
| `RAG_NEW_BLK_SIZE` | tile edge for every parallel family | 32 | ✗ `benchmarks/apps/CMakeLists.txt` |
| `DEBUG_SSCP` | dump image/correlation planes | off | ✗ deliberately left out of every variant (hundreds of MB per run) |
| `RAG_DIG_SPOT_ON`, `RAG_QSORT_ON`, `RAG_AFFINE_ON` | spotlighting / detect sort / registration | 0, 0, 1 | ✗ CMake |

Changing the problem size means building a different size target, not passing
an argument — which is why the sizes are separate catalog rows.

## Structure

With `Ix = Iy = 64`, tile edge `B = 32`, `N = ⌊√Nc⌋ = 2`, and the three
tiled window grids `T_bp = ⌈Ix/B⌉·⌈Iy/B⌉ = 4`,
`T_ccd = ⌈(Iy−Ncor+1)/B⌉² = 4`,
`T_cfar = ⌈(Iy−Ncor−Ncfar+2)/B⌉² = 4`:

| object | count | here | size |
|--------|-------|------|------|
| stage heads | 18 (`post_main`, `main_body`, `ReadData`×2, `FormImage`×2, `post_FormImage`×2, `BackProj`×2, `Affine`, `post_Affine`, `post_affine_async_1/2`, `CCD`, `CFAR`, `post_CFAR`, plus `mainEdt` itself — the OCR shim creates it as an EDT before its body runs (`arts_ocr.c:2194`), so it is not one of `mainEdt`'s own explicit `ocrEdtCreate` calls and the stage-head count above missed it) | 18 | — |
| tile EDTs | `2·T_bp` backprojection + `N²` correlation + `T_bp` resample + `T_ccd` CCD + `T_cfar` CFAR | 8 + 4 + 4 + 4 + 4 | — |
| global DBs | 14, all in `mainEdt` | 14 | `X` 133 KiB, `curImage`/`refImage` 32.5 KiB each, `corr_map` 42.7 KiB, `Y` 15.2 KiB, `Pt`/`Tp`/axis vectors, `file_args` 4 KiB, four parameter blocks (12–88 B) |
| `Affine` DBs | `5 + N²` | 9 | `Fx`/`Fy` (`Nc·4`), `A` (`Nc·32`), `output` 32.5 KiB, 56 B per control point |
| per-task scratch | `2·N²` correlation windows (392 B) + `6·T_bp` backprojection (8.4 KiB tile copy, two 32-vectors) + `T_cfar` clutter windows (2.7 KiB) + 5 fixed | 41 | — |
| EDT templates | 21 per rank | 21 | into a fixed `templateList[25]`, bounds-checked on every claim |

Totals: `18 + 3·T_bp + N² + T_ccd + T_cfar` = **42 EDTs**,
`24 + 3·N² + 6·T_bp + T_cfar` = **64 DBs**, and **20 events regardless of
size** — the app never calls `ocrEventCreate`, and the 10 `ocrEdtCreate` calls
that pass a non-NULL `outputEvent` are exactly the 10 that carry
`EDT_PROP_FINISH` (`main_body`, `Affine`, `post_affine_async_1/2`,
`post_Affine`, `CCD`, `CFAR`, `post_CFAR`, `BackProj`×2), giving 10 output +
10 finish events; every other create passes `NULL`.

Live set ≈ 0.4 MiB, dominated by `X` and the three image-shaped blocks — and it
is also the *total* allocated set, because `main.c`, `back_proj.c` and
`registration.c` each `#define` `bsm_free`/`dram_free`/`spad_free` to nothing;
only `cfar.c` destroys anything (a clutter window per CFAR tile, plus `Nd`).
The binary additionally carries 264 KiB of `.rodata` — the linked dataset as
`image_0/1`, `platform_0/1`, `pulse_0/1`.

Counter cross-check: verified (1 node, `sar_tiny` vs `sar_small`, counters ONCE,
measured totals 43/115 EDTs, 65/185 DBs, 20/20 events). NUM_DB_CREATE matches
`24 + 3·N² + 6·T_bp + T_cfar` (64/184) exactly plus the runtime's constant +1
DB per run; NUM_EVENT_CREATE matches the fixed 20 exactly, zero offset;
NUM_EDT_CREATE needed the `mainEdt` correction in the stage-heads row above —
it matches `18 + 3·T_bp + N² + T_ccd + T_cfar` (42/114) plus that same
constant +1 EDT per run, i.e. exactly the measured 43/115.

## Wiring

There is not one explicit event in the program: stages chain through **finish
events**, each head a finish EDT whose fan-out lives in its scope and whose
event lands on the next head's last dependence slot (`main.c:1180-1185`).
`post_FormImage` closes an imaging round by wiring the refilled `X`/`Pt`/`Tp`
into the next `ReadData` (first round) or `curImage` into `Affine` (second).

| datablock | RW writers | RO readers | max concurrent readers |
|-----------|-----------|------------|------------------------|
| `X`, `Pt`, `Tp` | `ReadData` (1) | `backproject_async` | 4 |
| `curImage` | `backproject_async` (4), `post_Affine` (1) | `affine_async_1`, `affine_async_2`, `ccd_async` | 4 |
| `refImage` | `FormImage` (1) | `affine_async_1`, `ccd_async` | 4 |
| `output` | `affine_async_2` (4) | `post_Affine` | 1 |
| `affine_params`, `Fx`, `Fy`, `A` | `affine_async_1` (4) | `post_affine_async_1/2` | 1 |
| `corr_map` | `ccd_async` (4) | `cfar_async` | 4 |
| `Y`, `Nd` | `cfar_async` (4) | `post_CFAR` | 1 |
| `image_params` | `ReadData` (twice) | *every* task in the program | all of them |

Two facts dominate.  **Every parallel family writes one whole-image block**,
and OCR `RW` is per-node exclusive — 4 backprojection tiles all take
`curImage` RW, 4 CCD tiles `corr_map`, 4 CFAR tiles `Y` and `Nd`,
4 correlation tasks `affine_params`/`Fx`/`Fy`/`A`.  Tiles are round-robin
placed, so each of those blocks is written from every node's tasks and a
family can never write on two nodes at once.  `curImage` is the contention
point: 8 exclusive acquisitions of a 32.5 KiB block in backprojection alone,
over disjoint pixel ranges — pure protocol cost, not an algorithmic
dependence.  And the fan-out is genuinely broadcast-shaped: `X` (133 KiB) read
concurrently by all 4 tiles of a round, `curImage`/`refImage` by up to
4 tasks, and the 88-byte `image_params` by essentially every task —
the widest fan-out object in the program, and one `ReadData` takes RW twice.
Tiles sharing a writable block coordinate with plain atomics on the shared
copy (`__sync_fetch_and_add` on `affine_params->Nc`, on `Nd`), varying the
*order* of rows and detects but not the counts or the fitted warp.  2-D blocks
are packed (row-pointer table then payload) and consumers rebuild that table
task-locally (`RAG_REMAP_2D`), so a reader never dirties a block it reads.

## Flow

Strictly serial stages, each a one-EDT head fanning out to a tile family and
joining on its own finish event:

1. `mainEdt` (rank 0, serial) — 14 `ocrDbCreate`s, axis vectors, `Detects.txt`
   truncated for validation.
2. `refReadData` → `refFormImage` — blit 133 KiB of pulse data out of `.rodata`,
   zero `curImage` (32.5 KiB).  Both serial.
3. `BackProj` → **4 `backproject_async`** — the heaviest stage; each tile
   sweeps all 132 pulses over `32×32` pixels, ≈ 135 k complex MACs.
4. `post_FormImage` → `ReadData` → `FormImage` (which now really does the
   `curImage → refImage` copy, 32.5 KiB) → `BackProj` → **4 tiles** again.
5. `Affine` → **4 `affine_async_1`**, one per control point,
   `(2·Rc+1)²·Sc²` ≈ 14 k operations each.
6. `post_affine_async_1` — serial `A'A`/`A'F` accumulation and two 6×6
   Gaussian eliminations → **4 `affine_async_2`** resampling tiles.
7. `post_affine_async_2` → `post_Affine` — serial full-image copy
   `output → curImage` (32.5 KiB).
8. `CCD` → **4 `ccd_async`**; `CFAR` → **4 `cfar_async`**;
   `post_CFAR` writes `Nd` lines; `post_main` shuts down.

Max parallel width is `max(T_bp, N², T_ccd, T_cfar)` = **4**, but what
bounds how much machine a rung keeps busy is the *narrowest* wide stage,
**4**.  Serial bottlenecks in cost order: the two `ReadData` blits, the
`FormImage` copy-and-zero pair, `post_Affine`'s full-image copy, `post_CFAR`'s
line-by-line write.  Stage boundaries are hard barriers, so no two tile
families overlap and every serial head is fully exposed.

## Placement (as-born)

Every `ocrEdtCreate` passes `NULL_HINT`, and every DB is created through
`bsm/dram/spad_malloc`, which pass `NULL_HINT` too (`rag_ocr.c:22-24`).  The
five tile-EDT creates route their hint through `ragTileEdtHint()`, guarded by
`OCR_APP_OPTIMIZED_PLACEMENT` and returning `NULL_HINT` as-born (the
`sar_tiny_opt` family compiles it in — the catalog's `optimized: true`).  So:
**EDTs** → the shim passes `ARTS_HINT_ANY_RANK` → runtime round-robin, and all
24 tile tasks plus every stage head land on arbitrary ranks; **DBs** →
home = creating rank, and since `mainEdt` runs on rank 0 all 14 global blocks
(`X`, both images, `corr_map`, `Y`, the parameter blocks) are homed there,
while `Affine_edt` and the two `BackProj_edt`s are round-robin so `output`,
`A`, `Fx`, `Fy` and the per-tile scratch home on scattered ranks.

Consequence: tiles spread uniformly over all `n` ranks, so every large
read-only block must materialise on every rank that runs a tile, and the
whole-image block a family writes changes owner on nearly every one of its
per-tile RW acquisitions.  The algorithm has ideal tile locality (disjoint
output tiles over shared read-only pulse data) and the as-born program
expresses none of it — not merely because hints are absent, but because a
stage's entire output is one datablock, so no placement could let two ranks
write it concurrently.

## Sizing

Nothing about a SAR run is chosen at launch: the four compiled-in rows are one
program at four scales and the ladder *is* the target list
(`sar_tiny → sar_small → sar_medium → sar_large`, continuing past `large` only
through `sar_pss`, which reads its parameters from a file).  A campaign should
pick **one** rung; running several multiplies machine time without adding a
data point.  `sar_tiny` is the first rung.

- `Ix`/`Iy` is the real dial: all three tile grids grow as `(Ix/32)²` and five
  of the blocks grow as `Ix²`.
- `P1`/`S1` change no object count — only the bytes of `X` and the inner-loop
  length of a backprojection tile.  They are how the reference datasets make
  the *work* per tile large without widening the DAG.
- `Nc` moves only the correlation stage, as `⌊√Nc⌋²`; `RAG_NEW_BLK_SIZE` (32)
  trades width against grain at fixed work — the one knob that would rebalance
  a rung against a machine, and it is compile-time.

Against the reference machine (15 workers + 1 progress thread per node,
1/2/4/8 nodes, strong scaling): pick the rung whose narrowest wide stage
comfortably exceeds `nodes × 15`.  `sar_tiny` is 4 wide everywhere,
`sar_small` 16, `sar_medium` 256/64/256/256 (its 64-wide correlation stage
starves past about two nodes), `sar_large` 1024/3600/1024/1024 — ≥ 8 tiles per
worker even at 8 nodes × 15 = 120 workers, which is why that rung carries an
explicit `timeout: 120`.  Use `sar_tiny` to check that a configuration runs and prints `SAR detects: 12`, never to measure — 42 EDTs and a 0.4 MiB live set finish before a multinode run has finished starting.

Width is not throughput here, though: every tile of a stage acquires the same
whole-image block RW, exclusive per node, so write concurrency across nodes is
1 however wide the stage — extra nodes buy 32.5 KiB-sized transfers, not
parallelism, and the strong-scaling curve should be expected to bend or invert.
Memory never constrains the choice on this machine (~0.4 MiB live plus 264 KiB
of read-only dataset per rank).
