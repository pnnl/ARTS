# sar_large

*Synthetic-aperture radar change detection — tiled backprojection, affine
registration, coherent change detection and CFAR, every stage driven through
one whole-image datablock.*
Source: `third_party/ocr-apps/apps/sar/ocr/src/` (11 files, ~4.6k lines).
All five SAR rows share these sources; `sar_large` is the build that compiles in
`ocr/large/Parameters.h` and links the `large` dataset.

## Overview

Forms two SAR images from a pulse stream and reports how many pixels changed
between them.  Per image: `ReadData` fills the pulse-return block `X` (plus
platform positions `Pt`, timestamps `Tp`), `FormImage` copies the previous
image into `refImage` and zeroes `curImage`, and `BackProj` fans out 1024
tiles that each accumulate all 4200 pulses into their `32×32` pixels.  After
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
`expect: '6523'` is a hard equality.  Arithmetic is real — 4.40 G complex
MACs of backprojection per image — but structurally this is a handful of
whole-image blocks read by thousands of tile tasks: it stresses broadcast of
large read-only blocks and exclusive re-acquisition of one image block per
tile.

## Parameters

`sar_large` takes **no command-line arguments**; the catalog's `args: []` is
literal.  `mainEdt`'s argv-parsing block sits inside
`#ifndef RAG_IMPLICIT_INPUTS` (`main.c:156-208`) and this target is built
*with* it: the dataset is linked in as `.rodata`, the detects path is the
compiled `argv_4` default, and every tunable is compile-time.

| knob | meaning | value here | CLI reachability |
|------|---------|------------|------------------|
| `argv[1..4]` | data / platform / pulse-time / detects paths | — | ⚠ parsed only in the runtime-input build (`sar_pss`); silently ignored here |
| `RAG_Ix`, `RAG_Iy` | image pixels | 1024 × 1024 | ✗ `ocr/large/Parameters.h` |
| `RAG_P1`, `RAG_S1` | pulses per image; complex samples per pulse | 4200, 4096 | ✗ same; must match the linked dataset |
| `RAG_Sx`, `RAG_Sy` | spotlight subimage | 1024 × 1024, so `TF = Ix/Sx = 1` | ✗ same; `TF > 1` aborts ("not yet supported") |
| `RAG_Nc` | affine control-point budget | 3629 → `N = ⌊√Nc⌋ = 60` per axis | ✗ same |
| `RAG_Sc`, `RAG_Rc`, `RAG_Tc` | correlation window / search radius / accept threshold | 15, 16, 0.7 | ✗ same |
| `RAG_Ncor`; `RAG_Ncfar`, `RAG_Nguard`, `RAG_Tcfar` | CCD window; CFAR window / guard cells / percentile | 5; 25, 17, 90 | ✗ same |
| `RAG_NumberImages` | images to process | 2 — **only 2 works**; any other value is a run-time error in `main_body_edt` | ✗ same |
| `RAG_NEW_BLK_SIZE` | tile edge for every parallel family | 32 | ✗ `benchmarks/apps/CMakeLists.txt` |
| `DEBUG_SSCP` | dump image/correlation planes | off | ✗ deliberately left out of every variant (hundreds of MB per run) |
| `RAG_DIG_SPOT_ON`, `RAG_QSORT_ON`, `RAG_AFFINE_ON` | spotlighting / detect sort / registration | 0, 0, 1 | ✗ CMake |

Changing the problem size means building a different size target, not passing
an argument — which is why the sizes are separate catalog rows.

## Structure

With `Ix = Iy = 1024`, tile edge `B = 32`, `N = ⌊√Nc⌋ = 60`, and the three
tiled window grids `T_bp = ⌈Ix/B⌉·⌈Iy/B⌉ = 1024`,
`T_ccd = ⌈(Iy−Ncor+1)/B⌉² = 1024`,
`T_cfar = ⌈(Iy−Ncor−Ncfar+2)/B⌉² = 1024`:

| object | count | here | size |
|--------|-------|------|------|
| stage heads | 18 (`post_main`, `main_body`, `ReadData`×2, `FormImage`×2, `post_FormImage`×2, `BackProj`×2, `Affine`, `post_Affine`, `post_affine_async_1/2`, `CCD`, `CFAR`, `post_CFAR`, plus `mainEdt` itself — the OCR shim creates it as an EDT before its body runs (`arts_ocr.c:2194`), so it is not one of `mainEdt`'s own explicit `ocrEdtCreate` calls and the stage-head count above missed it) | 18 | — |
| tile EDTs | `2·T_bp` backprojection + `N²` correlation + `T_bp` resample + `T_ccd` CCD + `T_cfar` CFAR | 2048 + 3600 + 1024 + 1024 + 1024 | — |
| global DBs | 14, all in `mainEdt` | 14 | `X` 131.3 MiB, `curImage`/`refImage` 8.01 MiB each, `corr_map` 11.9 MiB, `Y` 11.4 MiB, `Pt`/`Tp`/axis vectors, `file_args` 4 KiB, four parameter blocks (12–88 B) |
| `Affine` DBs | `5 + N²` | 3605 | `Fx`/`Fy` (`Nc·4`), `A` (`Nc·32`), `output` 8.01 MiB, 56 B per control point |
| per-task scratch | `2·N²` correlation windows (1.8 KiB) + `6·T_bp` backprojection (8.4 KiB tile copy, two 32-vectors) + `T_cfar` clutter windows (2.7 KiB) + 5 fixed | 14373 | — |
| EDT templates | 21 per rank | 21 | into a fixed `templateList[25]`, bounds-checked on every claim |

Totals: `18 + 3·T_bp + N² + T_ccd + T_cfar` = **8738 EDTs**,
`24 + 3·N² + 6·T_bp + T_cfar` = **17992 DBs**, and **20 events regardless of
size** — the app never calls `ocrEventCreate`, and the 10 `ocrEdtCreate` calls
that pass a non-NULL `outputEvent` are exactly the 10 that carry
`EDT_PROP_FINISH` (`main_body`, `Affine`, `post_affine_async_1/2`,
`post_Affine`, `CCD`, `CFAR`, `post_CFAR`, `BackProj`×2), giving 10 output +
10 finish events; every other create passes `NULL`.

Live set ≈ 208 MiB, dominated by `X` and the three image-shaped blocks — and it
is also the *total* allocated set, because `main.c`, `back_proj.c` and
`registration.c` each `#define` `bsm_free`/`dram_free`/`spad_free` to nothing;
only `cfar.c` destroys anything (a clutter window per CFAR tile, plus `Nd`).
The binary additionally carries 262 MiB of `.rodata` — the linked dataset as
`image_0/1`, `platform_0/1`, `pulse_0/1`.

Counter cross-check: not independently measured at this size. The family's
shared closed forms — `EDT = 18 + 3·T_bp + N² + T_ccd + T_cfar` (the `+18`
folding in `mainEdt` itself, created by the OCR shim's bootstrap before its
body runs, `arts_ocr.c:2194`), `DB = 24 + 3·N² + 6·T_bp + T_cfar`, and the
fixed 20 events — were verified exactly on the `sar_tiny`/`sar_small` pair (1
node, counters ONCE: measured 43/115 EDTs, 65/185 DBs, 20/20 events, each the
formula's 42/114, 64/184, 20/20 plus the runtime's constant +1 EDT/+1 DB/+0
EVT baseline). This row's Structure figures follow the same forms at its own
`T_bp = 1024`, `N² = 3600`, `T_ccd = 1024`, `T_cfar = 1024`.

## Wiring

There is not one explicit event in the program: stages chain through **finish
events**, each head a finish EDT whose fan-out lives in its scope and whose
event lands on the next head's last dependence slot (`main.c:1180-1185`).
`post_FormImage` closes an imaging round by wiring the refilled `X`/`Pt`/`Tp`
into the next `ReadData` (first round) or `curImage` into `Affine` (second).

| datablock | RW writers | RO readers | max concurrent readers |
|-----------|-----------|------------|------------------------|
| `X`, `Pt`, `Tp` | `ReadData` (1) | `backproject_async` | 1024 |
| `curImage` | `backproject_async` (1024), `post_Affine` (1) | `affine_async_1`, `affine_async_2`, `ccd_async` | 3600 |
| `refImage` | `FormImage` (1) | `affine_async_1`, `ccd_async` | 3600 |
| `output` | `affine_async_2` (1024) | `post_Affine` | 1 |
| `affine_params`, `Fx`, `Fy`, `A` | `affine_async_1` (3600) | `post_affine_async_1/2` | 1 |
| `corr_map` | `ccd_async` (1024) | `cfar_async` | 1024 |
| `Y`, `Nd` | `cfar_async` (1024) | `post_CFAR` | 1 |
| `image_params` | `ReadData` (twice) | *every* task in the program | all of them |

Two facts dominate.  **Every parallel family writes one whole-image block**,
and OCR `RW` is per-node exclusive — 1024 backprojection tiles all take
`curImage` RW, 1024 CCD tiles `corr_map`, 1024 CFAR tiles `Y` and `Nd`,
3600 correlation tasks `affine_params`/`Fx`/`Fy`/`A`.  Tiles are round-robin
placed, so each of those blocks is written from every node's tasks and a
family can never write on two nodes at once.  `curImage` is the contention
point: 2048 exclusive acquisitions of an 8.01 MiB block in backprojection alone,
over disjoint pixel ranges — pure protocol cost, not an algorithmic
dependence.  And the fan-out is genuinely broadcast-shaped: `X` (131.3 MiB) read
concurrently by all 1024 tiles of a round, `curImage`/`refImage` by up to
3600 tasks, and the 88-byte `image_params` by essentially every task —
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
2. `refReadData` → `refFormImage` — blit 131.3 MiB of pulse data out of `.rodata`,
   zero `curImage` (8.01 MiB).  Both serial.
3. `BackProj` → **1024 `backproject_async`** — the heaviest stage; each tile
   sweeps all 4200 pulses over `32×32` pixels, ≈ 4.3 M complex MACs.
4. `post_FormImage` → `ReadData` → `FormImage` (which now really does the
   `curImage → refImage` copy, 8.01 MiB) → `BackProj` → **1024 tiles** again.
5. `Affine` → **3600 `affine_async_1`**, one per control point,
   `(2·Rc+1)²·Sc²` ≈ 245 k operations each.
6. `post_affine_async_1` — serial `A'A`/`A'F` accumulation and two 6×6
   Gaussian eliminations → **1024 `affine_async_2`** resampling tiles.
7. `post_affine_async_2` → `post_Affine` — serial full-image copy
   `output → curImage` (8.01 MiB).
8. `CCD` → **1024 `ccd_async`**; `CFAR` → **1024 `cfar_async`**;
   `post_CFAR` writes `Nd` lines; `post_main` shuts down.

Max parallel width is `max(T_bp, N², T_ccd, T_cfar)` = **3600**, but what
bounds how much machine a rung keeps busy is the *narrowest* wide stage,
**1024**.  Serial bottlenecks in cost order: the two `ReadData` blits, the
`FormImage` copy-and-zero pair, `post_Affine`'s full-image copy, `post_CFAR`'s
line-by-line write.  Stage boundaries are hard barriers, so no two tile
families overlap and every serial head is fully exposed.

## Placement (base)

Every `ocrEdtCreate` passes `NULL_HINT`, and every DB is created through
`bsm/dram/spad_malloc`, which pass `NULL_HINT` too (`rag_ocr.c:22-24`).  The
five tile-EDT creates route their hint through `ragTileEdtHint()`, guarded by
`OCR_APP_OPTIMIZED_PLACEMENT` and returning `NULL_HINT` base (the
`sar_large_hinted` family compiles it in — the catalog's `hinted: true`).  So:
**EDTs** → the shim passes `ARTS_HINT_ANY_RANK` → runtime round-robin, and all
8720 tile tasks plus every stage head land on arbitrary ranks; **DBs** →
home = creating rank, and since `mainEdt` runs on rank 0 all 14 global blocks
(`X`, both images, `corr_map`, `Y`, the parameter blocks) are homed there,
while `Affine_edt` and the two `BackProj_edt`s are round-robin so `output`,
`A`, `Fx`, `Fy` and the per-tile scratch home on scattered ranks.

Consequence: tiles spread uniformly over all `n` ranks, so every large
read-only block must materialise on every rank that runs a tile, and the
whole-image block a family writes changes owner on nearly every one of its
per-tile RW acquisitions.  The algorithm has ideal tile locality (disjoint
output tiles over shared read-only pulse data) and the base program
expresses none of it — not merely because hints are absent, but because a
stage's entire output is one datablock, so no placement could let two ranks
write it concurrently.

## Placement (hinted)

Every parallel family in this pipeline tiles ONE whole-image block acquired
`DB_MODE_RW`.  Write permission is exclusive at rank granularity, so tiles
placed on different ranks cannot overlap in time anyway — they can only hand
the whole image around, one rank at a time, paying a full-image transfer per
hand-off.  Scattering buys no parallelism and costs the image each turn.

The layer (`ragTileEdtHint` in `rag_ocr.h`) therefore pins each tile family to
the PD that already holds the blocks its parent acquired (`ocrAffinityGetCurrent`),
keeping the pipeline's single mutable image resident.  `pdCount <= 1` returns
`NULL_HINT`.  This is containment of a structurally broadcast-bound program,
not a scaling fix — the multinode story for SAR is the size ladder, not spread.

## Sizing

Nothing about a SAR run is chosen at launch: the four compiled-in rows are one
program at four scales and the ladder *is* the target list
(`sar_tiny → sar_small → sar_medium → sar_large`, continuing past `large` only
through `sar_pss`, which reads its parameters from a file).  A campaign should
pick **one** rung; running several multiplies machine time without adding a
data point.  `sar_large` is the top rung of the compiled-in ladder.

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
explicit `timeout: 120`.  `sar_large` is the rung the 1→8 node sweep is meant to run: the only compiled-in size whose narrowest wide stage still gives every worker several tiles at 120 workers, with the widest single stage (3600 correlation tasks) and the heaviest single-block RW contention (2048 acquisitions of an 8 MiB image) on the roster.

Width is not throughput here, though: every tile of a stage acquires the same
whole-image block RW, exclusive per node, so write concurrency across nodes is
1 however wide the stage — extra nodes buy 8.01 MiB-sized transfers, not
parallelism, and the strong-scaling curve should be expected to bend or invert.
Memory never constrains the choice on this machine (~208 MiB live plus 262 MiB
of read-only dataset per rank).
