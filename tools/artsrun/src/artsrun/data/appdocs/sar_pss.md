# sar_pss

*The SAR pipeline with its inputs on disk instead of in the binary — one
executable and a ten-rung parameter ladder, the roster's largest single
datablock.*
Source: `third_party/ocr-apps/apps/sar/ocr/src/` (11 files, ~4.6k lines),
built from `ocr/problem_size_scaling/`.  Binary: `sar_problem_size_scaling`.

## Overview

`sar_pss` runs **exactly the same program** as `sar_tiny/small/medium/large`
— identical sources, identical stage graph, identical wiring — with three
build-level differences that make it a different benchmark:

1. **Inputs at runtime, not at link time.** It is the one SAR target built
   *without* `RAG_IMPLICIT_INPUTS`, so `mainEdt`'s argv block (`main.c:156-208`)
   is live: the pulse data, platform positions and pulse timestamps are read
   from files named on the command line, and the radar/image parameters come
   from a text file rather than from a compiled-in `Parameters.h`.  The
   problem size is data, so one binary covers the whole ladder.
2. **A ten-rung ladder ships with it.**  `Parameter0.txt … Parameter9.txt`
   sit beside the source with `Ix = Iy = 400, 800, … 4000` and everything else
   fixed (`P1 = 4200`, `S1 = 4000`, `Nc = 3629`).  The catalog names
   `Parameter2.txt` (1200²).
3. **A coarser tile: `RAG_NEW_BLK_SIZE = 50`, not 32** — this variant's own
   upstream `Makefile.x86` blocking factor, preserved by CMake.  Every
   parallel family tiles at `50×50`, so at equal image size it has ~2.4×
   fewer, ~2.4× fatter tiles than the compiled-in sizes.  Its CFAR percentile
   is also looser (`Tcfar = 75` vs 90), so it reports more detects per scene.

The pipeline itself: two SAR images are formed from a pulse stream and
compared.  Per image, `ReadData` fills the pulse-return block `X` (plus `Pt`,
`Tp`), `FormImage` copies the previous image into `refImage` and zeroes
`curImage`, and `BackProj` fans out 576 tiled tasks that each accumulate all
4200 pulses into their `50×50` pixels.  After the second image, `Affine`
registers current against reference (2-D correlation at 3600 control points →
6-parameter least-squares warp → tiled resampling), `CCD` builds a normalized
correlation map, `CFAR` declares a detection wherever the cell under test is
less correlated than its local clutter ring, and `post_CFAR` prints
`SAR detects: <Nd>` — the catalog marker and scalar — and writes the detect
list to the path given as `argv[4]`.  Provenance: the Georgia Tech Research
Institute Streaming Sensor Challenge Problem reference, ported to OCR through
the `RAG_*` macro layer.  The catalog pins **no `expect`** for this row, and
correctly so: the scalar is a function of the parameter file, not of the
binary.  It stresses broadcast of large read-only blocks and exclusive
re-acquisition of one whole-image block per tile, on top of 6.05 G complex
MACs of backprojection per image.

## Parameters

argv is **all-or-nothing**: the four path overrides are taken only when
`ocrGetArgc() >= 5` (program name + 4) and the parameter-file override only at
`>= 6`.  One to three arguments is a usage error that names the four paths and
exits; with none, the compiled `argv.h` defaults apply —
`../../datasets/huge/Data.bin` and friends, relative to the working directory
— and the validation `fopen` then fails with `Error opening ...`.

| arg | meaning | catalog value | CLI reachability |
|-----|---------|---------------|------------------|
| `argv[1]` | pulse-return data (`2·P1·S1` complex pairs) | `datasets/sar-huge/Data.bin` (268.8 MB) | ✓ carried to every node in the `file_args` block, reopened per task — multinode-safe |
| `argv[2]` | platform positions (`2·P1·3` floats) | `datasets/sar-huge/PlatformPosition.bin` | ✓ same |
| `argv[3]` | pulse timestamps (`2·P1` floats) | `datasets/sar-huge/PulseTransmissionTime.bin` | ✓ same |
| `argv[4]` | detects output path | `{repo}/scratch/sar_detects.txt` | ✓ passed by value in `post_CFAR`'s paramv |
| `argv[5]` | radar/image parameter file | `ocr/problem_size_scaling/Parameter2.txt` | ✓ read by `ReadParams` on rank 0 only, into the `image_params` DB — multinode-safe |
| `Ix`, `Iy` (param file) | image pixels | 1200 × 1200 | ✓ via `argv[5]`; **`Ix != Iy` is unsupported** (see below) |
| `P1`, `S1` (param file) | pulses per image; samples per pulse | 4200, 4000 | ✓ but must match the `.bin` fixtures byte for byte |
| `Sx`, `Sy` (param file) | spotlight subimage | 1200 (`TF = Ix/Sx = 1`) | ✓; `TF > 1` aborts ("digital spotlighting not yet supported") |
| `Nc`, `Sc`, `Rc`, `Tc` | control-point budget / window / radius / threshold | 3629 → `N = ⌊√Nc⌋ = 60`; 15, 16, 0.7 | ✓ |
| `Ncor`; `Ncfar`, `Nguard`, `Tcfar` | CCD window; CFAR window / guard / percentile | 5; 25, 17, 75 | ✓ |
| `NumberImages` | images to process | 2 — **only 2 works**; any other value is a run-time error in `main_body_edt` | ✓ but effectively fixed |
| `RAG_NEW_BLK_SIZE` | tile edge for every parallel family | 50 | ✗ `benchmarks/apps/CMakeLists.txt` |
| `DEBUG_SSCP` | dump image/correlation planes | off | ✗ deliberately left out of every variant |

`Ix != Iy` is accepted by the parser and then mis-indexed: `backproject_async`
walks `m` over the `Ix` extent against a row table built with `Iy` rows, and
`affine_async_2` clamps its `Y` loop against `Xend`.  All ten shipped
parameter files are square, so this never bites — but a hand-written one must
be too.

## Structure

With `Ix = Iy = 1200`, tile edge `B = 50`, `N = ⌊√Nc⌋ = 60`, and the three
tiled window grids `T_bp = ⌈Ix/B⌉² = 576`, `T_ccd = ⌈(Iy−Ncor+1)/B⌉² = 576`,
`T_cfar = ⌈(Iy−Ncor−Ncfar+2)/B⌉² = 576`:

| object | count | here | size |
|--------|-------|------|------|
| stage heads | 18 (`post_main`, `main_body`, `ReadData`×2, `FormImage`×2, `post_FormImage`×2, `BackProj`×2, `Affine`, `post_Affine`, `post_affine_async_1/2`, `CCD`, `CFAR`, `post_CFAR`, plus `mainEdt` itself — the OCR shim creates it as an EDT before its body runs (`arts_ocr.c:2194`), so it is not one of `mainEdt`'s own explicit `ocrEdtCreate` calls and the stage-head count above missed it) | 18 | — |
| tile EDTs | `2·T_bp` backprojection + `N²` correlation + `T_bp` resample + `T_ccd` CCD + `T_cfar` CFAR | 1152 + 3600 + 576 + 576 + 576 | — |
| global DBs | 14, all in `mainEdt` | 14 | `X` 128.2 MiB, `curImage`/`refImage` 11.0 MiB each, `corr_map` 16.4 MiB, `Y` 15.7 MiB, `Pt`/`Tp`/axis vectors, `file_args` 4 KiB, four parameter blocks (12–88 B) |
| `Affine` DBs | `5 + N²` | 3605 | `Fx`/`Fy` (`Nc·4`), `A` (`Nc·32`), `output` 11.0 MiB, 56 B per control point |
| per-task scratch | `2·N²` correlation windows (1.8 KiB) + `6·T_bp` backprojection (20 KiB tile copy, two 50-vectors) + `T_cfar` clutter windows (2.7 KiB) + 5 fixed | 11 237 | — |
| EDT templates | 21 per rank | 21 | into a fixed `templateList[25]`, bounds-checked on every claim |

Totals: `18 + 3·T_bp + N² + T_ccd + T_cfar` = **6498 EDTs**,
`24 + 3·N² + 6·T_bp + T_cfar` = **14 856 DBs**, and **20 events regardless of
size** — the app never calls `ocrEventCreate`, and the 10 `ocrEdtCreate` calls
that pass a non-NULL `outputEvent` are exactly the 10 that carry
`EDT_PROP_FINISH` (`main_body`, `Affine`, `post_affine_async_1/2`,
`post_Affine`, `CCD`, `CFAR`, `post_CFAR`, `BackProj`×2), giving 10 output +
10 finish events; every other create passes `NULL`.

Because every grid tiles at `B = 50` and every shipped `Ix` is a multiple of
400, all three grids collapse to `g² ` with `g = Ix/50`, giving the closed
forms `EDT = 3618 + 5·g²` and `DB = 10 824 + 7·g²` across the whole ladder.

Live set ≈ 229 MiB — and it is also the *total* allocated set, because
`main.c`, `back_proj.c` and `registration.c` each `#define`
`bsm_free`/`dram_free`/`spad_free` to nothing; only `cfar.c` destroys anything
(a clutter window per CFAR tile, plus `Nd`).  Unlike the compiled-in sizes the
binary carries no dataset: the 268.8 MB of input is read from disk instead,
twice (once per image), by whichever node runs each `ReadData`.

Counter cross-check: not independently measured at this size. The family's
shared closed forms — `EDT = 18 + 3·T_bp + N² + T_ccd + T_cfar` (the `+18`
folding in `mainEdt` itself, created by the OCR shim's bootstrap before its
body runs, `arts_ocr.c:2194`), `DB = 24 + 3·N² + 6·T_bp + T_cfar`, and the
fixed 20 events — were verified exactly on the `sar_tiny`/`sar_small` pair (1
node, counters ONCE: measured 43/115 EDTs, 65/185 DBs, 20/20 events, each the
formula's 42/114, 64/184, 20/20 plus the runtime's constant +1 EDT/+1 DB/+0
EVT baseline). This row's Structure figures (and the collapsed
`EDT = 3618 + 5·g²`, `DB = 10 824 + 7·g²` forms above) follow the same closed
forms at its own `T_bp = 576`, `N² = 3600`, `T_ccd = 576`, `T_cfar = 576`.

## Wiring

Identical to the compiled-in sizes.  There is not one explicit event in the
program: stages chain through **finish events**, each head a finish EDT whose
fan-out lives in its scope and whose event lands on the next head's last
dependence slot (`main.c:1180-1185`).  `post_FormImage` closes an imaging
round by wiring the refilled `X`/`Pt`/`Tp` into the next `ReadData` (first
round) or `curImage` into `Affine` (second).

| datablock | RW writers | RO readers | max concurrent readers |
|-----------|-----------|------------|------------------------|
| `X`, `Pt`, `Tp` | `ReadData` (1) | `backproject_async` | 576 |
| `curImage` | `backproject_async` (576), `post_Affine` (1) | `affine_async_1`, `affine_async_2`, `ccd_async` | 3600 |
| `refImage` | `FormImage` (1) | `affine_async_1`, `ccd_async` | 3600 |
| `output` | `affine_async_2` (576) | `post_Affine` | 1 |
| `affine_params`, `Fx`, `Fy`, `A` | `affine_async_1` (3600) | `post_affine_async_1/2` | 1 |
| `corr_map` | `ccd_async` (576) | `cfar_async` | 576 |
| `Y`, `Nd` | `cfar_async` (576) | `post_CFAR` | 1 |
| `image_params` | `ReadData` (twice) | *every* task in the program | all of them |

Two facts dominate.  **Every parallel family writes one whole-image block**,
and OCR `RW` is per-node exclusive — 576 backprojection tiles all take
`curImage` RW, 576 CCD tiles `corr_map`, 576 CFAR tiles `Y` and `Nd`, 3600
correlation tasks `affine_params`/`Fx`/`Fy`/`A`.  Tiles are round-robin
placed, so each of those blocks is written from every node's tasks and a
family can never write on two nodes at once.  `curImage` is the contention
point: 1152 exclusive acquisitions of an 11.0 MiB block in backprojection
alone, over disjoint pixel ranges — pure protocol cost, not an algorithmic
dependence.  And the fan-out is genuinely broadcast-shaped: `X` (128.2 MiB)
read concurrently by all 576 tiles of a round, `curImage`/`refImage` by up to
3600 tasks, and the 88-byte `image_params` by essentially every task.  Tiles
sharing a writable block coordinate with plain atomics on the shared copy
(`__sync_fetch_and_add` on `affine_params->Nc`, on `Nd`), varying the *order*
of rows and detects but not the counts or the fitted warp.

One wiring detail is specific to this build: a `FILE*` is process-local, so
the input paths (not handles) travel in the `file_args` block, and each
`ReadData_edt` reopens all three files on whatever node it lands on and
`fseek`s to its image's slice.  The fixtures must therefore be visible at the
same path on every node — trivially true for `launcher=local`, a real
requirement on a cluster.

## Flow

Strictly serial stages, each a one-EDT head fanning out to a tile family and
joining on its own finish event:

1. `mainEdt` (rank 0, serial) — reads the parameter file, validates the four
   paths by opening them, 14 `ocrDbCreate`s, axis vectors.
2. `refReadData` → `refFormImage` — read 134 MB of pulse data off disk, zero
   `curImage` (11.0 MiB).  Both serial.
3. `BackProj` → **576 `backproject_async`** — the heaviest stage; each tile
   sweeps all 4200 pulses over `50×50` pixels, ≈ 10.5 M complex MACs.
4. `post_FormImage` → `ReadData` (the second 134 MB read) → `FormImage` (which
   now really does the `curImage → refImage` copy) → `BackProj` → **576 tiles**.
5. `Affine` → **3600 `affine_async_1`**, one per control point,
   `(2·Rc+1)²·Sc²` ≈ 245 k operations each.
6. `post_affine_async_1` — serial `A'A`/`A'F` accumulation and two 6×6
   Gaussian eliminations → **576 `affine_async_2`** resampling tiles.
7. `post_affine_async_2` → `post_Affine` — serial full-image copy
   `output → curImage`.
8. `CCD` → **576 `ccd_async`**; `CFAR` → **576 `cfar_async`**; `post_CFAR`
   writes `Nd` lines; `post_main` shuts down.

Max parallel width is `max(T_bp, N², T_ccd, T_cfar)` = **3600**, but what
bounds how much machine a rung keeps busy is the *narrowest* wide stage,
**576** at `Parameter2` — and note that the correlation stage is 3600 wide at
*every* rung, because `Nc` is 3629 in all ten parameter files.  Serial
bottlenecks in cost order: the two 134 MB `ReadData` file reads, the
`FormImage` copy-and-zero pair, `post_Affine`'s full-image copy, and
`post_CFAR`'s line-by-line write.  Stage boundaries are hard barriers, so no
two tile families overlap and every serial head is fully exposed.

## Placement (as-born)

Every `ocrEdtCreate` passes `NULL_HINT`, and every DB is created through
`bsm/dram/spad_malloc`, which pass `NULL_HINT` too (`rag_ocr.c:22-24`).  The
five tile-EDT creates route their hint through `ragTileEdtHint()`, guarded by
`OCR_APP_OPTIMIZED_PLACEMENT` and returning `NULL_HINT` as-born (the
`sar_pss_opt` family compiles it in — the catalog's `optimized: true`).  So:
**EDTs** → the shim passes `ARTS_HINT_ANY_RANK` → runtime round-robin, and all
5880 tile tasks plus every stage head land on arbitrary ranks; **DBs** →
home = creating rank, and since `mainEdt` runs on rank 0 all 14 global blocks
(`X`, both images, `corr_map`, `Y`, the parameter blocks) are homed there,
while `Affine_edt` and the two `BackProj_edt`s are round-robin so `output`,
`A`, `Fx`, `Fy` and the per-tile scratch home on scattered ranks.

Consequence: tiles spread uniformly over all `n` ranks, so every large
read-only block must materialise on every rank that runs a tile, and the
whole-image block a family writes changes owner on nearly every one of its
per-tile RW acquisitions.  This build adds one more cross-rank effect the
compiled-in sizes do not have: `ReadData_edt` is round-robin like everything
else, so the 134 MB file read happens on an arbitrary rank and the freshly
filled `X` then has to reach every other rank's backprojection tiles.  The
algorithm has ideal tile locality (disjoint output tiles over shared read-only
pulse data) and the as-born program expresses none of it — not merely because
hints are absent, but because a stage's entire output is one datablock, so no
placement could let two ranks write it concurrently.

## Sizing

`sar_pss` is the continuation of the SAR ladder past `sar_large`, and the only
rung that is chosen at launch rather than at build time — swap `argv[5]` and
the same binary covers `Ix = 400 … 4000`.  It is also the SAR row that is
enabled by default (the four compiled-in sizes carry `default_enabled: false`),
and the one that needs `fixtures`: the 268.8 MB `datasets/sar-huge/` triple.
A campaign should still pick **one** rung.

| parameter file | `Ix = Iy` | tiles per grid | EDTs | DBs | image block | live set | backprojection |
|---|---|---|---|---|---|---|---|
| `Parameter0.txt` | 400 | 64 | 3 938 | 11 272 | 1.2 MiB | ~150 MiB | 0.67 G MAC/img |
| `Parameter2.txt` (catalog) | 1200 | 576 | 6 498 | 14 856 | 11.0 MiB | ~229 MiB | 6.05 G MAC/img |
| `Parameter5.txt` | 2400 | 2 304 | 15 138 | 26 952 | 44.0 MiB | ~495 MiB | 24.2 G MAC/img |
| `Parameter9.txt` | 4000 | 6 400 | 35 618 | 55 624 | 122.1 MiB | ~1.1 GiB | 67.2 G MAC/img |

How the knobs move the shape: `Ix`/`Iy` is the real dial — all three tile grids
grow as `(Ix/50)²` and five of the blocks grow as `Ix²`.  `P1`/`S1` change no
object count, only the bytes of `X` and the inner-loop length of a
backprojection tile (and they must match the fixture, so in practice they are
fixed at 4200×4000).  `Nc` moves only the correlation stage, as `⌊√Nc⌋²`, and
every shipped file leaves it at 3629 — which means the 3600-task correlation
stage is a *constant* floor of work that does not shrink when you pick a
smaller image.  `RAG_NEW_BLK_SIZE` (50) trades width against grain at fixed
work and is compile-time.

Against the reference machine (15 workers + 1 progress thread per node,
1/2/4/8 nodes, strong scaling): pick the rung whose narrowest wide stage
comfortably exceeds `nodes × 15`.  `Parameter0` (64 tiles) fits one node and
starves beyond it; `Parameter2` (576) gives ≥ 4 tiles per worker at 8 nodes ×
15 = 120 workers and is the calibrated choice for that reason — it is the
smallest rung whose *narrow* stages still cover the full node sweep, at a live
set that fits comfortably per rank.  `Parameter3`–`Parameter5` are the rungs to
reach for if 8 nodes finish too fast; `Parameter9` is a ~1.1 GiB, 67 G-MAC
per-image run and should be treated as a capacity experiment, not a sweep point.

Width is not throughput here, though: every tile of a stage acquires the same
whole-image block RW, exclusive per node, so write concurrency across nodes is
1 however wide the stage — extra nodes buy image-sized transfers, not
parallelism, and the strong-scaling curve should be expected to bend or invert.
Memory never constrains the choice on this machine below `Parameter7` or so.
