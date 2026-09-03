# smithwaterman

*Tiled sequence-alignment DP over EDTs — a wavefront over a tile grid,
every tile a task, three neighbors its dependencies.*
Source: `third_party/ocr-apps/apps/smithwaterman/ocr/smithwaterman.c`
(~540 lines).

## Overview

Computes an alignment score between two nucleotide sequences by dynamic
programming, tiled into a `tile_width × tile_height` grid and executed as
a diagonal wavefront: each `smith_waterman_task` fills its tile's DP cells
from the tile to its west, north and northwest, using a transition/
transversion-aware scoring matrix (`MATCH=+2`, `TRANSITION_PENALTY=-2`,
`TRANSVERSION_PENALTY=-4`, `GAP_PENALTY=-1`). Despite the app's name, the
recurrence has no zero-floor and no local traceback — border cells
accumulate `GAP_PENALTY` linearly from the (0,0) corner, i.e. this is a
global (Needleman–Wunsch-style) alignment score, not a local
Smith–Waterman one. The single bottom-right-most tile prints the score and
calls `ocrShutdown()` inline, checked externally by the harness against a
score fixture file (catalog `expect`). Reads two sequence files and a
score file from disk (rank 0 only, native `fopen`/`fread`); the DAG shape
depends only on the two sequence *lengths* and the tile size — sequence
*content* only affects the numeric scores, not the task graph. The
program stresses wavefront scheduling and border-cell data movement, not
raw compute (each tile does `O(tile_width·tile_height)` cheap DP steps).

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `tileWidth` | DP columns per tile | required | ✓ `atoi` in `ioHandling` (non-`TG_ARCH` build) |
| `argv[2]` = `tileHeight` | DP rows per tile | required | ✓ same |
| `argv[3]` = `fileName1` | sequence 1 input file | required | ✓ passed straight to `read_file`; catalog names it via `fixtures` |
| `argv[4]` = `fileName2` | sequence 2 input file | required | ✓ same |
| `argv[5]` = `scoreFile` | expected-score fixture (ASCII integer) | required | ✓ same; parsed with `atoi` |

`n_tiles_width`/`n_tiles_height` are **not** direct CLI knobs — they are
derived (`ceil(seqLen / tileSize)`) from the tile size against whichever
fixture pair is named. The scoring constants (`GAP_PENALTY`,
`TRANSITION_PENALTY`, `TRANSVERSION_PENALTY`, `MATCH`) are `#define`s with
no argv path (✗). A `TG_ARCH` branch reinterprets `argv[3..5]` as raw
character counts instead of filenames; it is dead code in this build
(`TG_ARCH` is never defined for the ARTS/x86 target).

## Structure

Let `W = ceil(len1/tileWidth)`, `H = ceil(len2/tileHeight)`, where
`len1`/`len2` count only `A`/`C`/`G`/`T` characters after whitespace
stripping (not raw file bytes). Unlike nqueens/quicksort this shape is
fully closed-form — content never affects it, only the two lengths and
the tile size:

| object | count | size |
|--------|-------|------|
| `smith_waterman_task` | `W·H` | — |
| `mainEdt` | 1 | — |
| DBs | `5·W·H + 2·W + 3·H + 7` | per compute tile: 2 temp (destroyed same task, `4·(tileWidth+1)·(tileHeight+1)` B + `8·(tileHeight+1)` B) + 3 output (`4` B / `4·tileHeight` B / `4·tileWidth` B); border init: `2W+2H+1` (≤ `4·max(tileWidth,tileHeight)` B each); tile-matrix structure: `H+2`; shared params DB: 1, `8·(8 + ⌈len1/8⌉ + ⌈len2/8⌉)` bytes; **+3** for the three input-file buffer DBs `read_file` allocates (one `ocrDbCreate` each for `fileName1`, `fileName2`, `scoreFile`, called from `ioHandling`) |
| Events | `3·(W+1)·(H+1)` STICKY events (readiness signals for the border-inclusive `(H+1)×(W+1)` tile grid) |

Worked numbers for the calibrated `args=[100, 100, ...cal...]` (the
`string{1,2}-cal.txt` fixtures: `len1=140000, len2=140400` after stripping)
→ `W=1400, H=1404`: EDTs = 1,965,601; DBs = 9,835,019; Events = 5,905,215.
The shared params DB is ≈280 KB (`8·(8 + ⌈len1/8⌉ + ⌈len2/8⌉)` = 280,464 B)
and is read (RO) by all 1,965,600 compute tasks.

Counter cross-check: verified (1 node, `4 4` tiny fixtures (`W=H=2`) vs
`4 4` small fixtures (`W=H=3`)): NUM_EDT_CREATE 6 → 11, NUM_DB_CREATE
38 → 68, NUM_EVENT_CREATE 27 → 48 — exactly `W·H+1` / `5WH+2W+3H+7` /
`3(W+1)(H+1)` (app values 5/10, 37/67, 27/48) plus the runtime's constant
+1 EDT/+1 DB/+0 EVT baseline. The DB formula's original `+4` constant
undercounted by exactly the 3 file-buffer DBs above; corrected to `+7`.

An HPX port (`benchmarks/hpx/smithwaterman.cpp`) mirrors both tiers as
`smithwaterman_hpx` / `smithwaterman_hinted_hpx`: locality 0 reads the files
once and broadcasts them, then posts every tile in one serial loop (base:
round-robin, so tile `n = (i-1)*W + (j-1)` lands on locality `n mod L`, the
same place the OCR base's blind creates put it; hinted: the band
`((i-1)*L)/H`); a tile is the continuation on its west column, north row and
north-west corner, pushed to it by their producers; the bottom-right tile
delivers the score to locality 0. Structural references at the calibrated
arguments: `tasks = posts_from_locality0 = W*H = 1,965,600`,
`strips = H(W-1) + (H-1)W + (H-1)(W-1) = 5,891,193`, `bytes = 1,579,209,588`.

## Wiring

- Every dependence is `DB_MODE_CONST` → ARTS RO: slot 0 = west neighbor's
  `right_column_event`, slot 1 = north neighbor's `bottom_row_event`, slot
  2 = northwest neighbor's `bottom_right_event`, slot 3 = the shared
  params DB. There is no `DB_MODE_RW` anywhere in this app.
- Every border-crossing output DB (`br`/`rc`/`brow`) has exactly one
  consumer — no fan-out — **except** the shared params DB, read by every
  one of the `W·H` compute tasks: the one broadcast point in the app, but
  pure RO fan-out (never written), the natural case for RO-snapshot
  caching rather than lock contention.
- No separate reduction/finish EDT: the single bottom-right tile both
  computes its cell *and* performs verification/shutdown in the same EDT
  body once its own 3-way AND-join (which transitively needs the whole
  grid) fires.

## Flow

`mainEdt` (rank 0): reads both sequence files and the score file, builds
the `(H+1)×(W+1)` event grid (`3·(W+1)(H+1)` STICKY events plus the
`Tile_t` matrix DBs), seeds border row/column 0 with linear gap-penalty
values (`2W+2H+1` DBs), builds the shared params DB, then issues `W·H`
`ocrEdtCreate` calls in a plain nested loop — **all** of this, including
the create loop itself, is serial, single-rank work that must finish
before the first compute tile can run (over 1M creates at the calibrated
size). Execution then follows an anti-diagonal wavefront: tile `(i,j)`
is ready once its west/north/northwest neighbors have published their
border arrays. Parallel width at diagonal step `k = i+j`
(`2 ≤ k ≤ W+H`) is `|{i : max(1,k-W) ≤ i ≤ min(H,k-1)}|`, ramping
`1 → min(W,H) → 1` over `W+H-1` steps; average concurrency
`≈ W·H/(W+H-1)`, well under the peak. Completion is implicit in the last
tile's own dependency join — no separate barrier phase.

## Placement (base)

The source carries an `OCR_APP_OPTIMIZED_PLACEMENT` guard around one
helper, `swBandEdtHint` (built as `smithwaterman_hinted`, `HINTED_PLACEMENT`
in `benchmarks/apps/CMakeLists.txt`; catalog `hinted: true`) — see the next
section. Outside that guard every `ocrEdtCreate`/`ocrDbCreate` call passes
`NULL_HINT`, and base is the guard off. Effective policy:

- **EDTs**: NULL hint → round-robin (`ARTS_HINT_ANY_RANK`) —
  `smith_waterman_task` instances scatter across ranks with no relation to
  the wavefront's 2D adjacency.
- **DBs**: NULL hint → home = creating rank. Border/output DBs are homed
  wherever the *producing* tile's round-robin-placed EDT landed, so a
  consumer usually acquires each of its 3 inputs from a different, likely
  remote, rank.

Consequence: nearly every one of the wavefront's dependency edges is a
remote acquire of a small (≤400 B) block — the algorithm's real locality
(adjacent tiles) is never expressed by placement — layered under one
large, always-resident, RO-fan-out params DB (homed once at rank 0,
read-shared everywhere, never migrated).

## Placement (hinted)

As-born scatters the W x H wavefront round-robin, so a tile's three inputs
(West's right column, North's bottom row, NW's corner) almost always live on
three different remote ranks (see above).

The layer (`swBandEdtHint` in `smithwaterman.c`, the single create loop) is
the same contiguous row-band map as LCS_all: tile (i,j) pins to rank
`((i-1) * nranks) / n_tiles_height`.  A tile's West neighbour shares its row
and therefore its rank, North/NW share its band on all but the nranks-1
band-boundary rows, so the dominant row-to-row payload stays rank-local while
the anti-diagonal frontier still reaches every band once it is a band tall.
The border/output DBs keep `NULL_HINT` — creator home puts each output on its
producer's band rank, which is its consumer's rank for the in-band edges.
`nranks <= 1` returns `NULL_HINT` (verified: 1-node run PASSED, score 80).

## Sizing

The total DP work is the product of the two sequence lengths and does **not**
depend on the tiling.  So the tile size sets width and grain, while the run
length is set by the DATASET -- which makes the dataset this row's size knob.

| geometry, 178k pair | time |
|---|---|
| 1 node x 15 workers | 16.37 s |
| 2 nodes | 316.06 s |
| 4 nodes | 283.89 s |

A 19x degradation across the first node boundary, so the window is 10-30 s and
the calibrated pair is 140,000/140,400: 14.5 s, 14.7 s and 15.3 s on the three
coherence families, holding 3 GB.  Its expected score, 86360, comes from an
independent sequential reference of the same recurrence, validated by
reproducing the 515,000-pair's long-pinned 318128.

This row also anti-scales INSIDE a node, and the phase measurement says why:
7.91 s at 15 workers against 10.88 s at 112, on the same problem.  The run is
bounded below by a single-threaded creation loop, so workers added around it
only contend -- `max(creation, work/workers)` predicts every point of that
sweep, with creation at 9.82 s.

**Deliberate deviation from the width rule, and not for the reason recorded
here before.**  A wavefront needs `W^2` tasks to offer width `W`: the rule's 4x
slack (13,824) is 191 M tasks and even 1x (3,456) is 11.9 M.  This row builds
its whole graph in `mainEdt` -- 9.82 s of a 9.93 s run, against 0.001 s to read
the input -- so creation cost tracks the task count, and width cannot be bought
without leaving the window: tile 100 puts the anchor at 11.8 s with width
0.41x, tile 50 at 100.8 s with 0.81x.  The window binds for an anti-scaler, so
the tile stays 100 and the width is given up.  Both tiers share these
arguments and both land in the window: 11.8 s base, 11.0 s hinted, each
holding 3 GB.

An earlier note blamed 210 GB of tile memory.  That was wrong twice over: the
tiles ARE reclaimed -- the task destroys the three blocks it read -- and the
figure was arithmetic for a 515,000 pair this row no longer runs, while the
arguments beside it were a 140,000 one at 0.41x rather than the 1.49x claimed.
What did accumulate was the readiness events, three per tile and never
destroyed; they are reclaimed now by the same consumer that frees the blocks
they carried, which is 9.10 s and 2 GB before against 7.99 s and 1 GB after.

The placement layer is worth its tier here, and the counters say exactly what
it buys.  Over four nodes the EDT counts are already even without it -- 122,500
finished per rank either way, an imbalance of 1.00x, because a hintless create
round-robins -- so what the layer changes is not who works but where the data
is: remote acquires fall from **49.95% to 0.27%** (978,954 of 1,959,999 against
5,250) and the bytes crossing from 605 MB to 262 MB.  Banding by row makes each
tile's neighbours rank-local.  Banding by row makes each tile's row neighbour rank-local, so the ranks
stop waiting on remote acquires.
