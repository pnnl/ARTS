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

Worked numbers for the calibrated `args=[100, 100, ...large...]`
(`len1=100713, len2=101133` after stripping) → `W=1008, H=1012`: EDTs =
1,020,097; DBs = 5,105,539; Events = 3,066,351. The shared params DB is
≈198 KB and is read (RO) by all 1,020,096 compute tasks.

Counter cross-check: verified (1 node, `4 4` tiny fixtures (`W=H=2`) vs
`4 4` small fixtures (`W=H=3`)): NUM_EDT_CREATE 6 → 11, NUM_DB_CREATE
38 → 68, NUM_EVENT_CREATE 27 → 48 — exactly `W·H+1` / `5WH+2W+3H+7` /
`3(W+1)(H+1)` (app values 5/10, 37/67, 27/48) plus the runtime's constant
+1 EDT/+1 DB/+0 EVT baseline. The DB formula's original `+4` constant
undercounted by exactly the 3 file-buffer DBs above; corrected to `+7`.

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

## Placement (as-born)

There is no `OCR_APP_OPTIMIZED_PLACEMENT` guard anywhere in this source —
every `ocrEdtCreate`/`ocrDbCreate` call passes `NULL_HINT` directly, and
the catalog correctly carries no `optimized` flag for this app. Effective
policy:

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

## Sizing

`tileWidth`/`tileHeight` set per-task grain (compute grows with
`tileWidth·tileHeight`; message sizes with `tileWidth` or `tileHeight`
alone); `W`, `H` are *derived* from a chosen dataset's fixed sequence
lengths against that tile size — the datasets fixture set spans
`tiny`/`small`/`medium`/`medium-large`/`large`.

- To keep `N` nodes × `C` workers busy, aim for the wavefront's peak width
  `min(W,H)` at or above `N·C`: pick a longer dataset and/or shrink the
  tile size.
- Shrinking tiles grows EDT/DB/event counts roughly with `1/tileSize²`
  while per-tile compute shrinks with `tileSize²` — past a point this
  turns the app into a scheduling/coherence-churn probe rather than a
  compute benchmark.
- The calibrated `args=[100, 100, ...large...]` gives `W=1008, H=1012`
  (peak width 1008) — comfortably above 8 nodes × 15 workers = 120 workers
  even accounting for the wavefront's ramp-up/ramp-down edges.
