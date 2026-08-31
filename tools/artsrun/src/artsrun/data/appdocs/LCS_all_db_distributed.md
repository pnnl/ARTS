# LCS_all_db_distributed

*Same recursive quad-tree wavefront DP, but now the score matrix is
materialized as a genuine `L×L` grid of per-tile labeled DBs too — a real
tile-to-tile wavefront DAG, not a single shared DB coordinated by events.*
Source: `third_party/ocr-apps/apps/LCS/refactored/ocr/intel-jesmin-lcs_all_db_distributed/lcs_distributed.cpp`
(~1,085 lines, C++; author Jesmin Jahan Tithi, Intel 2016).

## Overview

The third point in this family's decomposition spectrum: where
`LCS_shared` keeps `S`, `T`, and the score matrix each in one DB, and
`LCS_distributed_ST` distributes only `S`/`T`, this variant distributes
**everything**, including the score matrix itself — hence "all DB
distributed". The score is no longer the `O(N)` antidiagonal-collapsed
array the other two variants use; it is the *full* `(N+1)×(N+1)` DP
table, tiled `base×base` and materialized as `(L+1)²` separate labeled
DBs (`L = N/base`), each with exactly one writer and up to three
read-only consumers (its east/south/southeast neighbors) — a real
tile-grid wavefront DAG at the DB level, not just at the EDT level. The
recursion shape is unchanged (`recLCSEdt` quad-tree: `x11` unblocked,
`x12`/`x21` gated on `x11`, `x22` on both; `seqLCSEdt` fills the base
case), but reaching a leaf now wires real dependencies onto four
neighboring score tiles, not one shared DB. Initialization is also
parallelized here (a separate `InitEdt` finish-scope, absent from the
other two variants) rather than done inline in `mainEdt`. Unlike the
other two variants, `CHECK_RESULTS`/`PRINT` are commented out in this
file, so there is **no internal native-recomputation self-check** — the
binary just prints `LCS length: N`; correctness for this row depends
entirely on the external harness comparing that scalar against the
catalog's pinned `expect`. The catalog marks this row `hinted: true`
(a companion `OCR_APP_OPTIMIZED_PLACEMENT`-guarded build exists; see
Placement) — DAG shape depends only on `N`/`base`, never on string
content.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `N` | string length | 1024 | ✓ `atol` in `mainEdt`, reaches the recursion via `LCS_task_params.N` (`paramv`) — multinode-safe |
| `argv[2]` = `base` | recursion base case / tile side length (both the recursion's and the score grid's) | 256 | ✓ same, via `LCS_task_params.base` |

`GAP_PENALTY` is compile-time only. `LCS_ROW_BANDS_PER_RANK` and the
`OCR_APP_OPTIMIZED_PLACEMENT`-guarded hint helpers (`scoreIdx`,
`blockRowHint`) belong to the hinted-placement build only — see
Placement.

## Structure

Correctness of the tile-index arithmetic (`block_i_end = (xi-1)/base+1`,
etc.) requires leaf tile width to equal `base` exactly, which holds only
when `N/base` is a power of two (see the notes file's Findings for the
general risk; this app's own catalog `args`/`expect_args` both use safe
ratios). Let `L = N/base = 2^d`, where `d` is the recursion depth (shifts
of `N` until `≤ base`); `grid_block_size = L+1`.

| object | count | size |
|--------|-------|------|
| `recLCSEdt` (incl. root) | `(4^(d+1)-1)/3` | — |
| `seqLCSEdt` (compute tile) | `L²` (`=4^d`) — one per interior score tile | — |
| `InitEdt` | 1 (finish scope around all data setup) | — |
| `randInitEdt` | `2L` — fills `S`/`T` tile 0 and each of tiles `1..L-1` | — |
| `scoreInitEdt` | `2L+1` — fills the boundary row-0/col-0 tiles and the single corner tile; the `L²` interior score tiles are created **empty** (written only by `seqLCSEdt`) | — |
| `mainEdt` / `shutDownEdt` | 1 each | — |
| DBs | `(L+2)²` — 3 pointer-array DBs (`S`,`T`,`score`) + `2L` labeled `S`/`T` tiles + `(L+1)²` labeled score tiles (1 corner + `2L` boundary + `L²` interior) | pointer arrays: `8L`/`8L`/`8·(L+1)²` B; `S`/`T` tiles: `4·(base+1)` B (tile 0), `4·base` B (others); score tiles: `4` B (corner), `4·base` B (boundary), `4·base²` B (interior — the dominant term) |
| Events | `6L² + 2` (`= 6·4^d + 2`) — every `recLCSEdt` create (root + `x11`/`x12`/`x21`/`x22`, count `(4^(d+1)-1)/3`) *and* the single `InitEdt` create are each `EDT_PROP_FINISH` with a non-NULL output-event argument, costing 3 events apiece (1 app STICKY + 1 runtime finish + 1 runtime output); every `seqLCSEdt` create (count `L²`) is `EDT_PROP_NONE` with a non-NULL output event, costing 2 (1 app STICKY + 1 runtime output, no finish); `randInitEdt`/`scoreInitEdt`/`shutDownEdt` all pass `EDT_PROP_NONE` with a NULL output event and make no explicit `ocrEventCreate` call, costing 0 | — |

Worked numbers at the calibrated `args = [65536, 1024, 48]` (`N=65536,
base=1024`, exact power-of-two ratio `L=64`, `d=6`): `recLCSEdt` =
5,461; `seqLCSEdt` = `L²` = 4,096; `randInitEdt` = 128; `scoreInitEdt` =
129; total EDTs = 9,817; Events = `6·64² + 2` = 24,578; DBs = `(64+2)² =
4,356`. Score payload is the headline number: `(N+1)²` total DP cells
(interior tiles alone total exactly `N²`; boundary/corner add `2N+1`
more) × 4 bytes ≈ **16.0 GiB** at this `N` — three to four orders of
magnitude more than `LCS_shared`/`LCS_distributed_ST`'s `O(N)` score
payload (a few hundred KB), because this is the only variant that
materializes the full DP table instead of the antidiagonal-collapsed
`O(N)` representation.

Counter cross-check: verified (1 node, `8 2 1` (`d=2, L=4`) vs `16 2 1`
(`d=3, L=8`)): ΔNUM_EDT_CREATE = 128, ΔNUM_DB_CREATE = 64,
ΔNUM_EVENT_CREATE = 288 — exactly the formulas' deltas (EDTs `(7·4^d-1)/3
+ 4L + 4` = 57→185; DBs `(L+2)²` = 36→100; Events `6·4²+2=98` →
`6·4³+2=386`); the runtime adds a constant baseline of +1 EDT and +1 DB
per run (Events' baseline is +0), giving measured totals 58/37/98 →
186/101/386.

## Wiring

- `InitEdt` (`depc=3`): `S`/`T`/`score` pointer-array DBs, all RW. Its
  body creates every labeled `S`/`T`/score tile and wires the
  `randInitEdt`/`scoreInitEdt` children that fill the non-interior ones.
- Root `recLCSEdt` (`depc=1`) is wired to `InitEdt`'s finish-scope
  completion event — the entire computation is gated on *all* of
  `InitEdt`'s DB creation and boundary-fill work finishing first.
- Internal `recLCSEdt` calls: identical `x11`/`x12`/`x21`/`x22`
  event-chain wiring to the other two variants, still carrying no DB
  dependence.
- Each leaf's `seqLCSEdt` (`depc=6`) is wired to **six** DBs: `S` tile
  (RO, slot 0), `T` tile (RO, slot 1), its own **current** score tile
  (RW, slot 2), and its **left**/**above**/**diagonal** neighbor score
  tiles (RO, slots 3-5) — computed from the tile's row/column via
  `block_i_end`/`block_j_end` arithmetic. This is a genuine tile-to-tile
  DAG: unlike the other two variants, the dependency structure *is* the
  data structure.
- **DB concurrency — the contrast with the other two variants.** Every
  score tile has **exactly one writer** (the one `seqLCSEdt` that owns
  it as "current") and **at most three RO readers** — the leaves
  immediately east, south, and southeast of it, which need it as their
  left/above/diagonal neighbor respectively; never more than three.
  There is no DB anywhere in this program that more than a handful of
  tasks ever touch concurrently — concurrency here tracks the DAG's real
  data dependencies exactly, with no artificial DB-level serialization.
  `S`/`T` tiles keep the same broadcast-RO shape as
  `LCS_distributed_ST` (up to `L` leaves sharing a fixed row or column
  index can read one tile concurrently).
- Peak parallel width of the compute wavefront is `min(rows,cols)` of
  the `L×L` interior tile grid, i.e. `L` (`=2^d`) — the formula the
  quad-tree recursion is a scheduling strategy *for*, not a departure
  from.

## Flow

Phase 1 (`InitEdt`, finish scope): a single task body serially issues
`(L+2)²` `ocrDbCreate` calls (dominated by the `L²` interior score-tile
creates, which are created empty with no fill work) and spawns `4L+1`
`randInitEdt`/`scoreInitEdt` children to fill `S`, `T`, and the boundary/
corner score cells in parallel across ranks; `mainEdt` blocks on this
whole finish scope before creating the root `recLCSEdt`. Phase 2: the
same quad-tree recursion as the other two variants unfolds to depth `d`,
but now each of the `L²` `seqLCSEdt` leaves is a true wavefront tile,
ready once its three neighbor tiles (left, above, diagonal) are
released; peak concurrency `L` (see Wiring), ramping up and down across
the tile grid's antidiagonals the way a standard tiled-DP wavefront
does. Phase 3: `shutDownEdt` prints the scalar with no
internal verification (`CHECK_RESULTS` undefined in this file).

## Placement (base)

An `OCR_APP_OPTIMIZED_PLACEMENT` guard exists in this file
(`scoreIdx`/`blockRowHint`, re-encoding a score tile's label index and
computing an `EDT_AFFINITY` hint from the tile's row band) for the
companion `_hinted` build; the base (`#else`) branches make both
functions no-ops: `scoreIdx` returns the identity index, `blockRowHint`
returns `NULL_HINT`. So base, every create in this file — `InitEdt`,
`recLCSEdt`, `seqLCSEdt`, and `baseEdt` alike — passes `NULL_HINT`, same
as the other two variants. But (as established for `LCS_distributed_ST`)
**labeled GUIDs place differently from ordinary ones regardless of
hint**: a range from `ocrGuidRangeCreate` gets each index's home fixed
round-robin (`home = index % nranks`) at range-creation time, baked into
the GUID itself. Effective policy:

- **EDTs** (`InitEdt`, `recLCSEdt`, `seqLCSEdt`): NULL hint → round-robin
  — `InitEdt` itself is *not* pinned to rank 0, unlike the other two
  variants' initialization work (which is inline `mainEdt` code); it may
  need a remote RW acquire of the `S`/`T`/`score` pointer-array DBs
  (homed at rank 0, ordinary creates) just to begin its own create-loop.
- **`S`/`T`/`score` pointer-array DBs**: ordinary create, NULL hint →
  home = rank 0 (`mainEdt`).
- **`S`/`T`/score labeled tiles**: home = `index % nranks`, fixed
  independent of hint or of which rank's `InitEdt` instance actually
  issues the `ocrDbCreate`.

Consequence: this is the one LCS variant where the data genuinely lands
distributed (score tiles spread round-robin by grid index, not
concentrated anywhere) — but base EDT placement is still an
independent round-robin, uncorrelated with a tile's `index % nranks`
home, so a leaf's RW acquire of its own "current" tile (let alone its
three RO neighbors) is remote far more often than not. The `_hinted` build
exists specifically to close this gap by deriving `EDT_AFFINITY` from
the same row-band arithmetic that decides a tile's home, so the EDT and
its data agree on a rank; base, they don't.

## Placement (hinted)

As-born is wavefront work fed from labeled-GUID score blocks whose homes
round-robin on the raw linear label index — neighbours in the block grid land
on unrelated ranks, so the heavy north-south payload of the wavefront crosses
the wire nearly every step.

The layer places by contiguous ROW BANDS (`blockRowHint`, one band per rank;
`LCS_ROW_BANDS_PER_RANK` widens to k cyclic bands): a block's West neighbour
shares its row and therefore its rank, North/NW share its band except on the
nranks-1 boundary rows, so the dominant payload stays rank-local while the
anti-diagonal frontier still reaches every band once it is a band tall.  The
same band function is applied on BOTH sides: base-case EDTs pin to their
block's band rank, and the labeled index is re-encoded as `band(row) +
nranks*t` so the label-derived round-robin home of each score block lands on
its band's rank too — pure index arithmetic, agreed by every producer and
consumer without communication.  (Measured at the time of the v2 rework: 4n
-23% from the EDT pins alone.)

The 2026-08-21 rework extends the layer on both remaining fronts: S string
tiles are band-steered too (`sIdx` — row i's tile is read only by block-row
i+1's band), and every creation-phase block is created `NO_ACQUIRE`, so a
block whose home is remote is born there instead of materializing on the
creating rank — without it the eager creation loop pulls the whole table
onto one rank.

## Family shape (measured, 15w+1p x 1/2/4/8 nodes, `32768 1024`)

e2e seconds, both versions:

| arm | base 1n/2n/4n/8n | hinted 1n/2n/4n/8n |
|---|---|---|
| val_wb_nocomb | 1.03 / 2.79 / 3.28 / 3.34 | 1.02 / 1.65 / 2.05 / 2.34 |
| val_wb | 1.09 / 2.80 / 3.30 / 3.36 | 1.08 / 1.66 / 2.06 / 2.34 |
| inv_wb | 1.02 / 2.99 / 3.60 / 3.81 | 1.02 / 1.69 / 2.04 / 2.40 |
| excl_retain | 1.02 / 7.76 / 6.70 / 5.86 | 1.01 / 1.79 / 2.29 / 2.64 |

At the calibrated workload (`131072 1024`, L=128) the sweep reads
base 41.6 / 57.6 / 52.7 / 40.7 s and hinted 41.7 / 45.2 / 31.4 /
19.0 s across 1/2/4/8 nodes: base never beats its own single node,
while the band placement does scale INSIDE the wiring cap — 2.2x at 8
nodes against the `(4/3)^7 = 7.5x` span bound, helped by NO_ACQUIRE
spreading the eager table's materialization across the nodes' homes.
Lowering the cap
shows the plateau directly: the same N at `base 8192` (L=16, cap 3.16x)
runs 36.0 / 40.8 / 36.5 / 34.9 s across 1/2/4/8 nodes — eight nodes buy
nothing.  The cap itself no placement can move: the recursion's quadrant gating is the only ordering (the
leaf's datablock dependences satisfy immediately), so creation itself
trickles behind completion and the span caps speedup at `(4/3)^d`.
Measured on one 108-worker node at fixed work (`131072 1024`): 1..8
workers give 47.4 / 29.1 / 24.2 / 16.5 / 16.0 / 14.9 / 14.2 / 13.6 s —
saturation at ~3.5x — and 112 workers run SLOWER (15.5 s) than 12.
Memory is the second wall: the creation phase materializes the whole
`(N+1)²`-cell table before the first leaf runs, so the destroy-at-diag
only trims the tail, never the peak.  Both walls are the 2016 source's
own structure; the answer to both is the restructured version
(`LCS_wavefront` — real tile dependences, lazy creation), which runs the
same `131072 1024` in 0.82 s on the same node.

## Sizing

Both `N` and `base` must keep `N/base` a power of two for correctness
(Structure). Given that, `base` trades tile *count* (`L²`, i.e. wavefront
parallel width `L`) against tile *size* (`base²` cells) at **fixed total
score memory** — the `(N+1)²`-cell DP table's size depends only on `N`,
never on `base`; `base` only decides how finely it's diced. `N` alone
sets memory (`≈4·N²` bytes for the score data), which is the binding
constraint this variant has and the other two do not:

- The calibrated `args = [65536, 1024, 48]` (`L=64`, ~16 GiB score
  payload) is sized to be observable and to give a `64`-wide wavefront —
  comfortably above 8 nodes × 15 workers = 120 workers only if multiple
  tiles per worker overlap in flight; at 1 node the peak width (`64`) is
  already above the 15-worker budget, so this app does offer genuine
  worker-bound scaling, unlike `LCS_shared`/`LCS_distributed_ST`'s
  single-DB-capped shape.
- At `N=65536` the ~16 GiB score payload is spread `1/nranks` per rank
  under the labeled round-robin homing — negligible at 8 nodes but the
  **full** 16 GiB lands on the single rank of a 1-node run (ferrari's
  ~1 TB/node makes this comfortable here, but it is the one LCS variant
  where memory belongs in the sizing decision at all).
- Shrinking `base` (holding `N` fixed) grows `L` (more wavefront
  parallelism, smaller per-tile compute and per-DB payload) without
  changing total score memory; growing `base` does the reverse. Either
  way, keep `N/base` a power of two.

Measured on the Dane-mirror geometry (1 node, 108w+4p, Release,
hinted): `131072 1024` = 41-42 s (68 GB eager table, slab
prepopulation included), and one step up
(`196608 1024`, 155 GB) already exceeds a 570 s ceiling — the per-tile
overhead grows with tile count on top of the serial floor.  The
calibrated arguments stay **`131072 1024`** — also the workload the
two sequential siblings share, so the ladder compares cell by cell: this
row is the paper's no-scale-by-wiring exhibit, and that is the largest
size whose eager footprint and wall time both behave (NO_ACQUIRE spreads
the table across nodes' homes multinode, but a strong-scaling sweep's
one-node cell still carries all of it); the scaling story continues in
`LCS_wavefront`.
