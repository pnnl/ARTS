# Stencil2D_intel_chandra

*A 2-D five-point star stencil over a rank-per-tile SPMD grid, with a native
OCR-affinity-driven 2-D tile-block→node placement baked into the port.*
Source: `third_party/ocr-apps/apps/Stencil2D/refactored/ocr/intel-chandra/stencil_2d.c`
(~2000 lines, `PROBLEM_TYPE=2` is the header default) + `timers.c`. Byte-for-byte
the same source as `Stencil1D_intel_chandra`'s `stencil_1d.c` — the two
targets differ only in the `PROBLEM_TYPE` compile define.

## Overview

Ports the Intel PRK "Stencil" kernel onto OCR: a square `NP×NP` domain is
tiled into an `NR_X×NR_Y` grid of `NR` tiles ("ranks" in the port's own
vocabulary, distinct from ARTS ranks — see Placement), each running `NT+1`
timesteps of a radius-2 discrete-divergence stencil plus a 4-neighbor
(left/right/top/bottom) halo exchange over pairwise sticky events. At the
final round every tile feeds its local L1 norm and elapsed time into two
independent 10-ary reduction trees (`ADD`, `MAX`); rank 0 compares the
reduced norm against the kernel's analytic answer `(NT+1)·2` and prints
`SUCCESS: L1 norm = …` / `Solution validates` on a match, `ERROR: …`
otherwise — the catalog's `L1 norm = ([\-+0-9.eE]+)` regex reads that value
off either line. `FULL_APP=1` (always, in this build) means the compute is
real, not skipped; the program stresses per-tile task/event churn and
4-way halo-exchange DB traffic at a grain set independently of the
per-tile compute (`np_x·np_y = (NP/NR_X)·(NP/NR_Y)`).

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `NP` | side of the square `NP×NP` domain | 1000 | ✓ parsed in `FNC_init_globalParamH`, propagated through `globalParamH_t` (`DB_MODE_CONST`/RO downstream) — multinode-safe |
| `argv[2]` = `NR` | number of tiles; factored into `NR_X×NR_Y` via `splitDimension` (near-square, largest divisor ≤ `√(NR+1)`) | 16 | ✓ same DB propagation |
| `argv[3]` = `NT` | timesteps; each tile runs `NT+1` rounds (round 0 untimed) | 10 | ✓ same DB propagation |
| all three, or none | `argc==4` (exactly 3 user args) is the only recognized case; 1 or 2 args are silently ignored, defaults used | — | ⚠ no partial-override path |
| `HALO_RADIUS` | stencil radius (2, 5-point-per-axis → 9-point 2-D stencil) | 2 | ✗ compile-time `#define` |
| `FULL_APP` | 1 = real compute; 0 = OCR-overhead skeleton | 1 | ✗ compile-time `#define` |
| `ARITY` | reduction-tree fan-out | 10 | ✗ compile-time in `reduction.h` |

`expect: '22.0'` in the catalog is a pin for the **default-args** run
(`NP=1000, NR=16, NT=10` → `(10+1)·2 = 22`), not for the calibrated
strong-scaling args below — a consensus check only fires when a cell's
actual args equal `expect_args` (empty here, i.e. no args at all).

## Structure

`NR_X,NR_Y = splitDimension(NR)`; `np_x≈NP/NR_X`, `np_y≈NP/NR_Y`. Setup is
the same fixed 10-EDT chain as the 1-D port, then 3 one-time EDTs per tile
(`rankInit`→`init_rankH`→`init_rankDataH`, one `rankInitSpawner` loop),
then `NT+1` rounds of 11 EDTs per tile (`rankMultiTimestepper`, `timestep`
[FINISH], `Lsend`/`Rsend`/`Lrecv`/`Rrecv`/`Bsend`/`Tsend`/`Brecv`/`Trecv`,
`update`) chained serially per tile via `timestep`'s output event.

Both reduction trees (norm `ADD`, timer `MAX`) route through the same
shared `reduction.c` library as the 1-D port (`type=REDUCE`, `ARITY=10`,
one `reductionLaunch` per tile per tree at its last round). With
`C(NR) = ⌊(NR-2)/10⌋ + 1` the count of tiles with at least one child in the
`ARITY`-ary tree (`NR≤11` ⇒ `C=1`, a single-level star), each tree
contributes exactly `3·NR + 2·C(NR) - 1` EDTs and `3·NR - 2` DBs — see the
1-D doc for the call-by-call derivation (identical logic, this port only
differs in the stencil dimensionality, not the reduction plumbing). Two
trees together: `6·NR + 4·C(NR) - 2` EDTs, `6·NR - 4` DBs.

| object | count | notes |
|--------|-------|-------|
| EDTs | `8 + 9·NR + 11·NR·(NT+1) + 4·C(NR)` | `8+9·NR` folds the two reduction trees into the fixed 10-EDT chain (incl. `mainEdt`) + `3·NR` one-time per-tile setup |
| DBs | `28·NR - 1` | `3+22·NR` (global + per-tile handle/payload DBs, NT-independent) plus the two reduction trees' `6·NR-4` |
| DB payload | `xIn`: `8·(np_x+4)·(np_y+4)` B; `xOut`: `8·np_x·np_y` B; L/R halo bufs `8·HALO_RADIUS·np_y` B; T/B halo bufs `8·HALO_RADIUS·np_x` B | `xIn`+`xOut` dominate |
| Events | `5 + 34·NR + 25·NR·(NT+1) + 2·C(NR)` | `25` fresh events per tile per round (not 22 — see below) plus `24·NR` one-time per-tile phase-buffer events at init, plus the two reduction trees' `10·NR+2·C(NR)-10` |

Each round's 25 events are 13 from EDT creation (`timestep`'s
explicit+output+finish = 3; the 8 halo EDTs' (`Lsend`/`Rsend`/`Bsend`/
`Tsend`/`Lrecv`/`Rrecv`/`Brecv`/`Trecv`) output events; `update`'s
explicit+output = 2) plus 12 more from `update`'s own body, which destroys
and recreates the tile's 12 phase-buffered sticky events (the 1-D port's 6
event types, doubled for the extra top/bottom direction) every round. The
initial 12-events-×-2-phase-buffers (`init_rankH`, once per tile) is the
`24·NR` term.

Calibrated args `['25600', '128', '30']` (NP=25600, NR=128, NT=30):
`splitDimension(128)=8×16`, `np_x=3200,np_y=1600` →`xIn`+`xOut`≈82 MB/tile
→ **≈10.5 GB** across 128 tiles; `C(128)=13`; EDTs ≈ `8+1,152+43,648+52` =
**44,860**; DBs ≈ `28·128-1` = **3,583**; Events ≈ `5+4,352+99,200+26` =
**103,583**.

Counter cross-check: verified (1 node, `NP=64 NR=4 NT=2` vs `NP=64 NR=4
NT=4`): predicted absolutes 181/112/443 and 269/112/643 (`NUM_EDT_CREATE` /
`NUM_DB_CREATE` / `NUM_EVENT_CREATE`) match the measured counters exactly,
against a runtime baseline of `+1 EDT, +1 DB, +0 EVT`. As in the 1-D port,
the fixed control-chain/payload-DB formulas (`10+3·NR+11·NR·(NT+1)` /
`3+22·NR`) were already exactly right — the gap was the un-derived
reduction trees (`+26` EDT / `+20` DB at `NR=4`, identical to the 1-D
port since the reduction plumbing doesn't depend on stencil dimension)
plus an event slope of 22 instead of the source's actual 25 per
tile-round.

## Wiring

Each tile's 12 payload DBs plus its double-buffered `rankEventH` pair are
private state, touched only by that tile's own chain and its 4 immediate
neighbors' send/recv EDTs. `Lsend`/`Rsend`/`Bsend`/`Tsend` take `xIn` RO
plus their own phase send buffer RW; `Lrecv`/`Rrecv`/`Brecv`/`Trecv` take
`xIn` RW (halo write-in, same buffer as the send side reads, but disjoint
regions by construction — no runtime-visible conflict). `update` takes
`xIn`/`xOut`/`refNorm` RW, `weight` CONST. Cross-tile edges are event-only:
`Lsend` on tile `(ix,iy)` satisfies `EVT_Rrecv_start` on the `(ix-1,iy)`
tile's current-phase `rankEventH`, wired RO into that neighbor's `Lrecv`
(and symmetrically for the other 3 directions) — a halo DB has exactly one
producer and one consumer per round, never concurrent RW from two tiles.
The neighbor's whole `rankH` family (passed `DB_MODE_CONST`, resolving the
same as RO) is the only cross-tile-boundary object besides the halo buffer
itself. The two reduction trees are the sole many-to-one fan-in, mediated
entirely by `reduction.c`'s own labeled-GUID channel events, outside the
app's own DB graph.

## Flow

Setup is a short serial prefix, parallel only in its last step (`NR`
independent tile-init chains, launched from one `rankInitSpawner` loop).
Compute is `NR` independent per-tile chains of `NT+1` serial rounds; each
round's own 8-EDT halo exchange is internally parallel, and `update` gates
the next round via its own output event so one tile's slow round only ever
stalls its 4 immediate neighbors, not the whole grid. Parallel width is
`NR` throughout compute (bounded, does not grow with `NT`). Every tile
independently launches its two reductions at its own final round; both
trees converge on rank 0's `summary` (output-event-gated so its own
dependence releases aren't truncated), then a dedicated `shutdown` EDT
calls `ocrShutdown()`.

## Placement (base)

Not a NULL-hint program: `ENABLE_EXTENSION_AFFINITY` is on project-wide
(`OCR_EXT_DEFINES`), and the source uses it unconditionally.
`ocrAffinityCount(AFFINITY_PD, …)` returns the ARTS run's actual node count
(`arts_get_total_ranks()`), independent of the app's own `NR`.
`rankInitSpawner` factors that count into a `PD_X×PD_Y` grid
(`splitDimension`) and maps each tile's 2-D coordinate `(id_x,id_y)` onto a
PD-grid block coordinate via `getPartitionID` on each axis independently —
the same block-partition algorithm used to carve the spatial domain — then
hints the tile's `rankInit` EDT and its `rankH` DB (`DB_PROP_NO_ACQUIRE`,
so the creator never touches the payload) onto that PD via
`ocrAffinityGetAt(AFFINITY_PD, pd, …)`. Everything created from `rankInit`
onward re-derives `ocrAffinityGetCurrent()` and stays put, so a tile's
whole private state and compute chain are pinned to one ARTS rank for the
run. Net effect: the `NR_X×NR_Y` tile grid decomposes into up-to-`PD_X×PD_Y`
contiguous 2-D sub-blocks, one per ARTS rank — **neighbor tiles in both
directions are usually co-resident**, and cross-rank halo traffic is
limited to the tile-boundary pairs that straddle a block edge (perimeter of
the blocks, not their area) — real algorithmic locality the port
expresses. Global bookkeeping (`globalH`, `globalParamH`, event ranges,
created from `mainEdt`/`globalInit`/`globalCompute`) uses `NULL_HINT` and
so lives on rank 0 by the ordinary creator-home policy.

## Sizing

`NR` sets SPMD width and (via `splitDimension`) the 2-D PD-block shape;
`NP` sets per-tile compute grain and memory (`np_x·np_y`); `NT` sets serial
chain depth. For N nodes × C workers, pick `NR` a small multiple of `N·C`
(and prefer `NR` with a near-square `splitDimension` factorization, so the
2-D PD-block partition is itself near-square and per-block halo perimeter
stays low relative to block area), then size `NP` for the target memory
footprint (`8·2·np_x·np_y` bytes resident per tile), and `NT` for wall-clock
length. The calibrated args `25600 128 30` give an 8×16 tile grid,
`np_x=3200,np_y=1600` (≈82 MB/tile, ≈10.5 GB total) — long enough at 1 node
(15 workers stealing across 128 tiles) to run minutes rather than seconds,
while 8 nodes still get 16 tiles apiece. As in the 1-D port, memory scales
with `NP²/NR` per tile while `NR` itself is free to raise (more cross-node
parallelism) without moving memory per tile.
