# Stencil1D_intel_chandra

*A 1-D five-point star stencil over a rank-per-tile SPMD grid, with a
native OCR-affinity-driven tile→node placement baked into the port.*
Source: `third_party/ocr-apps/apps/Stencil1D/refactored/ocr/intel-chandra/stencil_1d.c`
(~2000 lines, compiled with `-DPROBLEM_TYPE=1`) + `timers.c`.

## Overview

Ports the Intel PRK "Stencil" kernel (1-D variant of the same PRK-derived
divergence-operator stencil used by `Stencil2D_intel_chandra`, whose source
this file is byte-for-byte identical to — `PROBLEM_TYPE=1` selects the 1-D
`#elif` branches at compile time) onto OCR. The 1-D domain of `NP` points is
sliced into `NR` contiguous tiles ("ranks" in the source's own vocabulary,
distinct from ARTS ranks — see Placement). Each tile runs `NT+1` timesteps:
apply the radius-2 discrete-divergence stencil to its local strip, exchange a
2-point halo with its left/right neighbor via a pair of sticky events per
direction, then move on. At the final timestep every tile feeds its local L1
norm and elapsed time into two independent 10-ary reduction trees (`ADD` for
the norm, `MAX` for wall time); rank 0 compares the reduced norm against the
kernel's known-analytic answer `(NT+1)·1` and prints `Solution validates`
(the scalar the catalog checks) only on a match. With no compute payload
disabled (`FULL_APP=1` always in this build), the program stresses per-tile
task/event churn and halo-exchange DB traffic at a grain set independently of
the actual per-tile compute (`np_x = NP/NR`).

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `NP` | 1-D domain length (points); tile width is `NP/NR` | 1000 | ✓ parsed in `FNC_init_globalParamH` via `ocrGetArgc`/`ocrGetArgv`, propagated through the `globalParamH_t` DB (`DB_MODE_CONST`/RO to every downstream EDT) — multinode-safe |
| `argv[2]` = `NR` | number of tiles ("ranks" in the port's terms); sets SPMD width, independent of physical ARTS node count | 16 | ✓ same DB propagation |
| `argv[3]` = `NT` | timesteps; each tile actually runs `NT+1` rounds (round 0 is untimed warm-up) | 10 | ✓ same DB propagation |
| all three, or none | the parser only overwrites its defaults when `argc==4` (all 3 args); a partial argv silently falls back to the compiled defaults | — | ⚠ no partial-override path — passing 1 or 2 args is silently ignored, not an error |
| `HALO_RADIUS` | stencil radius (2-point halo, 5-point stencil) | 2 | ✗ compile-time `#define` in `stencil.h` |
| `FULL_APP` | 1 = run the real compute+halo work; 0 = OCR-overhead-only skeleton | 1 | ✗ compile-time `#define`, not overridden by the build |
| `ARITY` | fan-out of the two reduction trees | 10 | ✗ compile-time `#define` in the shared `reduction.h` library |

## Structure

Let `NR_X = NR` (1-D — no second axis) and `np_x ≈ NP/NR_X` (last tile
absorbs the remainder). Setup is a fixed 10-EDT chain (`mainEdt` →
`globalInit`(FINISH) → `init_globalParamH` → `init_globalH` →
`init_globalH_part1` → `rankInitSpawner` → `globalCompute` →
`rankComputeSpawner` → `summary` → `shutdown`), independent of `NR`/`NT`.
`rankInitSpawner` then fans out 3 one-time EDTs per tile
(`rankInit`→`init_rankH`→`init_rankDataH`), and each tile's steady state is
`NT+1` rounds of 7 EDTs (`rankMultiTimestepper`, `timestep` [FINISH],
`Lsend`/`Rsend`/`Lrecv`/`Rrecv`, `update`) chained one round to the next
through `timestep`'s output event — i.e. a tile's own rounds run serially,
while the `NR` tiles run those chains in parallel.

Both reduction trees (norm `ADD`, timer `MAX`) route through the shared
`reduction.c` library (`type=REDUCE`, `ARITY=10`): each of the `NR` tiles
calls `reductionLaunch` once per tree at its last round, and every tile
whose id has at least one child in the `ARITY`-ary tree (`id·ARITY+1 <
NR`) additionally runs the channel-setup + up/down relay logic. Let
`C(NR) = ⌊(NR-2)/10⌋ + 1` be that "has-children" tile count (`NR≥2`; for
`NR≤11` this is exactly 1 — a single-level star with tile 0 as root). Per
tree this contributes exactly `3·NR + 2·C(NR) - 1` EDTs and `3·NR - 2` DBs,
independent of `ARITY`'s exact branching once `C(NR)` is known — derived
call-by-call from `reduction.c`'s `reductionLaunch`/`reductionEdt`/
`reductionSendChannelEdt`/`reductionRecvChannelEdt` (root: `Recv`-only +
its `reductionRecvUp`-spawned phase-1 instance; interior/leaf tiles: `Send`
+, if they themselves have children, `Recv` too). Two trees together:
`6·NR + 4·C(NR) - 2` EDTs, `6·NR - 4` DBs — this is the "extra" the bounded
`O(NR)` estimate below used to gesture at without deriving.

| object | count | notes |
|--------|-------|-------|
| EDTs | `8 + 9·NR + 7·NR·(NT+1) + 4·C(NR)` | `8+9·NR` folds in the two reduction trees' `6·NR+4·C(NR)-2` against the fixed 10-EDT chain (incl. `mainEdt`) and `3·NR` one-time per-tile setup |
| DBs | `24·NR - 1` | `3+18·NR` (global + per-tile handle/payload DBs, NT-independent) plus the two reduction trees' `6·NR-4` |
| DB payload | `xIn`: `8·(np_x+4)` B; `xOut`: `8·np_x` B; halo bufs: `8·HALO_RADIUS`=16 B each | dominant memory is `xIn`+`xOut` per tile |
| Events | `5 + 22·NR + 15·NR·(NT+1) + 2·C(NR)` | `15` fresh events per tile per round (not 12 — see below) plus `12·NR` one-time per-tile phase-buffer events at init, plus the two reduction trees' `10·NR+2·C(NR)-10` |

Each round's 15 events are 9 from EDT creation (`timestep`'s explicit STICKY
plus its FINISH output+finish pair = 3; `Lsend`/`Rsend`/`Lrecv`/`Rrecv` each
carry one output event = 4; `update`'s explicit STICKY plus its output event
= 2) plus 6 more from `update`'s own body, which destroys and recreates the
tile's 6 phase-buffered sticky events (`EVT_Lsend_fin`/`EVT_Rsend_fin`/
`EVT_Lrecv_start`/`EVT_Rrecv_start`/`EVT_Lrecv_fin`/`EVT_Rrecv_fin`) every
round — `ocrEventDestroy` doesn't count, the 6 `ocrEventCreate`s that follow
it do. The initial 6-events-×-2-phase-buffers (`init_rankH`, once per tile)
is the `12·NR` term.

Calibrated args `['700000000', '128', '30']` (NP=700M, NR=128, NT=30): tile
width `np_x ≈ 5.47M` points, `xIn`+`xOut` ≈ 87.5 MB/tile → **≈11.2 GB**
payload across 128 tiles; `C(128)=13`; EDTs ≈ `8+1,152+27,776+52` =
**28,988**; DBs ≈ `24·128-1` = **3,071**; Events ≈ `5+2,816+59,520+26` =
**62,367**.

Counter cross-check: verified (1 node, `NP=64 NR=4 NT=2` vs `NP=64 NR=4
NT=4`): predicted absolutes 133/96/275 and 189/96/395 (`NUM_EDT_CREATE` /
`NUM_DB_CREATE` / `NUM_EVENT_CREATE`) match the measured counters exactly,
against a runtime baseline of `+1 EDT, +1 DB, +0 EVT`. The original doc's
fixed control-chain/payload-DB formulas (`10+3·NR+7·NR·(NT+1)` / `3+18·NR`)
were already exactly right — the gap was entirely the un-derived reduction
trees (`+26` EDT / `+20` DB at `NR=4`) plus an event slope of 12 instead of
the source's actual 15 per tile-round.

## Wiring

Each tile's private state after setup is 8 payload DBs plus its
`rankEventH` pair (double-buffered by `itimestep%2`, so round `k`'s sends can
run before round `k-1`'s receive-side events have been destroyed). Every
`Lsend`/`Rsend`/`Rrecv`/`Lrecv` EDT takes `xIn` RO (send side) or RW (recv
side, in place) plus its own phase's send/recv buffer RW; `update` takes
`xIn`/`xOut`/`refNorm` RW and `weight` CONST (CONST resolves to the same
per-node-exclusive/shared-RO ARTS mapping as RO — see the OCR-shim mode
table). Cross-tile deps are event-only, never a direct DB dependence: `Lsend`
on tile `i` satisfies `EVT_Rrecv_start` on tile `i-1`'s *current* rankEventH
DB with tile `i`'s send buffer; the neighbor's `Lrecv` is wired to that same
event RO. So a halo DB (16 bytes) has exactly one producer and one consumer
per round — no DB ever has concurrent RW from two tiles, and the neighbor's
whole `rankH`-family DB tree (passed `DB_MODE_CONST` into
`rankMultiTimestepper`/`timestep`) is the only object read across the tile
boundary besides the halo buffer itself. The two reduction trees (`ADD`
norm, `MAX` time) are the only place where an object's provenance fans in
from many tiles — mediated by the `reduction.c` library's own channel/sticky
events, not the app's own DB graph.

## Flow

Setup is a short serial prefix (global → per-parameter → per-tile init, all
depth ≤5) whose parallel width is `NR` only at the very last step
(`rankInit`/`init_rankH`/`init_rankDataH`, one chain per tile, launched from
a single `rankInitSpawner` loop). Compute is `NR` independent per-tile
chains of `NT+1` serial rounds each — parallel width is `NR` throughout
(bounded, does not grow with `NT`), and each round's own halo exchange (4
EDTs) is itself fully parallel within the round. `update` gates the next
round via its own output event, so a slow round on one tile does not stall
its neighbors beyond their own halo dependence. The last round on every tile
independently launches its reduction; both trees converge on rank 0's
`summary`, which is gated by a dedicated output event so `summary`'s own
dependence releases are not truncated by shutdown, then a final `shutdown`
EDT calls `ocrShutdown()`.

## Placement (base)

This port is NOT a NULL-hint program: `ENABLE_EXTENSION_AFFINITY` is defined
project-wide for every ARTS benchmark build (`OCR_EXT_DEFINES` in
`benchmarks/apps/CMakeLists.txt`), and the source uses it unconditionally.
`ocrAffinityCount(AFFINITY_PD, …)` resolves (via the OCR shim) to
`arts_get_total_ranks()` — the actual ARTS node count of the run, decoupled
from the app's own `NR` tile count. `rankInitSpawner` partitions the `NR`
tile ids into `affinityCount` contiguous blocks (`getPartitionID`, the same
algorithm used for the spatial domain split) and hints each tile's `rankInit`
EDT — and its `rankH` DB, created `DB_PROP_NO_ACQUIRE` so the creator never
touches the payload — onto the owning ARTS rank via
`ocrAffinityGetAt(AFFINITY_PD, pd, …)`. Every EDT and DB created from inside
`rankInit` onward re-derives its own current affinity
(`ocrAffinityGetCurrent`) and stays there, so a tile's whole private state and
compute chain live on one ARTS rank for the run's duration. Net effect: at 1
ARTS rank all 128 (calibrated) tiles collapse onto that rank (parallelism
comes from worker-thread stealing only); at N ARTS ranks the tile-id range
splits into N contiguous blocks, so **neighbor tiles that halo-exchange are
usually co-resident**, and cross-rank halo traffic is limited to the ~`N-1`
tile-boundary pairs that straddle a block edge — genuine algorithmic
locality the port expresses, unlike a NULL-hint program. Global bookkeeping
objects created from `mainEdt`/`globalInit`/`globalCompute` (the `globalH`,
`globalParamH`, event ranges) use `NULL_HINT` and so follow the ordinary
creator-home policy (all live on rank 0, where `mainEdt` runs).

## Sizing

`NR` sets the SPMD width (independent of the ARTS node count — the tile grid
just repartitions across however many ranks are actually running); `NP` sets
per-tile compute grain (`np_x = NP/NR`) and memory; `NT` sets serial chain
depth per tile. For N nodes × C workers, pick `NR` a small multiple of
`N·C` so every worker gets several tiles to steal (tiles are the only unit
of cross-node parallelism — parallel width never exceeds `NR`), then size
`NP` for the target memory/compute-per-round footprint (`8·2·np_x` bytes of
resident payload per tile), and `NT` for wall-clock length (`update`'s own
timer only counts rounds 1..`NT`, so `NT` past a few dozen barely changes
setup-vs-steady-state ratio). The calibrated args `700000000 128 30` give
`np_x≈5.47M` (87.5 MB/tile, ≈11.2 GB total across 128 tiles) — sized so 1
node's 15 workers stay busy across 128 tiles for minutes rather than
seconds, while 8 nodes still get 16 tiles apiece. Memory is the binding
resource before task count is: doubling `NP` doubles memory and per-round
compute together, while `NR` is free to raise for more cross-node
parallelism without touching memory per tile.
