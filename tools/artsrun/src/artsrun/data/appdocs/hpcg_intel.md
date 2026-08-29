# hpcg_intel

*HPCG's preconditioned conjugate gradient over an SPMD grid of cube tiles —
a 26-neighbour halo exchange per operator application, a 4-level multigrid
V-cycle inside every iteration, three global sums to close it.*
Source: `third_party/ocr-apps/apps/hpcg/refactored/ocr/intel/hpcg.c`
(~2400 lines) + `timers.c`, linked against the `reduction` library
(`apps/libs/src/reduction/reduction.c`, `ARITY = 10`).

## Overview

Solves `Ax = b` for the HPCG matrix — a 27-point stencil, diagonal 26,
off-diagonals −1, `b` set to the row sums so the exact solution is all ones —
by preconditioned CG. The global grid is cut into `N = npx·npy·npz` cube tiles
of `m³` points; one *tile* (the port calls it a "rank", distinct from an ARTS
rank) is one serial EDT chain owning one large private datablock. The
preconditioner is a 4-level multigrid V-cycle smoothed by symmetric
Gauss-Seidel sweep pairs. At termination each tile computes `Σ(1−xᵢ)²` and a
final REDUCE prints `final deviation: …` — a convergence residual, not a
checksum, so the catalog's scalar moves with `m` and `maxIter`. Three stresses
at once: a ~100 MB per-tile working set touched RW by every EDT in the chain,
11 halo exchanges per iteration each fanning out to up to 26 neighbours, and
three latency-bound global reductions.

## Parameters

The parser accepts 0, 3, 4, 5 or 6 user arguments; each prefix extends the
previous one.

| arg | meaning | default | CLI reachability |
|-----|---------|---------|------------------|
| `argv[1..3]` = `npx,npy,npz` | tile grid; `N = npx·npy·npz` | 3,4,5 (`N=60`) | ✓ parsed in `mainEdt`, copied into the shared DB every tile reads RO — multinode-safe |
| `argv[4]` = `m` | tile cube edge, **rounded up to a multiple of 16** | 16 | ✓ same path |
| `argv[5]` = `maxIter` | CG iterations; rejected above `HPCGMAXITER` (50) | 50 | ✓ same path |
| `argv[6]` = `debug` | print level 0/1/2 | 0 | ✓ same path |
| 1 or 2 arguments | `bomb()` prints `ERROR … TERMINATING`, calls `ocrShutdown()` and exits(1) | — | ✓ fatal, as the message says; same for `npx<=0`, `m==0` and `maxIter>50` |
| `PRECONDITIONER`, `COMPUTE`, `AFFINITY`, `TIMER`; multigrid depth; `ARITY` | V-cycle / real arithmetic / native PD placement / per-phase timestamps; 4 levels (`m/2^l`); reduction fan-out | on,on,on,off; 4; 10 | ✗ compile-time (`NO_*` variants, hardwired levels, `reduction.h`) |

`expect_args ['1','1','1','16','5']` is a **degenerate pin**: `N=1` nulls every
halo direction and takes `reductionEdt`'s `nrank==1` short circuit, so the
pinned `0.001279` checks the local linear algebra and the V-cycle, not a byte
of the communication the app exists to exercise.

## Structure

Write `mt_l = (m/2^l)³`, `ht_l = (m/2^l+2)³`, let
`L = (3·npx−2)(3·npy−2)(3·npz−2) − N` be the number of directed neighbour links
(`26N` only if every tile were interior) and `I = ⌈(N−1)/10⌉` the ranks with a
child in the reduction tree.

| object | count | size |
|--------|-------|------|
| private block | `N` (one per tile) | `≈ 427·m³` B — 4 levels of matrix (`27·Σmt·8`) + column indices (`27·Σmt·4`) + diagonal indices, the halo'd `Z`/`P` vectors (`ht·8`) and `R/AP/X/B` (`mt·8`), behind a ~3 kB header |
| shared / reduction-private / scalar blocks | 1 / `N` / `N` | 56 B (RO to all `N` `initEdt`s) / struct / 8 B |
| halo blocks | `11·L` per iteration | 8 corners × 8 B, 12 edges × `8m_l` B, 6 faces × `8m_l²` B — `8((m_l+2)³−m_l³)` B per exchange |
| EDT creates | `55N + 3(2N+I−1)` per iteration | 55 per tile: 5 `hpcg` + 17 `mg` + 11 `haloExchange` + 11 `unpack` + 7 `smooth` + 4 `spmv` |
| Event creates | `37N` per iteration | all `OCR_EVENT_ONCE_T`: 1 (`hpcg` p1) + 3 (`hpcg` p3) + 10 per MG level 0–2 + 3 at level 3 |
| DB creates | `11L + 9N − 6` per iteration | halo blocks + `3N−2` per ALLREDUCE × 3 |
| one-time | `9N + 3I` EDTs, `9N + L − 3` DBs, `6N + 3L + I − 4` events | `mainEdt` and `wrapUpEdt`, then four EDTs per tile: `initEdt`, `channelInitEdt`, the phase-0 `hpcgEdt` and the phase-1 clone that one creates. Blocks: the shared block, a private, a reduction-private and an 8-byte scalar block per tile, and a GUID carrier per link. Events: a `returnEVT` channel per tile, and per link a CHANNEL event plus a labeled sticky created from *both* ends (the shim counts both attempts). No finish EDTs |

On top of that the reduction contributes `2N+I−1` EDTs, `N−1` DBs and
`5(N−1)+I` events of one-off tree setup — per tree link a channel pair and a
labeled sticky attempted from both ends, plus the output events of its two
setup EDTs, the only `ocrEdtCreate`s in the program that pass one — and then
runs `3·maxIter + 2` times. Two of those calls
are outside the iteration loop: the phase-0 `rtr` ALLREDUCE (`2N+I−1` EDTs,
`3N−2` DBs, like any other) and the closing REDUCE, which skips the down pass
and costs `N+I` EDTs and `2N−1` DBs. All of this assumes `N > 1`; a single tile
short-circuits the tree entirely.

Calibrated args `['8','4','4','64','10']`: `N=128`, `m=64`, `L=2072`, `I=13`.
Private blocks **≈107 MiB each → ≈13.7 GiB resident**; halo traffic ≈0.97 MiB
per tile per iteration. Setup is 1,191 EDTs / 3,221 DBs / 6,993 events; per
iteration cluster-wide **7,844 EDT creates, 23,938 DB creates, 4,736 event
creates**; over the run ≈79.6k / ≈243k / ≈54.4k.

Counter cross-check: verified (1 node, `N=8`, `L=56`, `I=1`). `2 2 2 16 2` and
`2 2 2 16 5` measure 1,052/2,516 EDTs, 1,490/3,536 DBs, 805/1,693 events — the
per-iteration formulas give those deltas exactly (488/682/296 per iteration),
and the one-time terms give the absolutes (75/125/213 here) once the runtime's
constant +1 EDT and +1 DB per run are added. `2 2 2 16 3` and `2 2 2 32 3`
measure identical triples, 1,540/2,172/1,101, confirming that `m` moves block
sizes and flops but no object count. Getting there corrected all three setup
terms: the EDT count had omitted `wrapUpEdt`, both `hpcgEdt` creates per tile
and every reduction EDT; the DB count had omitted the per-tile scalar block and
the reduction's blocks; the event count had omitted each tile's `returnEVT`.

## Wiring

A tile's private block is the spine: `hpcgEdt` → `mgEdt` → `haloExchangeEdt` →
`unpackEdt` → `smoothEdt`/`spmvEdt` → back, each hop a fresh
`OCR_EVENT_ONCE_T` carrying the same block **RW**. Every stage releases before
satisfying, so exactly one EDT holds it at a time — a 107 MiB block with a
strictly serial single-writer history and no sharing at all.
`haloExchangeEdt` is the only fan-out: it creates `unpackEdt` (26 RO slots),
packs each face/edge/corner into a fresh small DB, releases it and satisfies
the matching **channel** event, then hands the private block on and satisfies
a control `packEVT` (NULL payload) so the consumer waits for both the local
pack and the remote unpack. A halo block therefore has exactly one writer (the
packing tile, at create time) and exactly one RO reader (the neighbour's
`unpackEdt`, which destroys it) — one-shot migration, never shared.

Neighbour channels are established once: `initEdt` reserves `26N` labeled
STICKY GUIDs, creates a CHANNEL event per outgoing direction, ships its GUID in
an 8-byte DB through the sticky at index `26·myrank+ind`, and collects the
mirror at `26·neighbour + (25−ind)`; `channelInitEdt` installs the arrivals as
`haloRecvEVT[]`. Reductions use the same idiom: `reduction.c` builds a 10-ary
tree over labeled stickies and moves 8-byte blocks up (**RO** into the parent's
`yourdata` slots, destroyed there) and back down. The only DB with many
concurrent readers anywhere is the 56-byte shared block (`N` RO readers at
startup); the only many-to-one fan-in is the reduction tree's `ARITY`-way join.

## Flow

`mainEdt` (rank 0 only) reserves two GUID ranges and spawns `N` `initEdt`s;
each runs the serial preamble — four `matrixfill` passes building `27·Σmt`
column indices, ≈8.1 M entries per tile at `m=64`, the dominant startup cost —
then the channel rendezvous. Each tile then runs its own chain: `hpcg` p0 seed
(`rtr`) → **p1** convergence test → `mg` V-cycle → **p2** `rtz` → **p3** halo +
SpMV → **p4** `pAp` → **p5** update `x`,`r`,`rtr` → p1. The V-cycle is `mgStep`
0..6 with `level = 3−|mgStep−3|`: levels 0–2 each smooth → SpMV → restrict →
recurse → prolong → smooth (5 `mgEdt` bodies, 3 exchanges), level 3 smooths
once and returns — 17 `mgEdt`, 10 exchanges, 7 smoothers, 3 SpMVs per cycle.

Parallel width is `N` and only `N`: a tile's chain is serial end to end,
V-cycle included, so the machine is busy only while `N ≳ nodes × workers`. The
coarse levels do **not** narrow the width — every tile stays active at every
level — they narrow the *work* by 8× per level while the exchange keeps its
full 26-message shape, so level 3 is essentially pure message latency. Each
reduction is a global barrier `2⌈log₁₀N⌉` hops deep; an iteration is 11
exchange barriers plus 3 reduction barriers. Serial points: `mainEdt`'s spawn
loop and the single `wrapUpEdt`.

## Placement (base)

This is **not** a NULL-hint program, and no `OCR_APP_OPTIMIZED_PLACEMENT`
layer exists in the source — what follows is as published. `mainEdt` calls
`ocrAffinityCount(AFFINITY_PD, &PDcount)` (the ARTS node count) and maps each
tile through `getMyPD`, a **recursive bisection**: halve the PD range and the
tile grid's currently longest axis together, recurse. Tile `myrank`'s `initEdt`
is hinted onto the resulting PD; from `initEdt` on, every create re-reads
`ocrAffinityGetCurrent()` into `pbPTR->myAffinityHNT` and passes it to every
`ocrEdtCreate`, so a tile's whole chain — `hpcg`, `mg`, `haloExchange`,
`unpack`, `smooth`, `spmv` and its reduction EDTs — is pinned to one ARTS rank
for the run. DBs carry NULL hints, so home = creating rank = that same node.

The consequence is the good one: the 107 MiB private block never leaves its
node, and the tile grid decomposes into contiguous 3-D sub-blocks, one per node
(at `8 4 4`: `4×4×4` per node on 2 nodes, `2×2×4` on 8), so cross-node halo
traffic is the *surface* of those sub-blocks rather than all `26N` links. What
still crosses: halo blocks straddling a block face, and every 8-byte reduction
block — the reduction tree is indexed by tile id, not by node, so its links
ignore the placement entirely. Global objects created in `mainEdt` (shared
block, GUID ranges, `finalOnceEVT`, `wrapUpEdt`) are NULL-hinted, on rank 0.

## Sizing

`npx·npy·npz` sets the number of tiles and `m` the edge of the cube each tile
owns.  Memory is `≈427·m³·N` bytes.  `m` is not only a size: a tile computes
`∝m³` and exchanges `∝m²`, so **`m` is the granularity**, and the ratio the
runtime has to cover is `6/m`.

Four levels of multigrid halve the edge four times, so `m` must be a multiple
of 16 (the program rounds up) and 16 is the floor -- the coarsest grain the
program can be given.  That floor is where the previous cycle left this row, and
it is why the row looked like something it is not:

| | `8 8 8 16 50` | `8 8 8 32 50` |
|---|---|---|
| 1 node x 15 workers | 6.31 s | 50.19 s |
| 2 nodes | 32.86 s | 45.49 s |
| | **0.19x -- collapses** | **1.10x -- scales** |

Same tile count, same number of exchanged blocks, four times the face.  At the
floor the exchange is not covered by the compute and the row falls apart at the
first node boundary; one step above it, the row scales.  The earlier reading ran
the causation backwards -- it saw the collapse at `m = 16`, concluded the row was
an anti-scaler, and then used the 10-30 s window that judgment implies to keep
`m` at the floor.

At the campaign's own arguments the row is a strong scaler, and the gain does
not run out by eight nodes:

| geometry | `16 16 16 32 50` | cumulative | per doubling |
|---|---|---|---|
| 1 node x 15 workers | 524.7 s | 1.00x | -- |
| 2 nodes | 288.6 s | 1.82x | 1.82x |
| 4 nodes | 188.4 s | 2.78x | 1.53x |
| 8 nodes | 130.7 s | **4.02x** | 1.44x |

All four give the same deviation, holding 87-141 GB.  So it is calibrated
against ~150 s, which those arguments reach at 155.2 s, 160.9 s and 159.0 s on
the three coherence families, holding 125 GB at the anchor.

A trend read at a smaller grid understates this row badly -- `8 8 8 32 50` gives
2.05x over the same eight nodes against 4.02x here.  The reason is the same
surface-to-volume ratio that makes `m` matter: at two nodes the bisection puts
25% of an 8³ grid's tiles on the cut plane against 12.5% of a 16³ one, so the
small grid is the least favourable size this application has.

The 4096 tiles are 1.19x the largest geometry's 3456 workers -- above the width
floor, but not by much, and there is no room to raise it.  Work is
`tiles · m³ · maxIter`; the iteration count is the benchmark's own 50 and the
edge cannot go below 32 without losing the scaling, so more tiles only leave the
window.  Granularity wins that trade, because without it the row does not scale
at all.

`maxIter` stays at the application's own cap of 50: in HPCG the CG iteration
count is fixed by the algorithm and the benchmark's variable is the grid.  `T`
is that cap's default when the run gives no iteration argument -- its comment
used to read "number of time steps", which is what led this catalog to describe
the run as 50 timesteps of 50 iterations.

## Placement (hinted)

There is no hinted tier, and the source carries no placement guard.  A layer was
written and measured before that conclusion: a consumer home for the per-exchange
halo blocks, computed once per direction from the neighbour's rank, applied only
to directions that actually leave the domain.

At `m = 16` it pays -- 1.09x at two nodes, 1.11x at four, 1.21x at eight, the
gain growing with node count exactly as more crossings would predict.  At
`m = 32` it is neutral to three digits (50.21 / 45.54 / 31.38 / 24.63 s against
50.19 / 45.49 / 31.35 / 24.44 s).  The layer was mitigating the cost that correct
sizing removes outright, not fixing a placement defect, so it is not kept.

The application already places itself well, which is why little was left to take.
An unguarded `AFFINITY` define pins every EDT to its rank's domain, and the
rank→domain map is a recursive bisection of the 3D grid (`getMyPDc`, splitting
the longest axis), so each domain holds a compact box rather than a stride of
scattered ranks.  A reduction tree renumbered into domain order was also tried --
the library builds its tree on the participant index while the ranks are placed
by bisection, so the two maps are unrelated -- and it lost: 1.00x, 0.92x, 0.84x
at two, four and eight nodes, worsening with node count, since domain-ordered
numbering concentrates the tree's upper levels in one domain.

There is no restructured tier either: the row scales.  The structural fix a
restructure would bring is the one `hpgmg_dist` made -- face buffers created once
at init and reused instead of a fresh datablock per exchange -- and this row does
not need it.
