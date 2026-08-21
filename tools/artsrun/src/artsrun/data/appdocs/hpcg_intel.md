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

`npx·npy·npz` sets parallel width and nothing else; `m` sets per-tile grain
(`∝m³` flops and bytes, `∝m²` halo) and the whole memory footprint; `maxIter`
sets chain length and wall time linearly. Because a tile is a serial chain,
**pick `N` first**: `N ≥ 2×(nodes × workers)` keeps every deque fed, and `N`
should factor so bisection lands compact blocks (powers of two per axis are
ideal). Then pick `m` for grain — `m=64` gives ~107 MiB and ~14 Mflop per
level-0 smoother sweep per tile, enough that per-EDT overhead is not the story;
`m=16` (1.7 MiB) turns the app into a pure message-rate probe. Memory is
`≈427·m³·N` bytes and is the real ceiling.

Worked: 1 node × 15 workers → `N=128`, `m=32` (≈14 MiB/tile, 1.7 GiB) runs in
seconds; 8 nodes × 120 workers wants `N ≥ 240` unless `m` is large enough that
halo latency dominates anyway. The catalog's `8 4 4 64 10` fixes `N=128`,
`m=64` — 13.7 GiB resident, 128 tiles across 15…120 workers, so 1 node is 8.5
tiles per worker and 8 nodes is 1.07: strong scaling is a race between
shrinking per-node compute and a halo surface that shrinks only as fast as the
block partition allows. Raising `m` rather than `N` buys wall time without
eroding the 8-node width.
