# hpcg_intel_Eager

*The same HPCG solver as `hpcg_intel`, rewritten so no message buffer is ever
allocated twice: every halo and reduction buffer is created once at startup,
double-buffered, and reused for the whole run.*
Source: `third_party/ocr-apps/apps/hpcg/refactored/ocr/intel-Eager/hpcgEager.c`
(~2500 lines) + `timers.c`, linked against `reductionEager`
(`apps/libs/src/reductionEager/reductionEager.c`, `ARITY = 2`) and `timer`.

## Overview

"Eager" names the **application's own buffer/communication scheme**, not a
runtime coherence mode: the port targets an OCR branch with *eager
datablocks*, whose recommended usage (`READMEeager`) is (1) channel events
only, (2) the **receiver** owns the channel event, (3) create the datablocks
once and reuse them. Requirement (3) is what changed the program. Against
`hpcg_intel`:

- **Halo buffers are preallocated and reused.** `initEdt` creates two blocks
  per outgoing direction, sized at *level 0* (the largest), and keeps their
  GUIDs in the private block; the steady state never calls `ocrDbCreate` or
  `ocrDbDestroy` on the halo path.
- **`packEdt` returns as a real EDT.** In `hpcg_intel` packing is inline
  because it creates the buffer it fills; here the buffer is a dependence, so
  packing is its own EDT taking the 26 reused blocks RW, private block RO.
- **Double buffering by `toggle`.** `haloExchangeEdt` flips a bit in the
  private block; each direction owns two buffers and two channel events and
  the exchange alternates, so generation `k+1` cannot overwrite a buffer
  generation `k`'s reader may still hold.
- **Channel-event ownership is reversed**: the receiver creates both channel
  events and ships their GUIDs to the sender through the labeled sticky.
- **The reduction tree becomes binary and buffer-free.** `reductionEager` has
  `ARITY = 2` (against `reduction`'s 10), preallocates and double-buffers its
  up/down blocks, and adds `reductionSendUpEdt` /
  `reductionSendDownAndBackEdt` so a send is an EDT over a resident buffer.

Mathematics, scalar, argument surface and pinned answer are unchanged; the
catalog leaves this row `default_enabled: false`. Halo blocks carry
`OCR_HINT_DB_EAGER`, which the ARTS shim documents as advisory and ignores —
coherence here is a build-wide protocol, not a per-DB policy — so on this
runtime the variant is purely a *DB-churn* ablation of `hpcg_intel`.

## Parameters

Byte-identical parser to `hpcg_intel`: 0, 3, 4, 5 or 6 user arguments.

| arg | meaning | default | CLI reachability |
|-----|---------|---------|------------------|
| `argv[1..3]` = `npx,npy,npz` | tile grid; `N = npx·npy·npz` | 3,4,5 | ✓ `mainEdt` → shared DB → every tile — multinode-safe |
| `argv[4]` = `m` | tile cube edge, rounded up to a multiple of 16 | 16 | ✓ same path |
| `argv[5]` = `maxIter` | CG iterations, capped at 50 | 50 | ✓ same path |
| `argv[6]` = `debug` | print level 0/1/2 | 0 | ✓ same path |
| 1 or 2 arguments | `bomb()` prints + `ocrShutdown()`, **then returns**; run continues on defaults | — | ⚠ diagnosed, not fatal |
| `CONSECUTIVE_PLACEMENT` | replace the bisection tile→node map with `myrank/(N/PDcount)` | undefined | ✗ compile-time; **not** set in this build |
| `ARITY`; `ENABLE_EAGER_DATABLOCKS`; `PRECONDITIONER`, `COMPUTE`, `AFFINITY`, `TIMER` | reduction fan-out (binary here); sets `OCR_HINT_DB_EAGER`; as `hpcg_intel` | 2; on; on,on,on,off | ✗ compile-time (`reductionEager.h`; the EAGER hint is ignored by the ARTS shim) |

`expect_args ['1','1','1','16','5']` is the same degenerate single-tile pin:
`N=1` leaves no halo directions and takes `reductionEdt`'s `nrank==1` short
circuit, so it validates the arithmetic and nothing this variant changed.

## Structure

With `L = (3·npx−2)(3·npy−2)(3·npz−2) − N` directed neighbour links and
`I = ⌈(N−1)/2⌉` ranks with a child:

| object | count | size |
|--------|-------|------|
| private block | `N` | `≈ 427·m³` B — `hpcg_intel`'s layout plus `toggle`, `haloDBK[26][2]` and the doubled event arrays |
| halo buffers | `2L`, **created once** | level-0 sized: 8 B corners, `8m` B edges, `8m²` B faces; `≈8((m+2)³−m³)` B per direction-set per copy |
| reduction buffers | `2(N−1)` up + `2(N−1)` down, **created once** | 8 B each |
| shared / reduction-private / scalar blocks | 1 / `N` / `N` | 56 B / struct / 8 B |
| EDT creates | `66N + 3(3N+2I−2)` per iteration | 66 per tile: 5 `hpcg` + 17 `mg` + 11 `haloExchange` + 11 `pack` + 11 `unpack` + 7 `smooth` + 4 `spmv` |
| Event creates | `37N` per iteration | unchanged from `hpcg_intel` — the ONCE chain events |
| DB creates | `3N` per iteration | **only** the three reduction launch blocks (8 B); zero on the halo path |
| one-time | `11N + 3I − 1` EDTs, `15N + 3L − 9` DBs, `10N + 4L − 7` events | `mainEdt` and `wrapUpEdt`, then four EDTs per tile: `initEdt`, `channelInitEdt`, the phase-0 `hpcgEdt` and the phase-1 clone that one creates. Blocks: the shared block, a private, a reduction-private and an 8-byte scalar block per tile, and per valid direction two halo buffers and a GUID carrier. Events: a `returnEVT` channel per tile, and per direction two CHANNEL events and a labeled sticky created from *both* ends. No finish EDTs, no output events |

The reduction adds `2N` EDTs (a clone and a `channelRecvEdt` per rank),
`10(N−1)` DBs and `N + 8(N−1)` events of one-off tree setup: a ONCE per rank,
and per tree link each end creates a GUID carrier, two resident transfer
buffers, two channel events and two labeled-sticky attempts — then
`channelRecvEdt` allocates each end's two buffers a second time, which is why
a link costs ten blocks to install four. The tree then runs `3·maxIter + 2`
times; the two calls outside the iteration loop are the phase-0 `rtr`
ALLREDUCE and the closing REDUCE (`2N+I−1` EDTs, `N` DBs, no down pass).
`N > 1` throughout — a single tile short-circuits.

Calibrated args `['4','4','3','64','15']`: `N=48`, `m=64`, `L=652`, `I=24`.
Private blocks **≈107 MiB × 48 ≈ 5.0 GiB**; halo buffers ≈396 KiB per interior
tile (≈19 MiB total). Setup is 599 EDTs / 2,667 DBs / 3,081 events. Per
iteration cluster-wide: **3,738 EDT creates, 144 DB creates, 1,776 event
creates** — against `hpcg_intel`'s `11L + 9N − 6` DB creates for the same
shape, a **~50× reduction in DB churn per iteration**. Over 15 iterations
≈56.7k EDTs, ≈4.8k DBs, ≈29.7k events.

Counter cross-check: verified (1 node, `N=8`, `L=56`, `I=4`). `2 2 2 16 2` and
`2 2 2 16 5` measure 1,336/3,190 EDTs, 328/400 DBs, 889/1,777 events — the
per-iteration formulas give those deltas exactly (618/24/296 per iteration,
which pins the reduction's `3N + 2I − 2` EDT term), and the one-time terms give
the absolutes (99/279/297 here) once the runtime's constant +1 EDT and +1 DB
per run are added. The event count was already right; the EDT count had omitted
`wrapUpEdt`, both `hpcgEdt` creates per tile and the reduction's real cost, and
the DB count had omitted the two off-loop `reductionLaunch` blocks.

## Wiring

The private-block spine is unchanged — `hpcg` → `mg` → `haloExchange` →
{`pack`,`unpack`} → `smooth`/`spmv` → back, one `OCR_EVENT_ONCE_T` per hop,
the block always **RW** and held by one EDT at a time. The exchange is where
the variants diverge. `haloExchangeEdt` flips `toggle`, creates `packEdt` with
all 26 `haloDBK[i][toggle]` blocks **RW** and the private block **RO**, creates
`unpackEdt` with the 26 `haloRecvEVT[i][toggle]` channel events **RO** and the
private block **RW**, then releases the private block and lets both run.
`packEdt` gathers each face/edge/corner into its resident buffer, releases it
and satisfies the neighbour's channel event; the neighbour's `unpackEdt` reads
it RO, releases it — it does **not** destroy it, the producer will reuse it —
and copies into the halo tail of the vector.

A halo buffer is therefore a genuinely **long-lived shared block**: written RW
by its owner's `pack` on the local node, read RO by exactly one remote
consumer, every other exchange, for the whole run — the opposite of
`hpcg_intel`'s one-shot create/read/destroy. Its home is the producing tile's
node (the `OCR_HINT_DB_EAGER` hint carries no affinity, so creator-home
applies). Note that `pack` (private block RO) and `unpack` (private block RW)
are wired to the same block in the same exchange; they touch disjoint regions
(interior vs the halo tail) and `unpack` also waits on 26 remote events, so
`pack` runs first in practice, but nothing orders them.

Reductions build a binary tree over `2N` labeled stickies (indices `r` and
`N+r`), again with the receiver creating the channel events. Steady state moves
8 bytes per link through resident, toggled buffers; only the per-launch 8-byte
`mydata` block is fresh, and it becomes that tile's result block.

## Flow

Identical phase structure to `hpcg_intel`: `hpcg` p0 seed → p1 convergence →
4-level MG V-cycle (`mgStep` 0..6, `level = 3−|mgStep−3|`; 17 `mgEdt`, 10
exchanges, 7 smoothers, 3 SpMVs) → p2 `rtz` → p3 halo + SpMV → p4 `pAp` → p5
update → p1. Eleven exchange barriers and three reduction barriers per
iteration; parallel width is `N` tiles, each a serial chain.

The one addition is that each exchange now offers **two** runnable EDTs per
tile (`pack` and, once its 26 arrivals land, `unpack`) instead of one, so
per-tile EDTs per iteration rise 55 → 66 while the critical path is unchanged.
The binary reduction tree makes each global sum `2⌈log₂N⌉` hops deep instead of
`2⌈log₁₀N⌉` — at `N=48` that is 12 hops against 4, a deliberate trade of
latency for the absence of per-message allocation. The coarse MG levels are
still full-width and still pay 26 messages for `1/512` of the level-0 work.
Serial points: `mainEdt`'s spawn loop and `wrapUpEdt`.

## Placement (as-born)

As published and identical to `hpcg_intel`; there is no
`OCR_APP_OPTIMIZED_PLACEMENT` layer in this source. `mainEdt` reads
`ocrAffinityCount(AFFINITY_PD, …)` — the ARTS node count — and maps tile
`myrank` through `getMyPD`, a recursive bisection halving the PD range and the
tile grid's longest axis together, giving each node a contiguous 3-D
sub-block. (`CONSECUTIVE_PLACEMENT` would substitute a linear
`myrank/(N/PDcount)` split; it is not defined here.) From `initEdt` on,
`pbPTR->myEdtAffinityHNT` carries `ocrAffinityGetCurrent()` into every
`ocrEdtCreate`, so the tile's whole chain — `pack`, `unpack` and its reduction
EDTs included — is pinned to one node, and every DB is created there with a
NULL affinity so homes follow.

Reuse changes what crosses the wire. In `hpcg_intel` a cross-node exchange is a
fresh block created remotely, acquired once, destroyed. Here the same block is
re-acquired RO by the same remote reader every other exchange for the whole
run — a stable, repeating producer→consumer pair per link, the pattern a
directory or a retained grant can amortise and a purge-on-release policy
cannot. Cross-node *volume* is unchanged (≈0.97 MiB per tile per iteration at
`m=64`); what changes is that it rides `2L` durable blocks instead of `11L`
ephemeral ones per iteration. Global objects from `mainEdt` live on rank 0.

## Sizing

The dials behave as in `hpcg_intel`: `N = npx·npy·npz` sets parallel width (a
tile is a serial chain, so `N ≥ 2×nodes×workers`), `m` sets grain and the
`≈427·m³` bytes per tile that dominate memory, `maxIter` sets wall time
linearly. Two variant-specific notes. Halo buffers are sized at level 0 and
reused at every coarser level, so raising `m` raises resident buffer memory as
`m²` even though coarse levels use a fraction of it. And the binary reduction
tree means the three per-iteration barriers grow as `log₂N`, so pushing `N` up
for width has a visible latency cost here that it does not have in
`hpcg_intel`.

The catalog's `4 4 3 64 15` gives `N=48`, `m=64`, 5.0 GiB resident, 15
iterations. At 1 node × 15 workers that is 3.2 tiles per worker — a reasonable
steal pool; at 8 nodes × 120 workers it is 0.4, so **the machine is
under-subscribed past about 3 nodes** and multinode points from this row read
as a communication measurement rather than a scaling curve. If width matters,
`4 4 4` (`N=64`) or `8 4 4` (`N=128`, matching the other two hpcg rows) at the
same `m` is the change to make; the extra iterations (15 against 10) are what
buy the 1-node run its wall time.
