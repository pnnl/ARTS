# hpcg_intel_Eager_Collective

*The Eager HPCG with its hand-built reduction tree deleted: the three global
sums per iteration are handed to the runtime as an OCR collective event
("Redevt" = reduction event), leaving the app with only the halo exchange.*
Source: `third_party/ocr-apps/apps/hpcg/refactored/ocr/intel-Eager-Collective/hpcgEagerRedevt.c`
(~2560 lines) + `timers.c`. No reduction library is linked; built with
`ENABLE_EXTENSION_COLLECTIVE_EVT` + `ENABLE_EXTENSION_MULTI_OUTPUT_SLOT`.

## Overview

Same solver, same scalar (`final deviation: …`), same argument surface as
`hpcg_intel`. "Eager" and "Collective" both name **application-side algorithm
choices**, never a runtime coherence mode. This variant is
`hpcg_intel_Eager` — preallocated double-buffered halo blocks, a separate
`packEdt`, receiver-owned channel events — plus one further change:

- **The reduction library is gone.** `reductionPrivate_t` shrinks to a single
  `ocrGuid_t returnEVT`. `initEdt` instead creates two *labeled*
  `OCR_EVENT_COLLECTIVE_T` events (`redEvtGuid` for the CG dot products,
  `redEvtTimerGuid` for the final residual) with `nbContribs = N`,
  `op = REDOP_F8_ADD`, `type = COL_ALLREDUCE`.
- **A dot product becomes two calls**: `ocrAddDependenceSlot(redEvt, myrank,
  returnEVT, 0, RO)` registers where this tile's result should land, then
  `ocrEventCollectiveSatisfySlot(redEvt, myDataPTR, myrank)` contributes the
  scalar. No tree, no rendezvous, no buffers on the app side.
- **The final reduce is an allreduce too** — the source notes `COL_REDUCE` is
  "not supported", so rank 0 wires the result into `finalOnceEVT` and every
  other tile wires it into a throwaway ONCE event.
- **Placement was changed as well**, and not for the better: `getMyPD`'s
  recursive bisection is commented out in favour of a linear block split of
  consecutive tile ids (see Placement).

On ARTS the collective event is implemented by the OCR shim, which builds a
**binary** reduce/broadcast tree per generation over deterministic labeled
edge events. The runtime is therefore not doing anything the app could not:
the variant relocates the tree from application code into shim code, and its
interest is that the shim's tree is object-churny where `reductionEager`'s is
buffer-free. The catalog gives this row no `expect`/`expect_args`, so its
correctness rides entirely on cross-configuration consensus.

## Parameters

Identical parser to the other two (0, 3, 4, 5 or 6 user arguments), reading
through `getArgc`/`getArgv` rather than the `ocr…` wrappers.

| arg | meaning | default | CLI reachability |
|-----|---------|---------|------------------|
| `argv[1..3]` = `npx,npy,npz` | tile grid; `N = npx·npy·npz` | 3,4,5 | ✓ `mainEdt` → shared DB → every tile — multinode-safe |
| `argv[4]` = `m` | tile cube edge, rounded up to a multiple of 16 | 16 | ✓ same path |
| `argv[5]` = `maxIter` | CG iterations, capped at 50 | 50 | ✓ same path |
| `argv[6]` = `debug` | print level 0/1/2 | 0 | ✓ same path |
| 1 or 2 arguments | `bomb()` prints + `ocrShutdown()`, **then returns**; run continues on defaults | — | ⚠ diagnosed, not fatal |
| tile→node map | block distribution of the `N` tile ids over `PDcount` nodes, the first `N mod PDcount` nodes taking one tile more | — | ✓ total for any `(N, PDcount)`, including `N < PDcount` |
| `PRINT_DIAGNOSE`, `RANK_TIMER`; `PRECONDITIONER`, `COMPUTE`, `AFFINITY`, `TIMER` | per-iteration `rtr`/`rtz` lines and per-tile elapsed times; as `hpcg_intel` | undefined; on,on,on,off | ✗ compile-time — this build is quiet until `final deviation` |
| collective params | `arity = 2`, `maxGen = 1`, `nbContribsPd = N/PDcount`, `reuseDbPerGen = true` | — | ✗ in-source; the ARTS shim honours only `nbContribs`, `nbDatum`, `op`, `type` |

## Structure

With `L = (3·npx−2)(3·npy−2)(3·npz−2) − N` directed neighbour links, and the
per-generation collective cost taken from the shim's binary tree over `N`
contributors:

| object | count | size |
|--------|-------|------|
| private block | `N` | `≈ 427·m³` B — same 4-level matrix/index/vector layout as the other two |
| halo buffers | `2L`, **created once** | level-0 sized; 8 B corners, `8m` B edges, `8m²` B faces |
| collective metadata | 2 per **node** | `ARTS_DB_PIN`, `sizeof(CollectiveMetadata) + N·sizeof(CollectiveContribState)` |
| shared / reduction-private / scalar blocks | 1 / `N` / `N` | 56 B / 8 B / 8 B |
| EDT creates | `66N` app + `6N` shim = `72N` per iteration | app: 5 `hpcg` + 17 `mg` + 11 `haloExchange` + 11 `pack` + 11 `unpack` + 7 `smooth` + 4 `spmv` per tile; shim: one up-reducer and one down-forwarder per contributor per generation |
| Event creates | `37N` app + `3(4N−2)` shim per iteration | app: the ONCE chain events; shim: each tree edge is `ensure`d from both endpoints, so `4N−2` creates install `2N−1` distinct edge events |
| DB creates | `3(4N−1)` per iteration | **entirely shim-side** — own-datum, partial, forward and delivery blocks of 8 B each; the app creates none in steady state |
| one-time | `8N + 2` EDTs, `11N + 3L + 2·nodes − 1` DBs, `10N + 4L − 4` events | `mainEdt` and `wrapUpEdt`, then four EDTs per tile: `initEdt`, `channelInitEdt`, the phase-0 `hpcgEdt` and the phase-1 clone that one creates. Blocks: the shared block, a private, a reduction-private and an 8-byte scalar block per tile, per valid direction two halo buffers and a GUID carrier, and one collective-metadata block per collective event per node. Events: a `returnEVT` channel per tile, per direction two CHANNEL events and a labeled sticky created from *both* ends, and `N−1` throwaway ONCE events at termination |

Two collective generations fall outside the iteration loop — the phase-0 `rtr`
and the final residual — and each costs the shim's usual `2N` EDTs, `4N−1` DBs
and `4N−2` edge-event `ensure`s. The two `OCR_EVENT_COLLECTIVE_T` creates
themselves make no event: the shim backs a collective with a node-local
`ARTS_DB_PIN` metadata block and the first create on a node wins, so `N` tiles
sharing a node yield one block, not `N`.

Calibrated args `['8','4','4','64','10']`: `N=128`, `m=64`, `L=2072`. Private
blocks **≈107 MiB × 128 ≈ 13.7 GiB**; halo buffers ≈396 KiB per interior tile.
Setup is 1,026 EDTs / 7,625 DBs / 9,564 events at one node. Per iteration
cluster-wide: **9,216 EDT creates, 1,533 DB creates, 6,266 event creates**;
over 10 iterations ≈93k EDTs, ≈23k DBs, ≈72k events. Against `hpcg_intel` at
the same args that is ~16× fewer DB creates per iteration (the halo path is
free) but ~1.2× more EDTs and ~1.3× more events.

Counter cross-check: verified (1 node, `N=8`, `L=56`). `2 2 2 16 2` and
`2 2 2 16 5` measure 1,219/2,947 EDTs, 444/723 DBs, 1,072/2,230 events — the
per-iteration formulas give those deltas exactly (576/93/386 per iteration,
which pins the shim's `2N` / `4N−1` / `4N−2` per generation), and the one-time
terms give the absolutes (66/257/300 here) once the runtime's constant +1 EDT
and +1 DB per run are added. Two metadata blocks at one node also settles that
the collective registry is per node rather than per tile. All three setup
counts had to be corrected, each having omitted the two off-loop generations;
the EDT count additionally omitted `wrapUpEdt` and both `hpcgEdt` creates per
tile, and the DB count the per-tile 8-byte scalar block.

## Wiring

The halo half is `hpcg_intel_Eager`'s, unchanged: `haloExchangeEdt` flips
`toggle`, wires the 26 resident `haloDBK[i][toggle]` blocks **RW** into
`packEdt` (private block **RO**) and the 26 `haloRecvEVT[i][toggle]` channel
events **RO** into `unpackEdt` (private block **RW**), then releases the
private block. Each halo buffer is a durable shared block: written by its
owner's `pack`, read RO by exactly one remote `unpack`, alternating between two
copies so generation `k+1`'s write cannot overtake generation `k`'s read. The
private block remains a strictly serial single-writer spine, ~107 MiB, one
holder at a time, never leaving its node.

The reduction half is entirely different. A tile's `returnEVT` is a CHANNEL
event created once in `initEdt` and reused for every generation; each round the
tile calls `ocrAddDependenceSlot` (recorded as the pending dependent for its
contributor slot — no object created) then `ocrEventCollectiveSatisfySlot`. The
shim then, on the contributor's own node, creates an **up reducer** (slot 0 = a
fresh 8-byte copy of the datum, slots 1.. = that contributor's children's
up-edges, RO) and a **down forwarder** (slot 0 = its own down-edge, RO). Edges
are labeled STICKY events at GUIDs derived from `(collective, nrank,
generation, contributor, direction)` and homed round-robin over the ARTS ranks,
so the reduction's message graph ignores the tile→node map entirely:
contributor `r`'s parent is `⌊(r−1)/2⌋` by index, wherever that tile lives, and
the edge event may be homed on a third node. Each edge is `ensure`d by both
endpoints (install-if-absent), which is why the event-create count is about
twice the number of distinct edges. Every DB here is 8 bytes, one writer, one
reader.

## Flow

Phase structure is `hpcg_intel`'s: `hpcg` p0 seed → p1 convergence → 4-level MG
V-cycle (`mgStep` 0..6, `level = 3−|mgStep−3|`; 17 `mgEdt`, 10 exchanges, 7
smoothers, 3 SpMVs) → p2 `rtz` → p3 halo + SpMV → p4 `pAp` → p5 update → p1.
Eleven exchange barriers and three reduction barriers per iteration; parallel
width is `N` tiles, each exchange offering two runnable EDTs per tile (`pack`,
then `unpack` once its 26 arrivals land).

Around each collective the width briefly grows by `2N` short shim EDTs whose
dependence graph is a binary tree of depth `⌈log₂N⌉` in each direction — 14
hops at `N=128`, against 4 for `hpcg_intel`'s 10-ary tree. Those EDTs are
microscopic (an 8-byte add and a satisfy), so the collective is a pure latency
structure: three per iteration, each a global barrier. The coarse MG levels
remain full-width, paying 26 messages for `1/512` of the level-0 work. Serial
points: `mainEdt`'s spawn loop and `wrapUpEdt`.

## Placement (as-born)

No `OCR_APP_OPTIMIZED_PLACEMENT` layer exists here either; what follows is as
published. `mainEdt` reads `ocrAffinityCount(AFFINITY_PD, …)` (the ARTS node
count) and maps tiles with

```
nrankPD = N / PDcount;  extraPD = N % PDcount;   // getMyPD(...) is commented out
myPD    = block index of myrank, first extraPD nodes taking nrankPD+1 tiles
```

— a **linear** split into runs of consecutive tile ids, where the other two
variants use `getMyPD`'s recursive bisection. Since `myrank = x + npx·y +
npx·npy·z`, consecutive ids are `x`-major slabs, so a node's block is long and
thin: at the calibrated `8 4 4` on 8 nodes each node owns 16 consecutive ids =
an `8×2×1` slab (22 of 48 face adjacencies internal), where bisection would
give a `2×2×4` cube (28 of 48) — measurably more cross-node halo surface, and
the gap widens with node count. `initEdt` onward is the usual discipline:
`pbPTR->myEdtAffinityHNT` carries `ocrAffinityGetCurrent()` into every
`ocrEdtCreate`, DBs are NULL-hinted so homes follow the creator, and each
tile's chain and its ~107 MiB private block stay on one node.

The collective adds a second, unrelated traffic pattern: its tree is over
contributor *indices*, its EDTs run on the contributing tile's node, and its
edge events are homed `(coll_rank + r + direction) mod nodes`. Nothing ties
that layout to the tile→node map, so every non-local tree edge is a cross-node
8-byte hop chosen independently of the domain decomposition —
`hpcg_intel`'s 10-ary tree has the same property but a third of the depth.
Global objects from `mainEdt` live on rank 0; the two collective metadata
blocks per node are `ARTS_DB_PIN`, created locally.

## Sizing

`N = npx·npy·npz` sets width, `m` sets grain and the `≈427·m³` bytes per tile
that dominate memory, `maxIter` sets wall time linearly — as in the other two
variants, and a tile is still a serial chain, so `N ≥ 2×nodes×workers`. One
constraint is specific to this row: the collective's barriers grow as `log₂N`,
so extra width costs more reduction latency here than in `hpcg_intel`.  An `N`
divisible by the node count is still the tidier choice — the block map then
gives every node the same number of tiles — but an indivisible one only skews
the load, it does not misplace anything.

The catalog's `8 4 4 64 10` matches `hpcg_intel` exactly — the point of the row
is to compare reduction implementations at a fixed problem — giving `N=128`,
`m=64`, 13.7 GiB resident, and 128 divisible by 1/2/4/8 so the whole node sweep
is legal. At 1 node × 15 workers that is 8.5 tiles per worker; at 8 nodes × 120
workers, 1.07. For a quick functional check use `2 2 2 16 5` (`N=8`, ≈14 MiB
total, seconds); `1 1 1 16 5`, the other variants' pinned argument set, also
runs here at any node count and reproduces their `0.001279`.
