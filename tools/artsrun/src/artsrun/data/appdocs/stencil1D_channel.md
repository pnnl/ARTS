# stencil1D_channel

*1-D three-point stencil, cloning style, halo exchange over persistent
CHANNEL events, with explicit block-affinity chain placement.*
Source: `third_party/ocr-apps/apps/Stencil1D/refactored/ocr/intel-david/stencil1Dchannel.c`
(~550 lines; requires the `ENABLE_EXTENSION_CHANNEL_EVT` OCR extension, which
this build always enables — see Parameters).

## Overview

Same 1-D three-point elliptic relaxation as `stencil1D_sticky`
(`anew(i) = 0.5*a(i) + 0.25*(a(i-1)+a(i+1))`) and the same
`N`-chain/`M`-point/`T`-timestep decomposition; its per-timestep compute
code is in fact verbatim identical to the lineage-A trio
(`stencil1D_sticky`/`guid`/`once`), including the same variable names and
the same private-DB-with-embedded-data layout (`dataOffset`). Two things
set this variant apart from all six others. First, the halo handoff uses a
**CHANNEL event created once per internal boundary-direction and reused for
the entire run** — every other STICKY/ONCE variant creates a fresh event
object every iteration; this one creates it exactly once and satisfies /
consumes it `T` times, making its synchronization-object churn independent
of `T` altogether (like `stencil1D_guidPI`, but via a different mechanism —
persistent channels instead of no-events-at-all). Second, and unique among
all seven, it explicitly **places** its `N` chains via
`ocrAffinityGetAt(AFFINITY_PD, ...)`/`OCR_HINT_EDT_AFFINITY` in a
contiguous block partition across the actual run's rank count, and pins
every subsequent clone to stay on that same rank via self-affinity
(`ocrAffinityGetCurrent`) — the only one of the seven Stencil1D variants
that expresses real locality in its base placement (see Placement).
Termination combines a clean `EDT_PROP_FINISH`/output-event join (`realMain`
launched as a finish EDT, `wrapupEdt` depending on its output event) with a
left-to-right serialized print relay riding the same per-boundary channels,
much like the lineage-A trio's serialized shutdown wave.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `nrank` | number of independent chains | 4 | ✓ parsed in `mainEdt`, only when exactly 3 args given |
| `argv[2]` = `npoints` | points per chain | 10 | ✓ same |
| `argv[3]` = `maxt` | timesteps | 100 | ✓ same |

Local literal defaults (`4 10 100`), no `stencil1D.h`, `argc` must be
exactly 4 — same pattern as the lineage-A trio; no zero-argument guard
exists in this file. The whole file is gated behind `#ifdef
ENABLE_EXTENSION_CHANNEL_EVT`; without it, `mainEdt` just prints a notice
and shuts down. This build always defines it (and
`ENABLE_EXTENSION_AFFINITY`/`ENABLE_EXTENSION_LABELING`/etc.) globally for
every app via `OCR_EXT_DEFINES` in `benchmarks/apps/CMakeLists.txt`, so the
real (channel-event) path is what ships. No CLI-exposed knob controls node
placement; the block-affinity bucketing (see Placement) is derived purely
from `nrank` and the run's actual rank count, not from a separate argument.

## Structure

Let `N=nrank`, `M=npoints`, `T=maxt`.

| object | count | size |
|--------|-------|------|
| EDTs total | `N·T + 4N + 3` — `mainEdt`, `realMainEdt` (FINISH), `N` `stencilInitDBsEdt`, `N` `stencilInitEdt`, `N·(T+1)` `stencilEdt` chain generations (single clone per iteration, no grandchild lookahead), `N` `stencilReportEdt` (one per chain, at the final timestep), 1 `wrapupEdt` | — |
| EDT templates | `3N+3` — `realMainTML`/`wrapupTML`/`stencilInitDBsTML` created once centrally; `stencilInitTML`, `stencilTML`, and `reportTML` are each (re-)created once per chain rather than shared | — |
| DBs | `3N-1` — 1 shared + `N` private (data embedded via `dataOffset`, same single-DB layout as lineage A) + `2(N-1)` halo buffer | shared ≈ 40 B (`nrank`/`npoints`/`maxt` + `startDirs[2]`); private ≈ 72 B (`private_t`: 5×`u32` + `dataOffset` + 5×`ocrGuid_t`) + `M×8` B data; buffer = 16 B |
| Events | `6(N-1)+2` `ocrEventCreate`/`ocrEventCreateParams` calls, **all made once at bootstrap, none during the `T`-iteration steady state**: `4(N-1)` labeled STICKY calls (2 endpoints × 2 roles per boundary — each boundary's address is independently "created" by both neighbors under `GUID_PROP_CHECK`, so only `2(N-1)` distinct objects actually result, see Uncertain) used solely to exchange channel-event GUIDs, plus `2(N-1)` CHANNEL events (one per rank-side that has a neighbor), each satisfied/consumed `T` times over the run without being recreated, plus a constant **+2** from `mainEdt`'s single `ocrEdtCreate` of `realMainEdt` as a FINISH EDT with a non-NULL `outputEvent` (`&finishEVT`) — per the runtime's event-accounting rule this materializes *two* separate event objects, the output event itself and the internal FINISH-latch event `EDT_PROP_FINISH` always creates (chained together by the shim), independent of `N`/`T` | — |

Worked numbers at the calibrated args (`48 50 340000`): EDTs ≈
`48×340000 + 4×48 + 3 = 16,320,195`; DBs = `3×48-1 = 143`, ≈24 KB payload;
Events = `6×47+2 = 284` API calls total, for the entire 340,000-iteration
run — the flattest event-churn profile of any variant that uses events at
all (only `stencil1D_guidPI`'s `188` is smaller, and that variant uses no
events past bootstrap either).

Counter cross-check: verified (1 node, `4 4 5` vs `6 4 8`): NUM_EDT_CREATE
40 → 76, NUM_DB_CREATE 12 → 18, NUM_EVENT_CREATE 20 → 32 — exactly
`N·T+4N+3` / `3N-1` / `6(N-1)+2` (app values 39/75, 11/17, 20/32) plus the
runtime's constant +1 EDT/+1 DB/+0 EVT baseline. `EDTs`/`DBs` were
already exact; the `6(N-1)` event formula was short by a constant +2 —
the FINISH-EDT output/latch pair above accounts for it.

## Wiring

- `mainEdt` creates a shared DB and `realMainEdt` **as a finish EDT**
  (`EDT_PROP_FINISH`, output event `finishEVT`), then `wrapupEdt` depending
  on `finishEVT`.
- `realMainEdt` creates `N` `stencilInitDBsEdt` EDTs, each explicitly
  placed at a computed rank (see Placement) and RW-dependent on the shared
  block.
- `stencilInitDBsEdt` creates the private DB (data embedded via
  `dataOffset`) and 0–2 buffer DBs, then launches `stencilInitEdt`.
- `stencilInitEdt` computes, via `ocrGuidFromIndex`, the labeled STICKY
  addresses both neighbors of a boundary will independently attempt to
  create (idempotent, checked); creates its own CHANNEL event(s) for each
  side it has, writes each channel's GUID into the corresponding buffer
  DB's `.EVT` field, and satisfies the matching labeled STICKY event with
  that buffer — this is how a rank's channel GUID reaches its neighbor,
  used exactly once. It then creates the single chain-head `stencilEdt` (no
  grandchild-ahead pipelining — the persistent channel makes it
  unnecessary) wired to the two labeled receive-events for the neighbors'
  channels.
- Each `stencilEdt` generation creates its immediate successor and
  satisfies its own channel event(s) with the outgoing buffer DB — the
  *same* channel object every iteration, never recreated.
- At the final timestep, the chain creates a `stencilReportEdt` instead of
  a further clone; it depends on the chain's private DB (RO) and on the
  chain's own `leftRcvEVT` channel (RO) — an extra, `(T+1)`-th
  satisfy/consume on the same persistent channel used as a one-time
  left-to-right print-order signal. `stencilReportEdt` prints its `M`
  values then, if not the last chain, satisfies its own `rightSendEVT`
  channel to let its right neighbor's report proceed.
- Same DB-concurrency shape as every other variant: no DB has more than
  one simultaneous accessor.

## Flow

`O(N)` rank-0 setup (the `stencilInitDBsEdt` creation loop, each iteration
doing `O(1)` work), then `N` parallel chains each a `T`-deep serial
dependency chain. Finalization has two layers: the `stencilReportEdt`s form
a left-to-right serialized relay (rank 0 first, cascading via the shared
channel infrastructure — mirroring lineage A's shutdown wave), and the
*whole* computation additionally waits on `realMainEdt`'s finish-EDT output
event before `wrapupEdt` runs — so completion is only signaled once every
descendant EDT of every chain (including every report) has actually
finished, not merely once the report relay has been kicked off. Critical
path ≈ `T + N` (steady state plus the report relay's `N`-deep tail).

## Placement (base)

This is the one Stencil1D variant with genuine, non-`NULL_HINT` placement
outside any `OCR_APP_OPTIMIZED_PLACEMENT` guard (that guard does not appear
in this file at all — this is base behavior, always on):

- `realMainEdt` computes `bucket = ceil(N / affinityCount)` where
  `affinityCount = ocrAffinityCount(AFFINITY_PD, …)` = the run's actual
  rank count (`arts_get_total_ranks()` in the shim), and places chain `i`'s
  `stencilInitDBsEdt` at PD `i / bucket` via `OCR_HINT_EDT_AFFINITY` — a
  contiguous block partition of the `N` chains across whatever ranks the
  run actually has.
- Every subsequent EDT in that chain (`stencilInitEdt`, every `stencilEdt`
  clone, `stencilReportEdt`) re-derives its own hint via
  `ocrAffinityGetCurrent` (= "the rank I am currently running on") and
  passes it to its own successor — so once a chain lands on a rank, it
  **stays there** for the rest of the run; no round-robin churn.
- DBs: still `NULL_HINT` (home = creating rank) — but since the *creating*
  EDT (`stencilInitDBsEdt`, itself explicitly placed) is on the target
  rank, every DB in a chain still ends up homed on that chain's assigned
  rank. A `HNT_db` hint object is initialized in three functions
  (`stencilEdt`, `stencilInitEdt`, `stencilInitDBsEdt`) but never actually
  passed to any `ocrDbCreate` call — dead but harmless, since the
  NULL_HINT/creator-homed default already lands in the right place here.

Consequence at multinode: unlike every other Stencil1D variant, most DB
acquires are **local** — a chain's private/buffer DBs live on the same
rank as the chain's own executing EDT for the chain's entire lifetime.
Cross-rank traffic is confined to the boundaries that fall on a bucket
seam (at most `nodes-1` of the `N-1` internal boundaries), not every single
dependence edge. This is a genuine, deliberate locality-aware base
design — the opposite conclusion from the other six rows in this
directory.

## Sizing

`N`, `M`, `T` play the same roles as elsewhere (`N` caps parallel width and
sets the block-partition granularity, `T` drives EDT churn/wall-time, `M`
only moves per-chain payload/compute), but the placement story changes the
node-count guidance: because chains are bucketed contiguously across
whatever rank count the run actually has, growing the node count (1→2→4→8,
strong scaling) automatically redistributes the same `N=48` chains into
finer contiguous blocks (6 chains/node at 8 nodes) rather than leaving
placement to chance — so this row is the one where adding nodes should
plausibly *reduce* per-chain remote-access rate rather than leave it flat,
the opposite of the round-robin siblings. Pick `N` ≳ target worker count
for parallel occupancy (as elsewhere); `T=340000` (the shared calibrated
value) sizes wall time and total EDT churn the same way it does for every
other row in this family.
