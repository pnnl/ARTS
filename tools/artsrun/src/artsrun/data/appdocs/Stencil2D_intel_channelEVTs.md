# Stencil2D_intel_channelEVTs

*The same 2-D star-stencil SPMD kernel as `Stencil2D_intel_chandra`, ported
onto persistent CHANNEL events for halo exchange instead of a per-round
sticky-event handshake — this variant exists specifically to exercise that
event idiom.*
Source: `third_party/ocr-apps/apps/Stencil2D/refactored/ocr/intel-channelEVTs/stencil_2d.c`
(~1540 lines, built with `EXTRA_DEFINES STENCIL_WITH_DBUF_CHRECV
CHANNEL_EVENTS_AT_RECEIVER`) + `timers.c`.

## Overview

Same physical kernel as `Stencil2D_intel_chandra` (radius-2, 9-point
discrete-divergence stencil over an `NP×NP` domain tiled `NR_X×NR_Y`,
`NT+1` rounds, final-round `ADD`/`MAX` reduction trees, rank-0 correctness
check against the analytic `(NT+1)·2`), but restructured around OCR's
`OCR_EVENT_CHANNEL_T`: each of a tile's 4 neighbor directions gets **one
persistent channel event established once at setup**, not a pair of sticky
events recreated every round. `CHANNEL_EVENTS_AT_RECEIVER` picks which side
of a direction creates that channel (here: the receiver creates it and
publishes its GUID to the sender over a one-time labeled-sticky-event
handshake — `#else` would have the sender create+publish instead);
`STENCIL_WITH_DBUF_CHRECV` doubles the channel count per direction (4→8) so
consecutive rounds' sends don't serialize behind the previous round's still
-draining channel — a double-buffering of the *channel event itself*,
independent of the halo-buffer double-buffering (`LsendBufs[2]` etc., which
this port also keeps). The catalog's marker line is `Computed L1 norm =
…`, printed by rank 0's `FNC_summary` alongside `Solution validates` on a
match. Same stress profile as the chandra variant (per-tile task/event
churn + halo DB traffic, independent of per-tile compute grain), but with
the per-round event-object churn moved out of the steady state and into a
one-time setup cost.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `NP` | side of the square domain | 1000 | ✓ parsed in `init_settings` (called from `mainEdt`), propagated via `globalParamH_t` (`DB_MODE_RO` downstream) — multinode-safe |
| `argv[2]` = `NR` | tile count, factored `NR_X×NR_Y` via `splitDimension_Cart2D` | 16 | ✓ same DB propagation |
| `argv[3]` = `NT` | timesteps (`NT+1` rounds run, round 0 untimed) | 10 | ✓ same DB propagation |
| all three, or none | `argc==4` required for any override; 1–2 args silently ignored | — | ⚠ no partial-override path |
| `HALO_RADIUS` | stencil radius | 2 | ✗ compile-time `#define` |
| `CHANNEL_EVENTS_AT_RECEIVER` | which side of a halo direction owns the channel event | on (this target) | ✗ build-time target selection, not a runtime knob |
| `STENCIL_WITH_DBUF_CHRECV` | double-buffer the channel events themselves (4→8 per tile) | on (this target) | ✗ build-time target selection |
| `USE_STATIC_SCHEDULER` | two-level PD-then-tile SPMD fork instead of the direct per-tile fork | off | ✗ not defined for this build — `forkSpmdEdts_Cart2D` (direct fork, below) is what actually runs |
| `ARITY` | fan-out of the (now 3) reduction trees | 10 | ✗ compile-time in `reduction.h` |

## Structure

Setup is only 2 fixed EDTs (`mainEdt`, a `shutdownEdt` gated on a
cluster-wide join reduction — there is no separate `globalInit`/
`globalCompute` chain; the SPMD fork happens directly from `mainEdt` via
`forkSpmdEdts_Cart2D`), then 4 one-time EDTs per tile
(`initEdt`→`channelSetupEdt`→`FNC_stencil`→`FNC_initialize`), then `NT+1`
rounds of 11 EDTs per tile (`timestepLoopEdt`, `timestepEdt` [FINISH],
`Lsend`/`Rsend`/`Lrecv`/`Rrecv`/`Bsend`/`Tsend`/`Brecv`/`Trecv`, `update`),
plus one `FNC_summary` per tile at its final round.

This port runs *three* reduction trees per tile through the shared
`reduction.c` library, not two, and the norm/timer pair use `ALLREDUCE`
(every tile needs the answer, unlike chandra's root-only `REDUCE`) while
the `spmdJoin` shutdown barrier stays plain `REDUCE`. With
`C(NR) = ⌊(NR-2)/10⌋ + 1` the count of tiles with at least one `ARITY=10`-ary
child (`NR≤11` ⇒ `C=1`), a single `ALLREDUCE` tree costs `4·NR + 2·C(NR) - 2`
EDTs and `4·NR - 3` DBs (the extra EDT per non-root tile over `REDUCE` is
`reductionRecvDown`'s clone, waiting on the broadcast; the extra DBs are
`reductionSendDown`'s one-per-child payload, sent by every tile that has
children, root included); a `REDUCE` tree costs `3·NR + 2·C(NR) - 1` EDTs
and `3·NR - 2` DBs (same as the chandra ports' two trees). Both tree types
cost the same `5·NR + C(NR) - 5` events. Two `ALLREDUCE` + one `REDUCE`
together: `11·NR + 6·C(NR) - 5` EDTs, `11·NR - 8` DBs, `15·NR + 3·C(NR) - 15`
events.

| object | count | notes |
|--------|-------|-------|
| EDTs | `-3 + 16·NR + 11·NR·(NT+1) + 6·C(NR)` | folds the three reduction trees into the fixed 2-EDT chain (incl. `mainEdt`) + `5·NR` one-time per-tile setup (4 setup EDTs + 1 `FNC_summary`) |
| DBs | `32·NR - 7` | `1+21·NR` (global + per-tile control-block/payload/transient DBs, NT-independent) plus the three reduction trees' `11·NR-8` |
| DB payload | same as `Stencil2D_intel_chandra`: `xIn` `8·(np_x+4)·(np_y+4)` B, `xOut` `8·np_x·np_y` B | `rankH_t` itself is a few hundred bytes (control-block only, no bulk payload) |
| Events | `10 + 31·NR + 11·NR·(NT+1) + 3·C(NR)` | `11` fresh events per tile per round (not 9 — see below) plus `16·NR` one-time per-tile handshake events at setup, plus the three reduction trees' `15·NR+3·C(NR)-15` |

Each round's 11 events are the `OCR_EVENT_COUNTED_T` helper that
`createEventHelper` attaches to each of the 8 halo EDTs (`Lsend`/`Rsend`/
`Bsend`/`Tsend`/`Lrecv`/`Rrecv`/`Brecv`/`Trecv`, `EDT_PROP_OEVT_VALID` so no
*extra* event is materialized beyond the one `createEventHelper` already
made) plus `timestepLoopEdt`'s own loop-continuation `COUNTED` join, plus
`timestepEdt`'s own creation: `EDT_PROP_FINISH` with a *freshly materialized*
(non-`OEVT_VALID`) output event, so it costs 2 more (output + finish) on top
of the explicit `COUNTED` already counted — 8+1+2 = 11, not the 9 the
original count found by only tallying `createEventHelper` calls and missing
`timestepEdt`'s own FINISH/output pair. The `16·NR` one-time setup figure
(8 labeled sticky "own publish" + 8 more labeled sticky "reverse
pre-declare", one pair per tile per of 4 directions — the same
double-creation pattern as the reduction tree's labeled events, see
Uncertain in the notes file — plus 8 `OCR_EVENT_CHANNEL_T` from
`STENCIL_WITH_DBUF_CHRECV`) matches the original doc's "8 labeled + 8
channel = 16" claim; only the *count of labeled creates* needed
correcting, not their sum.

Calibrated args `['41472', '13824', '400']`: 108×128 tile grid,
`np_x=384,np_y=324`, ≈1.9 MB/tile payload, ≈27.5 GB total; `C(13824)=1383`;
EDTs ≈ `-3+221,184+60,977,664+8,298` = **61.2 M**; DBs ≈ `32·13824-7` =
**442,361**; Events ≈ `10+428,544+60,977,664+4,149` = **61.4 M**.

Counter cross-check: verified (1 node, `NP=64 NR=4 NT=2` vs `NP=64 NR=4
NT=4`): predicted absolutes 200/122/269 and 288/122/357 (`NUM_EDT_CREATE` /
`NUM_DB_CREATE` / `NUM_EVENT_CREATE`) match the measured counters exactly,
against a runtime baseline of `+1 EDT, +1 DB, +0 EVT`. As in both chandra
ports, the fixed control-chain/payload-DB formulas
(`2+5·NR+11·NR·(NT+1)` / `1+21·NR`) were already exactly right — the gap
was the un-derived three reduction trees (`+45` EDT / `+36` DB at `NR=4`)
plus an event slope of 11 instead of the previously-claimed 9 per
tile-round. A later `NR=16` re-check (`NP=2048 NT=100/400`) confirms the
EDT/DB formulas and the per-tile-round slopes exactly (11 events created,
10 destroyed — the surviving one is the `timestepEdt` FINISH/output pair's
runtime-minted half), but finds the event TOTAL under-predicted by a flat,
NT-independent 72: the setup/reduction terms vary with the grid's actual
corner/edge/interior neighbor mix, which the `NR=4` (2×2, all-corner)
verification point could not expose. The formula is kept with that caveat
rather than re-fit.

## Wiring

`rankH_t` is one consolidated per-tile control-block DB carrying the
command-line-derived params, the 11 EDT template GUIDs, both
affinity-hint structs, and the 8/16 halo-channel-event GUIDs, alongside
GUID handles to the payload DBs — where chandra spreads that same
information across 9 separate small handle DBs, this port keeps it as one
DB passed `DB_MODE_CONST` into almost every EDT (resolves the same as RO
under the shim). Halo data flow per direction: `Lsend`/`Rsend`/`Bsend`/
`Tsend` copy `xIn` (RO) into their phase's send buffer (RW), then
`ocrAddDependence` that buffer's guid directly into the neighbor-owned
persistent CHANNEL event (`haloSendEVTs[GET_CHANNEL_IDX(face,phase)]`) —
no separate "recv-ready" sticky leg, the channel itself provides the
producer/consumer rendezvous and FIFO ordering across generations.
`Lrecv`/`Rrecv`/`Brecv`/`Trecv` depend RO on their own `haloRecvEVTs` slot
and write into `xIn` RW at the matching halo region. Every send/recv EDT is
created with `EDT_PROP_OEVT_VALID` against a pre-made `OCR_EVENT_COUNTED_T`
helper (`createEventHelper`), so the same event GUID both signals `update`
that this leg finished (the join) and — on the send side — is itself the
producer end of the neighbor's channel. As with the chandra port, a halo
region has exactly one producer and one consumer per round, and no DB ever
takes concurrent RW from two tiles; the three reduction trees (norm ADD,
timer MAX, and the `spmdJoin` REDUCE-typed barrier every tile calls before
shutdown) are the only cross-tile fan-in, again mediated by `reduction.c`'s
own channel/sticky machinery outside the app's own graph.

## Flow

`mainEdt`'s only work is command-line parsing, allocating the three
`ocrGuidRangeCreate` label spaces plus the join event, and one call to
`forkSpmdEdts_Cart2D`, which itself issues `NR` hinted `ocrEdtCreate` calls
for `initEdt` in a single loop (no separate spawner EDT, unlike chandra) —
parallel width reaches `NR` immediately once that loop completes. Each
tile's `initEdt`→`channelSetupEdt`→`FNC_stencil` sequence is a 3-deep serial
per-tile setup (the one-time channel-event handshake happens inside this
window), then `FNC_initialize` runs in parallel with the *first*
`timestepLoopEdt`'s early setup (both depend only on `channelSetupEdt`'s
output, not on each other) before the timestep recursion actually touches
data. From there, compute is `NR` independent per-tile chains of `NT+1`
serial rounds (`timestepLoopEdt` re-creates itself on its own `FINISH`
child's join event), each round's 8-EDT halo exchange internally parallel.
Parallel width is `NR` throughout compute and does not grow with `NT`. Every
tile's final-round `update` triggers its own `FNC_summary`, which destroys
all of that tile's now-dead per-tile objects (`destroyOcrObjects`) and joins
the `spmdJoin` reduction; only after all `NR` tiles have joined does the
dedicated `shutdownEdt` (created once, from `mainEdt`, gated on
`EVT_OUT_spmdJoin_reduction`) call `ocrShutdown()`.

## Placement (base)

Not a NULL-hint program, and not gated by any hint-layer guard (this
source carries no `OCR_APP_OPTIMIZED_PLACEMENT` code at all — there is no
`_hinted` build). `forkSpmdEdts_Cart2D` (`ocrAppUtils.c`, shared with the
chandra port's hand-rolled equivalent) queries
`ocrAffinityCount(AFFINITY_PD, …)` — the ARTS run's actual node count,
independent of the app's own `NR` — factors it into a `PD_X×PD_Y` grid via
`splitDimension_Cart2D`, and maps each tile's `(id_x,id_y)` onto a PD-block
coordinate via `getPartitionID` per axis (the identical block-partition
algorithm chandra uses), hinting each `initEdt` (and, through it, every DB
and EDT that tile ever creates) onto that PD. Net effect is the same 2-D
contiguous tile-block-per-rank locality as `Stencil2D_intel_chandra`:
neighbor tiles in both directions are usually co-resident, and cross-rank
halo channel traffic is limited to the tiles straddling a block edge. The
one-time channel-event handshake (`initEdt`'s labeled-sticky exchange) is
the only place this port does cross-rank traffic that chandra's per-round
sticky pair does not — but it happens once per run, not once per round, so
it does not change the steady-state locality story.

## Family shape (measured, 15w+1p x 1/2/4/8 nodes, `6144 768 200`)

base (there is no `_hinted` — the port's own affinity layer IS the
placement), e2e seconds, counters all off:

| arm | 1n | 2n | 4n | 8n |
|---|---|---|---|---|
| val_wb_nocomb | 7.33 | 3.94 | 2.05 | 1.14 |
| val_wb | 7.29 | 3.90 | 1.98 | 1.13 |
| inv_wb | 7.33 | 3.97 | 2.06 | 1.27 |
| excl_retain | 7.31 | 3.97 | 2.03 | 1.24 |

Every arm strong-scales near-ideally (6.4x at 8 nodes) and the arms are
indistinguishable: the block placement confines halo traffic to tile-block
edges, each halo buffer has one producer and one consumer per round, and no
hot globally-shared RO DB exists — so VAL has nothing to re-validate en
masse (combining changes nothing) and INV/EXCL pay only boundary-edge
rounds. This is the coherence-friendly pole of the application set, the
structural opposite of the one-hot-DB fan-in programs.

## Sizing

The CLI is `NP NR NT`, all three or none (`argc==4`). `NR` fixes the SPMD
width and must factor near-square through `splitDimension_Cart2D` — a prime
`NR` degenerates to a 1×NR strip. The campaign pins `NR = 13824 = 108·128`:
it spans 32 nodes × 108 workers at 4 tiles per worker, and the calibrated
`NP` divides both axes exactly. `NP` is the size dial (per-tile arrays scale
as `NP²/NR`); `NT` is the real timestep count — rounds are usage, not
repetition padding.

Measured on the Dane-mirror geometry (1 node, 108w+4p, Release, val_wb_nocomb,
counters all off, `NR=13824`):

| NP | NT | e2e |
|---|---|---|
| 13824 | 100 | 5.2 s* |
| 27648 | 100 | 16.8 s* |
| 13824 | 400 | 18.6 s* |
| 27648 | 400 | 69.1 s |
| 41472 | 400 | **144.4 s** |
| 55296 | 200 | 131.6 s |

(*) measured with an object-counter set still compiled in; a clean-tree
re-measure of `27648 400` moved 67.2→69.1 s, so that instrumentation is
within run-to-run noise and the starred points stand for shape. e2e is
linear in `NT` and tracks `NP²`, so the lattice extrapolates cleanly. The
calibrated arguments are the feasible point nearest the ~150 s anchor:
`41472 13824 400` (np `384×324` per tile, ≈27.5 GB total payload);
`55296 13824 200` stands as the alternate if a larger domain at fewer
rounds is ever preferred. Peak object load is bounded by the frontier, not
the run: the only per-round survivor is one runtime-minted FINISH/output
event per tile-round (measured slope exactly 1.0), ≈5.5 M events ≈
single-digit GB at the calibrated size — no Dane budget concern, uniform
across coherence arms, so the base source is left untouched.
