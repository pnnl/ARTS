# stencil1D_guid

*1-D three-point stencil, cloning style, halo exchange by passing raw EDT
GUIDs — no OCR events at all.*
Source: `third_party/ocr-apps/apps/Stencil1D/refactored/ocr/intel-david/stencil1Dguid.c`
(~640 lines; author David S. Scott, Intel 2015).

## Overview

Same computation, decomposition and convergence behavior as
`stencil1D_sticky` (see that doc for the shared elliptic-relaxation
formula, the `N`-chain/`M`-point/`T`-timestep decomposition, and the
"cloning" idiom every Stencil1D variant in this directory shares). What
distinguishes this variant is that it uses **no OCR events whatsoever** for
the neighbor halo exchange: instead of creating and satisfying an event, a
chain hands its neighbor the raw `ocrGuid_t` of an EDT it has *already
created*, and the neighbor wires a plain `ocrAddDependence` straight onto
that GUID. This only works because each `stencilEDT` clones its own
**grandchild** (two generations ahead) rather than its immediate successor
— by the time a chain tells its neighbor "here is the GUID to depend on",
that EDT already exists and is already known, so no race between
dependence-add and satisfaction is possible (the reasoning the README gives
for why "guid" and "once" both need this two-ahead trick, and why "sticky"
does not). The result is the cheapest of the three original idioms in pure
event-object terms: zero events, ever. Like `stencil1D_sticky`, there is no
separate wrapup/join EDT — the last chain prints and shuts down directly.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `nrank` | number of independent chains ("workers") | 4 | ✓ parsed in `mainEdt`, only when exactly 3 args are given |
| `argv[2]` = `npoints` | points per chain | 10 | ✓ same |
| `argv[3]` = `maxt` | timesteps | 100 | ✓ same |

Identical parsing/validation to `stencil1D_sticky`: local literal defaults
(`4 10 100`, no `stencil1D.h` involved), `argc` must be exactly 4, and
`nrank==0 || npoints==0 || maxt==0` aborts with a message. No other runtime
knobs. As with `stencil1D_sticky`, an `#ifdef PARALLEL` branch (labeled-GUID
parallel init) is present but dead — never defined by this app's CMake
registration.

## Structure

Let `N=nrank`, `M=npoints`, `T=maxt`.

| object | count | size |
|--------|-------|------|
| EDTs total | `N·(T+1) + 2` — `mainEdt`, `realMainEDT`, `N` chains × `(T+1)` generations. `realMainEDT` pre-creates 2 generations per chain up front (the grandchild-ahead bootstrap) rather than 1; the *total* instance count is unaffected — only who-creates-whom shifts | — |
| EDT templates | 1 (`stencilTML`) | — |
| DBs | `3N-2`, created once, never recreated | private ≈ 48 B (`private_t`: 6×`u32` + 3×`ocrGuid_t`) + `M×8` B; buffer = 16 B (1 `double` + 1 `ocrGuid_t GUID` field) |
| Events | **0** — no `ocrEventCreate` call anywhere in this file's compiled path | — |

Worked numbers at the calibrated args (`48 50 340000`): EDTs ≈
`48×340001+2 = 16,320,050` (identical to `stencil1D_sticky`'s EDT count —
the idiom changes wiring mechanics, not object counts); DBs = `142`,
≈22 KB total payload (marginally smaller than sticky's `private_t`, which
carries two extra `ocrGuid_t` fields for the old-event cleanup this variant
doesn't need); Events = `0`. The live DB/EDT frontier is `O(N)`, same
reasoning as `stencil1D_sticky`.

Counter cross-check: verified (1 node, `4 4 5` vs `6 4 8`): NUM_EDT_CREATE
27 → 57, NUM_DB_CREATE 11 → 17, NUM_EVENT_CREATE 0 → 0 — exactly
`N(T+1)+2` / `3N-2` / `0` (app values 26/56, 10/16, 0/0) plus the
runtime's constant +1 EDT/+1 DB/+0 EVT per-run baseline; the zero event
count is confirmed at both sizes.

## Wiring

- `realMainEDT` creates, per chain, **two** initial generations
  (`stencilGUID` = the one that will run first, `myChildGUID` = its
  already-created successor) and wires the chain head's 3 slots (0=leftIn,
  1=private, 2=rightIn) the same way as `stencil1D_sticky`.
- Each `stencilEDT` clone creates its grandchild (`myGrandChildGUID`) and,
  on send, writes that GUID directly into the buffer DB's `.GUID` field
  alongside the halo value, then calls `ocrAddDependence(bufferGUID,
  neighborTargetGUID, slot, DB_MODE_RW)` — the dependence add IS the
  handoff; there is no intervening satisfy step.
- The private DB and the pre-created immediate-child GUID are handed to
  the *child* (not the grandchild) at the end of each invocation — the
  chain always keeps exactly one generation "in the pipe" ahead of the one
  currently executing.
- Same DB-concurrency shape as `stencil1D_sticky`: no DB ever has more than
  one simultaneous accessor; ownership of each buffer alternates strictly
  between the two neighboring chains.

## Flow

Identical shape to `stencil1D_sticky`: an `O(N)` rank-0 setup preamble
(DB creation + the doubled chain-head bootstrap), `N` parallel chains each
running a `T`-deep serial dependency (now with a 2-generation pipeline
lookahead, which does not change the critical-path depth, only how far
ahead the *next* EDT already exists when the *current* one starts), and an
`O(N)`-deep serialized finalization wave identical to sticky's. Total
critical-path depth ≈ `T + N`.

## Placement (as-born)

`NULL_HINT` everywhere, no affinity code outside the dead `#ifdef PARALLEL`
block — same effective policy as `stencil1D_sticky`: EDTs round-robin
per-create (chain identity has no bearing on rank), DBs home at rank 0 (all
created inside `mainEdt`/`realMainEDT`). The grandchild-ahead pipelining
does not change this: the grandchild is *also* created with `NULL_HINT` and
lands on an independently round-robined rank. Same consequence as sticky —
essentially every RW acquire is a remote round-trip to rank 0, a worst-case
fine-grain coherence stress with no locality expressed.

## Sizing

Same dials, same effect, as `stencil1D_sticky`: `N` caps parallel width,
`T` drives total churn and wall time, `M` only moves per-DB payload and
compute. Use the same sizing guidance — `N` ≳ target worker count so no
worker starves, `T` sized for the desired wall-time at that width. The
calibrated `48 50 340000` is identical across all seven Stencil1D rows (a
deliberate cross-idiom comparison at fixed work), so at a given node/worker
profile this variant and `stencil1D_sticky` see the same parallel width and
critical-path depth — the difference a sweep would surface is in wire/event
traffic (zero event objects here vs. `~32M` for sticky), not in EDT
scheduling pressure (their EDT counts are equal).
