# stencil1D_once

*1-D three-point stencil, cloning style, halo exchange over ONCE events.*
Source: `third_party/ocr-apps/apps/Stencil1D/refactored/ocr/intel-david/stencil1Donce.c`
(~660 lines; author David S. Scott, Intel 2015).

## Overview

Same computation, decomposition and convergence behavior as
`stencil1D_sticky`/`stencil1D_guid` (see `stencil1D_sticky`'s Overview for
the shared relaxation formula and cloning idiom). This variant's
distinctive trait: the halo handoff uses **ONCE events**
(`OCR_EVENT_ONCE_T`) instead of STICKY events or raw GUIDs. A ONCE event is
self-destructing — it fires exactly one satisfy and then frees itself, so
(unlike `stencil1D_sticky`) the app never calls `ocrEventDestroy`. It is
safe here (unlike a naive use of ONCE, which would race against an
unordered dependence-add) for the same structural reason `stencil1D_guid`
is safe: each `stencilEDT` clones its **grandchild** before sending,
guaranteeing the dependence is wired before the event can possibly be
satisfied. So this variant sits between the other two: it keeps the
grandchild-ahead pipeline of `guid` but re-adds an explicit synchronization
object per handoff (still churns one event object per direction per
iteration, same count as sticky, but with automatic rather than manual
lifetime management). Like both siblings, there is no separate wrapup EDT —
the last chain prints and shuts down directly.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `nrank` | number of independent chains ("workers") | 4 | ✓ parsed in `mainEdt`, only when exactly 3 args are given |
| `argv[2]` = `npoints` | points per chain | 10 | ✓ same |
| `argv[3]` = `maxt` | timesteps | 100 | ✓ same |

Identical parsing/validation/defaults to the other two lineage-A variants:
local literals (`4 10 100`, no `stencil1D.h`), `argc` must be exactly 4,
zero in any of the three is rejected. No other runtime knobs. The
`#ifdef PARALLEL` branch is again present and dead (never defined by CMake
for this target).

## Structure

Let `N=nrank`, `M=npoints`, `T=maxt`.

| object | count | size |
|--------|-------|------|
| EDTs total | `N·(T+1) + 2` — same formula as `stencil1D_sticky`/`stencil1D_guid`; `realMainEDT` pre-creates 2 generations per chain (grandchild-ahead bootstrap, same as `guid`) | — |
| EDT templates | 1 (`stencilTML`) | — |
| DBs | `3N-2`, created once, never recreated | private ≈ 48 B (`private_t`: 6×`u32` + 3×`ocrGuid_t` — no old-event-cleanup fields, same shape as `guid`'s) + `M×8` B; buffer = 16 B |
| Events | `2(N-1)·T` ONCE events created over the run — same formula and magnitude as `stencil1D_sticky`'s STICKY events, but zero explicit `ocrEventDestroy` calls (self-destructing) | — |

Worked numbers at the calibrated args (`48 50 340000`): EDTs ≈
`16,320,050` (same as both siblings); DBs = `142`, ≈22 KB; Events ≈
`2×47×340000 = 31,960,000` ONCE-event creates — numerically identical churn
to `stencil1D_sticky`, but with no matching destroy-call stream (sticky
issues a comparable number of explicit `ocrEventDestroy` calls that once
never needs, since a ONCE event releases its own resources on satisfy).

Counter cross-check: verified (1 node, `4 4 5` vs `6 4 8`): NUM_EDT_CREATE
27 → 57, NUM_DB_CREATE 11 → 17, NUM_EVENT_CREATE 30 → 80 — exactly
`N(T+1)+2` / `3N-2` / `2(N-1)T` (app values 26/56, 10/16, 30/80) plus the
runtime's constant +1 EDT/+1 DB/+0 EVT per-run baseline, numerically
identical to `stencil1D_sticky`'s counts despite the different event type
(ONCE vs STICKY).

## Wiring

- `realMainEDT` bootstraps 2 generations per chain, same as `guid`
  (`stencilGUID` head + `myChildGUID` already-created successor), and
  creates the initial ONCE send events for each direction.
- Each `stencilEDT` clone creates its grandchild first, then for each
  direction (if a neighbor exists and it is not the final deferred
  handoff): creates a fresh ONCE event, wires it as the grandchild's
  dependence for that direction, stores its GUID into the outgoing buffer's
  `.EVT` field, releases the buffer, and satisfies the *previous*
  generation's send event with it.
- Same DB-concurrency shape as the other two: no DB has more than one
  simultaneous accessor; private DBs are single-owner for the whole run;
  buffer DBs alternate strictly between the two neighboring chains.

## Flow

Identical shape to `stencil1D_guid`: `O(N)` rank-0 setup (now creating the
initial ONCE events too, not just DBs and EDTs), `N` parallel chains each a
`T`-deep serial dependency chain with a 2-generation lookahead, and an
`O(N)`-deep serialized finalization wave. Total critical-path depth ≈
`T + N`.

## Placement (base)

`NULL_HINT` everywhere, no affinity code outside the dead `#ifdef PARALLEL`
block — identical effective policy to `stencil1D_sticky`/`stencil1D_guid`:
EDT creates round-robin per-call (chain identity and rank are unrelated
across generations), DBs home at rank 0 (all created inside
`mainEdt`/`realMainEDT`). Same consequence: essentially every RW acquire a
clone performs is a remote round-trip to rank 0 — worst-case fine-grain
coherence stress, no algorithmic locality expressed in the base program.

## Sizing

Same dials as the other lineage-A variants: `N` caps parallel width, `T`
drives total churn/wall-time, `M` only moves per-DB payload/compute and is
free to size independently. Same guidance — `N` ≳ target worker count, `T`
sized for desired wall time at that width. At the shared calibrated args
(`48 50 340000`) this variant's parallel width and critical-path depth
match both siblings exactly; a cross-idiom sweep would surface differences
in event-object churn and destroy-call overhead (sticky pays an explicit
destroy per superseded event, once does not), not in scheduling pressure.
