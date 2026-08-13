# stencil1D_stickyLG

*1-D three-point stencil, cloning style, halo exchange over labeled
double-buffered STICKY events, with parallelized initialization.*
Source: `third_party/ocr-apps/apps/Stencil1D/refactored/ocr/intel-david/stencil1DstickyLG.c`
(~565 lines; author Bryan Pawlowski, Intel 2015).

## Overview

Same 1-D three-point elliptic relaxation as `stencil1D_sticky`
(`anew(i) = 0.5*a(i) + 0.25*(a(i-1)+a(i+1))`, boundaries pinned to 1,
converges toward all-1s) and the same `N`-chain/`M`-point/`T`-timestep
decomposition, but a from-scratch reimplementation (different author, README
calls it "sticky with labeled GUIDs") with two structural differences from
`stencil1D_sticky`: **initialization is parallel** (`N` independent `init`
EDTs instead of one serial `mainEdt` loop creating all DBs up front), and
the halo handoff uses **labeled GUIDs with double buffering**. Every rank
independently computes, via `ocrGuidFromIndex` over shared GUID ranges, the
*same* two fixed addresses per neighbor direction (one per `toggle`
generation, 0 and 1) — so a sender and its receiver agree on an event's
identity without ever exchanging it, and only the alternating toggle needs
distinguishing across iterations. Events are still created fresh every
iteration (the address, not the object, is what's reused), and the prior
generation's receive event is explicitly `ocrEventDestroy`'d two iterations
later once superseded — mechanically closer to `stencil1D_sticky`'s
create/destroy churn than to a genuinely reused object. Termination differs
from the lineage-A trio too: a `wrapupEdt` with `N` direct DB dependencies
(one per chain's final data block) does the printing, rather than the last
chain shutting down inline — the README notes this `N`-not-`N+1` dependency
count is what distinguishes every parallel-init variant from the ones with
a `realMain`-as-finish-EDT join.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `numNodes`/`N` | number of independent chains | 10 | ✓ parsed in `mainEdt`, only when exactly 3 args given |
| `argv[2]` = `chunkSize`/`M` | points per chain | 50 | ✓ same |
| `argv[3]` = `timesteps`/`T` | timesteps | 10000 | ✓ same |

Unlike the lineage-A trio, this file `#include`s `stencil1D.h` and its
defaults **are** that header's `N`/`M`/`T` `#define`s (10/50/10000), used
when `argc != 4`. No other runtime knobs; no `nrank==0` style guard exists
in this file (unvalidated zero/negative inputs are not rejected — see
Uncertain).

## Structure

Let `N=numNodes`, `M=chunkSize`, `T=timesteps`.

| object | count | size |
|--------|-------|------|
| EDTs total | `N·T + 3N + 3` — `mainEdt`, `realmainEdt`, `N` `init` EDTs, `N` `stencilInitEdt` EDTs, `N·(T+1)` `stencilEdt` chain generations, 1 `wrapupEdt` | — |
| EDT templates | `2N+2` — `initTemplate`/`wrapupTemplate` created once centrally, but `stencilInitTemplate` (in `initEdt`) and `stencilTML` (in `stencilInitEdt`) are each (re-)created once *per chain* (`N` times) for what is the same function every time, rather than created once and shared (see Findings in notes) | — |
| DBs | `4N+1` — 1 shared block + `N` private blocks + `N` data blocks (double-buffered, `2M` doubles) + `2N` halo buffer blocks | shared ≈ 64 B (`ocrGuid_t wrapup` + 3×`u64` + `ranges[4]`); private ≈ 128 B (double-buffered event-GUID arrays: `left/right/leftrcv/rightrcv[2]` = 8×`ocrGuid_t`, plus scalars); data = `2M×8` B; buffer = 16 B |
| Events | `4(N-1)·T` labeled STICKY event creates — 2 receive + 2 send per internal boundary per iteration, at one of 2 fixed double-buffered addresses per role | — |

`stencilInitEdt` (`stencil1DstickyLG.c:264-274`) creates `sendleftDb` and
`sendrightDb` **unconditionally** on every one of its `N` invocations —
unlike the paired `ocrGuidFromIndex` calls just below, which *are*
boundary-guarded (`private->left[...]`/`private->rightrcv[...]` set to
`NULL_GUID` at rank 0/`N-1`). So rank 0's `sendleftDb` and rank `N-1`'s
`sendrightDb` are created but never wired to anything (their matching
event GUID stays `NULL_GUID`) — 2 orphaned DBs beyond the `2(N-1)` a
naive "one buffer pair per internal boundary" count would predict; hence
`2N`, not `2(N-1)`, halo buffers, and `4N+1` DBs total, not `4N-1`.

Worked numbers at the calibrated args (`48 50 340000`): EDTs ≈
`48×340000 + 3×48 + 3 = 16,320,147`; DBs = `4×48+1 = 193`, ≈45 KB payload;
Events ≈ `4×47×340000 = 63,920,000` — roughly **double** `stencil1D_sticky`'s
event churn, because this variant creates 4 distinct event objects per
boundary per iteration (its own receive *and* send pair, per direction)
where sticky's single shared per-direction event object suffices. The old
generation's receive event is destroyed two iterations later (double
buffering means a create at address A must wait for the *previous*
occupant of A to be freed), so the live event count stays `O(N)` despite
the larger churn.

Counter cross-check: verified (1 node, `4 4 5` vs `6 4 8`): NUM_EDT_CREATE
36 → 70, NUM_DB_CREATE 18 → 26, NUM_EVENT_CREATE 60 → 160 — exactly
`N·T+3N+3` / `4N+1` / `4(N-1)T` (app values 35/69, 17/25, 60/160) plus the
runtime's constant +1 EDT/+1 DB/+0 EVT baseline. The original `4N-1` DB
formula was short by exactly 2 (predicted 16/24 instead of measured
18/26) — the two orphaned boundary buffer DBs above account for it.

## Wiring

- `realmainEdt` creates 2 label-index ranges of size `N` each
  (`ranges[0]`, `ranges[1]`), the shared `wrapupEdt` (depc = `N`), and `N`
  `init` EDTs, each RW-dependent on the shared block.
- `init` creates a private DB, then launches `stencilInitEdt` with 4 slots:
  shared (RO), private (RW), left buffer (RW, absent at rank 0), right
  buffer (RW, absent at rank `N-1`).
- `stencilInitEdt` computes, via `ocrGuidFromIndex`, all 8 double-buffered
  labeled addresses this rank will ever use (`left[0..1]`, `right[0..1]`,
  `leftrcv[0..1]`, `rightrcv[0..1]`) and creates the first chain-head
  `stencilEdt` wired to `leftrcv[0]`/`rightrcv[0]`.
- Each `stencilEdt` generation creates its immediate successor (1-ahead,
  not the 2-ahead grandchild trick of `guid`/`once`), creates the
  receive-event pair for the *next* generation at the toggle-flipped
  address, wires them as the successor's dependences, then creates the
  send-event pair at the same fixed addresses and satisfies them with the
  outgoing halo buffer — dependence-add always precedes satisfy because
  they target different event objects (receive vs. send), so no
  grandchild-ahead pipelining is needed despite the 1-ahead cloning.
- At the final timestep, the chain instead wires its data DB (RO) directly
  onto `wrapupEdt`'s slot `mynode` — the only join mechanism in this
  variant, no output event involved.
- Same DB-concurrency shape as lineage A: no DB has more than one
  simultaneous accessor.

## Flow

Parallel initialization removes the `O(N)` *serial* rank-0 preamble that
lineage-A pays (DB creation is now spread across `N` independently
schedulable `init`/`stencilInitEdt` pairs), at the cost of `2N` extra EDT
stages before steady state begins. Steady state is the same `N`-wide,
`T`-deep parallel chain structure as lineage A. Finalization is a true
`N`-way *join* (via `wrapupEdt`'s `N` dependences) rather than a serialized
wave — all `N` chains can reach their last timestep concurrently and
`wrapupEdt` fires once all have. Critical-path depth ≈ `T` (the
initialization and join stages add `O(1)` depth each, not `O(N)`).

## Placement (as-born)

`NULL_HINT` on every create — no affinity code in this file. Effective
policy differs from lineage A in one structural respect: **DB homes are no
longer concentrated at rank 0**. `init`/`stencilInitEdt` (which create the
private/data/buffer DBs) are themselves round-robin-placed EDTs, so each
chain's DBs home wherever its `init` EDT happened to land — potentially any
rank, not always rank 0. This does not, however, produce locality: every
subsequent `stencilEdt` clone is *independently* round-robin-placed too, so
a chain's DBs stay fixed at whatever rank its `init` drew while the chain's
executing EDT keeps moving — the same worst-case "DB home and executing EDT
are on unrelated ranks" pattern as lineage A, just with the DB homes
scattered across all ranks instead of pinned to rank 0.

## Sizing

`N`, `M`, `T` play the same roles as in lineage A: `N` caps parallel width,
`T` drives total churn/wall-time (now roughly double the event churn of
lineage A for the same `N,T`), `M` only moves data-DB payload. Use `N` ≳
target worker count and size `T` for desired wall time at that width. The
calibrated `48 50 340000` differs from this file's own built-in default
(`10 50 10000`) by both `N` and `T` — a deliberate strong-scaling
calibration shared identically across all seven Stencil1D rows, so a
cross-idiom comparison at fixed `(N,M,T)` isolates the effect of the
event/GUID idiom rather than the workload size.
