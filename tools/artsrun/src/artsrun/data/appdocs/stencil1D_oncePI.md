# stencil1D_oncePI

*1-D three-point stencil, cloning style, halo exchange over plain ONCE
events, with parallelized initialization.*
Source: `third_party/ocr-apps/apps/Stencil1D/refactored/ocr/intel-david/stencil1DoncePI.c`
(~635 lines; author Bryan Pawlowski, Intel 2015).

## Overview

Same 1-D three-point elliptic relaxation and `N`-chain/`M`-point/`T`-timestep
decomposition as `stencil1D_sticky` (see that doc's Overview for the
formula and convergence behavior). This is the "once, parallel init"
member of the family: like `stencil1D_stickyLG`, initialization is spread
across `N` independent `init`/`stencilInitEdt` pairs rather than one serial
`mainEdt` loop, and termination is an `N`-dependency `wrapupEdt` join
instead of a serialized shutdown wave. Unlike `stickyLG`, the steady-state
halo handoff does **not** use labeled/double-buffered addressing — it uses
plain (non-labeled) ONCE events, self-destructing, created fresh every
iteration and explicitly propagated forward through the buffer DB's
`.control` field (the receiving neighbor's *grandchild* is wired to depend
on the event before it is ever satisfied — the same grandchild-ahead
pipelining `stencil1D_once`/`stencil1D_guid` use, now combined with
parallel init). Labeled GUIDs are used only once, at bootstrap, to kick off
the very first generation's handoff without any runtime coordination.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `N` | number of independent chains | 10 | ✓ parsed in `mainEdt`, only when exactly 3 args given |
| `argv[2]` = `M` | points per chain | 50 | ✓ same |
| `argv[3]` = `T` | timesteps | 10000 | ✓ same |

Like `stencil1D_stickyLG`, this file `#include`s `stencil1D.h` and its
defaults are that header's `N=10, M=50, T=10000` `#define`s, used when
`argc != 4`. No other runtime knobs; no validation guard against
zero/negative arguments (see Uncertain).

## Structure

Let `N`, `M`, `T` as above.

| object | count | size |
|--------|-------|------|
| EDTs total | `N·T + 4N + 3` — `mainEdt`, `realmainEdt`, `wrapupEdt`, `N` `init`, `N` `stencilInitEdt` (each of which creates *two* `stencilEdt` instances — the head and its pre-created grandchild), and `N·T` further `stencilEdt`-created grandchildren over the run | — |
| EDT templates | `2N+2` — `initTemplate`/`wrapupTemplate` created once centrally; `stencilInitTemplate` and `private->template` (for `stencilEdt`) are each (re-)created once *per chain* rather than shared (same pattern as `stickyLG`; see Findings) | — |
| DBs | `4N-1` — 1 shared + `N` private + `N` data (double-buffered, `2M` doubles) + `2(N-1)` buffer | shared ≈ 48 B (`ocrGuid_t wrapup` + `ranges[2]` + 3×`u64`); private ≈ 80 B (`private_t`: `wrapup`/`template`/`mychild` `ocrGuid_t` + `toDestroy[2]`, no double-buffer arrays); data = `2M×8` B; buffer = 16 B |
| Events | `2(N-1)·(T+3)` — `2(N-1)·T` steady-state ONCE events (1 per direction per interior rank per iteration, self-destructing) plus `6(N-1)` one-time bootstrap events `stencilInitEdt` creates per interior boundary | — |

Each chain's `stencilEdt` grandchild-ahead pipeline creates one new EDT
per generation `0..T-1` (`T` creates) on top of the *two* instances
`stencilInitEdt` pre-creates directly (`stencil` = generation 0, and
`private->mychild` = generation 1) — `T+2` `stencilEdt` instances per
chain, not `T+1`: generation `T-1`'s invocation still creates a
generation-`T+1` grandchild before generation `T` (which runs the
`timestep==t` finalize branch and creates nothing) is known to be the
last one needed, so that final grandchild is created but never invoked —
one genuinely orphaned EDT per chain. `stencilInitEdt` also creates, once
per interior boundary and independent of `T`: 2 ONCE events (the initial
grandchild's left/right dependences), 2 STICKY events (`leftrcv`/
`rightrcv`, the receive side of the very first cross-rank handoff), and 2
STICKY events (`leftsend`/`rightsend`, its send side) — 6 events per
interior boundary, `stencil1DoncePI.c:288-394`.

Worked numbers at the calibrated args (`48 50 340000`): EDTs ≈
`48×340000 + 4×48 + 3 = 16,320,195`; DBs = `191`, ≈43 KB payload
(marginally less than `stickyLG`'s, whose private struct carries the
double-buffer address arrays this one doesn't need); Events ≈
`2×47×(340000+3) = 31,960,282` — within 0.001% of the naive `2(N-1)T`
estimate, matching the same per-iteration magnitude as
`stencil1D_once`'s plus a fixed one-time bootstrap term.

Counter cross-check: verified (1 node, `4 4 5` vs `6 4 8`): NUM_EDT_CREATE
40 → 76, NUM_DB_CREATE 16 → 24, NUM_EVENT_CREATE 48 → 110 — exactly
`N·T+4N+3` / `4N-1` / `2(N-1)(T+3)` (app values 39/75, 15/23, 48/110)
plus the runtime's constant +1 EDT/+1 DB/+0 EVT baseline. The original
`N·T+3N+3` EDT formula was short by exactly `N` (the orphaned
grandchild-per-chain above); the original `2(N-1)T` event formula was
short by `6(N-1)` (the bootstrap events above) — both corrected.

## Wiring

- `realmainEdt` creates 2 label ranges of size `N` (`ranges[0..1]`), the
  `wrapupEdt` (depc = `N`), and `N` `init` EDTs.
- `init` creates a private DB plus left/right buffer DBs (absent at the
  respective boundary) and launches `stencilInitEdt`.
- `stencilInitEdt` bootstraps the first-generation ONCE events for its own
  grandchild (`leftEVT`/`rightEVT`) and, via `ocrGuidFromIndex` over the
  shared ranges, the labeled STICKY events used *only* for the very first
  cross-rank handoff (`leftinGUID`/`rightinGUID`, `leftoutGUID`/
  `rightoutGUID`) — creating and immediately satisfying the outbound ones
  with the (still-empty, first-iteration) buffer DBs.
- Each `stencilEdt` generation creates its grandchild, creates fresh ONCE
  events for the grandchild's two dependences, then — for each direction —
  reads the previous generation's control GUID out of the buffer
  (`leftin->control`/`rightin->control`), overwrites it with the new ONCE
  event's GUID, releases the buffer, and satisfies the *old* control GUID
  with it. This is the explicit control-field propagation `stickyLG` avoids
  via labeled double-buffering.
- Final timestep: RW-dependence from the chain's data DB directly onto
  `wrapupEdt`'s slot `mynode`, same join mechanism as `stickyLG`.
- Same DB-concurrency shape throughout: no DB has more than one
  simultaneous accessor.

## Flow

Same shape as `stencil1D_stickyLG`: `O(1)`-per-chain parallel
initialization (no serial rank-0 preamble beyond the one-time label-range
and `wrapupEdt` setup), `N` parallel chains each a `T`-deep serial
dependency chain (grandchild-ahead, so 2 generations always exist before
the currently-running one completes), and a true `N`-way join via
`wrapupEdt`. Critical-path depth ≈ `T`.

## Placement (as-born)

`NULL_HINT` everywhere. Same effective policy as `stickyLG`: DB homes are
scattered across whichever rank each chain's round-robin-placed `init` EDT
landed on (not concentrated at rank 0, unlike the lineage-A trio), but
every subsequent `stencilEdt` clone is independently round-robin-placed
too, so the executing EDT and the DBs it acquires are on unrelated ranks
just as often — no locality is expressed by the as-born program.

## Sizing

Same dials and guidance as `stencil1D_stickyLG`: `N` caps parallel width,
`T` drives churn/wall-time, `M` only moves data-DB payload. `N` ≳ target
worker count, `T` sized for desired wall time at that width. At the shared
calibrated `48 50 340000`, this variant's event churn is half of
`stickyLG`'s at equal `N,T` — the two make a natural within-family A/B pair
for isolating the cost of labeled double-buffering vs. explicit
control-field propagation at otherwise-identical parallelism and depth.
