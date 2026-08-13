# stencil1D_guidPI

*1-D three-point stencil, cloning style, halo exchange by passing raw
control GUIDs — no steady-state OCR events — with parallelized
initialization.*
Source: `third_party/ocr-apps/apps/Stencil1D/refactored/ocr/intel-david/stencil1DguidPI.c`
(~610 lines; author Bryan Pawlowski, Intel 2015).

## Overview

Same 1-D three-point elliptic relaxation and `N`-chain/`M`-point/`T`-timestep
decomposition as `stencil1D_sticky` (see that doc's Overview for the
formula and convergence behavior). This is the "no events, parallel init"
member of the family — structurally the parallel-init sibling of
`stencil1D_guid`, combining that variant's headline trait (zero OCR event
objects in steady state — the neighbor's grandchild dependence is wired
directly via a raw GUID carried in the buffer DB's `.control` field) with
`stencil1D_stickyLG`/`stencil1D_oncePI`'s parallel `init`/`stencilInitEdt`
bootstrap and `N`-dependency `wrapupEdt` join. Labeled STICKY events appear
only once, at bootstrap, to seed the very first handoff at each internal
boundary without runtime coordination; nothing is created or destroyed at
any of the remaining `T-1` iterations beyond the EDTs themselves. This
makes it the cheapest of the seven variants in synchronization-object
terms — comparable to `stencil1D_guid`'s zero steady-state events, but
without that variant's serial rank-0 initialization bottleneck.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `N` | number of independent chains | 10 | ✓ parsed in `mainEdt`, only when exactly 3 args given |
| `argv[2]` = `M` | points per chain | 50 | ✓ same |
| `argv[3]` = `T` | timesteps | 10000 | ✓ same |

Like `stencil1D_stickyLG`/`stencil1D_oncePI`, this file `#include`s
`stencil1D.h` and defaults to that header's `N=10, M=50, T=10000` when
`argc != 4`. No other runtime knobs; no zero/negative-argument guard (see
Uncertain).

## Structure

Let `N`, `M`, `T` as above.

| object | count | size |
|--------|-------|------|
| EDTs total | `N·T + 3N + 3` — same skeleton as `stickyLG`/`oncePI`: `mainEdt`, `realmainEdt`, `N` `init`, `N` `stencilInitEdt`, `N·(T+1)` `stencilEdt` generations (2-ahead grandchild pipeline, same as `stencil1D_guid`), 1 `wrapupEdt` | — |
| EDT templates | `2N+2` — `initTemplate`/`wrapupTemplate` created once centrally; `stencilInitTemplate` and `stencilTML` are each (re-)created once per chain rather than shared (same pattern as `stickyLG`/`oncePI`; see Findings) | — |
| DBs | `4N+1` — 1 shared + `N` private + `N` data (double-buffered, `2M` doubles) + `2N` buffer | shared ≈ 48 B (`ocrGuid_t wrapup` + 3×`u64` + `ranges[2]`); private ≈ 80 B (`private_t`: `wrapup`/`template`/`mychild` `ocrGuid_t` + `toDestroy[2]`, same shape as `oncePI`'s); data = `2M×8` B; buffer = 16 B |
| Events | `4(N-1)` labeled STICKY `ocrEventCreate` calls, **all at bootstrap, none afterward** — independent of `T` | — |

`initEdt` (`stencil1DguidPI.c:417-432`) creates `leftDb`/`rightDb`
**unconditionally** on every one of its `N` invocations, with no boundary
guard — unlike `stencilInitEdt`'s later use of them (`lbuf!=NULL`
gating, line 291), the `ocrDbCreate` calls themselves run for every node.
So node 0's `leftDb` and node `N-1`'s `rightDb` are created but never
populated by a real neighbor — 2 orphaned DBs beyond the naive
`2(N-1)`, the same pattern as `stickyLG`'s `sendleftDb`/`sendrightDb`
bug; hence `2N` buffer DBs and `4N+1` total, not `4N-1`.

Each of the `2(N-1)` distinct labeled STICKY addresses at an interior
boundary is computed independently by **both** sides of that boundary via
`ocrGuidFromIndex` — the receiving node's `stencilInitEdt` call (`leftrcv`/
`rightrcv`) and the sending node's own `stencilInitEdt` call
(`rightsend`/`leftsend` one node over) resolve to the *same* address, and
**each side calls `ocrEventCreate` on it independently**
(`stencil1DguidPI.c:344-389`); `GUID_PROP_CHECK` makes the second call an
idempotent no-op rather than a duplicate object, but `NUM_EVENT_CREATE`
still counts both calls. So the create-call count is `4(N-1)`, double the
`2(N-1)` distinct event objects that actually exist.

Worked numbers at the calibrated args (`48 50 340000`): EDTs ≈
`48×340000 + 3×48 + 3 = 16,320,147` (same formula as `stickyLG`/`oncePI`);
DBs = `4×48+1 = 193`, ≈43 KB payload; Events = `4×47 = 188` total, for the
*entire* 340,000-iteration run — five to six orders of magnitude below
every other event-using variant, since no event object is ever created
past the first handoff.

Counter cross-check: verified (1 node, `4 4 5` vs `6 4 8`): NUM_EDT_CREATE
36 → 70, NUM_DB_CREATE 18 → 26, NUM_EVENT_CREATE 12 → 20 — exactly
`N·T+3N+3` / `4N+1` / `4(N-1)` (app values 35/69, 17/25, 12/20) plus the
runtime's constant +1 EDT/+1 DB/+0 EVT baseline. `EDTs` was already exact;
`DBs` was short by 2 (the orphaned boundary buffers) and `Events` was
short by exactly half (the double-create-per-address effect) — both
corrected.

## Wiring

- `realmainEdt` creates 2 label ranges of size `N`, the `wrapupEdt`
  (depc = `N`), and `N` `init` EDTs.
- `init` creates a private DB plus left/right buffer DBs and launches
  `stencilInitEdt`.
- `stencilInitEdt` pre-creates **two** chain-head generations (`stencil`
  and `private->mychild`, the grandchild-ahead bootstrap, same trick as
  `stencil1D_guid`), sets each boundary buffer's `.GUID`-equivalent
  (`control`, via `leftPTR->GUID`/`rightPTR->GUID` in this file) to
  `myChildGUID`, and — via `ocrGuidFromIndex` over the shared ranges —
  creates and immediately satisfies the labeled bootstrap STICKY events
  that seed the very first cross-rank handoff.
- Each `stencilEdt` generation creates its grandchild, stores the
  grandchild's GUID into the outgoing buffer's control field, releases the
  buffer, and calls `ocrAddDependence(bufferGUID, targetGUID, slot,
  DB_MODE_RW)` directly — again, the add-dependence call *is* the handoff,
  no satisfy step, no event object.
- Final timestep: RW-dependence from the chain's data DB directly onto
  `wrapupEdt`'s slot `mynode`.
- Same DB-concurrency shape throughout: no DB has more than one
  simultaneous accessor.

## Flow

Same shape as `stencil1D_stickyLG`/`stencil1D_oncePI`: parallel
`O(1)`-per-chain initialization, `N` parallel chains each a `T`-deep serial
chain (now with the grandchild-ahead lookahead), and a true `N`-way join
via `wrapupEdt`. Critical-path depth ≈ `T`.

## Placement (as-born)

`NULL_HINT` everywhere. Same effective policy as the other parallel-init
variants: DB homes scatter across whichever rank each chain's
round-robin-placed `init` EDT drew, while every subsequent `stencilEdt`
clone is independently round-robin-placed — the executing EDT and the DBs
it acquires are on unrelated ranks for essentially every iteration, no
locality expressed.

## Sizing

Same dials as the rest of the family: `N` caps parallel width, `T` drives
EDT churn and wall time, `M` only moves data-DB payload. `N` ≳ target
worker count, `T` sized for desired wall time at that width. Because this
variant's *event* churn is flat in `T` (only `4(N-1)` create calls ever, vs. millions for
the STICKY/ONCE siblings), a sweep across this row isolates pure
EDT-scheduling and DB-coherence cost from event-plane overhead — useful as
the "zero-synchronization-object" reference point when comparing against
`stickyLG`/`oncePI` at the same `(N,M,T)`.
