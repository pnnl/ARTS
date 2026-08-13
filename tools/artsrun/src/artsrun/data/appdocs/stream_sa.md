# stream_sa

*STREAM's "shared-array form": the same fresh-DB-per-iteration churn as
`stream`, but with `numThreads` and the array size fixed at compile time
(same 32 threads / 9,000,000 elements as `stream_org`) instead of read from
argv.*
Source: `third_party/ocr-apps/apps/stream/ocr/stream_sa.c` (~500 lines; the
final third is a `#if 0`-disabled serial reference checker, dead code).

## Overview

Structurally this is `stream`'s DB-churn pattern (every kernel creates a
fresh output DataBlock; `triad` destroys the previous iteration's `a`, `b`
and `c`) grafted onto `stream_org`'s fixed sizing (`STREAM_ARRAY_SIZE =
9,000,000`, `NUM_THREADS = 32`, both compile-time constants — the CMake
target defines neither, so these are the source's own `#define` defaults).
As in the other two ports, `scale` reads directly from `a` rather than from
`copy`'s output, so `copy`'s freshly-created array is written once and then
destroyed unread — its `RW` dependence into `add` exists only to authorize
the destroy. The result scalar is `STREAM_VALID relerr`: `finalize` replays
the recurrence natively (`scalar = 3.0`) and reports `a[0]`'s relative error,
independent of the `MB/s` diagnostic line `finalize` prints first (the
catalog's completion marker). That timing line reads a plain, unguarded,
per-process static array (the same pattern `stream_org` uses, not
`stream`'s NULL-guarded heap pointer), so it inherits `stream_org`'s
multinode timing-reliability caveat even though its DB lifetime management
matches `stream`. The provenance label "shared-array form" does not map
cleanly onto the DB-churn behavior actually in this file — see Uncertain in
this app's notes.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `ntimes` | iterations per chain (DAG depth); accepted only if `2 ≤ v ≤ NTIMES_MAX(4000)` | 1000 | ⚠ parsed in `mainEdt`, forwarded through paramv (multinode-safe) — an out-of-range or unparsable value falls back to the default and prints a one-line warning naming the accepted range and the rejected value |

Compile-time only (✗, no CLI path): `STREAM_ARRAY_SIZE` (9,000,000),
`NUM_THREADS` (32 — `PER_THREAD_SIZE` is their quotient, 281,250), `scalar`
(3.0). `NTIMES_MAX` (4000) bounds the static `times[NUM_THREADS][NTIMES_MAX]`
timing array and is why `ntimes` is capped.

## Structure

Let `T = 32` (fixed), `K = ntimes`, `pts = 281250` (fixed). Per thread: 1
`mainLet` EDT, `K` `loop` EDTs (one `FINISH`, the rest plain), `4·K` kernel
EDTs, and — like `stream`, unlike `stream_org` — a fresh DB from every
kernel call each iteration (nothing persists across iterations).

| object | count | size |
|--------|-------|------|
| EDTs total | `T·(1 + 5K) + 2` | — |
| DBs | `T·(1 + 4K)` — one initial `a`, then `copy`/`scale`/`add`/`triad` each create one fresh DB per iteration | `281250·8 = 2,250,000` bytes (~2.15 MiB) each |
| Events | `T·(3 + 4K)` — per thread: the ONCE completion event, the first `loop`'s output event, its finish event (it is the `FINISH` EDT), plus one output event per kernel EDT each iteration (the runtime materializes an event for every `ocrEdtCreate` with a non-NULL output-event slot) | — |
| EDT templates | 7 (`mainLet`, `loop`, `copy`, `scale`, `add`, `triad`, `finalize`) | — |

At the default `ntimes = 1000` (the catalog's `args: []` takes no override):
EDTs total = `32·5001+2 = 160,034`; DBs = `32·4001 = 128,032`; events =
`32·4003 = 128,096`. Each thread
holds only a handful of DBs live at any instant (creation → single
consumption → destruction within one iteration), so the live set is a small
multiple of `2.15 MiB` per thread (~8.6 MB), not the full 128,032-DB total;
`finalize` briefly holds all 32 threads' final `a[]` at once (≈68.8 MB).

Counter cross-check: verified (1 node, `ntimes` 5 vs 8): NUM_EDT_CREATE
835/1315, NUM_DB_CREATE 673/1057, NUM_EVENT_CREATE 736/1120 — exactly
`32·(1+5K)+2`, `32·(1+4K)` and `32·(3+4K)` plus the runtime's constant
+1 EDT / +1 DB baseline.

## Wiring

- `mainEdt` creates the 7 templates, one `finalize` EDT (`depc = 32`), then
  per thread one ONCE event `evt_finalize_i` (RO into `finalize`) and one
  `mainLet` EDT.
- `mainLet` creates the thread's initial `a` (`= 2.0`, no explicit release —
  held RW by the creating EDT), then one `EDT_PROP_FINISH` `loop(iter=0)`
  wired `a` RW; `mainLet`'s own output event is discarded, since the finish
  wrapper only fences the per-thread subtree, not `mainLet` itself.
- `loop(iter)` creates `copy`/`scale`/`add`/`triad`, wiring `a`(RO)→
  `copy`,`scale`; `scale_output`(RO)+`copy_output`(RW, destroy-only)→`add`;
  `add_output`(RO)+`scale_output`(RO)+`a`(RW, destroyed)→`triad`; then
  either creates `loop(iter+1)` with `triad_output` RW, or (last iteration)
  satisfies `evt_finalize_i` with it.
- Per iteration, max concurrent readers on any one DB is 2 (`a` read by
  `copy` and `scale`; `scale_output` read by `add` and `triad`). Every DB has
  exactly one writer across its whole lifetime — a fresh GUID every time —
  so no DB ever takes `RW` from more than one node's tasks concurrently, and
  no DB is shared across threads; the 32 chains join only at `finalize`.
  There is no contention point in the concurrency sense (identical to
  `stream` in this respect; see `stream_org`'s notes for the contrasting
  persistent-array case).

## Flow

Same per-iteration shape as `stream`: `copy`∥`scale` (width 2), then `add`,
then `triad` (width 1). Loop iterations form a strict serial chain of depth
`K` per thread — `loop(iter+1)` is not created until `triad(iter)`
completes, so no intra-thread pipelining. The 32 independent, unsynchronized
chains give instantaneous parallel width up to `2·32 = 64`. Every thread's
`loop(iter=0)` is `EDT_PROP_FINISH`, fencing that subtree's completion
tracking away from `mainLet`; the real cross-thread join is the explicit
`evt_finalize_i` event, satisfied by the last iteration's `triad` and
consumed by the single `finalize` EDT. `mainEdt`'s `preamble()` runs once,
natively, before any EDT exists.

## Placement (as-born)

Every DB and EDT create in this source uses `NULL_HINT`, and `mainLet` is
additionally given an inert `OCR_HINT_EDT_DISPERSE`/`NEAR` hint — the shim's
affinity extractor only reads `OCR_HINT_EDT_AFFINITY`, so it has no effect.
Every EDT is round-robin-placed; every DB homes at its creator's rank.

Because every DB is fresh (as in `stream`, not `stream_org`'s persistent
arrays), the traffic pattern is: each phase transition is one coin flip
(probability `(N-1)/N` at `N` ranks) on the new consumer landing away from
where the producer created its output — `a` into `copy`/`scale`,
`scale_output` into `add`/`triad`, `add_output` into `triad`, and the
churned `a` into the next `loop`. With `pts = 281,250` fixed (2.15 MiB per
DB, larger than `stream`'s calibrated 162.8 KiB), each such remote hop moves
more data than `stream`'s. `finalize` is itself round-robin-placed, and its
`printTimes()` reads the same unguarded, per-process static
`times[NUM_THREADS][NTIMES_MAX]` array `stream_org` uses (not `stream`'s
`NULL`-guarded pointer) — a cell is only meaningful when the `loop` that set
it and the `triad` that diffed it land on the same rank, which independent
round-robin placement makes unlikely at multinode. The `MB/s` line should
not be trusted there; `STREAM_VALID relerr` is computed from a real,
coherence-resolved acquire of `depv[0].ptr` and is unaffected.

## Sizing

`ntimes` is the only reachable dial; it adds depth (more EDT/DB churn, a
longer per-thread chain), not width. Parallel width is fixed at `2·32 = 64`
instantaneous EDTs regardless of arguments.

- 1–2 nodes (15–30 workers): 64 ≥ workers, fine.
- 4–8 nodes (60–120 workers): 64 < 60 and 64 ≪ 120 — this port cannot widen
  itself from the command line; raising `NUM_THREADS`/`STREAM_ARRAY_SIZE`
  requires recompiling. This mirrors `stream_org` and is consistent with the
  catalog carrying it with `args: []` and `default_enabled: false` — a
  fixed-shape reference variant, not a swept row. Use `ntimes` to trade run
  length for timing-sample count (tens of iterations for a quick check, up
  to `NTIMES_MAX = 4000` for longer). Memory never binds: the live set stays
  a small multiple of one thread's 2.15 MiB working set regardless of
  `ntimes`, since DBs are churned rather than accumulated.
