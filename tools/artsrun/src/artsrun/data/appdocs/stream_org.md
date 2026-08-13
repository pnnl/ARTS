# stream_org

*STREAM with McCalpin's original in-place arrays: three DataBlocks per thread,
allocated once and mutated `nTimes` times — no per-iteration churn.*
Source: `third_party/ocr-apps/apps/stream/ocr/stream_org.c` (~510 lines; the
final third is a `#if 0`-disabled serial reference checker, dead code).

## Overview

The "original single-node form" of the STREAM port: each thread creates its
`a`/`b`/`c` arrays exactly once (`mainLet`) and every iteration's kernels
write in place into the same three DataBlocks, rather than allocating fresh
ones (contrast `stream`/`stream_sa`). As in the other two ports, `scale`
reads directly from `a` (not from `copy`'s output), so `copy`'s write into
`c` is immediately clobbered, unread, by `add`'s own write into the same `c`
— a legitimate write-after-write, ordered by `add`'s `RW` dependence on
`copy`'s output DB, not a race. The result scalar is `STREAM_VALID relerr`:
`finalize` replays the copy/scale/add/triad recurrence natively (`scalar =
3.0`, matching the kernels' own constant) and reports `a[0]`'s relative error
against it — a correct run prints a value at or near zero. `finalize` also
prints an `MB/s` line first (the catalog's completion marker), which is a
separate diagnostic, not the validated scalar; per Placement, its wall-clock
inputs are unreliable at multinode. `STREAM_ARRAY_SIZE` (9,000,000) and
`NUM_THREADS` (32) are compile-time constants here — this port cannot be
resized from the command line at all, only run for more or fewer iterations,
which is reflected in the catalog leaving it disabled by default with no
argument list to sweep.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `ntimes` | iterations per chain (DAG depth); accepted only if `2 ≤ v ≤ NTIMES_MAX(4000)` | 1000 | ⚠ parsed in `mainEdt`, forwarded through paramv (multinode-safe) — an out-of-range or unparsable value falls back to the default and prints a one-line warning naming the accepted range and the rejected value |

Compile-time only (✗, no CLI path): `STREAM_ARRAY_SIZE` (9,000,000 total
elements), `NUM_THREADS` (32, DAG width — `PER_THREAD_SIZE` is their
quotient, 281,250), `scalar` (3.0, a plain global initializer). `NTIMES_MAX`
(4000) is the static bound of the `times[NUM_THREADS][NTIMES_MAX]` timing
array — the reason `ntimes` is capped at all.

## Structure

Let `T = 32` (fixed), `K = ntimes`, `pts = 281250` (fixed). Per thread: 1
`mainLet` EDT, `K` `loop` EDTs (one `FINISH`, the rest plain), and `4·K`
kernel EDTs — but, unlike the churned variants, only **3** DataBlocks ever
exist per thread, created once and reused for all `K` iterations.

| object | count | size |
|--------|-------|------|
| EDTs total | `T·(1 + 5K) + 2` | — |
| DBs | `3T` — constant, independent of `K` (no per-iteration create/destroy) | `281250·8 = 2,250,000` bytes (~2.15 MiB) each |
| Events | `T·(3 + 4K)` — per thread: the ONCE completion event, the first `loop`'s output event, its finish event (it is the `FINISH` EDT), plus one output event per kernel EDT each iteration (the runtime materializes an event for every `ocrEdtCreate` with a non-NULL output-event slot) | — |
| EDT templates | 7 (`mainLet`, `loop`, `copy`, `scale`, `add`, `triad`, `finalize`) | — |

At the default `ntimes = 1000` (the catalog's `args: []` takes no override):
EDTs total = `32·5001+2 = 160,034`; DBs = `96`; events = `32·4003 =
128,096`. Total array
memory is `96·2.15 MiB ≈ 206.3 MiB`, held for the entire run (matches the
program's own printed "Total memory required" line, `3·8·9,000,000` bytes)
— this is the one STREAM port where memory footprint is a real, fixed,
sizing input rather than a transient churn peak.

Counter cross-check: verified (1 node, `ntimes` 5 vs 8): NUM_EDT_CREATE
835/1315, NUM_DB_CREATE 97/97 (constant — no churn), NUM_EVENT_CREATE
736/1120 — exactly `32·(1+5K)+2`, `3·32` and `32·(3+4K)` plus the
runtime's constant +1 EDT / +1 DB baseline.

## Wiring

- `mainLet` creates `a`/`b`/`c` once, initializes them (`a=2, b=2, c=0`),
  releases all three, then wires them **RO** into the first `loop` (a
  `FINISH` EDT) — the loop shell only ever forwards these GUIDs onward, it
  never dereferences their payload.
- Each `loop(iter)` re-wires the same three GUIDs with the modes that
  actually touch data: `a`(RO)→`copy`,`scale`; `a`(RO)+`scale_output`
  (b, RO)+`copy_output`(c, RW)→`add`; `a`(RW)+`add_output`(c,
  RO)+`scale_output`(b, RO)→`triad`. It then forwards `triad_output`,
  `scale_output` and `add_output` **RO** into `loop(iter+1)` (or, on the
  last iteration, `triad_output` into `evt_finalize_my`).
- Per iteration: `a` has 2 concurrent RO readers (`copy`, `scale`); `b` has 2
  concurrent RO readers (`add`, `triad`); `c` is written twice in strict
  sequence (`copy` then `add`, ordered by `add`'s `RW` hold on `copy`'s
  output) with no reader in between — `copy`'s value is computed and
  discarded. `a` and `b` each have exactly one writer per iteration (`triad`,
  `scale`); no DB ever has two writers active at once.
- `a`, `b` and `c` are the only objects that live the whole run: each is
  re-written `K` times by `K` independently-placed EDT instances (see
  Placement) — over the run, each array's current owner effectively
  random-walks across ranks, rather than each write starting fresh at its
  own creator's rank as in `stream`/`stream_sa`. No DB is ever shared across
  threads; the `T=32` chains join only at `finalize`.

## Flow

Same per-iteration shape as `stream`: `copy`∥`scale` (width 2), then `add`,
then `triad` (width 1 each); loop iterations are a strict serial chain of
depth `K` per thread (no intra-thread pipelining, since `loop(iter+1)` isn't
even created until `triad(iter)` finishes). The 32 chains are mutually
independent and unsynchronized, giving instantaneous parallel width up to
`2·32 = 64`. Every thread's `loop(iter=0)` is an `EDT_PROP_FINISH` EDT
fencing that thread's whole `K`-deep subtree into its own OCR scope
(`mainLet` discards the finish EDT's own output event and completes
immediately after creating it); the actual cross-thread join is the explicit
`evt_finalize_i` ONCE event, satisfied by the last iteration's `triad` and
consumed by the single `finalize` EDT (`depc = 32`). `mainEdt`'s
`preamble()` (banner + size printout) runs once, natively, before any EDT
exists.

## Placement (as-born)

All DB and EDT creates in this source use `NULL_HINT`, and `mainLet` is
additionally given an `OCR_HINT_EDT_DISPERSE`/`NEAR` hint that the shim
never inspects (it only reads `OCR_HINT_EDT_AFFINITY`) — so, exactly as in
`stream`, every EDT (`mainLet` included) is placed by runtime round-robin,
and every DB homes at its creator's rank.

This port is the more exposed case for multinode traffic: because `a`, `b`
and `c` are the *same three DataBlocks* for all `K` iterations, and each
iteration's writer (`triad`, `scale`, or `copy`/`add`) is independently
round-robin-placed, each array's owning rank changes with probability
`(N-1)/N` on essentially every one of the `K` writes — not once at creation
(as in `stream`), but repeatedly, for the life of the run. At `N` ranks this
turns a 3-array, 206 MiB working set into a target for `K`-many full-array
(2.15 MiB) ownership migrations per thread, dwarfing whatever the algorithm
itself needed to move. Separately, `finalize` is itself round-robin-placed
and its `printTimes()` reads a plain, unguarded, per-process static array
(`times[NUM_THREADS][NTIMES_MAX]`); a cell is only meaningful when the
`loop` that set it and the `triad` that later diffed it ran on the *same*
rank, which — given both are independently round-robin-placed — is rare at
multinode. The printed `MB/s` line should not be trusted there; the
`STREAM_VALID relerr` scalar the harness reads is computed from `depv[0].ptr`
directly (a real, coherence-resolved acquire) and is unaffected.

## Sizing

`ntimes` is the only reachable dial, and it only adds depth (more EDTs, more
DBs... no — more EDTs but the *same* 3 DBs, reused). Parallel width is fixed
at `2·32 = 64` instantaneous EDTs, independent of everything a caller can
pass on the command line.

- 1 node × 15 workers / 2 nodes × 30: 64 ≥ workers, fine.
- 4 nodes × 60 workers / 8 nodes × 120 workers: 64 < 60 and 64 ≪ 120 — this
  port structurally cannot fill a wide profile; growing it requires
  recompiling with a different `NUM_THREADS`/`STREAM_ARRAY_SIZE`, which is
  exactly why the catalog carries it with `args: []` and
  `default_enabled: false` rather than as a strong-scaling row. Use `ntimes`
  only to trade run length for timing-sample count (tens of iterations for a
  quick check, up to `NTIMES_MAX = 4000` for a longer one); it does not
  change what a sweep across node counts would show. Memory is fixed at
  ~206 MiB regardless of `ntimes` — never the limiting factor here either.
