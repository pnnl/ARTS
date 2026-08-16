# stream

*The four McCalpin STREAM kernels (copy/scale/add/triad) run over `numThreads`
independent per-thread arrays, `nTimes` times each, with every kernel's output
array freshly created and the previous one destroyed.*
Source: `third_party/ocr-apps/apps/stream/ocr/stream.c` (~580 lines; the
final third is a `#if 0`-disabled serial reference checker, dead code).

## Overview

Computes the classic STREAM bandwidth kernels — `copy: c=a`, `scale:
b=scalar*a`, `add: c=a+b`, `triad: a=b+scalar*c` — but unlike McCalpin's
canonical STREAM, `scale` reads directly from `a`, not from `copy`'s output
`c`; `copy`'s result is computed and then discarded (its `RW` dependence into
`add` exists only to authorize destroying the DB, never to read it). The
array is split into `numThreads` disjoint, mutually independent chains
(no data or control dependence between them), each iterated `nTimes` times
with a fresh set of `a`/`b`/`c` DataBlocks churned every iteration. The result
scalar the harness reads is `STREAM_RESULT a[0] = ...`, printed by `finalize`
alongside an independent "Solution Validates" check that replays the same
recurrence natively for every thread's final `a[]`. `finalize` also prints an
`MB/s` line aggregating all four kernels' 10-word-per-element traffic over one
thread's best (minimum) per-iteration wall time; that figure is a separate
diagnostic, not what the harness validates, and (see Placement) its inputs
are unreliable at multinode. The catalog lists a restructured twin,
`stream_dist`, that redecomposes the same benchmark and is a live, separately
registered target with its own arguments and pinned answer. With near-zero arithmetic per
element and DBs churned on every step, the program is a data-movement
(memory-bandwidth) probe locally — and, once EDTs and DBs spread across
ranks, a data-*migration* probe instead (Placement).

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `streamArraySize` | total element count, split evenly across threads | 1000 (build's `EXTRA_DEFINES`; source `#define` default is 9000000) | ✓ parsed in `mainEdt` from the argument DB, forwarded through paramv to `mainLet`/`loop`/every kernel — multinode-safe |
| `argv[2]` = `numThreads` | number of independent chains (DAG width) | 2 (build default; source default 32) | ✓ same path |
| `argv[3]` = `nTimes` | iterations per chain (DAG depth) | 3 (build default; source default 1000) | ✓ same path |

If `streamArraySize % numThreads != 0` the run truncates to
`perThreadSize = streamArraySize / numThreads` elements/thread and prints a
warning (the tail elements are simply not represented in any DB). Compile-time
only: `SCALAR` (0.42, the recurrence multiplier), `STREAM_TYPE` (`double`) —
a user cannot select single precision or a different scalar without
recompiling. `mainEdt` also builds an `OCR_HINT_EDT_DISPERSE`/`NEAR` hint and
attaches it to every `mainLet` create; it is not a CLI argument and, per
Placement below, has no effect under the ARTS shim.

## Structure

Let `T = numThreads`, `K = nTimes`, `pts = perThreadSize`. Per thread:
1 `mainLet` EDT, `K` `loop` EDTs (one `FINISH`, the rest plain), and `4·K`
kernel EDTs (`copy`/`scale`/`add`/`triad`, one set per iteration).

| object | count | size |
|--------|-------|------|
| EDTs total | `T·(1 + 5K) + 2` (per-thread chain + `mainEdt` + `finalize`) | — |
| DBs | `T·(1 + 4K)` — one initial `a`, then `copy`/`scale`/`add`/`triad` each create one fresh DB per iteration | `pts·8` bytes each |
| Events | `T·(3 + 4K)` — per thread: the ONCE completion event, the first `loop`'s output event, its finish event (it is the `FINISH` EDT), plus one output event per kernel EDT each iteration (the runtime materializes an event for every `ocrEdtCreate` with a non-NULL output-event slot) | — |
| EDT templates | 7 (`mainLet`, `loop`, `copy`, `scale`, `add`, `triad`, `finalize`), created once, reused for every instance | — |

Worked numbers at the catalog's calibrated args (`1000000 48 750`):
`1000000 % 48 = 16 ≠ 0`, so the run truncates to `pts = 20833`
(999,984 of 1,000,000 elements used) and warns. EDTs total =
`48·(1+3750)+2 = 180,050`; DBs = `48·3001 = 144,048`; events =
`48·3003 = 144,144`. Each DB is
`20833·8 = 166,664` bytes (~162.8 KiB). Because every DB is created, consumed
and destroyed within one iteration (nothing is retained across iterations
except the freshly-churned `a`), the live set per thread is a handful of DBs
at a time (≈4 × 162.8 KiB), not the full `4K` created over the run; `finalize`
briefly holds all 48 threads' final `a[]` at once (≈7.8 MB total).

Counter cross-check: verified (1 node, `1000 4 3` vs `1000 4 5`):
NUM_EDT_CREATE 67/107, NUM_DB_CREATE 53/85, NUM_EVENT_CREATE 60/92 —
exactly `T·(1+5K)+2`, `T·(1+4K)` and `T·(3+4K)` plus the runtime's
constant +1 EDT / +1 DB baseline.

## Wiring

- `mainEdt` creates the 7 templates, one `finalize` EDT (`depc = numThreads`),
  then per thread: one ONCE event `evt_finalize_i` (wired RO into
  `finalize`'s slot `i`) and one `mainLet` EDT carrying that event's GUID plus
  the template GUIDs and sizes through paramv.
- `mainLet` creates the thread's initial `a` (initialized to 2.0, RW-held by
  the creating EDT, no separate release), then one `EDT_PROP_FINISH` `loop`
  EDT (`iter=0`) with `a` wired RW. `mainLet`'s own output event is discarded
  (its creator passes no `outputEvent` slot) — the finish wrapper's only job
  is to fence the whole `K`-deep per-thread subtree into its own OCR scope, so
  `mainLet` itself completes without waiting on it.
- `loop(iter)` creates `copy`/`scale`/`add`/`triad`, wires
  `a`(RO)→`copy`,`scale`; `scale_output`(RO)+`copy_output`(RW, destroy-only)→
  `add`; `add_output`(RO)+`scale_output`(RO)+`a`(RW, destroyed)→`triad`; then
  either creates `loop(iter+1)` with `triad_output` RW, or (last iteration)
  satisfies `evt_finalize_i` with `triad_output`.
- `copy`'s output DB is never read by anything — `add` takes it RW purely to
  authorize `ocrDbDestroy`. The real producer chain is `a →{copy∥scale}→
  add→triad→(new) a`; per-iteration max concurrent readers on any one DB is 2
  (`a` read by `copy` and `scale`; `scale_output` read by `add` and `triad`).
- No DB is ever written by more than one EDT (fresh GUID every write), and no
  DB is shared across threads — the `T` chains touch disjoint DB sets and
  join only at `finalize`. There is no contention point in the concurrency
  sense; the bottleneck is the strictly serial `K`-deep hand-off within each
  thread (see Flow).

## Flow

Per thread, iterations are a serial pipeline of depth `K`: `loop(iter+1)`
cannot even be created until `triad` of iteration `iter` has produced its
output DB, so a thread never has two iterations' kernels in flight at once.
Within one iteration, `copy` and `scale` are parallel (width 2), then `add`
(needs both), then `triad` (needs `add` and `scale`) — width collapses to 1
for the back half of every iteration. Across the `T` independent, unsynchronized
chains, instantaneous parallel width ranges up to `2T` (all chains happening
to be in their copy/scale half simultaneously) down toward `T` (all in
add/triad); realized width is whatever the scheduler achieves given actual
staggering. The `T` chains converge only at the single `finalize` EDT
(`depc = T`), which validates every thread's array and shuts down. There is
no rank-0-only or native-preamble phase inside the DAG itself — `mainEdt`'s
own `preamble()` (banner + size printout) runs once, natively, before any EDT
is created.

## Placement (as-born)

Every `ocrDbCreate` in this source passes `NULL_HINT`, so every DB homes at
its creator's own executing rank (creator/first-touch) — a copy of the fresh
`c`, `b` or `a` lives wherever the kernel that made it happened to run. Every
`ocrEdtCreate` also passes `NULL_HINT` *except* `mainLet`, which is given an
`OCR_HINT_EDT_DISPERSE`/`NEAR` hint — but the shim's affinity extractor only
inspects the `OCR_HINT_EDT_AFFINITY` bit of a hint's propMask, never
`OCR_HINT_EDT_DISPERSE`, so this hint has no effect: `mainLet`, `loop`,
`copy`, `scale`, `add`, `triad` and `finalize` are *all* placed by the same
policy — runtime round-robin (a per-creating-rank atomic counter modulo rank
count).

Consequence at multinode: since each phase's consumer EDT is independently
round-robin-placed, essentially every dependence edge — `a` into `copy`/
`scale`, `scale_output` into `add`/`triad`, `add_output` into `triad`, the
churned `a` into the next `loop` — is a coin flip (probability `(N-1)/N` at
`N` ranks) on being a *remote* acquire of a `pts·8`-byte array. What STREAM
means to measure as local memory bandwidth becomes, at `N>1`, dominated by
RDMA transfers of full per-thread arrays on nearly every phase transition;
the algorithm has no exploitable locality left to lose (there was never any
data reuse to place near), but the as-born program adds gratuitous network
traffic on top of the minimum the algorithm requires. A second, independent
consequence: `finalize` is itself round-robin-placed, so it does not
reliably land on rank 0. Its `printTimes()` reads the file-scope `times`
pointer, which is allocated only on the rank that ran `mainEdt` and left
`NULL` everywhere else (every write is `if(times)`-guarded); a set/diff pair
recorded by a same-rank `loop`/`triad` is meaningful, but `loop` and `triad`
for a given (thread, iteration) are independently round-robin-placed too, so
most cells are never both set and diffed on the same process. The printed
`MB/s` figure should not be trusted at multinode; the `STREAM_RESULT`/
"Solution Validates" scalar the harness reads does not depend on `times` and
is unaffected.

## Sizing

`streamArraySize` and `numThreads` jointly set task grain: `pts =
streamArraySize/numThreads` is both the DB payload (bytes moved per
dependence edge) and the per-kernel work size. `numThreads` alone sets DAG
width (up to `2·numThreads` instantaneous EDTs); `nTimes` sets depth only —
more iterations means more total EDTs/DBs and a longer serial chain per
thread, not more parallelism.

- 1 node × 15 workers: pick `numThreads` a small multiple of 15 (e.g. 30–60)
  so the `2×` copy/scale width comfortably covers the workers; keep `pts` in
  the tens of thousands of elements (tens–hundreds of KiB per DB) so
  per-kernel overhead stays small relative to the memcpy-like body.
- 8 nodes × 120 workers: `numThreads` should scale toward 120–240 to keep the
  DAG at least as wide as the worker count; the calibrated args do not do
  this (see below), which is a known limitation of holding args fixed across
  a strong-scaling sweep, not a bug in this document's guidance.

The calibrated args (`1000000 48 750`) give `pts ≈ 20.8K` elements (~162.8
KiB DBs) — comfortably oversubscribed at 1 node × 15 workers (`2·48 = 96 ≫
15`) but under-filled at 8 nodes × 120 workers (`96 < 120`), and — per
Placement — every one of those 96-wide instants pays a `(N-1)/N` chance of a
remote array transfer once `N>1`. Read this benchmark's multinode numbers as
a placement/coherence stress test, not an achievable-bandwidth measurement;
memory footprint is never the binding constraint (DBs are churned, not
accumulated), so it does not factor into sizing.
