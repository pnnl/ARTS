# stencil1D_sticky

*1-D three-point stencil, cloning style, halo exchange over STICKY events.*
Source: `third_party/ocr-apps/apps/Stencil1D/refactored/ocr/intel-david/stencil1Dsticky.c`
(~650 lines; author David S. Scott, Intel 2015).

## Overview

`N` independent "chains" (`nrank` in the source) each own a contiguous block
of `M` points (`npoints`) and iterate the update
`anew(i) = 0.5*a(i) + 0.25*(a(i-1)+a(i+1))` for `T` timesteps (`maxt`), a
three-point elliptic-solver relaxation. All points start at 0 except the two
global boundary points (leftmost point of chain 0, rightmost point of chain
`N-1`), which are pinned at 1; the field converges slowly toward all-1s. Each
chain needs its neighbors' edge value every iteration, so a scalar "halo"
value is exchanged left/right each timestep.

Because an OCR EDT is single-shot, a chain re-launches itself as a fresh
clone every iteration to regain RW access to its own state and its neighbor
buffers — the "cloning" style shared by all seven Stencil1D variants in this
directory. What is distinctive about **this** variant is the synchronization
idiom used to hand the halo value to the clone that will consume it: a
**STICKY event**, freshly created every iteration and explicitly destroyed
the following iteration once superseded. The README (same directory)
explains why STICKY specifically: the receiving clone is wired to depend on
an event whose GUID it did not choose and whose satisfaction it cannot
order against the dependence-add — a STICKY event tolerates satisfy-before-
add (or after), unlike a one-shot event that must be consumed exactly once
in a fixed order. This is the only one of the seven with a catalog-pinned
correctness `expect` (`1`, at default args — see Verify in the notes file);
the app prints the whole final field itself and terminates by direct
`ocrShutdown()` from the last chain, with no separate join/wrapup EDT.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `nrank` | number of independent chains ("workers") | 4 | ✓ parsed in `mainEdt` via `ocrGetArgc`/`ocrGetArgv`, only when exactly 3 args are given |
| `argv[2]` = `npoints` | points per chain | 10 | ✓ same |
| `argv[3]` = `maxt` | timesteps | 100 | ✓ same |

All three defaults are local literals in `mainEdt` (`nrank=4; npoints=10;
maxt=100;`) — this file does **not** include `stencil1D.h`, so the header's
`N`/`M`/`T` `#define`s (10/50/10000) play no role here (unlike the
labeled/parallel-init variants, see Uncertain). `mainEdt` requires exactly 4
`argc` (program name + 3 args); any other count silently falls back to the
defaults and prints a notice. `nrank==0 || npoints==0 || maxt==0` is rejected
with a message and immediate `ocrShutdown()`. No other runtime knobs exist.
The source also carries an entire `#ifdef PARALLEL` branch (parallel
initialization, labeled GUIDs) that is dead in this build: nothing in the
CMake registration (`add_benchmark_app(NAME stencil1D_sticky ...)`) defines
`PARALLEL`, so only the non-parallel path below is ever compiled.

## Structure

Let `N=nrank`, `M=npoints`, `T=maxt`.

| object | count | size |
|--------|-------|------|
| EDTs total | `N·(T+1) + 2` — `mainEdt`, `realMainEDT`, and `N` chains × `(T+1)` clone generations each (timesteps `0..T` inclusive) | — |
| EDT templates | 1 (`stencilTML`, shared by every clone) | — |
| DBs | `3N-2`, created once by `mainEdt`/`realMainEDT` and **never recreated** — `N` private blocks + `2(N-1)` buffer blocks | private ≈ 64 B (`private_t`: 6×`u32` + 5×`ocrGuid_t`, struct-field estimate, padding not verified) + `M×8` B data; buffer = 16 B (1 `double` + 1 `ocrGuid_t`) |
| Events | `2(N-1)·T` STICKY events created over the run (2 per internal boundary per iteration, deferred by one on the very last handoff) | — |

Worked numbers at the catalog's calibrated args (`48 50 340000`, i.e.
`N=48, M=50, T=340000`): EDTs ≈ `48×340001+2 = 16,320,050`; DBs = `3×48-2 =
142` totaling only ≈23 KB of payload (persistent for the whole run); Events
≈ `2×47×340000 = 31,960,000` STICKY event creates. Every superseded STICKY
event is explicitly `ocrEventDestroy`'d the following iteration, so the
*live* event count at any instant stays `O(N)` even though the cumulative
churn is `O(N·T)` — the same is true of the live DB set (fixed at `3N-2`)
and the live EDT frontier (`O(N)`, one active generation per chain).

Counter cross-check: verified (1 node, `4 4 5` vs `6 4 8`): NUM_EDT_CREATE
27 → 57, NUM_DB_CREATE 11 → 17, NUM_EVENT_CREATE 30 → 80 — exactly
`N(T+1)+2` / `3N-2` / `2(N-1)T` (app values 26/56, 10/16, 30/80) plus the
runtime's constant +1 EDT/+1 DB/+0 EVT per-run baseline.

## Wiring

- `realMainEDT` creates the `N` chain-head `stencilEDT`s, each with 3 slots:
  0 = `leftIn` buffer (RW), 1 = `private` (RW), 2 = `rightIn` buffer (RW).
  Boundary chains get `NULL_GUID` in the missing slot.
- Each `stencilEDT` clone creates its own successor (`stencilGUID`) at the
  end of its body and wires it directly: slot 0 = the STICKY event that will
  carry the *next* left-neighbor value, slot 1 = its own (released) private
  DB, slot 2 = the STICKY event for the next right-neighbor value.
- Neighbor handoff: on send, the chain writes its edge value into the
  buffer DB it currently holds, releases it, creates a fresh STICKY event,
  stores that event's GUID back into the buffer, and satisfies the
  *previous* generation's send event with the (now-released) buffer DB —
  the neighbor's already-wired dependence on that event fires and it
  receives the DB as its `leftIn`/`rightIn` for the next iteration.
- No DB is ever read/written by two chains at once: a buffer DB alternates
  ownership strictly left-then-right-then-left across iterations, so the
  maximum simultaneous accessor count for any DB is 1 (dataflow-serialized).
  Private DBs are never shared at all — one owner for the whole run.
- There is no separate wrapup/join EDT: the last chain (`myrank==nrank-1`)
  prints its final `M` values and calls `ocrShutdown()` directly from inside
  its own final `stencilEDT` invocation; every other chain prints then
  satisfies its right-send event so its neighbor's finalization can proceed
  — print order is enforced purely by this right-to-left dependency chain,
  not by a join.

## Flow

Two serial preambles inside `mainEdt`/`realMainEDT` (rank-0 only, `O(N)`
work): allocating the `3N-2` DBs and creating the `N` chain heads. Then `N`
independent chains run `T` iterations in parallel — parallel width is
`min(N, workers×nodes)` since each chain's `T` iterations are a strictly
serial dependency chain (a clone cannot start before its predecessor
finishes and hands off both its private DB and the fresh halo events). The
finalization is a *serialized wave*, not a join: at `timestep==maxt`, chain
`myrank` prints, then (if not last) unblocks chain `myrank+1`'s
finalization by satisfying its right-send event — an `O(N)`-deep purely
serial tail after the `T`-deep parallel body, so total critical-path depth
is `T + N`, dominated by `T` at any realistic size.

## Placement (base)

The source passes `NULL_HINT` on every `ocrEdtCreate`/`ocrDbCreate` call —
no affinity code exists outside the dead `#ifdef PARALLEL` block. Effective
policy:

- **EDTs**: NULL hint → shim's `ARTS_HINT_ANY_RANK` → runtime round-robin
  (per-creating-rank atomic counter, modulo rank count). `realMainEDT`, all
  `N` chain heads, and every subsequent clone are round-robin scattered
  independently — a clone's rank has no relationship to its predecessor's.
- **DBs**: NULL hint → home = creating rank. All `3N-2` DBs are created
  inside `mainEdt`/`realMainEDT`, which always run on rank 0 — so every
  private and buffer DB is homed at rank 0 for the entire run.

Consequence at multinode: essentially every RW acquire a clone performs (its
own private DB, plus the two halo buffers) is a remote round-trip back to
rank 0, since the executing clone is round-robin-scattered while the DBs it
touches are permanently pinned there. Locality (each chain's data belongs
together, and neighbor chains are cheap to co-locate) exists in the
algorithm but the base program expresses none of it — this is a
worst-case fine-grain coherence stress by construction, structurally
identical in spirit to fibonacci's remote-4-byte-block pattern but shaped as
a long, narrow (`N`-wide) pipeline instead of a wide recursive tree.

## Sizing

`N`, `M`, `T` independently move three different things: `N` sets
parallelism (max concurrent chains), `M` sets per-clone compute (and DB
payload) with no effect on object counts, `T` sets the length of each
chain's serial dependency and dominates total EDT/event churn.

- Parallel width is capped at `N` — `T` and `M` do not add parallelism, only
  depth and per-task work. Pick `N` ≳ `nodes×workers` so every worker has a
  live chain; the calibrated `N=48` covers up to 3 workers/node at 1 node
  (15 workers) or fits within a single node's worker count at any of
  {1,2,4,8} nodes without idling threads, since only `N` chains are ever
  runnable in parallel regardless of node count.
- `T=340000` is what actually drives runtime and churn (`O(N·T)` EDTs and
  events) — pick it so the run takes a useful number of seconds at the
  target width: a 1-node×15-worker run processes the 48-chain pipeline with
  ~3 chains per worker; wall time ≈ `T` × (per-iteration cost, dominated at
  multinode by the remote-rank-0 round-trips from Placement above, not by
  worker count).
- `M` only changes memory-per-DB and per-iteration compute; it does not
  change the EDT/DB/event counts at all, so it is the cheapest knob to grow
  if only wall time (not churn) needs adjusting up or down.
- At 8 nodes × 15 workers (120 workers total) the same `N=48` chains
  under-fill the machine (48 < 120) — `N` would need to grow for that
  profile to have every worker holding a live chain, but the calibrated
  args keep `N` fixed at 48 across the whole node sweep (strong scaling),
  so past 3-4 nodes this app is chain-count-bound rather than worker-bound
  by design, and its interesting signal is the remote-rank-0 traffic from
  Placement, not raw parallel occupancy.
