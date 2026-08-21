# p2p

*PRK synch_p2p — a P-wide column-partitioned wavefront pipeline built for one
purpose: measuring per-hop synchronization latency, not bandwidth.*
Source: `third_party/ocr-apps/apps/p2p/refactored/ocr/intel/p2p.c` (~730 lines).

## Overview

p2p is the OCR/ARTS port of the Parallel Research Kernels `synch_p2p`
benchmark: a `p`-way column-partitioned pipeline computing the 2D recurrence
`ARRAY(i,j) = ARRAY(i-1,j) + ARRAY(i,j-1) - ARRAY(i-1,j-1)` over an `n`-row ×
`m`-column grid, where each of `p` logical "ranks" owns a contiguous stripe
of `~m/p` columns and needs its left neighbor's rightmost boundary column
before it can compute its own first column of a row-block. This wires the `p`
logical ranks into a closed pipeline/ring (rank `p-1` wraps back to rank 0),
each one a single persistent EDT chain — one `p2pEdt` self-recreation per
(timestep, phase) generation — that never migrates once born. The result
scalar checks the last cell (`ARRAY(n-1,k-1)`) against the closed form
`(t+1)·(n+m-2)`; timing is also printed (rank 0 stamps the start, rank `p-1`
the end). Because every hot-path datablock is either exclusively private to
one rank's chain or a single-producer/single-consumer 1:1 handoff — never
shared by more than two ranks, never read by more than one consumer — the app
has essentially no coherence contention; any slowdown is per-hop
create/acquire/wire latency along a long dependency chain, which is exactly
what this kernel is built to probe.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `p` | number of logical p2p "ranks" (independent of the actual ARTS node/worker count) | 10 | ✓ parsed in `mainEdt`, carried via `realMainEdt`'s paramv to every `initEdt` — multinode-safe |
| `argv[2]` = `m` | total columns, split ~evenly across `p` ranks | 100 | ✓ same path |
| `argv[3]` = `n` | rows per rank | 1000 | ✓ same path |
| `argv[4]` = `t` | timesteps (`t+1` total generations counting one untimed warm-up timestep) | 100 | ✓ same path |
| `argv[5]` = `gf` | group factor: rows processed per phase before a boundary send | 1 | ✓ optional 5th arg; same path |
| `BLOCK` / `AFFINITY` | compile-time `#define`s (unguarded, always on base): `BLOCK` = contiguous-block PD assignment for the `p` ranks (vs `WRAP` = round-robin); `AFFINITY` = actually apply the computed `EDT_AFFINITY` hints (vs `NULL_HINT`, which the source's own comment says gives "bad performance regardless") | both defined | ✗ compile-time — unlike fibonacci/fft/triangle, this app ships its own real base placement logic outside the hinted-placement guard |

`argc` must be exactly 1, 5, or 6 (program name plus 0, 4, or 5 args); any
other count is a hard `bomb()`/shutdown.

## Structure

Let `w = ceil((n-1)/gf)` (phases per timestep) and `G = (t+1)·w` (generations
per rank, including the untimed timestep 0).

| object | count | size |
|--------|-------|------|
| `p2pEdt` (chain generations) | `p·G` | — |
| `initEdt` / `initp2pEdt` / `initChannelEdt` | `p` each | — |
| `mainEdt` / `realMainEdt` / `p2pShutdownEdt` | 1 each | — |
| EDTs total | `3 + 3p + p·G` | — |
| `dataDBK` (per-rank working array) | `p`, persistent, migrates generation-to-generation, never touched by another rank | `n·(⌈m/p⌉+1)·8` bytes each |
| `privateDBK` (per-rank state) | `p`, same persistence pattern | `sizeof(private_t)` (~150-250 B) each |
| `sharedDBK` / `timerDBK` / `channelDBK` (×p) | 1 / 1 / `p` | small, ≤ a few hundred bytes |
| `bufferOutDBK` (boundary handoff) | `(p-1)·G + t` (every non-last-rank generation, plus one wrap-around send per completed timestep from rank `p-1` — `t`, not `t+1`: the final timestep's wrap-around is skipped because the termination check (`p2p.c:213`) returns before the wrap-around create (`p2p.c:236`) is ever reached) | `(gf+1)·8` bytes — 16 B at `gf=1` |
| labeled STICKY rendezvous (`sendRightEVT`/`recvLeftEVT`) | `p` distinct objects, `2p` `ocrEventCreate` calls — `GUID_PROP_CHECK` makes the loser of each racing pair an idempotent *install*, but the shim still increments the create counter for the loser too (`arts_event_create`'s `INCREMENT_NUM_EVENT_CREATE_BY(1)` runs before the install-or-fail check), so all `2p` calls count | — |
| CHANNEL events (per-rank boundary transport) | `p`, one per rank, satisfied once per generation except the last | — |
| shim-materialized output event | `1`, constant — the one `p2pEdt` create that spawns rank `p-1`'s terminal instance passes a non-NULL `outputEvent` (`termOETp`, `p2p.c:251-254`) so the shim materializes an event feeding `p2pShutdownEdt`; every other `p2pEdt`/`initEdt`/`initp2pEdt`/`initChannelEdt` create passes `NULL` for it | — |

Events total: `3p + 1`. Worked numbers at calibrated `args = [128, 256, 100,
1400]` (`p=128, m=256, n=100, t=1400, gf=1`): `w=99`, `G=138,699`, `p2pEdt`
total ≈ 17,753,472, EDTs total ≈ 17,753,859, `bufferOutDBK` = 17,616,173, all
other DBs = `3p+2` = 386 (total DBs = 17,616,559), events = `3·128+1` = 385.

Counter cross-check: verified (1 node, `args=[2,4,5,2]` vs `args=[2,4,5,3]`,
i.e. `p=2, m=4, n=5, gf=1, t=2` vs `t=3`, so `w=4`, `G=12` vs `G=16`): measured
absolutes EDT 34/42, DB 23/28, EVT 7/7; subtracting the runtime's constant
baseline (+1 EDT, +1 DB, +0 EVT per run) gives app-side EDT 33/41 (exactly
`3+3p+p·G`) and EVT 7/7 (exactly `3p+1` at `p=2`, unaffected by `t`). The
corrected DB formula — `bufferOutDBK = (p-1)·G+t` (not the earlier `+(t+1)`)
— gives `bufferOutDBK` = `1·12+2=14` and `1·16+3=19`; adding the constant
`3p+2=8` gives app-side DB 22/27, exactly reproducing the measured totals
(23/28) once the runtime's +1 DB baseline is added back.

## Wiring

p2p's DB graph is almost entirely private per rank: `dataDBK` and
`privateDBK` are each acquired RW by exactly one generation of exactly one
rank's chain at a time and never touched by any other rank — no fan-out, no
contention. The only cross-rank object is `bufferOutDBK`: created fresh every
generation by the sender, released, delivered via that rank's persistent
CHANNEL event, and acquired+destroyed exactly once by the immediate right
neighbor's next generation — strict single-producer/single-consumer, at most
1 concurrent accessor. `sharedDBK` fans out RO to all `p` `initEdt` instances
once at startup only. The one-time labeled-STICKY rendezvous
(`ocrGuidFromIndex` into a pre-reserved GUID range) exists purely so each rank
can learn its left neighbor's CHANNEL guid without a central directory — it
fires once per rank, not per generation. `timerDBK`'s guid is broadcast to
every rank's private state but is only actually acquired twice in the whole
run (rank 0 at timestep 1, rank `p-1` at the terminal generation).

## Flow

This is a diagonal wavefront/systolic pipeline, not a width-1 serial chain:
rank `i`'s generation `N` depends on rank `i-1`'s generation `N` (the same
(timestep, phase) index, paired FIFO through the CHANNEL), and each rank's
own generations execute strictly one after another (a single persistent
chain). Once filled (after the first `p` hops), essentially all `p` ranks are
concurrently active, each on a different generation index — parallel width
approaches `p`, not 1. What makes it a *latency* probe is the length of the
critical path, not its narrowness: completion requires `p + G - 1` sequential
hops end-to-end, and since every hop's payload is 16 bytes, wall time is
dominated by per-hop create/acquire/wire round-trip cost rather than
bandwidth or compute — the `Rate (MFlops/s)` the program prints is a derived,
secondary number. Termination is a single rank-`p-1` special case (the last
generation returns without cloning; a dedicated `p2pShutdownEdt` waits on
that generation's own output event, which fires only after its dependence
releases complete, so print/shutdown never truncates the measured run).

## Placement (base)

Unlike fibonacci/fft/triangle, p2p's placement is *not* NULL_HINT-throughout
base: `BLOCK`/`AFFINITY` are plain, unguarded `#define`s (not gated by
`OCR_APP_OPTIMIZED_PLACEMENT`), so `realMainEdt` explicitly computes a BLOCK
partition (`myPD = i / block`) and creates each rank's `initEdt` with an
explicit `EDT_AFFINITY` hint pinning it there; `initEdt`/`initp2pEdt`
propagate `ocrAffinityGetCurrent()` downward so each rank's *entire* chain
(all `G` generations) stays pinned to the PD it was born on — real,
load-bearing base locality, by design. The one genuinely
`OCR_APP_OPTIMIZED_PLACEMENT`-gated piece is `p2pBufHint` (home the boundary
DB at its consumer instead of its creator); base it returns `NULL_HINT`,
so `bufferOutDBK` homes at the sender (creator/first-touch) and the receiving
rank's acquire is always a one-hop remote fetch from its immediate left
neighbor — exactly the point-to-point traffic the benchmark is named for.

## Placement (hinted)

As-born already ships real placement (unguarded `BLOCK`/`AFFINITY` defines pin
each rank's chain EDT to its PD; see Parameters).  What it does not place is
the 16-byte boundary block each rank mints per generation for its right
neighbour — `NULL_HINT` homes it on the CREATOR, so every consumer acquire
pays a remote directory round on the critical path of the chain.

The layer adds exactly one thing: a consumer-home `OCR_HINT_DB_AFFINITY` for
those boundary blocks (`p2pBufHint`).  Each rank has a single fixed consumer
(myRank+1, wrapping to 0), so the hint is invariant and computed once in
`initp2pEdt`, mirroring realMain's BLOCK/WRAP rank->PD map.  This is the
measured one-shot/high-frequency/small-DB exception to the EDT-only rule
(2026-07-10 A/B: consumer-home recovered 2n 7.4x / 4n 6.3x, and consumer beats
producer-home by ~35%).

## Sizing

`p` sets the pipeline's parallel width and its critical-path floor (`~p`
warm-up hops); `t` and `w` (driven by `n`, `gf`) set the steady-state length.
To saturate `W` workers, want `p` close to or above `W` (`p ≈ W` keeps
roughly one rank per worker at steady state) — the calibrated `p=128`
slightly oversubscribes even the 8-node×15-worker profile (120 workers), by
design (BLOCK placement then packs more than one rank onto some PDs). `m`
(total columns) mostly sets per-rank grain (`k ≈ m/p`) and is cheap to move;
`n`, `t`, `gf` set generation count and thus wall time roughly linearly
(`G = (t+1)·⌈(n-1)/gf⌉`) — `gf=1` maximizes hop count (and thus the latency
signal), so it should not be raised except to deliberately trade fewer,
larger messages for less synchronization overhead. 1 node × 15 workers:
`p ≈ 15`, keep `t`/`n` small for a quick run. 8 nodes × 120 workers: `p ≈
120-128` (matches calibrated), `t=1400` gives a multi-second-to-minutes run.
Memory is negligible throughout — every DB is at most a few hundred bytes per
rank, and `bufferOutDBK` instances are transient and tiny.
