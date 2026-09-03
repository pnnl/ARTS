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

An HPX port (`benchmarks/hpx/p2p.cpp`) mirrors the base tier as `p2p_hpx`: one
persistent rank object per logical rank on locality `i / ⌈p / L⌉` (the
program's own BLOCK map), each generation a continuation on the left rank's
pushed boundary (receiver-owned, keyed by the consumer's generation), the chain
tail-posted to the same locality; rank `p−1`'s terminal generation delivers the
checksum to locality 0.  Structural references at the calibrated
`6912 1347840 6913 32`: `tasks = p·G = 1,576,599,552`,
`sends = (p−1)·G + t = 1,576,371,488`,
`bytes = 16·(p−1)·G + 8·t = 25,221,943,552` (every boundary carries
`gf + 1 = 2` doubles; only the `t` wrap-around sends carry one).

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
load-bearing base locality, by design. The source carries no
`OCR_APP_OPTIMIZED_PLACEMENT` guard at all (the consumer-home layer that was
tried is gone — see the next section), so `bufferOutDBK` homes at the
sender (creator/first-touch) and the receiving rank's acquire is always a
one-hop remote fetch from its immediate left neighbor — exactly the
point-to-point traffic the benchmark is named for.

## Sizing

`p` is the number of virtual ranks the program decomposes into.  It is an
application-level knob with no relation to the worker threads `arts.cfg` starts,
so it is free to choose; what constrains it is memory, not a rule.

The wavefront's width is `min(p, n-1)`: rank `i` phase `j` waits on rank `i-1`
phase `j` and on its own phase `j-1`, so the runnable set is the diagonal and
**both** `p` and the row count cap it.  The catalog had this wrong twice --
first at `3456 8192 50 2100` (width 49) and then at `3456 1347840 500 515`,
where `p` was read as "the pipeline's width" while the 500 rows held the real
width to 499 against 3456 workers.

Averaged over a sweep the width is `p(n-1)/(p+n-2)`, because half of every sweep
is the pipeline filling and draining.  `p = n-1` is where the two caps balance,
and for a given array it buys the most width.  At `p = 6912` -- twice the
largest geometry's 3456 workers -- with `n = p+1`:

| | |
|---|---|
| peak width `min(p, n-1)` | 6912 = **2.00x** the workers |
| average width `p(n-1)/(p+n-2)` | 3456 = **1.00x** the workers |
| columns a rank `m/p` | 195 |
| resident | 159 GB at the one-node anchor |

so the bound is met across the whole sweep, fill and drain included, rather than
touched at the peak alone.

Columns stay at the application's own `m = 1,347,840`, so raising `p` costs the
array nothing -- it only divides it into more, smaller slices.  What raising `p`
*does* cost is live boundary blocks: the channel's `maxGen` is `n`, so a producer
may run a full sweep ahead and its unconsumed 16-byte blocks accumulate, one
runtime object each.  That term, not the array, is what puts `p = 6912` with
`n = 10369` at 271 GB, past the 256 GB ceiling, while the same `p` at `n = 6913`
holds 159 GB.

`gf` stays 1: the README calls any other value cheating and negates the flop
rate the program reports.

This is the one strong scaler of its cycle, and it does not run out on the way to
eight nodes -- the gain per doubling grows rather than fades.  Measured at the
arguments the campaign itself runs, so there is no short-run proxy to correct
for (two runs a cell, all eight giving the same checksum):

| geometry | time | cumulative | per doubling |
|---|---|---|---|
| 1 node x 15 workers | 389.8 / 390.4 s | 1.00x | -- |
| 2 nodes | 331.4 / 336.8 s | 1.17x | 1.17x |
| 4 nodes | 202.0 / 200.9 s | 1.94x | 1.66x |
| 8 nodes | 111.3 / 111.6 s | **3.50x** | **1.81x** |

which is what a pipeline does as it widens: the fill is a fixed number of phases,
so it costs a smaller share of a shorter run.  From two nodes to eight that is
3.00x on 4x the workers, 75% efficiency; the residual is a per-task cost that
grows with the node count and has not been isolated to a mechanism -- it is not
the boundary crossings, which stay under one percent of handoffs at every
geometry, and not the fill, which is a node-count-independent constant.

So the row is calibrated against ~150 s rather than 10-30 s, and the iteration
count reaches it at `t = 32`: 146.2 s, 145.3 s and 152.8 s on the three coherence
families.

A short run is not a smaller version of this one.  Splitting end-to-end time into
a fixed and a per-timestep part put setup at more than half the run at two
timesteps, and setup is memory-bandwidth bound, so it scales *better* than the
wavefront and inflates any trend read there.  That is why the table above is
measured at `t = 32` and not at a cheaper length.

There is no restructured tier either, and the reason is the same measurement: a
row that scales does not need one.  The structure a restructure would attack is
the boundary handoff -- one block of 16 bytes per phase per rank -- and the knob
that batches them exists already: the group factor widens a block to `(gf+1)*8`
bytes and divides the phase count by `gf`.  The PRK README calls a group factor
other than 1 cheating, so the one aggregation available here is the one the
benchmark forbids.

## Placement (hinted)

There is no hinted tier, and no placement guard in the source.  The layer that
was tried homed each per-phase boundary block at its one consumer, and it was
measured and **discarded**: 0.87x at two nodes over five interleaved runs a
side, 0.99x at four, 0.97x at eight.

The reason is that the application had already claimed the locality.  `p2p.c`
carries an unguarded `#define BLOCK`, so ranks go to policy domains in
contiguous runs rather than the strided map the `#else` branch would give.  That
choice is worth measuring: forcing the strided map costs **11x** at one rank per
worker (428 s against 37.7 s).  With the contiguous map only one handoff per
domain crosses a boundary at all.

A consumer home therefore helps only that handful of handoffs while being paid
on every phase of every rank.  Holding the map fixed and varying only the home,
at one rank per worker and fifteen ranks a domain:

| map | producer home | consumer home |
|---|---|---|
| strided | 428.3 / 431.0 s | 303.1 / 305.8 s |
| contiguous (base) | 37.7 / 38.3 s | 30.9 / 31.5 s |

so the home is worth 1.22x-1.41x *there*, where one handoff in fifteen crosses.
At the catalog's shape far fewer do, and the balance goes the other way -- which
is what the two-node cell measures.

The resident blocks are not a placement surface either: homing `data` and
`private` explicitly measures the same as leaving them to first touch, because
the rank's own init task is what creates them.  A rank's EDT chain is already
pinned to its domain by the application's `AFFINITY` define.  Nothing is left, so
the source carries no guard and the build produces no hinted target.
