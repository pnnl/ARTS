# npb_cg_dist

*The restructured version of `npb_cg`: the same NPB CG power iteration on a
**row-band SPMD decomposition** — one persistent chain per rank, a
channel-event fragment exchange in place of the whole-vector broadcast, and
library allreduces in place of single-EDT dot products.*
Source: `third_party/ocr-apps/apps/npb-cg/sdsc-ocr/` — a separate program
(`cg_dist_edt.c` + `cg_dist.h` + `cg_dist_makea.c`, ~1300 lines) sharing the
base port's random stream and utilities (`makea_ocr.c`, `util_ocr.c`) but
not its matrix construction.  Selected as `npb_cg:restructured`.

## Overview

The base program anti-scales for four structural reasons no hint reaches:
every matvec broadcasts a *fresh* whole vector to `na/blk` readers, gathers
`na/blk` result fragments back into one EDT, runs its dot products and
axpys as single whole-vector EDTs on the critical path, and pays task
overhead per a few thousand flops of grain.  The rewrite dissolves all
four at once.  Rank `i` owns rows `[i·na/R, (i+1)·na/R)`: its band of the
matrix, its fragments of every vector (`x z r p q` packed in one per-rank
datablock), and a persistent chain of EDTs pinned to its own rank that
never hands a vector datablock to anyone.  Per matvec, each rank publishes
one immutable `nloc`-long fragment per peer over persistent channel events
— written once, read once, destroyed — and the two CG reductions travel as
3-double scalars through the reduction library's allreduce.  The chain
fuses the vector arithmetic into the same EDTs that consume the reduction
results, so an inner iteration's control plane is four EDTs per rank
(bcast → spmv → join → alpha, plus beta on the continuing path) regardless
of problem size.

**Within** a rank the product is a fan-out, not a single task: the band is
cut into row slices (`CG_DIST_ROWS_PER_TASK` rows each, capped at
`CG_DIST_MAX_CHUNKS`), one task per slice writes its own rows of `q` and
returns the partial dots those rows contribute, and a join sums them before
launching the allreduce.  Without it a rank's chain occupies exactly one
worker — the whole point of a 108-worker node is lost, and the run gets
*slower* as the node gets wider.  The slices are also where the band lives:
each slice datablock is created by the task that builds it, so the band is
first touched across the rank's workers instead of being written by one
worker and then read by all of them through that worker's memory
controller.

**The matrix is built where it lives, in parallel.**  The published
generator is one serial sweep: for every source row it scatters a row of
contributions into whatever destination rows that row's entries name,
inserting each into a sorted-by-column list.  It cannot be split by source
row, because two sources write the same destination — which is why the
serial form looks inherent, and why it was this program's largest floor.
Inverting the pairs removes it.  For every DESTINATION row, list the
`(source row, entry)` pairs that name it; building that index is one pass
over the `na·(nonzer+1)` generated entries, not over the
`na·(nonzer+1)²` contributions.  Afterwards each destination row is built
alone, by exactly the insertions the serial sweep would have performed on
it and in the same order, so there is no exchange, no bucketing, no sort,
and the matrix comes out **bit identical** to the published one.  Each row
slice is therefore constructed by the task that will multiply it: the band
is never assembled anywhere else, nothing is copied to a rank, and the
slices are first touched across the rank's workers.

What stays serial is what cannot be anything else: the draw stream
(`sprnvc` rejects on both range and duplicate, so a row's draw count is
data-dependent and no jump-ahead exists) and the running scale factor (a
product accumulated in row order).  Nothing is generated centrally and
shipped, though — as in the reference MPI implementation, **every rank
replays the same stream itself** and indexes only the rows it owns.  So the
replay costs one rank's time however many ranks there are, the construction
inputs never cross the wire, and they are released as soon as the build
that consumed them finishes.  Measured at the anchor node: 0.756 s of draws
at class D, 0.072 s at class C.

Built-in run discipline: an untimed warm-up pass precedes the measured
passes; the reporting rank's shutdown EDT is gated on the final EDT's
output event (releases complete before shutdown); and a last collective
holds every rank in the run until the report, so no rank's chain is still
executing when the runtime comes down.

## Parameters

`-t` class, `-i` iteration override, `-c` row slices per rank (0 = derive).
`-b` is **not** accepted: nothing in this program blocks the matrix, so a
blocking dial would be a knob with no effect.  Loud-fails: at most
`CG_DIST_MAX_RANKS` (64) ranks, and `na >= nrank`.

## Wiring

Each rank's chain hands four datablocks RW from link to link — the private
state, the packed vector block, the timer (reporting rank only) and the
reduction-library private block — so they live on their creating rank
forever.  Channel GUIDs are exchanged once at setup through a labeled
sticky range (`nrank²` events, racing creators legal); fragments ride the
channels as fresh datablocks with `maxGen 8` headroom, and the per-
iteration allreduce bounds chain skew, so a generation can never be
overrun.  The final pass's residual matvec reuses the same bcast/spmv
links with the `residual` flag switching the operand from `p` to `z` and
the successor from `alpha` to the outer-iteration EDT.

## Structure

Per rank per inner iteration: 1 bcast + 1 spmv + `nchunk` slice tasks +
1 join + 1 alpha (+1 beta except on the last); `R−1` fragment datablocks
created and destroyed; `nchunk` partial datablocks created and destroyed;
one allreduce of 3 doubles.  `nchunk = min(256, ceil(nloc/16))`, so a rank
always offers its workers more tasks than they have hands.  Per outer iteration: 25 inner
iterations, one residual matvec, one outer EDT.  Setup: `R` rank-init EDTs,
`R·nchunk` row builders, `R` slice joins, `R` channel-init EDTs.  Events:
persistent channels (`2·R·(R−1)` total) plus the one-time labeled
stickies; nothing grows with the iteration count.

## What still bounds it

- **The draw stream — the benchmark's own Amdahl term, and worth reading off
  the curve rather than hiding.** `sprnvc` rejects both out-of-range draws
  and duplicates, so a row's draw count is data-dependent and no jump-ahead
  into the LCG exists; drawing in parallel would consume the stream in a
  different order and produce a different matrix, i.e. a different answer.
  It is therefore replayed whole on every rank and does not shrink with the
  node count: **0.756 s at class D** (0.072 s at class C), against a
  32-node cell of roughly 16 s — about 5%, and flat.  Everything else in
  init does scale: the capacity and index passes are 0.93 s at class D and
  the assembly folds from 566 s serial to 10.2 s over one node's workers and
  0.32 s over 32 nodes'.  All timings here come from a tree built with every
  counter OFF; the earlier ones carried a previous campaign's `attribution`
  set and read up to 16% differently.
- **Two collectives per inner iteration.** That is conjugate gradient, not
  this port: `p·q` must be reduced before `α`, and `r·r` before `β`.
  Removing one needs a different CG variant (Chronopoulos–Gear), i.e. a
  change of numerics rather than of decomposition.
- **The product is gather-latency bound, not bandwidth bound** — which is
  what NPB CG is *for*.  Widening a node buys memory-level parallelism, not
  bandwidth: at class C one NUMA node's 15 workers **beat** all 108
  (12.7 s vs 16.8 s of solve for the same 40 iterations), and at class D
  7.2× the workers buy 1.76× (69.9 s vs 39.6 s at `-i 5`, ~44 GB/s of
  matrix stream).  Page
  placement is not the reason — `numactl --interleave=all` moves it 2–4%.
  Cache-blocking the gather would change the summation order, i.e. measure
  a different benchmark.
- **~5% serial work inside each chain link** — publishing `R−1` fragments,
  assembling the `na`-long operand (every rank needs the whole vector,
  because NPB CG's sparsity is random), and the two `nloc` axpys.  It is
  fan-out-able but not currently binding, so it is left simple.
- **The class ladder has no rung at the anchor's budget.** Measured at
  dane1: class C 32.6 s, class D 471 s, class E over 19 h.  Class E is a
  cliff rather than a cost — its operand vector is 72 MB, so every gather
  is a DRAM access plus a TLB page walk (class C's is 1.2 MB, class D's
  12 MB), and one outer iteration alone takes over 14 minutes.  `-i`
  cannot bridge the gap either: class D reaches the benchmark's own 1e-8
  verification bar only around iteration 85 of its 100, so the published
  count is barely padded.  Class D is therefore the largest class that
  runs, and it is what the catalog pins.
