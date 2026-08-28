# cholesky_dist

*The restructured version of `cholesky_blas`: the same factorisation, the same
2-D block-cyclic ownership and the same BLAS3 kernels, with each place building
its own tasks where they run.*
Source: `third_party/ocr-apps/apps/cholesky/ocr-mkl/cholesky_dist.c` (~430 lines).

## Overview

The row this re-implements builds its whole DAG in `mainEdt` -- about `t^3/6`
task creations on one rank -- and then relies on placement to keep each kernel
near the tile it writes.  The placement is already the canonical one and cannot
be improved on: it is ScaLAPACK's 2-D block-cyclic map, every kernel sits on
the coordinate of the tile it updates, and first touch carries the tile's home
with it.  Measured at eight nodes it cuts remote acquires sevenfold.  The row
still degrades 3.24x across the node sweep with that layer on.

What this version changes is where the graph is built.  `mainEdt` does O(P)
work; each place creates the tasks for the tiles it owns, on itself.  The
ownership map, the kernels and the answer are untouched.

## Parameters

`cholesky_dist --ds <n> --ts <n> --fib <tile-stream> [--places <n>]`.
The catalog runs `--ds 90000 --ts 500 --places 32`.

`ts` is the tile edge and `t = ds/ts` the tiles per dimension.  `--fib` is the
tile-stream binary, the same layout the base row reads -- lower-triangular
tiles in `(i, j<=i)` order, `ts^2` doubles each.

`places` is the ownership decomposition and is an **argument, not the rank
count**: the task count, the datablock count and the order the factor is
computed in are identical in every geometry, and the machine enters only
through the map from a place onto a rank.  Verified by running it: the answer
does not move over places 1, 2, 4, 8, 16, 32 and 64, nor over one, two, four
and eight nodes -- checked on the built binary, not inferred.

`ts` is also what decides whether a shrunk version of this row is the same
program.  A tile costs `ts^3` of arithmetic and `ts^2` of movement, so compute
per byte moved IS `ts`; a trend taken at a fifth the tile is a fifth the
arithmetic intensity and reverses the verdict.  The trend below therefore holds
`ts` at the catalog value and shrinks `ds` alone.

## Structure

`mainEdt` creates the versioned tile-event grid, one `placeInit` per place, and
the finisher.  Everything else is built by `placeInit` on the place that will
run it.

A tile is rewritten at every step below its column, so the events carrying it
are **versioned**: `tileEvt[(i,j), k]` is tile (i,j) after step k.  One event
per tile would be satisfied once per step, which is not a thing an event does.
Tile (i,j) exists in versions `0..j+1` and no further, and the grid is indexed
accordingly -- numbering every tile up to the last step instead would spend two
thirds of it on names nothing ever satisfies or reads (1.94 M of 2.95 M at the
catalog decomposition).

The events are sticky, and that is the flavour the dependence pattern needs
rather than a fallback: a place satisfies a tile's first version before it has
created the tasks that read it, and the producer of a later version is built by
one place while its consumers are built by another, concurrently. Binding after
the satisfy is therefore normal here, which a once-event does not survive. A
counted event would fit too -- the consumer count is exact and derivable -- and
would reclaim the grid as the factorisation walks it, but the flavour is not
portable across the runtimes this row is compared on, and what it saves is
small against a working set that is the matrix. Binding before satisfying
everywhere would need a startup barrier gating every one of the `t^3/6` tasks,
to save a grid that is well under one percent of the residency.

Per step k, on each place, for the tiles it owns:

- **POTRF** on (k,k): version k becomes k+1.
- **TRSM** on (j,k) for j>k: reads the factored (k,k), produces the panel.
- **the trailing update**, one task per tile -- SYRK on (j,j), GEMM on (j,i) --
  each reading its own two panel tiles.

Each place reads **its own tiles** from the file: the stream is fixed-size
records in a regular order, so a tile's offset is arithmetic and a place seeks
to the ones it owns.  Nothing streams the matrix through one rank.

There is no per-place panel gather.  One was built and measured: it loses at
every geometry that scales, because the copy costs more than the coalescing
saves and because the gather is a barrier, making all of a place's updates at a
step wait on the whole panel instead of on the two tiles they read.

## Wiring

A tile is one block of `ts*ts` doubles throughout; nothing is packed, gathered
or copied between places, so there is no transfer format to describe. What a
task receives is the tile it writes at slot 0 `DB_MODE_RW` and the one or two
panel tiles it reads at slots 1 and 2 `DB_MODE_RO`, and its `paramv` carries the
name of the event to satisfy with the block it wrote -- release first, then
satisfy, so nothing is exposed while the writer still holds it.

The event grid and the input path are the only objects `mainEdt` hands out, and
every place reads both `DB_MODE_RO`.

Reclamation splits by object kind.  Every **event** carries the exact number of
tasks that will name it and is reclaimed when the last of them has, so the grid
does not accumulate across the run.  The **blocks** cannot be treated the same
way: the finisher destroys the diagonal tiles, which it is the last reader of
by construction, but the off-diagonal tiles are left.  A tile's final version is
what the trailing updates of its own column read, so a reaper hung on it would
be a destroy with no ordering edge against those readers; freeing them needs a
per-tile last-reader count the DAG does not carry, and one block per tile is the
program's own working set either way.  What is left at shutdown is the matrix,
not an accumulation.

## Flow

Tiles are read and satisfied by their owners.  Each step's POTRF fires on the
diagonal tile's current version, the TRSMs on that and their own, and the
updates on their tile and the two panel tiles they read.  The finisher takes
every diagonal tile at its final version, prints the trace and shuts down.

## Placement (base)

The tiles are owned by places on ScaLAPACK's two-dimensional block-cyclic map,
and a place is then mapped onto a rank by folding **both** axes of the place
grid onto a near-square rank grid.  That second step is the one that is easy to
get wrong: a linear `place * nranks / places` keeps only the axis the place
index varies slowest against, so the column coordinate never reaches the rank
and the distribution is one-dimensional exactly where messages are generated --
the whole property the two-dimensional map exists to provide, lost at the last
step.

What that costs is a question of how much data crosses, so it shows where the
data is.  At the catalog arguments over eight nodes, folding both axes runs
**1.22x** faster (127.8 s against 104.7 s, two replicates each whose own spread
is 0.5%) and holds **a third less** resident memory -- 69 GB a rank against 47.
At the trend size the same comparison shows no difference in time at all (16.85
against 16.93 at four nodes, 13.48 against 13.73 at eight): a fifth the data
puts the run against this host's aggregate arithmetic ceiling rather than
against its network, and a placement map cannot be seen from there.  The
residency still separates -- 57 GB against 73 at four nodes -- because that is
set by how many ranks a panel reaches, not by how long it takes to get there.

Everything a place creates -- tasks and blocks alike, including the tiles it
reads from the input -- carries its rank as a hint, so a tile is first-touched
on the place that owns it.  That is the only thing the program asks the
machine, and it asks it for a hint: the tile partition above it is fixed by
argument, so where a place lands changes nothing about what the program
computes.

There is no separate `hinted` version.  The block-cyclic map is already the
canonical placement for this factorisation and the base row carries it as its
optimised layer; here it is structural rather than optional.

## Correctness

The identity input has a unit factor, so the trace is exactly `ds` and the
expected value is analytic.  The trace alone inspects only the diagonal, so the
factor is checked two further ways on non-trivial SPD inputs (three of them:
`1/(1+|i-j|)`, a cosine band, and a modular pattern, each diagonally dominant).

An independent sequential Cholesky factors the same tile stream, and
`cholesky_blas` is made to write its factor: **all 500,500 elements agree to
1.5e-15** on every one of the three matrices.  That establishes the reference.
This row is then compared against it on the trace, which is not the weak check
it looks like: a tile's final version is what the trailing updates of its own
column read, so every off-diagonal element feeds a diagonal tile and the trace
is a function of the whole factor.  The three matrices give
`31622.766542 / 44719.846738 / 31748.460760`, and this row reproduces all three
at places 1, 4 and 16 -- and at one, two and four nodes.

## Sizing

Two knobs, and they divide cleanly -- neither is chosen for speed.

`ts` is the WIDTH knob: peak width is `t(t-1)/2` with `t = ds/ts`, and the class
rule asks for four times the largest geometry's 3456 workers, so `t >= 167`.
`ds` is the SIZE knob and the window fixes it.  At `ds = 90000` with `t = 180`
that makes `ts = 500` and a peak of 16,110 -- 4.66x the workers.

The base row derives the same way and lands on the same `t`: `ds = 16700`,
`ts = 100`, `t = 167`, peak 13,861.  The tile sizes differ by a factor of five
only because the two rows sit in different windows -- the base anti-scales and
is calibrated against 10-30 s, this one scales and gets ~150 s.  `ts` is not a
value anyone picks; it falls out of `ds/t`.

A larger tile does run faster here, and that is exactly why it is not used: it
would give a peak width of 4,005, which is 1.16x the workers rather than 4x,
and narrowing the width for a timing gain is the one move the width rule
forbids.  The experiment's subject is what the decomposition costs.

**At the anchor** (one node, 108 workers + 4 progress, catalog arguments) the
row runs 145.5 s / 89 GB under `val_wb_comb`, **159.7 s / 83 GB** under
`inv_wb` and 151.8 s / 85 GB under `excl_retain`.  The tightest family decides
and that is `inv_wb` at 160 s, the window a strong scaler asks for; the largest
residency any family holds is 89 GB, a third of what a node has.  The catalog
arguments are run at the anchor and nowhere else here: a node sweep at that
size is what a campaign produces, not what calibrates one.  The families
spread 1.10x in time, and the same binary in the same family moves about 5%
between runs on this host, so the row is stated as a range rather than one
figure.  Memory is not what bounds this row: the
largest geometry measured holds **45 GB a node**, less than one node does,
because the tiles divide while the runtime's own structures do not grow with
the node count.  Resident figures here are per RANK; the runner sums VmRSS over
the processes it started, which on a host that carries every rank itself is a
whole-machine total and not what a node has to hold.

**The trend** is taken at `ds = 40000`, the same `ts = 500`, at 15 workers and
one progress thread a node:

| nodes | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| time | 32.76 s | 22.26 | 16.67 | **13.77** |

**2.38x from one node to eight**, where the base row degrades 4.21x over its
own sweep.

The trend size is a shrink of the catalog cell rather than a smaller problem:
`ts` is held and only `ds` moves.  Taken at `ds = 10000, ts = 100` instead --
a fifth the tile -- the same binary anti-scales, 2.30 s at one node against
6.07 s at eight, because arithmetic per byte moved goes with `ts` and that
version is a fifth as intense.  The verdict follows the tile, so the trend has
to keep it.

The design's live set is one block per tile and nothing else.  A serial
emulator running this program's real task graph at `ds = 40000, ts = 500`
reports 88,593 tasks (every one of them fired), 3,243 datablocks, 91,800 events
and a peak live set of 6,180 MB against a 6.03 GiB matrix -- the data, not a
multiple of it.  Those counts are what the decomposition derives on paper, so
what the runtime holds beyond them is its own coherence copies.

About half the reads in that emulation are of a block homed on another rank,
and that is structural rather than a placement failure: a trailing update reads
one panel tile from its own row and one from its own column, and no assignment
of tiles to ranks makes both local.  What the ownership map decides is not
whether a panel crosses a rank but how many ranks it must cross to.
