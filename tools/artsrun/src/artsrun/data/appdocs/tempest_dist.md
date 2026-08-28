# tempest_dist

*The restructured version of `tempest`: the same cube-sphere exchange, the same
neighbour topology and the same cross-check, with the halo batched per rank
pair instead of one datablock per patch edge.*
Source: `third_party/ocr-apps/apps/tempest/refactored/ocr/intel-bryan/tempest_dist.c`
— a separate program that reuses the base port's geometry (the 363-line
cube-sphere neighbour finder) unchanged.  Selected as `tempest:restructured`.

## Overview

The base program anti-scales harder than anything else in the roster, and the
placement layer cannot reach the reason.  Hints already do everything a hint
can: they take remote acquires from **74.93% to 1.32%** and what crosses from
**14.4 GB to 265 MB** at four nodes, leaving under half a percent of the
442,368 directed patch edges crossing a rank line.  The row still degrades
**2.75x across the first node boundary**.

What is left is not the size of the cut but the price of crossing it.  A patch
edge owns one datablock that the two patches bounce: a timestep acquires it RW
on one side, stamps eight bytes into it and hands it back, so a crossing edge
migrates exclusive ownership twice per timestep.  The payload is eight bytes --
`nbData_t` is a single `s64`, and the source says "there will be other stuff
here later" -- while a remote acquire moves **349 bytes** of protocol on
average.  Ninety-eight percent of what crosses is not data.

This tier changes exactly that.  A rank keeps its own patches' inbound stamps
in memory and batches everything bound for one peer into a single block per
timestep.  The same stamps cross, the same number of times; they cross
together.

Why placement cannot do this: the dependence graph is a mesh, not a tree.  A
divide-and-conquer program (`nqueens`, `fib`) has independent subtrees, so
placing the top levels and letting the rest follow its creator removes cross-
rank traffic entirely -- an edge is traversed once.  Here every patch talks to
eight neighbours every timestep forever, so any partition leaves a cut and the
cut is paid `duration` times.  Placement minimises the cut; only a change of
what a cut edge costs can remove what remains.

## Parameters

The same two dials as `tempest`: `patchRange` (k, giving `6k^2` patches) and
`duration` (timesteps).  Both are arguments in both tiers.

## Structure

Per rank per timestep: one task per patch, exactly as the base program has, each
doing the same eight stores into that patch's own persistent slice; a fan-in
level that concatenates the slices; and the rank's single apply, which turns the
result into memory writes for the patches it owns and into one outgoing block
per peer.  The fan-in exists because a block written by many tasks at once
would serialise them and an apply taking every slice directly would carry one
dependence per patch -- it is the shape of a reduction, not a limit on how many
patches run at once.

## Wiring

Setup runs in two phases so no rank ever registers a dependence on a name that
has not been created: `mainEdt` creates every labeled name and per-rank block,
phase one fills them, phase two reads them.  A channel is created by the rank
that RECEIVES on it and published at the label `sender*nranks + receiver`;
labeled channel ranges are not among this OCR's labeled kinds, so the guid
rides a labeled sticky, as the other rank-persistent ports here do.  Each rank
seeds its own incoming channels once: without a seed generation every rank's
first apply waits on peers whose first apply is waiting on it.

## Flow

A patch task writes the triples `(to, dir, from)` its eight neighbours need --
the triple travels rather than an agreed ordering, so the two sides need agree
on nothing beyond the geometry they both compute.  The reverse direction is
derived from the topology rather than assumed: the cube sphere reverses
orientation across some face seams, so a patch's north neighbour does not
always have it to the south, which is why the base port learns each reverse
link by exchange instead of computing it.  The fan-in concatenates, the apply
sorts local from remote, publishes one block per peer, and starts the next
timestep; the last one prints the cross-check and announces the rank done.

## Placement (base)

Patches have spatial homes on the minimum-cut `P x Q` map the hinted tier
offers as a hint, here structural.  Every task and every block a rank owns
carries that rank's affinity, so a patch's slice, its group block and its
outgoing blocks are all born where they are used.

There is no separate `hinted` version: the decomposition is the placement.

## Sizing

`duration` keeps the shipped default -- the README calls the run "a few
timesteps" and `go.sh`'s recommended argument is the patch range -- so `k`
reaches the window, and the width follows for free: `6k^2` patches is far past
the 3456 workers the largest geometry offers.

This row scales, so its window is the scaler's ~150 s -- the row it
re-implements anti-scales and is calibrated against the short one, which is
why the two tiers run different sizes on purpose.  At the anchor (one node,
108 workers + 4 progress) the ladder measures 22.6 s at k=96, 118.7 s at 224,
149.7 s at 248 and 189.4 s at 272, so **k is 248**.  There the three coherence
arms run **155.1 s**, 121.9 s and 131.2 s -- the tightest decides, and it lands
on the window.

Memory never enters the choice: the anchor holds **2 GB**, against the base
row's 1 GB and a node's 256.  It does not grow with the run either -- the
slices and group blocks are made once and reused, the only per-timestep
allocation is one outgoing block per peer, and the channel's two generations
bound what is in flight -- so raising `k` from 96 to 248 left the resident set
where it was.

**The trend** is taken at `k=96` -- smaller than the anchor, and chosen so that
the worst cell of BOTH tiers lands inside the short window: the hinted tier's
worst is 18.5 s at two nodes and this row's is 11.8 s at one.  `duration=100`,
15 workers a node, two runs a cell:

| nodes | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| hinted | 6.72 s | 18.45 | 16.52 | 14.09 |
| this row | 11.76 s | 5.96 | 3.07 | **1.75** |

**6.72x from one node to eight, 84% of linear**, where the row it replaces
degrades to 0.48x.  The price is at one node, where a geometry with no
communication pays the fan-in and the apply for nothing: 1.76x.  From two nodes
on it is ahead, and at eight it is 8.1x faster in absolute terms.

The counters say what moved.  Over four nodes at the same arguments, against
the hinted tier: remote acquires **1,206 against 667,022** -- 0.01% of
acquires against 1.32% -- and what crosses **15.85 MB against 264.60 MB**.
The task count is unchanged, **5,734,811 against 5,640,202**: this tier batches
what crosses, it does not make the tasks bigger.
