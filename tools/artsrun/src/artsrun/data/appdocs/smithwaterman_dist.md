# smithwaterman_dist

*The restructured version of `smithwaterman`: the same wavefront and the same
kernel, decomposed so each place builds the tasks for its own band of rows.*
Source: `third_party/ocr-apps/apps/smithwaterman/ocr/smithwaterman_dist.c`.

## Overview

The row this re-implements prescribes its whole DAG from `mainEdt`: a readiness
event trio per tile, then a task per tile with four dependences each. Measured,
that is **9.82 s of a 9.93 s run** — against 0.001 s to read the input. A tile
cannot start before the creator has reached it, so the run cannot end before
the loops do. More workers make it slower (7.91 s at 15, 10.88 s at 112) and
more ranks make it far worse, because a task placed elsewhere is a message.

The wavefront itself was never the constraint. A `W x W` tile grid has `W^2`
tasks against an anti-diagonal critical path of `2W-1`, so about `W/2` of
average concurrency — thousands of ready tasks at these sizes, far more than
the machine offers.

## Parameters

`smithwaterman_dist <tileW> <tileH> <seq1> <seq2> <score> [places]`.
The catalog runs `100 100 ... 32`.

The two knobs are independent: total DP work is the product of the sequence
lengths, and the wavefront's peak width is `min(len)/tile`. This row is a
**scaler**, so the window fixes the length -- the anchor goes as `L^2.2`
(7.9 s at 200k, 40.1 s at 400k, 92.7 s at 600k, 178.5 s at 800k), which puts
the window at `L = 800,000`. The tile then sets the width.

`places` is the ownership decomposition and is an **argument, not the rank
count**: the answer does not move over places 1, 2, 4, 8 and 16, nor over one,
two, four and eight nodes.

## Structure

`mainEdt` reads the sequences, reserves one labeled event range, and creates
one `placeInit` per place. That is all of its work.

Each `placeInit` owns a **contiguous band of rows** and builds, for its band
only, the border cells and one task per tile. Names are what make this
possible: a readiness event's name follows from the tile coordinate, so a
place can name a tile it does not own -- which is exactly what the top edge of
a band needs -- while only the owner ever creates it. Each name is therefore
created exactly once, and a consumer may register a dependence on a name whose
object does not exist yet.

No event grid is built up front. A readiness event is created by the task that
satisfies it, immediately before, so a name is alive only from the moment it is
produced to the moment its single consumer reclaims it: the live set follows
the wavefront rather than the tile count -- but only while creation is itself
paced. Creating every name in a parallel phase lifts the live set: at the
catalog cell this row holds 106 GB against the base row's 3.

## Wiring

A tile's task takes the west tile's right column, the north tile's bottom row
and the north-west tile's bottom-right corner, plus the shared parameter block.
It writes three blocks of its own and satisfies the three events it owns, then
destroys the three blocks it read **and the three events that carried them** —
each readiness event is named by exactly one dependence, so its single consumer
is also its last user.

## Flow

Band `r` starts as soon as band `r-1` has produced its first column, so the
bands run as a pipeline rather than in turn.

## Placement (base)

Rows are banded, not scattered, because a wavefront is a pipeline. Only a
band's top edge crosses a rank -- `O(P*W)` against the `O(W^2)` a cyclic or
round-robin map pays, every neighbour of which is remote -- and the price is
pipeline fill, `P` of the `2W-1` anti-diagonal steps, under a percent at these
sizes. Measured, a cyclic map was worse than the base row's own hinted banding
(155 s at two nodes against 105 s), and banding is 192x the base row at eight.

Places are bands and ranks are bands, so the linear place-to-rank map is the
right one here: it keeps a rank's rows adjacent, which is the whole point. (A
two-dimensional ownership would lose an axis through that map. A
one-dimensional one does not — the same line of code is a defect in one
structure and correct in the other.)

The event names are laid out so a tile's events are homed on the rank that owns
the tile: the tiles are numbered densely within each rank's band and then
interleaved by rank, and the three events of a tile are strided by a multiple
of the rank count so they share that home.

Over four nodes the counters put this row and the hinted tier at the same
locality -- 0.21% of acquires remote against 0.27%, both from an imbalance of
1.00x -- and separate them by what crosses: **2.4 MB against 274 MB**.  The
acquire ratio cannot see the difference because it counts what a task reads,
and both band their tiles; the bytes can, because the row this replaces builds
every task on one rank and each creation and dependence registration is itself
a message.

## Correctness

The score is the bottom-right cell of the DP, which every tile feeds. At the
trend size the two ports agree on `86360`, and this row reproduces it at one,
two, four and eight nodes. At the
catalog size a sequential reference is not available -- 800,000 x 800,000 is
6.4e11 cells -- so the pin is agreement between two independently built graphs:
this row and the port it re-implements both give `493680`.

## Sizing

**At the anchor** (one node, 108 workers + 4 progress, catalog arguments) the
tightest of the three coherence arms runs **178.5 s holding 106 GB**, 41% of
what a node has; the other two run 163.6 s and 155.6 s at 88 and 106 GB.

**The trend** is taken at `L = 140,000`, the same tile, at 15 workers a node,
two runs a cell:

| nodes | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| time | 6.07 s | 3.69 | 1.88 | **0.98** |

**6.19x from one node to eight** — 77% of linear. The price is at one node,
where the parallel creation phase costs 1.42x against a serialised one; from
two nodes on it wins, and at eight it is 4.6x faster in absolute terms.

**DELIBERATE DEVIATION on width**, structural and smaller than the base row's.
A wavefront needs `W^2` tasks to offer width `W`, and distributing the creation
does not reduce the count: at `L = 700,000` a tile of 50 buys 4.05x the
workers but costs 493 s and 199 GB against tile 100's 108 s and 48 GB. The tile
stays 100 and the row runs at **2.31x** the workers -- against the base row's
0.41x, which is the width this tier actually bought.
