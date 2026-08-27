# quicksort_dist

*The restructured version of `quicksort`: the same array and the same answer,
decomposed as a sample-splitter distributed sort on a place-persistent SPMD
structure whose exchange is aggregated per place, and whose input is generated
where it is classified.*
Source: `third_party/ocr-apps/apps/quicksort/ocr/quicksort_dist.c` (~720 lines).

## Overview

`quicksort` is a recursion in which every task acquires the SAME datablock
`DB_MODE_RW` -- the children are handed the same GUID and differ only in their
index range -- so its `hinted` version can only keep the chain where the block
is. This rewrite replaces the decomposition with the sample-splitter
(PSRS/samplesort) shape: a sorted sample of the input is cut into `nbuckets-1`
ascending splitters, every element is routed to the bucket its value falls in,
and each bucket sorts what it receives with a local in-place quicksort. Because
the buckets are in splitter order, concatenating them in order is the sorted
array, and no two tasks ever write the same object.

The result scalar is a real check, not a sample of the output: each bucket
reports `{count, sum, sorted, first, last}`, each place combines its buckets'
verdicts, and the finisher confirms order across place boundaries, the element
count, and the permutation-preserving element sum against the input sum the
chunks computed as they generated.

## Parameters

`quicksort_dist <arraySize> <range> [nbuckets] [nchunks] [places]`. The catalog
runs `4000000000 1000000 6912 6912 32`.

`arraySize` is the size knob and `range` is the program's own default (`RANGE`
in `quicksort.c`). `range` is the value the `quicksort` row runs and `arraySize`
is not: a restructured tier is its own row with its own window, and this one
sorts hundreds of times more elements in the same span. The generator is a pure
function of the index, so a chunk producing its own slice produces exactly the
values the base program would have generated for those indices. `nbuckets`
and `nchunks` are the compute decomposition and the width rule sets them.

`places` is the ownership and communication decomposition. It is an **argument,
not the rank count**: the program never asks how many ranks exist except to
compute a placement hint, so its task count, its datablock count and the order
its partial sums combine in are identical in every geometry. 32 is one place per
node at the largest geometry, the convention XSBench's `-p 32` already uses.

## Structure

`mainEdt` does O(P) work and nothing else: the `P*P` exchange events, one
verdict event per place, the sticky splitter event, one `placeInitTask` per
place, and the sampling that feeds the splitters. Everything else is created by
`placeInitTask` **on the place that will run it**, because a task created with a
remote affinity is a message and a graph built in one place cannot scale.

- **`sampleTask`** (`nchunks`) draws its share of the
  `SAMPLES_PER_BUCKET * nbuckets` sample; **`splitterTask`** sorts the whole
  sample and cuts the splitters, publishing them on a sticky event every chunk
  and every unpack reads.
- Per place, with `cpg = nchunks/places` and `bpg = nbuckets/places`:
  - **`chunkTask`** (`cpg`) generates its own slice, groups it by destination
    place, and carries its own partial input sum in its own block -- no two
    chunks share a writable object.
  - **`packTask`** (one per wave, several waves per place) concatenates a run
    of this place's chunks per destination, sends one block per peer, and
    destroys the chunks it read -- so the generated form is released a wave at
    a time rather than all at once.
  - **`unpackTask`** (one) cuts the `P` arrivals into this place's `bpg` bucket
    blocks, so a bucket is handed its own elements rather than scanning every
    arrival for them.
  - **`bucketTask`** (`bpg`) sorts its bucket, reports five words, and
    destroys the sorted block: the checks are computed before it is released
    and the finisher works from the verdicts, so holding it would keep a second
    copy of the whole array alive for no reader.
  - **`placeJoinTask`** (one) combines this place's verdicts into one.
- **`finishTask`** combines the `P` verdicts, prints the scalar and shuts down.

## Wiring

A block is `[count][elements]` at every stage but the chunk's, which prefixes
per-destination counts so the pack takes a peer's share as a contiguous run.
Every transfer block is destroyed by the task that consumes it: the pack
destroys the chunks of its wave, the unpack destroys the arrivals, the bucket
destroys both its input and its sorted output once checked, the join destroys
the verdicts, the finisher destroys the reports. Only the splitters outlive
their readers, and they are 55 KB.

The chunk's own sum riding in its own block is what removed the previous
version's one shared writable object: 6,912 chunks acquiring one block
`DB_MODE_RW`, exclusive at rank granularity and migrating across ranks, measured
at **57% of that version's eight-node time**.

## Flow

Samplers run first; the splitter task fires on all of them and releases the
chunks. A wave's pack fires on that wave's chunks -- not on the place's -- the
unpack on all arrivals, the buckets on the unpack, the join on its buckets, and
the finisher on the `P` verdicts.

## Placement (base)

A place is mapped to rank `place * nranks / places` and everything the place
creates carries that hint. That is the only thing the program asks the machine,
and it asks it for a hint. Blocks bound for a peer are created on the peer's
home, since a block consumed exactly once cannot amortize an ownership
migration.

## Sizing

A place's resident data is `arraySize / places` elements throughout, because the
exchange redistributes it rather than growing it, and nothing accumulates: the
emulator that runs this program's real task graph reports a peak live set equal
to the data, not a multiple of it.

Measured at the catalog arguments, trend geometry (15 workers + 1 progress per
node), against the version this replaces:

| nodes | staged exchange | place-persistent |
|---|---|---|
| 1 | 8.89 | 8.99 |
| 2 | 11.44 | 5.38 |
| 4 | 11.74 | 3.08 |
| 8 | 12.51 | **2.03** |

**4.4x from one node to eight**, where the version it replaces got 1.4x worse
over the same span. Every cell prints the same answer.

One property of the program is worth stating because it is not this tier's to
fix: the base recursion's pivot is `getRandNum(size/2) % (high-low)`, so every
subrange of a given length picks the same relative pivot and the base tree is
far from balanced. This tier does not inherit that, but its splitters come from
a random sample, so bucket occupancy is balanced only to the accuracy
`SAMPLES_PER_BUCKET` (32) buys.
