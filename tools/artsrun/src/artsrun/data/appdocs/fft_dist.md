# fft_dist

*The restructured version of `fft`: the same transform, decomposed as a Bailey
four-step transpose over a tile grid, on a place-persistent SPMD structure whose
transpose is aggregated per place.*
Source: `third_party/ocr-apps/apps/fft/ocr/fft_dist.c` (~610 lines).

## Overview

`fft` is one datablock every task in a recursion acquires `DB_MODE_RW`; nothing
about it can be distributed, which is why its `hinted` version only contains the
tree rather than spreading it. This rewrite replaces the decomposition. The
length-N transform is viewed as an `N1 x N2` matrix (`N1 = 2^ceil(m/2)`,
`N2 = N/N1`) and computed in Bailey's four steps: an FFT down every column, a
twiddle, a transpose, and an FFT along every row. The matrix is cut into `t`
tiles in each direction, so no two tasks ever write the same object -- a tile
owns `N2/t` columns on the way in and `N1/t` rows on the way out.

The input is a single tone, whose spectrum is known in closed form
(`X[5] = X[N-5] = N/2`, every other bin zero), so each row tile verifies its own
output block analytically and the partial checksums reduce to exactly `N`. That
is the result scalar. A run that got the transform wrong prints
`FFT_DIST INVALID` and extracts nothing.

## Parameters

`fft_dist <power> [tiles] [places]`. The catalog runs `32 8192 32`.

`power` is the size knob. It is NOT the `fft` row's: a restructured tier is
its own row with its own window, and this one transforms 4 times the length in
about the time the recursion takes for a sixteenth of it. `tiles` is the compute decomposition, clamped
down to the largest power of two dividing both `N1` and `N2`; the width rule
sets it -- a dataflow DAG asks for about 4x the 3456 persistent units the
largest geometry provides, and `t` is the width of every phase.

`places` is the ownership and communication decomposition: how many partitions
the program divides its tiles into and aggregates its exchange in. It is an
**argument, not the rank count**. The program never asks how many ranks exist
except to compute a placement hint, so its task count, its datablock count and
the order its partial checksums combine in are identical in every geometry. 32
is one place per node at the largest geometry, which is where the calibration is
taken, and it is the convention XSBench's `-p 32` already uses here.

## Structure

`mainEdt` does O(P) work and nothing else: it creates the `P * waves * P`
events the places hand blocks through, one report event per place, and one
`rankInitTask` per place, hinted onto `place * nranks / places`. The events a
place hands to ITSELF -- the cut from a wave's unpack to its row tiles -- never
leave the place and are created there, not here. Everything else is created by
`rankInitTask` **on the place that will run it**, because a task created with a
remote affinity is a message and a graph built in one place cannot scale.

Per place, with `tpg = t/places`:

- **`colTileTask`** (`tpg`, no dependences) generates its own `N2/t` columns of
  the tone, runs a length-`N1` FFT down each, applies the twiddle, and writes a
  first-touch block in row-block-major order.
- **`packTask`** (one per wave, several waves per place) reads a run of this
  place's column tiles and writes one block per destination PLACE, then destroys
  the tiles it read. Aggregating per place is what keeps the message count off
  the tile count, and packing in waves is what lets the column form be released
  a wave at a time rather than all at once. Each block is created on its
  consumer's home.
- **`unpackTask`** (one per wave) takes this place's arrival from every place
  and cuts it into one block per row tile, then destroys the arrivals. This is
  the stage that lets the two properties hold at once: a block on the wire is a
  place pair's whole share, and a block a row tile waits on belongs to it alone.
  Emitting per destination ROW BLOCK instead removes this stage, but multiplies
  the transfer count by the tile count -- `places * waves * t` blocks, 2 KB each
  at the catalog decomposition -- and buys nothing back, because a row tile
  waits on every place and every wave regardless, so no arrival is freed any
  earlier for it.
- **`rowTileTask`** (`tpg`) takes one arrival per wave, reads its rows as
  contiguous runs, runs a length-`N2` FFT along each, checks every output bin
  against the closed-form spectrum, reports `{partial checksum, max err}`, and
  **destroys its arrivals**.
- **`rankJoinTask`** (one) sums this place's partial checksums, takes the worst
  of their errors, and destroys the partials.
- **`finishTask`** combines the `P` reports, prints the scalar and shuts down.

## Wiring

A column tile's block is `[row block][row][column]`; row-block major is what
makes a destination's whole share of it one contiguous run, so the pack copies
it per column tile rather than per row block. A transferred block is
`[source's column tile][row block of the destination][point]`, and the unpack
turns it into `[column tile of the wave, places in order][point]` per row tile,
which is the order the row tile walks -- so it gathers rather than searches.

Every block is destroyed by the task that consumes it: the pack destroys the
column tiles of its wave, the unpack destroys the arrivals, the row tile
destroys what the unpack gave it, the join destroys the partials, the finisher
destroys the reports. Nothing outlives its reader.

## Flow

Column tiles have no dependences and run as soon as their place is created. A
wave's pack fires when that wave's tiles are done -- not when the place's are --
a wave's unpack when that wave has arrived from every place, the row tiles when
every wave has been cut for them, the join on its place's partials, and the
finisher on the `P` reports.

## Placement (base)

A place is mapped to rank `place * nranks / places` and everything the place
creates carries that hint. That is the only thing the program asks the machine,
and it asks it for a hint: the decomposition above it is fixed by argument, so
where a place lands changes nothing about what the program is.

Blocks bound for a peer are created on the peer's home, since a block consumed
exactly once cannot amortize an ownership migration.

## Sizing

A place's resident data is `N/places` complex numbers throughout, because the
transpose redistributes it rather than growing it. Live across the machine that
is `N * 16` bytes of column-tile blocks -- 68.7 GB at the catalog's `power` 32 --
and about twice that during a wave, while the column form the wave still holds
and the row-major blocks it has emitted are both alive. Measured peak resident
set at the anchor is 161 GB under `val_wb`, 176 under `inv_wb` and 164
under `excl_retain` -- a 1.09x spread, which is the resident set following the
protocol only weakly because every transfer block has one writer and one
reader. `power` 33 is excluded on memory alone.

Measured at `power` 28 in the trend geometry (15 workers + 1 progress per
node), against the per-row-block split this replaces:

| nodes | per row block | per place pair |
|---|---|---|
| 1 | 6.88 | 5.55 |
| 2 | 43.39 | 4.44 |
| 4 | 52.59 | 1.99 |
| 8 | 38.10 | **1.16** |

**4.8x from one node to eight**, where the split it replaces lost a factor of
five over the same span -- and 33x at eight nodes outright. The split had been
measured on memory and on one node, and one node is exactly where its cost
cannot appear: every block is local there, so only the allocation shows.

At the anchor (one node, 108 workers + 4 progress) `power` 32 runs in 22.78 s
holding 161 GB under `val_wb`, 23.17 s and 176 GB under `inv_wb`, 22.89 s
and 164 GB under `excl_retain`. The tightest family decides, and 176 GB is what
the row is admissible at against a 256 GB node. The time is well inside the
window a strong scaler asks for; **memory is what fixes the size here**, since a
power doubles it and 33 does not fit. The checksum is one value in every cell
and every family -- an earlier version whose decomposition followed the rank
count instead had it drift with the node count, because the order the partials
combined in drifted too.

What this replaced, in turn, was a direct transpose of `t^2` blocks: 67.1 million
datablocks and as many never-reclaimed events, which at eight nodes took the
host to 807 GB of 1007 and had to be killed.
