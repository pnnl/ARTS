# mapreduce_wordcount

*The canonical MapReduce application (Dean & Ghemawat, OSDI 2004) in
native OCR, weak-scaling form — a synthesized corpus streamed through map
tiles and a K-ary reduction tree whose lower levels stay node-local.*
Source: `benchmarks/apps/mapreduce_wordcount.c` (in-repo).

## Overview

Every map tile generates its own text from a PRNG seeded by the tile's
GLOBAL id, so the input never exists as a file: no ingest I/O at any node
count, and the corpus for a given (nodes, seed) is identical across runs
and runtime configurations.  The vocabulary is CLOSED — each token is
drawn from exactly V words (word k spelled in variable-length base-26),
the same fixed-word-list scheme as Hadoop's RandomTextWriter — so the
histogram width is V by construction.  Tokens are drawn Zipf-like via the
continuous inverse CDF (O(1) per token); with dense V-slot partials the
draw skew shapes count values only, never the communication volume.

A spawner tree unfolds the K-ary reduction over M = TPN x nodes tiles
(rank-contiguous, tile t on rank t/TPN): each combine EDT is created on
its FIRST child's rank, so the lower levels of the tree merge inside a
node (the map-side combiner) and only the upper ~log_K(nodes) levels
cross ranks — those crossings are the coherence traffic the campaign
measures.  Every partial histogram is created LOCALLY by the EDT that
fills it (creator-local home) and wired straight into its parent's slot
after release; there are no events and no pre-created blocks, and setup
is O(log) deep with nothing serialized on the main rank.

The verifier requires the tree total to equal M x WORDS exactly (one
lost update anywhere breaks it) and prints an FNV checksum of the final
histogram — the scalar the cross-configuration consensus votes on.
Partial blocks are retained for the run (a combine destroying blocks it
still holds would race its own release), so V x partial-count x ITER is
the memory envelope; big-V campaigns carry fewer iterations.

## Parameters

`V WORDS TPN K ZIPF GRAIN ITER SEED`

- **V** — vocabulary size = histogram width; the reduce payload is
  V x 8 bytes (16384 -> 128 KB partials; 1M -> 8 MB), which makes V the
  campaign's payload dial.
- **WORDS** — tokens per map tile; with token synthesis+parse costing
  tens of ns, 1M words is a ~40 ms map tile.
- **TPN** — map tiles per node; M = TPN x nodes.  Tiles are homogeneous
  by construction, so any TPN >= workers load-balances; 2 x workers is
  the convention.
- **K** — reduction fan-in (tree depth = ceil(log_K M)).
- **ZIPF** — draw skew x100 (0 uniform, 100 = s=1).  Shapes count values
  only; traffic-invariant with dense partials.
- **GRAIN** — extra spin per tile in us, to emulate heavier map work.
- **ITER** — map+reduce rounds; the verifier of round i spawns round i+1.
- **SEED** — corpus seed.

## Structure

Three working EDT kinds plus a driver: `map_edt` (one per tile: stream-
synthesize text in 64 KB chunks, tokenize, count into a fresh local
partial), `combine_edt` (depc = K: sum child partials into a fresh local
partial; short slots of the last node arrive empty), `spawn_edt` (unfold
one tree node: create the combine and one child per chunk; single-tile
chunks become maps directly), and `verify_edt` (depc = 1: exact-total
check, checksum, next round or shutdown).  EDTs per round:
M maps + ceil((M-1)/(K-1)) combines + about as many spawners.

## Wiring

Pure DB-to-slot wiring, no events: a producer releases its partial and
calls `ocrAddDependence(partial, parent, slot, DB_MODE_RO)` — release
strictly before wiring, so the satisfy that travels to the parent's rank
finds a published block.  The combine template's fan-in is fixed at K;
slots the last chunk cannot fill are satisfied with NULL_GUID /
DB_MODE_NULL and skipped by the sum.

## Flow

mainEdt parses knobs, creates the four templates and the round-0
verifier, then unfolds the root tree node in place.  Spawners fan the
creation out (O(log) depth); maps run as they are created; combines fire
as their K slots fill; the root partial reaches the verifier, which
either spawns the next round's verifier+tree or prints the marker and
shuts down.

## Placement (base)

Every EDT and every block carries an explicit affinity hint — nothing is
left to the no-hint defaults.  Tile t (map EDT and its partial) lives on
rank t/TPN; a combine and its partial live on the FIRST child's rank; the
verifier on rank 0.  Lower tree levels therefore merge node-locally and
only the top ~log_K(nodes) levels cross ranks.  There is no `_hinted`
twin: the placement layer IS the application (the same explicit-assignment
discipline MPI map-reduces use), so the base build is already fully hinted.

## Sizing

Weak scaling: per-node work is TPN x WORDS tokens regardless of node
count; the tree adds ceil(log_K M) levels of which only the top
~log_K(nodes) cross ranks.  e2e(N) = T_map + (cross-rank levels) x
T_edge(protocol) — the protocol coefficient rides the log term and V
scales T_edge.  Defaults (V 16K, WORDS 1M, TPN 2 x workers, K 8) give
~40 ms map tiles, 128 KB partials, and a few-second round; retained
partials bound memory at V x 8B x (M x ITER x K/(K-1)) per campaign
cell, so big-V points carry small ITER.
