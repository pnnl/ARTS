# LCS_wavefront

*The restructured version of `LCS_all_db_distributed`: the same computation
with the control plane rebuilt — per-tile boundary strips travel on labeled
counted events, leaves run in true anti-diagonal wavefront order, and memory
follows the frontier.*
Source: `third_party/ocr-apps/apps/LCS/refactored/ocr/intel-jesmin-lcs_all_db_distributed/lcs_wavefront.c`
(~430 lines, C).  Selected as `LCS_all_db_distributed:restructured`.

## Overview

Computes the identical edit-distance-style score as the tiled original —
same per-tile string seeds, same analytic row-0/column-0 boundary, same
per-cell recurrence — so the two programs print the same `LCS length` and
cross-validate each other.  What changes is everything the original's
scaling died of: the quadrant recursion whose FINISH gating capped speedup
at (4/3)^d is replaced by real tile-to-tile data dependences, and the
upfront creation of the whole `(N+1)²`-cell score matrix is replaced by
transient per-leaf scratch, so the program both parallelizes to the
anti-diagonal width `L = N/base` and keeps only the wavefront frontier
resident.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `N` | string length; total work is `N²` cells | 1024 | ✓ parsed in `mainEdt`, carried via paramv — multinode-safe |
| `argv[2]` = `base` | tile side; `L = N/base` is the wavefront width dial (`N % base == 0` required, loud-failed otherwise — no power-of-two constraint) | 256 | ✓ same |
| `LCS_ROW_BANDS_PER_RANK` | bands per rank in the row-band placement | 1 | ✗ compile-time `#ifndef` |

## Structure

Per run: `2L` string-tile creates + `2L` init EDTs (each with an
app-supplied COUNTED(1) completion event — the gate below), `L` row-spawner
EDTs in a free-running chain, `L²` leaf EDTs, one wrapup and one dedicated
shutdown EDT.  Per interior leaf: 3 strip DBs out (right column `base`
ints, bottom row `base` ints, corner 1 int) + 3 labeled COUNTED(1) events;
edge tiles skip the outputs nobody consumes.  Every strip is destroyed by
its single consumer, so steady-state resident data is the frontier's strips
plus the `2L` string tiles — the score matrix itself never exists.  The
leaf's working set is 3 rolling rows of `base` ints in transient scratch.

## Wiring

Leaf `(bi,bj)` has exactly five dependences: its S tile (RO, band-steered
home), its T tile (RO, round-robin home), and the west/north/northwest
strips — each an ordinary DB riding a **labeled COUNTED(1) event** the
producer creates at satisfy time (a consumer that registered first parks on
the absent label and is fired by the install; the declared count of one
reclaims the event as soon as its consumer has registered — nothing
lingers).  Row-1/column-1 leaves take `NULL_GUID` in the missing slots and
synthesize the boundary strips analytically.  The wavefront must not start
before the strings exist (an RO read of a tile still being written is
unordered), so the first row spawner is gated on all `2L` init-completion
COUNTED events.

## Flow

Row spawners free-run ahead creating leaves (creation is cheap and parks on
events; it never constrains the wavefront), leaves fire in anti-diagonal
order as their strips arrive, width ramps 1→L→1.  The last tile's corner
event feeds the wrapup (prints the scalar), and shutdown runs in its own
EDT behind the wrapup's completion.

## Placement (base)

Row-band, baked in (a restructured program owns its placement, so its
base form IS the placed form — there is no `_hinted` flavour): block-row `bi` maps to `band(bi-1)`, leaves are hinted
there, and strip/S-tile homes follow through labeled-index re-encoding
(`band + nranks·idx`).  West strips are always band-local; north/corner
strips cross ranks only at band boundaries; T tiles stay round-robin (a
column's readers span every band).

## Family shape (measured, 15w+1p x 1/2/4/8 nodes, `131072 512`)

restructured, e2e seconds — the point of the rewrite: it SCALES, where the
tiled original was capped by wiring at every node count:

| arm | 1n | 2n | 4n | 8n |
|---|---|---|---|---|
| val_wb | 3.24 | 2.31 | 1.51 | 1.11 |
| val_wb_comb | 3.27 | 2.32 | 1.51 | 1.12 |
| inv_wb | 3.26 | 2.01 | 1.42 | 0.74 |
| excl_retain | 3.24 | 1.92 | 1.36 | 1.10 |

At this small instance (`L=256`, average width 128 ≈ the 8-node worker
count) ramp-up/down is a visible fraction, so 2.9–4.4x at 8 nodes is a
width-limited floor, not the asymptote.  On one 108-worker node the rewrite
runs `131072 1024` in 0.82 s where the tiled original needs 15.5 s (19x) —
the wiring, not the machine, was the original's limit.

## Sizing

`N` sets total work (`N²` cells) and `base` dices it: `L = N/base` is the
wavefront width, so `base` trades per-leaf grain against parallel width at
fixed work.  Fixed-work granularity on the Dane-mirror geometry (1 node,
108w+4p, Release, val_wb): `2097152/512` = 122.6 s vs `2097152/256` =
202.1 s — finer tiles cost runtime object churn, so pick the coarsest
`base` whose `L` still spans the largest geometry.  Measured lattice:
1048576/256 = 45.3 s, 1572864/256 = 104.6 s, 2097152/256 = 202.1 s,
2097152/512 = 122.6 s, 3145728/512 = 275.8 s.  The calibrated point is
**`1769472 256`** (141.0 s): `L = 6912`, whose average anti-diagonal width
(3456) equals the 32-node x 108-worker campaign's total worker count, and
the nearest feasible point to the ~150 s anchor.  Answer at that point:
`LCS length: 890574` (pinned; also cross-checked against the tiled
original's identical-strings oracle at small sizes).  Memory is
frontier-bound by construction; the eager 200 GB-class footprint of the
tiled original does not exist here.
