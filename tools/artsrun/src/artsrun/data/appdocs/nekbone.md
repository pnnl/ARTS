# nekbone

*Nek5000's spectral-element CG proxy as an SPMD program built out of EDTs: a
binary fork tree spawns `Rx·Ry·Rz` virtual MPI ranks, each running a
fixed-length conjugate-gradient solve.*
Source: `third_party/ocr-apps/apps/nekbone/refactored/ocr_src/` (~21k lines;
the DAG is the generated `z_nekbone_inOcr.c`, 3083 lines).

## Overview

Nekbone solves a Poisson problem on a 3-D brick of spectral elements with CG
and no real preconditioner (`solveM` is an identity copy). Each iteration is
a local sum-factorised matvec (`ax`), a nearest-neighbour gather-scatter over
the 26-neighbour cubic lattice, and three global dot products. The result
scalar (`CGloop_stop> rnormfinal`, printed by virtual rank 0 once the tail
recursion runs out) is the residual the CG loop ends on — the one printed
quantity that observes the matvec, the halo exchange and the reductions, and
the only one that depends on `CGcount`. The *initial* residual
(`CGstep0_stop> rnorminit`, printed before the loop) is kept as a secondary
scalar; the completion marker `NKTIME> FinalEDT=` comes from the terminal
`finalEDT`. The program stresses
three things at once: real double-precision FLOPs (O(pDOF⁴) per element), a
fixed-size neighbour exchange, and a global all-reduce on every iteration's
critical path — at scale the reduction latency, not the arithmetic, is what
a run pays for.

## Parameters

Exactly eight positional arguments are required; anything else prints
`ERROR: 8 cmd line arguments are needed` and shuts the runtime down
(`neko_globals.c:83`). Each must be a run of decimal digits — a token with any
other character is named in an error and stops the run, rather than being
folded into a number. There are no defaults.

| arg | meaning | default | CLI reachability |
|-----|---------|---------|------------------|
| `argv[1..3]` = `Rx Ry Rz` | virtual-rank lattice; `Rtotal = Rx·Ry·Rz` is the SPMD width | none (required) | ✓ parsed in `mainEdt`, stored in the `NEKOstatics` DB and copied down the fork tree — multinode-safe |
| `argv[4..6]` = `Ex Ey Ez` | elements per virtual rank; `Etotal = Ex·Ey·Ez` | none (required) | ✓ same path |
| `argv[7]` = `pDOF` | DOF per axis per element; polynomial order is `pDOF-1`; must be ≥ 2 | none (required) | ✓ same path |
| `argv[8]` = `CGcount` | CG iterations — a fixed count, never a convergence test | none (required) | ✓ same path |

Two ordering constraints abort the run if violated (`neko_globals.c:185`):
`Rx ≥ Ry ≥ Rz` and `Ex ≥ Ey ≥ Ez`. Nekbone's polynomial-order *sweep* is
collapsed (`pDOF_end = pDOF_begin + 1`) — one order per run. Compile-time
knobs, none argv-reachable: `ARITY` = 10 (all-reduce fan-in,
`libs/src/reduction/reduction.h`), `Nfoliation` = 2 (binary fork tree),
`NEKbone_neighborCount` = 26 (hard-asserted), the four `REDUCTION_*` and
three `NKEBONE_USE_CHANNEL_FOR_HALO_*` switches (all on), and
`NEK_OCR_ENABLE_AFFINITIES` (on — see Placement).

## Structure

Write `R = Rtotal`, `E = Etotal`, `P = pDOF`, `N = CGcount`,
`D = Σ_ranks neighbours`, and `I = ⌈(R-1)/ARITY⌉` (ranks with children in the
reduction tree). Per-rank DOF count is `P³·E`; `NBN_REAL` is `double`.

| object | count | size |
|--------|-------|------|
| app EDTs | `12·N·R + 18·R + 2` — 12 per iteration per rank, plus a `2R-1`-node fork tree with its join twin and a 10-EDT setup chain per rank | — |
| reduction EDTs | `(1+3N)·(2R+I-1)`, plus a one-time `2R-1+I` channel install | — |
| app DBs | `26·N·R + 57·R - 7` | see below |
| — solution vectors C, F, R, X, W, P, Z | 7 live per rank | `(P³·E + 1)·8` B each |
| — per-element scratch G1/G4/G6/UR/US/UT/temp | 7 per rank, reused for every element | `(P³ + 1)·8` B |
| — derivative matrices `dxm1`, `dxTm1` | 2 per rank | `(P² + 1)·8` B |
| halo envelopes | `D·(N+3)` | `(ddof+1)·16` B |
| reduction scalars | `(3R-2)·(1+3N)` | 8 B |
| events | `5·D + 6R + I - 4` — 3 CHANNEL creates plus 2 labeled-STICKY `ocrEventCreate` *attempts* per directed halo edge (both endpoints race to install the same rendezvous GUID; the losing call still increments the counter), plus the reduction's per-rank `returnEVT` channel, per-non-root-rank up/down channels, 2 labeled-STICKY attempts per tree edge, and output events on the channel-install EDTs and on `finalEDT` | — |

A rank's neighbour count is `n(rx,Rx)·n(ry,Ry)·n(rz,Rz) - 1`, with `n = 3`
interior, `2` on a boundary, `1` on a degenerate axis: 26 for an interior
rank, 7 for a corner. Per-neighbour payload `ddof` is 2025, 45 or 1 DOF for a
face, edge or corner contact at `E = 4³, P = 12`.

Worked, at the calibrated `8 4 4 4 4 4 12 50` (`R=128, E=64, P=12, N=50`,
`D=2072`, `I=13`): ~79.1k app + ~40.7k reduction ≈ **120k EDTs**; ~173.7k app
+ 57.8k reduction + 109.8k halo ≈ **341k DBs**; 10,360 halo + 777 reduction =
**11,137 events**. Each solution vector is 864 KiB, so the live set is ~5.3 MB
per rank plus one transient generation of Z/P/R/X — **0.7–1.1 GB across the
128 ranks**. Halo traffic is ~19.5 MiB per gather-scatter round, ~975 MiB over
the run; the all-reduces move 8 bytes each and cost latency, not bandwidth.

Counter cross-check: verified (1 node, three points — `2 2 2 2 2 2 4 2`,
`2 2 2 2 2 2 4 4`, `2 2 1 2 2 2 4 2`): NUM_EDT_CREATE (467/755/235) and
NUM_DB_CREATE (1,307/1,967/563) match the formulas above exactly (+1
runtime-baseline EDT/DB per run). NUM_EVENT_CREATE needed the labeled-GUID
correction above — the losing `ocrEventCreate` attempt counts, in both the
halo handshake and the reduction tree's channel install — to match 325/325/81
exactly; the old `4D+5R+I-3` predicted only 262/262/66.

## Wiring

The generated code calls `ocrXHookup(OCR_EVENT_ONCE_T, ...)` everywhere, but
that helper is a bare `ocrAddDependence` wrapper (`app_ocr_util.c:100`) —
**its event type and flags are ignored and no event is created**. Every edge
in the main DAG is a direct DB→EDT dependence; the only real events belong to
the halo and the reduction.

- **Fork/join**: `SetupBtForkJoin` seeds `BtForkIF` over `[1, R]`; a node with
  `low < hi` splits in two (`BtForkFOR` → two `BtForkIF`), a node with
  `low == hi` is one virtual rank and starts `BtForkTransition_Start`.
  `BtJoinIFTHEN` folds 16-byte checksums back up.
- **Per-rank setup**: `BtForkTransition_Start` → `channelExchange_start/stop`
  (installs the halo channels) → `nekMultiplicity_start/stop` →
  `nekSetF_start/stop` → `nekCGstep0_start/stop` → `setupTailRecursion`.
- **CG iteration**, a strictly serial 12-EDT chain per rank:
  `tailRecursionIFThen` → `tailRecurTransitBEGIN` → `nekCG_solveMi` →
  `beta_start/stop` → `axi_start/stop` → `alpha_start/stop` →
  `rtr_start/stop` → `tailRecurTransitEND` → next `tailRecursionIFThen`.
- **Halo**: `start_channelExchange` creates, once per directed neighbour edge,
  one labeled `STICKY` event from a `Rtotal`-wide range (27 ranges, one per
  lattice direction) and three `CHANNEL` events (`maxGen=2, nbSat=1,
  nbDeps=1` — single producer and consumer per generation), one each for the
  multiplicity, set-f and ax rounds. A round satisfies the neighbour's channel
  with a freshly created payload DB and hooks its own channel onto the `_stop`
  EDT at one of 27 slots (unused directions take a `NULL_GUID` dependence), so
  `nekCG_axi_stop` has 45 slots — 18 fixed, 27 halo.
- **Reduction**: `reductionLaunch` builds an `ARITY=10` tree with per-rank
  up/down `CHANNEL` events, installed once through labeled `STICKY` events and
  reused for every later all-reduce; each rank threads one `reducPrivate` DB,
  taken `RW`, through the whole chain.

DB concurrency is low by construction: every solution vector is created, read
and written only by its own rank's chain, so no DB has two concurrent writers
and the maximum reader fan-out is one. The contention point is not a DB at
all — it is the reduction tree's root, which every iteration passes through
three times. One irregularity: `nekCG_axi_start` acquires `nekW` as `RO` and
writes the whole matvec result into it, which is safe only because the block
never leaves its node (see the notes file).

## Flow

`mainEdt` does no heavy work — it parses argv, reserves the labeled-GUID
ranges and seeds the fork tree, which is `log₂R` deep, doubles in width per
level, costs `~5R` EDTs and is over quickly.

Steady-state parallel width is exactly **`Rtotal`**: a virtual rank is a
serial chain with no intra-rank parallelism whatsoever, so the runtime sees
`R` independent 12-EDT chains per iteration plus the reduction and halo EDTs
connecting them. The serial bottlenecks are the three all-reduces per
iteration — each a `⌈log₁₀R⌉`-deep tree through a single root rank that every
rank blocks on — and the halo `_stop` EDTs, which cannot fire until all of a
rank's up-to-26 neighbours have satisfied their channels. A rank is never
more than one iteration ahead of its neighbours, nor ahead of the root at all.

## Placement (base)

This application carries an `OCR_APP_OPTIMIZED_PLACEMENT` layer (built as
`nekbone_hinted`, `HINTED_PLACEMENT` in `benchmarks/apps/CMakeLists.txt`;
catalog `hinted: true`), described in the next section. Outside that guard the
affinity hints it carries are genuinely base:
`ENABLE_EXTENSION_AFFINITY` is defined for the benchmark build, so
`NEK_OCR_ENABLE_AFFINITIES` is on and the program places explicitly.
`BtForkIF` computes `pdID = rankID % ocrAffinityCount(AFFINITY_PD)`, i.e.
`rankID % nodes`, and creates that rank's `BtForkTransition_Start` with an EDT
affinity hint for it; everything downstream asks for `NEK_OCR_USE_CURRENT_PD`,
so the whole setup chain and CG chain stay on the node the rank landed on, and
DBs created there with `NULL_HINT` are homed on that same node. Only the
fork/join tree itself (`BtForkIF`, `BtForkFOR`, `BtJoinIFTHEN`) is hint-less
and scatters round-robin — and it carries only 16-byte checksum blocks.

The *only* inter-node traffic is therefore the physical one: halo envelopes
created on rank A and RO-acquired by rank B, plus 8-byte reduction scalars
climbing the tree. The algorithm's locality is expressed; the lattice is not.
`rankID` linearises `(rx,ry,rz)` as `rx + Rx·ry + Rx·Ry·rz`, so with `Rx = 8`
and 8 nodes the map degenerates to `rx` — an x-slab decomposition, y- and
z-neighbours node-local, every x-neighbour remote. A good cut, but by
arithmetic accident: change `Rx` against the node count and locality moves.

## Placement (hinted)

The layer changes exactly one function: the rank-to-place map. Base ships
`calcPDid_S` = `rankID % places` (`neko_globals.c`), which puts a rank's
x-neighbours on other places by construction — consecutive rank ids are
x-neighbours, and consecutive ids land on consecutive places — so with more
than one place most of the 26-neighbour halo is remote. Under the guard,
`calcPDid_lattice` calls `nekbone_placeGrid`: it factors the place count into
`nx·ny·nz` boxes that divide the rank lattice, ranks the candidates by volume
per surface (the faces a place does not own are exactly the halo it exchanges),
and maps each rank to the box its lattice coordinate falls in. Every place still
holds the same number of ranks and every rank the same number of elements, so
balance is untouched; only which ranks share a place changes. One guard was
earned by measurement: a box one rank thick on an axis keeps none of that axis's
neighbours — it narrows the same exchange onto fewer peers rather than making it
local, and measured worse than spreading it (8 nodes, 120 ranks: 4.52 s against
4.45 s, the only legal factorisation there being 2×1×4) — so such a split falls
back to the shipped map. The restructured tier's participant numbering reuses
`nekbone_placeGrid` so both agree about which ranks share a place.

Measured (15w+1p per node, `18 16 12 2 2 2 12 100`): base 134.46 / 280.75 /
209.76 / 127.14 s at 1/2/4/8 nodes, never beating its own one-node time; hinted
133.15 / 108.05 / 71.37 / 56.49 s, monotone, 2.36× over one node. At the anchor
the two tiers are identical (one place — the map is irrelevant), which is the
check that the change is placement and nothing else.

## Sizing

`Rtotal` is the only dial that moves parallelism; `Etotal` and `pDOF` move
grain and memory; `CGcount` moves duration only.

- **Parallel width** is `Rtotal`: pick `Rx·Ry·Rz` ≳ 2× the total worker count,
  keeping `Rx ≥ Ry ≥ Rz`. Below one rank per worker the machine idles; well
  above it the reduction tree deepens as `log₁₀ R`.
- **Task grain** is `E·(12·P⁴ + 22·P³)` flops per rank per iteration, dominated
  by the `12·E·P⁴` matvec. `pDOF` is the cheap dial (compute ~P⁴, halo ~P²,
  memory ~P³); `Etotal` buys grain and memory linearly, halo as `E^(2/3)`.
- **Memory** is `7·(P³·E + 1)·8` bytes per rank, times `Rtotal`, plus a
  transient generation of four vectors per iteration.
- 1 node × 15 workers: `4 2 2 2 2 2 8 100` → 16 ranks × 8 elements × 512 DOF,
  a few seconds. 8 nodes × 120 workers: keep `Rtotal` ≥ ~256 for headroom.
- The calibrated `8 4 4 4 4 4 12 50` gives `Rtotal = 128`, one virtual rank per
  physical core of the reference machine, so the strong-scaling sweep moves
  from ~8.5 ranks per worker at 1 node to ~1.07 at 8 without starving a worker.
  `pDOF = 12` puts ~250 kflop in each element's matvec, enough that the run is
  not pure task overhead, and `CGcount = 50` puts the 1-node run in the minutes
  range. Note that the catalog's result scalar is fixed at CG step 0 and does
  not move with `CGcount`; only the `rnormfinal` line does.
