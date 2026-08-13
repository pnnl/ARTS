# XSBench_intel_sharedDB

*Same Monte Carlo cross-section lookup kernel as `XSBench_intel`, but the
nuclide/energy grid is held as a handful of large contiguous DBs shared
(read-only) by every worker "thread" of a rank, instead of exploded into one
DB per grid item — and this port is genuinely SPMD, with each rank pinned to
its own node.*
Source: `third_party/ocr-apps/apps/XSBench/refactored/ocr/intel-sharedDB/src/`
(`Main.c`, `io.c`, `GridInit.c`, `CalculateXS.c`, `XSutils.c`, `Materials.c`,
`timers.c`).

## Overview

Computes the same H-M reactor cross-section lookup workload as
`XSBench_intel`, but structured as `-p` independent SPMD "ranks", each
forked via `forkSpmdEdts_Cart1D` and pinned to one ARTS node through the OCR
affinity extension. Each rank independently *regenerates its own full copy*
of the nuclide/energy grid — there is no cross-rank sharing of grid content.
The "sharedDB" in the name is intra-rank: the grid is a handful of large
contiguous DBs (one nuclide-grid DB, one unionized-energy-grid DB, one
flattened cross-section-index DB, ...) that every one of that rank's `-t`
worker "threads" reads concurrently, in contrast to `XSBench_intel`'s
per-item DB fan-out. Lookups are not exploded into per-lookup EDTs either:
each worker processes a large contiguous chunk of lookups inline, in a
straight-line C loop inside one EDT.

The result scalar, `Workload    (unit): <lookups>`, is printed by the shared
`print_throughput_custom` timer helper and is a plain echo of the `-l`
argument — not a computed checksum. It can only catch an argv-propagation
failure, not a computation bug (see notes/Findings).

A restructured twin, `XSBench_dist`, is held back in the catalog; it derives
from this variant's companion `Main_dist.c`.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `-s <size>` | H-M size, same semantics as `XSBench_intel` | `small` | ✓ |
| `-g <gridpoints>` | overrides `n_gridpoints` | 11303 | ✓ |
| `-l <lookups>` | total lookups (`L`) | 15000 | ✓ |
| `-p <nprocs>` | number of SPMD ranks (`P`), each pinned to one ARTS node | 1 | ✓ — genuinely drives node placement (`forkSpmdEdts_Cart1D`); `XSBench_intel` has no equivalent flag. Validated `≥ 1` (0 previously hung the SPMD fork, a negative value previously wrapped to a huge rank count) |
| `-t <threads>` | number of independent per-rank lookup-decomposition chains (`T`) | 1 | ✓ — genuinely drives intra-rank parallel width (`lookUpKernelEdt` spawns `T` chains); contrast `XSBench_intel`, where `-t` is inert |
| `CHUNK_SIZE` (= 1000) | lookups per generation per thread-chain | 1000 | ✗ `#ifndef`-guarded compile-time constant |
| `SCHEDULER_TYPE` (= 1) | `0` = one static shot per thread, `1` = dynamic chunked generations | 1 | ✗ compile-time |
| `VERIFICATION` | enables a per-lookup result hash and a fixed RNG seed | off | ✗ compile-time `#ifdef`, not built in this catalog entry |

## Structure

Let `N_i`, `N_g`, `N_m=12`, `L` as before, `P` = nprocs, `T` = nthreads,
`G = ⌈L / (1000·T)⌉` (generations per thread-chain).

| object | count | size |
|--------|-------|------|
| Per-rank scaffolding EDTs, incl. `mainEdt` itself (`mainEdt`→`initEdt`→`channelSetupEdt`→`FNC_xsbenchMain`→`FNC_initSimulation`→`lookUpKernelEdt`→`launchReductionEdt`→`summaryEdt`) | `1 + 7·P` | — |
| `lookUpKernelPerThreadEdt` + `iterationsPerThreadEdt` (one pair per thread-chain generation) | `2·P·T·G` | — |
| Top-level (`wrapUpEdt`) | 1 | — |
| **EDTs total** | `2 + 8·P + 2·P·T·G` | — |
| Per-rank "handle" DBs (rankH, rankDataH, rpPerfTimer, the 6 grid/material DBs, 3 pointer-array DBs, **plus 3 more from the nested `gpmatrix` call inside `generate_energy_grid`'s own unionized-grid sort**) | `15·P` | grid DBs scale with `N_i·N_g`; the rest are small |
| Per-thread scratch DBs (`DBK_seed`, `DBK_xs`; created once per tid, reused across generations) | `2·P·T` | tiny |
| Per-generation ephemeral pointer DBs (created and destroyed within each `iterationsPerThreadEdt`) | `3·P·T·G` | tiny |
| Top-level (`argv`, `globalParamH`) | 2 | small |
| **DBs total** | `2 + P·(15 + 2·T + 3·T·G)` | — |
| Events (`mainEdt`'s `finalOnceEVT`, plus per-rank `rpPerfTimerEVT`/`loopCompletionLatchEVT`, plus 3 per thread-chain generation) | `1 + 2·P + 3·P·T·G` | — |

`generate_energy_grid` (`GridInit.c`) calls `gpmatrix` a *second* time
internally to build a temporary sorted copy of the nuclide grid before
destroying it (`DBK_nuclide_grids`/`DBK_nuclide_grid_ptrs`, immediately
`ocrDbDestroy`'d — `NUM_DB_CREATE` counts the create regardless), so
`initSimulation`'s own DB fan-out is 11 creates (`gpmatrix`×2 + 
`generate_energy_grid`'s own `uEnergy_grid`/`xs_grid` + `load_num_nucs` +
`load_mats`×2 + `load_concs`×2), not the 9 the original count assumed. Each
generation's 3 events are `iterationsPerThreadEdt`'s FINISH-scoped output
event pair (`ocrEdtCreate(...,EDT_PROP_FINISH,...,&iterationsPerThreadOEVT)`
→ output + finish) plus `createEventHelper`'s `OCR_EVENT_COUNTED_T` join —
the original `2·P·T·G` term missed the third.

For the catalog's args (`-s small -g 10 -l 100`, no `-p`/`-t` ⇒ `P=T=1`):
`N_i=68`, `N_g=10`, `U=680`, `G=⌈100/1000⌉=1`. EDTs ≈ `2+8+2 = 12`. DBs ≈
`2+(15+2+3) = 22`. Events ≈ `1+2+3 = 6`. The dominant DB is `DBK_xs_grid` at
`N_i·U·4 B ≈ 185 KB`; `DBK_nuclide_grids` and `DBK_uEnergy_grid` are each
`U·48 B ≈ 32.6 KB` (`DBK_uEnergy_grid` is allocated at `NuclideGridPoint`
size — 48 B/point — but only ever indexed as the 16-byte `GridPoint`, so
two-thirds of that buffer is never addressed; harmless, just wasted). This is
a debug-sized smoke configuration (see notes/Findings), not a scaling
workload — everything above is O(10) objects, in contrast to
`XSBench_intel`'s calibrated ~1.27M EDTs.

Counter cross-check: verified (1 node, `-s small -g 3 -l 100` vs `-s small -g
5 -l 2500`): predicted absolutes 13/23/6 and 17/29/12 (`NUM_EDT_CREATE` /
`NUM_DB_CREATE` / `NUM_EVENT_CREATE`) match the measured counters exactly,
against a runtime baseline of `+1 EDT, +1 DB, +0 EVT` on top of the app
totals above.

## Wiring

`iterationsPerThreadEdt` is the sole reader of the grid: one EDT per
(rank, thread, generation) holds RO (`DB_MODE_RO`) deps on six DBs —
`nuclide_grids`, `uEnergy_grid`, `xs_grid`, `num_nucs`, `mats_all`,
`concs_all` — plus RW on its own private per-thread `seed`/`xs` scratch, and
performs its entire `[ibegin,iend]` lookup range as one straight-line C loop
with `calculate_macro_xs`. No further EDT fan-out per lookup.

Max concurrent RO readers on one rank's grid DB is `T` (bounded by how many
thread-chains run concurrently within that rank); ranks never share a DB
instance (each rank privately regenerates its own grid), so cross-rank
concurrency does not compound onto one DB object — it instead multiplies
memory footprint by `P` (each rank pays the full grid cost independently, as
opposed to `XSBench_intel`'s single shared grid). No DB is ever taken RW by
more than one node's tasks: init writes are single-writer, and the per-tid
scratch DBs and per-generation pointer DBs are private and never contended.
The contention point (when `T>1`) is the rank's large `xs_grid`/
`nuclide_grids` DBs under `T`-way concurrent RO acquire at the start of each
generation; at the catalog's default `T=1` there is no contention at all.

The one cross-rank data dependency in the whole program is
`launchReductionEdt` → `reductionLaunch`, a labeled-GUID `ALLREDUCE`
(`REDUCTION_F8_MAX`) over the `P` ranks' elapsed times, feeding the printed
throughput.

## Flow

Per rank: `initEdt` (pinned to its node) → `channelSetupEdt` →
`FNC_xsbenchMain` → `FNC_initSimulation` → `lookUpKernelEdt` spawns `T`
independent generation-chains → `launchReductionEdt` (waits on a `LATCH`
event gated by all `T` chains reaching their last generation) →
`summaryEdt` → top-level `wrapUpEdt` shuts down.

`FNC_initSimulation` is a single long straight-line EDT: `gpmatrix` /
`generate_energy_grid` / `set_grid_ptrs` / `load_mats` / `load_concs` all run
serially in C, with **no EDT fan-out at all** during init — the mirror
opposite of `XSBench_intel`, which parallelizes the equivalent
grid-alignment work into `U=N_i·N_g` independent EDTs. `set_grid_ptrs` in
particular does `N_i·N_g·N_i` binary searches single-threaded inside that one
EDT; it is the largest serial bottleneck in the program and does not shrink
with node or worker count.

Compute-phase parallel width is `P·T` (assuming ranks run concurrently, and
each rank's `T` chains run concurrently); within a chain, generations are
strictly serial (each `lookUpKernelPerThreadEdt` gates the next on its
predecessor's `FINISH` scope draining) — `G` sync points per chain, same
batching structure as `XSBench_intel`'s `NB` but over coarse per-thread
chunks (`CHUNK_SIZE=1000`) rather than per-lookup fan-out.

## Placement (as-born)

Unlike `XSBench_intel`, this port makes real, non-`NULL_HINT` placement
decisions. `forkSpmdEdts_Cart1D` pins each of the `P` rank-init chains to a
specific ARTS node via `OCR_HINT_EDT_AFFINITY`, computed by a **block**
partition of the `P` ranks over `ocrAffinityCount(AFFINITY_PD)` nodes (rank
`i` → a contiguous block, not round-robin). Confirmed against the shim:
`AFFINITY_PD`'s count is `arts_get_total_ranks()`, and the hint value is
consumed directly as an ARTS rank id (`extract_edt_affinity` in
`benchmarks/ocr_shim/arts_ocr.c`). Every DB and EDT a rank creates thereafter
inherits that rank's captured `myDbkAffinityHNT`/`myEdtAffinityHNT`
(`getAffinityHintsForDBandEdt`, itself reading `ocrAffinityGetCurrent` — "the
node I'm running on right now") — so a rank's grid, materials, and lookup
EDTs are co-located on one node by construction: real locality, unlike
`XSBench_intel`'s scatter.

The catalog's calibrated args do **not** pass `-p`, so `nprocs=1`: with a
single rank, the block partition always resolves to policy-domain 0 — **all
work lands on node/rank 0 regardless of how many nodes the run launches**;
every other node does nothing. Multinode scaling requires the user to pass
`-p <node_count>` explicitly; it is not automatic (see notes/Findings).

## Sizing

Two independent axes: `-p` spreads work *across* nodes (one rank pinned per
node, up to node count; beyond that ranks share a node via the block
partition). `-t` spreads work *within* a rank across concurrent
generation-chains. `-g`/`-s` scale the one-time per-rank init cost *and*
memory footprint, replicated `P` times (unlike `XSBench_intel`'s single
shared grid instance).

Guidance for N nodes × C workers: set `-p N` (one rank per node, matching the
as-born affinity mapping) and `-t` up to roughly `C` (too far above starves
at each generation barrier from oversubscription, too far below leaves
workers idle); pick `-l` large enough that `G=⌈L/(1000·T)⌉` spans several
generations, or the run is dominated by the one-time serial `set_grid_ptrs`
cost, which does not shrink with more nodes or workers. Worked examples: 1
node × 15 workers → `-p 1 -t 12..15`; 8 nodes × 120 workers → `-p 8 -t 14`
(one rank per node, leaving headroom for the progress thread) — but note
that even at 8 ranks, each rank still pays the *full* serial `set_grid_ptrs`
cost for its own grid copy, so wall time is bounded below by that per-rank
serial cost regardless of N.

The catalog's calibrated args (`-s small -g 10 -l 100`, no `-p`/`-t`) run
`P=1, T=1` — the same tiny configuration as the correctness `expect_args`,
not a scaling workload; treat it as a smoke config rather than tuned for the
reference machine (see notes/Findings).
