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

Smoke-sized example (`-s small -g 10 -l 100`, `P=T=1`): `N_i=68`, `N_g=10`,
`U=680`, `G=⌈100/1000⌉=1`. EDTs ≈ `2+8+2 = 12`. DBs ≈ `2+(15+2+3) = 22`.
Events ≈ `1+2+3 = 6`. The dominant DB is `DBK_xs_grid` at `N_i·U·4 B`;
`DBK_nuclide_grids` and `DBK_uEnergy_grid` are each allocated at `U·48 B`
(`DBK_uEnergy_grid` is allocated at `NuclideGridPoint` size — 48 B/point —
but indexed as the 8-byte energy-only `GridPoint`, so most of that buffer is
never addressed; harmless, just wasted). For the calibrated args (`-s large
-g 96 -l 50000000 -t 108 -p 32`, fixed at every node count): per instance,
`U=34080` puts `xs_grid` at ~48 MB (well out of cache) and the grids at
~1.6 MB each; the compute plane is 32 instances × 108 chains ×
`G=⌈50M/(1000·108)⌉=463` generations — coarse-grain (1000 inline lookups
per generation EDT), in contrast to `XSBench_intel`'s per-lookup 3-EDT
chains, and identical at every node count.

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

## Placement (base)

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

Without `-p` the block partition resolves every rank to policy-domain 0 —
all work on one node no matter how many the run launches — so the catalog's
`args_by_nodes` scales `-p` to the cell's node count (the same arrangement
as `RSBench_intel_sharedDB`).

## Sizing (measured)

Every input is fixed at the largest geometry's calibration and held across
the node sweep (the campaign convention, shared with the tiled rank grids):
`-p 32` instances, `-t 108` chains each, `-l 50M` per instance — an
aggregate of 1.6G lookups, a constant program at every node count.  At 32
nodes the fork's block partition puts one instance per node; at fewer
nodes the same 32 instances pack evenly (e.g., 32 on one node at 1n, like
a fixed rank grid).  `-l` is **per instance** (each instance is an
independent MC replica); `-g`/`-s` scale the per-instance replica and its
one-time init, replicated 32 times.

`-t` is a logical chain count, not a thread count — nothing an app passes
can change the runtime's rank or worker count. The calibrated `-t 108`
follows the campaign's SPMD width rule at the calibration geometry:
exactly 1× the node's persistent workers, one chain per worker at 32
nodes — oversubscription only adds churn. The knob is real and measured:
`-t 1` → 2.55M lookups/s single-chain, `-t 15` on 15 workers → 28M/s;
widths from 15 to 3456 on 15 workers measure within noise of each other.

The port originally kept a pointer field (`xs_ptrs`) inside the shared
unionized-grid DB and had every generation EDT re-derive it — ~544 KB of
identical-value stores into RO-acquired memory per generation, because a
DB's mapping is acquire-relative. Those concurrent same-line stores
ping-ponged across NUMA domains and froze per-worker scaling (9.1M lookups/s
at 15 workers ≈ 10.4M/s at 108). The field is gone — the kernel indexes the
flat `xs_grid` arithmetically — and the wall went with it: 20.1M/s at 15
workers, **131.5M/s at 108** (91% per-worker efficiency, `-g 1000` small),
2.2× and 12.6× over the pointer-fixup form.

`-g` sets the table size and thus how memory-bound the inline kernel is; at
the calibrated `-s large -g 96` the kernel runs ~1.2-1.3 µs/lookup at 15
workers (14.8M/s). `-l` is per instance and int-typed (2³¹ cap); the node
ladder divides it to hold the aggregate at 1.6G, which sizes the 1-node
ferrari-geometry cell at ~108 s and lets the high-node cells shrink as the
row's near-perfect scaling dictates.

## Family shape (measured, 15w+1p × 1/2/4/8 nodes, `-s large -g 96 -l 50M -t 108 -p 32` fixed)

e2e seconds — near-perfect strong scaling of the fixed 32-instance
program, arms indistinguishable (the decomposition shares nothing across
instances, so no coherence arm has anything to do):

| arm | 1n | 2n | 4n | 8n |
|---|---|---|---|---|
| val_wb_nocomb | 119.5 | 66.2 | 35.2 | 17.5 |
| val_wb | 119.7 | 66.5 | 35.2 | 17.5 |
| inv_wb | 119.9 | 66.2 | 35.4 | 17.6 |
| excl_retain | 119.4 | 66.2* | 35.2 | 17.6 |

Speedup 1.81 / 3.40 / 6.83 at 2/4/8 nodes (efficiency 85-90%; the
residual is the 32 replicas'-worth of init packed onto fewer nodes). A
Dane-geometry single node (108w+4p) runs the same program in 23.0 s.
(*) The excl_retain 2-node cell reproducibly measures ~122 s with a
normal kernel (122.3/122.4 across two runs) — ~56 s sits outside the
kernel, an arm-specific non-compute stall this configuration alone
shows; queued for a runtime-side look.
