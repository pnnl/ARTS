# XSBench_intel

*Monte Carlo cross-section lookup proxy (ANL) — the grid is exploded into one
DB per nuclide and one DB per unionized energy gridpoint; every lookup is a
3-EDT chain touching a handful of those DBs.*
Source: `third_party/ocr-apps/apps/XSBench/refactored/ocr/intel/src/`
(`Main.c`, `io.c`, `CalculateXS.c`, `XSutils.c`, `Materials.c`, `qsort.c`,
`timers.c`; ~1300 lines, mostly `Main.c`).

## Overview

XSBench is a proxy for the macroscopic-cross-section lookup kernel at the
heart of Monte Carlo neutron-transport codes (the OpenMC family). The program
builds a Hoogenboom-Martin (H-M) reactor's per-nuclide microscopic
cross-section energy grids, unions them into one grid, then issues `-l`
randomized lookups: pick an energy and a material, binary-search the
unionized grid, and for every nuclide in that material interpolate a 5-vector
of cross sections. Every lookup's numeric result is computed and then
discarded — this port does not verify the lookup arithmetic.

The result scalar is `XSBench grid checksum: <u64>`, printed once during grid
construction: the sum of the IEEE-754 bit patterns of the unionized energy
grid's values. Grid construction uses a fixed seed independent of `-l`, so
the checksum is deterministic and is exactly what the catalog's `expect`
checks — it proves the grid was built identically, not that any lookup
computed the right answer.

This layout — one DB per nuclide grid and one DB per unionized-grid point,
each lookup expanding into an independent 3-EDT chain, everything placed with
`NULL_HINT` — makes the program a stress test of task-creation churn and
fine-grain DB creation/RO-acquire traffic; there is negligible floating-point
work per task. Contrast `XSBench_intel_sharedDB`, which holds the same grid
as a handful of large DBs instead. A restructured twin, `XSBench_dist`, is
held back in the catalog (commented out) pending a real port.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `-s <size>` | H-M benchmark size: `small`/`large`/`XL`/`XXL`; only `small` sets `n_isotopes=68`, others leave the 355 default (`XL`/`XXL` also set `n_gridpoints` unless `-g` was given) | `small` (`read_CLI` pre-sets `HM="small"`) | ✓ parsed in `mainEdt` |
| `-g <gridpoints>` | overrides `n_gridpoints` per nuclide | 11303 | ✓ |
| `-l <lookups>` | number of XS lookups (`L`) | 15000 | ✓ |
| `-t <threads>` | printed as "Threads:" and fed into `estimate_mem_usage`'s neighbourhood, but never changes decomposition — every lookup gets exactly one 3-EDT chain regardless of this value | 1 | ⚠ parsed but has no effect on parallelism or task count; passing `-t` prints a one-line warning that it has no effect in this port |
| *(no flag)* `nprocs` | number of "rank" domains | — | ✗ `FNC_settingsInit` unconditionally sets `nprocs=1`; there is no `-p` in this port (contrast `XSBench_intel_sharedDB`, which has one). The whole program is always a single logical rank, at any node count |
| `NL_SYNC` (= 1024) | lookups per compute-phase sync batch | 1024 | ✗ compile-time literal in `FNC_globalComputeSpawner`/`FNC_rankCompute`, not a `#define` and not user-reachable |

`n_mats` is fixed at 12 (hardcoded, matches H-M's material count either size).
`print_CLI_error`'s "Default is equivalent to" line now names the defaults the
code actually applies (`-s small -l 15000`); it previously advertised `-s large
-l 15000000`, neither half of which was true.

## Structure

Let `N_i` = n_isotopes, `N_g` = n_gridpoints, `U = N_i·N_g` (unionized grid
points), `L` = lookups, `NB = ⌈L / 1024⌉` (compute-phase sync batches).

| object | count | size |
|--------|-------|------|
| Nuclide grid DBs | `N_i` | `N_g × 48 B` each (6 `double`s) |
| Unionized-gridpoint DBs (`uEnergy_grid[i]`) | `U` | `N_i × 4 B` each (per-point nuclide index array) |
| Material DBs (index list + concs list) | `2 × 12 = 24` | varies (`num_nucs[mat]×4 B` / `×8 B`) |
| Scaffolding/handle DBs (Inputs, rank/data handles, array-of-GUID holders, timers) | 17 | small |
| **DBs total** | `41 + N_i + U` | — |
| Init-phase scaffolding EDTs, incl. `mainEdt` itself | 15 (fixed) | — |
| `init_uEnergy_i` (grid-alignment binary searches, one per unionized point) | `U` | — |
| `rankCompute` / `rankMultiLookupSpawner` (sync-batch pair) | `2·NB` | — |
| `rankLookup` / `macroxs` / `microxsAggregator` (one chain per lookup) | `3·L` | — |
| **EDTs total** | `15 + U + 2·NB + 3·L` | — |
| **Events total** | `22 + 3·NB` | — |
| EDT templates | 3 persistent (`rankLookup`/`macroxs`/`microxsAggregator`), created once and reused for every lookup — no per-lookup template churn | — |

The 15 fixed EDTs are `mainEdt` (the shim materializes it as its own EDT,
distinct from the runtime's own bootstrap wrapper — see the cross-check note
below) plus the 14-deep scaffolding chain
(`globalInit`→`init_InputsH`→`rankInitSpawner`→`rankInit`→`init_InputsH`→
`init_rankH`→`init_dataH`→`init_uEnergy`→`init_materials`→`globalCompute`→
`globalComputeSpawner`→`globalFinalize`, plus `timer`) that runs regardless of
`nprocs` (hardcoded to 1, so none of this chain's own loops contribute a
second factor). The event total's `22` fixed events are the output/finish
events the accounting rule attaches to those same 15 EDTs (`mainEdt`'s four
FINISH-scoped children `settingsInit`/`globalInit`/`globalCompute`/
`globalFinalize` alone contribute 4 explicit STICKY events + 8 output/finish
events) plus the always-present `init_InputsH`/`globalComputeSpawner`/`timer`
output events and the `init_dataH`-spawned `init_uEnergy`/`init_materials`
finish events; `3·NB` is `rankCompute`'s explicit
`rankMultiLookupSpawner`-completion STICKY event plus that spawner's own
output+finish events, once per sync batch. `init_uEnergy_i`,
`rankLookup`/`macroxs`/`microxsAggregator` all pass a NULL output event and
carry no FINISH scope, so the per-lookup and per-gridpoint fan-out contributes
zero events regardless of `U` or `L`.

For the calibrated args (`-s small -g 1000 -l 400000`): `N_i=68`, `N_g=1000`,
`U=68000`, `L=400000`, `NB=391`. DBs ≈ `41+68+68000 = 68,109` (the `uEnergy_grid`
fan-out dominates DB count at ~18.5 MB total; the 68 nuclide grids total
~3.3 MB). EDTs ≈ `15+68000+782+1,200,000 = 1,268,797`, overwhelmingly the
per-lookup chains. Events ≈ `22+3·391 = 1,195`, staying in the low thousands
regardless of `L` (only `NB` moves it, and `NB` saturates at `⌈L/1024⌉`).
`-l` growth is pure EDT churn (no DB growth); `-g`/`-s` growth grows DB count
*and* size together.

Counter cross-check: verified (1 node, `-s small -g 3 -l 10` vs `-s small -g 5
-l 20`): predicted absolutes 252/314/25 and 418/450/25 (`NUM_EDT_CREATE` /
`NUM_DB_CREATE` / `NUM_EVENT_CREATE`) match the measured counters exactly,
against a runtime baseline of `+1 EDT, +1 DB, +0 EVT` on top of the app
totals above (the baseline is the runtime's own bootstrap EDT that invokes
`mainEdt` and the argv DB it hands it — a layer below the app's own `mainEdt`,
which the 15-EDT fixed count already includes).

## Wiring

`rankLookup` (per lookup) takes `CONST` (→ ARTS RO) deps on the `InputsH`,
`templatesH`, and four *handle* DBs — the `nuclide_grids` GUID array, the
flat `uEnergy` grid, the `uEnergy_grid` GUID array, and the three
materials-handle arrays — none of these are the actual grid payload, only
GUID indirections. It binary-searches `uEnergy` locally, then creates
`macroxs` with RO deps on the handle DBs plus the one `uEnergy_grid[idx]`
leaf DB for the energy bucket it landed on. `macroxs` in turn creates
`microxsAggregator`, adding one RO dep per nuclide in the chosen material —
`num_nucs[mat]` (4 to 34, randomly chosen per lookup) individual nuclide-grid
leaf DBs. `microxsAggregator` is the only EDT that actually reads nuclide-grid
payload.

Every lookup-phase access is RO (`DB_MODE_CONST`); grid/material DBs are
written exactly once during init (single writer, home = whoever created it)
and never touched RW again. Max concurrent RO readers on one nuclide-grid
leaf DB is bounded by the sync-batch width (`NL_SYNC=1024`) and, within that,
by how many concurrently-scheduled lookups happen to draw a material
containing that nuclide (`pick_mat`'s distribution favors material 0, "fuel",
at 14%) — this is data- and schedule-dependent, not statically fixed. The
contention point is therefore the small set of nuclide-grid leaf DBs (`N_i`
of them) crossed against up to 1024 concurrently in-flight lookups, not the
per-gridpoint `uEnergy_grid[idx]` DBs (each lookup's `idx` is usually
distinct at these grid sizes).

## Flow

Three sequential program-level phases, each gated by a `FINISH` join:
**init** (single logical rank; ends with a `U`-wide parallel fan-out of
`init_uEnergy_i` grid-alignment tasks under one `FINISH` scope) → **compute**
→ **finalize**. The compute phase is `NB` strictly serial sync batches: each
`rankMultiLookupSpawner` is a `FINISH` EDT covering up to 1024 lookup chains
(up to `1024×3` EDTs in flight at the batch's peak); the *next* batch's
`rankCompute` only starts once the previous batch's `FINISH` scope — every
descendant of all 1024 chains — has fully drained. For the calibrated args
that is 391 hard synchronization barriers. `NL_SYNC` is a compile-time
constant, so no argument raises the in-flight width above ~1024 chains; more
nodes/workers beyond that only shortens each batch, not the barrier count.

`mainEdt`'s only native (non-EDT) work is `read_CLI`/`print_inputs`, which is
negligible; everything else, including grid construction, runs as EDTs.

## Placement (as-born)

Every `ocrEdtCreate`/`ocrDbCreate` in this port passes `NULL_HINT`.
Effective policy: **EDTs** → shim round-robin (`ARTS_HINT_ANY_RANK`, atomic
counter modulo rank count); **DBs** → home = creating rank.

The init-phase chain (`globalInit → rankInitSpawner → rankInit → init_rankH →
init_dataH`) is a sequence of 1:1 nested creates, each round-robining
independently, so the grid DBs (all created inside `FNC_init_dataH`) end up
homed on whichever single node that EDT happened to land on — an arbitrary
node, decided by the chain's cumulative round-robin state, not by any
locality intent. Because `nprocs` is hardcoded to 1, there is no way to get a
grid replica per node; the grid always lives on exactly one node.

Compute-phase `rankLookup`/`macroxs`/`microxsAggregator` are *also*
independently round-robined, with no relationship to where the grid lives.
Nearly every hop of every lookup chain is therefore a remote acquire of
whichever grid DB it needs — the same worst-case locality pattern as
`fibonacci`, but at payload sizes up to tens of KB per nuclide-grid DB
instead of 4 bytes. At multinode this app is a genuine bandwidth-and-message-rate
stress, by construction, with the intact-but-unexploited locality of the
algorithm (a lookup only ever touches its own material's nuclides) never
expressed in placement.

## Sizing

`-l` (`L`) is the only parameter that scales *parallel work* — each +1 lookup
adds exactly 3 EDTs and zero DBs. `-g`/`-s` (`N_g`, `N_i`, via `U=N_i·N_g`)
scale the *one-time* grid-construction DB fan-out (count and size together);
oversizing the grid inflates a largely single-node-homed, one-time cost far
out of proportion to the repeatable, embarrassingly-parallel lookup phase.

Guidance for N nodes × C workers: pick `L` so `3·L ≫ N·C` (thousands of
lookup-EDTs per worker keeps every deque busy across the run), independent of
node count — since `NL_SYNC=1024` caps in-flight concurrency per batch
regardless of cluster size, more workers only help once `1024` exceeds
`N·C`'s working set per generation, which holds comfortably at both 1 node ×
15 workers and 8 nodes × 120 workers. Keep `-g`/`-s` modest (the grid build
does not parallelize across nodes — it is always homed on one node) and grow
`-l` to scale wall time. The calibrated strong-scaling args (`-g 1000 -l
400000`) size the grid to a moderate ~68K-DB, ~22 MB one-time fan-out and the
lookup phase to ~1.27M EDTs, large enough that the per-hop remote-acquire tax
described above — not raw worker count — dominates wall time at multinode.
