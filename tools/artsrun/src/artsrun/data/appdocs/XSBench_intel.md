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
| `-b <batch>` | lookups per compute-phase sync batch (`NL_SYNC`); the batch is a FINISH scope and the next batch starts only when it drains, so this is the compute phase's in-flight width — the program's standing offer of concurrency to the machine | 1024 | ✓ parsed and carried through the `Inputs` copy chain (paramv slot 5 → `settingsInit` → `init_InputsH`). The calibrated args pass 3456 — the Master/Worker width rule, calibrated once at the largest geometry: the pool is the work count (`-l`, already ≫ workers) and the window covers the machine exactly (32×108 total workers), so the row's anti-scaling is attributable to the central serial production, never to an under-provisioned window.  Throughput is insensitive to the width within ~7% (1024↔13824 probed: <3% at one node, −7% at 2 nodes, +5% at 8) — production, not the window, is the cap |
| *(no flag)* `nprocs` | number of "rank" domains | — | ✗ `FNC_settingsInit` unconditionally sets `nprocs=1`; there is no `-p` in this port (contrast `XSBench_intel_sharedDB`, which has one). The whole program is always a single logical rank, at any node count |

`n_mats` is fixed at 12 (hardcoded, matches H-M's material count either size).
`print_CLI_error`'s "Default is equivalent to" line now names the defaults the
code actually applies (`-s small -l 15000 -b 1024`); it previously advertised
`-s large -l 15000000`, neither half of which was true.

## Structure

Let `N_i` = n_isotopes, `N_g` = n_gridpoints, `U = N_i·N_g` (unionized grid
points), `L` = lookups, `NB = ⌈L / b⌉` (compute-phase sync batches, `b` =
`-b`; calibrated args use 3456 → `NB=87`).

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

For the calibrated args (`-s large -g 96 -l 300000 -b 3456`): `N_i=355`,
`N_g=96`, `U=34080`, `L=300000`, `NB=87`. DBs ≈ `41+355+34080 = 34,476`
(each `uEnergy_grid` leaf is `N_i·4 B = 1.4 KB` — ~48 MB of plane total; the
355 nuclide grids are `96×48 B` each, ~1.6 MB total). EDTs ≈
`15+34080+174+900,000 = 934,269`, overwhelmingly the per-lookup chains.
Events ≈ `22+3·87 = 283`, staying small regardless of `L` (only `NB` moves
it). `-l` growth is pure EDT churn (no DB growth); `-g`/`-s` growth grows DB
count *and* size together, and `-s` additionally multiplies the aggregator
fan-in (`num_nucs[fuel]`: 34 small / 321 large).

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
`rankMultiLookupSpawner` is a `FINISH` EDT covering up to `b` lookup chains
(up to `b×3` EDTs in flight at the batch's peak); the *next* batch's
`rankCompute` only starts once the previous batch's `FINISH` scope — every
descendant of all `b` chains — has fully drained. For the calibrated args
that is 87 hard synchronization barriers. Within a batch the chains are
spawned serially by the one spawner EDT (create + 8 dependence registrations
per lookup, ~4 µs each), which is the single-node throughput cap — see
Sizing.

`mainEdt`'s only native (non-EDT) work is `read_CLI`/`print_inputs`, which is
negligible; everything else, including grid construction, runs as EDTs.

## Placement (base)

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

## Placement (hinted)

As-born round-robins every link of every lookup independently: rankLookup
lands somewhere, spawns macroxs there, which lands somewhere else, which spawns
the aggregator on a third rank — each chain's intermediate results cross the
wire twice for nothing (see above).

The layer has two halves, both compiled only under
`OCR_APP_OPTIMIZED_PLACEMENT` (`Main.c`; base resolves every call to
`NULL_HINT`):

- **Chain pinning** (`mcChainEdtHint`): pins the two SPAWNED links of each
  chain to the rank the chain's first link landed on.  The first link stays
  round-robin — that IS the load balance across independent lookups — so
  the distribution across ranks is untouched and only the chain's interior
  becomes local.
- **Home spreading** (`mcSpreadDbHint`/`mcSpreadEdtHint`): the readers are
  uniformly random, so no placement can make the reads local — but base
  every grid object is homed on the one rank that ran its init EDT, which
  then serves the whole machine.  The layer spreads the per-gridpoint plane
  and the nuclide grids round-robin (`i % N`), co-locates each gridpoint's
  alignment (writer) EDT with its block, spreads the 24 material tables
  (`mat % N`), and puts each of the per-lookup handle singletons
  (`InputsH`/`templatesH`/`dataH`/the GUID arrays) on a different rank —
  a singleton cannot be split, but the SET's aggregate serving load can be.

Rejected while designing the layer: sending `macroxs` to its gridpoint DB's
home (a remote EDT creation plus six remote dependence registrations costs
more than the one 1.4 KB remote fetch it saves), and pinning the spawner
chain (three acquires per batch — negligible).  Measured effect at 2 nodes
(`-s large -g 192 -l 65536`): inv_wb compute 28.4→4.66 s (6.1×), val_wb
50.0→37.6 s (1.33×) — under VAL the spread homes still charge a
re-validation round trip per acquire, which is the arm's structural cost,
not the placement's.

## Sizing (measured)

Single-node compute throughput is **spawn-limited**, not worker-limited: the
one live spawner EDT emits a lookup chain every ~4 µs, so 15 workers do
~260K lookups/s while 108 workers do 152K/s — MORE workers are slower,
because the idle workers' steal traffic interferes with the one spawning
worker. `-t` cannot change this (no-op) and the batch width moves it only
marginally (1024↔8192: <3% at one node, −7% at 2 nodes, +5% at 8 —
production, not the window, binds; the calibrated 3456 is the Master/Worker
window rule — cover the largest machine — not a tuning device). Multinode collapses
~100× further: each chain's
creation and its ~dozens of dependence registrations cross the wire, and
this control-plane traffic — not the grid payload — is the bill. That makes
the app a wiring-plane worst case in every arm, mildly anti-scaling with
node count.

Parameter roles, measured: `-l` is the only pure work dial (linear, zero DB
growth). `-s` selects the H-M configuration — large (the benchmark's
canonical case) changes no structure and not the 1-node rate (261K vs 256K
lookups/s vs small), but sets the fuel fan-in (321 vs 34) and multiplies
init's dependence registrations (`N_i²·g`). `-g` scales the exploded plane
and init linearly and leaves the compute rate untouched at one node; at
multinode the `U`-wide alignment fan-out acquires remotely, so init is
~150 s at 2-8 nodes for `-s large -g 96` (roughly flat in node count,
2× that at g=192, minutes-to-hours at g≥1000 under large). The calibrated
`-g 96` keeps a 34K-DB plane while holding init inside the 600 s bentley
ceiling with the compute phase still the majority of the worst cell.

## Family shape (measured, 15w+1p × 1/2/4/8 nodes, `-s large -g 96 -l 300000 -b 3456`)

hinted, e2e seconds (compute seconds in parentheses; e2e−compute ≈ the
init phase). All lookups discarded by design; the checksum pin held in
every cell:

| arm | 1n | 2n | 4n | 8n |
|---|---|---|---|---|
| val_wb | 3.0 (1.1) | 349.7 (186.0) | 360.2 (199.1) | 398.7 (238.4) |
| val_wb_comb | 3.0 (1.1) | 172.6 (26.7) | 176.8 (33.1) | 205.5 (45.0) |
| inv_wb | 4.0 (1.3) | 87.2 (18.8) | 132.5 (26.0) | 169.3 (32.9) |
| excl_retain | 5.8 (1.8) | 103.4 (26.9) | 136.8 (31.6) | 171.5 (39.0) |

base val_wb: 376.7 (229.7) / 390.0 (241.4) / 420.7 (274.6) at 2/4/8n —
the hinted layer wins 1.15-1.24× on val here (and 6.1× on inv, probed).
A Dane-geometry single node (108w+4p) runs the instance in 3.9 s (compute
1.9 s). The arm separations are the row's point: on write-once data at 8
nodes, VAL's re-validate-per-acquire costs 7.2× INV's covering reads;
combining recovers VAL to 5.3× better; EXCL sits near INV. Every arm is
mildly anti-scaling — the wiring plane, not the data, sets the wall.
Width-ladder evidence (1024/3456/8192/13824 all measured): every cell
moves within ~7%, the window is never the binder.
