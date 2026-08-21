# RSBench_intel

*Every one of the `-l` lookups gets its own three-EDT chain and its own slice
of the cross-section dataset (355 nuclides → 3×355 small datablocks); nothing
is reduced or checked at the end — see `RSBench_intel_sharedDB` for the
opposite design (one big DB per array, coarse per-thread EDTs, an actual
verified checksum).*
Source: `third_party/ocr-apps/apps/RSBench/refactored/ocr/intel/src/` (7 C
files, ~880 lines; `main.c` builds and drives the whole graph, `init.c` /
`material.c` generate the synthetic cross-section data, `rs_kernel.c` is the
per-lookup compute).

## Overview

RSBench is ANL's proxy for the multipole-representation resonance-lookup
kernel of Monte Carlo neutron transport — XSBench's sibling benchmark: the
same "sample a random (material, energy) pair and evaluate a macroscopic
cross-section" workload, evaluated with a different (window/pole-based)
kernel. This port fragments the synthetic cross-section dataset into one
datablock **per nuclide** per array (poles, windows, pseudo-K0RS each get
their own small DB — 3×355 of them at default sizing) and turns every one of
the `-l` lookups into an independent `rankLookup → macroxs → microxsAggregator`
EDT chain that computes a macroscopic cross-section vector and **discards
it** — `macro_xs` is a stack-local accumulator in `FNC_microxsAggregator`,
never written to a DB, never reduced, never printed. The completion marker
`Lookups:` (from the results banner) only proves the run reached the end;
`scalar_kind: bool` and no `expect` mean the catalog uses this app as a
**liveness / scheduler probe**, not a correctness oracle — it stresses
task-creation churn and fine-grain remote-DB-acquire traffic, not verified
arithmetic.

## Parameters

| flag | meaning | default | CLI reachability |
|------|---------|---------|-------------------|
| `-l <lookups>` | XS lookups; sizes the whole graph (3 EDTs each) | 10,000,000 | ✓ parsed in `mainEdt`, carried inside the `Inputs` datablock to every consumer — multinode-safe |
| `-s small\|large` | H-M benchmark size; `small` also forces `n_nuclides=68` | large (355 nuclides) | ✓ |
| `-n <n>` | nuclide count, overrides whatever `-s` set | 355 (68 with `-s small`) | ✓ — validated: the H-M material tables hold fixed nuclide IDs, so only 68 (small) or ≥355 (large) is accepted; other values used to index the per-nuclide arrays out of bounds |
| `-p <poles>` | average poles per nuclide — sizes each pole DB | 1000 | ✓ |
| `-w <windows>` | average windows per nuclide — sizes each window DB | 100 | ✓ |
| `-d` | disable Doppler broadening (skip the temperature-dependent Faddeeva kernel) | Doppler ON | ✓ |
| `-t <threads>` | "OpenMP thread count" | 1 | ⚠ **dead**: parsed and range-checked (`≥1`) but never read again anywhere in the EDT graph — this port creates one EDT chain per lookup regardless of thread count; `Threads:` in the results banner is the only place the value is used. Passing `-t` prints a one-line warning that it has no effect in this port |
| *(none)* `n_mats` | material zones — the H-M reactor model has exactly 12 | 12 | ✗ compile-time only: no `-m` flag, and `load_mats`'s per-material nuclide tables are hardcoded for exactly 12 zones, so this isn't a knob a run could reasonably move |
| *(none)* `numL` | Legendre moments per nuclide (pseudo-K0RS array width) | 4 | ✗ compile-time only — never read from argv in either RSBench port |

## Structure

Let `n` = `n_nuclides`, `m` = `n_mats` (=12), `L` = `lookups`, and
`G = ⌈L / 1024⌉` (`NL_SYNC`, the hardcoded per-rank sync-batch size). `nprocs`
is hardcoded to 1 in `FNC_settingsInit` regardless of any parameter — this
port has no rank-partitioning axis at all.

| object | count | size |
|--------|-------|------|
| global/handle DBs (`InputsH`, `globalH`, `InputsHs[]`, `rankHs[]`, `timers`) | 6 | tens of bytes each |
| per-rank handle DBs (`InputsH[i]`, `rankH`, `settingsH`, `dataH`, `templatesH`) | 5 | small (GUIDs/ints) |
| `dataH`'s 8 index arrays (n_poles, n_windows, 3×per-nuclide GUID arrays, numNucs, 2×per-material GUID arrays) | 8 | `int`/`ocrGuid_t` arrays, `n` or `m` elements |
| per-nuclide DBs: pole, window, pseudo-K0RS | `3n` | Pole ≈72 B ×~`avg_n_poles`/nuclide; Window 32 B ×~`avg_n_windows`/nuclide; K0RS 32 B (`numL`×8 B) |
| per-material DBs: nuclide-ID list, concentration list | `2m` | `num_nucs[i]`×4 B / ×8 B |
| **DBs total** | `6 + 13 + 3n + 2m` | — |
| EDTs: setup (`mainEdt`…`init_dataH` chain) | 13 | — |
| EDTs: `rankCompute` + `rankMultiLookupSpawner` (chained sequentially, one pair per sync batch) | `2G` | — |
| EDTs: `rankLookup` + `macroxs` + `microxsAggregator` (one lookup) | `3L` | — |
| **EDTs total** | `13 + 2G + 3L` | — |
| explicit STICKY events (global + per-rank + one per sync batch) | `7 + G` | — |
| shim-materialized events (finish events for `EDT_PROP_FINISH` creates, output events for creates with a non-NULL `outputEvent`) | `13 + 2G` — the 4 `mainEdt`-level `EDT_PROP_FINISH` phases (`settingsInit`/`globalInit`/`globalCompute`/`globalFinalize`) each carry a finish event *and* an output event (`4·2=8`); `TS_init_InputsH` (created twice total, once from `globalInit` and once per-rank from `rankInit` since `nprocs=1`) and `TS_timer` each pass a non-NULL `outputEvent` with no `FINISH` (`+3`); `TS_globalComputeSpawner` (`FINISH` + output, `+2`); and each of the `G` `rankMultiLookupSpawner` creates carries both (`+2G`) | — |
| **Events total** | `20 + 3G` | — |

Nothing here is data-dependent in *count*: the random draws (`rand()`-seeded
per-nuclide pole/window counts, per-lookup material pick) change DB **payload
sizes** and per-`microxsAggregator` dependence-slot counts, never how many
objects exist. The shim-materialized events are easy to miss reading the
source alone: they never appear as an `ocrEventCreate` call, only as a
non-NULL last argument to `ocrEdtCreate`
(e.g. `main.c:480-482`) or an `EDT_PROP_FINISH` property flag, but the OCR
shim (`benchmarks/ocr_shim/arts_ocr.c:1022-1063`) materializes a real ARTS
event for each, and `arts_event_create` increments `NUM_EVENT_CREATE`
regardless of whether the call came from `ocrEventCreate` or from
`ocrEdtCreate`'s own output/finish-event machinery.

Worked numbers at the calibrated `-l 400000` (all else default: `n=355`,
`m=12`): **1,108 DBs**, **1,200,795 EDTs** (`G=391`), **1,193 events**
(`20+3·391`). Est. cross-section payload ≈26.7 MB (the app's own `Est. Memory
Usage` print, ~25.5 MiB), fragmented across the 1,065 per-nuclide/per-material
DBs above — independent of `-l`.

Counter cross-check: verified (1 node, `-s small -p 50 -w 10 -l 20` vs `-l
40`, both `G=1`): measured absolutes EDT 76/136, DB 248/248, EVT 23/23;
subtracting the runtime's constant baseline (+1 EDT, +1 DB, +0 EVT per run)
gives app-side EDT 75/135, DB 247/247, EVT 23/23 — DB and EDT already matched
the formulas above exactly; the event formula (`20+3G`, corrected from `7+G`)
now also reproduces the measured absolute (`20+3·1=23`) exactly, at both
`-l`.

## Wiring

`mainEdt → TS_settingsInit(FINISH) → TS_globalInit(FINISH: TS_init_InputsH +
TS_rankInitSpawner → TS_rankInit → TS_init_rankH → TS_init_dataH, whose body
calls `generate_n_poles`/`generate_poles`/`generate_window_params`/
`generate_pseudo_K0RS`/`get_materials` directly — no further EDTs) →
TS_globalCompute(FINISH: TS_globalComputeSpawner + TS_timer) →
TS_globalFinalize(FINISH)`. Inside `TS_globalComputeSpawner`, one
`rankCompute` EDT is created (`nprocs=1`); each `rankCompute` spawns one
`rankMultiLookupSpawner` (FINISH) covering up to 1024 lookups and, if more
remain, chains to the next `rankCompute` — the sync batches are **sequential**,
not spawned in parallel. Inside a batch, `rankMultiLookupSpawner`'s own C loop
creates up to 1024 independent `rankLookup → macroxs → microxsAggregator`
triples with no dependence on each other.

Every per-nuclide/per-material DB is written exactly once (during
`init_dataH`, before release) and only `DB_MODE_CONST`/RO thereafter — no
writer ever returns. `microxsAggregator` RO-acquires the pole/window/K0RS DBs
of every nuclide in its lookup's material (`num_nucs[mat]` of them, 5–321
depending on which material `pick_mat` drew). Because sync batches serialize,
**at most 1024 lookups are ever in flight** per rank; within that window a
given nuclide's DBs see concurrent RO readers in proportion to how many live
lookups picked a material containing that nuclide — the fuel material
(`num_nucs[0]=321`, picked with probability 0.14) drives the highest fan-out,
on the order of `1024×0.14≈140` concurrent readers, further capped by worker
count. No single DB is a contention point — the per-nuclide fragmentation
trades a hot DB for EDT/DB churn instead (contrast `RSBench_intel_sharedDB`,
which makes the opposite trade).

## Flow

Setup (`settingsInit`→`globalInit`→its rank-init chain) is a strict
FINISH-scoped serial pipeline on one worker producing the fixed `6+13+3n+2m`
DBs. The lookup phase is `G` sequential sync batches — a hard serialization
point independent of `-l`'s absolute size, since batch width is capped at
`NL_SYNC=1024` regardless of workload or worker count. Within one batch, max
concurrent EDTs is on the order of `min(3×1024, workers)` once
`rankMultiLookupSpawner`'s own (single-worker, sequential) 1024-iteration
creation loop has run; between batches there is no overlap — batch `g+1`'s
`rankCompute` depends on batch `g`'s `rankMultiLookupSpawner` output event.
`mainEdt`'s four top-level FINISH phases (`settingsInit → globalInit →
globalCompute → globalFinalize`) never overlap either.

## Placement (base)

Every `ocrEdtCreate` and `ocrDbCreate` in this port passes `NULL_HINT` — there
is no affinity/labeling code at all. Effective policy: **EDT → runtime
round-robin** (per-creating-rank atomic counter via `ARTS_HINT_ANY_RANK`);
**DB → home = creating rank**. Because `nprocs` is hardcoded to 1, the *entire*
per-nuclide dataset is created by whichever rank the single
`rankInit`/`init_dataH` chain happens to land on (call it rank R, itself
round-robin-placed from `mainEdt`) — then every lookup's `rankLookup` /
`macroxs` / `microxsAggregator` triple is independently round-robin-placed
too, so on an `N`-rank run only ~1/`N` of lookups execute where the data
lives. The rest remote-RO-acquire the touched per-nuclide DBs over the
network — up to `num_nucs[mat]` nuclides × 3 arrays per lookup (as many as
963 DBs for a fuel-material draw). The algorithm's real locality — one lookup
only ever needs one material's handful-to-few-hundred nuclides — is never
expressed in placement, the same "worst-case coherence stress by
construction" shape as `fibonacci`, but at per-nuclide-DB (KB-scale) rather
than 4-byte granularity.

## Placement (hinted)

As-born round-robins every link of every lookup independently: rankLookup
lands somewhere, spawns macroxs there, which lands somewhere else, which spawns
the aggregator on a third rank — each chain's intermediate results cross the
wire twice for nothing (see above).

The layer (`mcChainEdtHint` in `main.c`) pins the two SPAWNED links of each chain
to the rank the chain's first link landed on.  The first link stays
round-robin — that IS the load balance across independent lookups — so the
distribution across ranks is untouched and only the chain's interior becomes
local.  The nuclide/energy grids are read-only and replicate per rank on
first touch, so chain locality, not data placement, is what a hint can win
here.  `pdCount <= 1` returns `NULL_HINT`.

## Sizing

`-l` is the only lever that matters — `-t` is dead (see Parameters).
Doubling `-l` linearly doubles EDT count (`3` per lookup, `+2` per additional
1024-lookup batch) and wall time; DB count and memory are fixed by
`-s`/`-n`/`-p`/`-w` alone and never move with `-l`.

- **1 node × 15 workers**: `-l 50000`–`200000` gives 150k–600k EDTs — several
  hundred per worker live within a batch, enough to keep 15 workers busy for
  tens of seconds. Heavier `-l` mostly lengthens the sequential batch chain
  (`G`) rather than widening per-batch parallelism, since batch width is
  capped at 1024 regardless of `-l` or worker count.
- **8 nodes × 120 workers**: the same `-l` scaling applies, but `nprocs` stays
  pinned at 1 — the one-time, one-rank data-generation phase never spreads
  across nodes, and a 1024-wide batch leaves 120 workers comfortably fed but
  gains nothing from more nodes beyond spreading the remote-acquire traffic
  described in Placement.
- The calibrated `-l 400000` (≈1.2M EDTs, 391 sequential batches) is sized for
  a multi-minute 1-node run. Because there is no work-partitioning axis across
  nodes at all, treat multinode runs of this app as testing remote-acquire /
  fine-grain-churn cost, not throughput scaling — unlike `_sharedDB`'s `-t`.
