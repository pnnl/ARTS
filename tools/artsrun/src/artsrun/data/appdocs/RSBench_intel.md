# RSBench_intel

*Every one of the `-l` lookups gets its own three-EDT chain and its own slice
of the cross-section dataset (355 nuclides → 3×355 small datablocks); the
lookup results are discarded by design, and the pinned scalar is a checksum
of the deterministic pole data itself — see `RSBench_intel_sharedDB` for the
opposite design (one big DB per array, coarse per-thread EDTs).*
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
never written to a DB, never reduced, never printed. Because no lookup result
is observable, the pinned scalar is `RSBench pole checksum:` — a bit-exact
sum over the IEEE-754 bit patterns of the generated pole data (`init.c`,
printed once from the serial `init_dataH` phase). It verifies that the
fixed-seed dataset every lookup reads is identical on every configuration —
not the lookup arithmetic itself, which the app deliberately discards. Every
family-sweep cell held the pin.

## Parameters

| flag | meaning | default | CLI reachability |
|------|---------|---------|-------------------|
| `-l <lookups>` | XS lookups; sizes the whole graph (3 EDTs each) | 10,000,000 | ✓ parsed in `mainEdt`, carried inside the `Inputs` datablock to every consumer — multinode-safe |
| `-b <batch>` | lookups per compute-phase sync batch (`NL_SYNC`) — the in-flight width of the whole compute phase, since each batch is a FINISH scope and the next starts only when it drains | 1024 | ✓ parsed, validated `≥1`, carried in the `Inputs` datablock through both settings-init hops — multinode-safe |
| `-s small\|large` | H-M benchmark size; `small` also forces `n_nuclides=68` | large (355 nuclides) | ✓ |
| `-n <n>` | nuclide count, overrides whatever `-s` set | 355 (68 with `-s small`) | ✓ parsed (only `≥1` checked) — ⚠ the H-M material tables hold fixed nuclide IDs (up to 67 for small, 354 for large) and the tables are picked by count alone, so any value other than exactly 68 or ≥355 indexes the per-nuclide arrays out of bounds in the kernel; the campaign only ever reaches this through `-s` |
| `-p <poles>` | average poles per nuclide — sizes each pole DB | 1000 | ✓ |
| `-w <windows>` | average windows per nuclide — sizes each window DB | 100 | ✓ |
| `-d` | disable Doppler broadening (skip the temperature-dependent Faddeeva kernel) | Doppler ON | ✓ |
| `-t <threads>` | "OpenMP thread count" | 1 | ⚠ **dead**: parsed and range-checked (`≥1`) but never read again anywhere in the EDT graph — this port creates one EDT chain per lookup regardless of thread count; `Threads:` in the results banner is the only place the value is used. Passing `-t` prints a one-line warning that it has no effect in this port |
| *(none)* `n_mats` | material zones — the H-M reactor model has exactly 12 | 12 | ✗ compile-time only: no `-m` flag, and `load_mats`'s per-material nuclide tables are hardcoded for exactly 12 zones, so this isn't a knob a run could reasonably move |
| *(none)* `numL` | Legendre moments per nuclide (pseudo-K0RS array width) | 4 | ✗ compile-time only — never read from argv in either RSBench port |

## Structure

Let `n` = `n_nuclides`, `m` = `n_mats` (=12), `L` = `lookups`, and
`G = ⌈L / batch⌉` (`NL_SYNC`, the `-b` sync-batch width, default 1024).
`nprocs` is hardcoded to 1 in `FNC_settingsInit` regardless of any parameter —
this port has no rank-partitioning axis at all.

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

Worked numbers at the calibrated `-l 150000 -b 3456` (all else default:
`n=355`, `m=12`): **1,108 DBs**, **450,101 EDTs** (`G=44`), **152 events**
(`20+3·44`). Est. cross-section payload ≈26.7 MB (the app's own `Est. Memory
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
`rankMultiLookupSpawner` (FINISH) covering up to `batch` lookups and, if more
remain, chains to the next `rankCompute` — the sync batches are **sequential**,
not spawned in parallel. Inside a batch, `rankMultiLookupSpawner`'s own C loop
creates up to `batch` independent `rankLookup → macroxs → microxsAggregator`
triples with no dependence on each other.

Every per-nuclide/per-material DB is written exactly once (during
`init_dataH`, before release) and only `DB_MODE_CONST`/RO thereafter — no
writer ever returns. `microxsAggregator` RO-acquires the pole/window/K0RS DBs
of every nuclide in its lookup's material (`num_nucs[mat]` of them, 5–321
depending on which material `pick_mat` drew). Because sync batches serialize,
**at most `-b` lookups are ever in flight** per rank; within that window a
given nuclide's DBs see concurrent RO readers in proportion to how many live
lookups picked a material containing that nuclide — the fuel material
(`num_nucs[0]=321`, picked with probability 0.14) drives the highest fan-out,
on the order of `3456×0.14≈480` concurrent readers at the calibrated width,
further capped by worker count. No single DB is a contention point — the per-nuclide fragmentation
trades a hot DB for EDT/DB churn instead (contrast `RSBench_intel_sharedDB`,
which makes the opposite trade).

## Flow

Setup (`settingsInit`→`globalInit`→its rank-init chain) is a strict
FINISH-scoped serial pipeline on one worker producing the fixed `6+13+3n+2m`
DBs. The lookup phase is `G` sequential sync batches — a hard serialization
point independent of `-l`'s absolute size, since batch width is capped at
`-b` regardless of workload or worker count. Within one batch, max
concurrent EDTs is on the order of `min(3·batch, workers)` once
`rankMultiLookupSpawner`'s own (single-worker, sequential) `batch`-iteration
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

The layer (in `rsbench.h`, because the creates span several files) is the
same two-part recipe as `XSBench_intel`'s. **Chain pinning**
(`mcChainEdtHint`): the two SPAWNED links of each chain pin to the rank the
chain's first link landed on; the first link stays round-robin — that IS the
load balance across independent lookups — so the distribution across ranks
is untouched and only the chain's interior becomes local. **Home spreading**
(`mcSpreadDbHint`): every dataset object is created inside the one
`init_dataH` task and would otherwise be homed on that single rank, which
then serves the whole machine's reads — the per-nuclide pole/window/K0RS
blocks spread round-robin by nuclide (one nuclide's three blocks
co-located), the material tables by material, and the ten handle/index
singletons every lookup acquires each land on a different rank.
`pdCount <= 1` returns `NULL_HINT` for both.

Measured A/B of the two parts (chain-pin-only vs chain-pin+spread, single
runs): spreading is ≈ neutral here — val 8n improved 331.0→318.7 s but most
other multinode cells moved 0 to +12% — unlike the chain pinning, which
carries the layer's whole win over base. The likely reason spreading buys
less than the serving-load argument suggests: the binder is the requesting
side's per-acquire latency and the serial spawner, not the home's serving
throughput. The layer keeps both parts — they are what hints alone can
express in this structure.

## Sizing

`-l` sets total work and `-b` the in-flight width — `-t` is dead (see
Parameters). Doubling `-l` linearly doubles EDT count (`3` per lookup, `+2`
per additional batch) and wall time; DB count and memory are fixed by
`-s`/`-n`/`-p`/`-w` alone and never move with `-l`.

- The campaign runs `-l 150000 -b 3456` at every node count (the sweep
  invariant: total logical quantity fixed). `-b 3456 = 32×108` is the
  master/worker window rule — one in-flight batch spanning the full campaign
  machine (32 nodes × 108 workers, the Dane anchor) — so the nominal width
  can occupy every worker at the largest geometry; smaller runs simply hold
  more of the window per worker.
- Throughput is spawn-serial-capped: the single `rankMultiLookupSpawner` loop
  creates chains at ~4 µs/lookup, so a batch's tail is creation-bound before
  it is compute-bound (~98 K lookups/s at 15 workers, ~88 K/s at 108 — more
  workers do not help a serial spawner). `-l 150000` is sized from that cap
  for a minutes-scale ceiling on the slowest arm×geometry.
- Because `nprocs` is hardcoded to 1, there is no work-partitioning axis
  across nodes at all: the one-time, one-rank data-generation phase never
  spreads, and multinode runs measure remote-acquire / wiring cost, not
  throughput scaling — unlike `_sharedDB`'s fork axis.

## Family shape (measured, 15w+1p × 1/2/4/8 nodes, `-l 150000 -b 3456`)

hinted, e2e seconds (kernel `Runtime:` ≈ e2e here — init is a fixed ~0.1 s).
The pole-checksum pin held in all 21 cells:

| arm | 1n | 2n | 4n | 8n |
|---|---|---|---|---|
| val_wb_nocomb | 1.6 | 218.8 | 247.7 | 318.7 |
| val_wb | 1.5 | 30.1 | 22.4 | 44.3 |
| inv_wb | 1.8 | 10.5 | 15.1 | 19.1 |
| excl_retain | 1.8 | 17.6 | 19.6 | 36.4 |

base val_wb_nocomb: 234.3 / 302.6 / 342.2 at 2/4/8n — the hinted layer wins
1.07-1.22× on val. A Dane-geometry single node (108w+4p) runs it in 1.5 s.
The arm separations dwarf the hint deltas: on this write-once,
read-fine-grained dataset at 8 nodes, VAL's re-validate-per-acquire costs
**16.7×** INV's covering reads (318.7 vs 19.1) and 8.8× EXCL; combining
recovers VAL to 7.2× better than plain VAL but still 2.3× behind INV. Every
arm is anti-scaling 2n→8n — one rank's serial spawner feeds all nodes, so
extra nodes only add wire distance. The 1n→2n cliff (1.6 s → 10.5-218.8 s)
is the fine-grain remote-acquire regime switching on: per-nuclide KB-scale
DBs, up to 963 acquires per fuel-material lookup.
