# CoMD_intel_chandra

*One SPMD rank per link cell — 50 000 "ranks" on a 60³ box, each a nine-deep
task chain per step, all of them joined by a global allreduce at every period.*
Source: `third_party/ocr-apps/apps/CoMD/refactored/ocr/intel-chandra/` (6 C
files, ~3k lines; `CoMD.c` is the whole DAG, `lj.c` the kernel).

## Overview

**Unsupported — not selectable.** Atom migration mutates two cells through
`DB_MODE_EW` dependences, and the exclusive-write guarantee they rely on is
one this runtime deliberately does not implement (EW maps onto per-node RW);
concurrent same-node moves then race the cells' non-commuting counter/compact
updates.  The sibling ports cover the same physics without EW —
`CoMD_sdsc` (one serial redistribute task) and `CoMD_intel_chandra_tiled`
(message-passing halos) are the family's contrasting pair, and future
optimized/restructured CoMD work starts from the tiled port.

Intel's OCR port of the ExMatEx CoMD proxy: FCC copper, velocity Verlet,
Lennard-Jones.  Its decomposition is the extreme of the family — the *link cell
is the rank*.  `mainEdt` factors the policy domains into a 3-D grid, maps the
cell grid onto it, and forks one `initEdt` per cell; from there every cell runs
its own SPMD program (private block, its own 27 halo channels, its own leaf in
the reduction trees) and never sees a per-phase fan-out EDT.

The result scalar is `Initial energy`, and that is the only energy this port
prints in a machine-readable form: `printThingsEdt` prints the initial value at
step 0 and then one status row per period, with no final-energy validation
line at all.  So the catalog scalar checks the lattice, the temperature rescale
and the first force evaluation — it does **not** check the integration.

What the program stresses is neither arithmetic (~17 atoms per cell against 27)
nor parallel width (width is the cell count, always ≫ workers) but **depth and
synchronisation**: ~9 dependent EDTs per cell per step, a global allreduce
whose result gates the step's finish event, and two rank-0-homed read-only
blocks that every cell acquires twice per step.

## Parameters

| flag | meaning | default | CLI reachability |
|------|---------|---------|------------------|
| `-x/--nx`, `-y/--ny`, `-z/--nz` | unit cells per dimension | 20 | ✓ parsed in `mainEdt`, carried in the command DB every EDT depends on — multinode-safe |
| `-N/--steps` | time steps | 100 | ✓ exact (no period quantisation); the requested step always ends the run.  `0` is legal — the initial state is printed and the run shuts down without integrating |
| `-n/--period` | steps between the kinetic-energy allreduce and a status row | 10 | ✓ any value; the last step reduces whether or not it is a period boundary (its status row is what reaches the shutdown event), so a final partial period prints one extra status row.  It is also the global-barrier interval |
| `-D/--dt` | time step (fs) | 1.0 | ✓ applied to both half kicks and the drift (`0.5·dt`, `1.0·dt`) |
| `-T/--temp` | initial temperature (K) | 600.0 | ✓ read from the command DB inside each cell's `comdEdt` |
| `-r/--delta` | initial random displacement (Å) | 0.0 | ✓ same path |
| `-l/--lat` | lattice constant (Å); `<0` = the potential's 3.615 | -1.0 | ✓ — moves the cell grid, see Sizing |
| `-e/--doeam` | use EAM | 0 | ✗ not implemented: `eam.c`'s `init_eam` is an empty stub that reports "EAM potentials are not implemented in this port" and returns an error; `main1Edt` ends the run on it |
| `-d/-p/-t` (`pot_dir`, `pot_name`, `pot_type`) | EAM tables | `pots`, auto, `funcfl` | ✗ only consumed by the unimplemented EAM path |
| `-h/--help` | print the option table | — | ⚠ prints the usage table instead of the parameter dump, then runs anyway |

Compile-time: `MAXATOMS` (64) fixes the cell payload, `MAXSPECIES` (1) the mass
table, `ARITY` (10, in the reduction library) the allreduce tree fan-in.  There
is no rank-grid argument — the "rank" count is derived from the box grid, not
chosen.

## Structure

With `lat = 3.615`, `cutoff = 5.7875`, `g_i = floor(0.62462·n_i)`,
`B = g_x·g_y·g_z` cells, `S` steps, `P` period, `K = ceil(S/P)` reducing steps
(every period boundary, plus the last step when it is not one), and
`Q = ceil((B−1)/10)` interior nodes of the reduction tree:

| object | count | size |
|--------|-------|------|
| `atomData_t` | `B` | 5656 B (64 slots: gid, species, r, p, f, u) |
| `linkCellH_t` | `B` | 504 B (bounds + 27+27 neighbour GUIDs) |
| `privateBlock_t` | `B` | 1552 B (6×27 event GUIDs, affinity hint, templates) |
| reduction private | `2B` | 368 B each |
| halo rendezvous blocks | `27B` | 48 B each (6 GUIDs) |
| command / shared / mass | 1 each | ~38 KB / 432 B / 32 B |
| DBs | `33B + 4` app blocks `+ 2(B−1) + (3+K)(3B−2)` reduction transients | — |
| events | `B·(157 + 17S − 2 + 4K) + 1` + `2(5(B−1)+Q)` reduction setup | 54 labeled sticky (27 directions × both endpoints) + 81 channel (position, redistribute, force) + 2 channel per cell at init; per step 12 in the body, 3 for the finish EDT, 2 for the loop, 4 per period |
| EDTs | `B·(13 + 9S − 1) + 3` + `(2+3+K)(2B+Q−1)` reduction + `M` moves | init 13/cell; steady 9/cell/step at `P = 1` |

`move_edt` is the one data-dependent count: a cell's `redistributeAtomsEdt`
spawns one move per direction that actually loses an atom (0–26), so `M` carries
a small tail on the EDT counter while DB and event counts stay exact.  A
direction that leaves the simulation box names the cell on the opposite face and
the atom travels as its periodic image, so a departing atom is moved once and
its direction stops being flagged.  `M` is small but not zero at production
sizes: the calibrated `-x 60 -y 60 -z 60 -N 6` moves 3178 atoms in 2993 move
EDTs — 0.4 % of the atoms over six steps at 600 K, and reproducible run to run.
At `-x 8 -y 8 -z 8` over two or four steps nothing migrates (`M = 0`): the same
rate applied to 2048 atoms and a third of the steps rounds to no event at all,
which is why the verify pair's counters are exact.  A move compacts the source
cell by swapping its last atom into the freed slot, and the scan re-examines
that slot, so every atom bound for the same direction leaves in the same pass
rather than one step later.

The per-cell init constant is `157`, where counting the source's distinct
`ocrEventCreate` calls suggests `103`.  Twenty-seven of the difference are the
force-acknowledgment channels — the third of the three per-direction channels
(see Wiring).  The rest is double-counted rendezvous.  `comdInitEdt`'s halo
rendezvous calls `ocrEventCreate(GUID_PROP_CHECK)` on each of a
cell's 27 labeled sticky indices from *both* sides — the owning cell's "send"
create and the neighbouring cell's "receive" create land on the same index —
and the shim increments `NUM_EVENT_CREATE` before the check resolves the race
(`arts_event_create`, `event.c:219-221`), so both attempts count: 54 sticky
creates per cell, not 27 (+27 per cell, i.e. +1728 at the calibrated small
grid).  The reduction library's own labeled-GUID pairing has the identical
shape — `reductionEdt`'s parent-side `recvEVT` create
(`reduction.c:481-482`) and the child's `reductionSendChannelEdt`'s `sendEVT`
create (`reduction.c:271-272`) touch the same tree-edge index — so the setup
term is undercounted by one edge-event per non-root cell per object:
`2(4(B−1)+Q)` is `2(5(B−1)+Q)` (+2(B−1) per run, i.e. +126 at the calibrated
small grid).  DB and EDT counts were already exact; only events needed the fix.

Worked numbers for the calibrated `-x 60 -y 60 -z 60 -N 6` (default period 10,
so `K = 1`) → `g = 37`, `B = 50 653`, `S = 6`, `Q = 5066`: **≈4.0 M EDTs**,
**≈2.4 M DBs**, **≈13.7 M events**, 864 000 atoms (17.1 of 64 slots per cell),
and **≈494 MB** of live payload (9.5 KB per cell, of which 5656 B is atom
data).  At `-n 1` the same six steps cost ≈4.5 M EDTs, ≈3.1 M DBs and ≈14.8 M
events — the extra is five more allreduces.  For a seconds-scale `-x 8 -y 8
-z 8 -N 2 -n 1`: `g = 4`, `B = 64` → ≈2.9k EDTs, ≈3.2k DBs, 13 253 events.

Counter cross-check: verified (1 node, `-x 8 -y 8 -z 8 -N 2 -n 1` vs `-x 8 -y 8
-z 8 -N 4 -n 1`): measured absolutes EDT 2862/4282, DB 3193/3573, EVT
13253/15941; subtracting the runtime's constant baseline (+1 EDT, +1 DB, +0
EVT per run) and counting the app's own `mainEdt` (the shim-created EDT that
`main1Edt`/`wrapUpEdt` hang off of, folded into the `+3` constant above)
against the formulas gives an exact match in both absolute value and delta —
`move_edt`'s data-dependent tail is `M = 0` at this size in both runs.

## Wiring

`mainEdt → main1Edt` sizes the grid and forks `B` `initEdt`s.  Each cell then
runs `initEdt → comdInitEdt → channelInitEdt → comdEdt`.  `comdInitEdt` is the
halo bootstrap: for each of the 27 directions it creates a **labeled sticky
event** at index `27·myCell + dir` of a `27B`-wide GUID range, satisfies it with
a 48-byte block holding {its own position channel, its own redistribute channel,
its own force channel, its `linkCellH` GUID, its `atomData` GUID}, and registers
on the *neighbour's* labeled event at index `27·nbr + (26−dir)` — a symmetric
rendezvous in which whichever side reaches `ocrEventCreate(GUID_PROP_CHECK)`
first wins.  `channelInitEdt` unpacks the 27 replies into the cell's
neighbour-GUID cache.  All three channels are pairwise and one-per-direction, so
each carries exactly one satisfy and one dependence per step.

Per cell per step, `timestepLoopEdt` creates a **finish** `timestepEdt` and its
own successor, and `timestepEdt` builds the step:

- `advanceVelocityEdt` (own `atomData` RW) → `advancePositionEdt` (own
  `atomData` RW, mass RO, plus the **27 incoming force-acknowledgment channel
  events** of the previous step) — chained through sticky output events.
- the position EDT's output satisfies the cell's **27 outgoing position channel
  events**; `redistributeAtomsEdt` (FINISH; own `linkCellH` + `atomData` RO)
  waits on the **27 incoming** ones, then spawns `move_edt`s that take its own
  and a neighbour's `atomData`/`linkCellH` **EW** (→ ARTS per-node-exclusive RW).
- its finish event satisfies the cell's 27 outgoing redistribute channels;
  `ljforce_edt` (FINISH, 54 slots: own `atomData` RW at slot 13, 26 neighbour
  `atomData` RO, 27 `DB_MODE_NULL` control edges) waits on the incoming ones.
- the force EDT's finish event satisfies the cell's **27 outgoing force
  channels** — the acknowledgment the next step's `advancePositionEdt` waits
  on, so no cell moves atoms a neighbour is still reading.
- `advanceVelocityEdt` again; every `P` steps `kineticEnergyEdt` launches the
  allreduce and `printThingsEdt` waits on its result.

The neighbour handshake is therefore a full round trip: positions are published
before forces are computed *and* force completion is published back before the
positions change again.  `comdEdt` opens the run with the same three
publications (random displacement → position, redistribute, force), which is
the generation the first `timestepEdt` consumes.

The run ends on one ONCE event with one consumer: `main1Edt` creates
`finalOnceEVT` and hangs `wrapUpEdt` (`ocrShutdown`) on it, and cell 0's *last*
`printThingsEdt` — the task that prints the final status row — publishes its
output event into it, the same event→event idiom the halo channels use.  So the
shutdown is ordered behind the last print rather than behind the reduction that
feeds it, and a zero-step run reaches it through the step-0 print.

DB concurrency:

- **`atomData[b]`** — one writer (its own chain), up to **26 concurrent RO
  readers** (neighbour force EDTs), and up to 26 `move_edt` writers **from other
  ranks**.  It is the contention point and the only multi-node-writer block.
  The RO readers overlap the block's own force EDT, which writes only `f`/`u`
  while they read only `r`; the writes that would collide with a reader —
  `advancePositionEdt` and `move_edt` — are ordered against them by the force
  channels.  One overlap remains unsynchronised — see the *Correctness caveats*
  note under Sizing.
- **command block (~38 KB) and shared header (432 B)** — RO by *every* cell's
  `timestepLoopEdt` and `timestepEdt`, i.e. `2B` acquires per step, all against
  blocks homed on rank 0.  The 32-byte mass block is the same shape.
- **channel events** are strictly pairwise (`nbSat = nbDeps = 1`), so the halo
  is point-to-point signalling, not a shared datablock.

## Flow

Per cell per step the critical path is `timestepLoop → timestep → velocity →
position → redistribute → moves → force → velocity → (kinetic energy →
allreduce → print)` — nine dependent EDTs of a few microseconds each.  Parallel
width is `B` (one runnable task per cell), so at any realistic grid the machine
is saturated many times over; the cost is depth, not width.

Three serialisations dominate.  First, **there is no lookahead**: `timestepEdt`
is a FINISH EDT and `timestepLoopEdt` for step `t+1` depends on its finish
event, so a cell builds step `t+1` only after step `t` is fully drained.
Second, the step both opens and closes on a neighbour rendezvous — the force
acknowledgment gates `advancePositionEdt`, the position publication gates
`redistributeAtomsEdt`, the redistribute publication gates `ljforce_edt` — so a
cell can lead its neighbours by at most one force pass and the wavefront stays
tight without a barrier.  Third, every
`P`th step the finish scope contains a `printThingsEdt` that waits on the
kinetic-energy **allreduce over all `B` cells** — a 10-ary tree, `≈2·log₁₀ B`
(10 hops at the calibrated size) of 24-byte messages up and down — so the whole
grid re-synchronises and the slowest cell sets the pace for all of them.  At the
calibrated `-n 1` that barrier is on **every** timestep.

The serial preamble is `main1Edt`'s loop: `B` affinity queries and `B` EDT
creations on one worker, followed by `B` cells each creating 27 labeled events,
27 blocks and 81 channel events — 1.4 M blocks and 5.5 M events of pure
bootstrap at the calibrated size, before any physics runs.

## Placement (as-born)

There is no `OCR_APP_OPTIMIZED_PLACEMENT` layer in this port; the placement
below is the program's own and is compiled in.  `main1Edt` calls
`ocrAffinityCount(AFFINITY_PD)`, factors the rank count into a 3-D grid
(`splitDimension`) and gives each `initEdt` the `OCR_HINT_EDT_AFFINITY` of the
policy domain that owns its cell (`getPoliyDomainID`, a block partition per
axis).  Every later EDT of that cell reuses `ocrAffinityGetCurrent` cached in
its private block, and every DB it creates passes `NULL_HINT` — which, with
home = creating rank, lands it on the owning rank anyway.  **A cell's entire
pipeline and all of its state therefore live on one rank for the whole run.**

What crosses ranks: the block *surface* (~11 % of the `26·B` neighbour edges for
8 ranks over a 37³ grid, 0 % at one node), the `move_edt`s that write into a
neighbour rank's cell, and — this one is not a surface effect — the command
block, shared header and mass block, which are created in `mainEdt`/`main1Edt`
and so are **homed on rank 0 while `2B` tasks per step read them**.  The halo
rendezvous events are labeled GUIDs from a round-robin range, so the bootstrap
handshake is spread over all ranks regardless of cell ownership; that cost is
one-off.

`splitDimension` factors every rank count into a grid that covers it, so no rank
is left without cells.

## Sizing

`n_x,n_y,n_z` set width *and* memory: cells `≈ (0.6246·n)³`, atoms `4·n³`, and
8.9 KB of runtime-resident state per cell regardless of occupancy (~17 of 64
atom slots used).  Task grain is fixed by the cutoff — a force EDT is always one
cell against 27 — so `n` buys more tasks, never bigger ones.  `-N` scales time
linearly; `-n` scales the number of global barriers and nothing else (the
integrator is a plain Verlet chain either way).

- **1 node × 15 workers**: `-x 24` (`g = 14`, `B = 2744`, 24 MB) to `-x 32`
  (`g = 19`, `B = 6859`, 61 MB) already gives hundreds of cells per worker;
  larger grids buy wall time, not utilisation.
- **8 nodes × 120 workers**: the calibrated `-x 60` gives `B = 50 653`, ~6.3k
  cells per rank and ~422 per worker, each rank owning an ~18.5³ block whose
  surface is the only neighbour traffic.
- The calibrated `-x 60 -y 60 -z 60 -N 6` is sized so the 1-node run costs
  ≈4.0 M tasks and 494 MB resident.  It runs at the default period 10, so the
  six steps carry a single global barrier (the last step always reduces).  `-n
  1` puts one on **every** timestep — a choice about what is being measured,
  and worth making explicitly if the barrier is the subject.
- Bootstrap, not the timestep loop, is what makes large grids expensive here:
  every cell costs 27 labeled events, 81 channel events and 27 blocks before the
  first force.  Doubling `n` multiplies that by ~8.

**Correctness caveats.**  Two overlaps on `atomData[b]` and one on the
allreduce still carry no dependence and are decided by the schedule rather than
by the DAG:

- `redistributeAtomsEdt` publishes its completion only to the 26 cells it can
  move atoms *to*, but a `move_edt` writes the **destination** cell, which a
  third cell two steps away is reading in the same generation's `ljforce_edt`.
  So a cell at distance 2 can add an atom to a neighbour after that neighbour
  has been read.  Reachable only when atoms migrate (`M > 0`), which excludes
  the `-x 8` verify sizes but not the calibrated one.
- `move_edt` takes both cells **EW**, which OCR defines as exclusive but which
  ARTS maps onto per-*node*-exclusive `DB_MODE_RW`, so several move EDTs on one
  node hold the same cell concurrently and their `nAtoms` read-modify-writes
  race.  `EW` is required here rather than decorative: a cell's redistribute
  spawns up to 26 moves that all take that cell as the source, and a cell can be
  the destination of up to 26 more, while `move_atom` does a non-atomic
  `--nAtoms` / `++nAtoms` pair and appends at the index it then increments.
  Measured at `-x 8 -y 8 -z 8 -N 50 -n 10 -T 3000`, 14 workers, 5 repeats per
  setting: the reported atom count settles off `4·n³` in 12 of 20 runs
  (2043–2053 against 2048) and is not reproducible; at one worker, and with the
  move bodies serialised in-app, it is exactly 2048 in every run.  The default
  `-r 0` does not avoid it — thermal drift alone migrates 3178 atoms at the
  calibrated arguments — but with only ~3k moves spread over 50 653 cells the
  collision rate is low enough that no count drift was observed there.
- The Ke allreduce honors the reduction library's one-in-flight contract:
  the first `timestepLoopEdt` is gated on the *consumer* of the initial
  fold's answer (`printThingsEdt`, which takes `returnEVT`), and every later
  step is covered by `timestepEdt`'s FINISH scope containing its own
  `printThingsEdt`.  This edge was originally missing (the gate was the
  *launcher*, `kineticEnergyEdt`), which let the initial-energy fold and the
  first timestep's fold each take the wrong generation's contribution by
  equal and opposite amounts (reported atom counts of 2049 then 2047 around
  2048) — deterministically, even at one worker.  The defect class was
  reproduced against the unmodified library with a standalone 64-rank
  driver (12/12 wrong with one unordered transition, 0 with none).

`Initial energy` is bit-reproducible: identical at 1/2/4/14 workers, across
20 repeats at `-x 8 -y 8 -z 8 -N 2`, at 2 nodes, and at the calibrated
`-x 60 -y 60 -z 60 -N 6` (all `-1.166063303478`).  It was not before the force
acknowledgment described under Wiring existed — the value then wandered over
~1.1·10⁻⁵ because 48–63 of every 320 force passes read a neighbour that had
already advanced into the next step.
