# CoMD_sdsc2

*The same molecular dynamics, re-decomposed: two datablocks per link cell, a
neighbour-GUID cache inside each cell, local atom migration instead of a global
redistribute, and a rank-block placement the program asks for itself.*
Source: `third_party/ocr-apps/apps/CoMD/refactored/ocr/sdsc2/` (8 C files, ~2.3k
lines; `simulation.c` builds the graph, `timestep.c` + `lj.c` run it,
`reductions.c` builds the trees).

## Overview

**Unsupported — not selectable.** Per-cell atom migration takes both cells
`DB_MODE_EW` and needs one-EDT-at-a-time exclusivity this runtime
deliberately does not implement (EW maps onto per-node RW).  The family's
selectable pair is `CoMD_sdsc` (serial redistribute) and
`CoMD_intel_chandra_tiled` (message-passing halos); future
hinted/restructured CoMD work starts from the tiled port.

SDSC's second OCR port of the ExMatEx CoMD proxy — same physics as `CoMD_sdsc`
(FCC copper, velocity Verlet, Lennard-Jones), same `Final energy` scalar, a
different program.  What changed, in the order it matters:

- **Per-cell state is split in two.** `atomData_t` (5208 B) holds what changes
  every step; `linkCellH_t` (744 B) holds geometry plus a *cached array of the
  26 neighbours' GUIDs*, so a cell wires its own halo without a global list.
- **The whole-grid join is gone.** `CoMD_sdsc` redistributed atoms in one EDT
  holding every cell read-write; here each cell's `exchange_edt` is a FINISH EDT
  that spawns a `move_edt` only toward neighbours that actually received an
  atom — data-dependent, purely local migration.
- **Barriers became neighbour-local signals.** Force and position EDTs chain
  through per-cell signal blocks (`ff`) and exchange output events (`pf`), so a
  cell advances as soon as its 26 neighbours have, not the whole grid.
- **Reductions became 8-ary trees** of COUNTED events (`reductions.c`, `FANIN
  8`) instead of one `B+1`-dependence sum EDT.
- **The shared simulation header is read-only in the hot path**, not `RW` on
  every force EDT.
- **The program places itself**: it queries the policy-domain count and maps the
  cell grid onto a 3-D grid of ranks (see Placement) — base, not an added
  hint layer.
- **`-N` is exact** (`CoMD_sdsc` quantizes it to `--period`), and **EAM is
  not implemented** (`eam.c` is an empty shell that says so and fails).

Halo sharing is still the stress — 26 concurrent readers per cell — joined now
by multi-writer datablocks (a cell is written by move EDTs from up to 26 other
cells, on other ranks) and a deep event tree.

## Parameters

| flag | meaning | default | CLI reachability |
|------|---------|---------|------------------|
| `-x/--nx`, `-y/--ny`, `-z/--nz` | unit cells per dimension | 20 | ✓ grid via the simulation DB, lattice counts via EDT paramv — multinode-safe |
| `-N/--steps` | time steps | 100 | ✓ exact (`ds = min(period, steps−step)` per block) |
| `-n/--period` | steps per block: one global reduction and one status line per block | 10 | ✓ blocking only — a block closes with a half velocity kick and the next opens with the matching one, so the trajectory does not depend on how the steps are divided into blocks |
| `-D/--dt` | time step (fs) | 1.0 | ⚠ applied everywhere except the *initial* half kick, which is hard-coded `0.5` (`ukvel_edt`) instead of `0.5·dt` |
| `-T/--temp` | initial temperature (K) | 600.0 | ✓ carried in `PRM_FNC_init` paramv to every rank |
| `-r/--delta` | initial random displacement (Å) | 0.0 | ✓ same path |
| `-l/--lat` | lattice constant (Å); `<0` = the potential's | -1.0 | ✓ |
| `-e/--doeam` | use EAM | 0 | ✗ not implemented: `init_eam` is commented out; it reports "EAM potentials are not implemented in this port" and the initialization-error path ends the run immediately |
| `-d/-p/-t` (`pot_dir`, `pot_name`, `pot_type`) | EAM tables | `pots`, auto, `funcfl` | ✗ only consumed by the unimplemented EAM path |
| `-h/--help` | print the option table | — | ✓ (prints, then runs anyway) |

Compile-time: `MAXATOMS` (64) fixes the cell payload, `MAXSPECIES` (1) the mass
table, `FANIN` (8) the reduction arity.  `ENABLE_EXTENSION_AFFINITY` is defined
for every benchmark app, so the placement code below is live.

## Structure

With `g_i = floor(n_i·lat/cutoff) = floor(0.6246·n_i)`, `B = g_x·g_y·g_z` cells,
`S` steps, one block when `S ≤ period`, and `T(B)` = the node count of an 8-ary
reduction tree over `B` leaves (`Σ_d ceil(B/8^d)`, ≈ `B/7`):

| object | count | size |
|--------|-------|------|
| `atomData_t` DBs | `B` | 5208 B (64 slots: r, p, f, u, species, neighbour links) |
| `linkCellH_t` DBs | `B` | 744 B (bounds, gids, 26+26 neighbour GUIDs) |
| per-cell schedule DBs | `B` (init only) | 208 B |
| per-cell signal DBs (`ff`) | `B` per block | 8 B |
| per-cell scalars (`vcm`, `ek`, `uk`) | `3B` init + `B` per block | 8–40 B (`vcm` and `uk` each carry a `u64` live atom count after their reals) |
| simulation header / mass / reduction header | 1 each | 368 B / 32 B / 40 B |
| EDTs | `6 + 4T(B) + 5B + B·(3S+1)` + moves | init: `FNC_init` + sched + vcm + force + ukvel per cell; steady: position + exchange + forcevel per cell per step |
| DBs | `17 + 9B` | independent of `S` |
| events | `10 + 4T(B) + 7B + 3B·S` | 2 per exchange (FINISH + output), 1 per force, `B+1` COUNTED per tree |

`move_edt` is the one count that is **not** static: an exchange spawns one move
per neighbour direction that actually receives an atom (0–26), so
`NUM_EDT_CREATE` carries a data-dependent tail while DB and event counts stay
exact.  A direction that leaves the simulation box names the cell on the
opposite face, and the atom is tested and stored as its periodic image, so a
departing atom is moved once and its direction stops being flagged.  A move
compacts the source cell by swapping its last atom into the freed slot, and the
scan re-examines that slot, so every atom bound for the same direction leaves in
the same pass rather than one step later.

The DB constant is `17`, not `11`: six of the one-time DBs are each
function's own `DBK_affinityGuids` — a `ocrGuid_t[affinityCount]` array
queried via `ocrAffinityGet` and destroyed again a few lines later, present
in `init_simulation`, `EDT_init_fork`, and **every** `build_reduction` call
(`reductions.c:47`) — one per reduction tree built. For a single-block run
(`S ≤ period`, the scope of this closed form) that is `init_simulation`(1)
+ `EDT_init_fork`(1) + `EDT_init_fork`'s three `build_reduction` calls
(ured/tred/vred, +3) + `period_edt`'s own one `build_reduction` call (ured
only, +1) = 6, on top of the `5` non-affinity one-time DBs (simulation
header, mass, reduction header, `DBK_linkCellGuidsH`, `DBK_atomDataGuidsH`)
already counted in the old `11`. A multi-block run would add one more
`DBK_affinityGuids` per additional block (`period_edt` runs once per
block), which this single-block-scoped formula does not need to express.

Worked numbers for the calibrated `-x 44 -y 44 -z 44 -N 8` → `g = 27`,
`B = 19683`, `T = 2814`, one block of 8 steps: ~602k EDTs (plus moves), 177k
DBs, 621k events, 340 736 atoms, **117 MB** of cell payload.  For the `expect`
args `-x 4 -y 4 -z 4 -N 2`: `B = 8`, `T = 1` → 106 EDTs, 89 DBs, 118 events.

Counter cross-check: verified (1 node, `-x4 -y4 -z4 -N1` vs `-x6 -y4 -z4
-N3`): NUM_EDT_CREATE 84 → 200, NUM_DB_CREATE 90 → 126, NUM_EVENT_CREATE
94 → 214 — exactly `6+4T(B)+5B+B(3S+1)+m` / `17+9B` / `10+4T(B)+7B+3BS`
(app values 82+m₁/198+m₂, 89/125, 94/214) plus the runtime's constant +1
EDT/+1 DB/+0 EVT baseline, with the data-dependent move count `m₁=m₂=1`
in both runs. DB was short by exactly 6 (the `DBK_affinityGuids` DBs
above); EVT was already exact.

## Wiring

`mainEdt → FNC_preInit → FNC_setUpGraph` allocates and shapes the graph;
`EDT_init_fork` (a FINISH EDT) creates one `FNC_init` per cell plus three
chained reduction trees (centre-of-mass velocity → temperature rescale →
energy), each root feeding the next through the reduction header.  `period_edt`
then builds one block's worth of per-cell EDTs and its own successor.

Per cell per step: `position_edt` (29 slots: own `atomData` RW, mass RO, its own
and 26 neighbours' `ff` signal blocks RO) → `exchange_edt` (FINISH, 2 slots:
own `atomData` and `linkCellH` RW, wired by the position EDT) → `move_edt`s →
`ljforcevel_edt` (56 slots: own `atomData` RW, 26 neighbour `atomData` RO,
simulation header RO, own `ff` RW, and 27 `DB_MODE_NULL` control edges on the
exchange output events of itself and its neighbours).

DB concurrency:

- **`atomData[b]`** — one writer (its own pipeline) and up to **26 concurrent
  readers** (neighbour force EDTs): the same DB-granular read/write overlap as
  the first port, benign because the writer touches `f`/`u`/`p` and readers
  touch `r`.  It is additionally taken `EW` (→ ARTS RW) by up to 26 `move_edt`s
  **originating on other ranks** — the only datablock in either port with
  writers from many nodes.  The FINISH scope keeps every move ahead of the next
  force round, but the exclusivity the moves ask for does not survive the
  mapping: OCR `EW` becomes per-*node*-exclusive ARTS `RW`, so several move EDTs
  on one node hold the same cell at once and their `atoms` read-modify-writes
  race.  `EW` is load-bearing here, not decorative — `move_atom` appends at the
  index it then increments and compacts the source by moving its last atom into
  the freed slot, so two concurrent moves on one cell lose an atom either way —
  and the overlap is observed: instrumented at `-x 8 -y 8 -z 8 -N 50 -n 50
  -T 3000 -r 0.5`, move EDTs holding the same cell simultaneously are counted in
  the hundreds per run at 14 workers, none at one worker, and the live atom
  inventory then drifts off `4·n³`.  Reachable whenever atoms migrate, which at
  `expect_args` they do not (`M = 0`, so the pin is safe) but at the calibrated
  `-x 44 -y 44 -z 44 -N 8` they do (1463 move EDTs, 1528 atoms).  `end_edt`'s
  "no atoms lost" line does see it: the live total is re-summed from the cells
  through the `ured` tree (each `uk` leaf carries its cell's atom count
  alongside the two energies) and compared with the count the initial `vred`
  fold produced, so a lost decrement shows up as the warning.
- **`ff[b]`** — 8 bytes, RW by one force EDT, RO by 27 position EDTs per step.
- **simulation header and mass block** — RO fan-out to all `B` force/position
  EDTs each step (pure broadcast; the header is RW only in `period_edt` and at
  the reduction roots).  **schedule DBs** — init only, each RW by 26 different
  `FNC_init` EDTs on different ranks (the source argues the writes are disjoint).

Because concurrent `move_edt`s append into a cell in whatever order the runtime
grants them, the *slot order* of atoms inside a cell is not reproducible, and
neither is the order in which their energies are summed — the structural reason
this entry carries a looser `1e-4` tolerance than the first port's `1e-6`.
Measured at the calibrated arguments: `Final energy` spans 9·10⁻⁷ over three
repeats while the migration census is bit-identical (1463 moves, 1528 atoms) in
all of them, so that band is re-association and slot order, not lost atoms.

## Flow

Two nested rhythms.  Inside a block the DAG is a **wavefront**: cell `b` may
start step `s+1` once itself and its 26 neighbours have finished step `s`, so
distant cells drift apart in time and there is no whole-grid barrier.  Maximum
concurrent EDTs is on the order of `B` (each cell has one runnable task at a
time, up to 26 moves during migration).  Between blocks the reduction tree is a
**global barrier**: every cell contributes a leaf, `T(B)` internal EDTs fold it
in `ceil(log8 B)` levels (5 at the calibrated size), and the root satisfies the
next `period_edt`.

The serial sections are all *graph construction*, not physics: `FNC_preInit`
creates `2B` datablocks in one loop on rank 0; `EDT_init_fork` creates `2B` more
datablocks, `B` initializer EDTs and three trees in a single task; and
`period_edt` builds `3B` EDTs, `B` blocks and roughly `85·B` dependences per
block — about 1.7 M `ocrAddDependence` calls at the calibrated size, on one
worker.  A small `--period` multiplies that cost.

## Placement (base)

There is no `OCR_APP_OPTIMIZED_PLACEMENT` layer here; the placement below is
the program's own and is compiled in.  `init_simulation` asks for the
policy-domain count (= rank count), factors it into a 3-D rank grid
(`splitDimension`) and maps each cell to a rank by block-partitioning the cell
grid in each dimension (`getPoliyDomainID`).  Both cell datablocks and the
schedule block are created with that `OCR_HINT_DB_AFFINITY`, i.e. **homed on
the owning rank**; `FNC_init` and the per-step position/exchange/force EDTs take
`OCR_HINT_EDT_AFFINITY` from `ocrAffinityQuery` on the cell's GUID (its home
rank), and every EDT they spawn uses `ocrAffinityGetCurrent` — so an entire
cell's pipeline stays on the rank that owns the cell for the whole run.

What crosses ranks is therefore only the block *surface*: of the 26 neighbour
`atomData` blocks a force EDT reads, only those in another rank's block are
remote (~14 % of the `26·B` neighbour edges for 8 ranks over a 27³ grid), plus
the `move_edt`s that write into a neighbour rank's cell.  Everything else — the
simulation header, mass block, reduction roots — is small and read-mostly.
Unhinted objects are few: the guid arrays, `leaves_g`, the reduction header and
`period_edt` itself keep the defaults (EDT round-robin, DB home = creating rank).

## Sizing

`n_x,n_y,n_z` set width and memory: cells `≈ (0.6246·n)³`, atoms `4·n³`,
`5952 B` of payload per cell independent of occupancy (~17 of 64 atom slots
used).  Task grain is fixed by the cutoff (a force EDT is always one cell against
27), so `n` buys tasks, not bigger tasks; `-N` scales time and adds no memory.
Unlike the first port there is no dependence-count ceiling — the widest EDT has
56 slots — so grids are limited only by memory.

- **1 node × 15 workers**: `-x 20` (`B = 1728`, 10 MB) up to `-x 28`
  (`B = 4913`, 29 MB) keeps every worker several hundred cells deep.
- **8 nodes × 120 workers**: `-x 44` gives `B = 19683`, ~164 cells per worker,
  each rank owning a ~14³ block whose surface is the only remote traffic.
- The calibrated `-x 44 -y 44 -z 44 -N 8` is sized so the 1-node run takes
  minutes (≈6·10⁵ tasks, 117 MB resident) and so that 8 steps ≤ the default
  period — one block and one global barrier.  Splitting the same steps into more
  blocks costs barriers and graph construction, not accuracy.
- `splitDimension` factors every rank count into a grid that covers it, so no
  rank is left without cells.
