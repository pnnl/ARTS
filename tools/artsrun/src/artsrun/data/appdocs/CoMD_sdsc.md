# CoMD_sdsc

*Molecular dynamics on a link-cell grid — one datablock per cell, one EDT per
cell per phase, and a whole-grid join in the middle of every step.*
Source: `third_party/ocr-apps/apps/CoMD/refactored/ocr/sdsc/` (9 C files, ~2.3k
lines; `comd.c` drives, `timestep.c` + `lj.c` + `cells.c` hold the DAG).

## Overview

CoMD is the ExMatEx molecular-dynamics proxy: an FCC copper lattice integrated
with velocity Verlet under a Lennard-Jones (default) or EAM potential.  Space is
cut into *link cells* at least one cutoff wide, so the force on an atom needs
only its own cell and the 26 neighbours.  This is SDSC's first OCR port: one
datablock per cell, one EDT per cell per phase, and a barrier between phases.

The result scalar is `Final energy` — total energy per atom after the last step,
printed by `validate_result` beside the initial energy, so a value off by more
than the catalog's `1e-6` means the integration went wrong, not that the machine
was slow.  Arithmetic per task is real but modest (~17 atoms per cell against 27
cells); what the program stresses is **halo sharing** — every cell block is read
by 26 tasks while its owner writes it — and two whole-grid serializations per
step: the atom-redistribution EDT and a read-write dependence on one shared
simulation block.

## Parameters

| flag | meaning | default | CLI reachability |
|------|---------|---------|------------------|
| `-x/--nx`, `-y/--ny`, `-z/--nz` | unit cells per dimension | 20 | ✓ parsed in `mainEdt`, stored in the simulation DB that every EDT depends on — multinode-safe |
| `-N/--steps` | requested time steps | 100 | ⚠ honoured only at period boundaries: the run executes `period·ceil(steps/period)` steps, so `-N 2` with the default period simulates **10** |
| `-n/--period` | steps per outer block, status print interval | 10 | ✓ — and it, not `-N`, is the real step quantum |
| `-D/--dt` | time step (fs) | 1.0 | ✓ positions use `dt/mass`, the velocity kicks `0.5·dt` (block ends) and `dt` (inside a block) |
| `-T/--temp` | initial temperature (K) | 600.0 | ✓ applied in `set_temperature` during init |
| `-r/--delta` | initial random displacement (Å) | 0.0 | ✓ |
| `-l/--lat` | lattice constant (Å); `<0` = the potential's | -1.0 | ✓ — moves the cell grid, see Sizing |
| `-e/--doeam` | use EAM instead of Lennard-Jones | 0 | ✓ but reads `pot_dir/pot_name` from disk at init; needs the `pots/` dataset in the working directory |
| `-d/-p/-t` (`pot_dir`, `pot_name`, `pot_type`) | EAM table selection | `pots`, auto, `funcfl` | ✓ under `-e`, dead otherwise |
| `-h/--help` | print the option table | — | ✓ (prints, then runs anyway) |

Compile-time only: `MAXATOMS` (64) fixes the per-cell atom capacity and hence
the DB payload; the LJ parameters (`sigma` 2.315, `epsilon` 0.167, cutoff
`2.5·sigma`, lattice 3.615 Å) are constants of `init_lj`.  `sanity_checks`
rejects a box smaller than two cutoffs, i.e. `n_i ≥ 4`.

## Structure

With `g_i = floor(n_i·lat/cutoff) = floor(0.6246·n_i)` cells per dimension,
`B = g_x·g_y·g_z` cells, `P` = period, `K = ceil(steps/period)` blocks:

| object | count | size |
|--------|-------|------|
| cell DBs (`box`) | `B`, created serially in `mainEdt` | 5896 B each (64 atom slots: gid, r, p, f, U, a) |
| simulation DB | 1, dependence of nearly every EDT | 1024 B |
| cell-GUID list, timer, command | 3 | `8B` B, 352 B, ~38 KB |
| per-cell scalar results | `B` per force phase, `B` per block-end KE | 8 B each, destroyed by the reducer |
| EDTs total | `(2B+6) + K·(P·(3B+8) + 2B+5)` | `mainEdt` + 4 control EDTs + `3(B+1)` fan-outs + 1 redistribute per step |
| DBs total | `(3B+6) + K·(P·(B+1) + B+1)` | dominated by the `B` cells |
| events total | `2 + K·(P·(2B+4) + B+2)` | ONCE events plus one output event per fan-out EDT |

Per step the fan-outs are: advance-velocity (`B` + 1 join), advance-position
(`B` + 1), redistribute (**1** EDT), force (`B` + 1 join). The leading
`(2B+6)` is `mainEdt`(1) + `main_edt2`(1) + the initial force fan-out
(`B+1`) + `main_edt3`(1) + the initial kinetic-energy fan-out (`B+1`) +
`end_edt`(1, created once by the final block's `bot_edt`) — `mainEdt`
itself is easy to miss since it never appears as a Wiring-section actor
(it only ever creates `main_edt2`), but it is a genuine, distinct
`NUM_EDT_CREATE`-counted instance like every other app's entry EDT.

Worked numbers for the calibrated `-x 36 -y 36 -z 36 -N 2` → `g = 22`,
`B = 10648`, `P = 10`, `K = 1`, i.e. **10 simulated steps**: ~362k EDTs, ~149k
DBs, ~224k events, 186 624 atoms, **63 MB** of cell payload (live for the whole
run) plus ~85 KB of transient 8-byte results per phase.  For the `expect` args
`-x 4 -y 4 -z 4 -N 2`: `B = 8`, 363 EDTs, 129 DBs, 212 events.

Counter cross-check: verified (1 node, `-x4 -y4 -z4 -N1 -n1` vs
`-x6 -y4 -z4 -N3 -n1`): NUM_EDT_CREATE 76 → 250, NUM_DB_CREATE 49 → 121,
NUM_EVENT_CREATE 32 → 128 — exactly `(2B+6)+K(P(3B+8)+2B+5)` /
`(3B+6)+K(P(B+1)+B+1)` / `2+K(P(2B+4)+B+2)` (app values 75/249, 48/120,
32/128) plus the runtime's constant +1 EDT/+1 DB/+0 EVT baseline. DB/EVT
were already exact; the original EDT formula's leading `(2B+5)` was short
by exactly 1: `mainEdt` (created by the ARTS-native `main_edt` bootstrap
in `benchmarks/ocr_shim/arts_ocr.c:2194`) was never counted as one of the
app's own EDTs. Corrected leading constant: `2B+6`.

## Wiring

`mainEdt` builds the lattice natively, then hands a chain of control EDTs the
timer DB (RW), the simulation DB and the cell-GUID list.  Each control EDT
creates the next one and a fan-out whose join EDT satisfies the next control
EDT's last slot; there are no channel or latch events, only ONCE events and EDT
output events.

- **force** (`lj_edt`, 28 slots): simulation **RW**, own cell **RW**, 26
  neighbour cells RO.  Writes its cell's `f`/`U` and wires an 8-byte energy DB
  straight into the reducer's slot.
- **advance-velocity** (`av_edt`): simulation RO, own cell **RW**, a shared
  8-byte `dt` block **RW**; **advance-position** (`ap_edt`): simulation RO, own
  cell **RW**.
- **redistribute** (`redistribute_edt`, `B+1` slots): **every** cell **RW**
  plus the simulation DB **RW**, in one task.
- **kinetic energy** (`ke_edt`): simulation **RW**, own cell RO; `ke_red_edt`
  sums `B` scalars and satisfies a ONCE event.

DB concurrency, in decreasing order of pain: the **simulation block** is taken
RW by all `B` force EDTs and all `B` KE EDTs although both only read it — that
is a per-node-exclusive write grant migrating around the whole force phase, and
it is the contention point.  The **redistribute EDT** takes all `B` cells RW at
once (63 MB at the calibrated size) — every cell's ownership converges on one
node per step and scatters again.  The 8-byte **`dt` block** is taken RW by all
`B` velocity EDTs that only read it.  Each **cell** has one writer and up to
**26 concurrent readers** in the force phase — a genuine DB-granular
read/write overlap, benign only because the writer touches `f`/`U` and the
readers touch `r`.

## Flow

Strictly bulk-synchronous: `velocity(B) → position(B) → redistribute(1) →
force(B) → velocity(B) …`, with a join EDT between every pair of phases, so no
cell ever runs ahead of another.  Maximum concurrent EDTs is `B`; minimum is 1,
three times per step (the two join EDTs and the redistribute EDT).  A block of
`period` steps ends with a kinetic-energy fan-out and a status line.

The serial preamble is large: `mainEdt` creates `B` datablocks, builds the FCC
lattice over `4·n_x·n_y·n_z` atoms, sets temperatures and redistributes atoms —
all native code on the rank-0 worker before any parallelism exists (10 648
`ocrDbCreate` calls and 186 624 atom placements at the calibrated size).
`end_edt` likewise destroys all `B` cells serially.

## Placement (base)

There is no `OCR_APP_OPTIMIZED_PLACEMENT` layer in this port and no affinity
call anywhere: every `ocrDbCreate`/`ocrEdtCreate` passes `NULL_HINT`.
Effective policy:

- **EDTs** → runtime round-robin (`ARTS_HINT_ANY_RANK` from the shim), so a
  cell's force/position/velocity EDT lands on an arbitrary rank and on a
  *different* rank each step.
- **DBs** → home = creating rank.  Every cell, the simulation block, the timer
  and the cell list are created inside `mainEdt`, so **all of them are homed on
  rank 0**; only the transient 8-byte scalars are homed where their producer
  ran.

The algorithm's locality (a cell's neighbours are fixed for the whole run) is
therefore never expressed.  Each force EDT pulls 27 cells ≈ 159 KB, so the force
phase issues `27·B` cell acquires per step — 287k at the calibrated size — of
which round-robin placement makes a `(nodes−1)/nodes` fraction remote before any
reuse, on top of a write grant on the simulation block that migrates `2B` times
per step.  Treat this port as a worst-case coherence stress, not a scaling
benchmark.

## Placement (hinted)

As-born homes every cell on rank 0 and lands each cell's force/energy/advance
task on an arbitrary — and each step a different — rank, so every box's atoms
travel every timestep.

The layer (`comdSlabEdtHint` in `cells.h`; applied at the force-pair spawn in
`lj.c` and the kinetic-energy / advance-velocity / advance-position loops in
`timestep.c`) places box b's tasks on the band rank `(b * nranks) / boxes_num`.
Boxes are linearized x-fastest, so a contiguous index band is a slab of whole
x-y planes: a box's 26 neighbours are in its own or an adjacent plane, i.e.
the same or the neighbouring band — each box's RW data and most of its
neighbour reads stay on one rank, step after step.  EDT affinity only; the
box DBs keep `NULL_HINT` and settle with their pinned tasks.  The global
reductions stay base.  `nranks <= 1` returns `NULL_HINT`.

## Sizing

`n_x,n_y,n_z` set both parallel width and memory: cells `≈ (0.6246·n)³`, atoms
`4·n³`, payload `5896 B` per cell regardless of occupancy (~17 of 64 slots used,
a 3.7× padding).  Task grain is fixed by the cutoff — a force EDT always compares
one cell against 27 — so `n` buys *more* tasks, never bigger ones; `-l/--lat` is
the only knob that changes atoms per cell.  `-N`/`-n` scale time and nothing
else.

- **1 node × 15 workers**: `B ≥ ~10³` keeps every deque fed through the joins —
  `-x 16` (`B = 729`) to `-x 20` (`B = 1728`, the app default).
- **8 nodes × 120 workers**: `-x 36` gives `B = 10648`, ~89 force tasks per
  worker per step; `-x 48` (`B = 24389`, 144 MB) if a longer run is wanted.
- The calibrated `-x 36 -y 36 -z 36 -N 2` is sized so one node runs minutes:
  10 648 cells × 10 steps (the period quantization) ≈ 3.6·10⁵ tasks and 63 MB
  resident.  Use `-N k -n 1` when an exact step count matters.
- **Hard ceiling**: the join EDTs take `B+1` dependences and the shim's
  template encoding rejects a `depc` above 65534, so `B` must stay below that —
  `-x 65` (`B = 64000`) is the largest cubic run.  A larger grid is reported and
  the run ends: every join checks the create and shuts down naming the
  dependence count it could not satisfy.
