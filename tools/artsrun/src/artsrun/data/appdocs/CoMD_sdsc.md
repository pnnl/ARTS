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
by 26 tasks while its owner writes it — one whole-grid serialization per step
(the atom-redistribution EDT) — and, above all, the wiring plane: the whole
`B`-wide five-phase DAG is re-created over the wire every step.  (A second
serialization the port used to fabricate — every per-box task taking the
shared simulation block RW while only reading it — was a mode misdeclaration
and is repaired to CONST; see Wiring.)

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

Worked numbers for the campaign `-x 48 -y 48 -z 48 -N 15 -n 5` → `g = 29`,
`B = 24 389`, `P = 5`, `K = 3`, i.e. **15 simulated steps**: ~1.29M EDTs,
~512k DBs, ~805k events, 442 368 atoms, **144 MB** of cell payload (live for
the whole run) plus ~190 KB of transient 8-byte results per phase.  The
completion-event conversion is count-neutral (each ONCE/minted output event
became one COUNTED event), so the verified formulas stand.

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
EDT's last slot; there are no channel or latch events, only COUNTED completion
events (single consumer declared, reclaimed on delivery) and EDT output
events carried the same way.

- **force** (`lj_edt`, 28 slots): simulation CONST, own cell **RW**, 26
  neighbour cells RO.  Writes its cell's `f`/`U` and wires an 8-byte energy DB
  straight into the reducer's slot.
- **advance-velocity** (`av_edt`): simulation RO, own cell **RW**, a shared
  8-byte `dt` block CONST; **advance-position** (`ap_edt`): simulation RO, own
  cell **RW**.
- **redistribute** (`redistribute_edt`, `B+1` slots): **every** cell **RW**
  plus the simulation DB **RW** (it writes `max_occupancy`), in one task.
- **kinetic energy** (`ke_edt`): simulation CONST, own cell RO; `ke_red_edt`
  sums `B` scalars (simulation **RW** — it writes `e_kinetic`) and satisfies
  a COUNTED completion event.

The simulation block and `dt` used to be declared **RW** by every per-box
reader — a misdeclaration OCR's racy model never charges for, but under a
runtime that honours write exclusivity it made every phase serialize on one
singleton's write grant migrating node to node.  The conformance repair
declares the access each task performs (CONST), leaving RW only where a task
writes (the two reducers, redistribute).  Every completion event is a COUNTED
event with its single consumer declared (`comdJoinEvt` /
`OEVT_COUNTED_PRE`+`PROP`, guarded by `OCR_APP_COUNTED_OEVT`): the reclaim
contract needs the count, and the undeclared per-box output events the av/ap
fan-outs used to mint lingered forever — ~48.8K events per step, unbounded in
the step count.

Remaining DB concurrency, in decreasing order of pain: the **redistribute
EDT** takes all `B` cells RW at once — every cell's ownership converges on
one node per step and scatters again.  Each **cell** has one writer and up to
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

The layer (in `cells.h`) has three parts.  **`comdSlabEdtHint`** (the
force-pair spawn in `lj.c` and the kinetic-energy / advance-velocity /
advance-position loops in `timestep.c`) places box b's tasks on the band rank
`(b * nranks) / boxes_num`.  Boxes are linearized x-fastest, so a contiguous
index band is a slab of whole x-y planes: a box's 26 neighbours are in its
own or an adjacent plane, i.e. the same or the neighbouring band.
**`comdSlabDbHint`** (`init_atoms`) homes box b's datablock on that same band
rank, so a box's directory lives where its tasks run instead of all `B`
boxes being homed on the one rank that ran `mainEdt`.  **`comdHomeEdtHint`**
pins the control spine — the per-phase continuations, every join, the serial
redistribute — to rank 0, so the simulation singleton's writers all run
where it lives instead of the spine round-robining to a fresh rank each
phase and dragging the write grant along.  `nranks <= 1` returns
`NULL_HINT` everywhere.  Measured on the reduced trend ladder: 2n base
263.8 s vs hinted 192.0 s (1.37x), and at 4-8n base val exceeds the 590 s
ceiling while hinted runs 299-377 s (>=2x).

## Sizing

`n_x,n_y,n_z` set both parallel width and memory: cells `≈ (0.6246·n)³`, atoms
`4·n³`, payload `5896 B` per cell regardless of occupancy (~17 of 64 slots used,
a 3.7× padding).  Task grain is fixed by the cutoff — a force EDT always compares
one cell against 27 — so `n` buys *more* tasks, never bigger ones; `-l/--lat` is
the only knob that changes atoms per cell.  `-N`/`-n` scale time and nothing
else.

- The campaign cell is `-x 48` (`B = 24,389`, 144 MB): per-phase width
  24,389 = 7.1 tasks per worker even at the largest campaign geometry
  (32 nodes x 108 workers = 3,456).
- **`-N` is sized from the 32-node end, not from 1n** — this port
  anti-scales (every step re-wires the whole `B`-wide five-phase DAG over
  the wire, measured 12.4/19.2/25.5 s per step at 2/4/8 ferrari nodes, an
  arm-invariant wall), so the longest cell is the largest geometry and
  `-N 15` puts its extrapolated ~38 s/step near the 600 s budget.  The 1n
  cell is then ~12 s and the dane1 anchor cell (one 108w+4p node) 14.7 s —
  a small 1n start point is correct for this class.  The period must
  divide `-N` or the final print's kinetic term is stale (`-n 5`).
- **Hard ceiling**: the join EDTs take `B+1` dependences and the shim's
  template encoding rejects a `depc` above 65534, so `B` must stay below that —
  `-x 65` (`B = 64000`) is the largest cubic run.  A larger grid is reported and
  the run ends: every join checks the create and shuts down naming the
  dependence count it could not satisfy.

## Family shape (measured, reduced trend ladder: 15w+1p × 1/2/4/8 nodes, `-x 48 -N 15 -n 5`)

hinted, e2e seconds.  The energy pin held in every cell:

| arm | 1n | 2n | 4n | 8n |
|---|---|---|---|---|
| val_wb | 11.5 | 192.0 | 298.9 | 377.0 |
| val_wb_comb | 11.5 | 139.2 | 191.7 | 250.3 |
| inv_wb | 11.9 | 150.2 | 222.1 | 267.2 |
| excl_retain | 13.2 | 150.3 | 235.0 | 312.2 |

base val_wb: 11.7 / 263.8 / >=590 (censored) / >=590 (censored) — the hint
layer is worth 1.37x at 2n and >=2x at 4-8n.  Every arm anti-scales
monotonically: the per-step wall is the wiring plane (the arm-invariant DAG
re-wiring census established on the sibling cell-grain port), with VAL's
re-validation adding ~1.4-1.5x over the combining arm on top.  The structural
answer to this wall is the persistent-chain SPMD rewrite — which is what
`CoMD_intel_chandra_tiled` is; the pair is the family's ablation.
