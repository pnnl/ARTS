# CoMD_intel_chandra_tiled

*The same physics as a real MPI code: a rank grid you choose, one subdomain per
rank, and a face halo that travels as packed buffers over paired channel
events instead of shared datablocks.*
Source: `third_party/ocr-apps/apps/CoMD/refactored/ocr/intel-chandra-tiled/`
(12 C files, ~7.9k lines; `haloExchange.c` is the reason this port exists).

## Overview

Intel's second OCR port of the ExMatEx CoMD proxy, and structurally the
opposite of `CoMD_intel_chandra`: instead of promoting every link cell to a
rank, it keeps upstream CoMD's `xproc × yproc × zproc` domain decomposition and
runs one SPMD EDT chain per rank.  Each rank owns a brick of link cells plus one
shell of halo cells, holds its atoms in six flat arrays, and exchanges ghost
atoms with its **six face neighbours** in three sequential axis sweeps — the
classic CoMD communication pattern, expressed with OCR channel events in place
of `MPI_Sendrecv`.

The result scalar is `Final energy`, printed by `validateResult` at the last
step beside the initial energy, so a value off by more than the catalog's `1e-6`
means the integration went wrong.  What the program stresses is **structured
point-to-point data movement**: no datablock is ever shared between ranks, the
per-rank arrays are single-writer, and the only cross-rank objects are the
packed halo buffers.  It is the reference point against which the other CoMD
ports' shared-halo coherence traffic should be read — and, because the
per-rank chain is strictly sequential, the one CoMD entry whose parallel width
is a *command-line choice* rather than a consequence of the box size.

## Parameters

| flag | meaning | default | CLI reachability |
|------|---------|---------|------------------|
| `-x/--nx`, `-y/--ny`, `-z/--nz` | unit cells per dimension (global) | 20 | ✓ parsed in `mainEdt`, copied into the global-parameter DB and then into every rank's `rankH_t` — multinode-safe |
| `-i/--xproc`, `-j/--yproc`, `-k/--zproc` | **rank grid** — this is the parallel width | 1, 1, 1 | ✓ same path.  Product = number of SPMD EDT chains; defaults give **one** rank, i.e. a fully serial run.  `mainEdt` rejects a non-positive extent before the fork (an extent of 0 forks no rank at all, so nothing would be left to reach `sanityChecks`) |
| `-N/--nSteps` | time steps | 100 | ✓ exact; the last step always prints and validates regardless of `-n`.  `0` is legal — the initial state is printed and validated and the run joins and shuts down without integrating |
| `-n/--printRate` | steps between the kinetic-energy allreduce and a status row | 10 | ✓ pure diagnostic/sync interval; it does not change the trajectory and does not gate termination |
| `-D/--dt` | time step (fs) | 1.0 | ✓ applied to both half kicks and the drift |
| `-T/--temp` | initial temperature (K) | 600.0 | ✓ |
| `-r/--delta` | initial random displacement (Å) | 0.0 | ✓ |
| `-l/--lat` | lattice constant (Å); `<0` = the potential's 3.615 | -1.0 | ✓ — moves the cell grid, see Sizing |
| `-e/--doeam` | use EAM instead of Lennard-Jones | 0 | ⚠ implemented (unlike the sibling ports) but only for `funcfl`; needs a `pots/` directory in the working directory, and `setfl` aborts |
| `-d/-p/-t` (`potDir`, `potName`, `potType`) | EAM table selection | `pots`, auto, `funcfl` | ✓ under `-e`, dead otherwise |
| `-h/--help` | print the option table | — | ✓ prints and calls `ocrShutdown` |

The target's three `EXTRA_DEFINES`: **`DOUBLE_BUFFERED_EVTS`** doubles the halo
channel and send-buffer arrays (`NB_SEND_CHANNELS` 6 → 12) and indexes them by
`phase = step % 2` — load-bearing, see Wiring.  **`CHANNEL_EVENTS_AT_RECEIVER`**
decides which side *owns* each halo channel: with it a rank creates its own
receive channels and publishes their GUIDs, so the peer's send is a remote
satisfy (push); without it the sender owns them and the receiver's
`ocrAddDependence` is remote (pull).  **`WITH_COUNTED_EVT`** is **dead in this
source** — nothing under `intel-chandra-tiled/` references it; counted events
are used unconditionally through `createEventHelper`.

`MAXATOMS` (64), `ARITY` (10, reduction fan-in) and `numberOfTimers` (24) are
genuine compile-time constants.  `USE_STATIC_SCHEDULER` is in the upstream
Makefile but *not* in this target, so the plain `forkSpmdEdts_Cart3D` fork is
used (see Placement).

## Structure

With `R = i·j·k` ranks, `lat = 3.615`, `cutoff = 5.7875`,
`gs_a = floor(n_a·lat / (proc_a·cutoff))` local boxes per axis,
`L = gs_x·gs_y·gs_z`, `H = 2·((gs_x+2)(gs_y+gs_z+2) + gs_y·gs_z)`,
`T = L + H`, `A = 64·T` atom slots, `S` steps and `Q = ceil((R−1)/10)`:

| object | count | size |
|--------|-------|------|
| atom arrays (`gid`, `iSpecies`, `r`, `p`, `f`, `U`) | `6R` | `4A`, `4A`, `24A`, `24A`, `24A`, `8A` bytes |
| `nAtoms` | `R` | `4T` |
| halo send buffers | `12R` | `bufCapacity` each (largest of the three face areas × 2 × 64 × 56 B) |
| tag buffers / cell lists / exchange parms | `12R` / `6R` / `R` | 8 B / `4·nCells` / ~456 B |
| `rankH_t` / `SimFlat` / potential / 5 reduction privates | `R` each | ~7.2 KB / ~2.5 KB / ~112 B / 368 B |
| DBs | `52R + 4` app blocks + the reduction library's transients | **none created per timestep** |
| events | `R·(89 + 41S + 5·boundaries) + 3` + reduction setup | 24 halo channels + 12 labeled sticky per rank at init; 41 per rank per step |
| EDTs | `R·(30 + 23S + 2·boundaries) + 3` + reduction | 23 per rank per step |

The reduction library costs, per collective, one launch EDT and one launch block
per rank plus a tree pass (`2R+Q−1` EDTs and `3R−2` blocks for the three
ALLREDUCE objects; `R+Q` and `2R−1` for the two REDUCE ones), and the same tree
pass once more per object the first time it is used — `5(5(R−1)+Q)` events
across the five objects (see below).  Four collectives run at init
(centre-of-mass velocity, kinetic energy twice, max occupancy), one per print
boundary, and two in the epilogue (performance timers, SPMD join).  Total
reduction cost: `22R + 12Q − 10` EDTs, `24R − 17` blocks, `5(5(R−1)+Q)` events.

The 23 steady-state EDTs per rank per step are: `timestepLoop` + `timestep`
(2), the five body tasks (`advanceVelocity`, `advancePosition`,
`redistributeAtoms`, `computeForce`, `advanceVelocity`), `updateLinkCells` +
`haloExchange(x)` + `sortAtomsInCells` (3), two chained `haloExchange` EDTs for
the y and z axes, three `exchangeData` EDTs, six `loadAtomsBuffer`/
`unloadAtomsBuffer` EDTs, and the two-step force dispatch
(`ljForce_edt → ljForce1_edt`).

Two constants needed a fix past the first pass, both invisible in a same-`S`
delta (which is why Pair A's deltas already checked out) and both explained by
the same mechanism as the sibling port: the runtime counts *every*
`ocrEventCreate` call, not just the one that wins a `GUID_PROP_CHECK` race
(`arts_event_create` increments before the check, `event.c:219-221`).

- **The event constant is `89`, not `80`** (and the EDT constant needs a flat
  `+3`, not `+2`).  `initEdt`'s halo rendezvous (`CoMD.c:925-926, :971-972`)
  creates each of the `6R` labeled sticky GUIDs from *both* endpoints — a
  rank's own "send" create and the matching neighbour's "receive-mapping"
  create land on the same index — so it's 12 sticky creates per rank, not 6
  (+6 events per rank).  The reduction library's own send/recv channel pairing
  (`reductionEdt`'s parent-side `recvEVT` and the child's
  `reductionSendChannelEdt`'s `sendEVT`, same file as the sibling port) has the
  identical shape, so the setup term is `5(5(R−1)+Q)` across the five
  reduction objects (Vcm, Ke, max-occupancy, perf-timer, SPMD-join), not
  `5(4(R−1)+Q)` (+5(R−1) events).
- **The epilogue was missing entirely from the compact per-step formula**,
  because it fires exactly once per run (at whichever boundary has
  `itimestep == nSteps`), not once per boundary: `finalizeEdt`
  (`timestep.c:416-433`) runs on every rank (+1 EDT, +3 events: output +
  finish + `createEventHelper`), and `printPerformanceResultsEdt`
  (`timestep.c:394-412`) runs on rank 0 only (+1 EDT, +2 events) plus one
  `createEventHelper` shared by both branches (+1 event, every rank).  Summed
  over `R` ranks that is `+(R+1)` EDTs and `+(4R+2)` events.  A separate,
  opposite-sign effect offsets part of it: the *last* `timestepLoopEdt`
  iteration never creates a successor (it's the terminal step), so it's
  missing that create's own output event — `−R` EDTs and `−R` events.  For EDTs
  the two exactly cancel to a rank-independent `+1`, which is why only the
  flat constant moves (`+2` → `+3`, `30` stays `30`).  For events the
  coefficients don't match (`−1` vs `+4`), so the net `+3R` folds into the
  per-rank coefficient instead (`80` → `89`, with the halo-doubling fix above
  supplying the other `+6`) and the epilogue's own flat remainder (rank 0's
  extra `printPerformanceResultsEdt`, `+2`) lands in the constant (`1` → `3`).

Worked numbers for the calibrated `-x 144 -y 144 -z 144 -N 8 -i 8 -j 4 -k 4`
→ `R = 128`, `gs = (11,22,22)`, `L = 5324`, `H = 2164`, `T = 7488`,
`A = 479 232`: **≈31k EDTs**, **≈10k DBs**, **≈57k events**, 11 943 936 atoms
(93 312 per rank, 17.5 of 64 slots per local box), and per rank **≈42 MB** of
atom arrays plus **≈50 MB** of halo buffers — **≈11.7 GB in total, independent
of the node count** (the rank grid, not the machine, sets the footprint).  The
half of that which is halo buffer is worst-case capacity: the x face is the
largest here (`(gs_y+2)(gs_z+2) = 576` cells against 312 for the other two), and
every face buffer is sized for the largest.  For
the `expect` args `-x 8 -y 8 -z 8 -N 2 -i 2 -j 2 -k 1`: `R = 4`, `gs = (2,2,4)`,
`T = 96` → this is also the calibrated `-N 2` verify point: 406 EDTs, 292 DBs,
787 events exactly, 0.54 MB of atom arrays per rank.

Counter cross-check: verified (1 node, `-x 16 -y 16 -z 16 -N 2 -i 2 -j 2 -k 1`
vs `-x 16 -y 16 -z 16 -N 4 -i 2 -j 2 -k 1`): measured absolutes EDT 406/590, DB
292/292, EVT 787/1115; subtracting the runtime's constant baseline (+1 EDT, +1
DB, +0 EVT per run) against the formulas gives an exact match in both absolute
value and delta — this app has no data-dependent tail (redistribution always
runs the same fixed EDT/event chain, unlike the sibling port's `move_edt`).

## Wiring

**The halo is the point of this port, so start there.**  Each rank keeps, per
face and per phase, a *pair* of channel events: one carrying the packed
`AtomMsg` buffer and one carrying an 8-byte count — 6 faces × 2 phases × 2 =
**24 channel events per rank**, all with `nbSat = nbDeps = 1`.  They are matched
at startup through a rendezvous on labeled sticky GUIDs: `initEdt` publishes its
four channel GUIDs for face `f` into event `6·rank + f` of a `6R`-wide range and
registers on the neighbour's event `6·nbr + opposite(f)`; `channelSetupEdt`
unpacks the six replies into `haloSendEVTs`.  From then on the halo is pure
message passing — `loadAtomsBufferEdt` packs a face into its own send buffer and
`ocrEventSatisfy`s the channel with it; the neighbour's `unloadAtomsBufferEdt`
takes it **RO** off the channel and calls `putAtomInBox`.

`redistributeAtomsEdt` drives one step of that: `updateLinkCellsEdt` (empties
halo cells, moves atoms that crossed a cell boundary) → `haloExchangeEdt`, a
self-chaining FINISH EDT that runs the x, y and z sweeps **strictly in order**
(so corner and edge cells are filled by data that already arrived on the
previous axis) → `sortAtomsInCellsEdt`.  Each sweep's `exchangeDataEdt` is a
FINISH EDT holding exactly one load and one unload.

The second idiom is `createEventHelper`, used for **every** intermediate join in
the program: an `OCR_EVENT_COUNTED_T` with `nbDeps = 1`, fed by an EDT's output
event and consumed as a `DB_MODE_NULL` control edge.  Under the ARTS shim
COUNTED collapses to LATCH(1) fire-and-linger, so the count is advisory — the
app only ever asks for one — but it is what makes the whole per-rank DAG a chain
of one-shot latches rather than a web of sticky events.

Why double buffering is a **correctness** requirement, not a throughput tweak: a
rank reuses `sendBuf[face][phase]` every second step, and the pairwise handshake
supplies the missing edge.  Rank B's load at step `t+2` follows B's unload at
`t+1`, which follows A's load at `t+1`, which follows A's whole step `t`
finishing — and A's unload of B's step-`t` buffer is inside that scope.  With a
single buffer set the reuse would be at `t+1`, and no path in the DAG orders it
after A's unload at `t`; the OCR channel's `maxGen` would be the only thing
holding it back, and ARTS does not enforce `maxGen`.

DB concurrency, in one line: **there is none.**  Every per-rank array is written
only by that rank's own chain; every send buffer has exactly one writer and, one
step later, exactly one remote RO reader.  No datablock in this port is written
from more than one rank, and none has more than one concurrent reader.  The
reduction private blocks and the 24-byte reduction payloads are the only other
cross-rank objects.

## Flow

Per rank per step the DAG is a **single chain** about 15 EDTs deep:
`timestepLoop → timestep → velocity → position → redistribute →
{updateLinkCells → haloExchange(x → y → z), each exchangeData → load → unload}
→ sortAtoms → computeForce → ljForce → ljForce1 → velocity`.  Nothing inside a
rank runs concurrently with anything else in the same rank, so

    max concurrent EDTs ≈ R = xproc · yproc · zproc

and that is the whole parallelism story.  With the default `-i 1 -j 1 -k 1` the
program is *serial* whatever the machine — a single chain of ~23 EDTs per step,
each doing a full subdomain's work (this is why an uncalibrated run of this
entry looks like a few hundred EDTs of ~1 s grain).

Ranks are coupled only pairwise, through the face channels, so there is no
whole-grid barrier during the timestep loop; neighbours drift apart by at most
the one step the double-buffered channels allow.  The exceptions are the
diagnostic points: every `printRate` steps (and always on the last step) a
kinetic-energy **allreduce over all `R` ranks** gates `printThingsEdt`, which
sits inside the step's finish scope.  Initialisation adds three more collectives
(centre-of-mass velocity, kinetic energy, max occupancy) and the epilogue two
(performance timers, and an SPMD-join reduction whose root satisfies the
shutdown event) — so termination is a proper barrier, not a race.  The epilogue
hangs off the last step's `printThingsEdt`; with `-N 0` there is no such step,
so `FNC_initSimulation` builds the same epilogue behind the step-0 print
instead, and that print is then created the way the last step's is (finish
scope, timer-reduction block RW) because it is the one launching the timer
reduction the epilogue consumes and destroys.

## Placement (as-born)

No `OCR_APP_OPTIMIZED_PLACEMENT` layer exists here; everything below is the
program's own and is compiled in.  `mainEdt` calls `forkSpmdEdts_Cart3D`, which
queries `ocrAffinityCount(AFFINITY_PD)`, factors the rank count into a 3-D
policy-domain grid (`splitDimension_Cart3D` — correct for 1, 2, 4 and 8) and
gives each `initEdt` the `OCR_HINT_EDT_AFFINITY` of the policy domain that owns
its block of the rank grid (`getPolicyDomainID_Cart3D`, a block partition per
axis, so a 3-D sub-brick of ranks per node).  Every subsequent EDT takes its
hint from `getAffinityHintsForDBandEdt`, i.e. `ocrAffinityGetCurrent`.

Datablocks are placed two ways, and both land correctly: the halo send buffers
carry an explicit `OCR_HINT_DB_AFFINITY` of the current PD, and everything else
(the six atom arrays, `nAtoms`, `rankH_t`, `SimFlat`, the potential, the
reduction privates, the cell lists) passes `NULL_HINT` — which, with home =
creating rank, still homes them on the owning rank because they are created
*inside* the rank's own EDT.  The only rank-0-homed objects are the argv and
global-parameter blocks, read once per rank at init.

The resulting multinode traffic is therefore exactly the algorithm's: per rank
per step, six packed halo buffers out and six in (4.13 MB apiece at the
calibrated size, whole-DB granularity regardless of how many atoms actually
travel), plus one 24-byte reduction payload per collective.  Locality that
exists in the algorithm is fully expressed — this is the port to compare the
others against, not a coherence stress.

`USE_STATIC_SCHEDULER` would replace the per-rank affinity hints with one fork
EDT per policy domain plus an `OCR_HINT_EDT_DISPERSE` hint; it is not defined
for this target, so the direct affinity mapping is what runs.

## Sizing

Two dials that do different jobs.  **`-i/-j/-k` set parallelism** — width *is*
`R`, so pick it at a few times the total worker count and keep the factorisation
close to cubic so face areas stay balanced; the floor is structural
(`sanityChecks` demands `n_a·lat ≥ 2·cutoff·proc_a` and that the rank grid match
the number of ranks the run forked, `initLinkCells` asserts `gs_a ≥ 2`, i.e.
**`n_a ≥ 3.2 · proc_a`** on every axis).  **`-x/-y/-z` set work
and memory** — atoms `4·n_x·n_y·n_z`, per rank `≈ 5.5 KB × T` of atom arrays
plus 12 worst-case face buffers; shrinking the rank grid does not shrink the
total footprint, it concentrates it.

- **1 node × 15 workers**: `-i 4 -j 2 -k 2` (`R = 16`) with `-x 64 -y 32 -z 32`
  keeps every worker busy at ~1 rank each and ~8 MB per rank.
- **8 nodes × 120 workers**: the calibrated `-i 8 -j 4 -k 4` gives `R = 128` —
  16 ranks per node against 15 workers at the widest point, and 8×
  oversubscription at one node so the strong-scaling sweep starts saturated.
  `-x/-y/-z 144` clears `3.2·8 = 25.6` comfortably and yields `gs = (11,22,22)`,
  5324 local boxes and 17.5 atoms per box — the same occupancy as the other CoMD
  entries, so per-task arithmetic is comparable.
- `-N 8` with the default `-n 10` means one collective, at the last step; raise
  `-n` above `-N` to measure the loop with no barrier in it, lower it to study
  the collective.
- Do **not** leave `-i/-j/-k` at their defaults for a measurement: the run is
  correct but single-chain, and its numbers describe one core.
