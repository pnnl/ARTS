# miniAMR_intel_bryan

*miniAMR's refinement/coarsening protocol without miniAMR's mesh: every block
carries a single `double`, and what the program actually exercises is the
channel-event neighbour handshake that AMR needs to keep a changing 6-neighbour
graph connected.*
Source: `third_party/ocr-apps/apps/miniAMR/refactored/ocr/intel-bryan/mainOCR.c`
(~830 lines) + `refineOCR.h` (~2400) + `coarsenOCR.h` (~800) + `utils.h`; the
directory's `main.c`, `refine.c`, `comm.c` … are the reference MPI sources and
are **not compiled** into this target.

## Overview

An `npx × npy × npz` grid of blocks is forked as SPMD tasks, wired to its six
face neighbours through OCR **channel events**, and then iterated: each timestep
a block sends its value to all six neighbours, a `haloRcv` task per direction
collects either 1 value (neighbour same or coarser) or 4 (neighbour finer), and a
`stencilEdt` averages the seven numbers into the block's new value. Every
`refine_freq` timesteps the block enters a refinement round instead: a
three-phase intent protocol (`willRefine` → `communicateIntent` →
`updateIntent`, iterated to consensus) decides whether it splits into eight
children or coarsens back with its siblings, and the channel wiring is rebuilt
around the new neighbour cardinality.

The block payload is one `double` (`block_t.data`), not a mesh — `--nx/--ny/--nz`
never size an array here. So this port is a **protocol and task-graph
benchmark**: the interesting cost is the per-timestep six-way channel handshake
(6 `haloRcv` EDTs, 6 tiny datablocks, 6 event satisfies per block per timestep)
and the refinement rounds that tear the wiring down and rebuild it.

`wrapupEdt` prints `miniAMR complete!` after the `EDT_PROP_FINISH`
`realMainEdt`'s finish event fires, i.e. once every descendant has ended; that is
the catalog's marker. The catalog records `scalar_kind: bool` — correctly, because
the initial value is `srand(time(0)); rand()/RAND_MAX` (`mainOCR.c:322`) and the
refinement dice are `srand(time(NULL) * id)` (`refineOCR.h:1535`), so no printed
number is reproducible across runs.

## Parameters

Parsed by `realMainEdt` (`mainOCR.c:527`) into file-scope globals, then packed
into a 36-`int` datablock (`params[0..35]`) that every SPMD `blockInit` receives
as a dependence — the values that matter downstream are therefore
multinode-safe. The globals themselves are set on rank 0 only and never read
again after packing.

| flag | meaning | default | CLI reachability |
|------|---------|---------|-------------------|
| `--npx --npy --npz` | SPMD block grid; `npx·npy·npz` is the number of initial blocks and of `blockInit` tasks | 1 | ✓ used directly by `realMainEdt` to size the fork and the six labeled GUID ranges |
| `--max_time` | timestep cap; the block chain runs while `timestep < maxTime` | 1000 | ✓ `params[35]` → `block_t.maxTime` |
| `--num_tsteps` | timestep count in the reference program | 20 | ⚠ parsed into `params[20]` and **never read** — `--max_time` is this port's timestep dial |
| `--refine_freq` | timesteps between refinement rounds | 5 | ✓ `params[17]` → `block_t.refineFreq` |
| `--nx` | doubles as the *physical* block edge length used for the object-intersection geometry (`blockSize`, `mainOCR.c:299`) | 10 | ⚠ reachable, but it sizes geometry only — there is no cell array to size |
| `--ny --nz` | — | 10 | ⚠ parsed into `params[5]`/`params[6]`, never read |
| `--num_objects` | allocates the object array and enables `--object` | 0 | ⚠ reachable and copied into every block, but the refinement decision that would read it is compiled out (see below) |
| `--object t b cx cy cz mx my mz sx sy sz ix iy iz` | one object; must follow `--num_objects` | — | ⚠ same |
| `--num_refine --uniform_refine --block_change --num_vars --comm_vars --init_x/y/z --reorder --max_blocks --target_* --inbalance --lb_opt --report_diffusion --error_tol --stages_per_ts --checksum_freq --stencil --permute --report_perf --plot_freq --code --blocking_send --refine_ghost` | reference miniAMR knobs | see `param.h` | ⚠ all parsed and packed into `params[]`, none read by any EDT |
| `MAX_REF` | maximum refinement level (`block_t.maxRefLvl`) | 1 | ✗ `#ifndef` in `mainOCR.c:26`; the ARTS build passes no `-DMAX_REF`, so blocks refine at most one level |
| `OBJECT_DRIVEN` | selects the real, object-geometry refinement test in `willRefineEdt` | **not defined** | ✗ compile-time. Undefined, the `#else` branch runs: *block id 10 always refines*, and blocks descended from id 10 refine again at timesteps 100/200/300/400. Refinement is a fixed test hook, not a moving object |

Because `OBJECT_DRIVEN` is off and `MAX_REF` is 1, a run refines exactly once, in
exactly one place: block 10, at the first timestep divisible by `refine_freq`.
That requires `npx·npy·npz > 10` for any refinement to occur at all.

## Structure

Let `R = npx·npy·npz` (blocks) and `T = max_time`. `blockEdt` recreates itself
as its own successor at the top of every timestep body (a one-EDT
continuation chain, not a fresh fork per timestep), so the very first
`blockEdt` of each chain is created by `myConnect`, not by the chain itself —
and `mainEdt` itself (the trampoline entry) is the app's own first EDT, one
level above `realMainEdt`.

| object | count | size |
|--------|-------|------|
| setup EDTs | `1` `mainEdt` + `2` (`realMainEdt` [FINISH], `wrapupEdt`) + `R` `blockInit` + `R` `myConnect` + `R` first `blockEdt` (created by `myConnect`) | — |
| per-timestep EDTs | `R · T · 8` — 1 successor `blockEdt`, 1 `stencilEdt`, 6 `haloRcv` | — |
| refinement EDTs | per round per block: `refineControlEdt`, `willRefineEdt`, `communicateIntentEdt`, then `updateIntentEdt` + up to 6 `rcv` EDTs per consensus iteration; a block that splits adds 8 child `blockEdt` chains | — |
| `block_t` DB | `R` live (one per block, created by `myConnect`, replaced on refine) | 19,056 B — 960 B of channel-GUID tables and a fixed `object[100]` array (17.6 kB) that is copied whether or not objects exist |
| `comm_t` / `connect_t` DB | `R` (`blockInit`'s scratch, consumed by `myConnect`) + `6R` (`blockInit`'s per-direction rendezvous payload, created unconditionally for all 6 directions regardless of boundary) | 96 B / 16 B |
| `data_t` DB | `6` created and destroyed per block per timestep | 16 B |
| `dataBundle_t` DB | `6` created per block per timestep (`haloRcv` output) | 72 B |
| scratch DB in `stencilEdt` | 1 per stencil (`sizeof(double)·depc`) | 56 B |
| args / range DBs | `1` `int[36]` + `1` `range_t` + `1` `numRanks`-GUID scratch array (`realMainEdt`'s `redChannels` — allocated, never referenced again) + `1` objects DB (only if `--num_objects`) | — |
| labeled GUID ranges | `6 × R` sticky events reserved (`mainOCR.c:685`) | key space only |
| channel events | `12` per block, unconditional (`rcv[6]`, `rRcv[6]`, `maxGen 2`, `nbSat 1`, `nbDeps 1`) | — |
| labeled sticky events | `6R` unconditional "send" announcements + one "receive" pickup per real neighbour direction (boundary directions alias send↔receive instead) | — |

Event accounting: explicit `ocrEventCreate` gives `12R` channel events (always)
plus `6R` unconditional labeled-sticky "send" announcements (one per
direction per block) plus one "receive" pickup per real neighbour direction —
`2(E_x+E_y+E_z)` where `E_x=(npx−1)·npy·npz`, `E_y=npx·(npy−1)·npz`,
`E_z=npx·npy·(npz−1)` are the grid's adjacent-block-pair counts per axis (a
"receive" pickup targets the SAME labeled GUID the neighbouring block's own
"send" installs, so `2(E_x+E_y+E_z)` is both sides' attempts on
`E_x+E_y+E_z` links, not new objects) — **plus one harmless duplicate**:
`blockInit`'s `case 2` (the y⁻ direction) re-issues the "send"
`ocrEventCreate` a second time whenever `yPos>0`, redundantly re-announcing an
already-installed GUID (source quirk, see notes) — `E_y` more per grid.
Output events add one per `stencilEdt` (`&stencilOutEVT`) and one per
`haloRcv` (`&rcvOUT`) — i.e. `7` per block per timestep, `7RT` in total, the
dominant term. `realMainEdt` is the only `EDT_PROP_FINISH` EDT and also takes
an output event, so it contributes one finish event plus one output event;
every other create passes `NULL`.

Setup totals `3+3R` EDTs, `3+8R` DBs, `2+18R+2E_x+3E_y+2E_z` events — at
`R=2` (the verify args below) that is `9`/`19`/`40`; at the calibrated `R=16`
(`4×2×2`) it is `51`/`131`/`354`, negligible next to the steady-state totals
below.

Worked numbers for the calibrated args (`--npx 4 --npy 2 --npz 2 --max_time
800`): `R = 16`, `T = 800` → ~102k per-timestep EDTs, ~90k output events, ~154k
small datablocks created and destroyed, and one refinement round (block 10, at
`ts = 5`) that adds eight child chains. Payload is negligible; the whole run
lives in a few MB.

Counter cross-check: verified (1 node, `--npx 2 --npy 1 --npz 1 --max_time 10`
vs `--max_time 20`, no refinement in range — measured totals 170/280/180 EDT
/DB/EVT and 330/540/320). Deltas match `R·ΔT·8 = 160`, `R·ΔT·13 = 260`,
`R·ΔT·7 = 140` exactly. Absolutes match the setup formulas above once the
runtime's constant +1 EDT / +1 DB baseline is added (`R=2`: `9+1=10` EDT-,
`19+1=20` DB-worth of setup, `40` EVT; `170 = 10+16·10`, `280 = 20+26·10`,
`180 = 40+14·10`, and identically at `T=20`) — the original doc's setup count
(`2+2R` EDTs, no DB total, `12R` events) undercounted `mainEdt` itself, the
first `blockEdt` `myConnect` creates, `realMainEdt`'s three scratch DBs,
`blockInit`'s `comm_t`/`connect_t` DBs, and the sticky-rendezvous half of the
event count.

## Wiring

Setup is a three-EDT chain per block. `blockInit` claims its six outbound
rendezvous events from labeled GUID ranges (`ocrGuidFromIndex` over
`rStruct->range[dir]`, indexed by the *neighbour's* id so both sides compute the
same GUID), creates twelve channel events, and satisfies each outbound rendezvous
with a `connect_t` naming its own receive channels. `myConnect` fires when all six
neighbour `connect_t`s have arrived and folds them into a `block_t` holding
`snd[30]`/`rcv[30]` — stride 5, so each direction has one same-level slot and
four finer-neighbour slots. A boundary direction aliases its receive event to its
own send event, so an edge block satisfies and consumes its own channel.

Steady state per timestep: `blockEdt` creates `stencilEdt` (7 slots), six
`haloRcv` EDTs (1 or 4 slots depending on `neighborRefineLvls[i]`) wired
`rcvOUT → stencilEdt[i+1]`, then creates six `data_t` datablocks and satisfies
its six outbound channels. `stencilEdt` takes the block DB **RW** at slot 0,
averages the seven values, releases, and returns the block DB as its output —
which is the successor `blockEdt`'s slot 0. So a block's DB is a strictly serial
RW baton down its own chain; no datablock in this program is ever RW from two
chains. `data_t` and `dataBundle_t` blocks are single-producer/single-consumer
and explicitly `ocrDbDestroy`ed on consumption.

The refinement round replaces `blockEdt` with `refineControlEdt`, which forks
`willRefineEdt` (decides) and `communicateIntentEdt` (broadcasts the decision to
the six neighbours over the *refine* channels `rSnd`/`rRcv`) and iterates
`updateIntentEdt` until neighbour dispositions stop changing — the classic AMR
2:1-balance consensus. Only then are children created and new channels
distributed (`newConnections`). Max simultaneous readers is 4: a finer
neighbour's four quarter-face bundles feeding one `haloRcv`.

## Flow

`mainEdt` → `realMainEdt` (FINISH) → `forkSpmdEdts_Cart3D` creates `R`
`blockInit` tasks in one loop → per block `blockInit` → `myConnect` → an
unbounded `blockEdt` chain. Parallel width is `R` in the base phase and grows to
`R + 7` after the single refinement (block 10 becomes eight). Within a block the
timestep is fully serial (`blockEdt → 6 haloRcv → stencilEdt → blockEdt`), so
per-block latency, not width, sets the wall clock once `R` ≤ worker count.

Serial points: `realMainEdt`'s single-threaded argv parse and `R`-iteration fork
loop; each refinement round, which is a global-ish barrier only among a block and
its six neighbours (not across the mesh); and termination, which waits for the
finish event of every descendant before `wrapupEdt` runs. Blocks are otherwise
free-running — there is no timestep barrier across the mesh, only the pairwise
channel dependences, so blocks drift apart by up to `maxGen = 2` generations.

## Placement (as-born)

There is no `OCR_APP_OPTIMIZED_PLACEMENT` layer in this port; the placement below
*is* the as-born program, and it is explicit rather than defaulted.

- The initial fork distributes: `forkSpmdEdts_Cart3D` (`ocrAppUtils.c:341`, built
  with `ENABLE_EXTENSION_AFFINITY`) splits the policy domains into a 3-D grid,
  maps the `npx × npy × npz` block grid onto it by contiguous 3-D partitioning,
  and sets `OCR_HINT_EDT_AFFINITY` per `blockInit`. Neighbouring blocks therefore
  start on the same node wherever the partition allows.
- **Everything after that is pinned to the creating rank.**
  `ocrAffinityGetCurrent()` + `OCR_HINT_EDT_AFFINITY` is applied to `myConnect`,
  every `blockEdt`, `stencilEdt`, `haloRcv`, and the refinement control/intent
  tasks (`mainOCR.c:158/340/462`, `refineOCR.h` ×4). A block's entire future —
  including its eight children — stays on the node its `blockInit` landed on.
  There is no migration and no rebalancing after refinement.
- Datablocks are all created with `NULL_HINT` → home = creating rank, which given
  the pin above means every block's data is homed with its block. Cross-node
  traffic is exactly the halo channel satisfies on partition boundaries.
- A minority of creates in the coarsening path (`coarsenOCR.h`, and
  `refineOCR.h:196/940/951/1032`) pass `NULL_HINT` and therefore round-robin away
  from their block's node — a placement inconsistency inside the same port.

Net effect: distribution granularity is `npx·npy·npz`, decided once at fork. With
the default `--npx 1 --npy 1 --npz 1` there is a single block and the whole run
is one rank's work no matter how many nodes are configured; the catalog's args
give 16 blocks, which partition 8/8 across two nodes and 2 per node at eight.

## Sizing

- `npx·npy·npz` is the *only* parallelism dial and also the only distribution
  dial. It must exceed the total worker count for the machine to be busy, and it
  must exceed 10 for any refinement to happen.
- `max_time` sets duration linearly; it does not change width or grain.
- `refine_freq` changes how often the consensus protocol runs, but with
  `MAX_REF = 1` and the id-10 test hook only one round ever does anything.
- Grain is fixed and tiny: no argument makes a task do more work. Raising
  `npx·npy·npz` raises task *count*, never task size, so this port stays a
  scheduler / event-plumbing probe at every size.

1 node × 15 workers: `--npx 4 --npy 2 --npz 2` (16 blocks, one per worker with
slack) and `--max_time 800` — the calibrated set, chosen so the 1-node cell runs
in the tens of seconds. 8 nodes × 120 workers: 16 blocks would leave 104 workers
idle, so scale the grid with the node count (`--npx 8 --npy 4 --npz 4` = 128
blocks) and keep `--max_time` fixed; note that the calibrated strong-scaling
sweep does *not* do this, so the multinode cells of this row measure a fixed
16-block graph spread thinner, and past two nodes most of the machine is idle.
