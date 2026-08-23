# miniAMR_intel_chandra

*The same AMR proxy as an SPMD program: an `npx × npy × npz` grid of "ranks" is
forked onto the policy domains, and the entire potential octree beneath each rank
— every block at every refinement level — is materialized up front, then
activated or deactivated as refinement moves.*
Source: `third_party/ocr-apps/apps/miniAMR/refactored/ocr/intel-chandra/`
(15 `.c` files; `main.c` `init.c` `driver.c` `refine.c` `comm*.c` carry the
structure), linked against `ocrAppUtils`, `reduction` and `timer`.

## Overview

`mainEdt` parses the command line, packs it into one `globalParamH_t`
datablock, and forks `npx·npy·npz` SPMD `initEdt`s over a 3-D policy-domain grid.
Each of them walks levels `0 … num_refine` and, for every octree node it would
ever own, creates a `rankH_t` handle datablock, a `channelSetupEdt` that
rendezvouses with the node's 6 neighbours + parent + 8 children + 8 siblings
through labeled sticky events, and (inside that setup) the node's cell array.
Only level-0 nodes then start the driver; deeper nodes sit pre-wired and
pre-allocated, waiting to be switched on by a refinement.

The timestep DAG is a chain of small loop-driver EDTs per active block —
`timestepLoop → stageLoop → stage → varsLoop → vars → comm | calcLoop → calc` —
with a `checkSum` chain folded in on checksum timesteps and a `refineLoop` on
refinement timesteps. Cross-block coupling is a `reduction`-library tree over the
octree (block counts, refinement intent, checksums, coarsening consensus) plus
double-buffered halo events. `wrapUpEdt` prints `Shutting down`, calls
`ocrShutdown()` and prints `Done` — the catalog's marker.

`VERIFICATION_RUN` is compiled in by the ARTS build, so every cell starts at
`pow(8, num_refine) + var` (`init.c:215`) instead of the default unseeded
`rand()`; the checksum is therefore reproducible across runs and runtimes, which
is what makes this row usable for cross-configuration consensus at all. What the
program stresses is fine-grain task chaining (hundreds of tiny loop EDTs per
block per timestep) plus a deep reduction tree — the cell arithmetic is a small
fraction of it.

## Parameters

`parseCommandLine` (`main.c:191`) fills a `Command` struct from `param.h`
defaults, `mainEdt` copies it into `globalParamH_t`, and every SPMD EDT receives
that datablock as a dependence — the whole surface is multinode-safe. Unknown
flags fall through the `else if` chain silently.

| flag | meaning | default | CLI reachability |
|------|---------|---------|-------------------|
| `--npx --npy --npz` | **SPMD rank grid**; `npx·npy·npz` is the number of forked tasks and the unit of node distribution | 1 | ✓ — the single most consequential argument here |
| `--init_x --init_y --init_z` | coarse blocks per rank per axis | 1 | ✓ |
| `--nx --ny --nz` | cells per block per axis (even, > 0) | 10 | ✓ |
| `--num_vars` | variables per cell | 40 | ✓ (capped by `MAX_NUM_VARS`) |
| `--comm_vars` | variables per comm batch; 0 or > `num_vars` → `num_vars` | 0 | ✓ |
| `--num_refine` | refinement levels — **also the eager allocation depth** | 5 | ✓ (capped by `MAX_REFINE_LEVELS`) |
| `--num_tsteps` | timesteps | 50 | ✓ |
| `--stages_per_ts` | stages per timestep | 20 | ✓ |
| `--checksum_freq` | checksum every N *timesteps* (`driver.c:435` uses `ts % checksum_freq`, not the reference's `istage %`) | 5 | ✓ |
| `--refine_freq` | timesteps between refinement rounds | 5 | ✓ |
| `--report_diffusion` | **gates the only checksum printout** (`driver.c:665/676`) | 0 | ✓ — without it the program prints no per-timestep number at all |
| `--num_objects` / `--object …` | refinement-driving objects | 0 | ✓ — with 0 objects `check_block` never intersects, so **no block ever refines** and the octree stays at level 0 |
| `--error_tol` | checksum tolerance exponent | 8 | ✓ |
| `--stencil` | 7 or 27 | 7 | ✓ |
| `--uniform_refine` | refine every block to `num_refine` regardless of objects | 0 | ✓ — the object-free way to exercise the refinement path |
| `--block_change --code --permute --refine_ghost --plot_freq --report_perf --blocking_send --max_blocks --target_* --inbalance --reorder` | reference knobs | `param.h` | ✓ parsed; `code`, `refine_ghost`, `stencil` are honoured, the rest are stored and mostly unread |
| `--lb_hinted` | load balancing (0 none / 1 each refine / 2 each phase) | 0 | ⚠ parsed; `FNC_loadbalance`/`FNC_redistributeblocks` exist but the default 0 never reaches them |
| `VERIFICATION_RUN` | deterministic cell initialization | **defined** by the ARTS build | ✗ compile-time (`benchmarks/apps/CMakeLists.txt`) |
| `USE_STATIC_SCHEDULER` | use the static-scheduler fork | not defined | ✗ compile-time |
| `USE_LAZY_DB_HINT` | tag payload DBs with `OCR_HINT_DB_LAZY` | not defined | ✗ compile-time |
| `CHANNEL_EVENTS_AT_RECEIVER` | which side owns the halo event | not defined | ✗ compile-time |
| `MAX_OBJECTS` / `MAX_REFINE_LEVELS` / `MAX_NUM_VARS` | 10 / 8 / 40 | — | ✗ `param.h` |

## Structure

Let `P = npx·npy·npz` (SPMD ranks), `Q = init_x·init_y·init_z` (coarse blocks per
rank), `R = num_refine`, `V = num_vars`, `C = (nx+2)(ny+2)(nz+2)`, `S =
stages_per_ts`, `T = num_tsteps`. Octree nodes materialized at startup:
`N = P·Q·Σ_{l=0..R} 8^l`.

| object | count | size |
|--------|-------|------|
| `initEdt` | `P` | — |
| `channelSetupEdt` | `N` (one per octree node, all levels) | — |
| `rankH_t` DB | `N` | ~10 kB (block handle + shared-object GUID tables + timers) |
| cell array DB (`DBK_array`) | `N` | `V·C·8` B |
| work DB (`DBK_work`) | `N` | `C·8` B |
| halo send buffers | `N × 6 × {curr, coar, 4×refn} × 2` phases | `msg_len·8` B each (`msg_len` from `init.c:106`) |
| refinement-intent buffers | `N × (6 × {curr, coar, 4×refn} × 2 + 8 siblings × 2)` | 4 B each |
| reduction scratch DBs | `N × (1 + 3·MAX_REDUCTION_HANDLES)` — `octTreeRedH`, then `in`/`out`/`redRootH` per handle (`MAX_REDUCTION_HANDLES` = 26) | small |
| handshake copies | `N × 6` — a `sharedOcrObj_t` published into each outgoing setup event | ~10 kB |
| labeled sticky events | `6·(P·Q·8^l)` reserved per level `l` (`main.c:132`) | key space only |
| per-timestep EDTs per active block | `1 + S·(16 + 2V)` at `comm_vars ≥ num_vars` | — |
| per-timestep events per active block | `4 + S·(30 + 4V)` | — |
| checksum EDTs | `2V` per active block on a checksum timestep (`checkSum` + `print` per variable) + a reduction tree per variable | — |

The timestep chain is exact. Per stage a block runs `stageLoop`, `stage`,
`varsLoop`, `vars`, then `V` × (`calcLoop` + `calc`) — and a **comm sub-chain of
12 EDTs**: `FNC_comm` walks the three axes as a linear chain of three instances,
and each instance creates one `commHaloNbrsEdt`, which in turn creates one
`packHalosEdt` and one `unpackHalosEdt`. The axis count is fixed at 3
(`comm.c:94`) whatever the neighbour topology; `blockNnbrs` only sizes the
dependence lists, so a degenerate axis costs the same three tasks with fewer
dependences. Adding the one `timestepLoop` per timestep gives `1 + S·(16 + 2V)`.

Events come from three sources and all three are on the steady-state path.
Every `EDT_PROP_FINISH` create that also takes an output event contributes
**two** (its finish event and the materialized output event); every such site is
immediately followed by a `createEventHelper` counted event that the successor
waits on, so the loop drivers cost **three** events per link. Per stage:
`stage`, `varsLoop`, `vars` 3 each, `vars` again for the `calcLoop` link 3, each
of the `V` `calcLoop`s 4 (its `calc` link plus a continuation event), each of the
3 `FNC_comm` instances 3 for its `commHaloNbrsEdt`, and each `commHaloNbrsEdt` 2
for `packHalosEdt` (output event only — that one is `EDT_PROP_NONE`). Setup adds
the rendezvous: per octree node, 4 channel events per reduction handle, 53
double-buffered halo channels, 7 per neighbour direction and 17 for
parent/children/siblings — 216, level-independent, since a slot that has no
peer at this level gets a pre-satisfied counted event instead of a labeled one.
Datablocks are allocated only at setup: **249 per octree node**, nothing on the
timestep path.

Worked numbers for the calibrated args (`--nx 12 --ny 12 --nz 12 --init_x 2
--init_y 2 --init_z 2 --num_tsteps 10 --num_refine 1 --npx 2 --npy 2 --npz 2`):
`P = 8`, `Q = 8`, `R = 1` → `N = 8·8·9 = 576` octree nodes, of which 64 are
active level-0 blocks; `C = 2744`, so each node's cell array is 878,080 B and the
eager allocation alone is ~505 MB in ~143k datablocks before a single timestep
runs. With `S = 20` and `V = 40`, each active block issues
`1 + 20·(16 + 80) = 1921` EDTs and `4 + 20·(30 + 160) = 3804` events per
timestep — ~1.23 M EDTs and ~2.4 M events over ten timesteps across 64 blocks,
and not one datablock. Because `--num_objects` is absent nothing ever refines,
so the 512 level-1 nodes are allocated and wired but never activated. Every
count is a closed form of the arguments.

Counter cross-check: measured at 1 node with `--npx 2 --npy 1 --npz 1 --nx 4
--ny 4 --nz 4 --num_vars 2 --num_refine 0 --stages_per_ts 1 --checksum_freq 0`
(`N = A = 2`, `S = 1`, `V = 2`), the per-timestep forms give exactly the
observed 21 EDTs and 42 events per (block, timestep) and zero datablocks, and
the setup terms — `3 + N·(17 + 10V)` EDTs, `2 + N·249` datablocks and
`1 + N·(222 + 3V)` events, plus what the reduction library allocates for the
`V + 2` all-reduces it runs — reproduce the absolute totals 163/521/649 at
`num_tsteps 2` and 247/521/817 at `num_tsteps 4` once the shim bootstrap
(2 EDTs, 1 datablock) is subtracted.

## Wiring

Setup is a rendezvous, not a tree. Each octree node computes the labeled GUIDs of
its 6 face neighbours (at its own level, at the coarser level, and the 4 finer
sub-faces), its parent, its 8 children and its 8 siblings — indices into
`haloRangeGUID[level]`, derived arithmetically from the block's global (i,j,k) —
creates each event with `GUID_PROP_CHECK` so whichever side gets there first
installs it, satisfies its own `sharedOcrObj_t` handle into it, and makes its
`channelSetupEdt` depend on all `6·nNbrs + 17` of them. When that fires, every
node holds its peers' event GUIDs and datablock keys; `init()` then allocates the
cell array and the level-0 nodes start the driver.

Steady state: `FNC_comm` packs the outgoing faces into the current phase's send
buffers and satisfies the neighbour's halo event; `unpackHalos` writes into the
receiving block's array. Buffers are double-buffered per phase, so a block never
overwrites a buffer a neighbour has not yet consumed. Access modes: `DBK_rankH`
is RW on every loop EDT of its own block's chain (a strictly serial baton — the
chain is linear by construction), `DBK_octTreeRedH` is RW on the reduction path,
send buffers are RW on the sender and RO on the receiver. The genuinely shared
objects are the reduction handles: `FNC_createChildBlocks` takes the parent's
`rankH` **and all eight children's** RW simultaneously (`block.c:157–165`), which
is the widest RW fan-in in the program and the point where a refinement
serializes an octree family.

`--num_objects 0` (the default and the catalog's setting) means `check_block`
never intersects, `bp->refine` is never set to `REFINE`, and none of that
machinery ever runs.

## Flow

`mainEdt` (rank 0, serial: parse + `initGlobalOcrParamH`'s `MAX_REDUCTION_HANDLES
+ num_refine + 2` GUID-range reservations) → `P` `initEdt`s in one loop → each
runs an `N/P`-iteration serial loop creating handle datablocks and
`channelSetupEdt`s → a global-ish rendezvous as all `N` setups complete → an
init-checksum reduction → `FNC_miniamrMain` → `FNC_driver` → `FNC_timestepLoop`.

Parallel width is the number of **active** blocks: `P·Q` at level 0, ×8 per
level actually refined. Within a block everything is serial — the stage loop, the
vars loop and the calc loop are linear EDT chains, one variable at a time, so a
block's timestep is a chain of ~1900 tasks with no internal parallelism. The
machine is therefore kept busy by block count, never by per-block width.

Serial bottlenecks: `mainEdt` itself; each `initEdt`'s `N/P` setup loop; every
checksum timestep, which drives a full octree reduction and a `2V`-EDT
serial print chain on the sequential-rank-0 block; and every refinement round,
which is an intent reduction plus a parent-and-eight-children RW join.

## Placement (base — and why there is no hinted tier)

This port needs no placement layer added to it, which is a statement about the
port rather than about the app: `forkSpmdEdts_staticScheduler_Cart3D` maps a
3-D **subgrid** of ranks to each policy domain rather than striping them
modulo, `getAffinityHintsForDBandEdt` pins **both** EDT and datablock affinity
to the owning domain at 84 sites, and `load_balance.c` re-pins through
`...AtPD(newPD)` when a block migrates.  The counters at 8 nodes agree: EDT
spread across ranks **1.00x** and **98.8% of acquires local**.  A hinted tier
would have nothing left to say.

What the counters do show is a **2.24x spread in useful work** against that
1.00x spread in EDT count — the blocks are evenly distributed but not evenly
heavy, which is adaptive refinement rather than misplacement, and belongs to
the load-balance dials (`--lb_opt`, `--inbalance`, `--target_active`) rather
than to affinity.

## Placement (base)

No `OCR_APP_OPTIMIZED_PLACEMENT` guard exists in this port — the placement below
*is* the base program, and it is fully explicit.

- `forkSpmdEdts_Cart3D` (`ocrAppUtils.c:341`) splits the policy domains into a
  3-D grid via `splitDimension_Cart3D`, partitions the `npx × npy × npz` rank grid
  onto it contiguously, and sets `OCR_HINT_EDT_AFFINITY` per `initEdt`. This is
  real, structured, neighbour-preserving distribution.
- Inside a rank, `getAffinityHintsForDBandEdt` (`ocrAppUtils.c:36`) captures
  `ocrAffinityGetCurrent()` **once** into `rankH->myEdtAffinityHNT` and
  `myDbkAffinityHNT`, and every subsequent create — every loop EDT, every handle
  and cell and halo datablock, every child block on refinement — carries it.
  Both EDT placement and DB home are therefore pinned to the node the rank's
  `initEdt` landed on, for the whole run.
- Consequence: a block's data is always local to the tasks that touch it, and the
  only cross-node traffic is halo events and reduction edges on partition
  boundaries — exactly the MPI-like pattern the port is imitating. There is no
  migration and no rebalancing (`lb_hinted` defaults to 0), so a refinement wave
  concentrated in one octant loads one node.
- The one placement the app does not control: the labeled halo/reduction GUID
  ranges are reserved round-robin by the shim, so the *event* for a boundary face
  is homed at `index % nranks` rather than at either endpoint.

**Distribution granularity is `npx·npy·npz`, fixed at fork.** With the defaults
(`--npx 1 --npy 1 --npz 1`) there is a single SPMD rank: every EDT and every
datablock in the entire program is pinned to one node, and adding nodes changes
nothing. This is the port's dominant sizing constraint, not a subtlety.

## Sizing

- `npx·npy·npz` sets both the parallel width and the node partition, and must be
  a multiple of (or at least comfortably larger than) the node count; the PD grid
  is a contiguous 3-D partition, so a rank grid that does not divide well leaves
  nodes uneven.
- `init_x·init_y·init_z` multiplies blocks per rank without changing the node
  partition — the dial for giving each node's workers several independent block
  chains. Total level-0 blocks = `npx·npy·npz · init_x·init_y·init_z`.
- `nx·ny·nz` and `num_vars` set grain: `num_vars·(nx+2)(ny+2)(nz+2)·8` B per
  block, halo messages `msg_len·8` B.
- `num_refine` is expensive **even when nothing refines**: the octree is
  allocated eagerly, so memory and setup time scale as `Σ_l 8^l` — 9× at
  `num_refine 1`, 585× at 3, 37449× at the default 5. Never leave it at the
  default with a non-trivial `nx`.
- `num_tsteps × stages_per_ts` sets duration linearly.
- To make the refinement path run at all you need `--num_objects` with at least
  one `--object`, or `--uniform_refine 1`; and to get any printed per-timestep
  number you need `--report_diffusion 1`.

For `N` nodes × 15 workers, choose `npx·npy·npz ≥ N` (ideally a 3-D
factorization matching the node grid), then `init_x·init_y·init_z` so that the
block total covers the workers, then `nx/ny/nz` for the grain.

**The calibrated set mirrors the largest geometry**: `--npx 4 --npy 4 --npz 2`
is 32 ranks for 32 Dane nodes, and `--init_x 6 --init_y 6 --init_z 12` puts
13,824 blocks behind them.  It replaces a 12³ = 1728-rank grid that had been
chosen to force distribution.  The correction is worth stating because the
reasoning behind the old grid does not survive measurement: **the EDT count is
set by the block total, not by the rank count** — 518,117,863 EDTs at 32 ranks
against 518,119,559 at 1728, for the same mesh — so the oversized grid bought
no parallelism and cost 189.4 s against 143.3 s at the anchor.  The pin does
not move with the grid, because the checksum is a property of the mesh.

`--stages_per_ts 20` and `--num_vars 40` are stated even though they are the
defaults.  They are the multipliers: per block per timestep the program creates
about **2,082 EDTs** — 20 stages against roughly 104 for the 26-neighbour halo
pack/unpack and the per-variable calculation — and an argument list that leaves
them implicit hides where half a billion EDTs come from.

**Memory is an EDT-count phenomenon here, not a datablock one.**  At the anchor
the run holds **114.4 GB** resident while creating only 34 GB of datablocks:
what occupies the machine is the in-flight set of half a billion EDTs.  The
COUNTED output-event conversion is already complete (all 31 creates that ask
for an output event supply their own COUNTED event; the other 47 pass NULL, so
the runtime mints nothing), so there is no reclamation left to add — the count
itself is the size.  It fits: Dane nodes have 256 GB, and the residency divides
cleanly with the node count (15.5 / 8.5 / 4.5 / 2.7 GB over 1/2/4/8 nodes at
the trend size).

Anchor: **128.1 s**, inside the scaler window without further tuning.