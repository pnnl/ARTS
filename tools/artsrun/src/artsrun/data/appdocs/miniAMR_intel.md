# miniAMR_intel

*Block-structured adaptive mesh refinement as a continuation-cloning coroutine:
every block is a task that re-creates itself as a fresh EDT at each point where
it must wait for a neighbour or its parent.*
Source: `third_party/ocr-apps/apps/miniAMR/refactored/ocr/intel/` (13 `.c` files,
~1.2 MB; `root.c` `block.c` `parent.c` `comm.c` `refine.c` `chksum.c` carry the
structure).

## Overview

miniAMR (Mantevo) sweeps one or more geometric objects through a 3-D mesh; blocks
the objects touch refine 8-way, blocks they leave coarsen back, and between
refinement rounds every block runs a 7- or 27-point stencil over `num_vars`
variables with a 6-face halo exchange. This Intel port expresses each block as a
*suspendable function*: `blockClone_SoupToNuts` runs the whole algorithm as
straight-line C, and at every communication point a macro pair
(`SUSPEND__RESUME_IN_CLONE_EDT` / `SUSPENDABLE_FUNCTION_PROLOGUE`) snapshots a
1 kB software stack into the block's meta datablock and hands the continuation to
a newly created `blockClone` EDT. There is no persistent task per block — there
is a *chain* of thousands of them.

The completion marker is `Grand Total Checksum`, printed by
`checksum_RootFinalAggregation` (`chksum.c:288`) once per checksum round. The
catalog's scalar regex takes the **first** such line, which is the golden
checksum computed right after initialization and before any refinement, so it is
a pure function of `nx·ny·nz`, `npx·npy·npz` and `num_vars` (`init.c` fills every
interior cell with `(2·xPos+i)·100 + (2·yPos+j)·10 + (4·zPos+k)`; the `rand_r`
line above it is immediately overwritten). Every later round is compared against
that golden value inside the program and a divergence beyond `10^-error_tol` is
reported. What the program stresses is task creation and metadata churn far more
than arithmetic: a single interior block issues six EDT clones per stage.

## Parameters

Parsed twice — once in `rootLaunch_Func` (`root.c:88`, only `--num_objects` and
the `--np*` product, to size two datablocks) and once in full in `rootInit_Func`
(`root.c:248`). Everything lands in the `Control_t` datablock, which is copied
per block and reaches every rank as a dependence — so all of it is
multinode-safe. An unrecognized flag prints the help text and then executes
`*((int *) 123) = 456` — the port's assertion idiom is a deliberate SIGSEGV, not
a clean exit.

| flag | meaning | default | CLI reachability |
|------|---------|---------|-------------------|
| `--nx --ny --nz` | cells per block per axis (must be even, > 0) | 10 | ✓ propagated in `Control_t` |
| `--npx --npy --npz` | base (unrefined) mesh in blocks; their product is the block count and must be ≤ `MAX_NUM_UNREFINED_BLOCKS` | 1 | ✓ |
| `--num_refine` | max refinement levels (0–15) | 5 | ✓ |
| `--refine_freq` | timesteps between refinement rounds | 5 | ✓ |
| `--block_change` | levels a block may change per round; 0 → set to `num_refine` | 0 | ✓ |
| `--uniform_refine` | refine everything once, then never again | 0 | ✓ |
| `--num_vars` | variables per cell | 40 | ✓ |
| `--comm_vars` | variables exchanged per comm batch; 0 or > `num_vars` → `num_vars` | 0 | ✓ |
| `--num_tsteps` | timesteps | 20 | ✓ |
| `--stages_per_ts` | comm+calc stages per timestep | 20 | ✓ |
| `--checksum_freq` | stages between checksums (0 = none) | 5 | ✓ |
| `--stencil` | 7 or 27 | 7 | ⚠ both implemented (`stencil.c`), but the only implemented `--code` sends faces *without* edges and corners, so the 27-point form reads never-exchanged corner ghosts; `check_input` warns about divergence under non-uniform refinement |
| `--error_tol` | checksum tolerance exponent (`tol = 10^-e`) | 8 | ✓ |
| `--num_objects` | number of refinement-driving objects | 0 | ✓ validated `≥ 0` in both parses (a negative value used to underflow `sizeof_AllObjects_t` — see below) |
| `--object t b cx cy cz mx my mz sx sy sz ix iy iz` | one object (14 values); must follow `--num_objects` | — | ✓ |
| `--report_diffusion` | print per-variable checksum diffusion | 0 | ✓ |
| `--code` | 0 minimal sends / 1 send ghosts / 2 process on send | 0 | ⚠ only 0 implemented; 1 and 2 reach `"case not yet implemented"` + SIGSEGV |
| `--permute` | rotate comm axis order per stage | off | ✓ (flag, no value) |
| `--refine_ghost` | include ghost cells in the refinement test | off | ✓ (flag) |
| `--plot_freq` | plot every N timesteps (0 = none) | 0 | ✓ |
| `--report_perf` | perf report level | 4 | ⚠ parsed, stored, never read — the profiling path is commented out (`//PROFILE:`) throughout |
| `--max_blocks --target_active --target_max --target_min --inbalance --lb_hinted --reorder --init_x/y/z --blocking_send` | present in the reference miniAMR | — | ✗ not parsed here; an unknown flag is a hard error |
| `MAX_NUM_UNREFINED_BLOCKS` | cap on `npx·npy·npz` (one `rootClone` dependence slot each) | 1000 | ✗ `#define` in `root.h:38`; exceeding it prints advice and SIGSEGVs |
| `SIZEOFSTACK` | per-block continuation stack | 1024 B | ✗ `#define` in `clone.h:45` |

`--num_objects` defaults to **0** in both the sizing pre-scan
(`rootLaunch_Func`) and the full parse (`rootInit_Func`, whose value every
downstream count uses), and is validated non-negative in both (the latter via
`check_input`). The pre-scan's own default was `1` until the two were
reconciled, which over-sized the `allObjects` datablock by one `Object_t`
slot whenever the flag was omitted. `sizeof_AllObjects_t` is `sizeof(AllObjects_t) +
num_objects·sizeof(Object_t)` in `size_t` arithmetic, so a negative value
previously underflowed to ~2^64 and the per-block `allObjects` create in
`blockLaunch_Func` asked for an impossible size. A trailing `--num_objects`
with no following value is likewise rejected before the read goes out of
bounds. `rootInit_Func` zero-initializes the object array as soon as
`--num_objects` sets its size, so any object not covered by an `--object`
spec (as in the catalog's own args, which pass `--num_objects 1` with no
`--object`) is a well-defined inert entry rather than whatever bytes the
allocator returned.

## Structure

Let `B = npx·npy·npz`, `V = num_vars`, `C = (nx+2)(ny+2)(nz+2)`, `S =
stages_per_ts`, `T = num_tsteps`, `K = ceil(S / checksum_freq)` checksums per
timestep, `f` a block's in-mesh neighbour faces (6 interior, fewer at the mesh
edge) and `E` the number of adjacent block pairs, so `Σ_blocks f = 2E`. Counts
below are for the **unrefined** mesh; refinement multiplies the live block
population by up to 8 per level and is data-dependent (see Sizing).

| object | count | size |
|--------|-------|------|
| `blockLaunch` / `blockInit` EDTs | `B` each | — |
| `blockClone` EDTs | `B·(4 + T·(6S + K))` — 2 per comm axis per stage (an axis with no in-mesh neighbour is skipped), 1 per checksum, plus 4 outside the loop (first clone, two startup checksums, shutdown request) | — |
| `rootInit` + `rootClone` chain | `1 + (4 + 2KT)` — the root serves `2 + KT` checksum rounds and one shutdown round | — |
| `parentInit` + `parentClone` chain | one family per refine event | — |
| `block` DB | `B` live | `8 + C·V·8` B |
| `meta` (`BlockMeta_t`) DB | `B` live | ~1.8 kB (1088 B continuation stack + event tables) |
| `control` / `allObjects` DB | `B` each, per-block private copies | ~208 B / `40 + 176·num_objects` B |
| `whoAmI` DB | **one per EDT created** (`gasket__ocrEdtCreate`, `util.c:120`) | 8 B |
| `depv` copy DB | one per `rootInit`/`blockLaunch`/`blockInit`, and one per `rootClone` **firing** (`root.c:556`) | `depc·16` B |
| face DB | `f` per block per stage | `40 + comm_vars·(two block dims)·8` B |
| `Checksum_t` DB | `B·(2 + KT)` — two per block before the loop, then one per block per round | `40 + V·8` B |
| `scratchChecksum` DB | `1 + KT` — one per checksum round after the run's first | `40 + V·8` B |
| sticky events | 1 per block per service request; 36 labeled slots per block per level for halos | — |
| labeled GUID range | `(Σ_{l≤num_refine} 8^l)·B·36` reserved once (`root.c:459`) | key space only |

Two habits set the recurring cost. `gasket__ocrEdtCreate` gives every EDT an
8-byte `whoAmI` block, so `NUM_DB_CREATE` carries the whole EDT count; and
`rootClone_Func` copies its `depv` into a fresh datablock on **every** firing.
A steady-state checksum round therefore costs *two* root firings, not one:
`checksum_RootFinalAggregation` allocates a `scratchChecksum` and suspends
before aggregating on every round except the run's first, so the round is split
across a clone boundary. A level-0 block also contributes two checksums before
the timestep loop — `init()` emits one and `blockClone_SoupToNuts` emits another
— so the root sees `2 + KT` rounds, of which only the first is "golden".

Events come from three sources. Explicit `ocrEventCreate`: one
`conveyServiceRequestToParent` per base block at startup, one fresh one per
checksum/plot/unrefine service a block requests, one fresh outgoing halo event
per face sent, 16 per refine fork (8 `conveyEighthBlockToJoin` + 8 new
service-request events), and the labeled halo events materialized on first use
through `ocrGuidFromIndex` + `GUID_PROP_IS_LABELED|GUID_PROP_CHECK` — each
directed channel is created by both endpoints, the loser installing nothing but
still counting as a create. Output events: **none** — every
`gasket__ocrEdtCreate` passes `NULL` for `outputEvent`. Finish EDTs: **none** —
every create uses `EDT_PROP_NONE`.

With `--num_refine 0` every counter is a closed form:

    NUM_EDT_CREATE   = 5 + 6B + T·(B·(6S + K) + 2K)
    NUM_DB_CREATE    = NUM_EDT_CREATE + 10 + 9B + T·(B·K + 2E·S + 3K)
    NUM_EVENT_CREATE = 3B + 4E + T·(B·K + 2E·S)

Worked numbers for the calibrated args (`--nx 16 --ny 16 --nz 16 --npx 8 --npy 8
--npz 8 --num_tsteps 12 --num_objects 1`, defaults elsewhere): `B = 512`, `V =
40`, `C = 5832`, `S = 20`, `K = 4`; block DB = 1.87 MB, so the level-0 mesh
alone is ~955 MB of payload. Per timestep a block issues `6·20 + 4 = 124`
clones, so `T = 12` gives ~1488 clones per block and ~762k `blockClone` EDTs
(each with a `whoAmI` DB) before any refinement. Face traffic is ~82 kB per face
at `comm_vars = 40`, six faces per interior block per stage.

Counter cross-check: measured at 1 node with `--nx 4 --ny 4 --nz 4 --npx 2
--npy 2 --npz 2 --num_vars 4 --stages_per_ts 2 --checksum_freq 1 --num_refine 0
--num_objects 0` (`B = 8`, `S = 2`, `K = 2`, `E = 12`, `f = 3`), the three
formulas above give 169/321/136 at `T = 1` and 285/507/200 at `T = 2`; the
runtime reports 171/322/136 and 287/508/200, which is exact once the shim's own
bootstrap (1 `main_edt` + 1 trampoline EDT and 1 argv datablock) is subtracted.
The per-timestep deltas 116 EDT / 186 DB / 64 event match term for term.

## Wiring

The mesh is a two-level service hierarchy. `rootClone` holds one dependence slot
per base block (`serviceRequest_Dep[B]`, RO) plus `control` (RW),
`goldenChecksum` (RO) and `scratchChecksum` (RW); a block asks for a service by
satisfying its `conveyServiceRequestToParent` sticky event with a
`DbCommHeader_t`-prefixed datablock, and every request carries the *next*
("on-deck") event in its header so the root can rewire its successor clone. All
`B` slots must arrive before the root advances — the checksum is a hard barrier
across the whole mesh, and the root asserts that all `B` opcodes match.

When a block refines it forks into a `parentInit`/`parentClone` family plus eight
`blockClone` prongs; the parent then plays the same role for its eight children
that the root plays for base blocks. Unrefinement joins through the eight
`conveyEighthBlockToJoin` sticky events, with prong 000's clone becoming the
merged block.

Halo exchange never goes through a parent. Each face is a fresh `Face_t`
datablock satisfied into a **labeled sticky event** whose index encodes
(refinement level, linearized block position, one of 36 direction/quarter slots),
so a sender computes its neighbour's inbox GUID arithmetically without ever
learning the neighbour's identity. A face DB has exactly one producer and one
consumer and is destroyed after unpacking — no sharing.

Access modes: `block`, `meta`, `allObjects`, `control`(root's copy),
`scratchChecksum` are RW and single-owner; a *predecessor* block/control/meta
handed to eight fork prongs is RO with **eight simultaneous readers** — the only
real fan-out in the program, and the widest at the moment a refinement wave
passes. `control` is RO on the block side after `blockInit` copies it, so no DB
takes RW from more than one block's chain. The contention point is the root's
`B`-way checksum rendezvous, not any single datablock.

## Flow

`mainEdt` → `rootLaunch` → `rootInit`, which creates the `B` `blockLaunch` EDTs
and the first `rootClone`; each `blockLaunch` → `blockInit` → the first
`blockClone`. Then every block independently runs: init (which emits the golden
checksum) → a second, redundant checksum from the driver itself →
(optional) initial refine → `for ts in 1..T { for stage in 1..S { comm(); stencil
per variable; checksum every checksum_freq stages } ; move(); refine every
refine_freq timesteps } `.

Parallel width is the live block count — `B` at level 0, growing by up to ×8 per
refinement level in the region the objects occupy. Within a stage, a block's
three axes are serialized (axis 0 completes before axis 1 starts) because each is
a suspension point, so per-block latency is `3 · 2 · (clone + halo round-trip)`
per stage. The serial bottlenecks are (a) every checksum round, an all-blocks →
root join and a fresh root clone; (b) every refine round, a parent-mediated
neighbour-consensus protocol (`refine.c` alone has 11 suspension points) that
must converge before any block proceeds; (c) `rootInit`'s single-EDT loop
creating all `B` block launchers.

## Placement (base)

Every `gasket__ocrEdtCreate` passes its hint through `amrEdtHintForBlock` /
`amrEdtHintForPD` (`util.c:107`), whose `#else` branch — the base build —
returns `NULL_HINT`. The hinted flavour maps a block's (x,y,z,level) onto a
3-D policy-domain grid; that layer is out of scope here. Datablocks are always
created with `NULL_HINT` in both flavours.

Effective base policy: **EDT → runtime round-robin**, **DB → home = creating
rank**. Consequences at multinode:

- A base block's four datablocks are homed wherever its `blockLaunch` happened to
  land, but its `blockInit` and its whole 1000-clone continuation chain are
  round-robined independently — so a block's *own* `block`/`meta` datablocks are
  remote to it roughly `(nodes−1)/nodes` of the time, and they migrate on nearly
  every clone. The block payload is the largest object in the program
  (1.87 MB at calibrated args); this is the dominant traffic, and it is
  self-inflicted rather than algorithmic.
- Halo events live in a `GUID_USER_EVENT_STICKY` range reserved with ARTS's
  round-robin distribution, so the inbox for a given face is homed at
  `index % nranks` — unrelated to where either the sender or the receiver runs.
- The root's `B`-slot checksum join draws one datablock from every block on every
  round, from wherever that block last ran.

The algorithm has textbook spatial locality (a 3-D neighbour graph, refinement
strictly within a parent's octant) and the base program expresses none of it.

## Placement (hinted)

The layer (`amrBlockHomePD` / `amrEdtHintForPD` in `util.c`) places a block's
EDTs by the block's spatial position: the rank count is factored into a
near-cubic PDx x PDy x PDz grid, a block at (x,y,z) on refinement level L is
mapped through the effective mesh (npx<<L, npy<<L, npz<<L) onto that grid, and
the block's tasks pin there.  Neighbouring blocks — including parents and
children across refinement levels, whose coordinates nest — land on the same
or adjacent ranks, so halo exchange and refine/coarsen transfers stay mostly
rank-local.  EDT affinity only, and that is measured rather than assumed:
homing each refined child's datablocks on the domain its own tasks run at
changed nothing at 1, 2, 4 or 8 nodes (44.8 / 26.2 / 17.0 / 11.9 s against
44.7 / 26.2 / 17.0 / 11.9).  A block's datablocks are acquired RW by that
block's own task straight after creation, so the payload moves once either
way and the home is only a directory entry.

**Building the hint is on the hot path, and it showed.**  This program creates
23.2 M EDTs; resolving the domain affinity and rebuilding an `ocrHint_t` at
each of them cost more than the placement saved, and the layer as first
written ran the anchor cell in **241.6 s against the base's 155.1** — a hinted
tier that lost to the tier it was supposed to improve.  Caching one hint per
domain, caching the domain factorization, and handing back the cached hint by
reference bring it to **161.4 s**; the 4% that remains at one node is the cost
of offering a hint at all, where a single domain has nothing to gain from one.

What it buys, at the trend size over ferrari nodes:

| | 1n | 2n | 4n | 8n |
|---|---|---|---|---|
| base | 44.7 | 765.7 | 592.0 | 406.7 |
| **hinted** | 44.7 | **26.2** | **17.0** | **11.9** |

The base tier is an extreme anti-scaler — one node to two costs it 17x,
because a refined child is created wherever its parent ran and its halo
partners are elsewhere.  The hinted tier scales 3.8x to eight nodes and is
**34x** the base there.  The placement counters at eight nodes say the layer
is doing what it claims: `NUM_EDT_FINISH` spread across ranks 1.00x, useful
work 1.05x, and **97.3% of acquires local**.

## Sizing

- `npx·npy·npz` sets the **base width** and is capped at 1000 by
  `MAX_NUM_UNREFINED_BLOCKS` (`root.h`), which also sizes the root's `dbSize[]`
  array and the root clone's dependence count.  It is not the coverage
  constraint it looks like: refinement multiplies it, and the calibrated
  arguments produce **23,235,399 EDTs** and **44,316,336 datablocks** from 512
  base blocks — 6,725 EDTs per worker at 32 nodes x 108, so no worker goes
  unused.  Peak resident memory stays at 5.5 GB while 1,478 GB of datablocks
  pass through, which is the destroy path working.
- `nx·ny·nz` and `num_vars` set **grain**: block payload is
  `(nx+2)(ny+2)(nz+2)·num_vars·8` B and face payload `comm_vars·(dim1)(dim2)·8` B.
  Raising them makes each clone move more bytes without adding tasks — the lever
  for shifting the program from task-churn-bound to bandwidth-bound.
- `num_tsteps × stages_per_ts` sets **duration**: clones scale as
  `6·stages_per_ts + K` per block per timestep.
- `num_refine` and the object specification set how far the live block count
  grows. Because the growth depends on object geometry, block/EDT/DB counts
  past the golden checksum are **not** a simple closed form in the general
  case (an object left unspecified by `--object` is now a well-defined,
  zero-initialized entry rather than uninitialized memory, but it is still
  degenerate geometry — a zero-size, non-moving object at the origin — so it
  drives refinement differently than a deliberately-specified one). Use
  `--num_refine 0` when you need a run whose counts are a closed form.

For `N` nodes × 15 workers: pick `npx·npy·npz ≈ 4–8 × 15N` so every worker holds
several blocks through a refinement wave, then set `nx/ny/nz` so the block
payload is a few hundred kB to a few MB, then choose `num_tsteps` for the wall
time you want. 1 node × 15 workers: `--npx 4 --npy 4 --npz 4 --nx 16 --ny 16 --nz
16` (64 blocks × 1.87 MB). 8 nodes × 120 workers: `--npx 8 --npy 8 --npz 8`
(512 blocks) — which is the calibrated argument set, sized so the 1-node cell
still fits in memory (~1 GB of level-0 payload) while giving the 8-node cell four
blocks per worker. `--num_tsteps 12` with the default 20 stages keeps the run in
minutes, and `--num_objects 1` pins a reproducible (zero-initialized, unspecified) object rather than omitting the flag.

The calibrated set runs the Dane anchor node (108w+4p) in **155.1 s** at 5.5 GB
resident, which is the scaler window without further tuning.  Note that these
are counter-free numbers: the same cell reads 225.8 s on a tree still carrying
a campaign's `attribution` counters, a 31% instrumentation cost this
datablock-heavy program feels more than most.
