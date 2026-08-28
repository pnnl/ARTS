# tempest

*A cubed-sphere halo exchange with the physics removed: 6 panels of `k×k`
patches trade an 8-byte "who am I" block with up to 8 neighbours, 100
timesteps, over persistent CHANNEL events.*
Source: `third_party/ocr-apps/apps/tempest/refactored/ocr/intel-bryan/tempestCommunication.c`
(~980 lines).

## Overview

Bryan Pawlowski's (Intel, 2015) OCR-ification of the Tempest atmosphere model
(P. Ulrich, UC Davis) — but only the *communication* skeleton survived the
port. The program builds the cubed-sphere patch topology (6 panels, `k×k`
patches each, `patchNum = panel·k² + k·x + y`), works out each patch's 8
neighbours across the panel seams, and then exchanges one `nbData_t` (a single
`s64` patch number) per direction per timestep. There is no state vector and
no arithmetic: a patch's whole timestep is "stamp my number into each block I
received and hand it on".

Patch `TEST_PATCH` (0) prints its computed neighbour grid, then
`*CROSS-CHECKING NEIGHBOR DATA EXCHANGE*` and the 3×3 grid of patch numbers it
actually *received*; the two match iff every block travelled the edge it was
wired to. The catalog's scalar is the last cell of the received grid — the SE
neighbour — `21` at the default `k=2`, `46081` at `k=96`. It is a narrow
oracle: one direction of one patch, the other eight cells printed but not
extracted.

What it stresses is task churn and fine-grain exclusive data movement: every
timestep is `6k²` tasks, each acquiring **nine RW datablocks** and issuing
eight event satisfies, with nothing to compute in between. The persistent-
channel idiom keeps event objects out of the steady state (created once at
setup, reused for every generation), so what remains is EDT creation plus
per-node-exclusive block migration — a coherence probe, not a FLOPS benchmark.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|------------------|
| `argv[1]` = `k` | patches per panel side; total patches `6k²` | 2 | ✓ parsed in `mainEdt` (`atoi`), passed as `realmainEdt`'s `paramv[0]`, stored in every panel DB and copied into every patch DB — multinode-safe |
| (arg count) | `argc == 1` (no args) keeps the documented default `k=2`; `argc == 2` parses `argv[1]` | — | ✓ no-args default is intentional; a wrong argument count or a non-numeric `argv[1]` is now a loud usage error (prints `USAGE:` and shuts down) instead of silently running `k=2` |
| `duration` | timesteps, i.e. `patchEdt` generations per patch | **the second argument**; the source's `#define DURATION` is 100 and remains the default when the argument is absent | ✓ an argument — width comes from `k`, run length from here, and the two are therefore independent |
| `TEST_PATCH` | which patch prints the cross-check | 0 | ✗ compile-time (`#ifndef`-guarded, so `-DTEST_PATCH=` would work, but the CMake target does not set it) |
| channel `maxGen`/`nbSat`/`nbDeps` | requested `2`/`1`/`1` per halo channel | — | ✗ in-source; the shim *requires* `nbSat=nbDeps=1` and ignores `maxGen` (ARTS channels are unbounded MPSC), so there is no backpressure knob |

`k` moves parallel width, object count and memory together — nothing scales the
work *per* task — so `k` is the width dial and `duration` is the length dial.

## Structure

Every patch has 8 neighbours except the 4 corner patches of each panel, which
lack one diagonal, so the directed halo edges number `E(k) = 48k² − 24`. (The
topology was checked exhaustively up to `k=96`: every edge has a matching
reverse edge, no patch names the same neighbour twice, and every handshake slot
has exactly one publisher and one learner.)

| object | count | size / note |
|--------|-------|-------------|
| EDTs | `612k² + 9` | `600k²` `patchEdt` (`6k²` chains × 100 generations) + `6k²` `patchInit` + `6k²` `channelSetup` + 6 `panelInit` + `realmain` + `wrapup` + 1 `mainEdt` itself (the OCR shim creates it as an EDT — `arts_edt_create(mainEdtTrampoline, ...)` — before its body runs; not one of `mainEdt`'s own explicit `ocrEdtCreate` calls, so the earlier count missed it) |
| DBs | `102k² − 18` | 6 panel (80 B) + `6k²` patch (224 B) + `E(k)` channel-handoff (8 B) + `48k²` halo seeds (8 B) |
| Events created | `144k² − 70` | see accounting below |
| Live event objects | `96k² − 46` | `E(k)` CHANNEL + `E(k)` labeled sticky + 2; nothing is ever destroyed |
| EDT templates | `12k² + 9` | pure GUID encodings under ARTS, not runtime objects |

Event accounting: one `OCR_EVENT_CHANNEL_T` per directed edge, plus **two**
`ocrEventCreate` calls per labeled sticky slot — publisher and learner both
create the same `GUID_PROP_IS_LABELED | GUID_PROP_CHECK` GUID and the loser's
install is rejected, so `2·E(k)` creates yield `E(k)` objects. Exactly one
`ocrEdtCreate` passes a non-NULL `outputEvent` (`realmain`, feeding `wrapup`)
and that same EDT is the only `EDT_PROP_FINISH`, whose finish event the shim
pre-creates: `+2`. Every other `ocrEdtCreate` passes NULL and creates nothing.

Worked numbers at the calibrated `args: ['96']` — 55,296 patches, 442,344 halo
edges: **5,640,201** EDTs (5,529,600 of them `patchEdt`), **940,014** DBs,
**1,327,034** event creates for **884,690** live events, ≈18.6 MiB of payload,
≈49.8 M RW acquires over the run (`9 · 6k² · duration`). Default `k=2` → 24
patches, 2,457 EDTs, 390 DBs, 506 event creates.

Counter cross-check: verified (1 node, `k=4` vs `k=8`, measured totals
9,802/39,178 EDTs, 1,615/6,511 DBs, 2,234/9,146 events). NUM_EVENT_CREATE
matches `144k² − 70` exactly with no offset; NUM_DB_CREATE matches
`102k² − 18` plus the runtime's constant +1 DB per run; NUM_EDT_CREATE needed
the `mainEdt` correction above — it matches `612k² + 9` plus that same
constant +1 EDT per run (formula values 9,801/39,177, +1 = measured).

## Wiring

`mainEdt` (rank 0) creates 6 panel DBs and hands them RW to `realmainEdt`, the
run's single FINISH EDT, whose output event fires `wrapupEdt` (`DONE.` +
`ocrShutdown()`). `realmain` reserves eight labeled sticky GUID ranges of
`6k²` each — one per direction, used only for the one-time channel handshake —
stamps them into every panel DB and forks 6 `panelInit`s.

`panelInit` loops `k²` times: one 224 B patch DB and one `patchInit` per patch,
wired `panel DB → slot 0 (RW)`, `patch DB → slot 1 (RW)`. **No `patchInit`
writes the panel block** — it only reads `patchRange` and the GUID ranges — yet
all `k²` of them take it RW, i.e. exclusively.

`patchInit` computes the 8 neighbours, then per existing direction `i` creates
a CHANNEL event (its *receive* queue for that direction) and publishes it: an
8 B DB holding the channel GUID, satisfied into the labeled sticky at
`(range[rel], neighbour)`, where `rel` is the neighbour's direction back at me;
symmetrically it wires its `channelSetup`'s slot `i` (RO — the program's only
RO dependence) to `(range[i], me)`, where that neighbour publishes.
`channelSetup` records the learned GUIDs as `sendChannels[]`, mints 8 fresh 8 B
`nbData` seed blocks (unconditionally, corner slot included) and launches
generation 0. `patchEdt(g)` then creates generation `g+1`, wires
`recvChannels[i] → slot i (RW)` (one dependence = one pop from the FIFO),
writes its own patch number into each received block, releases it and satisfies
the neighbour's channel with it, and finally releases its patch block into slot
8 (RW). Blocks are never re-minted: `P`'s seed for direction `i` ping-pongs
across the `P↔Q` edge for the whole run.

DB concurrency is uniformly exclusive — one accessor at a time, no DB ever has
two readers, and no block takes RW from more than two patches (halo) or one
(patch state). The contention points are therefore not the halo but (a) the
**6 panel DBs**, each serialising `k²` exclusive turns during init, and (b) the
**single global finish-scope latch**, which every EDT create INCRs and every
completion DECRs — `2·(612k²+8)` satisfies on one event, 11.3 M at `k=96`.

## Flow

`mainEdt` is rank-0-only but O(1); `realmain` is one EDT. Width is then **6**
for the whole creation phase: each `panelInit` runs a `k²`-iteration serial
create+wire loop, and the `patchInit`s it spawns — nominally `6k²` ready tasks
— serialise behind their panel block's RW chain, `k²` deep per panel.
`channelSetup` is the first genuinely wide phase (`6k²`, each gated only on its
8 published slots). Steady state is 100 generations of `6k²` independent tasks
with no global barrier and no rank-0-only phase; because generation `g+1` of a
patch needs generation `g` of each neighbour, skew between two patches is
bounded by their graph distance rather than by a barrier. The tail is
symmetric: `patchEdt` at `timestep == 99` prints (patch 0 only) and returns
without a successor, the finish scope drains, `wrapup` shuts down. Nothing is
destroyed anywhere in the program — no `ocrDbDestroy`, no `ocrEventDestroy` —
so every object created stays live to the end.

## Placement (base)

Every create passes `NULL_HINT`: the `makePatchEdtHint`/`makeLocalEdtHint`
helpers return `NULL_HINT` unless `OCR_APP_OPTIMIZED_PLACEMENT` is defined (the
`_hinted` build), and there is no affinity use outside that guard — the base
program never calls `ocrAffinity*` at all. Effective policy: **EDTs
round-robin, DB home = creating rank.** Hence:

- The 6 panel blocks are born on rank 0, migrate to `realmain`'s rank, then to
  each `panelInit`'s rank, then bounce through `k²` random `patchInit` ranks —
  an 80-byte block dragged through `6k²` exclusive cross-rank grants at startup.
- A panel's `k²` patch blocks are all homed on that one `panelInit`'s rank (six
  ranks host every patch block), while the tasks that touch them are scattered.
  Each patch block is then RW-acquired by 102 successive EDTs (`patchInit`,
  `channelSetup`, 100 generations), each placed independently at random — 224
  bytes of per-patch state migrating once per timestep, remote with probability
  `(N−1)/N`.
- Halo seeds are homed wherever `channelSetup` ran and move once per timestep
  each: `99·E(k)` migrations of 8-byte blocks between two moving holders.
- Channel events live on their patch's `patchInit` rank, so every satisfy and
  every dependence registration is a message to a third, unrelated rank; the
  labeled sticky slots are spread by index, making each handshake a three-party
  rendezvous.

The algorithm has textbook nearest-neighbour locality on the sphere and the
base program expresses none of it: no two objects of a patch are placed
together, and re-placing the chain every generation means locality can never
even accumulate.

## Placement (hinted)

As-born is placement-blind: each timestep's patch EDT lands round-robin, so a
patch's halo exchange partners are arbitrary ranks and its persistent halo
blocks (created once at setup, reused every generation) are acquired remotely
almost every turn.

The layer (`patchHomeRank`) maps the cube-sphere's 6 x k x k patches onto a
P x Q rank grid chosen from the divisors of the rank count to minimise the cut
(the number of patch edges crossing rank boundaries), unrolling the six faces
along one axis; each patch EDT is pinned to its patch's home rank every
generation (`OCR_HINT_EDT_AFFINITY`).  With EDTs stationary, the reused halo
blocks' ownership settles on the consumer's rank after the first turn.  Below
6 ranks the map degenerates to contiguous patch bands.

## Sizing

`k` scales the *number* of tasks and blocks (`6k²` patches, `∝ k²` time) and
never their size -- a `patchEdt` does the same eight stores at any `k` -- so `k`
is the width dial and `duration`, the second argument, is the length dial.  Memory
never decides: 1-2 GB at any size measured here.

Width comes from the class rule: four times the largest geometry's 3456 workers
is 13,824, and `6k² = 13,824` puts `k` at 48 exactly.

The length then comes from the window, and this row's window is the short one,
because it anti-scales harder than anything else in the roster:

| geometry, k=16 | time |
|---|---|
| 1 node x 15 workers | 4.78 s |
| 2 nodes x 15 workers | ~1125 s (still running at the 900 s cap, 3513 of 4400 timesteps) |

**235x worse across one node boundary.**  That is what a program whose task does
eight stores and then exchanges with eight neighbours looks like once half those
exchanges cross a rank -- the communication is the entire program.  So the
anchor is calibrated against 10-30 s, and `duration=1900` at `k=48` measures
19.0 s, 19.3 s and 20.2 s on the three coherence families, holding 1 GB.

The placement layer matters here more than anywhere else in its cycle: at four
nodes it turns 454.2 s into **34.7 s, a factor of 13.1**.  Cutting the cube's six
faces into minimum-cut 2-D blocks makes almost every halo neighbour rank-local,
and in a program that is nothing but halo exchange that is nearly the whole
cost.
