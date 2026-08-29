# hpgmg_dist

*The restructured version of `hpgmg`: the same F-cycle, kernels and answer
on a **rank-persistent decomposition** — spatial box homes, per-rank slice
fan-outs in place of the central per-phase spawner, a face-slab ghost
exchange, and a fully distributed initialization.*
Source: `third_party/ocr-apps/apps/hpgmg/refactored/ocr/sdsc/` — a separate
program (`hpgmg_dist_main.c` + `mg_dist.c` + `exchange_dist.c` + `mg_dist.h`,
~1.6k lines) that links the base port's kernel and per-box task sources
unchanged; the base files carry no dist conditionals at all.  Selected as
`hpgmg:restructured`.

## Overview

The base program anti-scales for two structural reasons the hint tier
cannot reach.  First, its solve is a serial rope of ~1,200 phases, and
every phase is spawned centrally: one level EDT creates all `N` per-box
tasks — at 8 ranks, thousands of remote creations serialized through one
worker, once per phase, while the whole cluster waits.  Second, 94% of its
end-to-end time is initialization: one worker on rank 0 evaluates the
analytic problem over every cell of every box (~10² s) before any task
parallelism exists.  The restructure attacks exactly these two terms and
changes no numerics: every `||error||` digit matches the base program.

**Per-rank phase chains (no spine).**  The solve runs as one chain per
rank over the F-cycle's phase sequence, gated point-to-point through
per-rank-homed completion events: a phase touching only the rank's own
boxes waits for that rank's previous phase alone, the halo hand-offs
(pack/unpack) wait for the rank-grid neighbourhood — which bounds chain
skew at every exchange and also orders prolongation's writes into
straddling fine boxes — and level transitions, the bottom solve and the
norm final wait for everyone.  Each phase is one pinned FINISH *slice*
that creates its rank's per-box tasks locally (splitting into local
sub-slices when the box count is large, so creation itself parallelizes
within the rank) and counts their completions locally.  Nothing is
centrally spawned and no per-phase cost grows with the rank count.  Box
tasks are pinned with their spatially-homed boxes (`boxHomePD`), so a box
datablock's RW turns never leave its rank; the dist bodies drop the
per-operator timestamp writes, so the level datablocks stop changing
during the solve and every slice's RO acquire dedups to a header check.

**Face-slab exchange.**  Each box owns six slabs of `box_dim²` doubles.
An exchange phase packs the exchanged vector's interior boundary planes
into the box's own slabs (pack fan-out), then fills every ghost layer from
the neighbours' slabs (unpack fan-out, gated on the pack subphase's
finish).  Edge and corner ghosts of the 26-neighbour form read the
boundary lines/points a full face slab already contains.  The box
datablocks never cross ranks; the only recurring cross-rank payload is
slabs (~8 KiB against the 3.77 MB whole-box pull they replace).  An
out-of-domain neighbour's dependence slot carries the box's *own* slab —
mask-guarded, never read — so the base port's full-size constant box is
gone entirely.

**Distributed initialization.**  Per-rank creator EDTs allocate, zero and
describe their own boxes and slabs *on their own rank* (data is born where
it lives — under the base port, every box's payload was first-touched on
rank 0 and had to migrate out).  The fine level's analytic fill is spawned
as per-box tasks behind the creators.  A merge assembles the guid tables
into the level datablocks, and the operator build — coefficient
restriction down the hierarchy, the Gershgorin Dinv/L1inv sweep with its
eigenvalue reduction, and a 26-neighbour ghost exchange of both diagonals
over the face slabs — runs as sliced phases with per-rank partial-max
reductions.  The finalization error norm is reduced the same way (per-box,
per-rank, then a printing final), replacing the base finalizer's
acquisition of every fine box RW on rank 0.  `alpha_is_zero` is not
computed: nothing in the solve path reads it (it only ever fed the
periodic-BC branch this build compiles out).

## Parameters

Same dials as `hpgmg` (`log2_box_dim`, `target_boxes`).  The pinned
`||error||` 9.2779e-10 is O(h²)-consistent with the 512³ pin (ratio 3.985
≈ 4) and identical across arms and geometries.  At most 64 ranks (a
loud-failed compile-time bound of the slice tables).  What the dials are
set to, and why, is under **Sizing**.

## Structure deltas vs the base program

- **DBs**: +6 slabs per box (`6·Σ N_l` = 77 256 at `['5','4096']`,
  `box_dim²·8` B each); −1 constant box per level; small per-reduction
  result DBs (destroyed by their merges).  The per-box `temp` pointer
  array and its DB are gone (nothing holds cross-rank native pointers).
- **EDTs**: each spine phase adds R slices and each exchange phase becomes
  pack + unpack subphases (2 + 2R EDTs); each box's exchange work is one
  pack + one unpack instead of one pull task.  Initialization becomes
  R creators + per-box fill tasks + ~6 sliced phases per level, replacing
  a single serial call tree.
- **Events**: +2 lingering (FINISH + output) per subphase and ~R ONCE
  events per reduction — all bounded by the phase count (`TIMED = 1`).

## Wiring

`hpgmg.h`'s `HPGMG_DIST` block redirects `do_solves` to the lattice
builder (and `finalize` and the init-time phase names to their dist
bodies), so hpgmg.c's top chain is untouched.  The lattice is built in
two rounds: per-rank creators make and home their own phase events and a
rendezvous exchanges the guid table, then per-rank chain builders create
every phase task with its gate dependences — registration may trail a
neighbour's satisfaction, which is legal because events fire and linger.
A phase's slice and all its children complete inside its finish scope
before its completion event fires; the schedule is the exact serial
phase order, so the math is order-identical to the base program.  The
init chain (creators → merge → per-level operator phases) hands its
final event to `top_warm`.  Slab guids live in the level datablock's
reserved tail (`b_norms + N·8` — space the original allocation always
reserved), so `level_type` itself is unchanged.  Guids ride paramv as
64-bit images (`memcpy`, matching the base port's own PRM-struct idiom).

## Flow

A phase is one pinned FINISH slice.  It fires when its gate events have
fired — the rank's own previous phase for a phase that touches only that
rank's boxes, the rank-grid neighbourhood at a halo hand-off, everyone at a
level transition, at the bottom solve and at the norm — then creates its
rank's per-box tasks locally and counts their completions locally, and its
completion event fires when the slice and all its children have left the
finish scope.  An exchange is two subphases: pack fills a box's own six
face slabs from the exchanged vector's interior boundary planes, and
unpack, gated on the pack subphase's finish, fills every ghost layer from
the neighbours' slabs.  Finalization reduces `||error||` per box, then per
rank, then in a printing final that shuts down.  The schedule is the exact
serial phase order, so the arithmetic is order-identical to the base
program.

## Placement (base)

Boxes have spatial homes (`boxHomePD`) and every box task is pinned with
its box, so a box datablock's RW turns never leave its rank.  Per-rank
creator EDTs allocate, zero and describe their own boxes and slabs on their
own rank, so a box's payload is born where it lives; under the base program
every box was first-touched on rank 0 and had to migrate out.  Phase
completion events are homed per rank as well, which is what lets a phase
wait point-to-point instead of on a central spine.

There is no separate `hinted` version.  The decomposition is the placement
here -- it is structural, not a layer a hint could add or remove.

## Sizing

Calibrated as what this row now is, a scaling application: catalog args
`['6','4096']` — a 1024³ grid of 64³-cell boxes, ~113 GiB — whose dane1
anchor (108w+4p) runs ~128 s in the ~150 s class and IS the worst cell.

Trend sweep (bentley, 15w+1p, at the trend size `['5','4096']`, E2E / solve):

| arm | 1 n | 2 n | 4 n | 8 n |
|-----|-----|-----|-----|-----|
| val_wb | 22.3 / 6.1 | 21.3 / 12.1 | 17.3 / 11.7 | 17.5 / 13.8 |
| val_wb_comb | 22.2 / 6.1 | 12.9 / 4.3 | 7.5 / 2.9 | **5.1 / 2.5** |
| inv_wb | 21.9 / 6.2 | 12.7 / 4.6 | 7.7 / 3.2 | **5.1 / 2.5** |
| excl_retain | 22.4 / 6.3 | 12.5 / 4.6 | 7.7 / 3.2 | 5.3 / 2.8 |

The application's first positive scaling: 4.3× from 1 n to 8 n on the
comb/INV arms, against a hinted base that *degrades* 112 → 184 s over the
same geometries — a ~36× gap at 8 n.  The decomposition splits as:
initialization 106 s serial → ~16 s at 1 n → ~2 s at 8 n (rank-parallel
creation, zeroing and fill), and the 1 n solve, 6.1 s, matches the hinted
base's — the slicing costs nothing single-node.  Bare val_wb is the
outlier (12.1 s solve at 2 n): without request combining the slice
fan-outs' per-task RO acquires of the level datablock and slabs each pay
a validation round, so combining is this decomposition's natural partner;
INV needs no ledger at all (a slab read is a covering load) and matches
it.  There is no spine: the phases run as one chain per rank, gated
point-to-point (own-chain for the pure-local phases, the rank-grid
neighbourhood at the halo hand-offs, everyone at level transitions), so
no per-phase cost grows with the rank count — the term that would have
dominated the 32-node campaign is structurally absent.
