# hpgmg

*Full-multigrid Poisson solve on a cubical box grid — one EDT per box per
operator, one serial barrier per level visit.*
Source: `third_party/ocr-apps/apps/hpgmg/refactored/ocr/sdsc/` (11 files, ~3.5k
lines; SDSC's OCR port of Sam Williams' HPGMG-FV).

## Overview

Solves `-∇·(β∇u) = f` on the unit cube with Dirichlet boundaries by an **F-cycle
(FMG)**: restrict the right-hand side to the coarsest grid, solve there, then
prolong one level at a time, running a full V-cycle after each prolongation.
Each V-cycle level does 4 Chebyshev smooths, a residual, a restriction and (on
the way up) a prolongation; the bottom is closed by a serial BiCGStab
(`solve_edt.c`).  The domain is cut into `boxes_in_i³` cubical boxes, one
datablock each, holding 12 grid vectors (`u`, `f`, `f_Av`, `u_true`, `alpha`,
`beta_{i,j,k}`, `Dinv`, `L1inv`, `valid`, `vec_temp`) plus one ghost layer.

The result scalar `||error||` is the max-norm of `u − u_true` on the finest grid
— the *discretization* error, O(h²), a property of the grid rather than of the
solver's iteration count, so it moves with the arguments (hence `expect` is
pinned to `expect_args`); `f-cycle, norm=` reports the scaled residual norm.
What it stresses is a **bulk-synchronous DAG with collapsing parallel width**:
every operator is a fan-out of `num_boxes` short EDTs joined by a finish event
and the level chain is strictly serial, so the program is a long sequence of
barriers whose width falls 8× per agglomeration step down to a single box.  The
arithmetic is real (a 7-point variable-coefficient stencil), but at the coarse
end the cost is entirely barrier + coherence.

## Parameters

`mainEdt` requires exactly two arguments (`argc != 3` prints usage and shuts
down).

| arg | meaning | default | CLI reachability |
|-----|---------|---------|------------------|
| `argv[1]` = `log2_box_dim` | box edge in cells, `box_dim = 1 << log2_box_dim`; must be ≥ 4 | none (required) | ✓ parsed in `mainEdt`, consumed entirely inside `init_all`; every derived value lands in the level/box DBs and the app has no file-scope globals — multinode-safe |
| `argv[2]` = `target_boxes` | **cap** on total boxes: the largest `boxes_in_i` with `boxes_in_i³ ≤ target_boxes` is used | none (required) | ✓ same path.  ⚠ the error text calls it `target_boxes_per_rank`, but it is never divided by the rank count — it is a global cap |

Everything else is compile-time: ✗ `TIMED`/`WARMUP` (1 / 0 in `hpgmg.h` — timed
and warm-up F-cycle counts; `WARMUP = 0` makes `top_warm` a pass-through), ✗
`NUM_SMOOTHS`(1) × `CHEBYSHEV_DEGREE`(4) = 4 smooth sweeps per level visit, ✗
`MG_AGGLOMERATION_START`(8), the box edge below which the hierarchy stops
halving boxes and starts merging 8 into 1; and legitimately fixed:
`NUM_VECTORS`(12) / `NUM_GHOSTS`(1) (they set the box DB layout),
`STENCIL_VARIABLE_COEFFICIENT` + `STENCIL_FUSE_BC` (`operators.h`),
`minCoarseDim = 1`, `jMax = 200`, `desired_reduction_in_norm = 1e-3`.
`BC_DIRICHLET` is a literal at `hpgmg.c:56`, so `BC_PERIODIC` is unreachable.

## Structure

`mg_build` derives the level table: halve `box_dim` while it exceeds 8, then
halve `boxes_in_i` (8:1 agglomeration) until one box remains, then halve
`box_dim` again down to 1 — for a power-of-two `boxes_in_i` this always ends in a
**four-level single-box tail** with `box_dim` 8, 4, 2, 1.  The worked example
below uses `['5','512']` (`box_dim = 32`, `boxes_in_i = 8`, `L = 9`); the
catalog's campaign size `['5','4096']` is the same structure one doubling wider
— `boxes_in_i = 16`, `L = 10`, three 4096-box levels before agglomeration, and
every formula below scales accordingly:

| level | box_dim | boxes_in_i | boxes `N` | one box DB | all boxes |
|-------|---------|-----------|-----------|------------|-----------|
| 0 | 32 | 8 | 512 | 3 773 200 B | 1 842 MiB |
| 1 | 16 | 8 | 512 | 559 888 B | 273 MiB |
| 2 | 8 | 8 | 512 | 96 016 B | 46.9 MiB |
| 3 | 8 | 4 | 64 | 96 016 B | 5.9 MiB |
| 4 | 8 | 2 | 8 | 96 016 B | 0.7 MiB |
| 5–8 | 8,4,2,1 | 1 | 1 | 96 016 … 4 048 B | < 0.2 MiB |

A box DB is `16 + 12·(box_dim+2)·kStride·8` bytes (`kStride = (box_dim+2)²`,
floor-padded to 8 in the pencil); each level also carries one *constant box* of
that size (the out-of-domain neighbour sentinel) and a level DB of `240 + 64·N`
bytes.  **DBs = `2 + 3L + Σ N_l`** at init, plus one scratch per `solve_edt`
(`L`) and one in `print_timing_edt` → **1 651**, of which 1 612 are boxes;
payload **≈ 2.12 GiB**, live for the whole run.

EDTs come in two tiers.  The **level chain** is materialised up front —
`do_solves` is a plain C function running inside `top_loop`, so all of it exists
before most of it runs.  With `S = 4`, `T = L(L−1)/2` (= 36 here):
`exchange_level_edt` `(L−1)+(2S+1)T+1` = 333, `smooth_level_edt` `2S·T` = 288,
`time_edt` `3+(L−1)+5T` = 191, `restrict_level_edt` and `interpolate_level_edt`
`(L−1)+T` = 44 each, `residual_level_edt` `T+1` = 37, `zero_vector_level_edt`
`T` = 36, `solve_edt` `L` = 9, plus 2 `init_ur_level_edt`, 3 norm/mulv and 6
driver EDTs — **993**.  Each spawns one EDT per box of its level (restriction
and zeroing per *coarse* box); fan-out phases per level number `23` at level 0,
`21(l+1)` for `1 ≤ l ≤ L−2` and `3L+1` at the coarsest, so per-box EDTs
`= 23N_0 + Σ 21(l+1)N_l + (3L+1)N_{L−1}` = **72 221** (31 013 6-neighbour
exchanges, 27 112 smooths, 3 901 residuals, 2 886 restricts, 2 886
interpolations, 1 786 zeroings, 1 100 26-neighbour exchanges, 1 537
init/mulv/norm) — **73 214 EDT creates** in all.

Events, over all three shim sources: `ocrEventCreate` (`L+1`), the output event
of every `ocrEdtCreate` with a non-NULL `outputEvent`, and one finish event per
`EDT_PROP_FINISH`.  Nearly every level EDT is finish-with-output → **2 events
each**; `solve_edt`, `print_timing_edt` and `finalize_edt` are output-only → 1;
every per-box EDT passes NULL → 0.  Total **1 978**.
Dependence slots resolved ≈ 375 000: 189 k box-RO, 74 k box-RW, 77 k level-RO,
34 k on the nine constant boxes.

Counter cross-check: taken at `4 1` / `4 8` / `4 64` (L = 5/6/7) against a
since-reverted norm-collector variant; the formulas above are the base
wiring with that variant's `N_0` result blocks, output events and collector
subtracted analytically (ΔNUM_EDT_CREATE = +698 / +4 371 between the sizes is
unchanged; a fresh counter run should reconfirm the absolutes).  The absolutes
then read NUM_DB_CREATE = 29/48/172 and NUM_EVENT_CREATE = 586/856/1178 plus
the runtime's usual +1 DB / +0 event bootstrap — and NUM_EDT_CREATE needs **+2 EDT, not
+1**: `main_edt` (`libs/src/core/system/runtime.c:535`) is itself one EDT,
and its body — the OCR shim's own `main_edt`
(`benchmarks/ocr_shim/arts_ocr.c:2159-2196`) — creates the argv DB *and* a
second EDT, `mainEdtTrampoline`, to carry the app's `mainEdt` in as a DB
dependence. Both run before any app code, so every total here is
`app formula + 2` EDTs (73 214 → **73 216** at
`['5','512']`) — a shim-bootstrap fact, not specific to this app.

## Wiring

The level chain is a single dependence rope: every level EDT takes its
predecessor's finish event on its last slot, so phase `k+1` cannot start until
every child of phase `k` has completed and released.  The other events are one
ONCE event per restriction sequence, which `restrict_edt` satisfies **with
the coarsest box's GUID** (`ocrDbRelease` then `ocrEventSatisfy`,
`restrict_edt.c:77`) — that is how the bottom solve receives its datablock, on
an **RW** slot, since BiCGStab updates that box in place.

Per-box EDTs uniformly take the level DB on slot 0 and their own box on slot 1.
`smooth_edt`, `residual_edt`, `init_ur_edt`, `mulv_edt` and `zero_vector_edt`
take level RO + box **RW**; `norm_edt` takes the level **RW** and its box RO,
depositing one double into the level's `b_norms` slot — so the whole norm
fan-out serializes on the level datablock's per-node-exclusive turns, the
port's starkest one-datablock fan-in (kept as it ships: it is part of what the
base tier exhibits).  `exchange_edt` adds 6 (face) or 26 (face+edge+
vertex) neighbour boxes **RO**; out-of-domain neighbours resolve to the level's
single `constant_box_guid`, so one DB appears `6B²` times per 6-neighbour round
and `27B³ − (3B−2)³` times per 26-neighbour round — at level 0 of the calibrated
size, **384 and 3 176 RO slots on one 3.77 MB datablock**.  `restrict_edt` is
one per coarse box: coarse box **RW**, both level DBs RO, its `N_fine/N_coarse`
fine boxes RO; `interpolate_edt` mirrors it with the fine boxes **RW** and the
coarse box RO for the piecewise-constant (V-cycle) prolongation, **RW** for the
linear (FMG) one, which applies the boundary condition to that box's ghost
cells before reading it.

DB concurrency, worst first.  The **level DB is the contention point**: its
level EDT holds it RW while all `N_l` children hold it RO (children are created
and satisfied inside the parent's body, so the overlap is real, not nominal).
Next is the constant box, read-only but with the fan-in
above.  A regular box has at most 27 simultaneous accessors during a
26-neighbour exchange — its own writer plus 26 readers; the writer touches only
ghost cells and the readers only interiors, so the regions are disjoint even
though the datablock is not.  Finally `finalize_edt` takes **every level-0 box
RW in a single EDT** (`N_0 + 3` slots, 1.8 GiB at the calibrated size).

## Flow

One F-cycle: `time_all` → `init_ur` ×2 → `restrict_all` (`L−1` restrictions down
the hierarchy) → bottom `solve_edt` → for `l = L−1 … 1`: prolong to `l−1` and run
`vcycle(l−1)` → `scaled_residual_norm` → `time_all`.  `vcycle(ln)` descends
`ln … L−2` (smooth ×4, residual, restrict, zero), solves at the bottom, and
ascends with interpolate + smooth ×4.

Parallel width in a phase at level `l` is exactly `N_l = boxes_in_i(l)³` and is
the *only* parallelism — nothing at level `l` overlaps anything at level `l'`.
For `['5','512']` the widths are 512, 512, 512, 64, 8, 1, 1, 1, 1, while the
*number* of phases runs the other way (FMG visits a coarse level once per
V-cycle that reaches it): 23 at level 0, 168 at level 7.  So **469 of the 786
fan-out phases (60%) have width 1** and together carry 0.015% of the arithmetic;
a level-8 phase is a full barrier around one EDT that updates one cell.  That
inversion — barrier count rising as work falls — is the defining shape of the
program.

Serial bottlenecks, largest first: (1) the native preamble inside `mainEdt` —
`init_all` allocates and zeroes all 1 612 boxes, evaluates the analytic solution
and coefficients over `N_0·(box_dim+1)³` cells (18.4 M here, ~65 `pow`/`tanh`
calls each) and rebuilds the operator on every level, all on one worker of rank 0
before any EDT exists; (2) the `L` bottom solves, each a BiCGStab of up to 200
iterations inside one EDT on one box; (3) the single-box tail; (4)
`finalize_edt` over all level-0 boxes; (5) `top_loop`'s body, which issues all
993 level creates and ~2 000 `ocrAddDependence` calls itself.

## Placement (base)

Unusually for this catalog the base program places things explicitly —
`ENABLE_EXTENSION_AFFINITY` is on for every benchmark build and most of the
affinity code sits *outside* the `OCR_APP_OPTIMIZED_PLACEMENT` guard.  (The
guarded layer exists: it swaps the box home for a spatial 3-D partition and adds
hints to the four operators that lack them.)  As-born:

- **Box DBs** carry an explicit `OCR_HINT_DB_AFFINITY = box_num % rank_count`
  (`init.c:221`) at every level — round-robin over a linear 3-D index.
- **`smooth_edt`, `residual_edt`, `restrict_edt`, `interpolate_edt`** get EDT
  affinity from `ocrAffinityQuery(box)`, i.e. the box's home — these four *are*
  co-located with the box they write.
- **Everything else is hint-less** → runtime round-robin: `exchange_edt`,
  `init_ur_edt`, `zero_vector_edt`, `mulv_edt`, `norm_edt`, `norm_merge_edt`
  and all 993 level EDTs.  Level, `mg` and scratch DBs are `NULL_HINT` created in `mainEdt` →
  home = rank 0, and each level's constant box inherits box 0's affinity → also
  rank 0.

The resulting traffic: the ghost exchange — the most data-heavy phase, 32 113 of
the 72 221 per-box EDTs, each acquiring 1 box RW and 6–26 RO — is placed with no
relation to any of its boxes, so essentially every one of its ~215 k neighbour
RO slots is remote.  Round-robin over a linear box index also scatters spatial
neighbours onto consecutive ranks, so even a co-located operator's *neighbours*
are remote, and a coarse box and the 8 fine boxes it covers land on unrelated
ranks, making every restriction and prolongation edge cross.  Rank 0 further
homes all nine level DBs (77 k RO acquires, and the norm collector's RW turn)
and all nine constant boxes (34 k RO acquires).  The locality the
algorithm has — spatial adjacency within a level, parent/child nesting across
levels — is real and completely unexpressed.

## Placement (hinted)

As-born already places — box DBs home round-robin (`box_num % affinityCount`)
through the app's own `ENABLE_EXTENSION_AFFINITY` code — but round-robin
scatters both the intra-level neighbourhoods and the inter-level ladder: a
coarse box almost never lands with the fine boxes it restricts from / prolongs
to, so every V-cycle rung crosses ranks.

The layer replaces the map, not the mechanism: `boxHomePD` partitions the unit
cube once into a near-cubic PX x PY x PZ rank grid and sends every level's box
through THAT one partition by its normalized (i,j,k)/S position — neighbouring
boxes of a level co-locate AND a coarse box lands on the rank of its fine
children, so inter-level transfers stay rank-local.  EDTs follow their box via
`ocrAffinityQuery(box)` (`mg_edt.c`), keeping compute with data.  This is the
spatial box-home design adopted in the 2026-08-07 anti-scale verdicts; the
residual granularity amplification (~460x per remote face read) is structural
and out of a hint's reach.

Measured (ferrari trend sweep, 15w+1p, `['5','4096']`, val_wb_nocomb / best-of-arm
range): base 111 s → 348 / 330 / 350 s at 2/4/8 n — the whole-box exchange
saturates immediately and flattens; hinted 112 s → 160 / 175 / 184 s
(range across the four arms ≤ 8%, `val_wb` consistently best) — the
layer halves the multinode cost and turns the cliff into a decelerating
creep, but the program still anti-scales.  The Dane anchor cell (108w+4p)
runs 114.5 s.  Two facts locate the residual: the E2E is 94% *serial
initialization* (the app's own solve timer reads 6.2 s at 1 n under 112 s
E2E), and the solve itself anti-scales 10× even hinted (6.2 → 64 s at
8 n).  An exchange-payload-only rewrite (face slabs, ~460× fewer bytes,
structure otherwise unchanged) came back *slower* — the solve's multinode
term is the per-phase turnaround of the centrally-spawned fork-join, not
bandwidth.  Both terms fall to the full restructure: see `hpgmg_dist`
(rank-persistent decomposition + distributed init), which turns
112 → 184 s into 22 → 5.7 s and gives the application its first positive
scaling.

## Sizing

`log2_box_dim` sets the **grain** (`box_dim³` cells per per-box EDT: 4 096 at 4,
32 768 at 5, 262 144 at 6) and dominates memory, `≈ 1.18 · 96 · N_0 ·
(box_dim+2)³` bytes.  `target_boxes` sets the **width**: `N_0 = boxes_in_i³` with
`boxes_in_i = ⌊target_boxes^{1/3}⌋`, a cap rather than a count — 500 and 343 both
give 343 boxes.  Neither changes the number of F-cycles (compile-time
`TIMED = 1`), and `L` grows only logarithmically, so the level chain grows as
`O(L²)` while work grows as `N_0·box_dim³`.

Prefer a **power-of-two `boxes_in_i`** (1, 2, 4, 8, 16 → `target_boxes` 1, 8, 64,
512, 4096).  Other values make the table degenerate — `box_dim` falls below 8
while the box count stays high — and any `boxes_in_i` with an odd factor ≥ 7
drives a fine/coarse box ratio past the `MAX_FINE_BOXES` (125) the fine-box
lists hold, which `get_fine_boxes`/`get_fine_box_ids` refuse with a message
naming the level pair and abort on.  Pick `target_boxes` so `N_0` covers the workers with enough left over to
hide the exchange, and `log2_box_dim` so the grain is worth a task:

- **1 node, 15 workers**: `['5','512']` → 512 boxes, ~34 per worker, 32 768
  cells per task, 2.12 GiB.  `['5','64']` (64 boxes, 276 MiB) is the fast
  variant; `['4','1']` is a single box — strictly serial.
- **Campaign width**: `['5','4096']` → 16³ boxes of 32³ cells, a 512³ grid,
  ~17 GiB — 4 096 boxes cover a 108-worker rank 38-deep at level 0 and keep
  the first three levels at full width.

The catalog's `['5','512']` is the campaign calibration under the
anti-scaler rule: this program's worst cells are its LARGE geometries, so
the anchor is sized small — the dane1 anchor (1 node, 108w+4p) runs
14.5 s, 8 nodes runs 26.1 s, and the 32-node extrapolation stays in the
tens of seconds while the anti-scaling *shape* (the measurement) is fully
visible.  `['5','4096']` is the ferrari 15w+1p × {1,2,4,8} trend size
(~112 s at 1 n), where the measured family table above was taken.
`expect_args` equals `args`: the pinned `||error||` is grid-dependent
(discretization error) and prints identically across arms, node counts,
hint states and the restructured decomposition — and the two sizes'
pins sit exactly a factor ~3.97 apart, the O(h²) second-order signature.
Note that the serial `init_all` preamble does not shrink with node count,
so it grows as a fraction of a strong-scaling sweep.

## Family

The submodule holds one other HPGMG port, `hpgmg4`
(`apps/hpgmg4/refactored/ocr/intel/`): a *fourth-order* HPGMG with red-black
smoothing whose own README opens "Beginnings of an OCR implementation" — a
different discretization and an unfinished one, so it is neither registered
nor comparable.  This SDSC port
is the family's only member; there is no scaling sibling, which is why the
structural anti-scaling above has no in-family control and the catalog
carries a restructured decomposition instead: `hpgmg_dist` (see its own
appdoc) rebuilds the port on a rank-persistent data plane — spatial box
homes with per-rank slice fan-outs, a face-slab exchange, distributed
initialization — with the kernels, F-cycle and answer unchanged, and is
the pairing that supplies the fork-join-vs-persistent comparison for this
family.
