# npb_cg

*NAS Parallel Benchmarks CG, SDSC OCR port: an inverse power method whose
inner conjugate-gradient solve is a chain of sparse matvecs, each one a
fork-join over row blocks of the matrix.*
Source: `third_party/ocr-apps/apps/npb-cg/sdsc-ocr/` (~880 lines over six
files; the DAG is `cg_ocr.c` + `cg_graph.c` + `cg_edt.c`, the driver loop
`main_edt.c`, the matrix generator `makea_ocr.c`).

## Overview

NPB CG estimates the smallest eigenvalue of a random sparse symmetric
positive-definite matrix `A` by the inverse power method: `niter` outer
iterations, each solving `A·z ≈ x` with a **fixed 25-step** conjugate gradient
(no convergence test), then `zeta = shift + 1/(x·z)` and `x = z/‖z‖`. The
result scalar is that `zeta`, printed by `tail_edt` only after it is checked
against the class's reference value to a relative error of `1e-8`, so
`Verification SUCCESSFUL (zeta=…)` is a strong oracle — it observes the matrix
build, every matvec, every dot product and the whole outer loop. The program
stresses all three axes at once: real sparse arithmetic (~2·nnz flops per
matvec), heavy task churn (a fork-join of `na/blk` tasks per matvec, 26
matvecs per outer iteration), and a write-once/read-by-everyone vector that
changes writer every CG step. A restructured twin `npb_cg_dist` exists in the
source tree but is held back from both the build and the catalog.

## Parameters

Arguments come in flag/value pairs and are parsed **backwards** from the end
(`main_edt.c:28-35`); `argc` must be exactly 1, 3, 5 or 7 or the program
prints usage and shuts down.

| arg | meaning | default | CLI reachability |
|-----|---------|---------|------------------|
| `-t <c>` | problem class `T,S,W,A,B,C,D,E`; sets `na`, `nonzer`, `shift`, `niter` and the reference `zeta` from the table in `util_ocr.c:54-120` | `S` (`na=1400`) | ✓ parsed in `mainEdt`, stored in the class DB that every consumer takes `CONST` — multinode-safe |
| `-b <blk>` | rows per row block; the matrix is split into `nb = na/blk` datablocks and each matvec forks `nb` tasks | `1` | ✓ same path (`class->blk`), forwarded to `spmv_edt` in paramv — multinode-safe. **Must divide `na` exactly**, and be `>= 1`; both are checked in `mainEdt`, which prints the offending value and shuts down |
| `-i <n>` | overrides the class's `niter` (outer iterations only; the inner CG count is not affected) | class value | ✓ written into the class DB before its release — multinode-safe. `0` means "no override" |

Only the first character of `-t`'s value is read; an unrecognised class prints
usage and shuts down. Unknown flags are silently skipped, and since the walk
is positional a stray flag misaligns the remaining pairs.

Notable compile-time knobs, none argv-reachable: the **inner CG iteration
count is hardcoded** as `for(it=1; it<24; ++it)` plus an unrolled leading and
trailing iteration (`cg_ocr.c:19`) — 25 CG steps and a 26th matvec for the
residual; the verification tolerance `1e-8` (`main_edt.c:217`, `1e-3` under
`TG_ARCH`); the timing-detail flag `class->on`, pinned to `1` in `class_init`;
and `rcond = 0.1` in `makea`, an NPB constant.

## Structure

Write `n = na`, `b = blk`, `nb = n/b`, `k = nonzer+1`, `C = niter+1`
conj_grad invocations (one untimed warm-up from `mainEdt`, `niter` timed), and
`e = 2·⌈(b+1)/2⌉` (the `even` header pad). Each `conj_grad` issues 26 matvecs.

| object | count | size |
|--------|-------|------|
| matrix row blocks | `nb` | `12·Σ_{rows in block} nz + 4·e` B; `Σ` over all blocks `≈ 12·n·k² + 4·e·nb` |
| matrix GUID container | 1 | `8·nb` B |
| `makea` scratch (transient) | 6 | `4nk + 8nk + 4n + 4n + 8n` B + a 16 B RNG block |
| full-length vectors | `x` once per run; `r,p,rho,z` 4, `q` 26 and `pp` 24 per `conj_grad` | `8n` B each |
| row-block results | `26·nb` per `conj_grad` | `8b` B each |
| scalars `alpha, nalpha, dist` | 50 per `conj_grad` | 8 B each |
| **EDTs** | `2 + 2·niter + C·(128 + 26·nb)` | — |
| **DBs** | `10 + nb + C·(104 + 26·nb)` | — |
| **events** | `C·(150 + 26·nb)` | — |

Per inner CG iteration the loop body is `5 + nb` EDTs (1 `spmv_edt`, `nb`
`rowvec_edt`, 1 `assign_edt`, 1 `alphas_edt`, 1 `daxpy_edt`, 1 `update_edt`),
`4 + nb` DBs and `6 + nb` events; the unrolled first step adds a `square_edt`
and a `scale_edt`, the trailing block adds a second matvec plus `alpha_edt`
and `dist_edt`. No EDT is a finish EDT, so every event is either an explicit
`OCR_EVENT_ONCE_T` (74 per `conj_grad`) or an `ocrEdtCreate` output event
(`76 + 26·nb` per `conj_grad`). EDT templates are created and destroyed around
every single create, but the shim encodes a template into its GUID, so that is
pure app-side churn with no runtime object behind it.

Worked, at the calibrated `-t A -b 25` (`n=14000, k=12, niter=15, nb=560,
C=16, e=26`): **~235.0k EDTs, ~235.2k DBs, ~235.4k events**. The matrix is
`≈ 12·14000·144 + 4·26·560 ≈ 24.2 MB` over 560 blocks, ~42 KiB each; every
vector is 112 kB; peak live set ≈ 27 MB (24 MB matrix + ~8 concurrent
vectors), with a transient 2.2 MB of `makea` scratch. At `-t T -b 25`
(`expect_args`, `nb=2, C=4`): 728 EDTs, 636 DBs, 808 events, a 38 kB matrix.

Counter cross-check: verified (1 node). Both pairs — `-i 3`→`-i 6` (Δ =
+546 EDT / +468 DB / +606 EVT) and `-b 100`→`-b 50` (Δ = +5824 EDT / +5838 DB
/ +5824 EVT) — match the formulas' deltas exactly. NUM_DB_CREATE
(637/1105/7513/13351) and NUM_EVENT_CREATE (808/1414/8224/14048) match
`10+nb+C·(104+26·nb)` and `C·(150+26·nb)` exactly, with the runtime's usual
+1 DB / +0 event bootstrap. NUM_EDT_CREATE (730/1276/7906/13730) needs **+2
EDT, not +1**, over the app formula (728/1274/7904/13728): the OCR shim's
bootstrap creates two EDTs before `mainEdt` ever runs — `main_edt`
(`libs/src/core/system/runtime.c:535`) is one, and its body (the shim's own
`main_edt`, `benchmarks/ocr_shim/arts_ocr.c:2159-2196`) creates the argv DB
and a second EDT, `mainEdtTrampoline`, that carries the app's `mainEdt` in as
a DB dependence. `head_edt`/`tail_edt` (the formula's `2`) already cover
every EDT the app itself creates — `ocrShutdown()` runs directly inside
`tail_edt` (`main_edt.c:250`), no separate shutdown EDT — so the missing +1
is runtime bootstrap, not an app EDT the formula forgot; the same correction
reconciles hpgmg's EDT counts exactly.

## Wiring

`mainEdt` builds the class, timer and `x` blocks, runs `makea` inline, wires
`head_edt` (6 slots) and calls `conj_grad` to fill its last two. `head_edt`
discards that warm-up result, re-initialises `x` and starts the timed loop:
`loop_top_edt(it)` → `conj_grad` → `loop_bottom_edt(it)` → either the next
`loop_top_edt` or `tail_edt`; a dedicated shutdown EDT gated on `tail_edt`'s
output event calls `ocrShutdown` only after the tail's dependence releases
have completed, so no release work is truncated out of the measured run.
The driver EDTs otherwise pass `NULL` for their output events; the only
inter-EDT edges here are direct DB dependences.

Inside `conj_grad` every edge is either an EDT output event or a `ONCE` event
satisfied by hand:

- `spmv_edt` takes the GUID container `CONST` and the input vector `CONST`,
  then creates one `assign_edt` with `nb` slots and `nb` `rowvec_edt`s, wiring
  each `rowvec`'s output event into one `assign` slot and handing it its own
  row-block DB (`CONST`) plus the **same** input vector (`CONST`).
- `assign_edt` concatenates the `nb` fragments of `8b` bytes into one `8n`
  block, destroys them, releases and satisfies the matvec's `q` event.
- `alphas_edt` returns `alpha` through its output event and satisfies a
  separate `ONCE` event with `-alpha`; `update_edt` returns the new `p` and
  satisfies a `ONCE` event with a fresh copy `pp`.
- `update_edt` takes `p` and `r` **twice each**, in two `RW` slots — the
  aliasing is deliberate (`r += -α·q` and `p = β·p + r` are computed in
  place), so one EDT holds the same GUID on two dependences.

DB concurrency: the matvec input vector is the fan-out point — up to **`nb`
simultaneous `CONST` readers** (560 at the calibrated args), refreshed by a
single `RW` writer one CG step earlier. Six long-lived blocks (`x`, `r`, `p`,
`z`, `rho`, and the timer) take `RW` from a different, round-robin-placed task
every CG step, so their write right migrates ~25 times per `conj_grad`; the
row-block DBs are `CONST`-only after `makea` and never written again. The one
genuine overlap is `scale_edt` reading `x` `CONST` while the first
`update_edt` of the same `conj_grad` holds `x` `RW` — both are enabled by the
same `alphas_edt`, and `update` never actually writes `x` (see notes).

## Flow

The DAG is a necklace. Per `conj_grad`: 26 beads, each a width-`nb` fork
followed by a width-1 join, separated by two to four single-EDT links doing
full-length `O(n)` vector work (`alphas`' dot product, `update`'s two axpys +
dot + copy, `daxpy`'s axpy). Outer iterations never overlap — `loop_bottom`
creates the next `loop_top` — so:

- **max concurrent EDTs = `nb = na/blk`**, and only during a matvec;
- **`26·C` fork-joins per run** (416 at the calibrated args), each ending in a
  single `assign_edt` that gathers `nb` fragments and copies `n` doubles;
- **no reduction tree anywhere**: every dot product is one EDT looping over
  all `n` elements.

That serial spine is the scaling limit: per CG step the parallel matvec is
`≈ 2·n·k²` flops while the serial vector chain plus the gather is `≈ 12·n`
flop-equivalents — a serial fraction of roughly `6/k²` (~4% at class A), an
Amdahl ceiling near 25× however many workers are added. `mainEdt` also runs
`makea` natively on the rank-0 worker: an `O(n·k²)` insertion-sorted fill,
serial, and the only phase that touches the matrix.

## Placement (base)

This application carries no `OCR_APP_OPTIMIZED_PLACEMENT` layer and has no
`_hinted` target: every `ocrEdtCreate` and `ocrDbCreate` in the tree passes
`NULL_HINT`. Effective policy is therefore **EDT → runtime round-robin**, **DB
→ home = creating rank**.

Everything built in `mainEdt` — class, timer, `x`, the GUID container and all
`nb` row-block DBs — is homed on rank 0; everything created inside an EDT
(`q`, `z`, `rho`, `pp`, the scalars, every result fragment) is homed wherever
that EDT landed. At multinode:

- A `rowvec_edt` for row block `e` lands on an arbitrary rank and the rank
  rotates between matvecs, so an **immutable 42 KiB block is re-fetched from
  rank 0 by a different reader almost every matvec** — 416 reads of the whole
  24 MB matrix, cheap only on arms that keep a durable reader copy.
- The matvec input vector is written on one rank and `CONST`-read by `nb`
  tasks on every rank: a 112 kB broadcast per matvec from a fresh source.
- The `nb` result fragments are created on scattered ranks and gathered by one
  `assign_edt` on a single rank — `nb` small remote acquires per matvec,
  ~233k over the run, all on the critical path.

The algorithm has obvious locality (a row block and its result belong
together; the matrix never changes) and the base program expresses none.

## Placement (no hinted tier — measured, not assumed)

A hint layer was written for this application and then **removed**: every
form of it measured slower than the base program, at every geometry and
both problem sizes tried.  Solve time at 8 bentley nodes, class A (base
17.9/18.0/18.0 s over three runs):

| layer | solve |
|-------|-------|
| band-homed blocks only | 20.3 |
| band-pinned readers only | 20.3 |
| both | 19.3 |
| spine pinned to rank 0 only | 25.3 |
| all three | 25.0 |

and at class A's 4× larger sibling (class B, 180 MB matrix, `-i 5`,
end-to-end): 2 nodes 90.2 base vs 93.3 hinted, 8 nodes **108.9 base vs
147.7 hinted**.

The reason is in what the data does, not in how the hints were written.
The only placement-sensitive object is the matrix, and it is **read-only
after `makea`** — under a validating protocol every rank ends up holding
its own snapshot after the first read, so pinning a block's reader to a
fixed rank buys a locality the base program already has, while paying to
move the block's home and to resolve an affinity per spawn.  Everything
that actually moves — the operand vector broadcast to every reader each
matvec, and the `nb` result fragments gathered into one EDT — is
all-to-all or all-to-one, which no home assignment improves.  Worse,
pinning the serial spine makes one rank the permanent server for that
broadcast and the permanent sink for that gather; the base program's
round-robin rotates the role and spreads it, which is why the spine pin
alone costs 40%.

So the answer to this program's placement problem is not a hint but the
decomposition: see `npb_cg_dist`, where the vector never travels whole
and the matrix band is owned, not fetched.

## Sizing

The restructured `npb_cg_dist` shares this program's generator and CLI but
replaces the decomposition; see its appdoc for the sizes it is calibrated
at, and for the four floors that survive the rewrite — the serial
generator above being the one both versions pay.

`-b` is the only dial that changes parallelism without changing the answer:
`nb = na/blk` is both the task count per matvec and the fan-out. `-t` sets the
problem (`na` and `nonzer` move work as `na·(nonzer+1)²`, and the class also
fixes `niter`), `-i` moves duration only.

- **Width**: aim for `nb ≳ 2×` total workers. Task grain is `b·k²`
  multiply-adds with an indirect gather — `25·144 ≈ 3.6k` at the calibrated
  args, a few microseconds, so per-task runtime overhead is a real fraction of
  the cost. Raising `b` buys grain and costs width, both linearly; the serial
  spine means width past ~25× the serial cost buys nothing.
- **Constraints**: `blk` must divide `na` exactly, and `nb` must stay within
  the runtime's dependence-count limit (65534 under the shim) — `assign_edt`'s
  `depc` *is* `nb`, so classes B and above cannot run at the default `-b 1`.
  Both are rejected with a message rather than run: the blocking in `mainEdt`,
  the fan-out at the failing create in `spmv_edt`.
- **Memory** is `≈ 12·na·(nonzer+1)²` bytes for the matrix and negligible for
  everything else: 24 MB at class A, ~180 MB at B, ~430 MB at C.
- **The calibrated `-b` is set by the widest geometry, not by the anchor's
  clock.** `-t A -b 2` gives 7000 tasks per matvec — and 7000 matrix-block
  datablocks, and 7000 result fragments per matvec, since `na/blk` is all
  three at once. That is 2.0× the 3456 workers of 32 Dane nodes × 108;
  `-b 25`'s 560 would leave five of every six workers idle there. The count
  is node-invariant, so smaller geometries pack more of it onto each node.
  Width stops at 2× rather than the fork-join rule's 4× because the two
  constraints cross: `-b 1` reaches 4.05× and takes 73.1 s at the anchor,
  and this is an anti-scaler, so that cell only grows with node count (its
  two-node run was still going at 430 s). At 2× no worker is idle — the
  rest is stealing headroom, not coverage.
- **This is an anti-scaler, and the calibration says so.** Measured (hinted,
  val_wb): class A `-b 25` runs 1.3 s at one bentley node, 15.6 s at two and
  27.3 s at eight — every added node costs time, because the operand
  broadcast, the single-EDT gather and the serial vector spine all grow with
  the rank count while the per-task grain shrinks. One class up the wall is
  absolute: class B `-b 10` finishes in 116 s at one node and **times out
  past 590 s at two**. So the class stays at A — `-t A -b 2`, 27.0 s on the
  Dane anchor node (108w+4p), inside the 10–30 s an anti-scaler is sized to
  — and the 32-node cell, not the 1-node one, is what the budget has to
  hold. (`-b 25` at the same node is 2.3 s: enough to time, not enough to
  fill 32 nodes.)
- `expect_args` equals `args` and the pin is the class's own reference zeta
  (17.1721077015265): the previous pin was taken at `-t T`, an argument set
  no campaign runs, so the cross-check never fired.
