# fft

*Recursive Cooley-Tukey FFT — one big datablock, sliced by offset/step at
every recursion level, never copied.*
Source: `third_party/ocr-apps/apps/fft/ocr/{fft.c,verify.c}` (~450 + ~150 lines).

## Overview

Computes the discrete Fourier transform of an `N`-point impulse signal
(`x[1]=1`, else 0) via decimation-in-time Cooley-Tukey: `fftStartEdt(N)`
recursively splits into two `N/2` sub-transforms (even/odd samples, addressed
by doubling `step` and shifting `offset` — never by copying data) until the
block size drops to `serialBlockSize`, at which point the base case runs a
purely serial (function-call, no-EDT) recursive `ditfft2` down to size 1. On
the way back up, one `fftEndEdt` per split level performs the radix-2
butterfly combine, itself farmed out to `fftEndSlaveEdt` slave tasks chunked
by `serialBlockSize` elements. The whole tree operates in place on a single
shared datablock; once it completes, `fftVerifyEdt` recomputes the same
transform with a fully serial reference and compares. The printed `FFT
checksum` (sum of `|Re|+|Im|` over all outputs) is the result scalar. Unlike a
pure task-churn probe, every leaf and slave EDT does real floating-point work,
so the app stresses fine-grain task/DB scheduling *and* per-task compute, with
one very large, heavily-shared datablock at its center. The catalog also
carries a held-back, uncommitted restructured twin (`fft_dist`, a Bailey
four-step transpose rewrite) not yet registered in the build.

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `power` | `N = 2^power`, the transform length | required — exactly one argument, else usage error + shutdown | ✓ parsed in `parseOptions`/`mainEdt`, carried in EDT paramv structs to every rank — multinode-safe |
| `serialBlockSize` | recursion cutoff: below this size `fftStartEdt` computes serially via `ditfft2` instead of spawning more EDTs | `SERIAL_BLOCK_SIZE_DEFAULT` = 1024·16 = 16384 | ✗ `#define` only, no CLI path — the knob a user would most want to size against worker count |
| `iterations` | intended repeat count | 1 (hardcoded; `parseOptions` never reads argv for it) | ✗ constant; the `!=1` warning branch is dead code |
| `verify` | run `fftVerifyEdt` | `true` (hardcoded) | ✗ always on |
| `verbose` | extra `ocrPrintf` tracing | `true` (hardcoded) | ✗ always on |
| `printResults` | dump the full input/output arrays | `false` (hardcoded) | ✗ always off |

## Structure

Let `d = log2(N) - log2(serialBlockSize) = power - 14` be the number of split
levels (`d = 0` when `N ≤ serialBlockSize`, i.e. `power ≤ 14`).

| object | count | size |
|--------|-------|------|
| `fftStartEdt` | `2^(d+1) - 1` (internal splits `2^d - 1`, leaves `2^d`; `= 1` when `d = 0`) | — |
| `fftEndEdt` | `2^d - 1` (one per internal split node) | — |
| `fftEndSlaveEdt` | `d · 2^(d-1)` (`= 0` when `d = 0`) | — |
| `mainEdt` / `fftIterationEdt` / `fftVerifyEdt` / `finalPrintEdt` | 1 each | — |
| EDTs total | `2 + 3·2^d + d·2^(d-1)` (`= 5` when `d = 0`) | — |
| Datablocks | **3, constant regardless of `N`**: 1 shared data block + 2 verify blocks | data block `12N` bytes; each verify block `4N` bytes |
| Events | `5·2^d - 1` (`= 4` when `d = 0`) — finish latches + idempotent output events per FINISH create, plus 1 for the verify EDT | — |
| EDT templates | 6 created (5 app + 1 verify), all destroyed | — |

Worked numbers: `power = 6` (expect_args; `N = 64 ≤ serialBlockSize`, `d = 0`):
5 EDTs, 3 DBs, 4 events. `power = 23` (calibrated args; `N = 8,388,608`,
`d = 9`): 3,842 EDTs, 3 DBs (~96 MiB data block + 2×32 MiB verify blocks ≈
160 MiB total), 2,559 events. Each `+1` on `power` roughly doubles the
exponential terms; DB *count* never changes.

Counter cross-check: verified (1 node, `power=6` vs `power=15`): measured absolutes EDT 6/10, DB 4/4, EVT 4/9; subtracting the runtime's constant baseline (+1 EDT, +1 DB, +0 EVT per run) gives app-side EDT 5/9, DB 3/3, EVT 4/9 — exactly the formulas above, both in absolute value and in delta.

## Wiring

Every `fftStartEdt`/`fftEndEdt`/`fftEndSlaveEdt` node in the *entire* tree
operates on the *same* one data datablock, sliced by `offset`/`step`/local-`N`
pointer arithmetic rather than fresh per-level blocks — there are never more
than 3 DBs live for the whole recursion. Every one of those nodes declares
`DB_MODE_RW` on it (create-time dependency arrays default to RW; explicit
`ocrAddDependence` calls also ask for RW), even though sibling subtrees only
ever touch disjoint offset ranges — the coherence layer serializes at
whole-DB granularity regardless. This single datablock is therefore the app's
sole and severe contention point: at the widest wavefront (the last split
level) up to `2^d` `fftStartEdt`/leaf instances contend for RW on it
simultaneously (512 at the calibrated `power = 23`). `fftIterationEdt` and
`finalPrintEdt` hold it RO/CONST instead. `fftVerifyEdt` reads the finished
block RO and owns two fresh verify blocks exclusively. End-to-end:
`fftIterationEdt` (FINISH-wraps the whole tree) → its output event triggers
`fftVerifyEdt` → its output event triggers `finalPrintEdt`, which destroys the
data block and shuts down.

## Flow

The tree unfolds top-down: each `fftStartEdt(N)` immediately creates its two
`N/2` children and (if not a leaf) an `fftEndEdt` gated on both children's
nested FINISH scopes plus the data block — so descent (splitting) and ascent
(combine) are two passes over the same balanced binary tree of depth `d`, the
ascent strictly following each subtree's completion. Parallel width during
descent is `2^L` at split level `L` (max `2^d` at the leaves); each leaf then
runs a fully serial `ditfft2` of size `serialBlockSize` with no further EDT
parallelism inside it. During ascent, each `fftEndEdt` at level `L` fans out
to up to `2^(d-L-1)` slaves. `mainEdt` itself is a short rank-0-only preamble
(allocate, zero, seed the impulse, create templates); there is no other
serial bottleneck once the tree is launched.

## Placement (as-born)

The source passes `NULL_HINT` on every EDT and DB create — there is no
`OCR_APP_OPTIMIZED_PLACEMENT` guard anywhere in `fft.c` (unlike the held-back
`fft_dist`, which carries one). Effective policy: **EDTs** → runtime
round-robin (per-creating-rank atomic counter, modulo rank count), so a
node's two children and its `fftEndEdt` scatter across arbitrary ranks;
**DBs** → home = creating rank, but since there is only ever one data block
(created once by `mainEdt` on rank 0) and two verify blocks, "creator" only
applies at those few points, not per-node. Consequence: nearly every
`fftStartEdt`/`fftEndEdt`/`fftEndSlaveEdt` runs on a rank other than the data
block's home, so nearly every RW acquire of that one 12N-byte block is a
remote, whole-DB migration — the app never expresses the locality that exists
in principle (each subtree only touches a disjoint slice).

## Sizing

`power` is the only reachable dial, and it moves task count and per-task
grain together via `d = power - 14`: raising it multiplies the split/combine
tree width by 2 per level. Because `serialBlockSize` is compile-time fixed at
16384, `power` below ~15 collapses to `d = 0` (a single leaf, no parallelism)
— useless for a multi-worker run. To get width comparably above worker count:
1 node × 15 workers wants `d ≥ 6` (`power ≥ 20`); 8 nodes × 120 workers wants
`d ≥ 10` (`power ≥ 24`). The calibrated `power = 23` (`d = 9`, width 512, ~96
MiB data block) sits just under that deliberately — since the single shared
datablock serializes access anyway, adding more width mostly adds more
waiters on the same contention point rather than more real concurrency, so
this app reads as a coherence/contention stress case, not a pure scaling
benchmark. Memory is dominated by the one data block (`12·2^power` bytes) —
at `power = 30` that alone is 12 GiB, worth checking against node memory
before raising `power` much past the calibrated value.
