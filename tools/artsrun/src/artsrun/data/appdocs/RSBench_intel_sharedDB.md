# RSBench_intel_sharedDB

*The opposite design from `RSBench_intel`: one contiguous datablock per array
across ALL nuclides (not one per nuclide), coarse per-thread-per-chunk EDTs
(not one triple per lookup), and real policy-domain-affinity placement (not
`NULL_HINT` round-robin). The pinned scalar is the workload echo — the app's
computed counters travel through its reduction-tree library, whose delivery
is not deterministic under this runtime (see Overview).*
Source:
`third_party/ocr-apps/apps/RSBench/refactored/ocr/intel-sharedDB/src/` (7 C
files + `config.tpl`, ~1.1k lines; `main.c` runs the SPMD-fork / per-thread /
reduction machinery, `init.c` / `material.c` generate the shared
cross-section data, `rs_kernel.c` is the per-lookup compute — identical
physics to the plain port).

## Overview

Same RSBench multipole-lookup proxy as `RSBench_intel` (see its doc for the
kernel), restructured to track the MPI+OpenMP reference more literally.
`-p` SPMD "ranks" — a real ARTS-rank fork via policy-domain affinity, see
Placement — each independently regenerate the *same* fixed-seed dataset into
ONE contiguous datablock per array (`DBK_poles`, `DBK_windows`,
`DBK_pseudo_K0RS`, `DBK_mats_all`, `DBK_concs_all`), not one DB per nuclide.
`-t` OpenMP-style "threads" each become a persistent EDT lane that claims a
`CHUNK_SIZE=1000`-lookup slice at a time and computes all of it inside one
EDT body's C loop before creating its own successor lane. Two counters —
`abrarov` (how often the slow Abrarov/Faddeeva path is taken) and `alls`
(every Doppler-broadened pole evaluated) — accumulate per-lane and fold
through an `ARITY=10` reduction tree (`libs/src/reduction/reduction.c`).
The counters themselves are pure functions of the fixed seeds (`srand(42)`
for data generation, `42+1+tid` per lane for lookups), but their **delivery
is not deterministic on this runtime**: the reduction library's rendezvous
rides labeled-GUID event creation, and this runtime's labeled semantics is
create-replaces (racing creators install over each other), so the folded
sums come out different on every run — a family sweep printed 17 distinct
values in 17 cells on identical inputs, and the per-lane accumulation also
re-adds its running prefix each lookup (`*g += abrarov` inside the loop),
so even a faithful delivery would not equal the true count. The catalog
therefore pins the **workload echo** (`Lookups:` from the results banner,
`expect: 1000000`) — the same completion-level convention as
`XSBench_intel_sharedDB` — and the verified data-side oracle lives in the
plain port's pole checksum instead. The same caveat applies to the results
banner's `Runtime:` line (an `F8_MAX` allreduce over instances through the
same library): treat it as a lower bound, and use the runtime's own `[E2E]`
stamp for measurement.

## Parameters

| flag | meaning | default | CLI reachability |
|------|---------|---------|-------------------|
| `-t <threads>` | persistent EDT lanes; each claims `CHUNK_SIZE=1000` lookups per generation | 1 | ✓ drives both intra-rank task fan-out and (via `SINGLE_RUN_ACROSS_PD`) cross-node placement — see Placement |
| `-p <procs>` | SPMD "rank" replication count — each rank independently runs the FULL `-l` lookups, not a partition | 1 | ✓ reachable, but **replicates** the run rather than partitioning it — see Sizing. Validated `≥ 1` (0 previously hung the SPMD fork, a negative value previously wrapped to a huge rank count) |
| `-l <lookups>` | XS lookups *per rank* | 10,000,000 | ✓ |
| `-s small\|large` | H-M size, `small` forces `n_nuclides=68` | large (355) | ✓ |
| `-n <n>` | nuclide count | 355 | ✓ parsed (only `≥1` checked) — ⚠ constrained in practice: `load_num_nucs`/`load_mats` pick the H-M material tables by this count alone (exactly 68 → small tables, highest nuclide ID 67; anything else → large tables, highest ID 354), and every per-nuclide array is sized `n_nuclides`, so only `68` or `≥ 355` avoids out-of-bounds indexing in the lookup kernel; the campaign only ever reaches this through `-s` |
| `-a <poles>` | average poles per nuclide | 1000 | ✓ own flag, distinct from `-p` (was previously a second, unreachable `-p` branch shadowed by the nprocs match — see notes) |
| `-w <windows>` | average windows per nuclide | 100 | ✓ |
| `-d` | disable Doppler broadening | ON | ✓ |
| *(none)* `n_mats` | 12 material zones, same fixed H-M table as the plain port | 12 | ✗ compile-time only |
| *(none)* `numL` | Legendre moments per nuclide | 4 | ✗ compile-time only |

## Structure

Let `t` = `nthreads`, `p` = `nprocs`, `L` = `lookups`, and
`G = ⌈L / (1000·t)⌉` (`CHUNK_SIZE=1000`, `SCHEDULER_TYPE=1` dynamic-scheduling
equivalent — both compile-time constants). `n_nuclides` does **not** appear in
any of these counts: that is the structural point of "sharedDB" — the whole
cross-section dataset is a handful of contiguous arrays regardless of `n`.

| object | count (×`p`) | size |
|--------|------|------|
| global DBs (`argv`, `globalParamH`) | 2 | small |
| per-rank setup DBs (`rankH`, `rankDataH`, `rpPerfTimerDBK`) | 3 | small |
| per-rank shared-array DBs (`n_poles`, `n_windows`, `num_nucs`, `mat_ptrs`+`mats_all`, `conc_ptrs`+`concs_all`, `pole_ptrs`+`DBK_poles`, `window_ptrs`+`DBK_windows`, `K0RS_ptrs`+`DBK_pseudo_K0RS`) | 13 | `DBK_poles` ≈`n·avg_n_poles·72 B`; `DBK_windows` ≈`n·avg_n_windows·32 B`; rest small |
| per-thread persistent DBs (seed, xs, sigTfactors, reductionVars, loop-reduction) | `5t` | tens of bytes each |
| per-(thread,generation) ephemeral "ptrs" DBs, created **and destroyed** inside the same `iterationsPerThreadEdt` | `5tG` | tens of bytes each |
| reduction-tree fringe (loop-completion ×`t` + perf-timer ×1 launches, plus their `ARITY=10` fan-in machinery — `reductionLaunch`/`reductionSendChannelEdt` in `libs/src/reduction/reduction.c`) | **+5 DBs**, measured at the verified `t=2, p=1` point (`t+1=3` launches); not closed-form for general `t` — see notes | small |
| EDTs: global + per-rank setup | `2 + 7p` | — |
| EDTs: per (thread, generation) — `lookUpKernelPerThreadEdt` + `iterationsPerThreadEdt` | `2tGp` | — |
| EDTs: reduction-tree fringe | **+8 EDTs**, measured at `t=2, p=1`; not closed-form — see notes | — |
| STICKY/COUNTED events: global + per-rank | `1 + 3p` | — |
| events: per (thread, generation) — `iterationsPerThreadEdt` create (`main.c:408-409`, `EDT_PROP_FINISH` + non-NULL `outputEvent`, so 2 shim events) **plus** `createEventHelper(&iterationsPerThreadOEVTS, 1)` (`main.c:411`, an explicit `OCR_EVENT_COUNTED_T` create) — 3 events per (thread,generation), not 2 | `3tGp` | — |
| events: reduction-tree fringe | **+6 events**, measured at `t=2, p=1`; not closed-form — see notes | — |

At `p=1` the totals are: `DB = (2+16+5t) + 5tG + 5`, `EDT = (2+7) + 2tG + 8`,
`EVENT = (1+3) + 3tG + 6` — the fixed reduction-tree fringe terms (`+5`/`+8`/`+6`)
are pinned by measurement at `t=2` (see Counter cross-check below) but their
general-`t` scaling is not statically closed-formed (the `ARITY=10` tree's
depth grows with `t`; see notes).

Worked numbers at the calibrated `-l 1000000 -t 108 -p 32` (`n=355`
default): per instance `G=⌈10⁶/(1000·108)⌉=10` → base terms (excluding the
reduction-tree fringe, negligible at this scale) give per instance ≈576 DBs
(`18+5t+5tG`), ≈2.2K EDTs (`9+2tG`), ≈3.2K events (`4+3tG`) — ×32 instances
≈**18.4K DBs**, **70K EDTs**, **104K events** for the whole run. Contrast
the plain port's 450,101 EDTs at `-l 150000`: **~30× coarser task
granularity** for ~213× the lookups (32M aggregate). Shared-array payload
≈26.7 MB **per instance** (same total bytes as the plain port's fragmented
1,065 DBs, now 13 DBs), ≈854 MB across the 32 instances.

Counter cross-check: verified (1 node, `-s small -t 2 -l 30` vs `-l 2500`,
both `p=1`, `G=1` vs `G=2`): measured absolutes EDT 22/26, DB 44/54, EVT
16/22; subtracting the runtime's constant baseline (+1 EDT, +1 DB, +0 EVT per
run) gives app-side EDT 21/25, DB 43/53, EVT 16/22 — exactly the formulas
above (`t=2, p=1`: `DB=(2+16+10)+5·2·G+5=33+10G`, `EDT=9+4G+8=17+4G`,
`EVENT=4+6G+6=10+6G`, evaluated at `G=1` and `G=2`).

## Wiring

`mainEdt → forkSpmdEdts_Cart1D → initEdt (×p) → channelSetupEdt →
FNC_xsbenchMain → FNC_initSimulation (initOcrObjects + initSimulation: fills
the 13 shared-array DBs, RW then released) → lookUpKernelEdt` (creates the
`5t` per-thread persistent DBs, spawns `t` independent lanes) `→
lookUpKernelPerThreadEdt` chain, one link per generation per lane; each link
spawns one `iterationsPerThreadEdt` (FINISH) that RO-acquires the rank's
shared arrays plus its own 5 fresh "ptrs" DBs (created and destroyed inside
the same body) and runs the `CHUNK_SIZE`-lookup C loop. On a lane's last
generation it calls `reductionLaunch` into the loop-completion tree and
decrements `loopCompletionLatchEVT` (a LATCH counting down from `t`) →
`launchReductionEdt` (waits on the latch) launches the perf-timer reduction →
`summaryEdt` (RO on both reduction output events) prints the results banner,
destroys the shared arrays, satisfies `finalOnceEVT` → `wrapUpEdt` shuts down.

DB concurrency: the 8 shared-array DBs are written exactly once (during
`initSimulation`, released before any lookup runs) and RO thereafter — no
writer ever returns. Once released, up to **`t`** (128 at calibrated size)
`iterationsPerThreadEdt`s run concurrently — the lanes have no inter-lane
dependency — and can hold simultaneous RO acquires of the same rank's copy.
**`DBK_poles` (≈24.4 MB at default sizing: `355×1000×72 B`) is the contention
point** — a single DB instead of the plain port's 355 small ones. The 5
per-generation "ptrs" DBs are private scratch, never shared across EDTs.

## Flow

Setup is a strict FINISH-scoped serial pipeline per rank (`initEdt →
channelSetupEdt → xsbenchMain → initSimulation`, one worker). The lookup phase
is `t` independent generation-chains running fully in parallel — steady-state
concurrency is close to `t` (a lane's `lookUpKernelPerThreadEdt` and its child
`iterationsPerThreadEdt` overlap only briefly at each generation hand-off) —
each lane working through `G` **sequential** generations of up to 1000
lookups apiece; a lane never runs ahead of its own generation counter, but the
`t` lanes have no cross-lane ordering until the final latch.
`partition_bounds` splits each generation's `1000·t` lookups evenly across
lanes, so the lanes are balanced by construction and wall time tracks
`G`×(per-generation compute). The reduction join
(loop-completion latch → `launchReductionEdt` → `summaryEdt`) is the one
whole-rank barrier, at the very end.

## Placement (base)

`forkSpmdEdts_Cart1D` hints each of the `p` SPMD rank-EDTs (`initEdt`) with a
genuine `OCR_HINT_EDT_AFFINITY` from `ocrAffinityGetAt(AFFINITY_PD,
getPolicyDomainID_Cart1D(i, {p}, {affinityCount}), …)` — a real base
(not `OCR_APP_OPTIMIZED_PLACEMENT`-gated; that guard does not exist in this
port at all) cart-1D spread of SPMD ranks across policy domains. At the
calibrated `p=32` the fork's block partition maps 32 instances onto however
many PDs the run has (8n → 4 per node; 1n → all 32 on the node). Inside
each instance, `getAffinityHintsForDBandEdt` snapshots
`ocrAffinityGetCurrent()` (wherever `initEdt` landed) into
`rankH_t.myEdtAffinityHNT`/`myDbkAffinityHNT`, and every same-rank object
thereafter — `channelSetupEdt`, `FNC_xsbenchMain`, `rankDataH_t`, and all 13
of `initSimulation`'s DBs — is explicitly pinned there (not `NULL_HINT`), so
each instance's ~26.7 MB shared dataset is deliberately homed on the rank
its fork slot chose.

The one exception is the per-thread lane: with `SINGLE_RUN_ACROSS_PD`
compiled in (`benchmarks/apps/CMakeLists.txt`'s `OCR_EXT_DEFINES`),
`lookUpKernelEdt` re-hints each `lookUpKernelPerThreadEdt` (and everything it
spawns) with `ocrAffinityGetAt(AFFINITY_PD,
getPolicyDomainID_Cart1D(tid, {t}, {affinityCount}), …)`, where
`affinityCount` is the **live** `ocrAffinityCount(AFFINITY_PD, …)` queried
at run time — so EVERY instance spreads its `t` lanes across ALL the run's
ranks, independent of where the instance's own dataset lives. At the
calibrated `-p 32 -t 108` on 8 nodes this makes a deliberate broadcast
texture: an instance's lanes run machine-wide and ~7/8 of them
remote-RO-acquire that instance's arrays from its home rank; every node's
cache ends up holding copies of many instances' datasets (protocol-dependent
reuse across a lane's `G` generations). At 1n the Cart1D collapses and
everything is local. The 5 ephemeral per-generation "ptrs" DBs use
`NULL_HINT`, so they are homed wherever the compute EDT itself runs — always
local, never remote. There is nothing left for a hint layer to add — every
create already carries an explicit affinity or is deliberately creator-local —
so this app has no `hinted` flavor.

## Sizing

The campaign fixes `-l 1000000 -t 108 -p 32` at every node count — the
strict sweep invariant (every logical count node-invariant, same convention
as `XSBench_intel_sharedDB`):

- **`-p 32`** is the SPMD instance count, 1× the largest campaign geometry
  (32 nodes); the fork's block partition maps the 32 instances onto however
  many PDs the run has, so smaller runs pack more instances per node instead
  of changing any count. Each instance regenerates the identical fixed-seed
  dataset and runs its own full `-l` — total work is a constant 32M-lookup
  aggregate.
- **`-t 108`** lanes per instance, 1× a campaign node's persistent workers
  (the SPMD width rule); `G=⌈10⁶/(1000·108)⌉=10` generations per lane.
- **`-l 1000000`** per instance sizes a ~2-minute 1-node ceiling at 15
  workers (~260K lookups/s end-to-end there — two orders below XSBench's
  rate, the multipole kernel being that much heavier) and shrinks toward
  ~17 s at 8 nodes.
- Memory is dominated by the ~26.7 MB shared dataset (`n_nuclides=355`,
  `avg_n_poles=1000` by default, settable with `-a`) resident once per
  instance — ≈854 MB total at `-p 32`, plus protocol-dependent per-node
  cached copies from the lane spread; `-n`/`-s`/`-w`/`-a` are the only levers
  that move it. `DBK_poles`/`DBK_windows` are allocated at exactly
  `n_nuclides·avg_n_poles` / `n_nuclides·avg_n_windows` slots, while the
  per-nuclide counts are a random multinomial draw with a "bump any zero bin
  up to 1" floor — the draw conserves the total, so the allocation can only
  overflow when the zero-bin floor injects extras, i.e. at very small
  `-a`/`-w`, unreachably far from the sizes above.

## Family shape (measured, 15w+1p × 1/2/4/8 nodes, `-l 1000000 -t 108 -p 32`)

base (this app has no hinted flavor), e2e seconds. All cells completed;
the counter sums differed per cell as described in Overview, which is why
the pin is the workload echo:

| arm | 1n | 2n | 4n | 8n |
|---|---|---|---|---|
| val_wb | 121.6 | 76.1 | 46.2 | 22.2 |
| val_wb_comb | 124.8 | 62.7 | 34.2 | 16.9 |
| inv_wb | 116.4 | 63.0 | 33.6 | 17.1 |
| excl_retain | 116.2 | 65.8 | 33.5 | 16.3 |

A Dane-geometry single node (108w+4p) runs it in 74.6 s. Unlike its
`XSBench_intel_sharedDB` sibling (arm-indifferent to <1%), the arms separate
mildly here at 2–4n (val_wb trails the pack by up to ~1.2×): the
`SINGLE_RUN_ACROSS_PD` lane spread makes every instance's lanes
remote-acquire its arrays machine-wide, so the read path is exercised
cross-node even though the datasets are block-homed. All four arms scale
5.5–7.1× from 1n to 8n — the ~60 s serial-init floor (32 instances
regenerating datasets) parallelizes with nodes along with the kernel. The
banner's `Runtime:` line is not used for any of this (see Overview); the
numbers above are the runtime's `[E2E]` stamp.
