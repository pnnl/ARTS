# RSBench_intel_sharedDB

*The opposite design from `RSBench_intel`: one contiguous datablock per array
across ALL nuclides (not one per nuclide), coarse per-thread-per-chunk EDTs
(not one triple per lookup), real policy-domain-affinity placement (not
`NULL_HINT` round-robin), and an actual verified `RS_CHECKSUM` (not a
completion marker only).*
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
(every Doppler-broadened pole evaluated) — accumulate per-thread and fold
through an `ARITY=10` reduction tree (`libs/src/reduction/reduction.c`) into
`RS_CHECKSUM: <abrarov> <alls>`. Both are pure functions of the fixed seed
(`srand(42)` for data generation, `42+1+tid` per thread for lookups) and of
associative integer addition, so the catalog pins `expect: 17079` at
`expect_args: -l 100` (nthreads=1, nprocs=1) as an actual correctness oracle
— unlike the plain port's marker-only check.

## Parameters

| flag | meaning | default | CLI reachability |
|------|---------|---------|-------------------|
| `-t <threads>` | persistent EDT lanes; each claims `CHUNK_SIZE=1000` lookups per generation | 1 | ✓ drives both intra-rank task fan-out and (via `SINGLE_RUN_ACROSS_PD`) cross-node placement — see Placement |
| `-p <procs>` | SPMD "rank" replication count — each rank independently runs the FULL `-l` lookups, not a partition | 1 | ✓ reachable, but **replicates** the run rather than partitioning it — see Sizing. Validated `≥ 1` (0 previously hung the SPMD fork, a negative value previously wrapped to a huge rank count) |
| `-l <lookups>` | XS lookups *per rank* | 10,000,000 | ✓ |
| `-s small\|large` | H-M size, `small` forces `n_nuclides=68` | large (355) | ✓ |
| `-n <n>` | nuclide count | 355 | ✓ but constrained: `load_num_nucs`/`load_mats` pick the H-M material tables by this count alone (exactly 68 → small tables, highest nuclide ID 67; anything else → large tables, highest ID 354), and every per-nuclide array is sized `n_nuclides`, so only `68` or `≥ 355` is legal. Other values are rejected at parse — they used to index `pseudo_K0RS`/`windows`/`poles` out of bounds in the lookup kernel |
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

Worked numbers at the calibrated `-l 7500000 -t 128` (`p=1`, `n=355` default):
`G=59` → base terms (excluding the reduction-tree fringe, negligible at this
scale) give **≈38.4K DBs**, **≈15.1K EDTs**, **≈22.7K events** — the event
figure is markedly higher than a `2tGp`-slope estimate would give, because
the per-(thread,generation) event count is 3, not 2 (see the table above).
Contrast the plain port's 1,200,795 EDTs at `-l 400000`: **three orders of
magnitude coarser task granularity** for ~19× the lookups. Shared-array
payload ≈26.7 MB per rank (same total bytes as the plain port's fragmented
1,065 DBs, now 13 DBs).

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
`summaryEdt` (RO on both reduction output events) prints `RS_CHECKSUM`,
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
calibrated `p=1` this places the sole rank on PD 0 and contributes no spread
by itself. Inside that rank, `getAffinityHintsForDBandEdt` snapshots
`ocrAffinityGetCurrent()` (wherever `initEdt` landed) into
`rankH_t.myEdtAffinityHNT`/`myDbkAffinityHNT`, and every same-rank object
thereafter — `channelSetupEdt`, `FNC_xsbenchMain`, `rankDataH_t`, and all 13
of `initSimulation`'s DBs — is explicitly pinned there (not `NULL_HINT`), so
the whole ~26.7 MB shared dataset is deliberately homed on one rank.

The one exception is the per-thread lane: `SINGLE_RUN_ACROSS_PD` is now
compiled in (`benchmarks/apps/CMakeLists.txt`'s `OCR_EXT_DEFINES` — historically
missing, which pinned every lane to rank 0 regardless of node count; fixed as
of this writing), so `lookUpKernelEdt` re-hints each `lookUpKernelPerThreadEdt`
(and everything it spawns) with `ocrAffinityGetAt(AFFINITY_PD,
getPolicyDomainID_Cart1D(tid, {t}, {affinityCount}), …)`, where
`affinityCount` is the **live** `ocrAffinityCount(AFFINITY_PD, …)` queried at
run time — every thread lane spreads round-robin across however many ARTS
ranks the run actually has, independent of `p`. So the calibrated `-l 7500000
-t 128` self-scales: the same binary spreads its 128 lanes over 1 node or 8
nodes without re-tuning `-t`, while the shared dataset stays resident on rank
0's node only. The resulting pattern is a genuine broadcast-once/
compute-everywhere shape: a lane on a non-owning node pays one remote RO
acquire per shared array it touches, then (protocol-dependent) reuses its
node-local cached copy for the rest of its `G` generations. The 5 ephemeral
per-generation "ptrs" DBs use `NULL_HINT`, so they are homed wherever the
compute EDT itself runs — always local, never remote.

## Sizing

`-t` and `-p` are independent levers that do **not** compose the way MPI
ranks × OpenMP threads would:

- **`-t` (nthreads)** is both the intra-rank task-fan-out axis AND (via
  `SINGLE_RUN_ACROSS_PD`) the cross-node placement axis — set it to roughly
  the total worker count across the whole job (e.g. `-t 120`–`128` for an
  8-node×15-worker run) so one lane lands near each worker; `-l` then sets how
  many generations (`G=⌈L/(1000t)⌉`) each lane works through.
- **`-p` (nprocs)** replicates the entire `-l`-lookup run once per rank rather
  than partitioning it — each rank regenerates the identical fixed-seed
  dataset and independently runs the full lookup count. Raising `-p`
  multiplies total work and memory (`p`× the ~26.7 MB dataset, one full copy
  per rank) without changing any single rank's task grain — it is a "run `p`
  independent replicas, one per PD" knob, not a strong-scaling knob. Leave it
  at the default `1` unless independent-replica behavior is actually wanted.
- **1 node × 15 workers**: `p=1`, `-t` ≈15–30 (a few lanes per worker), `-l`
  large enough that `G` gives each lane several generations — e.g. `-l 500000
  -t 16` (`G≈32`) — for a run of tens of seconds.
- **8 nodes × 120 workers**: `p=1`, `-t 120`–`128` so `SINGLE_RUN_ACROSS_PD`
  spreads one lane per worker across all 8 nodes; the calibrated `-l 7500000
  -t 128` (`G=59`) is sized for a multi-minute 1-node run and self-spreads
  unchanged at 8 nodes since placement re-queries `affinityCount` at run time.
- Memory is dominated by the ~26.7 MB shared dataset (`n_nuclides=355`,
  `avg_n_poles=1000` by default, now settable with `-a`) resident once per
  rank; `-n`/`-s`/`-w`/`-a` are the only levers that move it, and none of them
  scale with node count in this port. `DBK_poles`/`DBK_windows` are allocated
  at exactly `n_nuclides·avg_n_poles` / `n_nuclides·avg_n_windows` slots, while
  the per-nuclide counts are a random multinomial draw with a "bump any zero
  bin up to 1" floor; both now sum the actual counts first and reject a run
  whose distribution would overflow the allocation (reachable at very small
  `-a`/`-w`, not at the sizes above).
