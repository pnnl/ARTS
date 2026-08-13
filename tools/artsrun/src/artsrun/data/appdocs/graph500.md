# graph500

*A 2D-decomposed, level-synchronized BFS over a randomly generated graph —
Graph500-style Kernel 1 (graph construction) + Kernel 2 (single-root search).*
Source: `third_party/ocr-apps/apps/graph500/graph500.c` (~1975 lines).

## Overview

Builds a graph of `SIZE = 2^SCALE` vertices and `EDGE_SIZE = SIZE·EDGEFACTOR`
random undirected edges, partitions vertices/edges across an `R×C` logical
worker grid, then runs a single level-synchronized BFS from a fixed root,
timing the search phase and printing `nodes`/`edges`/`MTEPS` (edges per
second) as the result. The harness build (`EXTRA_DEFINES NO_FILES NO_MAP` in
`benchmarks/apps/CMakeLists.txt:520`) compiles the *in-EDT, PRNG-generated*
graph path and the *pre-created EDT lattice* addressing path — the file-I/O
graph generator and the labeled-GUID addressing scheme both exist in the
source but are dead code in this build (see Findings). Edges are drawn
`source = xorshift64star(seed) % SIZE`, `destination = xorshift64star(seed) %
SIZE` — a uniform-random (Erdős–Rényi-style) generator, not Graph500's
Kronecker/RMAT model. The program stresses task-creation *volume* (tens of
thousands of EDTs from a single-threaded setup phase) and cross-worker
*data movement* (the row-then-column BFS-frontier scatter each level); the
per-EDT compute is light (array scans, a handful of comparisons).

## Parameters

| arg | meaning | default | CLI reachability |
|-----|---------|---------|-------------------|
| `argv[1]` = `SCALE` | `SIZE = 2^SCALE` vertices | required (usage error if `argc<5`) | ✓ parsed via `ocrGetArgv` in `mainEdt`, packed into `paramDBK` (`DB_MODE_CONST`), read by every EDT kind — multinode-safe |
| `argv[2]` = `EDGEFACTOR` | `EDGE_SIZE = SIZE·EDGEFACTOR`; avg vertex degree ≈ `2·EDGEFACTOR` | required | ✓ same path as `SCALE` |
| `argv[3]` = `R` | worker-grid rows; `SIZE` must be divisible by `R` | required | ✓ same path; also fixes `W = R·C` |
| `argv[4]` = `C` | worker-grid columns; `SIZE` must be divisible by `C` and by `R·C` | required | ✓ same path |
| `ROOT` (BFS root vertex) | fixed | `4` | ✗ compile-time constant (`graph500.c:1758`), never read from argv |
| `NUMBER_OF_SEARCH` | # of BFS searches (Kernel‑2 repeats) | `1` | ✗ hardcoded (`graph500.c:1697`); the `NO_MAP` build additionally `assert(0)`s if a second search is ever chained (`graph500.c:1624`) — the multi-search wiring (`startMap`, per-search finish slots) is unreachable dead code in this build |
| `SEED` (graph PRNG seed) | fixed | `123456789` | ✗ hardcoded (`graph500.c:1825`); genuinely deterministic in this build (see Findings) |
| `MAX_LEVEL` (BFS depth capacity) | fixed | `10` | ✗ compile-time (`graph500.c:101`); a graph whose BFS reaches it aborts with a message naming the cap |
| `NO_MAP`, `NO_FILES` | addressing scheme / graph-source switches | both defined | build-fixed by `benchmarks/apps/CMakeLists.txt:520`, not app CLI |
| `NO_AFFINITIES` | disables the app's own EDT affinity hints | not defined (hints ON) | compile-time only |
| `VALIDATION_MODE` (0–3) | depth of result checking | `0` (lightweight vertex-count sum, no full-graph reconstruction) | compile-time only |

`PARAMDBK_MODE`/`ARRAYDBK_MODE` (both `DB_MODE_CONST`, i.e. RO) and
`EDGE_MODE_LIST` (edge-list vs. matrix storage — only the list mode is
implemented) are further compile-time knobs with no CLI surface.

## Structure

Let `W = R·C` (logical worker count) and `K` = the number of BFS levels for
which `distribute`/`search`/`apply` actually run — equivalently, the number
of `<level>: <SUM_COUNT>` lines `applyEdt` prints on worker 0
(`graph500.c:1317`), counted from level 0 through the first level whose
`SUM_COUNT` prints `0` (the level-synchronized grid still runs a full
distribute→search→apply pass for that terminal, all-empty level before the
grid stops; `1 ≤ K ≤ MAX_LEVEL=10`, data-dependent — see below).
`sizeof(vertexType) = sizeof(ocrGuid_t) = 8 B`, `sizeof(edge) = 16 B`,
`sizeof(vInfo) = 16 B`, `sizeof(evalData) = 56 B`.

| object | count | notes |
|--------|-------|-------|
| EDT templates | 10, each created+destroyed exactly once | distribute/search/apply/load/finish/stop/shutdown/start/create/finalShutdown |
| EDTs total | **`32W + 6`** | `30W`: `mainEdt` pre-creates distribute+search+apply for *all* `MAX_LEVEL=10` levels × `W` workers up front, regardless of the graph's true BFS depth; `+W` `createEdt`, `+W` `loadEdt`; `+6` singletons — `mainEdt` itself plus start/finish/stop/shutdown/finalShutdown. `mainEdt`'s own creation is a genuine `NUM_EDT_CREATE` (the OCR shim's runtime-created `main_edt` builds the argv DB, then `arts_edt_create`s the `mainEdtTrampoline` that runs this app's `mainEdt`), distinct from the fixed runtime baseline of `main_edt` + the argv DB (`+1 EDT`/`+1 DB` on every app, not counted here) |
| Events | **`W + 14`** | `W` per-worker STICKY `dataEVT` + 10 pre-created per-level LATCH `nextEVT` + 1 ONCE `startEVT` + 1 LATCH `loadEndEVT`, plus 2 shim-materialized output events (`stopEdt`'s and `shutDownEdt`'s non-NULL `outputEvent`); no `EDT_PROP_FINISH` EDTs exist in this app |
| DBs | **`8 + 6W + K·W·(R+5)`** | 5 one-time setup — `paramDBK`/`constguidDBK`/`timeDBK` plus `mainEdt`'s own `LOCAL_VAR_ARRAY` `affinities`/`hints` — + 2 one-time `startEdt` `LOCAL_VAR_ARRAY` `affinities`/`hints` + 1 one-time `eTimeDBK` (`stopEdt`) + `5W` per-worker setup (`createEdt`'s 4 + `loadEdt`'s `arrayDBK`) + `W` `cVisitedDBK` (terminal level only) + `K` executed levels × `W` workers × [2 in `applyEdt` (`toRunDBK` + `LOCAL_VAR_ARRAY` `toRunBool`) + (`R`+3) in `searchEdt` (`R` `toRunxDBK` + `LOCAL_VAR_ARRAY` `destVertices`/`counts`/`positions`)] |

**`LOCAL_VAR_ARRAY` is a hidden `ocrDbCreate`.** The app's own
`LOCAL_VAR_ARRAY(TYPE,NAME,SIZE)` macro (`graph500.c:67`) expands to an
`ocrDbCreate` (plus a `_DBK` guid) whenever `USING_DATABLOCKS` is defined —
which it unconditionally is (`graph500.c:47`) — so every "local array" the
source declares through it is actually a datablock, invisible to a reading
that only greps for literal `ocrDbCreate` call sites. Eight of its ten use
sites are live in this build: `mainEdt`'s and `startEdt`'s
`affinities`/`hints` (`graph500.c:1808,1810` and `870,872` — one-time each,
sized to the rank/PD count, not scaled by `W`/`K`), and, on the per-level
hot path, `searchEdt`'s `destVertices`/`counts`/`positions`
(`graph500.c:1062,1064,1065`, unconditional — one triple per `searchEdt`
call) and `applyEdt`'s `toRunBool` (`graph500.c:1207`, inside the
`TORUN_DESTROY_MODE` branch — one per `applyEdt` call). `TORUN_DESTROY_MODE`
is `#define`d unconditionally in the source (`graph500.c:189`) — not gated
by the harness's `EXTRA_DEFINES NO_FILES NO_MAP` — so that branch, and its
per-invocation `toRunDBK` create, is the one that actually runs;
`createEdt`'s `toRun0DBK`/`toRun1DBK` are correspondingly the tiny (2- and
1-`vertexType`) `TORUN_DESTROY_MODE`-sized placeholders, not the
`vSIZE`-sized buffers the other branch would size them as. The remaining two
sites (`applyEdt`'s non-`TORUN_DESTROY_MODE` `toRunBool` at
`graph500.c:1274`; `finishEdt`'s `visited` at `graph500.c:1462`, gated on
`VALIDATION_MODE>=2`) are dead in this build.

**EDT and Event totals are exact, closed-form and data-independent** — the
`NO_MAP` build eagerly instantiates the full `MAX_LEVEL`-deep EDT lattice, so
the counts don't depend on how far the BFS actually gets (levels beyond the
true depth simply never receive their dependences and never fire). **DB
totals are not** — they scale with `K`, which is a property of the generated
graph, not a formula of the args. `K` *is* exactly reproducible run-to-run
(the generator is a fixed-seed PRNG, no wall-clock/PID entropy — see
Findings), but it has no closed form; `K` must be read from the run's own
frontier prints (the `<level>: <SUM_COUNT>` stdout lines above), then
treated as "measure once, same every time" for that argument set.

Worked numbers at the catalog's calibrated args (`20 8 32 16` → `SIZE =
1,048,576`, `EDGE_SIZE = 8,388,608`, `W = 512`): EDTs = `32·512+6 =
16,390`; Events = `512+14 = 526`; DBs = `8 + 6·512 + K·512·37 = 3,080 +
18,944·K`, where `K` is read directly from that run's own frontier prints
(random-graph diameter theory suggests `K` on the order of 5–8 for this
size/density, but that is only a pre-run estimate — the exact value comes
from the run's own output, not a formula). Dominant persistent memory is the
per-worker edge lists (`arrayDBK`, total ≈ `32·EDGE_SIZE + 8W` B ≈ 256 MiB)
and visited arrays (`visitedDBK`, total ≈ `16·SIZE` B ≈ 16 MiB — independent
of `W`, since `W·vSIZE = SIZE` always); `constguidDBK` is `(27+31W)·8` B ≈
124 KiB. Per-level `toRun`/`toRunx`/`toRunBool`/`destVertices`/`counts`/
`positions` traffic is transient (created and destroyed each level), bounded
by O(`SIZE`) live data per level.

Counter cross-check: verified (1 node, args `6 8 1 1` (`W=1,R=1`) vs
`6 8 2 2` (`W=4,R=2`); same seed ⇒ same generated graph ⇒ `K=5` executed
levels in both runs, per the run's own frontier prints `0:1, 1:8, 2:51, 3:4,
4:0`). Raw counters: NUM_EDT_CREATE = 39/135, NUM_EVENT_CREATE = 15/18,
NUM_DB_CREATE = 45/173. Subtracting the runtime's constant baseline (+1
EDT, +1 DB, +0 EVT per run — `main_edt` plus the argv DB, common to every
app) gives the app-only totals `32W+6` = 38/134, `W+14` = 15/18, and
`8+6W+K·W·(R+5)` at `K=5` = 44/172 — all three formulas exact at both
points, constants and `K`/`R` coefficients included, not just the deltas
(ΔEDT=96, ΔEVT=3, ΔDB=128).

## Wiring

Setup (`mainEdt`, rank 0) creates 3 globally-shared DBs — `paramDBK` (56 B,
the 4 CLI args + derived sizes + `SEED`), `constguidDBK` (template/GUID
lookup table, `(27+31W)·8` B), `timeDBK` (kernel-1 timer) — all `DB_MODE_CONST`
(RO), read by essentially every EDT the run creates; this is the run's real
**contention point**, not by write conflict (RO has no exclusive lock) but
by request volume against a single rank-0 home. Each worker `w`'s `loadEdt`
builds its own edge-list `arrayDBK` (RO, `DB_MODE_CONST`) and feeds it
through a per-worker STICKY `dataEVT`; `createEdt` builds a per-worker
`visitedDBK` (`vSIZE` `vInfo`s) that stays `DB_MODE_EW` (ARTS RW) for the
worker's *entire* run, threaded serially level-to-level through that
worker's own `applyEdt` chain — never touched by any other worker, so this
RW traffic never crosses ranks. The actual cross-worker channel is the BFS
frontier: each level, `distributeEdt(w)` row-scatters its frontier DB (RO)
to the `C` `searchEdt`s in its row; each `searchEdt(w)` then column-scatters
`R` freshly-built `toRunxDBK`s (RO) to the `R` `applyEdt`s in its column.
This two-hop row-then-column pattern is O(`W·(R+C)`) fan-out per level, not
the O(`W²`) of a full all-to-all. `applyEdt` folds its column's `R` inputs
into a new frontier (`toRunDBK`) and either spawns level `K+1`'s
distribute/search/apply triple (already pre-created — just wires slots) or,
once its frontier is empty, wires its `visitedDBK` summary and `toRunGuidDBK`
into the single `finishEdt` and a `NULL`-mode signal into the single
`stopEdt`.

## Flow

`mainEdt` (rank 0, single-threaded): parses argv, computes `SIZE`/`EDGE_SIZE`,
creates the 10 templates, then issues the entire `32W`-EDT creation burst
(`createEdt`×`W`, `loadEdt`×`W`, the `3·MAX_LEVEL·W` pre-created
distribute/search/apply triple) plus `W+13` event creates — an O(`W`) serial
preamble before any parallel work exists, independent of graph size. `W`
`loadEdt`s then run in parallel (round-robin across ranks), each
*redundantly* replaying the full `EDGE_SIZE`-long PRNG edge stream twice
(count + fill) to extract its own share — O(`W·EDGE_SIZE`) total work for an
O(`EDGE_SIZE`) graph. A `loadEndEVT` (W-way LATCH) barriers Kernel 1 against
Kernel 2's `startEdt`. The BFS proper is **level-synchronized**: each level
is a 3-stage pipeline (distribute → search → apply) of width `O(W)` per
stage; a level cannot start until the previous level's `applyEdt`s have all
fired (no cross-level pipelining). Parallel width per level ≈ `3W`
(distribute+search+apply concurrently in flight for that level, bounded by
the runtime's steal-half scheduling — not all `W` instances of a stage need
be live simultaneously). The run ends in two rank-0 global joins: `finishEdt`
(`depc = 2W+3` — every worker's final visited-count and toRun-guid DB) and
`stopEdt` (`depc = W+1` — every worker's termination signal plus the kernel-2
timer), both single EDTs that cannot fire until all `W` workers have reached
their terminal level simultaneously (the level-sync design keeps `K` uniform
across the grid). `shutDownEdt` then prints the scalar, destroys the 10
templates, and a dedicated `finalShutdownEdt` (chained off its output event)
calls `ocrShutdown()` — deliberately split out so shutdown never truncates
`shutDownEdt`'s own dependence-release work out of the measured run.

## Placement (as-born)

No `OCR_APP_OPTIMIZED_PLACEMENT` guard exists anywhere in this source — the
affinity usage below is the app's own, unconditional mechanism (gated only
by its own `NO_AFFINITIES`, which is not defined in this build).

- **Data-plane EDTs** (`load`/`create`/`distribute`/`search`/`apply`, all
  `W`-scaled): each worker `w` gets an explicit `OCR_HINT_EDT_AFFINITY` hint
  computed once by `mainEdt`/`startEdt` as `w % rank_count` — a deterministic,
  evenly-spread round-robin *by worker index*, held constant across all 10
  pre-created levels for that worker (so one worker's whole per-level chain
  stays on one rank for the run). This index-modulo placement is **not**
  row/column-aware: workers in the same row or column of the `R×C` grid are
  not co-located, so the row/column BFS-frontier scatter described in Wiring
  is genuinely cross-rank traffic in general.
- **Control-plane EDTs** (`start`/`finish`/`stop`/`shutdown`/`finalShutdown`):
  each passes `local_hint = ocrAffinityGetCurrent()` of its *creator* — since
  `mainEdt` runs on rank 0 and `startEdt` (created by `mainEdt` with that same
  hint) re-captures its own (rank-0) affinity for what it creates, this whole
  chain is pinned to rank 0.
- **DBs**: every `ocrDbCreate` in this app passes `NULL_HINT` → home =
  creating rank. Combined with the EDT pinning above, each worker's own DBs
  (`visitedDBK`, `arrayDBK`, `toRunGuidDBK`, `toRun0/1DBK`) co-locate with
  that worker's pinned rank by construction; the 3 shared control DBs
  (`paramDBK`, `constguidDBK`, `timeDBK`) home on rank 0 and are read
  remotely by every other rank's workers.

## Sizing

`SCALE` drives memory and load-phase compute (`SIZE`, `EDGE_SIZE` both grow
with it); `EDGEFACTOR` drives edge density (and, inversely, expected BFS
depth `K` — a denser graph has a smaller diameter). `R×C` is a purely
*logical* decomposition width, independent of physical ARTS worker-thread
count: raising it increases `W` (hence EDT/Event/DB totals) without changing
the graph. Keep `EDGEFACTOR` generous enough (≥ ~4–8) that `K` stays
comfortably under the hardcoded `MAX_LEVEL=10` cap for the chosen `SCALE`;
`K` can't be predicted statically per argument choice, only observed from a
run's own level-count output, but a graph that does reach the cap now aborts
there instead of wiring the wrong EDTs.

For a 1-node × 15-worker debug run, a small `W` (tens) with `SCALE` in the
single digits keeps the run to seconds and, with a 1-node profile, `rank_count
= 1` so every data-plane EDT lands on the same rank trivially (no cross-rank
traffic to reason about). For multinode strong scaling (e.g. 8 nodes × 15
workers, 120 threads), pick `W` well above the thread count so the
level-synchronized `O(3W)`-wide stages have enough concurrent tasks to keep
every thread fed; the catalog's calibrated args (`SCALE=20 EDGEFACTOR=8 R=32
C=16` → `W=512`) follow exactly this: `SCALE=20` sizes the graph to a
~272 MiB dominant footprint, `EDGEFACTOR=8` keeps `K` safely low, and
`W=512 ≫ 120` keeps every worker thread fed while the fixed `32W+6 ≈
16.4k` EDT creates and `W+14 ≈ 526` event creates stay bounded and exactly
reproducible run to run.
