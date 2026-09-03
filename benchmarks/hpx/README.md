# HPX reference apps

The normal benchmark build source-builds the pinned HPX submodule in an
isolated build tree, installs it under the ARTS build directory, and uses that
package to build these apps. HPX uses its distributed runtime with the MPI
parcelport; the TCP parcelport is disabled. Like every other vendored runtime
in this repo, HPX is built as static libraries: an app binary embeds HPX,
Boost, and HPX's own pinned hwloc (the release this HPX version validates
against, built static from a checksummed dist tarball), and resolves nothing
through the loader beyond the host toolchain and MPI. (An MPI library may still
load the host's hwloc for itself; the embedded copy is kept out of the dynamic
symbol table so the two never mix.)

```bash
cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release
cmake --build build --target nqueens_hinted_hpx
```

## The six ports

One catalog row per port, one target per mirrored tier. The arguments are the
ones the `hpx-gate` roster runs; each section carries the calibrated size, the
budgets and the campaign it was verified by.

| catalog row | targets | gate arguments | result line | `[STRUCT]` fields |
|---|---|---|---|---|
| [nqueens](#n-queens) | `nqueens_hpx`, `nqueens_hinted_hpx` | `17 13 1 6` (`size cutoff [rounds [scatter-levels]]`) | `<n>-queens; <n>x<n>; sols: <count>` | `tasks edges parcels bytes reducers` |
| [p2p](#p2p) | `p2p_hpx` | `8 256 100 10` (`p m n t [gf]`) | `PASS checksum = <x>`, then `Rate (MFlops/s): …` | `tasks sends parcels bytes ranks inits` |
| [Stencil2D_intel_channelEVTs](#stencil2d) | `Stencil2D_intel_channelEVTs_hpx` | `1000 16 10` (`NP NR NT`) | `Computed L1 norm = <x>`, then `Solution validates` | `tasks strips parcels bytes inits` |
| [smithwaterman](#smith-waterman) | `smithwaterman_hpx`, `smithwaterman_hinted_hpx` | `10 10` over the `*-gate.txt` triple (`tileWidth tileHeight file1 file2 scoreFile`) | `score: <n>`, then `PASSED Expected score: <n>` | `tasks posts_from_locality0 strips parcels bytes` |
| [triangle](#triangle) | `triangle_hpx`, `triangle_hinted_hpx` | `13 1 5` (`[depth [rounds [rows]]]`) | `PASS  final count 29760` | `nodes summers tree boards board_bytes parcels bytes` |
| [tempest](#tempest) | `tempest_hpx`, `tempest_hinted_hpx` | `4 20` (`[patchRange [duration]]`) | the neighbour-grid cross-check, ending `DONE.` | `generations deliveries parcels bytes interests returns state_posts state_bytes map_exchange inits` |

Every port also prints `[E2E]` and `[HPX]` under `ARTS_E2E_MARKER`, and
`[PARCELS]` beside `[STRUCT]` under `ARTS_STRUCT_MARKER`. `[STRUCT] bytes` is
not one quantity across the roster: where the port counts a payload at every
send it is placement-independent (p2p, Stencil2D, smithwaterman), and where it
counts only what crossed a locality it moves with the placement (nqueens,
triangle, tempest). Each port's own bullet says which, and the accounting
table uses `[PARCELS]` rather than this field.

## Threading and pinning

One locality is one process (one MPI rank). Every app here bakes in six
runtime defaults:

- `hpx.use_process_mask=1` — an externally installed CPU affinity mask is
  authoritative: HPX sizes and pins its worker threads inside it instead of
  rebinding from the machine topology (which is HPX's stock behavior).
- `hpx.os_threads=cores` — the default worker count is one per *physical
  core* in the mask, so SMT siblings never carry a second worker. An
  explicit `--hpx:threads=N` overrides.
- `hpx.parcel.mpi.sendimm=1` — sends go out immediately instead of waiting
  for one of the few cached connections; a fire-and-forget delivery that
  waits parks its thread, and a completion burst parks thousands at once
  (each parked HPX thread is a stack, i.e. two memory mappings, against
  the kernel's per-process mapping budget).
- `hpx.run_hpx_main=1` — every locality runs `hpx_main` and stays inside it
  until the global completion edge, because `hpx::finalize()` on *any*
  locality begins global shutdown; a locality that returned early would tear
  the run down under its peers.
- `hpx.thread_queue.max_thread_count=100` — the per-queue ceiling on
  materialising staged work ahead of demand. Every thread object that runs
  takes a 64 KiB stack and a 4 KiB guard page, and the queue keeps the stack
  in its own cache for the rest of the run, so a process's mapping count
  follows how many thread objects it ever materialised rather than how many
  are live. The ceiling is soft — a queue that runs dry raises its own limit,
  and sweeps of it measure flat on the ports that reach a large frontier — so
  it trims a burst rather than bounding one. It is kept — it measured no
  slower than the stock 1000, and trimming a burst is still worth having on a
  wide node — but the bound belongs to the next entry.
- `hpx.scheduler=local-priority-lifo` — the pending queue pops most-recent
  first. A worker then runs its own newest child before its older siblings,
  so an eagerly spawned tree is walked depth-first and the materialised
  frontier is bounded by the tree's *depth*; the stock `local-priority-fifo`
  walks it breadth-first and materialises the whole frontier as started
  thread objects, each holding a stack. Measured on the widest port
  (triangle, hinted, 8 localities x 16): live thread objects per locality
  231 753 -> 5 688 (41x) at 5.8x the speed, and the calibrated size goes
  from exhausting `vm.max_map_count` outright to completing at roughly a
  quarter of it. Every other port is neutral to 8 % better and every answer is
  unchanged. Owner and thief take from the *same* end of that queue:
  `lockfree_lifo::pop` ignores its `steal` argument
  (`libs/core/schedulers/include/hpx/schedulers/lockfree_queue_backends.hpp:186-189`),
  so the pending queue is a stack rather than a work-stealing deque and a
  thief takes the victim's newest item. The backend that steals from the far
  end — `lockfree_abp_lifo`, the Chase-Lev/ABP discipline ARTS's own deque
  uses — is compiled in but unreachable from any option: `abp-priority-lifo`
  resolves to `local_priority_queue_scheduler<std::mutex, lockfree_lifo>`,
  the same type `local-priority-lifo` selects, differing only in the
  scheduler's name string
  (`libs/core/threadmanager/src/threadmanager.cpp:436-437` against
  `:250-251`), and no factory instantiates the ABP backends anywhere. So
  thief-end parity with the runtime this is compared against is not reachable
  without a source change, and this reference makes none. The axis-by-axis
  comparison is `research/hpx-scheduler-parity-2026-09-03.md`.

The adoption evidence for the other five, at their gate sizes, 2 localities x
16 workers, same binary, interleaved x3, medians — the whole roster, not just
the port that forced the choice:

| port (gate size) | stock FIFO | `local-priority-lifo` | ratio |
|---|---|---|---|
| `nqueens_hinted 17 13 1 6` | 2.531 s | 2.326 s | 0.92 |
| `p2p 8 256 100 10` | 0.056 s | 0.104 s | 1.86 |
| `Stencil2D 1000 16 10` | 0.009 s | 0.009 s | 0.94 |
| `smithwaterman_hinted 10 10 <gate>` | 0.171 s | 0.174 s | 1.02 |
| `triangle_hinted 13 1 5` | 5.876 s | **1.345 s** | **0.23** |
| `tempest_hinted 4 20` | 0.031 s | 0.031 s | 1.01 |

Every answer is unchanged. Four of the six cells run for 7-100 ms, entirely
inside startup noise — the p2p ratio in particular is 48 ms of scatter, not a
regression — so only nqueens, triangle and smithwaterman carry a reading:
triangle 4.4x faster, nqueens 8 % faster, smithwaterman 2 % slower.

This is the one entry spelled without HPX's forcing modifier (`key=value`,
not `key!=value`): a forcing entry outranks the command line, and a
scheduling policy has to stay selectable with `--hpx:queuing`. So
`--hpx:queuing=local-priority-fifo` still gets the stock policy, which is how
the two `static_smoke` ctest invocations tell a non-forcing default from a
forcing one.

The LIFO and ABP schedulers exist in every HPX source tree but are compiled
only where 128-bit atomics are available, and in v1.11.0 they never are:
`hpx_check_for_cxx11_std_atomic_128bit()` takes no parameters and never
forwards `${ARGN}`, so the `DEFINITIONS HPX_HAVE_CXX11_STD_ATOMIC_128BIT` its
caller passes is dropped and the macro cannot be defined by the feature test
on any toolchain. Independently, the feature's runtime probe demands
`std::atomic<16 bytes>::is_lock_free()`, which GCC reports false whatever the
hardware. The HPX build therefore forces the option, carries the macro and
`-mcx16` in `CMAKE_CXX_FLAGS`, and links `atomic`; every consumer of the
installed package repeats the macro and the library, because the installed
`defines.hpp` cannot receive what the feature test never defined. The app
project repeats them on its own targets (`arts_hpx_configure`) rather than
inheriting them, so it configures correctly on its own against the install;
the library entry goes in `CMAKE_CXX_STANDARD_LIBRARIES`, because a static
link resolves `__atomic_load_16` only from a library that follows every HPX
archive. `libatomic` is the one loader dependency the ports gain from this —
a GCC runtime library alongside `libstdc++`, not a new piece of vendored
stack, and a compute node without it cannot start a port.

Enabling the feature also switches the *default* FIFO backend from
`hpx::lockfree::queue` to `hpx::lockfree::deque`. Under
`local-priority-lifo` that no longer touches the pending queue, but the
**staged** queue is a `lockfree_fifo` by a fixed template argument
(`libs/core/thread_pools/src/scheduled_thread_pool.cpp`) and the terminated
queue follows the same macro, so both change container with the feature. That
is HPX's own design and this reference takes it as HPX makes it: no local
patch pins the container back. What it is worth was measured against a build
whose FIFO backend was pinned to the lock-free queue — three interleaved
pairs per cell, 27 cells, every answer identical. p2p at `256 8192 1025 32`
on one locality x 16 is **3.44x slower** with the deque (19.13 s against
5.56 s, 0/9 paired reps favouring it) and 1.44x slower at 2 x 16; the
frontier cell `triangle_hinted 8 1 7` at 8 x 16 is 1.62x slower and needs
1.6x the peak mappings; tempest hinted `64 100` at one locality x 16 is
**2x faster** (4.01 s against 8.01 s, 3/3 pairs). Everything else sits inside
its own run-to-run spread. So the staged container is worth a factor of two
to three in either direction on a port whose cells are dominated by posted
work, and which direction is not predictable from the port. It is disclosed
here as HPX behaviour, not removed.

So a bare run uses every physical core of the machine, a run under a
narrowed mask uses exactly the cores of that mask, and one locality per
host — the remote launcher shape — needs no flags at all.
`--hpx:print-bind` prints the realized per-worker binding for verification.

Colocated localities (several per host) additionally depend on the one
backported upstream fix this build carries: stock v1.11.0 shifts each newly registering locality's
binding by the core footprints its peers reported (AGAS `first_used_core`),
computed against the machine topology with no regard for the process mask, so
colocated localities land outside the disjoint per-rank masks the launcher set
up. Upstream fixed this on master (`6df14adf09`, "Don't apply PU offset for
localities that explicitly use core bindings"); the configure step applies that
commit verbatim onto the pristine submodule tree from
`third_party/patches/hpx/` — idempotently, so the submodule working tree stays
at v1.11.0 plus exactly that one patch. With it, a colocated run places every
locality inside its own mask:

```bash
mpiexec -bind-to none -n 2 \
    bash tools/artsrun/envelope.sh 16 2 rank -- \
    build/benchmarks/hpx/nqueens_hinted_hpx 12 8
```

### Thread-count sensitivity

A reference rank gets the same CPU block as an ARTS rank and divides it its
own way: ARTS spends one core of a 16-core block on a dedicated progress
thread and computes on 15, HPX polls the parcelport from its workers and
computes on all 16. What that difference is worth was measured rather than
argued — one locality under a 16-CPU envelope, `--hpx:threads=15` against the
default 16, five runs a side interleaved `A B A B …`, medians of the `[E2E]`
stamp on an otherwise idle host:

| port | arguments | 15 workers | 16 workers | 15/16 |
|---|---|---|---|---|
| nqueens, hinted | `19 15 1 6` | 205.10 s | 192.34 s | 1.066 |
| tempest, hinted | `48 100` | 4.611 s | 4.440 s | 1.039 |
| p2p | `64 4096 513 8` | 0.1829 s | 0.1739 s | 1.052 |
| p2p | `64 4096 513 250` | 6.112 s | 5.673 s | 1.077 |

The linear bound is `16/15 = 1.067` — what a run whose whole cost is
worker-parallel loses on one worker fewer, and the most a run can lose. Every
port sits at or beside it. nqueens, the longest and the most compute-bound,
lands on it to within 0.3 %; the two message-bound ports scatter around it,
p2p's larger size above it on a 20 % run-to-run spread against a 7 % effect.
So the sixteenth worker is worth about its share of the block and nothing
structural: the CPU-budget asymmetry between the two runtimes is a few per
cent, and the paper reports the band rather than correcting for it.

The cell runs outside the experiment driver on purpose — the realized-geometry
check would fail a 15-worker rank holding a 16-core block, which is exactly
the configuration being measured.

### MPI transport on a shared-memory host

Colocated localities send every parcel through the MPI layer's shared-memory
transport. Where that layer is UCX (MPICH `ch4:ucx`, the usual build on a
workstation-class host), its SysV variant grows the receive-descriptor pool
one System V segment at a time and never returns them, so a run whose parcels
run into tens of millions exhausts the system-wide segment limit
(`kernel.shmmni`, 4096 by default) and dies inside `shmget`. Selecting the
POSIX variant removes the limit — it maps files under `/dev/shm` instead:

```bash
UCX_TLS=^sysv mpiexec -bind-to none -n 8 ...
```

The experiment driver exports this for every HPX cell it launches under its
local launcher, and only there: one rank per host cannot reach the limit, so
remote launchers keep their site's MPI defaults. A run interrupted mid-flight
can still leave segments behind — `ipcs -m` shows them, and until they drain
the next multi-rank launch may fail in `MPI_Init`.

### The configuration every table was measured on

One configuration produced every number in this file, and naming it once is
what makes the tables comparable with each other and with the ARTS arms beside
them. It is HPX as HPX configures itself: 128-bit atomics on — by flags,
because this version's own feature test cannot enable them on any toolchain
(see above) — `local-priority-lifo` selected on the sixth configuration line,
and **no local source patch**. The one patch the build applies is upstream's
own `6df14adf09`, backported because a released HPX cannot place colocated
localities inside their masks without it; nothing here alters an HPX design
decision. In particular the staged and terminated thread queues are whatever
128-bit atomics make them, and what that is worth is disclosed above (a
factor of 2-3 either way on post-dominated cells) rather than patched away.
Comparing two runtimes is comparing their designs, so the one axis where they
provably differ is stated rather than closed: HPX's pending queue here is a
LIFO stack that thieves pop from the owner's end, ARTS's is a Chase-Lev deque
whose thieves pop the far end, and the HPX backend that would match it is
unreachable without a source change. The ARTS arms are compiled with
`configs/counters_off.cfg` — every counter compiled out — so no arm pays for
instrumentation the other does not have; symmetrically, each port's own
structural counters are gated on `ARTS_STRUCT_MARKER`, so a timing run never
executes their atomics and the `[STRUCT]` pass is the only run that counts.
One asymmetry survives that claim and is named here instead of folded into
it: the HPX library is built with `HPX_WITH_PARCELPORT_COUNTERS=ON`, which
takes a mutex per parcel in the parcelport's `gatherer::add_data`, so every
parcel a timing run sends pays a counter the ARTS side does not. It is an
unquantified cost against HPX in parcel-heavy cells — the estimate on the
38.8 M-parcel triangle cell is about 5 %, and it has not been measured.
It cannot simply be turned off either: `[PARCELS]` reads exactly those
counters. A counters-OFF library build, A/B against this one, is the recorded
follow-up.
On this host every multi-locality launch carries `UCX_TLS=^sysv` (see above).
Every Budgets table below reports peak `VmHWM`, which `/proc/<pid>/status`
gives in KiB; GB in this file means 1,048,576 KB throughout.

Rows carrying an **earlier build** label are the exception to the first
sentence and are labelled in place. They were measured before the 128-bit
atomics or the scheduling policy were adopted, on the same sources against an
HPX whose feature test had failed — `local-priority-fifo` on the lock-free
queue — and were not re-measured because they are multi-hour calibrated-size
gates. Every Verification table, every fair-geometry table and every Budgets
row without that label is from the configuration named above; so is the
policy-adoption table under Threading and pinning, whose *stock FIFO* column
is by construction the build being compared against.

Four smaller asymmetries in the `[E2E]` span itself, none of them corrected
for, all of them the same order as each other:

- One global collective sits **outside** the HPX span and inside the ARTS
  one. `struct_enabled()` broadcasts on every run, timing runs included, and
  every port calls it before starting its clock, whereas ARTS's
  `init_per_node` and node-start barriers are inside its stamp.
  Sub-millisecond at eight localities; it favours HPX.
- So does per-locality setup arithmetic before the clock — Stencil2D's and
  tempest's loops over 13,824 tiles or patches, triangle's move generator,
  p2p's rank map — each O(10^4) trivial operations, microseconds. The
  *allocating* work is inside the span in both runtimes: the ports post
  `init_tile`/`init_rank`/`patch_init` after the barrier.
- Result printing runs the other way: ARTS's terminal EDT prints before
  shutdown is recognised, the ports print after the stamp is taken. A handful
  of records either way.
- For nqueens, p2p and triangle the completion edge is the root or checksum
  future rather than the second barrier, which the non-zero localities reach
  immediately, so a leaf that has already delivered its count may still be
  returning when the stamp is taken. Sub-microsecond; Stencil2D and tempest
  have genuinely global edges.

Two changes to the ports post-date the **earlier-build** rows: the counter
gating just described (measured at up to 7.6 % on the widest single-locality
cell of the most overhead-dominated port, inside run-to-run noise elsewhere —
so those rows understate the ports rather than flatter them), and the start
stamp moving from after the first barrier to just before it, which adds one
barrier of the order of a millisecond at eight localities to every HPX row.

## N-Queens

One source, `nqueens.cpp`, builds two targets that differ only in where a
subtree goes — the same bitmask search, cutoff, rounds and result line as the
OCR N-Queens app:

- `nqueens_hpx` — the **base** tier: the runtime's no-preference policy
  carried in the program, so every spawn goes to the next locality
  round-robin, reducers included.
- `nqueens_hinted_hpx` — the **hinted** tier: above `scatter-levels`, a
  subtree is mapped to a locality by the same mixed column-mask key as the
  OCR hinted version, and deeper descendants stay where they are.

In **both** tiers the reduction that sums a node's children is a spawn of its
own, not a continuation registered inside the spawning task — the tier decides
only where it goes: the hinted layer keeps the summer on the locality that
created the subtree, which is where the OCR original pins its own summer EDT,
and the base policy places it blindly like every other spawn. Its children's
counts rendezvous by key at whichever locality it landed on. So
`[STRUCT] reducers` is non-zero in both tiers and equal at one locality, where
the two walk the same tree. The shape is not only fidelity: an inline
continuation runs on the delivering child's thread, so every delivery adds to
that worker's set of started-but-unfinished threads and each of them holds a
stack for the rest of the run, while a spawned reduction is demand-limited.

```bash
build/benchmarks/hpx/nqueens_hpx 8 3
mpiexec -bind-to none -n 2 \
    bash tools/artsrun/envelope.sh 16 2 rank -- \
    build/benchmarks/hpx/nqueens_hinted_hpx 8 3 1 3
```

The application arguments are `size cutoff [rounds [scatter-levels]]`.
Search nodes below the cutoff run sequentially. Both commands above print
`8-queens; 8x8; sols: 92`.

Four lines are gated, never printed by a plain run:

- `[E2E] <ns>` — locality 0 only, under `ARTS_E2E_MARKER`; the same stamp the
  other runtimes print, opening just before the first barrier below and
  closing on the second.
- `[HPX] locality=<i> localities=<n> threads=<t>` — under `ARTS_E2E_MARKER`,
  once per locality, so the realized geometry is on the record.
- `[STRUCT] tasks=… edges=… parcels=… bytes=… reducers=…` — under
  `ARTS_STRUCT_MARKER`, summed over localities and printed from locality 0.
  `parcels` counts the edges that left their locality and `bytes` is `8` per
  such crossing — the crossing payload only, so both move with the placement
  while `tasks`, `edges` and `reducers` do not.
- `[PARCELS] sent=… bytes=… wire=…` — under `ARTS_STRUCT_MARKER`, the
  runtime's own MPI parcelport counters (parcels sent, payload bytes,
  serialised sends on the wire), summed over localities and printed from
  locality 0 — an upper bound on the port's `parcels`, since no
  coalescing is configured.

The completion edge is global, not a locality-0 event: locality 0 waits on
the root count while every other locality waits at the second barrier, and
only then does any locality finalize (finalize on one begins shutdown for
all). The stamp opens immediately before the first barrier rather than after
it: no task can run until every locality has arrived there, so that barrier is
part of what the program costs, which is also where the other runtimes' stamps
sit relative to their own start barriers.

**Target rename**: `nqueens_hpx` was the hinted port before 2026-09-02 and is
the base tier since; the hinted port is `nqueens_hinted_hpx`.

`tests/nqueens_core_test.cpp` checks the shared solver core (known solution
counts, cutoff boundary) and runs automatically as the app project's test
step on every build.

### Budgets

Peak `/proc/<pid>/maps` line count and `VmHWM` sampled per process. The
1-locality rows are the current build (20 ms sampler; 641-653 samples on a
calibrated run, 16 on a gate run — a sweep of a 29,000-line `maps` costs more
than the period, so the period is the sweep). The 8-locality rows were
measured on an earlier build and are not re-measured here.

| tier | localities | arguments | workers (per process) | peak mappings (per process) | peak VmHWM (per process) |
|---|---|---|---|---|---|
| base | 1 | calibrated `19 15 1 6` | 128 (bare) | 28,795 | 206,472 KB |
| hinted | 1 | calibrated `19 15 1 6` | 128 (bare) | 28,796 | 225,948 KB |
| base | 1 | gate `17 13 1 6` | 128 (bare) | 29,280 | 169,388 KB |
| hinted | 1 | gate `17 13 1 6` | 128 (bare) | 29,067 | 173,252 KB |
| base | 8 (max across ranks) | calibrated, earlier build | 16 (envelope) | 4,822 | 752,012 KB |
| hinted | 8 (max across ranks) | calibrated, earlier build | 16 (envelope) | 4,707 | 718,244 KB |

Both 1-locality calibrated rows are within 0.1 % of what a build without
128-bit atomics measured (28,786 / 28,812), which is the port's
insensitivity to the pending-queue order: its frontier was already
demand-limited by the spawned reducer, so there was nothing for a
depth-first pop order to bound further. The gate-size rows sit within 2 % of
the calibrated ones for the same reason — the peak is the saturated stack
cache, not the search.

That cache is the whole quantity (see Threading and pinning): every thread
object that ran left a stack and a guard page mapped, so the count climbs with
the run's cumulative thread churn and stops when the cache saturates — which is
why neither the tier nor the scatter tree's shape moves it. It is per process
and follows that process's worker count, not the node count, which is why 2 and
4 localities are not tabulated: every multi-locality geometry runs the same 16
workers per rank as the 8-locality rows. Gate: every geometry stays under
32,000 mappings per process.

### Verification

A small-size (`17 13 1 6`) consensus campaign — roster `hpx-gate`,
`artsrun run -p ferrari -b hpx-gate --entries
arts_val_wb,arts_inv_wb,arts_excl_retain,hpx --apps nqueens --nodes
1,2,4,8` — ran all 32 cells (four coherence arms x two tiers x four node
counts) to `OK`, unanimously corroborated on `95815104`, no `DISAGREE`. The
HPX cells ran under `local-priority-lifo`. The
calibrated-size (`19 15 1 6`) structural gate (`[STRUCT] tasks=T
edges=T reducers=S` at 1 and 8 localities, both tiers) passed with
`T=516550`, `S=52202`.

The campaign's strong-scaling table follows — a small-size shape only, not
a performance measurement: `17 13 1 6` exists to make the consensus run
cheap, not to be representative of the calibrated size in the Sizing section
of the nqueens application document
(`tools/artsrun/src/artsrun/data/appdocs/nqueens.md`). Every cell below is
one observation (`repeats=1`), so the spread between neighbouring cells is
within run-to-run noise:

| app | configuration | 1n | 2n | 4n | 8n |
|---|---|---|---|---|---|
| nqueens:base | VAL·WB·RETAIN | 3.84 | 6.59 | 6.16 | 4.18 |
| nqueens:base | INV·WB·RETAIN | 3.83 | 6.43 | 6.30 | 4.10 |
| nqueens:base | EXCL·WB·RETAIN | 3.84 | 6.96 | 6.32 | 4.34 |
| nqueens:base | HPX | 3.57 | 2.45 | 1.53 | 0.97 |
| nqueens:hinted | VAL·WB·RETAIN | 3.84 | 5.75 | 4.74 | 4.20 |
| nqueens:hinted | INV·WB·RETAIN | 3.86 | 5.97 | 4.96 | 3.57 |
| nqueens:hinted | EXCL·WB·RETAIN | 3.85 | 6.83 | 5.27 | 3.82 |
| nqueens:hinted | HPX | 3.54 | 2.37 | 1.47 | 1.29 |

## p2p

`p2p.cpp` builds one target, `p2p_hpx` — the **base** tier of the PRK
`synch_p2p` wavefront. There is no hinted target, for the same reason the OCR
build has none: the application already carries its own placement. `p` logical
ranks split the `m` columns into contiguous stripes, rank `i` lives on locality
`i / ⌈p / L⌉` (the program's own unguarded `#define BLOCK`), and its whole
generation chain stays there — every generation tail-posts the next one to the
same locality, so a rank never migrates.

One generation is one unit of decomposition — one OCR `p2pEdt` clone — but
two HPX thread objects: the tail post schedules the generation and its
`.then(launch::async, ...)` continuation schedules again. `[STRUCT] tasks`
counts decomposition units, not scheduler threads, and the doubling runs
against the port. A generation consumes the left neighbour's boundary strip,
computes `gf` rows of its own columns, and pushes its right column strip on to
the next rank. The push is receiver-owned and keyed by the *consumer's*
generation index, so the edge exists before either side runs and no producer
ever waits for a consumer to ask. Rank `p−1` wraps one value back to rank 0
once per timestep, and its terminal generation delivers the checksum to
locality 0, which is what closes the run.

```bash
build/benchmarks/hpx/p2p_hpx 8 256 100 10
mpiexec -bind-to none -n 2 \
    bash tools/artsrun/envelope.sh 16 2 rank -- \
    build/benchmarks/hpx/p2p_hpx 8 256 100 10
```

The application arguments are `p m n t [gf]` — ranks, columns, rows, timesteps
and the group factor — with the OCR defaults (`10 100 1000 100 1`) when none
are given. `gf` is capped at 7 by the boundary strip's fixed capacity. Both
commands above print `PASS checksum = 3894.000000`, the same line and closed
form (`(t+1)·(n+m−2)`) the OCR program checks, followed by the same
`Rate (MFlops/s): … Avg time (s): …` line.

Four lines are gated, never printed by a plain run:

- `[E2E] <ns>` — locality 0 only, under `ARTS_E2E_MARKER`; the same stamp the
  other runtimes print, opening just before the first barrier and closing
  on the second.
- `[HPX] locality=<i> localities=<n> threads=<t>` — under `ARTS_E2E_MARKER`,
  once per locality.
- `[STRUCT] tasks=… sends=… parcels=… bytes=… ranks=… inits=…` — under
  `ARTS_STRUCT_MARKER`, summed over localities and printed from locality 0.
  `tasks` is `p·G` with `G = (t+1)·⌈(n−1)/gf⌉`, `sends` is `(p−1)·G + t`
  (every non-last rank's generation, plus one wrap per completed timestep)
  and `bytes` is `8·(p−1)·(t+1)·((n−1) + w) + 8·t` with `w = ⌈(n−1)/gf⌉` — a
  timestep's phases advance `n−1` rows however `gf` divides them, so a short
  last phase changes no total, and at `gf = 1` this is `16·(p−1)·G + 8·t`.
  `bytes` is every send's payload, crossed or not, and is therefore
  placement-independent; `parcels` counts the subset of those
  sends that left the locality, so it is 0 at one locality. `ranks` and
  `inits` are each `p`: the first counts the slots the block map handed out,
  the second the init tasks that filled them.
- `[PARCELS] sent=… bytes=… wire=…` — under `ARTS_STRUCT_MARKER`, the
  runtime's own MPI parcelport counters (parcels sent, payload bytes,
  serialised sends on the wire), summed over localities and printed from
  locality 0 — an upper bound on the port's `parcels`, since no
  coalescing is configured.

`tests/p2p_core_test.cpp` checks the shared core — the column geometry, the
block map, the closed-form structural counts, and a single-threaded replay of
the whole pipeline against the checksum, wrap-around included — and runs
automatically as the app project's test step on every build.

### Budgets

Peak `/proc/<pid>/maps` line count and `VmHWM` sampled once per second, per
process:

| geometry | arguments | workers (per process) | peak mappings (per process) | peak VmHWM (per process) |
|---|---|---|---|---|
| 8 localities | calibrated `6912 1347840 6913 32`, earlier build | 16 (envelope) | 23,520 (locality 0), 5,078-11,988 (the other seven) | 8.82-9.21 GB |
| 1 locality | probe `512 16384 513 64`, earlier build | 16 | 5,359 | 0.13 GB |
| 1 locality | gate `8 256 100 10` | 128 (bare) | 9,957 | 0.08 GB |

The gate row is the current build; the two calibrated rows above it were
measured on an earlier build and are not re-measured here. A sub-second
process is below the 20 ms sampler's useful resolution, so that row is
sampled from a directly launched pid instead, without a sleep between reads
(55 samples); the same run under the 20 ms sampler reads 10,659, which is how
far the two sampling regimes disagree on a run this short.

Gate: every geometry stays under 32,000 mappings per process, locality 0 of
the calibrated run with the least headroom at 23,520.

The calibrated size is **not** tabulated at one locality: the day-one sizing
probe (`128 4096 129 8`, `512 16384 513 8`, `2048 65536 2049 8` at one
locality bare, 2798 / 564 / 1742 ns per generation) extrapolates the
calibrated `p·G = 1,576,599,552` generations to roughly 2,750 s there, past
the 1,800 s a gate run is allowed. The campaign's 1-node cell still runs and
records its own outcome; only the structural/budget gate is 8-locality.

What sets the mapping count is worker count and cumulative thread churn, not
this application's edge count — the third row above is a sub-second run with
8712 tasks and it has 1.9x the mappings of the 15 s, 17-million-task run
above it, because it has 8x the workers. Every started thread's 64 KiB stack and its
guard page stay mapped for the run (see Threading and pinning), and the
runtime's per-worker stack cache retains them after the thread dies, so the
count grows sub-linearly with the number of thread objects ever created and
scales with workers per process. It is *not* live thread objects: the
`/threads{locality#0/total}/count/instantaneous/{pending,staged,all}` counters
peak at 73 / 197 / 247 over the second row's whole run, two orders of
magnitude below its 5,359 mappings — the same statement from the other side:
the mappings are the stacks the run's churn left in the caches, not the
threads alive at any moment.

That measurement also settled the shape of `step()`. Handing the generation
to the scheduler as staged work instead — the continuation runs
`hpx::launch::sync` and only posts a `generation_action` locally — was built
and measured against the shipped form, interleaved:

| form | probe `512 16384 513 64` @1x16 | probe `2048 65536 2049 8` @8x16 | peak pending/staged/all | peak mappings |
|---|---|---|---|---|
| async continuation (shipped) | 15.52 / 13.07 s | 11.29 / 11.44 s (299 / 303 ns per generation) | 73 / 197 / 247 | 5,359 |
| staged continuation | 35.35 / 36.51 s | 15.00 s (397 ns per generation) | 32 / 178 / 210 | 6,207 |

The staged form is 2.4-2.8x slower at one locality and 1.31x slower at eight,
and it does not buy the mappings back (6,207 against 5,359) because the two
forms materialise the same number of thread objects per generation — only
their queue differs. So the continuation stays asynchronous. This is not the
N-Queens reducer situation: there, an inline continuation would have run a
*reduction* on the delivering child's thread and grown that worker's set of
started-but-unfinished threads with every delivery; here the continuation is
one link of a chain that is already demand-limited by its own dependence, and
there is nothing to bound.

The calibrated 8-locality run's stdout carries UCX
`object ... was not returned to mpool ucp_requests` warnings at finalize.
They are MPICH/UCX teardown noise after the result is printed, the run exits
`rc=0`, and they are harmless.

### Verification

A small-size (`8 256 100 10`) consensus campaign — roster `hpx-gate`, `artsrun
run -p ferrari -b hpx-gate --entries
arts_val_wb,arts_inv_wb,arts_excl_retain,hpx --apps p2p --nodes 1,2,4,8` — ran
all 16 cells (four coherence arms x four node counts) to `OK`, unanimously
corroborated on `3894.000000` at every node count, no `DISAGREE`; the HPX cells
ran under `local-priority-lifo`. The calibrated-size (`6912 1347840 6913 32`)
structural gate at 8 localities passed `check_struct.py` with
`tasks=1576599552`, `sends=1576371488`, `bytes=25221943552`, `ranks=6912`,
`inits=6912` — `parcels=1596705`, the 0.10% of handoffs that cross a locality
under the block map — and printed `PASS checksum = 44706783.000000` in 597.2 s.

That 597.2 s is at 8 localities x 16 workers, on the earlier build like
the calibrated Budgets rows above. The ARTS side of the same arguments is the
application document's own strong-scaling row
(`tools/artsrun/src/artsrun/data/appdocs/p2p.md`): 389.8 / 331.4 / 202.0 /
111.3 s at 1 / 2 / 4 / 8 nodes x 15 workers + 1 progress thread. Port at
8 localities x 16 against runtime at 8 nodes x 15 + 1, that is about 5.4x —
the port's largest gap in the roster, and it stands as measured: the port is
the mirror of the OCR program's dependency structure, so a faster shape here
would be a different program, and no further optimisation was attempted in
this plan. There is no 1-locality-bare figure to set beside the ARTS 1-node
389.8 s because that run was never made — the sizing probe extrapolates the
calibrated 1,576,599,552 generations to roughly 2,750 s at one locality, past
the 1,800 s a gate run is allowed (see Budgets).

The campaign's strong-scaling table follows — a small-size shape only, not a
performance measurement: `8 256 100 10` exists to make the consensus run cheap,
not to be representative of the calibrated size in the Sizing section of the
p2p application document (`tools/artsrun/src/artsrun/data/appdocs/p2p.md`).
Every cell below is one observation (`repeats=1`), and at hundredths of a
second the whole table is inside run-to-run noise and startup cost:

| app | configuration | 1n | 2n | 4n | 8n |
|---|---|---|---|---|---|
| p2p:base | VAL·WB·RETAIN | 0.01 | 0.12 | 0.19 | 0.26 |
| p2p:base | INV·WB·RETAIN | 0.01 | 0.16 | 0.21 | 0.56 |
| p2p:base | EXCL·WB·RETAIN | 0.03 | 0.11 | 0.17 | 0.28 |
| p2p:base | HPX | 0.05 | 0.07 | 0.07 | 0.07 |

At this size the pipeline is 8 ranks over 99 phases, so nothing scales and
the node-count column measures distribution overhead only; read the
calibrated numbers in the application document for the shape that does.

### Fair-geometry comparison

Size `256 8192 1025 32` — 256 ranks over 1,025 rows and 32 timesteps, chosen
so a cell lasts seconds rather than hundredths of one. Both sides get 16
first-SMT CPUs per node, so `N x 16 <= 128` physical cores at every geometry:
HPX runs `mpiexec -bind-to none -n N bash tools/artsrun/envelope.sh 16 N rank`
(`16 1 fixed` at one node) with 16 worker threads per locality, ARTS runs
`configs/local/ferrari/{1n,2n_sc,4n_sc,8n_sc}.cfg` with 15 workers + 1
progress thread per node. Counters compiled out on both sides
(`ARTS_COUNTER_CONFIG=configs/counters_off.cfg`), `UCX_TLS=^sysv` for the
8-locality HPX runs. Medians of three interleaved pairs (HPX, ARTS, HPX,
ARTS, ...), single session, no other runner; the measurement is each runtime's
own `[E2E]` stamp. An unrelated resident load of about 20 of the 128 cores was
present for both sides of every pair, which is what the pairing is for —
absolute times here are not comparable with other sessions', ratios are. All 24 runs printed `PASS checksum = 304095.000000`.

| nodes | HPX median | ARTS `val_wb` median | HPX / ARTS |
|---|---|---|---|
| 1 | 20.215 s | 2.334 s | **8.66** |
| 2 | 12.816 s | 4.078 s | **3.14** |
| 4 | 6.342 s | 6.117 s | **1.04** |
| 8 | 3.280 s | 6.912 s | **0.475** |

The ratio falls monotonically with node count and crosses 1 just past four
nodes: the port's deficit is entirely a one-locality deficit, and it is spent
by four nodes because the two sides scale in opposite directions over this
sweep — HPX goes 20.2 -> 3.3 s from one node to eight while ARTS goes
2.3 -> 6.9 s. This is the port that the staged queue's container moves most
(see Threading and pinning): on a build with that container pinned to the
lock-free queue the same four cells read 5.636 / 8.923 / 5.724 / 2.857 s, so
the one-locality ratio would be 2.47 rather than 8.66. The number above is
the one HPX's own configuration produces.

## Stencil2D

`stencil2d.cpp` builds one target, `Stencil2D_intel_channelEVTs_hpx` — the
**base** tier of the Intel channel-events 2-D star stencil (radius 2). There
is no hinted target, for the same reason the OCR build has none: the
application carries its own placement map and no hint-layer guard. `NR` tiles
of an `NP x NP` grid are laid out `NR_X x NR_Y` by the program's own
`splitDimension_Cart2D` (the largest divisor of `NR` at or below
`floor(sqrt(NR+1))`, and its cofactor); the locality count is factored the
same way into `PD_X x PD_Y`, and a tile's `(id_x, id_y)` maps onto a place by
the same block-partition rule the points use — so a tile is a persistent
object on the locality the program's own Cartesian map names, and it never
migrates.

One unit of decomposition per tile per round, again materialised as two HPX
thread objects (the post and its dataflow continuation), so `[STRUCT] tasks`
is an edge count rather than a scheduler-thread count. The tile's four halo
strips of `IN` go out first, read as `IN` stands before the round's update,
and the update is the continuation on the four strips its neighbours pushed. The push is
receiver-owned and keyed by `(tile, side, round)`, so the edge exists before
either side runs and no producer waits for a consumer to ask. After round
`NT` a tile sums `|OUT|` over its active points; the norm `all_reduce` is the
completion edge, and the second barrier is the join before shutdown.

```bash
build/benchmarks/hpx/Stencil2D_intel_channelEVTs_hpx 1000 16 10
mpiexec -bind-to none -n 2 \
    bash tools/artsrun/envelope.sh 16 2 rank -- \
    build/benchmarks/hpx/Stencil2D_intel_channelEVTs_hpx 1000 16 10
```

The application arguments are `NP NR NT` — the domain edge, the tile count
and the timestep count — all three or none, with the OCR defaults
(`1000 16 10`) otherwise. Both commands above print
`Computed L1 norm = 22.000000000000` and `Solution validates`, the analytic
`(NT+1)*2` the OCR program checks against, followed by the same
`Rate (MFlops/s): ...  Avg time (s): ...` line.

Four lines are gated, never printed by a plain run:

- `[E2E] <ns>` — locality 0 only, under `ARTS_E2E_MARKER`; the same stamp the
  other runtimes print, opening just before the first barrier and closing
  on the second.
- `[HPX] locality=<i> localities=<n> threads=<t>` — under `ARTS_E2E_MARKER`,
  once per locality.
- `[STRUCT] tasks=... strips=... parcels=... bytes=... inits=...` — under
  `ARTS_STRUCT_MARKER`, summed over localities and printed from locality 0.
  `tasks` is `NR*(NT+1)`, one per tile per round; `strips` is
  `(4*NR - 2*(NR_X+NR_Y))*(NT+1)`, the directed interior edges of the tile
  grid once per round; `bytes` is the `8*R` doubles each of those carries
  (`8*R*NP/NR_Y` across a vertical edge, `8*R*NP/NR_X` across a horizontal
  one) — every strip's payload, crossed or not, so `bytes` here is
  placement-independent; `parcels` counts the subset of the strips that left
  the locality, so it is 0 at one locality; `inits` is `NR`, the init tasks
  that built the tiles.
- `[PARCELS] sent=… bytes=… wire=…` — under `ARTS_STRUCT_MARKER`, the
  runtime's own MPI parcelport counters (parcels sent, payload bytes,
  serialised sends on the wire), summed over localities and printed from
  locality 0 — an upper bound on the port's `parcels`, since no
  coalescing is configured.

`tests/stencil2d_core_test.cpp` checks the shared core — the dimension split,
the tile bounds, the place map, the closed-form structural counts, and a
single-threaded replay of the whole halo exchange against the analytic norm
on grids whose tiles do *not* divide evenly — and runs automatically as the
app project's test step on every build. The uneven-tile norms are the real
oracle of the halo/strip transcription: a strip read or written in the wrong
order, or off by a row, moves the norm off `(NT+1)*2`.

### Budgets

Peak `/proc/<pid>/maps` line count and `VmHWM` sampled alongside the
calibrated-size (`31104 13824 400`) structural gate run, per process:

| geometry | workers (per process) | peak mappings (per process) | peak VmHWM (per process) |
|---|---|---|---|
| 8 localities, earlier build | 16 (envelope) | 9,133 (locality 0), 4,111-4,555 (the other seven) | 1.98 GB (locality 0), 1.94 GB (the others) |
| 1 locality, bare, earlier build | 128 | **41,195** | 15.14 GB |
| 1 locality, bare, guard pages off, earlier build | 128 | 20,308 | 15.14 GB |
| 1 locality, bare, gate `1000 16 10` | 128 | 922 | 0.07 GB |

The gate row is the current build; the calibrated rows above it were measured
on an earlier build and are not re-measured here.
It is an 18 ms process, far below the 20 ms sampler's resolution, so it is
sampled from a directly launched pid without a sleep between reads (28
samples). It stands for the contrast with the calibrated row — the same 128
workers, 38x fewer mappings — which is the first evidence that the count
follows cumulative thread churn and not worker count.

The 15.14 GB is the grid itself and is expected: `(np_x+4)(np_y+4) + np_x·np_y`
doubles per tile at `288 x 243` is 1.08 MiB, times 13,824 tiles is 14.6 GB,
plus in-flight strips and the runtime.

**The 1-locality bare geometry misses the 32,000-mapping gate at 41,195.**
Every geometry a ferrari campaign actually runs is far inside it — the 1-node
cell runs 16 workers under the envelope, not 128 — so this is a headroom
figure for a wide single node, and the 96-core Tuolumne node is what it is a
proxy for. Root-caused rather than left as a number:

- A snapshot of `/proc/<pid>/maps` 60 s into the run (40,744 mappings, near the
  peak) classifies as 15,236 `---p` 4 KiB anonymous guard pages and 15,102
  `rw-p` 64 KiB anonymous stack bodies — **30,338 of 40,744, 74.5%, are thread
  stacks**, exactly the class Threading and pinning describes. The remaining
  10,406 are the tile arrays, heavily coalesced (the 564 K, 548 K and their
  1,112 K / 1,660 K / 2,208 K sums hold all 27,648 of them): a per-tile array
  is *not* a mapping of its own, because glibc raises its dynamic mmap
  threshold past these sizes.
- 15,102 stack bodies at 128 workers is 118 per worker, already past the
  per-queue ceiling of 100: as Threading and pinning says, that ceiling is
  soft and does not bound the count. What the number records is the run's
  cumulative thread churn, held in the queues' stack caches.
- The escape hatch, measured once rather than assumed:
  `--hpx:ini=hpx.stacks.use_guard_pages!=0` on the command line drops the peak
  to **20,308**, a 50.7 % cut that clears the gate with room to spare, at an
  unchanged answer and an `[E2E]` of 126.8 s against 131.2 s. It is left OFF
  and out of `runtime_defaults()`: a guard page is what turns a stack overflow
  into a fault instead of silent corruption of the neighbouring stack. This
  row records what the hatch buys should a node's `vm.max_map_count` ever make
  it necessary; ferrari's is 1,048,576, so nothing here is close to failing.

A shape that removes one of the two thread objects per tile-round was built and
measured before the shipped form was kept. In it `finish_round` calls
`tile_round(id)` directly instead of `hpx::post<round_action>`, so the next
round's strips go out on the continuation's thread — the same dependency
structure, one scheduling hop fewer:

| form | calibrated @1x128 bare | calibrated @8x16 | peak mappings @1x128 | peak mappings @8x16 |
|---|---|---|---|---|
| posted next round (shipped) | 131.2 s | 48.0 s | 41,195 | 9,133 |
| merged continuation | 120.7 s | 44.7 s | 39,484 | 8,923 |

Merging cuts the mapping peak by 4 % — not the 22 % it would need to reach the
gate, because the stack cache is driven by cumulative churn across 5.5 M
tile-rounds either way, not by the per-round hop count. The two `[E2E]` columns
were measured minutes apart rather than interleaved, which on this host is
inside the drift a comparison must control for, so they are recorded but are
not evidence that either form is faster. With no budget reason to change the
shape and no admissible timing reason either, the posted form stays.

### Allocator bound

ARTS puts its DB payloads on a vendored mimalloc; the ports use whatever
`malloc` the toolchain links, here glibc's. To bound what that asymmetry is
worth, this source is built a second time against an override-mode static
mimalloc — the same translation unit and the same HPX, differing in nothing
but the allocator — and the two binaries are run alternately, one locality
bare (128 workers), medians of three runs a side of the `[E2E]` stamp:

| size | glibc malloc | mimalloc override | plain/override |
|---|---|---|---|
| `8000 256 400` (~10 s) | 9.907 s | 9.989 s | **0.992** |
| `1000 16 10` (gate, tens of ms) | 0.0285 s | 0.0160 s | 1.78 |

The ten-second row is the bound: substituting the allocator moves the port by
0.8 % and in the *slower* direction, inside the spread the same rows carry
(1.0 % across the glibc runs, 4.7 % across the mimalloc ones). The gate row is
not a measurement — at tens of milliseconds it times process startup, where an
allocator's first arena reservation is most of the number — and stands only as
the sanity check that the pair runs and validates at all. The allocator is not
where this port's cost is, so the cross-runtime comparison carries no allocator
correction.

The probe is OFF by default and is never a catalog target: it is built only
when `ARTS_HPX_MIMALLOC_OVERRIDE` names an archive, and then as one extra
binary under `probes/`, beside the ports rather than among them. The archive
has to be linked whole — an override defines entry points the C library
already provides, so on-demand extraction pulls in no member at all — and the
proof that it took is that `MIMALLOC_SHOW_STATS=1` makes the probe print
mimalloc's statistics on exit while the plain binary prints none:

```bash
cmake -GNinja -S third_party/mimalloc -B build_mimalloc_override \
    -DCMAKE_BUILD_TYPE=Release -DMI_OVERRIDE=ON -DMI_BUILD_SHARED=OFF \
    -DMI_BUILD_STATIC=ON -DMI_BUILD_OBJECT=OFF -DMI_BUILD_TESTS=OFF
ninja -C build_mimalloc_override
cmake -Bbuild_release \
    -DARTS_HPX_MIMALLOC_OVERRIDE=$PWD/build_mimalloc_override/libmimalloc.a
ninja -C build_release hpx_apps_ep
MIMALLOC_SHOW_STATS=1 \
    build_release/benchmarks/hpx/probes/Stencil2D_intel_channelEVTs_mimalloc_hpx \
    1000 16 10
```

### Verification

A small-size (`1000 16 10`) consensus campaign — roster `hpx-gate`,
`artsrun run -p ferrari -b hpx-gate --entries
arts_val_wb,arts_inv_wb,arts_excl_retain,hpx --apps
Stencil2D_intel_channelEVTs --nodes 1,2,4,8` — ran all 16 cells (four
coherence arms x four node counts) to `OK`, unanimously corroborated on
`22.000000000000` at every node count, no `DISAGREE`, no `LONE`, no `FAIL`;
the HPX cells ran under `local-priority-lifo`.
The calibrated-size (`31104 13824 400`) structural gate passed
`check_struct.py` at **both** geometries with `tasks=5543424`,
`strips=21984424`, `bytes=93395607552`, `inits=13824`, printing
`Computed L1 norm = 802.000000000000` and `Solution validates` in each;
`parcels` is 0 at one locality and 362,504 at eight — 1.65 % of the strips,
the block-seam edges of the `108 x 128` tile grid cut into `2 x 4` places.

The campaign's strong-scaling table follows — a small-size shape only, not a
performance measurement: `1000 16 10` exists to make the consensus run cheap,
not to be representative of the calibrated size in the Sizing section of the
application document
(`tools/artsrun/src/artsrun/data/appdocs/Stencil2D_intel_channelEVTs.md`).
Every cell below is one observation (`repeats=1`), and at hundredths of a
second the whole table is inside run-to-run noise and startup cost:

| app | configuration | 1n | 2n | 4n | 8n |
|---|---|---|---|---|---|
| Stencil2D_intel_channelEVTs:base | VAL·WB·RETAIN | 0.02 | 0.03 | 0.02 | 0.03 |
| Stencil2D_intel_channelEVTs:base | INV·WB·RETAIN | 0.01 | 0.02 | 0.02 | 0.02 |
| Stencil2D_intel_channelEVTs:base | EXCL·WB·RETAIN | 0.02 | 0.02 | 0.02 | 0.02 |
| Stencil2D_intel_channelEVTs:base | HPX | 0.01 | 0.01 | 0.01 | 0.02 |

At this size the grid is 16 tiles over 11 rounds, so nothing scales and the
node-count column measures distribution overhead only. At the calibrated size
the port takes 131.2 s at one locality bare (128 workers) against the catalog
row's 134.0 / 127.6 / 154.0 s for the three ARTS families at one node, and
48.0 s at eight localities x 16 workers — the same order as the ARTS arms
rather than a different regime, which is the comparison the campaign exists to
make admissible.

## Smith-Waterman

`smithwaterman.cpp` builds two targets, `smithwaterman_hpx` and
`smithwaterman_hinted_hpx` — the **base** and **hinted** tiers of the tiled
global-alignment wavefront, from one source under `HPX_APP_HINTED_PLACEMENT`.
Locality 0 reads the two sequence files and the score file once and broadcasts
them, so every locality holds the whole input and no tile ever fetches a
sequence; it then posts every tile of the `W x H` grid in one serial loop,
exactly as the OCR `mainEdt` issues one `ocrEdtCreate` per tile. Where a tile
lands is the only difference between the tiers. Base takes the runtime's blind
round-robin, which for the `n`-th spawn of one serial loop on one locality is
`n mod L` with `n = (i−1)·W + (j−1)`; the port checks its closed form against
the shared blind counter at every spawn and aborts on a disagreement, so the
mirror is asserted rather than assumed. Hinted takes the application's own
row-band key, `((i−1)·L)/H` — the map `swBandEdtHint` applies in the OCR
source.

A tile is one unit of decomposition and two HPX thread objects — the post and
the dataflow continuation — so its `[STRUCT] tasks` is an edge count, not a
scheduler-thread count. It is the continuation on its three inputs — the west
tile's right column
(`tile_h` ints), the north tile's bottom row (`tile_w` ints) and the
north-west corner (1 int) — each pushed to the consumer's locality by its
producer and keyed by `(i, j, kind)`, so the edge exists before either side
runs and no producer waits for a consumer to ask. A tile on the matrix border
reads a seeded value instead of a pushed one: the linear gap-penalty column,
row and corner that OCR's `initialize_border_values` writes into the border
row and column of the event grid. The bottom-right tile delivers the score to
locality 0, which is what closes the run.

The prologue is where the two runtimes differ most, and the port keeps the
asymmetry rather than hiding it:

- The OCR original spends about eight runtime calls per tile before any tile
  can run — three `ocrEventCreate` on the border-inclusive grid, one
  `ocrEdtCreate` and four `ocrAddDependence` — all serial in `mainEdt`, which
  the application document measures at 9.82 s of a 9.93 s run at the
  calibrated size. The port's loop is one `hpx::post` per tile and nothing
  else: the dependency graph is implicit in the keys and is never
  materialised. That is a difference in what the two programs build, not a
  measurement artefact, and it is the point of the comparison.
- The port reads the input files on locality 0 and broadcasts them *after*
  the first barrier, so the read is inside its `[E2E]` exactly as the OCR
  side's read inside `mainEdt` is inside its own stamp. The broadcast still
  precedes every tile, so nothing computes on a locality that has not got the
  sequences. The application document measures that read at 0.001 s at the
  calibrated size, so this settles a millisecond rather than a regime — but it
  settles it in the direction of the two stamps covering the same work.

```bash
build/benchmarks/hpx/smithwaterman_hpx 10 10 \
    datasets/smithwaterman/string1-gate.txt \
    datasets/smithwaterman/string2-gate.txt \
    datasets/smithwaterman/score-gate.txt
mpiexec -bind-to none -n 2 \
    bash tools/artsrun/envelope.sh 16 2 rank -- \
    build/benchmarks/hpx/smithwaterman_hinted_hpx 10 10 \
    datasets/smithwaterman/string1-gate.txt \
    datasets/smithwaterman/string2-gate.txt \
    datasets/smithwaterman/score-gate.txt
```

The application arguments are `tileWidth tileHeight fileName1 fileName2
scoreFile`, all five required — the same list the OCR program takes, and a
wrong count prints the same usage line. `W` and `H` are derived
(`ceil(len/tile)`) from the mapped sequence lengths, where a mapped sequence
keeps only `A`/`C`/`G`/`T` and drops every other byte, as OCR's
`clear_whitespaces_do_mapping` does. Both commands above print the six
setup lines (`Size of input string 1 is 2000` and its four siblings, then
`Imported 200 x 200 tiles.`), then `score: 1176` and
`PASSED Expected score: 1176` — the score fixture's own number, checked the
way the OCR program's `VERIFY` checks it.

Four lines are gated, never printed by a plain run:

- `[E2E] <ns>` — locality 0 only, under `ARTS_E2E_MARKER`; the same stamp the
  other runtimes print, opening just before the first barrier and closing
  on the second.
- `[HPX] locality=<i> localities=<n> threads=<t>` — under `ARTS_E2E_MARKER`,
  once per locality.
- `[STRUCT] tasks=… posts_from_locality0=… strips=… parcels=… bytes=…` —
  under `ARTS_STRUCT_MARKER`, summed over localities and printed from
  locality 0. `tasks` and `posts_from_locality0` are both `W·H`, the tile
  grid: the first counts tiles that ran, the second posts the creating loop
  issued, and they differ only if a tile was lost. `strips` is
  `H(W−1) + (H−1)W + (H−1)(W−1)`, the directed interior edges of the grid —
  one west column, one north row and one corner per edge — and `bytes` is
  `4·tile_h·H(W−1) + 4·tile_w·(H−1)W + 4·(H−1)(W−1)`, since a strip carries
  the full tile pitch even on a partial edge tile — every edge's payload,
  crossed or not, so `bytes` here is placement-independent. `parcels` counts
  the subset of those edges that left the locality plus the score delivery, so
  it is 0 at one locality.
- `[PARCELS] sent=… bytes=… wire=…` — under `ARTS_STRUCT_MARKER`, the
  runtime's own MPI parcelport counters (parcels sent, payload bytes,
  serialised sends on the wire), summed over localities and printed from
  locality 0 — an upper bound on the port's `parcels`, since no
  coalescing is configured.

`tests/smithwaterman_core_test.cpp` checks the shared core — the character
mapping, the closed-form structural counts at the calibrated and gate sizes,
the partial last column, and the tiled wavefront replayed against an untiled
reference of the same recurrence — and runs automatically as the app project's
test step on every build. The tiled-vs-untiled equality on sequence lengths
that do *not* divide the tile is the real oracle of the kernel and border
transcription: an effective width or height off by one, or a border seeded
from the wrong position, moves the score off the untiled answer while leaving
a tile-aligned case correct.

### Budgets

Peak `/proc/<pid>/maps` line count and `VmHWM` sampled alongside the
calibrated-size (`100 100` over the `string{1,2}-cal.txt` pair, `W=1400`,
`H=1404`) structural gate runs, per process, at a 20 ms period (197-1,984
samples per run, so these are peaks rather than floors):

| geometry | tier | workers (per process) | peak mappings (per process) | peak VmHWM (per process) |
|---|---|---|---|---|
| 8 localities, earlier build | base | 16 (envelope) | 7,744 (locality 0), 3,716-3,744 (the other seven) | 0.19 GB (locality 0), 0.18-0.19 GB (the others) |
| 8 localities, earlier build | hinted | 16 (envelope) | 4,992 (locality 0), 3,756-3,793 (the other seven) | 0.40 GB (locality 0), 0.23-0.28 GB (the others) |
| 1 locality, bare, calibrated, earlier build | base | 128 | 27,973 | 1.72 GB |
| 1 locality, bare, calibrated, earlier build | hinted | 128 | 28,586 | 1.57 GB |
| 1 locality, bare, gate `10 10` | base | 128 | 26,656 | 0.13 GB |
| 1 locality, bare, gate `10 10` | hinted | 128 | 26,982 | 0.13 GB |

The two gate rows are the current build, sampled from a directly launched pid
without a sleep between reads (71 samples each on a half-second run); the
calibrated rows above them were measured on an earlier build and are not
re-measured here.

**Every geometry clears the 32,000-mapping gate**, the widest of them —
1 locality bare at 128 workers — with about 11 % of headroom. So the
`--hpx:ini=hpx.stacks.use_guard_pages!=0` hatch the Stencil2D section
measures was not needed here and was not measured; guard pages stay on, as
they are everywhere.

Two things about these numbers rather than one:

- The 20 ms sampler is not free at this mapping count — reading a
  27,000-line `/proc/<pid>/maps` every 20 ms roughly doubles the run
  (`[E2E]` 14.2 s under it against 6.09 s with a 1 s sampler, base at one
  locality bare). The peaks above are therefore measured on a perturbed run
  and the timings below are not; a longer run has *more* time to accumulate
  thread-stack churn, so the peaks are if anything conservative.
- Locality 0 carries about twice the mappings of any other locality in both
  tiers, which is the creating loop: it is the only locality that runs
  `W*H` posts through its own worker threads before the wavefront spreads.

The 1.5-1.8 GB at one locality is the pre-posted graph, not the input: the
two sequences are 280 KB together, while 1.97 M tiles are posted before the
first of them can retire, each holding a dataflow and up to three hub
entries until its inputs arrive.

### Verification

A small-size (`10 10` over the `string{1,2}-gate.txt` pair) consensus
campaign — roster `hpx-gate`, `artsrun run -p ferrari -b hpx-gate --entries
arts_val_wb,arts_inv_wb,arts_excl_retain,hpx --apps smithwaterman --nodes
1,2,4,8` — ran all **32 cells** (four coherence arms x two tiers x four node
counts) to `OK`, unanimously corroborated on `1176` at every node count and
in both tiers, no `DISAGREE`, no `LONE`, no `FAIL`; the HPX cells ran under
`local-priority-lifo`. The driver reported
"32 cells to run, 0 structurally ineligible": the gate pair and its score
file are listed in the catalog row's `fixtures:`, so the campaign stages them
itself rather than needing them pre-placed.

The calibrated-size structural gate passed `check_struct.py` at **all four**
geometries — both tiers x {8 localities x 16 workers under the envelope,
1 locality bare} — with `tasks=1965600`, `posts_from_locality0=1965600`,
`strips=5891193`, `bytes=1579209588`, each printing `score: 86360` and
`PASSED Expected score: 86360`. `parcels` is 0 at one locality; at eight it
is 3,926,994 for base (66.7 % of the strips: `W=1400` is a multiple of 8, so
under the round-robin residue every north edge stays local and every west and
corner edge crosses) and 19,594 for hinted (0.33 %: the seven band seams,
`7*(W + W-1)`, plus the score delivery). That ratio, 200x fewer crossings for
the same answer and the same task count, is what the hinted tier buys here.

The campaign's strong-scaling table follows — a small-size shape only, not a
performance measurement: `10 10` over a 2,000-character pair exists to make the
consensus run cheap, not to be representative of the calibrated size in the
Sizing section of the application document
(`tools/artsrun/src/artsrun/data/appdocs/smithwaterman.md`). Every cell
below is one observation (`repeats=1`):

| app | configuration | 1n | 2n | 4n | 8n |
|---|---|---|---|---|---|
| smithwaterman:base | VAL·WB·RETAIN | 0.09 | 4.83 | 3.15 | 3.74 |
| smithwaterman:base | INV·WB·RETAIN | 0.09 | 4.74 | 3.27 | 3.66 |
| smithwaterman:base | EXCL·WB·RETAIN | 0.10 | 4.80 | 3.24 | 3.76 |
| smithwaterman:base | HPX | 0.10 | 0.39 | 0.30 | 0.30 |
| smithwaterman:hinted | VAL·WB·RETAIN | 0.09 | 2.21 | 2.16 | 3.25 |
| smithwaterman:hinted | INV·WB·RETAIN | 0.10 | 2.34 | 2.06 | 3.12 |
| smithwaterman:hinted | EXCL·WB·RETAIN | 0.10 | 2.09 | 2.14 | 3.35 |
| smithwaterman:hinted | HPX | 0.09 | 0.19 | 0.23 | 0.45 |

At this size the grid is 200 x 200 tiles of 10 x 10 cells, so the node-count
column measures distribution overhead only — every arm anti-scales across the
first node boundary, which is this application's documented shape and not a
property of any one runtime.

At the calibrated size the port takes 6.09 s (base) and 5.84 s (hinted) at
one locality bare on 128 workers, against the catalog row's 11.8 s and 11.0 s
for ARTS at one node, and 13.27 s / 9.48 s at eight localities x 16 workers —
the same order as the ARTS arms rather than a different regime, which is the
comparison the campaign exists to make admissible, and it anti-scales across
the node boundary exactly as the ARTS arms do.

## Triangle

`triangle.cpp` builds two targets, `triangle_hpx` and `triangle_hinted_hpx` —
the **base** and **hinted** tiers of the peg-solitaire game-tree search, from
one source under `HPX_APP_HINTED_PLACEMENT`. A node applies its move to the
board its parent pushed and then does one of three things: at `depth` it
returns 1, on a dead position it returns 0, otherwise it posts one child per
legal jump plus one summer. The summer is a spawn of its own rather than a
continuation of whichever child delivers last: it opens one keyed receive per
child on its own locality's hub and forwards the sum to the reply address its
creator gave it — the same shape as N-Queens, for the same reason (a
continuation would run on the delivering child's thread and grow that
worker's set of started-but-unfinished threads, each holding a stack; a spawn
is demand-limited).

The board is what separates this port from N-Queens. A child's whole input is
its parent's post-move board, `8·holes` bytes, one fresh copy per child, so
every edge of this tree carries data rather than a search state that fits in a
register: at the calibrated `8 1 8` that is 288 bytes on each of 21,352,742
child edges, 6.15 GB of produced state over the run. It is the roster's one
produced-data fan-out, and it is why the port counts `board_bytes` separately
from what actually crossed a locality boundary.

Where a node lands is the only difference between the tiers, and both mirror
the OCR source:

- Base takes the runtime's blind round-robin for every child, every summer and
  the root — the policy `NULL_HINT` selects in the OCR base build, carried in
  the program by the shared blind counter.
- Hinted takes the application's own key: a child at level
  `<= TRIANGLE_SCATTER_LEVELS` (3) goes to `mix_key(childBits) % L`, a pure
  function of the position it is about to search, and every deeper child and
  every summer stay on the creating locality — the map
  `triChildEdtHint`/`triLocalEdtHint` apply. So a scattered subtree runs
  wire-free beneath its root, which is the whole point of the layer here.
- The root is created with `NULL_HINT` in *both* OCR tiers, so both HPX tiers
  post the root blindly. That is deliberate and is not a gap in the hint
  layer.

```bash
build/benchmarks/hpx/triangle_hpx 13 1 5
mpiexec -bind-to none -n 2 \
    bash tools/artsrun/envelope.sh 16 2 rank -- \
    build/benchmarks/hpx/triangle_hinted_hpx 13 1 5
```

The application arguments are `[depth [rounds [rows]]]`, all optional and
positional, the same list the OCR program takes and with the same defaults: no
`rows` is the author's 5-row board, a `rows` in `[3, 10]` picks a larger
triangle, `depth` outside `[1, holes−2]` falls back to `holes−2`, and `rounds`
repeats the whole search sequentially (the loop lives in `hpx_main`, so rounds
never overlap, as OCR's `wrapupTask` chain never overlaps them). Both commands
above print `triangle puzzle depth 13 rows 5` and then
`PASS  final count 29760` — the author's puzzle's constant, checked in the port
against the same literal and printed through the same `%d` the OCR program
uses. Off the author's board the line is `final count <n> at depth <d> rows
<r>` and consensus judges it.

One transcription difference is worth naming: the OCR program drives the
author's 5-row board from the author's hand-written jump table when `rows` is
absent, and from the generator when `rows` is given (5 included). The port
always drives from the generator, which the core test checks reproduces the
author's table **as a set** — so the two agree on every count and on every
structural number, and differ only in the order sibling children are indexed.
No reported cell touches even that: every roster argument names `rows`, so
both sides run the generator, and the placement key is the board bitmask,
which does not depend on sibling order.

Four lines are gated, never printed by a plain run:

- `[E2E] <ns>` — locality 0 only, under `ARTS_E2E_MARKER`; the same stamp the
  other runtimes print, opening just before the first barrier and closing
  on the second.
- `[HPX] locality=<i> localities=<n> threads=<t>` — under `ARTS_E2E_MARKER`,
  once per locality.
- `[STRUCT] nodes=… summers=… tree=… boards=… board_bytes=… parcels=… bytes=…`
  — under `ARTS_STRUCT_MARKER`, summed over localities and printed from
  locality 0. `nodes` is the search tasks that ran (the tree's nodes),
  `summers` the reduction spawns (its internal nodes), and
  `tree = nodes + summers` is the port's whole task count — the quantity the
  OCR side pins as `NUM_EDT_CREATE − 4`, the four being the three EDTs the OCR
  program creates outside the tree (`mainEdt`, `realmainTask`, `wrapupTask`)
  and the runtime's constant `+1` baseline. `boards` is `nodes − 1` (one board
  per child edge, the root's excepted) and `board_bytes` is `8·holes·boards`;
  both are placement-independent, so all four gate geometries print the same
  five numbers. `parcels` counts the edges that left their locality — a child
  post, a summer post or a count delivery — and `bytes` only the payload those
  crossings carried (`8·holes` per board, 8 per count), so a summer post,
  which carries a reply address and no payload, raises `parcels` alone.
- `[PARCELS] sent=… bytes=… wire=…` — under `ARTS_STRUCT_MARKER`, the
  runtime's own MPI parcelport counters (parcels sent, payload bytes,
  serialised sends on the wire), summed over localities and printed from
  locality 0 — an upper bound on the port's `parcels`, since no
  coalescing is configured.

`tests/triangle_core_test.cpp` checks the shared core — the hole and jump
counts on two boards, the generator against the author's table as a set, the
depth-1 tree by hand, and `count_solutions(5, 13).leaves == 29760` — and prints
the rows-5 structural references the gate runs are compared against. It runs
automatically as the app project's test step on every build. The
generator-vs-author set check is the one independent oracle of the jump
geometry: a mistyped direction or an off-board test that admits one wrong jump
changes the tree without changing anything else that is checked.

### Budgets

Peak `/proc/<pid>/maps` line count and `VmHWM` sampled alongside the
calibrated-size (`8 1 8`) structural gate runs, per process, at a 20 ms period
(523-3,079 samples per run, so these are peaks rather than floors):

| geometry | tier | workers (per process) | peak mappings (per process) | peak VmHWM (per process) |
|---|---|---|---|---|
| 8 localities | hinted | 16 (envelope) | 290,019 (max over ranks; 251,336-315,150 over three runs) | 1.07 GB |
| 8 localities, earlier build | base | 16 (envelope) | 31,078 (locality 0), 30,893-30,913 (the other seven) | 27.73 GB (locality 0), 26.81-27.49 GB (the others) |
| 8 localities, guard pages off, earlier build | hinted | 16 (envelope) | 52,783-315,642 | 1.44-7.99 GB |
| 1 locality, bare | base | 128 | **35,141** | 8.75 GB |
| 1 locality, bare | hinted | 128 | **33,974** | 6.12 GB |
| 1 locality, bare, gate `13 1 5` | base | 128 | 30,570 | 0.52 GB |
| 1 locality, bare, gate `13 1 5` | hinted | 128 | 30,992 | 0.53 GB |
| 1 locality, bare, guard pages off, earlier build | base | 128 | 3,675 | 9.31 GB |

Three things here, and one of them is a host limit this application is the
first in the roster to reach.

**Both 1-locality bare geometries miss the 32,000-mapping gate**, base at
35,141 and hinted at 33,974. Every geometry a ferrari campaign actually runs is
far inside it — the 1-node cell runs 16 workers under the envelope, not 128 —
so this is a headroom figure for a wide single node, and the 96-core Tuolumne
node is what it is a proxy for. The escape hatch was measured once rather than
assumed, on an earlier build whose base peak was 35,278:
`--hpx:ini=hpx.stacks.use_guard_pages!=0` drops that peak to **3,675**, at an
unchanged answer. That is an 89.6 % cut, far more
than the 50 % one saved guard page per stack would explain by itself, and the
reason is coalescing: with a `---p` guard page interleaved between them the
`rw-p` stack bodies can never merge, so the mapping count tracks live thread
objects one for one; without it the adjacent bodies fold into a handful of
VMAs. The hatch is left OFF and out of `runtime_defaults()` — a guard page is
what turns a stack overflow into a fault instead of silent corruption of the
neighbouring stack.

**The hinted tier at 8 localities x 16 workers is what the scheduling policy
bought.** Under the stock FIFO pending queue this geometry could not run at
all: it reached ferrari's `vm.max_map_count` of 1,048,576 within 20 s (471,587
mappings by the tenth second), HPX then threw `mmap() failed to allocate thread
stack due to insufficient resources` on every later thread creation, and the
run made no further progress — killed at 560 s having never printed a count.
The escape then was `--hpx:ini=hpx.stacks.use_guard_pages!=0` (11.2 s, 315,642
mappings), a constant factor on a symptom. Under `local-priority-lifo` the same
geometry finishes in **5.17 s at 290,019 mappings** (medians of three; ranges
4.13-7.09 s and 251,336-315,150) with guard pages ON, the right count and
`parcels=116` — a 3.6x margin under the budget with nothing turned off. Its
smallest reproducing size, `8 1 7`, runs in 1.74 s at 119,190 mappings. The mechanism is the placement, which is why this tier and not the
base one: base sends 88 % of its edges through the parcelport, whose
serialisation paces the arrival of new work, while the hinted tier posts
~100 % of its children locally with nothing pacing them, so a breadth-first pop
order materialises the whole frontier while a depth-first one bounds it by the
tree's depth. The 1-locality bare geometry never hit this at 128 workers,
because those same 21.4 M nodes retire eight times faster.

**The base tier at 8 localities exhausts the host's SysV shared-memory
segments.** It pushes 38,812,521 parcels, and MPICH's UCX shared-memory
transport grows its `mm_recv_desc` pool one SysV segment at a time: the
system-wide segment count climbs from 0 to ferrari's `kernel.shmmni` of 4096
within ten seconds and stays pinned there, every later `shmget` returns ENOSPC
(2.48 M `UCX ERROR` lines), and a rank takes a SIGSEGV. `UCX_TLS=^sysv` moves
the pool to POSIX shared memory (`/dev/shm`, 504 GB free here) and the run
completes unchanged — same count, same `[STRUCT]`. The hinted tier never
approaches it: 116 parcels.

`kernel.shmmni` is not writable without root here, so it is recorded as a host
requirement of the calibrated base cell, not as a port setting. It changes no
answer, and it never appears at the gate size the consensus campaign runs.

The base tier's 27 GB per process at eight localities against the hinted
tier's 1.01 GB at the same geometry is the frontier seen from the memory side:
what is resident is the in-flight set, each entry a `node_t` holding its own
288-byte board, and the pop order is what sets its size.

### Verification

A small-size (`13 1 5`, the author's puzzle at full depth) consensus campaign —
roster `hpx-gate`, `artsrun run -p ferrari -b hpx-gate --entries
arts_val_wb,arts_inv_wb,arts_excl_retain,hpx --apps triangle --nodes 1,2,4,8` —
ran all **32 cells** (four coherence arms x two tiers x four node counts) to
`OK`, unanimously corroborated on `29760` at every node count and in both
tiers, no `DISAGREE`, no `LONE`, no `FAIL`. The driver reported "32 cells to
run, 0 structurally ineligible"; the HPX cells ran under
`local-priority-lifo`.

The calibrated-size structural gate passed `check_struct.py` at **all four**
geometries — both tiers x {8 localities x 16 workers under the envelope, 1
locality bare} — with `nodes=21352743`, `summers=1372805`, `tree=22725548`,
`boards=21352742`, `board_bytes=6149589696`, each printing `final count
19979938`; the two 1-locality runs and the hinted 8-locality run were
re-measured under `local-priority-lifo` and print the same line. That `tree` is
the counted ARTS run's `NUM_EDT_CREATE − 4` exactly, which is what makes the
two runtimes' task counts comparable rather than merely similar. The base
8-locality run takes the host hatch `UCX_TLS=^sysv`; the hinted one needs none
any more.

`parcels` is 0 at one locality in both tiers. At eight it is 38,812,521 for
base — 88.0 % of the 44,078,290 directed edges of the run
(`boards + summers + one count delivery per node`), which is the blind
round-robin leaving about one edge in eight local — and **116** for hinted:
58 board crossings and 58 count crossings, `288·58 + 8·58 = 17,168` bytes, to
the byte. Those 58 are the scattered subtree roots: 68 children live at levels
`<= 3` (2, then 8, then 58, by the board's own branching), 58 of them landed
off their creating locality, and everything below them is wire-free. 334,000x
fewer crossings for the same answer and the same tree is what the hinted tier
buys here.

The pinned gate runs print the core test's rows-5 references in all four
geometries — `nodes=1293179 summers=724549 tree=2017728 boards=1293178
board_bytes=155181360`, `PASS  final count 29760` — with `parcels` 0 at one
locality and, at two localities x 16 workers, 1,708,753 for base against 52 for
hinted (26 board and 26 count crossings, `120·26 + 8·26 = 3,328` bytes).

The campaign's strong-scaling table follows — a small-size shape only, not a
performance measurement: `13 1 5` is the author's 15-hole puzzle and exists to
make the consensus run cheap, not to be representative of the calibrated size
in the Sizing section of the application document
(`tools/artsrun/src/artsrun/data/appdocs/triangle.md`). Every cell below is one
observation (`repeats=1`):

| app | configuration | 1n | 2n | 4n | 8n |
|---|---|---|---|---|---|
| triangle:base | VAL·WB·RETAIN | 0.44 | 71.21 | 60.53 | 42.01 |
| triangle:base | INV·WB·RETAIN | 0.69 | 130.81 | 87.92 | 54.96 |
| triangle:base | EXCL·WB·RETAIN | 0.78 | 77.53 | 56.90 | 48.06 |
| triangle:base | HPX | 2.55 | 7.66 | 7.18 | 5.19 |
| triangle:hinted | VAL·WB·RETAIN | 0.44 | 0.55 | 0.55 | 0.35 |
| triangle:hinted | INV·WB·RETAIN | 0.62 | 0.36 | 0.28 | 0.18 |
| triangle:hinted | EXCL·WB·RETAIN | 0.78 | 0.44 | 0.33 | 0.22 |
| triangle:hinted | HPX | 1.71 | 0.82 | 0.54 | 0.48 |

The hinted HPX row is where the scheduling policy shows up in a campaign
table: the same cells read 2.23 / 4.24 / 3.02 / 2.48 on a build without
128-bit atomics, so every multi-locality cell is 4.6-5.7x faster and only the
1-locality one (where the frontier was never the constraint) is unchanged.

At this size the tree is 2.0 M nodes, so the node-count column measures
distribution overhead only, and it shows the base tier's documented shape in
every arm: each of the 1.29 M child boards is a fresh remote object the moment
the node boundary is crossed, which costs the ARTS base arms two orders of
magnitude and the port a factor of three. The hinted tier removes it on both
sides.

At the calibrated size the port takes 21.6 s (base) and 18.7 s (hinted) at one
locality bare on 128 workers, against the catalog row's 10.2 / 18.9 / 13.9 s
for the three ARTS families at one node — the same order as the ARTS arms
rather than a different regime, which is the comparison the campaign exists to
make admissible. Those two and the 82.7 s the base tier takes at eight
localities x 16 workers are earlier-build figures; the hinted tier at that
geometry, re-measured on the configuration this file documents, is **5.17 s**
(median of three, 4.13-7.09 s). Base anti-scales across the node boundary as
every ARTS base arm does; hinted does not.

## Tempest

`tempest.cpp` builds two targets, `tempest_hpx` and `tempest_hinted_hpx` — the
**base** and **hinted** tiers of the cubed-sphere neighbour exchange, from one
source under `HPX_APP_HINTED_PLACEMENT`. Six panels of `k x k` patches; a patch
has eight neighbours from the panel-seam topology, except the 24 panel-corner
patches, which have seven. A generation consumes the values its neighbours sent
at the previous generation, writes its own patch number into one value per
neighbour and forwards them, and launches the next generation; the terminal
generation (`duration - 1`) only reports. Patch 0 prints `timestep: <g>` every
generation and, at the end, its computed neighbour grid, the header, and the
3 x 3 grid of the numbers it actually received — whose south-east cell is what
the catalog's regex reads. That per-generation line is an unbuffered `write(2)`
where the OCR program uses a buffered `printf`: one syscall per generation on
one patch, a couple of milliseconds at the calibrated length, and a cost the
port pays and the OCR does not. A generation is likewise one unit of
decomposition and two or three HPX thread objects (hinted: the post and its
continuation; base: those plus `serve_interest`), so `[STRUCT] generations`
counts decomposition units rather than scheduler threads — again against the
port.

A value lands at the **creating locality** of the patch it is addressed to,
keyed by `(patch, side, generation)`, in that locality's hub. That single rule
is what the two tiers differ over:

- Hinted keeps a patch's state resident on its home (`home_rank`: a patch-number
  band below six localities, otherwise the minimum-cut `P x Q` block of the
  `k x 6k` strip — the map `patchHomeRank` applies), and a generation is the
  continuation on the eight receives that are already local. Creator and home
  are the same locality by construction, so nothing has to be published.
- Base runs a generation wherever the blind round-robin put it, carrying the
  96-byte patch state in the post. That generation cannot receive its own
  inputs — they were keyed at the creating locality — so it opens one
  batched receive on its own hub and posts an *interest* to the creator, which
  gathers the <= 8 keyed values there and returns them in one parcel. This is
  the OCR base's three-party rendezvous exactly: the channel lives on the
  `patchInit` rank, the producer satisfies it there, and the consumer pulls
  from it wherever it happens to run.
- Because a patch's creator is not a function of the patch under blind
  placement, the base tier publishes the patch->creator map once after init
  (`all_gather` of each locality's created list, `6k^2` entries total) — the
  analogue of the OCR base's labeled-sticky handshake, and, like it, inside the
  measured stamp on both sides. The hinted tier computes the same answer and
  exchanges nothing.

The map is published *and installed everywhere* before the first generation
exists: a barrier separates the install from the launch loop. A generation is
placed blindly, so the first one may run on any locality and reads the map
there; gathering the lists only fills each locality's own copy, so without that
barrier a generation arriving early reads the pre-exchange default, asks the
wrong locality for its inputs, and waits forever on values that are being
delivered somewhere else. That is a real wedge, not a theoretical one — it
cost this port a hung 8-locality cell before the barrier was added.

```bash
build/benchmarks/hpx/tempest_hpx 4 20
mpiexec -bind-to none -n 2 \
    bash tools/artsrun/envelope.sh 16 2 rank -- \
    build/benchmarks/hpx/tempest_hinted_hpx 4 20
```

The application arguments are `[patchRange [duration]]`, both optional and
positional, the same list the OCR program takes and with the same defaults:
`patchRange` (`k`) is the width dial, `duration` the length dial, and no
arguments at all prints `NO PATCHRANGE ARG GIVEN. USING DEFAULT PARAMS
(patchRange=2).` and runs `k=2`, `duration=100`. A non-positive or non-numeric
argument, or more than two, is a usage error. Both commands above print the
neighbour grid, `*CROSS-CHECKING NEIGHBOR DATA EXCHANGE*`, the received grid
whose south-east cell is `81` at `k=4`, and `DONE.`

Four lines are gated, never printed by a plain run:

- `[E2E] <ns>` — locality 0 only, under `ARTS_E2E_MARKER`; the same stamp the
  other runtimes print, opening just before the first barrier and closing
  on the second.
- `[HPX] locality=<i> localities=<n> threads=<t>` — under `ARTS_E2E_MARKER`,
  once per locality.
- `[STRUCT] generations=... deliveries=... parcels=... bytes=... interests=...
  returns=... state_posts=... state_bytes=... map_exchange=... inits=...` —
  under `ARTS_STRUCT_MARKER`, summed over localities and printed from locality
  0. `generations` is `6k^2 . duration`, `deliveries` the value edges of the
  run, `(48k^2 - 24) . (duration - 1)` — both placement-independent, so all
  four gate geometries print the same two numbers. `inits` is `6k^2 + 6` (one
  `patch_init` per patch, one `panel_init` per panel). The five base-only
  fields are the price of blind placement and the hinted tier gates them at
  zero: `interests` and `returns` are the two halves of the batched pull
  (one each per generation), `state_posts` the generation posts that carry a
  state and `state_bytes` the bytes they carry (96 each), `map_exchange` the
  one publication of the patch->creator map. `parcels` counts the edges that
  left their locality and `bytes` only the payload those crossings carried, so
  a `done` signal, which carries a patch number and no payload, raises
  `parcels` alone.
- `[PARCELS] sent=… bytes=… wire=…` — under `ARTS_STRUCT_MARKER`, the
  runtime's own MPI parcelport counters (parcels sent, payload bytes,
  serialised sends on the wire), summed over localities and printed from
  locality 0 — an upper bound on the port's `parcels`, since no
  coalescing is configured.

The cross-check itself — the neighbour grid, the header and the received grid —
is one `write()`, not six. The driver reads the header and the grid under it as
a single record, and a launcher that merges several localities' stdout and
stderr into one stream interleaves them at write granularity, so a record split
across writes can take an unrelated line (the `[E2E]` stamp, for one) in the
middle of it and stop matching. Anything a driver parses across lines belongs
in one write.

`tests/tempest_core_test.cpp` checks the shared core, and it is the only
independent oracle of the transcribed topology: every neighbour of every patch
on five board sizes has a matching reverse edge (`find_neighbor_patch(Q, k,
neighbor_relation(P, Q, k)) == P`), exactly 24 patches per board have seven
neighbours rather than eight, the directed edge count is `48k^2 - 24`, and
patch 0's south-east neighbour at `k = 48` is the catalog's pinned `11521`. It
also prints `k 4: SE of patch 0 = 81`, the gate runs' expected scalar, and
checks that the hinted map is a band below six places and covers every place at
eight. It runs automatically as the app project's test step on every build.

### Budgets

Peak `/proc/<pid>/maps` line count and `VmHWM` sampled alongside the
calibrated-size (`48 1900`) structural gate runs, one process, at a ~110 ms
period (2,586 and 2,165 samples, so these are peaks rather than floors):

| geometry | tier | workers | peak mappings | peak VmHWM |
|---|---|---|---|---|
| 1 locality, bare, calibrated, earlier build | base | 128 | **44,114** | 1.30 GB |
| 1 locality, bare, calibrated, earlier build | hinted | 128 | **45,481** | 0.39 GB |
| 1 locality, bare, `48 1` (init only), earlier build | base | 128 | 28,639 | 0.15 GB |
| 1 locality, bare, `48 1` (init only), earlier build | hinted | 128 | 35,724 | 0.17 GB |
| 1 locality, bare, `48 100`, guard pages off, earlier build | base | 128 | 4,331 | 0.25 GB |
| 1 locality, bare, gate `4 20` | base | 128 | 6,757 | 0.08 GB |
| 1 locality, bare, gate `4 20` | hinted | 128 | 7,656 | 0.06 GB |

The two gate rows are the current build; the rows above them were measured on
an earlier build and are not re-measured here. A 0.1 s process is below the
20 ms sampler's useful resolution, so the gate rows are sampled from a
directly launched pid without a sleep between reads (41 samples each).

**Both 1-locality bare geometries miss the 32,000-mapping gate**, base at
44,114 and hinted at 45,481 — and, unlike the other ports in this roster, the
two tiers miss it by about the same amount. Every geometry a ferrari campaign
actually runs is far inside it (the 1-node cell runs 16 workers under the
envelope, not 128), so this is a headroom figure for a wide single node and the
96-core Tuolumne node is what it is a proxy for. The escape hatch, measured
once rather than assumed, at the smallest size that still shows the growth
(`48 100`, base): `--hpx:ini=hpx.stacks.use_guard_pages!=0` drops the peak from
**38,671 to 4,331**, an 88.8 % cut, at an unchanged answer. That is the
coalescing effect the other sections record — with a `---p` guard page
interleaved between them the `rw-p` stack bodies can never merge, so the count
tracks live thread objects one for one; without it the adjacent bodies fold
into a handful of VMAs. The hatch is left OFF and out of `runtime_defaults()`:
a guard page is what turns a stack overflow into a fault instead of silent
corruption of the neighbouring stack.

What this port adds to the roster's picture of that number is where it comes
from. The `48 1` runs above do one generation each — init and the terminal
report, no steady state at all — and they already stand at 28,639 and 35,724
mappings within a second. So the peak here is not the steady state
accumulating: it is the creation phase, `6k^2 = 13,824` patch inits and their
immediate generation-0 launches, materialising thread objects faster than 128
workers retire them, after which the stack cache stays saturated for the rest
of the run. The tiers landing within 3 % of each other follows: their creation
phases are identical and only their steady states differ.

Nothing calibrated here is a multi-locality figure, and that is a property of
the measuring host rather than of the port. Every multi-locality geometry on
this box is several localities sharing one node's cores and one node's memory,
so the catalog's calibrated arguments — sized for a whole node per rank —
make a local multi-locality run memory-bound rather than representative.
Multi-locality numbers in this section are therefore small-argument numbers by
design, and a calibrated multi-locality figure is something the target machine
produces, not this one. The reference point that exists: `48 50` — a
thirty-eighth of the calibrated length — at 8 localities x 16 workers takes
**7.2 s** for base and **4.9 s** for hinted, with base pushing 6,564,681
parcels and 154 MB against hinted's traffic-free steady state.

A multi-locality run of this port at a calibrated size would also need
`UCX_TLS=^sysv`. MPICH's UCX shared-memory transport grows its
receive-descriptor pool one SysV segment at a time and pins ferrari's
`kernel.shmmni` of 4096 once tens of millions of parcels fly, after which every
`shmget` fails and a rank takes a SIGSEGV; `^sysv` moves the pool to POSIX
shared memory and changes no answer. The base tier's parcel count at eight
localities is of that order — 6.6 M already at a fiftieth of the calibrated
length. It is not needed at one locality, where nothing crosses the
parcelport at all. The same segment limit is reachable from the other side
too: a killed multi-rank run does not release its segments, so a few
interrupted eight-rank runs in a row can make the *next* MPI job fail in
`MPI_Init_thread` until they drain.

### Verification

A small-size (`4 20`) consensus campaign — roster `hpx-gate`, `artsrun run -p
ferrari -b hpx-gate --entries arts_val_wb,arts_inv_wb,arts_excl_retain,hpx
--apps tempest --nodes 1,2,4,8` — ran all **32 cells** (four coherence arms x
two tiers x four node counts) to `OK`, unanimously corroborated on `81` at
every node count and in both tiers, no `DISAGREE`, no `LONE`, no `FAIL`. The
driver reported "32 cells to run, 0 structurally ineligible"; the HPX cells
ran under `local-priority-lifo`.

The calibrated-size structural gate passed `check_struct.py` at both tiers at
1 locality bare, with `generations=26265600`, `deliveries=209968632`,
`inits=13830` in both, the base tier adding `interests=26265600
returns=26265600 state_posts=26265600 state_bytes=2521497600 map_exchange=1`
and the hinted tier printing those five as zero. Both logs carry the pinned
`11521` as the received grid's south-east cell, which is the value the
catalog's own `scalar_re` extracts from them.

The pinned gate runs at `4 20` — both tiers, 1 locality bare (128 workers) and
2 localities x 16 workers under the envelope — printed the same neighbour grid
and the same received grid in all four, south-east cell `81`, and
`generations=1920 deliveries=14136 inits=102` in all four. `parcels` is 0 at
one locality in both tiers; at two it is 10,053 for base (240,664 bytes)
against 3,040 for hinted (24,320 bytes), the blind placement's cost against the
map's.

**Init share.** The catalog's arguments separate the two dials, so the port's
own start-up cost is directly measurable: `48 1` runs the same `6k^2 = 13,824`
patch inits and then one terminal generation, and nothing else. At 1 locality
bare it takes 0.40 s (base) and 0.54 s (hinted) against the calibrated run's
277.7 s and 117.2 s — an HPX init share of **0.14 %** and **0.46 %**. So the
structural counts above are counts of steady state, not of set-up, in both
tiers. (The received grid at `48 1` is all `-1`: with one generation there has
been no exchange yet to cross-check, which is why the init pair is a timing
probe and not a correctness one.)

**Timing shape.** At 1 locality bare on 128 workers the calibrated run takes
**277.7 s** (base) and **117.2 s** (hinted) — both measured on an idle host,
which is the only way these are comparable to anything; an earlier pair taken
while another job held the same cores read 41 % and 147 % high and is
superseded. The ARTS side of the same calibrated size is the application
document's one-node row (`tools/artsrun/src/artsrun/data/appdocs/tempest.md`):
`48 1900` at 1 node x 15 workers + 1 progress thread measures 19.0 / 19.3 /
20.2 s across the three coherence families. Port at 1 locality x 128 bare
workers against runtime at 1 node x 15 workers, that is about **14x** (base)
and **6x** (hinted) — the two sides are not the same machine width, which is
why both geometries are named. The catalog comment's 4.8 s is the Sizing
lattice's `k = 16` trend point, a different size, and is not the ARTS figure
for this row. This row is the roster's extreme case for the port and the gap
is under investigation — at 128 workers a port's time moves by up to 5x with
the thread-queue backend alone (see Threading and pinning), so a profiling
diagnosis of that regime is the recorded follow-up. What the split between the
two tiers measures is the base tier's protocol rather than the exchange
itself: every one of the 26.3 M generations pays an interest and a return (the
`interests`/`returns` columns are exactly the generation count) plus a 96-byte
state post, where a resident-state design pays none of the three — which is
most of what separates the two tiers' own 277.7 s and 117.2 s. The accounting
pass quantifies the split.

The campaign's strong-scaling table follows — a small-size shape only: `4 20`
exists to make the consensus run cheap and is not representative of the
calibrated size in the Sizing section of the application document
(`tools/artsrun/src/artsrun/data/appdocs/tempest.md`). Every cell below is one
observation (`repeats=1`), and at this size the whole run is 1,920 generations,
so the node-count column measures distribution overhead only:

| app | configuration | 1n | 2n | 4n | 8n |
|---|---|---|---|---|---|
| tempest:base | VAL·WB·RETAIN | 0.01 | 0.39 | 0.49 | 0.30 |
| tempest:base | INV·WB·RETAIN | 0.01 | 0.62 | 0.65 | 0.49 |
| tempest:base | EXCL·WB·RETAIN | 0.01 | 0.34 | 0.53 | 0.30 |
| tempest:base | HPX | 0.03 | 0.07 | 0.09 | 0.04 |
| tempest:hinted | VAL·WB·RETAIN | 0.01 | 0.13 | 0.19 | 0.10 |
| tempest:hinted | INV·WB·RETAIN | 0.01 | 0.30 | 0.30 | 0.14 |
| tempest:hinted | EXCL·WB·RETAIN | 0.01 | 0.22 | 0.10 | 0.09 |
| tempest:hinted | HPX | 0.02 | 0.03 | 0.03 | 0.04 |

### Fair-geometry comparison

Size `64 100` — `6k^2 = 24,576` patches for 100 generations, the smallest of
`16 100` / `48 100` / `64 100` whose **hinted** tier lands in 2-30 s at one
node on *both* runtimes (the two smaller sizes put ARTS at 0.116 s and
1.314 s; the size was calibrated on the build that preceded this one and is
still inside the window on this one, 4.007 s HPX against 2.609 s ARTS). The
size is held fixed across both tiers and all four node counts. Both sides get 16 first-SMT CPUs per node, so `N x 16 <= 128`
physical cores at every geometry: HPX runs `mpiexec -bind-to none -n N bash
tools/artsrun/envelope.sh 16 N rank` (`16 1 fixed` at one node) with 16 worker
threads per locality, ARTS runs `configs/local/ferrari/{1n,2n_sc,4n_sc,8n_sc}.cfg`
with 15 workers + 1 progress thread per node. Counters compiled out on both
sides (`ARTS_COUNTER_CONFIG=configs/counters_off.cfg`), `UCX_TLS=^sysv` for the
8-locality HPX runs. Medians of three interleaved pairs (HPX, ARTS, HPX, ARTS,
...), single session, no other runner; the measurement is each runtime's own
`[E2E]` stamp. The same unrelated resident load noted in the p2p section was
present for both sides of every pair. All 48 runs printed `20481` as the cross-check grid's south-east cell —
`find_neighbor_patch(0, 64, SE)`, the `k = 64` analogue of the core test's
published `81`.

Base tier:

| nodes | HPX median | ARTS `val_wb` median | HPX / ARTS |
|---|---|---|---|
| 1 | 8.814 s | 2.649 s | **3.33** |
| 2 | 45.037 s | 476.337 s | **0.095** |
| 4 | 34.100 s | 416.304 s | **0.082** |
| 8 | 24.580 s | 313.884 s | **0.078** |

Hinted tier:

| nodes | HPX median | ARTS `val_wb` median | HPX / ARTS |
|---|---|---|---|
| 1 | 4.007 s | 2.609 s | **1.54** |
| 2 | 4.997 s | 12.049 s | **0.415** |
| 4 | 2.521 s | 11.409 s | **0.221** |
| 8 | 1.483 s | 10.653 s | **0.139** |

Both tiers are ahead of the runtime at every geometry that crosses a node and
behind it at one locality, and they differ in both the size of that one-node
deficit and where the ratio settles. Base is 3.3x slower at one node, then
its blind placement costs both runtimes an order of magnitude at the first
node boundary and holds the ratio flat near 0.08 from two nodes to eight.
Hinted is only 1.54x slower at one node and its ratio keeps falling
(0.42 -> 0.14), because HPX goes on strong-scaling (5.0 -> 1.5 s) while the
runtime's time barely moves (12.0 -> 10.7 s). The hinted tier's one-node cell
is the other side of the staged-queue sensitivity p2p pays for: on a build
with that container pinned to the lock-free queue it reads 8.296 s, i.e.
twice as slow, for a ratio of 3.15. Base is insensitive to it (9.093 s
pinned against 8.814 s here), which is why the two tiers no longer agree on
the one-node figure the way they did on that build.

## Experiment driver

`artsrun` offers HPX as the off-plane selection entry `hpx`: it runs the
applications whose catalog row carries `hpx: [<tiers>]`, one target per
mirrored tier (nqueens, smithwaterman, triangle and tempest: `hpx: [base,
hinted]`; p2p and `Stencil2D_intel_channelEVTs`: `hpx: [base]`), launched like
the other references (mpirun or srun + the CPU envelope) with the same `[E2E]`
measurement and consensus vote. `artsrun run --entries hpx --apps nqueens ...`
runs it alone.
