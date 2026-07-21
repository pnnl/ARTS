.. _coherence_protocols:

Coherence Protocols
===================

.. contents:: On this page
   :local:
   :depth: 2

This page is **normative**. It defines the memory-model contracts an ARTS
build can provide, the coherence protocols that implement them, and the
surrounding design space — both implemented and roadmap.

Model, protocol, timing
-----------------------

A *memory (consistency) model* is a contract: it defines which values a read
may legally return, and therefore which programs are valid. A *coherence
protocol* is an implementation mechanism — replica management, admission
control, ownership transfer, write-back — that satisfies some contract as a
consequence of its design. ARTS keeps the two as **separate build axes**,
plus a third for *when* consistency actions run:

* ``ARTS_MEMORY_MODEL`` selects the **contract** the build implements:
  ``OCR`` (default; races legal, the runtime orders every conflict it must) or
  ``DB_WRF`` (write-race-free at DataBlock granularity: the program must
  event-order every write-write conflict on a DB; evaluation only).
* ``ARTS_COHERENCE_PROTOCOL`` selects the **mechanism**: ``RCU`` (default;
  readers acquire versioned snapshots and are never blocked or invalidated)
  or ``RWLOCK`` (queue-fair distributed reader–writer lock per DB).
* ``ARTS_PROTOCOL_TIMING`` selects the **timing** of consistency actions —
  when write-backs and ownership transfers are performed.
  Values: ``EAGER`` (release-time) or ``LAZY`` (acquire-time, default).

Valid combinations — five in total, one build directory each:
OCR×RCU×{EAGER,LAZY}, OCR×RWLOCK×{EAGER,LAZY}, DB_WRF×RCU×EAGER. Everything
else is a configure-time ``FATAL_ERROR``: DB_WRF×RWLOCK is rejected because a
reader–writer lock already serializes all writers, making the program-side
write-ordering obligation redundant; DB_WRF×RCU×LAZY is rejected because the
DB_WRF arm is home-canonical — the release-time write-back *is* what makes
the home copy canonical.

The model and protocol axes are genuinely orthogonal in one direction: the
same RCU read path serves both contracts. What changes between OCR×RCU and
DB_WRF×RCU is **who orders the update side** — the runtime (via a
single-owner lease) under OCR, or the program (via events) under DB_WRF.

Every rank in a multinode run must use the same build.

.. _ocr_contract:

The OCR contract (reference)
-----------------------------

The contract delivered by the default build (OCR × RCU) is the memory model
of the *Open Community Runtime Interface*, version 1.2.0, §1.6, with the
``synchronized-with`` relation completed as in Dokulil, "Consistency model for
runtime objects in the Open Community Runtime", J. Supercomputing 75:2725–2760,
2018 (doi:10.1007/s11227-018-2681-2), which this document incorporates by
reference. The version pin matters: the OCR specification itself reserves the
right to relax its race rules in future versions.

Its skeleton:

* *sequenced-before* orders operations within one EDT (host C semantics).
* *synchronized-with* is established **only through events** (including each
  EDT's output event). Three release rules apply: (1) all loads/stores by an
  EDT to a DB complete before that DB's release completes; (2) an EDT releases
  **all** of its DBs before its post-slot is satisfied; (3) before an EDT
  explicitly satisfies an event, any DB potentially exposed by that
  satisfaction must be written back and released first.
* *happens-before* is the transitive closure of the two, and defines when a
  write must be visible to a read of the same DB.

Data races are **legal** under this contract, with defined semantics:

* *Non-overlapping rule:* if two unordered EDTs write disjoint 8-byte-aligned
  ranges of one DB, both writes must appear in memory (no write may be masked
  by buffering or whole-block write-back).
* *Overlapping rule:* unordered overlapping writes resolve to one legal
  interleaving at the platform's N-byte store-atomicity granularity (N = 8 on
  AMD64).

Access modes are replication directives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ARTS deliberately provides only the minimal mode algebra:

* ``DB_MODE_RO`` — "this EDT will not write". The runtime may replicate
  freely. **No isolation is promised**: as in OCR's read-only mode, writes by
  concurrent unordered writers may become visible to the reader. (A remote RO
  acquire happens to receive a copy — a snapshot — but that is a distribution
  artifact, not a contract.)
* ``DB_MODE_RW`` — "this EDT may write". The runtime manages ownership.
* ``DB_MODE_NULL`` — control dependence only. ``DB_MODE_VAL`` (raw value) is
  an ARTS extension outside the OCR mode set.

A mode describes a *sharing pattern* the runtime needs for replica management;
it never carries synchronization semantics. Synchronization is the exclusive
business of events (plus intra-node hardware atomics). For this reason ARTS
**permanently omits** OCR's two service modes:

* OCR ``Exclusive write`` (a critical-section service embedded in a mode) —
  mutual exclusion is the program's job, exactly as on hardware shared memory.
  EW is a runtime scheduling service, not part of the memory model proper: the
  normative rules of OCR §1.6 never mention it.
* OCR ``Constant`` (an isolation promise embedded in a mode).

The OCR-compatibility shim translates ``CONST→RO`` and ``EW→RW``. Both
translations are **exact for data-race-free programs** and lossy only for racy
programs that rely on CONST isolation or EW interval exclusion; such programs
are outside the supported subset (this is a property of the translation, not a
non-conformance of ARTS).

Implementations may be stronger; programs must not rely on it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every conforming implementation (both ARTS protocols, and likewise XSOCR and
OCR-Vx) is incidentally *stronger* than the contract in places — e.g. the RCU
protocol serializes unordered inter-node RW sessions via single-owner leases,
and remote RO acquires observe an installed-version copy. These
strengthenings are **never** part of the contract. A program that depends on
them is *non-portable* (it may break on another conforming implementation or
protocol), not invalid. Keeping the contract maximally relaxed preserves
implementation freedom — including future multi-writer merge protocols.

.. _db_wrf_contract:

The DB-WRF contract
-------------------

``ARTS_MEMORY_MODEL=DB_WRF`` delivers a deliberately **weaker** contract —
*write-race-freedom at DataBlock granularity*:

  Any two unordered writes to the same DB (no happens-before edge between
  their acquire/release brackets, at whole-DB granularity — disjoint-range
  sibling writers included) make the program's outcome undefined: concurrent
  writes may be lost entirely, because release performs a whole-DB write-back
  and the last write-back wins.

Read-write races remain **legal**, with the same word-granularity floor as
racy reads under OCR: an unordered reader observes, per 8-byte-aligned word,
some value that was legitimately held by that word (a *regular register* in
Lamport's sense). The name follows the DRF → HRF lineage: where DRF requires
the program to order *all* conflicting pairs, WRF requires it only for
*write-write* pairs.

The lattice of program classes is ``DRF ⊂ DB-WRF ⊂ OCR-legal``: every
data-race-free program is DB-WRF-valid, and every DB-WRF-valid program is
OCR-legal — but OCR-legal programs that exploit the non-overlapping data-race
rule (e.g. unordered sibling writers to disjoint regions of one DB) are
**invalid under DB-WRF** and can silently produce wrong answers there.

.. warning::

   DB_WRF exists to measure the cost of coherence obligations in benchmarks.
   Never use it as a correctness baseline; the build emits a CMake warning
   when selected. The correctness harness marks apps whose wiring relies on
   guarantees outside this contract with ``wrf_rcu_skip`` (a
   contract-ineligibility declaration, not a bug mask).

.. _protocols:

The protocols
-------------

RCU — versioned snapshots, externally ordered updates (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Status:* implemented (``libs/src/core/coherence/rcu/`` — ``eager.c`` /
``lazy.c`` + ``home.c`` + ``ownership.c``); selected by
``-DARTS_COHERENCE_PROTOCOL=RCU`` (the default), timing by
``ARTS_PROTOCOL_TIMING={EAGER,LAZY}`` (default ``LAZY``).

The name is used in McKenney's sense: **readers are never blocked and never
invalidated**. A reader acquires a versioned snapshot and proceeds on it;
updates install new versions without disturbing readers in flight. The
update side's ordering comes from *outside* the read path — which is exactly
what the model axis selects:

* **Under OCR** (OCR×RCU): write access is granted to at most one node at a
  time via a single-owner lease. Within the owning node, multiple RW
  acquirers share a local buffer, so intra-node writes are multi-writer
  (hardware cache coherence keeps them consistent). A remote RW acquire
  requests ownership from the current owner; the owner completes its release,
  ships the payload and transfers the lease. This serializes inter-node
  writes — the OCR non-overlapping rule holds because intra-node concurrent
  writes are hardware-coherent and inter-node ones never overlap in time.
* **Under DB_WRF** (DB_WRF×RCU×EAGER, ``wrf_rcu/wrf_rcu.c``): no ownership
  machinery at all. The home rank always holds the canonical payload;
  release writes the whole DB back to the home synchronously; acquires pull
  from the home. True concurrent multi-writer access is admitted at every
  level — which is precisely why the contract must demand program-side
  write-write ordering (see :ref:`db_wrf_contract`).

The correspondence to canonical RCU is the read-side regime: updaters
serialized outside the read path, readers proceeding on possibly-stale
versions, old versions reclaimed only after their readers drain (refcounted
snapshot buffers standing in for grace periods). Two mechanics deliberately
diverge: updates are applied **in place** rather than copy-on-update —
versions materialize at *serve* time, as reader-bound copies (copy-on-read) —
and reads are consequently **not version-atomic**: a snapshot copied
concurrently with an in-place write may carry mixed content, which the OCR
RO contract legalizes at word granularity.

Reader-side invalidation — in any form — is a **non-goal** of this protocol
family: read-mostly cases where a lock or invalidation protocol wins are a
philosophy difference, not a defect. RO snapshot serving is deduplicated by a
per-rank ``cached_version`` ledger (a monotonic floor of what each rank's
cache holds, credited by serves, by write-back installs, and by the
releaser/shipper itself under LAZY): a request from a rank whose ledger
already matches the master version receives a header-only, no-data reply.

RWLOCK — queue-fair distributed reader–writer lock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Status:* implemented (``libs/src/core/coherence/rwlock/`` — ``eager.c`` /
``lazy.c`` + ``home.c`` + ``arbiters.c``); selected by
``-DARTS_COHERENCE_PROTOCOL=RWLOCK``, timing by
``ARTS_PROTOCOL_TIMING={EAGER,LAZY}``.

Each DB carries a distributed reader–writer lock whose authoritative state is
a single 64-bit ``lock_state`` word on the home rank: shared read grants
overlap; a write grant excludes everything; waiters queue fairly (in the
tradition of queue-based reader–writer synchronization, Mellor-Crummey &
Scott). Data ships with the grant, in the style of entry consistency — the
consistency actions are scoped to the DB whose lock is being acquired.
RWLOCK trivially satisfies the OCR contract (it is strictly stronger:
readers are isolated from writers by admission, not by promise) and serves as
the lock-philosophy point of the design-space comparison.

Retired and rejected arms
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **MRSW** (multi-reader, single *global* writer; version-isolated reads) —
  implemented in 2026-06 as a mirror of the RCU engine, retired in 2026-07 to
  ``archive/coherence-mrsw/``. Its extra strength over OCR×RCU (forbidding
  even node-interleaved writer overlap) bought no benchmark insight for its
  maintenance cost. No configure value selects it.
* **MSI / write-invalidate** — abandoned: on this workload class it was
  dominated by both RWLOCK and RCU, and reader invalidation contradicts the
  RCU philosophy above. OCR-Vx serves as the write-invalidate representative
  in cross-runtime comparisons instead.
* **SRSW** (fully serialized single-reader/single-writer) — never built;
  useful only as a debug lower bound, and RWLOCK already provides a strict,
  deterministic baseline.

.. _orthogonal_dimensions:

Orthogonal protocol dimensions
-------------------------------

The implemented axes fix three points in a larger design space. The following
dimensions are orthogonal to each other; entries marked *roadmap* are
documented options, not commitments.

Timing
~~~~~~

When are write-backs and ownership transfers performed?

* **EAGER** (release-time) — consistency actions run synchronously at
  release: the releasing node writes back modified data (and, under RCU×OCR,
  answers ownership transfers) before the release completes. Acquirers find
  the DB ready. Trades release-side latency for shorter acquire paths.
* **LAZY** (acquire-time, default) — consistency actions are deferred until
  the next acquire or ownership transfer: the releaser is fast (local
  metadata update only); the acquirer pulls the latest data on demand. Fewer
  messages per release, but acquire latency includes the round trip.
* **Invalidate-at-write** (rejected) — publishing invalidations at the moment
  a write begins is the write-invalidate timing; see the MSI entry above.

Propagation strategy
~~~~~~~~~~~~~~~~~~~~

How does fresh data reach readers?

* **Pull-on-acquire** (current RCU) — the acquirer pulls a versioned snapshot
  from the serving side; stale replicas are never invalidated, they simply
  age out when re-acquired. One round trip per (non-deduplicated) acquire.
* **Ship-with-grant** (current RWLOCK) — data travels with the lock grant;
  admission control makes staleness impossible by construction.
* **Update-and-push** (roadmap) — the releasing node proactively pushes the
  new value to current readers. Eliminates the pull round trip when the same
  DB is re-acquired RO by many readers in succession; costly if the push is
  discarded. Aggregation-style RO improvements are the accepted evolution
  path for the RCU family (never reader invalidation).

Directory / ownership
~~~~~~~~~~~~~~~~~~~~~

Where does coherence bookkeeping live?

* **Fixed home** (all current arms) — each DB has a designated home rank
  (set at creation; encoded in the GUID rank field). The home holds the
  authoritative directory entry (RCU×OCR: lease directory; RWLOCK:
  ``lock_state``; DB_WRF: the canonical payload itself). Simple and
  predictable; the home can become a hot-spot for highly shared DBs.
* **Migratory owner** (partially present) — under RCU×OCR×LAZY the payload
  and the per-rank serving ledger migrate with the ownership lease; the home
  keeps only directory state. Full pointer-forwarding migration (home
  consulted only on miss) remains roadmap.
* **Hierarchical NUMA/cluster** (roadmap) — directory partitioned into
  levels (intra-socket, inter-socket, inter-node), exploiting locality to
  reduce long-distance traffic.

Read consistency
~~~~~~~~~~~~~~~~

What version of the DB's contents does a reader observe?

* **Position-dependent** (current RCU) — a reader co-located with the owner
  binds to the live shared buffer; a remote reader receives an
  installed-version snapshot copy. Both are legal under both contracts.
* **Lock-isolated** (current RWLOCK) — readers are admitted only when no
  writer holds the lock, so they always observe a quiescent DB.

.. note::

   The RCU position dependence gives racy programs *location-variable*
   semantics: a co-located racy reader sits on live hardware-coherent memory
   (effectively the machine's TSO floor), while a remote racy reader sees a
   frozen snapshot whose guarantee is only per-word regularity — reading in
   inverse write order no longer proves a publication prefix. Both behaviors
   are within the contract (races promise no more than the word-granularity
   floor), but a racy flag-then-payload idiom can appear to work single-node
   and break distributed. This is a property of any snapshot-serving
   conforming implementation, and one more reason racy idioms are
   non-portable.

Multi-writer resolution
~~~~~~~~~~~~~~~~~~~~~~~

When two writers' accesses to one DB are unordered, how are their writes
reconciled?

* **Node-exclusive serialization** (OCR×RCU) — intra-node: concurrent
  hardware-coherent writes to a shared buffer; inter-node: the single-owner
  lease serializes writers one node at a time. Preserves the OCR
  non-overlapping rule.
* **Admission exclusion** (OCR×RWLOCK) — the lock never grants two writers
  concurrently anywhere; the question is moot.
* **Lossy whole-DB write-back** (DB_WRF×RCU) — each release writes the whole
  DB back to the home; the last write-back wins. Unordered writes are lost —
  hence the stricter program-side contract.
* **Non-lossy merge** (roadmap) — diff- or twin-based merging in the
  TreadMarks / Midway tradition would reconcile concurrent writers' disjoint
  changes, achieving the full OCR non-overlapping rule while admitting true
  inter-node multi-writer access. This is the only path to a genuinely
  OCR-conformant multi-writer protocol.

Per-DB-kind contracts
---------------------

The global axes govern regular DRAM DBs (``ARTS_DB``). The other storage
kinds carry their own fixed, documented semantics regardless of the global
axes: ``ARTS_DB_PIN`` and ``ARTS_DB_GPU_PIN`` perform no DB-level coherence;
``ARTS_DB_GPU`` allows concurrent per-device replicas merged by reduction at
release; ``ARTS_DB_CXL`` relies on hardware cache coherence intra-node and
application-driven ordering across nodes. All of these are **app-ordered
(full DRF) contracts**: the application must event-order *every* conflicting
access pair — read-write included — because there is no snapshot or admission
machinery underneath.

.. _cdag_dropped:

Considered and dropped: CDAG
-----------------------------

*Cache DAG Consistency* (Landwehr et al., 2017) is a variant of DAG
consistency (Blumofe et al., 1996) in which a task observes the write of the
immediately preceding ancestor version of a shared location ("last-ancestor"
read). It effectively maps to a globally single-writer admission policy with
version-isolated reads — the design point the retired MRSW arm occupied.

CDAG was considered as a coherence option for ARTS and was dropped:

1. **Small delta over the RCU engine.** The single-owner lease already
   serializes inter-node writers; adding global writer exclusion and
   last-ancestor reads is a refinement of MRSW, not a separate protocol —
   and MRSW itself was retired for lack of benchmark insight.
2. **Weaker parallelism.** Forbidding competing concurrent writes entirely
   reduces parallelism for programs that exploit the OCR non-overlapping
   data-race rule.
3. **Recoverable.** ``archive/coherence-mrsw/`` preserves the engine CDAG
   semantics would build on, should the design point be revisited.

.. note::

   The token "CDAG" formerly appeared as a *misnomer* in GPU-related source
   comments in this codebase, referring to the GPU Location-Consistency
   invalidation-drain mechanism (the ``invalidate_count`` gate that holds
   off an EDT's output-event satisfaction until all outstanding device-replica
   invalidations drain to zero). That mechanism is entirely unrelated to Cache
   DAG Consistency. The GPU-LC comments have been updated to "GPU LC
   invalidation drain" so that no use of "CDAG" remains in the source.

Historical vocabulary
---------------------

**Earlier ARTS release:** the build axis was ``ARTS_MEMORY_MODEL={RC,LRC,LC}``.
``RC``/``LRC`` were not distinct models — they were the eager/lazy protocols
of the one OCR contract — and ``LC`` was not the literature's Location
Consistency. Those terms are retired; the mapping is
``RC → OCR+EAGER``, ``LRC → OCR+LAZY``, ``LC → RELAXED``.

**Axis history:** an earlier ``ARTS_MEMORY_MODEL`` axis (``OCR`` /
``RELAXED``) was retired in 2026-06 in favour of a single
``ARTS_COHERENCE_PROTOCOL`` admission-policy axis (``MRNEW`` / ``MRMW`` /
``MRSW`` / ``LOCK``). In 2026-07 the memory-model axis was deliberately
**re-introduced** on a new theoretical footing (OCR vs DB-WRF —
write-race-freedom at DataBlock granularity; the earlier "DB-DRF" label for
the weak arm was corrected to DB-WRF, since read-write races remain legal),
orthogonal to the mechanism axis. The mapping from the single-axis era is
exact and behavior-preserving:

* ``MRNEW + EAGER`` → ``OCR × RCU × EAGER``
* ``MRNEW + LAZY``  → ``OCR × RCU × LAZY`` (default)
* ``LOCK + E/L``    → ``OCR × RWLOCK × E/L``
* ``MRMW``          → ``DB_WRF × RCU × EAGER``
* ``MRSW``          → retired (``archive/coherence-mrsw/``)

Using an old ``-DARTS_COHERENCE_PROTOCOL={MRNEW,MRMW,MRSW,LOCK}`` value causes
a CMake ``FATAL_ERROR`` with this mapping, so existing build scripts are
caught at configure time.

References
----------

* OCR working group, *The Open Community Runtime Interface*, v1.2.0, 2016 — §1.6.
* J. Dokulil, *Consistency model for runtime objects in the Open Community Runtime*, J. Supercomputing, 2018.
* P. McKenney, J. Slingwine, *Read-Copy Update: Using Execution History to Solve Concurrency Problems*, PDCS 1998.
* J. Mellor-Crummey, M. Scott, *Scalable Reader-Writer Synchronization for Shared-Memory Multiprocessors*, PPoPP 1991.
* L. Lamport, *On Interprocess Communication, Part II: Algorithms*, Distributed Computing 1(2), 1986 (regular registers).
* S. Adve, M. Hill, *Weak Ordering — A New Definition*, ISCA 1990 (DRF).
* D. Hower et al., *Heterogeneous-race-free Memory Models*, ASPLOS 2014 (HRF).
* T. Landwehr et al., *Designing Scalable Distributed Memory Models*, SC 2017 (Cache DAG Consistency).
* G. R. Gao, V. Sarkar, *Location Consistency — A New Memory Model and Cache Consistency Protocol*, IEEE TC 49(8), 2000.
* K. Gharachorloo et al., *Memory Consistency and Event Ordering in Scalable Shared-Memory Multiprocessors*, ISCA 1990.
* P. Keleher et al., *Lazy Release Consistency for Software Distributed Shared Memory*, ISCA 1992.
* B. Bershad, M. Zekauskas, W. Sawdon, *The Midway Distributed Shared Memory System*, COMPCON 1993.
* L. Iftode, J. P. Singh, K. Li, *Scope Consistency: A Bridge between Release Consistency and Entry Consistency*, SPAA 1996.
* R. Blumofe et al., *DAG-Consistent Distributed Shared Memory*, IPPS 1996.
