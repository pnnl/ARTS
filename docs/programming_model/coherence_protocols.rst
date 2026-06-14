.. _coherence_protocols:

Coherence Protocols
===================

.. contents:: On this page
   :local:
   :depth: 2

This page is **normative**. It defines the contract every ARTS build provides
and surveys the protocol design space — admission policies, timing variants,
and orthogonal dimensions — both implemented and roadmap.

Model vs protocol, reframed
----------------------------

A *memory (consistency) model* is a contract: it defines which values a read
may legally return, and therefore which programs are valid.  A *coherence
protocol* is an implementation mechanism — replica management, admission
control, ownership transfer, invalidation, write-back — that satisfies some
contract as a *consequence* of its design.  The contract is therefore not a
build axis: it is what falls out of the protocol you choose.

ARTS exposes one primary build axis and one conditional sub-axis:

* ``ARTS_COHERENCE_PROTOCOL`` selects the **admission policy** — how many
  concurrent readers and writers the protocol grants access to a DataBlock.
  Values: ``MRNEW`` (default) or ``MRMW``.
* ``ARTS_PROTOCOL_TIMING`` selects the **timing** of consistency actions —
  when write-backs, ownership transfers, and invalidations are performed.
  Values: ``EAGER`` (release-time) or ``LAZY`` (acquire-time, default).
  Meaningful only when ``ARTS_COHERENCE_PROTOCOL=MRNEW``; ignored (with a
  CMake STATUS message) under ``MRMW``.

Every rank in a multinode run must use the same build.

.. _ocr_contract:

The OCR contract (reference)
-----------------------------

The contract delivered by the default protocol (MRNEW) is the memory model of
the *Open Community Runtime Interface*, version 1.2.0, §1.6, with the
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
  interleaving at the platform's N-byte store-atomicity granularity.

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
OCR-Vx) is incidentally *stronger* than the contract in places — e.g. current
ARTS protocols serialize unordered inter-node RW sessions via single-owner
leases, and remote RO acquires observe an installed-version copy. These
strengthenings are **never** part of the contract. A program that depends on
them is *non-portable* (it may break on another conforming implementation or
protocol), not invalid. Keeping the contract maximally relaxed preserves
implementation freedom — including future multi-writer merge protocols.

.. _admission_ladder:

Admission-policy ladder
------------------------

The admission policy governs how many concurrent readers and writers the
protocol grants. The ladder below runs from most permissive to strictest. All
entries that are stronger than the OCR contract still *satisfy* it (a valid
OCR program gets correct results under any of them).

MRMW — Multi-Reader, Multi-Writer (evaluation only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Status:* implemented (``mrmw.c``); selected by
``-DARTS_COHERENCE_PROTOCOL=MRMW``. Emits a CMake configure warning.

Allows true concurrent multi-writer access at every level: intra-node and
inter-node. No exclusive owner is maintained. At release, the entire DataBlock
is written back (whole-DB write-back), which **loses concurrent writes** —
the last write-back wins. The contract is therefore weaker than OCR:

  Visibility is guaranteed **only** along event happens-before. Any two
  accesses to the same DB that are not ordered by happens-before, where at
  least one writes, make the program's outcome undefined — concurrent writes
  may be lost entirely (whole-DB write-back may mask them).

Equivalently: only programs that are data-race-free *at DB granularity* have
defined results (hence DB-DRF). This is in the spirit of Location Consistency
(Gao & Sarkar, 2000) — provably weaker than release consistency yet equivalent
to it for DB-DRF programs — but it is not the LC model or the LC cache
protocol of that paper: conflicts are tracked at DataBlock granularity, not per
location.

.. warning::

   Programs that are valid under the OCR memory model (which legalizes data
   races) can be **invalid** under MRMW and silently produce wrong answers.
   MRMW exists to measure the cost of coherence in benchmarks. Never use it
   as a correctness baseline. The build emits a CMake warning when selected.

MRNEW — Multi-Reader, Node-Exclusive Writer (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Status:* implemented (``eager.c`` / ``lazy.c`` + ``ownership.c``); selected
by ``-DARTS_COHERENCE_PROTOCOL=MRNEW`` (the default). Timing variant selected
by ``ARTS_PROTOCOL_TIMING={EAGER,LAZY}`` (default ``LAZY``).

Grants multi-reader access freely; grants write access to at most one node at
a time via a single-owner lease. Within a node, multiple RW acquirers share a
local buffer, so intra-node writes are multi-writer (hardware cache coherence
keeps them consistent). Between nodes, a remote RW acquire sends an
``OWNERSHIP_REQUEST`` to the current owner; the owner completes its release,
ships the DB payload and transfers the lease to the requester (the new writer
PARKs until the transfer arrives). This serializes inter-node writes without
forbidding concurrent inter-node reads or local multi-writer patterns.

The contract delivered is **exactly the OCR v1.2.0 §1.6 memory model**
(see :ref:`ocr_contract` above). Non-overlapping concurrent writes survive
because intra-node they are hardware-coherent (shared buffer), and inter-node
the single-owner serialization prevents them from occurring concurrently.

MRSW — Multi-Reader, Single-Writer (roadmap)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Status:* not implemented; documentation-only roadmap.

Would grant shared multi-reader access but restrict write access to one writer
at a time *globally* (no two writers anywhere, intra- or inter-node,
concurrently). Reads would observe the last-committed version of the DB
(version-isolated, last-ancestor reads), forbidding competing writes entirely.
This is **stronger than OCR**: the OCR contract legalizes concurrent
overlapping writes; MRSW would forbid them, giving stronger isolation in
exchange for lower concurrency.

MRSW is the spirit of CDAG-style write-exclusion (see
:ref:`cdag_dropped` for why it was not pursued).

RWLOCK — Reader/Writer Mutex (roadmap)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Status:* not implemented; documentation-only roadmap.

Full reader/writer mutual exclusion: granting a write lock excludes all
readers, and granting shared read locks excludes all writers. Deterministic
serialization; useful as a strict correctness baseline. Stronger than MRSW
(no concurrent reads during a write).

SRSW — Single-Reader, Single-Writer (roadmap)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Status:* not implemented; documentation-only roadmap.

Fully serialized: at most one reader or one writer holds the DB at any time.
Strictest admission; maximal isolation; intended as a debug lower bound.

.. _orthogonal_dimensions:

Orthogonal protocol dimensions
-------------------------------

Admission policy is one axis; the following dimensions are orthogonal to it
and to each other. Each can in principle be combined with any admission policy.

Timing
~~~~~~

When are write-backs, ownership transfers, and invalidations performed?

* **EAGER** (release-time) — consistency actions are performed synchronously
  at release: the releasing node writes back modified data and transfers
  ownership before the release completes. Acquirers find the DB ready. Trades
  release-side latency for shorter critical paths at acquire time. Selected
  by ``-DARTS_PROTOCOL_TIMING=EAGER``.
* **LAZY** (acquire-time) — consistency actions are deferred until the next
  acquire or ownership transfer: the releaser is fast (local metadata update
  only); the acquirer pulls the latest data on demand. Fewer messages per
  release, but acquire latency includes the round trip. Default; selected by
  ``-DARTS_PROTOCOL_TIMING=LAZY``.
* **Invalidate-at-write** — a third variant (not implemented) publishes
  invalidations at the moment a write begins rather than at release or acquire.
  This is the CDAG-style publish-at-write-acquire timing and is related to
  MRSW; see :ref:`cdag_dropped`.

Propagation strategy
~~~~~~~~~~~~~~~~~~~~

How are stale replicas evicted and fresh data delivered?

* **Invalidate-and-pull** (current ARTS) — on ownership transfer or acquire,
  stale remote copies receive an invalidation; the acquirer pulls the payload
  from the current owner. Simple ownership chain; one round trip per transfer.
* **Update-and-push** — the releasing node proactively pushes the new value to
  all current readers instead of invalidating them. Eliminates the pull round
  trip for readers, at the cost of sending the payload speculatively.
  Beneficial when the same DB is acquired RO by many readers in succession;
  costly if the push is discarded.

Directory / ownership
~~~~~~~~~~~~~~~~~~~~~

Where does ownership bookkeeping live?

* **Fixed home** (current ARTS) — each DB has a designated home node (set at
  creation; encoded in the GUID rank field). The home holds the authoritative
  directory entry. Lease grants and transfers always go through the home.
  Simple and predictable; home can become a hot-spot for highly shared DBs.
* **Migratory owner** (roadmap) — ownership migrates to the last acquirer via
  pointer forwarding; the home is consulted only on cache miss. Reduces
  home-node traffic for producer-consumer patterns where the DB moves through
  a chain of owners.
* **Hierarchical NUMA/cluster** (roadmap) — directory is partitioned into
  levels (intra-socket, inter-socket, inter-node, inter-cluster), exploiting
  locality to reduce long-distance traffic. Appropriate for systems with deep
  memory hierarchies.

Read consistency
~~~~~~~~~~~~~~~~

What version of the DB's contents does a reader observe?

* **Live-shared** (current ARTS) — the reader observes whatever data is
  present in the DB at acquire time (the installed-version snapshot on remote
  acquire, or the live shared buffer on local acquire). No version isolation:
  unordered concurrent writers can affect what the reader sees.
* **MVCC / versioned snapshot** (roadmap) — the reader receives a
  version-tagged copy that is isolated from subsequent writes; a concurrent
  writer produces a new version without disturbing the reader. Required by
  MRSW's "last-ancestor read" semantics.

Multi-writer resolution
~~~~~~~~~~~~~~~~~~~~~~~

When two writers access the same DB and their accesses are unordered (no
happens-before edge between them), how are their writes reconciled?

* **Node-exclusive serialization** (current MRNEW) — intra-node: concurrent
  hardware-coherent writes to a shared buffer; inter-node: single-owner lease
  serializes concurrent writers one node at a time. The OCR non-overlapping
  rule holds because each inter-node write is serialized; each intra-node write
  is cache-coherent.
* **Lossy whole-DB write-back** (current MRMW) — each release writes the
  entire DB back; the last write-back wins. Concurrent inter-node writes to
  disjoint ranges of the same DB are *not* preserved (violates the OCR
  non-overlapping rule). DB-DRF contract only.
* **Non-lossy merge** (roadmap) — diff-based or twin-based merging reconciles
  concurrent writers' changes. This is the approach of TreadMarks (Keleher
  et al., LRC / diff-and-patch) and Entry Consistency (Bershad et al.): each
  writer's modifications are recorded as diffs, and at synchronization time
  diffs from all writers are merged. Achieves the full OCR non-overlapping
  rule while permitting true inter-node multi-writer access; the admission
  policy would be MRMW in name but non-lossy (a genuinely OCR-conformant
  multi-writer protocol).

Per-DB-kind contracts
---------------------

The global axes govern regular DRAM DBs (``ARTS_DB``). The other storage
kinds carry their own fixed, documented semantics regardless of the global
axes: ``ARTS_DB_PIN`` and ``ARTS_DB_GPU_PIN`` perform no DB-level coherence
(the application orders accesses with events); ``ARTS_DB_GPU`` allows
concurrent per-device replicas merged by reduction at release; ``ARTS_DB_CXL``
relies on hardware cache coherence intra-node and application-driven event
ordering across nodes. All of these are DB-DRF-style contracts: order your
conflicting accesses with events.

.. _cdag_dropped:

Considered and dropped: CDAG
-----------------------------

*Cache DAG Consistency* (Landwehr et al., 2017) is a variant of DAG
consistency (Blumofe et al., 1996) in which a task observes the write of the
immediately preceding ancestor version of a shared location ("last-ancestor"
read). It effectively maps to the MRSW admission policy plus MVCC reads:
a single global writer per DataBlock at a time, and readers observe the
last-committed ancestor version rather than live data.

CDAG was considered as a coherence option for ARTS and was dropped for the
following reasons:

1. **Small implementation delta over MRNEW.** MRNEW's existing single-owner
   lease already serializes inter-node writers. The additional work for CDAG
   semantics — adding version tags, MVCC reads, and the "last-ancestor"
   copy at acquire time — is non-trivial but marginal relative to the
   machinery already in place. Implementing CDAG would be a refinement of
   MRSW, not a separate protocol.
2. **Weaker parallelism.** CDAG / MRSW forbids competing concurrent writes
   entirely, reducing parallelism for programs that exploit the OCR
   non-overlapping data-race rule.
3. **Deferred.** If MRSW is later pursued, CDAG's "last-ancestor" read
   semantics can be revisited as part of that work.

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

**Current → new axis (this release):** the ``ARTS_MEMORY_MODEL`` build axis
(``OCR`` / ``RELAXED``) is retired. A memory model is a consequence of the
protocol, not an independent knob. The new axis is
``ARTS_COHERENCE_PROTOCOL`` ∈ {``MRNEW``, ``MRMW``} (admission policy) plus
``ARTS_PROTOCOL_TIMING`` ∈ {``EAGER``, ``LAZY``} (timing; MRNEW only). The
mapping is exact and behavior-preserving:

* ``OCR + EAGER`` → ``MRNEW + EAGER``
* ``OCR + LAZY``  → ``MRNEW + LAZY`` (default)
* ``RELAXED``     → ``MRMW``

Using the old ``-DARTS_MEMORY_MODEL=`` option causes a CMake ``FATAL_ERROR``
with the mapping message above, so existing build scripts are caught at
configure time.

References
----------

* OCR working group, *The Open Community Runtime Interface*, v1.2.0, 2016 — §1.6.
* J. Dokulil, *Consistency model for runtime objects in the Open Community Runtime*, J. Supercomputing, 2018.
* T. Landwehr et al., *Designing Scalable Distributed Memory Models*, SC 2017 (Cache DAG Consistency).
* G. R. Gao, V. Sarkar, *Location Consistency — A New Memory Model and Cache Consistency Protocol*, IEEE TC 49(8), 2000.
* K. Gharachorloo et al., *Memory Consistency and Event Ordering in Scalable Shared-Memory Multiprocessors*, ISCA 1990.
* P. Keleher et al., *Lazy Release Consistency for Software Distributed Shared Memory*, ISCA 1992.
* B. Bershad, M. Zekauskas, W. Sawdon, *The Midway Distributed Shared Memory System*, COMPCON 1993.
* L. Iftode, J. P. Singh, K. Li, *Scope Consistency: A Bridge between Release Consistency and Entry Consistency*, SPAA 1996.
* R. Blumofe et al., *DAG-Consistent Distributed Shared Memory*, IPPS 1996.
* S. Adve, M. Hill, *Weak Ordering — A New Definition*, ISCA 1990.
