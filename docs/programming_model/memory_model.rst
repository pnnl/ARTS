.. _memory_model:

Memory Model
============

.. contents:: On this page
   :local:
   :depth: 2

This page is **normative**. It defines the contract every ARTS build provides,
separates that contract (the *memory model*) from the mechanisms that implement
it (the *coherence protocols*), and defines the build axes that select them.

Model versus protocol
---------------------

A *memory (consistency) model* is a contract: it defines which values a read
may legally return, and therefore which programs are valid. A *coherence
protocol* is an implementation mechanism — replica management, ownership
transfer, invalidation, write-back — that must *obey* the model. ARTS keeps
these namespaces strictly separate:

* ``ARTS_MEMORY_MODEL`` selects the **contract**: ``OCR`` (default) or
  ``RELAXED``.
* ``ARTS_COHERENCE_PROTOCOL`` selects the **implementation** of the OCR
  contract: ``EAGER`` or ``LAZY`` (default). It is meaningful only when
  ``ARTS_MEMORY_MODEL=OCR``.

The OCR memory model (the contract)
-----------------------------------

The contract is the memory model of the *Open Community Runtime Interface*,
version 1.2.0, §1.6, with the ``synchronized-with`` relation completed as in
Dokulil, "Consistency model for runtime objects in the Open Community
Runtime", J. Supercomputing 75:2725–2760, 2018 (doi:10.1007/s11227-018-2681-2),
which this document incorporates by reference. The version pin matters: the
OCR specification itself reserves the right to relax its race rules in future
versions.

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
---------------------------------------

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
-------------------------------------------------------------

Every conforming implementation (both ARTS protocols, and likewise XSOCR and
OCR-Vx) is incidentally *stronger* than the contract in places — e.g. current
ARTS protocols serialize unordered inter-node RW sessions via single-owner
leases, and remote RO acquires observe an installed-version copy. These
strengthenings are **never** part of the contract. A program that depends on
them is *non-portable* (it may break on another conforming implementation or
protocol), not invalid. Keeping the contract maximally relaxed preserves
implementation freedom — including future multi-writer merge protocols.

The two protocols implementing the OCR contract:

* ``EAGER`` — write-back/ownership actions performed synchronously at release
  (release-side round trips).
* ``LAZY`` — consistency actions deferred to the next acquire/transfer
  (acquire-side pull, fewer messages). Default.

A third, future axis (admission policy) is reserved: ``MRMW`` (current
behavior), ``MRSW``, ``RWLOCK``, ``SRSW`` — increasingly strict concurrent-
access grants, all conforming, intended for evaluating the parallelism/traffic
trade-off.

The RELAXED model (DB-DRF) — evaluation only
--------------------------------------------

``ARTS_MEMORY_MODEL=RELAXED`` selects a **different, weaker contract**, not
another protocol:

  Visibility is guaranteed **only** along event happens-before. Any two
  accesses to the same DB that are not ordered by happens-before, where at
  least one writes, make the program's outcome undefined — concurrent writes
  may be lost entirely (whole-DB write-back may mask them).

Equivalently: only programs that are data-race-free *at DB granularity* have
defined results (hence the name DB-DRF). This is in the spirit of Location
Consistency (Gao & Sarkar, 2000) — which is provably weaker than release
consistency yet equivalent to it for data-race-free programs — but it is not
the LC model or the LC cache protocol of that paper: conflicts are tracked at
DataBlock granularity, not per location.

.. warning::

   Programs that are valid under the OCR memory model (which legalizes data
   races) can be **invalid** under RELAXED and silently produce wrong answers.
   RELAXED exists to measure the cost of coherence in benchmarks. Never use it
   as a correctness baseline. The build emits a CMake warning when selected.

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

Historical vocabulary
---------------------

Earlier ARTS releases named the build axis ``ARTS_MEMORY_MODEL={RC,LRC,LC}``.
``RC``/``LRC`` were not distinct models — they were the eager/lazy protocols
of the one OCR contract — and ``LC`` was not the literature's Location
Consistency. The terms are retired; the mapping is
``RC→OCR+EAGER``, ``LRC→OCR+LAZY``, ``LC→RELAXED``.

References
----------

* OCR working group, *The Open Community Runtime Interface*, v1.2.0, 2016 — §1.6.
* J. Dokulil, *Consistency model for runtime objects in the Open Community Runtime*, J. Supercomputing, 2018.
* G. R. Gao, V. Sarkar, *Location Consistency — A New Memory Model and Cache Consistency Protocol*, IEEE TC 49(8), 2000.
* K. Gharachorloo et al., *Memory Consistency and Event Ordering in Scalable Shared-Memory Multiprocessors*, ISCA 1990.
* P. Keleher et al., *Lazy Release Consistency for Software Distributed Shared Memory*, ISCA 1992.
* B. Bershad, M. Zekauskas, W. Sawdon, *The Midway Distributed Shared Memory System*, COMPCON 1993.
* L. Iftode, J. P. Singh, K. Li, *Scope Consistency: A Bridge between Release Consistency and Entry Consistency*, SPAA 1996.
* R. Blumofe et al., *DAG-Consistent Distributed Shared Memory*, IPPS 1996.
* S. Adve, M. Hill, *Weak Ordering — A New Definition*, ISCA 1990.
