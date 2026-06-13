DataBlocks
==========

DataBlocks (DBs) are ARTS's primary data abstraction — explicit data
objects identified by GUIDs and shared between EDTs.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

A DataBlock is a contiguous, typed memory region with a globally unique
identifier.  DataBlocks decouple *data identity* from *data location*:
any node can reference a DB by GUID, and the runtime handles data
movement transparently.

DB Types
--------

All DataBlocks use a single GUID kind (``ARTS_GUID_DB``).  The storage
**subtype** is set at creation time via :c:enum:`arts_db_types_t` and
determines the allocation strategy and whether the runtime manages
coherence for the DB:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Subtype
     - Semantics
   * - ``ARTS_DB``
     - Regular DRAM, runtime-coherent.  The default type for most use
       cases.  Its consistency contract is the build-time memory model;
       the coherence protocol implementing it is also selected at build
       time (see :ref:`memory_model`).
   * - ``ARTS_DB_PIN``
     - Regular DRAM, node-pinned, no DB-level coherence.  Only directly
       accessible on the creating node; the application orders accesses
       via events.
   * - ``ARTS_DB_CXL``
     - CXL shared memory.  Hardware cache coherence intra-node;
       application-ordered (DB-DRF) across nodes.  Compiled only with
       ``ARTS_USE_CXL=ON``.
   * - ``ARTS_DB_GPU``
     - GPU staging.  Concurrent per-device replicas merged by reduction
       at release (DB-DRF style).
   * - ``ARTS_DB_GPU_PIN``
     - GPU staging (host pinned + per-device replica), no DB-level
       coherence.

The ``ARTS_DB_DEFAULT`` macro is the experiment-wide subtype every
benchmark uses; CMake (``ARTS_DEFAULT_DB_KIND``) decides whether it
expands to ``ARTS_DB`` or ``ARTS_DB_CXL``.

Access Modes
------------

The access mode is set per-dependency at :c:func:`arts_add_dependence`
time, not at creation.  **Access modes are replication directives, not
synchronization**: they tell the runtime whether it may replicate the DB
for a dependency, and nothing more.  Ordering between conflicting
accesses is exclusively the job of events (see :doc:`events`).

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Semantics
   * - ``DB_MODE_RO``
     - Read-Only.  The runtime may replicate the DB to the reader; the
       replica is never written back.  **No isolation is promised**: if
       a writer runs concurrently (i.e. not ordered against the reader
       by events), its writes may become visible to the reader.
   * - ``DB_MODE_RW``
     - Read-Write.  Ownership-managed: per-node exclusive access with
       writeback (OCR RW semantics).  Two RW accesses not ordered by
       events may still interleave per the memory model — RW grants
       exclusivity of the replica, not a happens-before edge.
   * - ``DB_MODE_NULL``
     - Placeholder / pure control dependency.  No data is delivered.
   * - ``DB_MODE_VAL``
     - The dependency carries a raw ``uint64`` value, not a GUID.

These four are the complete public set.  (Values at or above
``DB_MODE_INTERNAL_BASE`` are runtime-internal and never appear in
user-facing :c:func:`arts_add_dependence` calls.)

Consistency
-----------

For regular ``ARTS_DB`` DataBlocks, which values a read may return is
governed by the build-time configuration:

* ``ARTS_MEMORY_MODEL`` selects the **contract** (``OCR`` or
  ``RELAXED``);
* ``ARTS_COHERENCE_PROTOCOL`` selects the **protocol** implementing the
  OCR contract (``EAGER`` or ``LAZY``).

See :ref:`memory_model` for the normative definition of both axes.  The
other subtypes (``ARTS_DB_PIN``, ``ARTS_DB_CXL``, ``ARTS_DB_GPU``,
``ARTS_DB_GPU_PIN``) carry no DB-level runtime coherence; the
application orders conflicting accesses with events (DB-DRF).

Creating a DataBlock
--------------------

.. code-block:: c

   /* Create a 1024-byte ARTS_DB on this node.  The creator auto-acquires
      RW access; *addr points at the writable payload. */
   void *addr;
   arts_guid_t db_guid =
       arts_db_create(&addr, 1024, ARTS_DB, ARTS_DB_PROP_NONE, NULL);

   /* Create on a specific node */
   arts_db_hint_t h = ARTS_DB_HINT_DEFAULTS;
   h.rank = target_node;
   arts_guid_t db_guid =
       arts_db_create(&addr, 1024, ARTS_DB, ARTS_DB_PROP_NONE, &h);

   /* Create a node-pinned DB with a pre-reserved (local) GUID */
   arts_guid_t guid = arts_guid_reserve(ARTS_GUID_DB, arts_get_current_rank());
   arts_db_create_with_guid(guid, 1024, ARTS_DB_PIN, ARTS_DB_PROP_NONE, NULL);

   /* Create a DB with initial data: write through the returned pointer */
   int data[] = {1, 2, 3};
   void *ptr;
   arts_guid_t g =
       arts_db_create(&ptr, sizeof(data), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
   memcpy(ptr, data, sizeof(data));

Pass ``ARTS_DB_PROP_NO_ACQUIRE`` in ``flags`` to skip the creator's
automatic RW acquire (``*addr`` is then set to ``NULL`` and the first
consumer EDT performs a normal acquire).

Accessing Data
--------------

DB data is read and written through EDT dependencies: the runtime
acquires each DB dependency before the EDT runs and releases it when the
EDT returns.

.. code-block:: c

   void process(uint32_t paramc, const uint64_t *paramv,
                uint32_t depc, arts_edt_dep_t depv[]) {
       int *data = (int *)depv[0].ptr;   /* payload pointer */
       /* ... use data ... */
   }

   /* Wire a read dependency: satisfies the EDT's slot 0 with the DB */
   arts_add_dependence(db_guid, edt_guid, 0, DB_MODE_RO);

For write access, wire the dependency with ``DB_MODE_RW``; the EDT's
writes are published when the runtime releases the DB at EDT
completion.

Release and Lifetime
--------------------

A creating EDT holds RW access until it returns.  To publish writes
early — required when the EDT blocks inside its body (e.g. via
:c:func:`arts_event_wait`) — release explicitly:

.. code-block:: c

   arts_db_release(db_guid, DB_MODE_RW);  /* created/written DBs */

:c:func:`arts_db_destroy` destroys all copies of a DB system-wide; any
acquire held by the calling EDT is implicitly released first, and actual
deallocation is deferred until outstanding references drain:

.. code-block:: c

   arts_db_destroy(db_guid);
