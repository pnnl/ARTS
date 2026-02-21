DataBlocks
==========

DataBlocks (DBs) are ARTS's primary data abstraction — explicit data
objects identified by GUIDs and managed through the CDAG memory model.

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

The DB type is set at creation time and determines allocation strategy
and coherence behavior:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Type
     - Semantics
   * - ``ARTS_DB``
     - Distributed DataBlock (CDAG-managed).  The default type for most
       use cases.  Supports read sharing and write coherence across nodes.
   * - ``ARTS_DB_LOCAL``
     - Node-resident DataBlock (no CDAG).  Only directly accessible on
       the creating node.  Use put/get for remote interaction.
   * - ``ARTS_DB_GPU``
     - GPU-pinned DataBlock (CDAG-managed).  Allocated via GPU pinned
       memory for efficient host-device transfers.
   * - ``ARTS_DB_LC``
     - Locality-class DataBlock (CPU-GPU coherence).  Double-sized
       allocation with shadow copy for CPU-GPU data movement.

Access Modes
------------

The access mode is set per-dependency at :c:func:`arts_record_dep` time,
not at creation:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Semantics
   * - ``ARTS_MODE_RO``
     - Read-Only.  Shared readers, no writeback.
   * - ``ARTS_MODE_EW``
     - Exclusive Write.  Single writer with frontier progression
       and latch decrement on release.

Creating a DataBlock
--------------------

.. code-block:: c

   /* Create a 1024-byte DB on this node (mode-less; access mode set later) */
   void *addr;
   arts_guid_t db_guid = arts_db_create(&addr, 1024, NULL);

   /* Create on a specific node */
   arts_guid_t db_guid = arts_db_create(&addr, 1024,
                                        &(arts_hint_t){.route = target_node});

   /* Create a LOCAL (node-resident) DB */
   arts_guid_t guid = arts_guid_reserve(ARTS_DB_LOCAL, target_node);
   arts_db_create_with_guid(guid, 1024, NULL);

Writing Data
------------

.. code-block:: c

   /* Put data into a DB (any node, any type) */
   int data[] = {1, 2, 3};
   arts_put_in_db(data, db_guid, 0, sizeof(data));

Reading Data
------------

The simplest way to read DB data is through an EDT dependency:

.. code-block:: c

   void process(uint32_t paramc, const uint64_t *paramv,
                uint32_t depc, arts_edt_dep_t depv[]) {
       int *data = (int *)depv[0].ptr;   /* payload pointer */
       /* ... use data ... */
   }

   /* Record a read dependency and signal the EDT */
   arts_record_dep(db_guid, edt_guid, 0, ARTS_MODE_RO);

For explicit reads outside an EDT dependency:

.. code-block:: c

   arts_get_from_db(db_guid, edt_guid, slot);

CDAG Memory Model
-----------------

For ``ARTS_DB`` DataBlocks, ARTS uses a Canonical-owner DAG (CDAG)
model:

1. Each DB has a single **canonical owner** node (the creating node).
2. Remote reads are cached in the routing table.
3. Writes flow through the owner; the owner invalidates cached copies.

This gives sequential consistency for single-writer patterns while
allowing efficient read sharing.

Array DataBlocks
----------------

For large distributed arrays, ARTS provides **Array DBs** — a collection
of ``ARTS_DB_LOCAL`` DataBlocks distributed across nodes:

.. code-block:: c

   arts_guid_t array_guid = arts_new_array_db(
       sizeof(int),   /* element size      */
       1024,          /* elements per block */
       num_nodes      /* number of blocks  */
   );

See :doc:`/api/public_api` for the full Array DB API.
