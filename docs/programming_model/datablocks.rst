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

Access Modes
------------

The access mode is specified at creation time and controls the coherence
protocol:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Semantics
   * - ``ARTS_DB_READ``
     - Write-once, read-many.  Default mode for most use cases.  The
       runtime caches reads and aggregates remote requests.
   * - ``ARTS_DB_WRITE``
     - Exclusive write access.  Set at dependency-registration time
       via :c:func:`arts_record_dep`.
   * - ``ARTS_DB_PIN``
     - Pinned (node-local).  Bypasses the CDAG model; only accessible
       on the creating node.  Use put/get for remote interaction.
   * - ``ARTS_DB_ONCE``
     - Single-use.  Automatically freed after the first acquire.
   * - ``ARTS_DB_ONCE_LOCAL``
     - Like ``ARTS_DB_ONCE`` but guarantees co-location with the
       acquiring EDT.

Creating a DataBlock
--------------------

.. code-block:: c

   /* Create a 1024-byte read-mode DB on this node */
   arts_guid_t db_guid = arts_db_create(1024, ARTS_DB_READ);

   /* Create on a specific node */
   arts_guid_t db_guid = arts_db_create_remote(1024, target_node, ARTS_DB_PIN);

Writing Data
------------

.. code-block:: c

   /* Put data into a DB (any node, any mode) */
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

   /* Signal the DB into the EDT's slot 0 */
   arts_signal_edt(edt_guid, 0, db_guid);

For explicit reads outside an EDT dependency:

.. code-block:: c

   arts_get_from_db(db_guid, edt_guid, slot);

CDAG Memory Model
-----------------

For ``ARTS_DB_READ`` DataBlocks, ARTS uses a Canonical-owner DAG (CDAG)
model:

1. Each DB has a single **canonical owner** node (the creating node, or
   the node after a :c:func:`arts_db_move`).
2. Remote reads are cached in the routing table.
3. Writes flow through the owner; the owner invalidates cached copies.

This gives sequential consistency for single-writer patterns while
allowing efficient read sharing.

Array DataBlocks
----------------

For large distributed arrays, ARTS provides **Array DBs** — a collection
of DataBlocks distributed across nodes:

.. code-block:: c

   arts_guid_t array_guid = arts_new_array_db(
       sizeof(int),   /* element size      */
       1024,          /* elements per block */
       num_nodes      /* number of blocks  */
   );

See :doc:`/api/public_api` for the full Array DB API.
