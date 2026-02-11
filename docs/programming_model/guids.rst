GUIDs
=====

Every runtime object in ARTS — EDTs, DataBlocks, events, epochs — is
identified by a 64-bit **Globally Unique Identifier** (GUID).

.. contents:: On this page
   :local:
   :depth: 2

Bitfield Layout
---------------

.. code-block:: text

   ┌──────────┬──────────────────┬────────────────────────────────────────┐
   │ type (8) │   rank (16)      │              key (40)                  │
   └──────────┴──────────────────┴────────────────────────────────────────┘
    Bits 63–56    Bits 55–40                Bits 39–0

- **type** (8 bits): Object kind from :c:enum:`arts_type_t`.
- **rank** (16 bits): Node that owns the object (up to 65 535 nodes).
- **key** (40 bits): Node-local unique key (~1 trillion per node).

Inspecting GUIDs
----------------

.. code-block:: c

   arts_guid_t guid = ...;

   unsigned int rank = arts_guid_get_rank(guid);
   unsigned int type = arts_guid_get_type(guid);
   bool local = arts_guid_is_local(guid);

GUID Ranges
-----------

For bulk allocation, use GUID ranges to reserve a contiguous block of
keys:

.. code-block:: c

   arts_guid_range_t *range =
       arts_guid_range_create(ARTS_DB, 100, target_node);

   while (arts_guid_range_has_next(range)) {
       arts_guid_t g = arts_guid_range_get(range);
       arts_guid_range_next(range);
       /* use g ... */
   }

Round-Robin Allocation
~~~~~~~~~~~~~~~~~~~~~~

To distribute GUIDs evenly across nodes:

.. code-block:: c

   arts_guid_range_t *range =
       arts_guid_reserve_round_robin(ARTS_DB, total_count);

``NULL_GUID``
-------------

The sentinel ``NULL_GUID`` (``0x0``) represents an absent or invalid
GUID.  Always check against it before dereferencing:

.. code-block:: c

   if (guid != NULL_GUID) {
       /* safe to use */
   }
