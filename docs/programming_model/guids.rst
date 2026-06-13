GUIDs
=====

Every runtime object in ARTS — EDTs, DataBlocks, events — is
identified by a 64-bit **Globally Unique Identifier** (GUID).

.. contents:: On this page
   :local:
   :depth: 2

Bitfield Layout
---------------

.. code-block:: text

   ┌──────────┬──────────────────┬────────────────────────────────────────┐
   │ kind (2) │   rank (14)      │              key (48)                  │
   └──────────┴──────────────────┴────────────────────────────────────────┘
    Bits 63–62    Bits 61–48                   Bits 47–0

- **kind** (2 bits): Object kind from :c:enum:`arts_guid_kind_t`
  (``ARTS_GUID_DB`` / ``ARTS_GUID_EVENT`` / ``ARTS_GUID_EDT``; the
  all-zero pattern is the reserved/NULL sentinel).
- **rank** (14 bits): Node that owns the object (up to 16 384 nodes).
- **key** (48 bits): Node-local unique key, stored in the
  least-significant bits so that GUID-range arithmetic reduces to plain
  integer addition.

Inspecting GUIDs
----------------

.. code-block:: c

   arts_guid_t guid = ...;

   unsigned int rank = arts_guid_get_rank(guid);
   arts_guid_kind_t kind = arts_guid_get_kind(guid);
   bool local = arts_guid_is_local(guid);

GUID Ranges
-----------

For bulk allocation, use GUID ranges to reserve a contiguous block of
keys:

.. code-block:: c

   arts_guid_t start =
       arts_guid_reserve_range(ARTS_GUID_DB, 100, target_node);

   for (unsigned int i = 0; i < 100; i++) {
       arts_guid_t g = arts_guid_from_index(start, i);
       /* use g ... */
   }

Round-Robin Allocation
~~~~~~~~~~~~~~~~~~~~~~

To distribute the homes of a range evenly across nodes, pass the
``ARTS_HINT_ROUND_ROBIN`` sentinel rank (``home = idx % nrank``;
broadcast the range GUID to every rank that derives children):

.. code-block:: c

   arts_guid_t start =
       arts_guid_reserve_range(ARTS_GUID_DB, total_count,
                               ARTS_HINT_ROUND_ROBIN);

``NULL_GUID``
-------------

The sentinel ``NULL_GUID`` (``0x0``) represents an absent or invalid
GUID.  Always check against it before dereferencing:

.. code-block:: c

   if (guid != NULL_GUID) {
       /* safe to use */
   }
