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

.. _labeled-guid-reuse:

Reusing a labeled GUID: create replaces, and that is a deviation
----------------------------------------------------------------

A create that finds its labeled GUID already occupied **replaces** the
object in that slot; the displaced one is released.  ARTS implements
neither of the OCR standard's checked variants:

``GUID_PROP_CHECK``
    the standard returns an error code when the GUID already exists.

``GUID_PROP_BLOCK``
    the standard blocks the create until the GUID can be re-created.

Both properties are accepted and both behave as the standard's
*unchecked* default, whose outcome the standard itself declares
undefined ("potentially create the same object multiple times leading
to undefined behavior").

The consequence worth knowing is narrower than "replace is unsafe".
Creating the same label from several ranks at once, where one creator
wins and the others go on to use the winner's object, works: every
creator installs an equivalent object and the last one stands.  What
does **not** work is *reusing* a label across a lifetime boundary:

.. code-block:: c

   arts_db_destroy(g);                  /* generation A */
   arts_db_create_with_guid(g, ...);    /* generation B, same label */

The destroy and the create are independent messages and ARTS gives no
ordering contract between them, so the destroy can be applied after the
create.  Nothing distinguishes the two generations — the route table
carries no per-slot generation stamp — so the destroy meant for A tears
down B, and any protocol operation still in flight for A is applied to
B.

Programs that need a label per unit of work should therefore derive a
**distinct index per unit** (``ocrGuidFromIndex`` over a range sized to
the work) rather than cycling a small range with an index that wraps.
Every application in the benchmark suite already does this; the pattern
that does not is ``prodcon``, which is marked ``unsupported`` in the
driver's catalog for exactly this reason.

``tests/ocr/ooo_gen_crossgen_drop.c`` drives the reuse pattern and is
deliberately left unregistered: it is the executable statement of this
limitation, ready to run if the semantics are ever implemented.
