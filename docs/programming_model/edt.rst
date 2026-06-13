Event-Driven Tasks (EDT)
========================

An EDT is the fundamental unit of computation in ARTS.  EDTs are
lightweight, non-preemptive tasks that execute once all their
dependencies have been satisfied.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

An EDT consists of:

- A **function pointer** — the C function to execute.
- **Static parameters** (``paramv``) — up to ``paramc`` 64-bit values
  baked into the EDT at creation time.
- **Dependencies** (``depv``) — up to ``depc`` slots that receive
  DataBlocks or values at runtime.

When every dependency slot has been signaled, the runtime schedules the
EDT for execution on the target node.

EDT Function Signature
----------------------

.. code-block:: c

   void my_edt(uint32_t paramc, const uint64_t *paramv,
               uint32_t depc, arts_edt_dep_t depv[]);

- ``paramc`` / ``paramv`` — static parameters passed at creation time.
- ``depc`` / ``depv`` — satisfied dependencies.  Each slot contains:

  - ``depv[i].guid`` — the GUID of the dependency (or an encoded value).
  - ``depv[i].ptr``  — pointer to the DataBlock payload (for DB deps).

Creating an EDT
---------------

.. code-block:: c

   arts_edt_hint_t h = ARTS_EDT_HINT_DEFAULTS;
   h.rank = target_node;                 /* placement */
   arts_guid_t guid = arts_edt_create(
       my_edt,                /* function pointer      */
       paramc, paramv,        /* static params         */
       depc,                  /* dependency count      */
       &h                     /* hint (NULL = defaults) */
   );

The returned GUID identifies the EDT.  The ``hint`` parameter carries
all optional creation features in :c:type:`arts_edt_hint_t`: pass
``NULL`` for the defaults (current node, auto-allocated GUID, inherit
the ambient finish scope), or set ``.rank`` to place the EDT on another
node.  If ``depc > 0``, the EDT will not run until all slots are
signaled.

Signaling Dependencies
----------------------

Wire a DataBlock or value into a dependency slot:

.. code-block:: c

   /* Wire a DataBlock into slot 0 (read-write) */
   arts_add_dependence(db_guid, edt_guid, 0, DB_MODE_RW);

   /* Wire a raw 64-bit value into slot 1 */
   arts_add_dependence((arts_guid_t)42, edt_guid, 1, DB_MODE_VAL);

   /* Satisfy a slot without data (pure control dependency) */
   arts_add_dependence(NULL_GUID, edt_guid, 2, DB_MODE_NULL);

If ``depc == 0``, the EDT fires immediately after creation.

Creation Hint Fields
--------------------

The legacy ``arts_edt_create_*`` variants are collapsed into the single
:c:func:`arts_edt_create` entry point; optional features ride in
:c:type:`arts_edt_hint_t`:

- ``rank`` — home node (``ARTS_HINT_CURRENT_RANK`` = current node).
- ``guid`` — create at a pre-reserved GUID (``NULL_GUID`` =
  auto-allocate; when set, the GUID's rank overrides ``rank``).
- ``finish_event`` — join a finish scope for termination detection
  (see :doc:`finish_events`).
- ``output_event`` — per-EDT result channel satisfied after the EDT's
  DBs are released (payload set via :c:func:`arts_edt_set_result`).
- ``edt_id`` — compiler-assigned profiling identifier.

See :doc:`/api/public_api` for the full list.

Execution Guarantees
--------------------

- An EDT runs **exactly once** (or not at all if destroyed before firing).
- The runtime does **not** preempt a running EDT.
- EDTs are **node-affine**: the function executes on the node specified
  at creation time.
- Static parameters are copied at creation; the runtime owns the copy.
