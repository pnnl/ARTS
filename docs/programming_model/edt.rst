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

   arts_guid_t guid = arts_edt_create(
       my_edt,                /* function pointer      */
       paramc, paramv,        /* static params         */
       depc,                  /* dependency count      */
       &(arts_hint_t){.route = target_node}  /* placement hint */
   );

The returned GUID identifies the EDT.  The ``hint`` parameter controls
placement: pass ``NULL`` for the current node, or a pointer to an
``arts_hint_t`` with the ``.route`` field set to the target node rank.
If ``depc > 0``, the EDT will not run until all slots are signaled.

Signaling Dependencies
----------------------

Wire a DataBlock or value into a dependency slot:

.. code-block:: c

   /* Signal a DataBlock into slot 0 (exclusive write) */
   arts_signal_edt(edt_guid, 0, db_guid, ARTS_MODE_EW);

   /* Signal a raw 64-bit value into slot 1 */
   arts_signal_edt_value(edt_guid, 1, 42);

   /* Signal without data (just satisfy the slot) */
   arts_signal_edt_null(edt_guid, 2);

If ``depc == 0``, the EDT fires immediately after creation.

Convenience Variants
--------------------

ARTS offers several ``arts_edt_create_*`` variants:

- :c:func:`arts_edt_create_dep` — create EDT and pre-signal
  dependencies in one call.
- :c:func:`arts_edt_create_with_guid` — create at a pre-reserved GUID.
- :c:func:`arts_edt_create_with_epoch` — associate with an epoch for
  termination detection.

See :doc:`/api/public_api` for the full list.

Execution Guarantees
--------------------

- An EDT runs **exactly once** (or not at all if destroyed before firing).
- The runtime does **not** preempt a running EDT.
- EDTs are **node-affine**: the function executes on the node specified
  at creation time.
- Static parameters are copied at creation; the runtime owns the copy.
