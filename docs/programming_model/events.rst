Events
======

Events are ARTS's synchronization primitive.  They serve two roles at
once: **control flow** (an EDT runs only after the events it depends on
have fired) and **data plumbing** (a fired event delivers a data GUID —
typically a DataBlock — into each dependent's slot).

.. contents:: On this page
   :local:
   :depth: 2

Creating Events
---------------

All events are created through a single hint-driven entry point:

.. code-block:: c

   arts_guid_t arts_event_create(const arts_event_hint_t *hint);

Passing ``NULL`` is equivalent to ``ARTS_EVENT_HINT_DEFAULTS`` (a latch
event with an initial count of 1 — single satisfy fires).  The hint
selects between the two event kinds and their parameters:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Hint field
     - Meaning
   * - ``rank``
     - Home rank of the event (``ARTS_HINT_CURRENT_RANK`` = current
       node, the default).
   * - ``latch``
     - Initial latch counter (default 1).  Latch events only.
   * - ``channel``
     - ``true`` selects a CHANNEL event (multi-fire FIFO, see below).
   * - ``guid``
     - Pre-reserved GUID from :c:func:`arts_guid_reserve`
       (``NULL_GUID`` = auto-allocate).  The GUID's rank field then
       overrides ``rank``.
   * - ``check``
     - ``true`` makes a create at an already-occupied GUID fail
       (return ``NULL_GUID``) instead of replacing it — OCR
       rendezvous semantics for labeled GUIDs.
   * - ``finish``
     - ``true`` selects a FINISH event (see below); all other fields
       are ignored.

Convenience macros build the common hints:
``ARTS_EVENT_HINT_LATCH(counter_init)``, ``ARTS_EVENT_HINT_DEFAULTS``,
``ARTS_EVENT_HINT_CHANNEL``, and ``ARTS_EVENT_HINT_FINISH``.

Latch Events (Fire-and-Linger)
------------------------------

A latch event maintains an integer counter.  Satisfies on the
``ARTS_EVENT_LATCH_DECR_SLOT`` decrement it; satisfies on the
``ARTS_EVENT_LATCH_INCR_SLOT`` increment it.  When the counter reaches
zero the event **fires**, delivering its data GUID to every registered
dependent.

Firing is a pure state transition — *fire is not destroy*.  A fired
event **lingers**: any :c:func:`arts_add_dependence` registered after
the fire is satisfied immediately from the stored fire data, until the
event is explicitly removed with :c:func:`arts_event_destroy`.  A
satisfy arriving past the fire is silently absorbed.

The single-fire OCR event flavors (ONCE, IDEMPOTENT, STICKY, COUNTED)
are all subsumed by this unified fire-and-linger + silent-over-satisfy
model; the macros ``ARTS_EVENT_HINT_ONCE``,
``ARTS_EVENT_HINT_IDEMPOTENT``, ``ARTS_EVENT_HINT_STICKY``, and
``ARTS_EVENT_HINT_COUNTED(nb_deps)`` are kept as source-compatibility
aliases of ``ARTS_EVENT_HINT_LATCH(1)``.

Signaling (Satisfy)
-------------------

.. code-block:: c

   /* Common case: decrement, optionally carrying a data GUID. */
   void arts_event_satisfy(arts_guid_t event_guid, arts_guid_t data_guid);

   /* Explicit slot: ARTS_EVENT_LATCH_DECR_SLOT or
    * ARTS_EVENT_LATCH_INCR_SLOT. */
   void arts_event_satisfy_slot(arts_guid_t event_guid,
                                arts_guid_t data_guid, uint32_t slot);

:c:func:`arts_event_satisfy` is the OCR-aligned convenience wrapper for
the DECR slot; call :c:func:`arts_event_satisfy_slot` directly only
when you need INCR.  Cross-rank calls are forwarded to the event's home
rank.  There is no public API to inspect fire state — observe a fire by
chaining a dependent EDT off the event.

Wiring Dependencies
-------------------

.. code-block:: c

   void arts_add_dependence(arts_guid_t source, arts_guid_t destination,
                            uint32_t slot, arts_db_access_mode_t mode);

:c:func:`arts_add_dependence` is the OCR-standard dispatcher: with an
event as ``source``, it registers ``destination`` (an EDT or another
event) as a dependent; when the event fires, its data lands in
``destination``'s ``slot`` with the given access mode (``DB_MODE_RO``,
``DB_MODE_RW``, or ``DB_MODE_NULL`` for pure control dependencies).
The entity-specific form :c:func:`arts_event_add_dependence` takes the
same arguments and is what the dispatcher calls for an event source.

Channel Events
--------------

A CHANNEL event (``ARTS_EVENT_HINT_CHANNEL``) is persistent and
multi-fire: it pairs each satisfy with exactly one
:c:func:`arts_add_dependence`, in FIFO arrival order.  The *g*-th
satisfy delivers its data GUID to the *g*-th registered dependent — one
consumer per generation.  Producer and consumer sides may run ahead of
each other in either direction; the runtime queues the unmatched side.

.. code-block:: c

   arts_event_hint_t h = ARTS_EVENT_HINT_CHANNEL;
   arts_guid_t ch = arts_event_create(&h);

   arts_event_satisfy(ch, data_db);          /* generation g produced  */
   arts_add_dependence(ch, edt, 0, DB_MODE_RW); /* generation g consumed */

Channel events suit iterative algorithms where data is produced in
rounds and each round has a dedicated consumer.

Finish Events
-------------

A FINISH event (``ARTS_EVENT_HINT_FINISH``) is a latch used for
hierarchical termination detection: its counter tracks the live EDTs of
a *finish scope*, and it fires when the scope has fully drained.  Wait
on it with :c:func:`arts_event_wait` or chain a continuation EDT with
:c:func:`arts_add_dependence`.  See :doc:`finish_events` for scope
membership, inheritance, and nesting.

Destroying Events
-----------------

Because fired events linger, every non-finish event must eventually be
released with:

.. code-block:: c

   void arts_event_destroy(arts_guid_t guid);

Destroy removes the event's route-table entry; the same GUID may be
re-created afterward.  In-flight satisfies and dependence registrations
for the GUID are handled through the runtime's out-of-order queue.
Finish events auto-destroy when they fire and must not be destroyed
manually.

Minimal Example
---------------

A two-producer fan-in (after ``tests/event_basic.c``): the dependent
EDT runs only after both satisfies arrive.

.. code-block:: c

   void dependent_edt(uint32_t paramc, const uint64_t *paramv,
                      uint32_t depc, arts_edt_dep_t depv[]) {
       /* runs once the latch drained to zero */
   }

   void main_edt(uint32_t paramc, const uint64_t *paramv,
                 uint32_t depc, arts_edt_dep_t depv[]) {
       arts_event_hint_t h = ARTS_EVENT_HINT_LATCH(2);
       arts_guid_t ev = arts_event_create(&h);

       arts_guid_t dep = arts_edt_create(dependent_edt, 0, NULL, 1, NULL);
       arts_add_dependence(ev, dep, 0, DB_MODE_NULL);

       /* Two producers each decrement the latch; the second fires it. */
       arts_event_satisfy_slot(ev, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
       arts_event_satisfy_slot(ev, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
   }
