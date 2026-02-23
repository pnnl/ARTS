Events
======

Events are ARTS's synchronization primitive — latch-based counters that
fire when their count reaches zero, triggering dependent EDTs.

.. contents:: On this page
   :local:
   :depth: 2

Latch Events
------------

A latch event maintains an integer counter.  When the counter reaches
zero, the event fires and delivers its data to all registered
dependents.

.. code-block:: c

   arts_guid_t evt = arts_event_create(target_node, ARTS_EVENT_LATCH,
                                       initial_latch_count, NULL_GUID);

   /* Register an EDT to fire when the event completes */
   arts_add_dependence(evt, edt_guid, slot);

   /* Decrement the latch counter */
   arts_event_satisfy_slot(evt, data_guid, ARTS_EVENT_LATCH_DECR_SLOT);

.. note::

   ``arts_event_create`` takes the target node rank, the event type
   (``ARTS_EVENT_LATCH``, ``ARTS_EVENT_ONCE``, ``ARTS_EVENT_STICKY``,
   ``ARTS_EVENT_IDEM``, ``ARTS_EVENT_COUNTED``, or ``ARTS_EVENT_CHANNEL``),
   a latch count (used by LATCH and COUNTED), and a data GUID (used by
   CHANNEL).  Unused parameters are silently ignored.

Slot Types
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Slot
     - Effect
   * - ``ARTS_EVENT_LATCH_DECR_SLOT``
     - Decrement the latch counter by one.
   * - ``ARTS_EVENT_LATCH_INCR_SLOT``
     - Increment the latch counter by one.

When the counter reaches zero, the event fires.

Event Callbacks
~~~~~~~~~~~~~~~

Instead of wiring an EDT, you can attach an inline callback:

.. code-block:: c

   void my_callback(arts_edt_dep_t data) {
       /* runs inline on the thread that fires the event */
   }

   arts_add_local_event_callback(evt, my_callback);

.. warning::

   Callbacks execute on the signaling thread. Keep them short and
   avoid blocking operations.

Channel Events
--------------

Channel events are re-armable: they can fire multiple times, each
time delivering updated data via a coupled DataBlock.

.. code-block:: c

   arts_guid_t ch = arts_event_create(route, ARTS_EVENT_CHANNEL, 0, db_guid);

   /* Register a dependent — will be notified on every fire */
   arts_add_dependence(ch, edt_guid, slot);

   /* Increment and decrement the latch to control fire cycles */
   arts_event_increment_latch(ch);
   arts_event_decrement_latch(ch);

Use cases include iterative algorithms (e.g., graph analytics) where
data is updated in rounds and dependents need to be re-notified.

Common Patterns
---------------

**Fan-in (join):** create an event with ``latch_count = N``, register
one dependent EDT.  As N producers complete, each decrements the latch.
When the count hits zero, the join EDT fires.

**Barrier:** create an event with ``latch_count = num_workers``.  Each
worker signals the event when it reaches the barrier point.  The
continuation EDT fires after all workers check in.
