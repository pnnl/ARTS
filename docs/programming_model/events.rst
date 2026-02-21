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

   arts_guid_t evt = arts_event_create(target_node, initial_latch_count);

   /* Register an EDT to fire when the event completes */
   arts_add_dependence(evt, edt_guid, slot);

   /* Decrement the latch counter */
   arts_event_satisfy_slot(evt, data_guid, ARTS_EVENT_LATCH_DECR_SLOT);

.. note::

   ``arts_event_create`` takes two parameters: the target node rank and
   the initial latch count.

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

Persistent Events
-----------------

Persistent events are re-armable: they can fire multiple times, each
time delivering updated data.

.. code-block:: c

   arts_guid_t pevt = arts_persistent_event_create();

   /* Register a dependent — will be notified on every fire */
   arts_add_dependence_to_persistent_event(pevt, edt_guid, slot);

   /* Satisfy (fire) the persistent event with new data */
   arts_persistent_event_satisfy(pevt, data_guid);

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
