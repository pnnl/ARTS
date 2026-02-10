Epochs (Termination Detection)
==============================

Epochs provide hierarchical termination detection — a mechanism to know
when a dynamic set of tasks has completed.

.. contents:: On this page
   :local:
   :depth: 2

Problem Statement
-----------------

In a task-based runtime, tasks can spawn new tasks recursively.
Determining when *all* transitively spawned work has finished requires
a distributed consensus protocol.  ARTS solves this with **epochs**.

Usage
-----

.. code-block:: c

   /* Create and start an epoch */
   arts_guid_t epoch = arts_initialize_and_start_epoch(done_edt, slot);

   /* All EDTs created after this point belong to the epoch.
      When every EDT (and their children) finishes, done_edt fires. */

   arts_edt_create(my_task, 0, 0, NULL, 0);
   arts_edt_create(my_task, 1, 0, NULL, 0);

When the epoch detects global quiescence (no active or pending tasks
remain), it satisfies the specified EDT/slot.

Blocking Wait
~~~~~~~~~~~~~

For sequential idioms, use :c:func:`arts_wait_on_handle` to block the
current worker until the epoch completes:

.. code-block:: c

   arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);
   /* ... spawn work ... */
   arts_wait_on_handle(epoch);
   /* all epoch-scoped work is done here */

Separate Init and Start
~~~~~~~~~~~~~~~~~~~~~~~

For more control, split initialization and start:

.. code-block:: c

   arts_guid_t epoch = arts_initialize_epoch(done_edt, slot);
   /* EDT creation here does NOT yet count toward the epoch */
   arts_start_epoch(epoch);
   /* EDT creation here DOES count */

How It Works
------------

ARTS uses a three-phase distributed termination detection algorithm:

1. **Phase 1** — initial quiescence check: are local active/finished
   counts balanced?
2. **Phase 2** — counter stabilization: re-check after a communication
   round to rule out in-flight messages.
3. **Phase 3** — termination confirmed: signal the exit EDT.

Each epoch tracks ``activeCount`` and ``finishedCount`` per node, then
reduces across the cluster to confirm global quiescence.

Relationship with EDTs
----------------------

EDTs are automatically associated with the current epoch (the epoch
active at creation time).  You can also explicitly add an EDT to an
epoch:

.. code-block:: c

   arts_add_edt_to_epoch(epoch_guid);

Or create an EDT bound to a specific epoch:

.. code-block:: c

   arts_edt_create_with_epoch(func, node, paramc, paramv, depc, epoch_guid);
