Fibonacci Example
=================

.. warning::

   Part of the deprecated ``examples/`` tree — it does not build against the
   current public API (``ARTS_BUILD_EXAMPLES`` is forced OFF; enabling it is
   a configure error).  Kept as a reading reference until modernized; the
   run instructions below are historical.

The Fibonacci example (``examples/cpu/fib.c``) demonstrates recursive
EDT creation, value-based signaling, and distributed execution.

.. contents:: On this page
   :local:
   :depth: 2

Algorithm
---------

The classic recursive Fibonacci is parallelised by turning each
recursive call into an EDT:

- ``fib_fork`` — if *n* < 2 returns *n* directly (base case); otherwise
  creates a ``fib_join`` EDT and two child ``fib_fork`` EDTs for
  *n* − 1 and *n* − 2.
- ``fib_join`` — receives two results via dependency slots, adds them,
  and signals the sum to the parent.
- ``fib_done`` — prints the final result and calls
  :c:func:`arts_shutdown`.

Source Code
-----------

.. literalinclude:: ../../examples/cpu/fib.c
   :language: c
   :linenos:
   :lines: 39-
   :caption: examples/cpu/fib.c (after license header)

Walkthrough
-----------

Entry Point
~~~~~~~~~~~

.. code-block:: c

   int main(int argc, char **argv) {
       arts_rt(argc, argv);
       return 0;
   }

:c:func:`arts_rt` reads ``arts.cfg``, initializes the runtime, spawns
worker threads, and blocks until :c:func:`arts_shutdown` is called.

Initialization
~~~~~~~~~~~~~~

.. code-block:: c

   void main_edt(uint32_t paramc, const uint64_t *paramv,
                      uint32_t depc, arts_edt_dep_t depv[]) {
       (void)paramc;
       (void)depc;
       (void)depv;
       char **argv = (char **)paramv[1];
       uint64_t num = strtol(argv[1], NULL, 10);
       arts_guid_t done_guid =
           arts_edt_create(fib_done, 1, &num, 1, &(arts_hint_t){.route = 0});
       uint64_t args[3] = {(uint64_t)done_guid, 0, num};
       start = arts_get_time_stamp();
       arts_edt_create(fib_fork, 3, args, 0, &(arts_hint_t){.route = 0});
   }

``main_edt`` is scheduled by the runtime on rank 0 after init.
It receives ``argc``/``argv`` via ``paramv[0]``/``paramv[1]``:

1. Create ``fib_done`` with 1 static parameter (the input number) and
   1 dependency slot (to receive the result), placed on node 0.
2. Create the root ``fib_fork`` with 0 dependency slots (fires
   immediately), passing the done-EDT GUID, slot index, and the input
   number as static parameters.

Fork
~~~~

``fib_fork`` distributes work across nodes using round-robin placement:

.. code-block:: c

   unsigned int next = (arts_get_current_node() + 1) % arts_get_total_nodes();

Each recursive call creates a new EDT on the ``next`` node, spreading
the computation across the cluster.

Join
~~~~

``fib_join`` receives two values via ``depv[0].guid`` and
``depv[1].guid`` (using :c:func:`arts_signal_edt_value`), sums them,
and signals the result up to the parent EDT.

Running
-------

.. code-block:: bash

   cd build/examples/cpu
   cp ../../configs/local/test/1n.cfg ./arts.cfg
   ./fib 30

Multi-node:

.. code-block:: bash

   srun -N 4 -n 4 -c 16 ./fib 30
