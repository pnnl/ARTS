Fibonacci Example
=================

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

   void init_per_worker(unsigned int node_id, unsigned int worker_id,
                        int argc, char **argv) {
       if (!node_id && !worker_id) {
           uint64_t num = strtol(argv[1], NULL, 10);
           arts_guid_t done_guid = arts_edt_create(fib_done, 0, 1, &num, 1);
           uint64_t args[3] = {(uint64_t)done_guid, 0, num};
           start = arts_get_time_stamp();
           arts_edt_create(fib_fork, 0, 3, args, 0);
       }
   }

Only the master thread (node 0, worker 0) performs initialisation:

1. Create ``fib_done`` with 1 dependency slot (to receive the result).
2. Create the root ``fib_fork``, passing the done-EDT GUID and the
   input number as static parameters.

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
   cp ../../sample_configs/arts.cfg .
   ./fib 30

Multi-node:

.. code-block:: bash

   srun -N 4 -n 4 -c 16 ./fib 30
