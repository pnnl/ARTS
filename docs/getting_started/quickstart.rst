Quickstart
==========

This tutorial walks through a minimal ARTS program step by step.

.. contents:: On this page
   :local:
   :depth: 2

Program Structure
-----------------

Every ARTS program consists of three parts:

1. **EDT functions** — the async work units.
2. ``arts_main_edt()`` — entry-point EDT, scheduled on rank 0 after runtime init.
3. ``main()`` — calls :c:func:`arts_rt` to start the runtime.

.. code-block:: c

   #include "arts.h"

   /* 1.  EDT function ---------------------------------------------------- */
   void my_task(uint32_t paramc, const uint64_t *paramv,
                uint32_t depc, arts_edt_dep_t depv[]) {
       arts_printf("Hello from node %u, worker %u!\n",
                   arts_get_current_node(), arts_get_current_worker());
       arts_shutdown();
   }

   /* 2.  Entry-point EDT ------------------------------------------------- */
   void arts_main_edt(uint32_t paramc, const uint64_t *paramv,
                      uint32_t depc, arts_edt_dep_t depv[]) {
       (void)depc;
       (void)depv;
       /* paramv[0] = argc, paramv[1] = argv */
       arts_edt_create(my_task, 0, NULL, 0, NULL);
   }

   /* 3.  Entry point ----------------------------------------------------- */
   int main(int argc, char **argv) {
       arts_rt(argc, argv);   /* blocks until arts_shutdown() */
       return 0;
   }

Configuration File
------------------

ARTS looks for ``arts.cfg`` in the current directory (or the path in
the ``artsConfig`` environment variable).  A minimal configuration:

.. code-block:: ini

   [ARTS]
   worker_threads=4
   launcher=ssh
   node_count=1
   nodes=localhost
   default_ports=34739

Copy the sample config for a quick start:

.. code-block:: bash

   cp <arts-root>/sample_configs/arts.cfg .

Building and Running
--------------------

Link your program against ``arts_shared`` (or ``arts_static``):

.. code-block:: bash

   gcc -std=c17 -o hello hello.c -larts_shared -lpthread

Run with the config file in the same directory:

.. code-block:: bash

   ./hello

Multi-Node Execution
~~~~~~~~~~~~~~~~~~~~

For SSH-based multi-node runs, list nodes in ``arts.cfg``:

.. code-block:: ini

   node_count=4
   nodes=node1,node2,node3,node4

Then launch normally — ARTS SSHs into each node automatically.

For SLURM clusters:

.. code-block:: bash

   srun -N 4 -n 4 -c 16 ./hello

Next Steps
----------

- :doc:`/programming_model/edt` — learn about EDT creation and signaling.
- :doc:`/programming_model/datablocks` — understand the DataBlock memory model.
- :doc:`/examples/fib` — a complete Fibonacci example with distributed work.
