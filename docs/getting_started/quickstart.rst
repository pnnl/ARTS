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
2. ``main_edt()`` — entry-point EDT, scheduled on rank 0 after runtime init.
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
   void main_edt(uint32_t paramc, const uint64_t *paramv,
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
the ``ARTS_CONFIG`` environment variable).  A minimal configuration:

.. code-block:: ini

   [ARTS]
   worker_threads=4
   launcher=local

The annotated reference listing every recognized key is
``configs/example.cfg`` (documentation only — copy the entries you need
into your ``arts.cfg``). Ready-to-run localhost shapes live in
``configs/local/test/``:

.. code-block:: bash

   cp <arts-root>/configs/local/test/1n.cfg ./arts.cfg

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

   launcher=ssh
   node_count=4
   nodes=node1,node2,node3,node4
   default_ports=25000

Then launch normally — ARTS SSHs into each node automatically. Pick a
``default_ports`` base below the kernel ephemeral port range (32768–60999);
a base inside it can randomly collide with outgoing connections' source
ports.

For SLURM clusters:

.. code-block:: bash

   srun -N 4 -n 4 -c 16 ./hello

Next Steps
----------

- :doc:`/programming_model/edt` — learn about EDT creation and signaling.
- :doc:`/programming_model/datablocks` — understand the DataBlock memory model.
- :doc:`/examples/fib` — a complete Fibonacci example with distributed work.
