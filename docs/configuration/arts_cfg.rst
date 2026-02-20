Runtime Configuration (``arts.cfg``)
====================================

ARTS reads its runtime configuration from ``arts.cfg`` in the current
working directory.  Override the path with the ``artsConfig`` environment
variable:

.. code-block:: bash

   artsConfig=/path/to/custom.cfg ./my_program

.. contents:: On this page
   :local:
   :depth: 2

Threading
---------

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Key
     - Default
     - Description
   * - ``threads``
     - 4
     - Total threads per node.  For multi-node runs the actual worker
       count is ``threads - outgoing - incoming``.
   * - ``tmt``
     - 0
     - Temporal multi-threading depth (0–64).  Enables context switching
       within a worker.
   * - ``stack_size``
     - 0
     - Thread stack size in bytes (0 = OS default).

Hardware Pinning
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Key
     - Default
     - Description
   * - ``pin``
     - 1
     - Enable thread-to-core pinning.
   * - ``pin_stride``
     - 1
     - Core spacing between pinned threads (useful for SMT).
Scheduling
----------

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Key
     - Default
     - Description
   * - ``scheduler``
     - 0
     - Scheduler type: ``0`` = CPU, ``3`` = GPU.
   * - ``worker_init_deque_size``
     - 4096
     - Initial size of each worker's Chase-Lev deque.
   * - ``route_table_size``
     - 20
     - Routing table size as power of 2 (e.g., 20 → 2\ :sup:`20`).

GPU Support
-----------

To enable GPU support, set ``scheduler=3`` and ``gpu=<N>``.

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Key
     - Default
     - Description
   * - ``gpu``
     - 0
     - Number of GPUs per node.
   * - ``gpu_route_table_size``
     - 12
     - GPU routing table size (power of 2).
   * - ``gpu_locality``
     - 0
     - Locality policy: 0 = random, 1 = allOrNothing, 2 = atLeastOne.
   * - ``gpu_fit``
     - 0
     - Fit policy: 0 = firstFit, 1 = bestFit, 2 = worstFit,
       3 = roundRobinFit.
   * - ``gpu_max_edts``
     - -1
     - Max concurrent GPU EDTs (-1 = unlimited).
   * - ``gpu_max_memory``
     - -1
     - Max GPU memory in bytes (-1 = unlimited).
   * - ``gpu_p2p``
     - 0
     - Enable GPU peer-to-peer transfers.

Networking
----------

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Key
     - Default
     - Description
   * - ``outgoing``
     - 1
     - Number of outgoing (sender) network threads.
   * - ``incoming``
     - 1
     - Number of incoming (receiver) network threads.
   * - ``ports``
     - 1
     - Number of network ports/connections per node pair.
   * - ``net_interface``
     - auto
     - Network interface name (``eth0``, ``ib0``, etc.).

Launcher
--------

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Key
     - Default
     - Description
   * - ``launcher``
     - ssh
     - Launch method: ``ssh``, ``slurm``, ``lsf``, or ``local``.
   * - ``master_node``
     - (first)
     - Hostname of the master node.
   * - ``node_count``
     - (auto)
     - Number of nodes.
   * - ``nodes``
     - localhost
     - Comma-separated node list.  Supports per-node ports
       (``host:port``) and range expansion (``node[01-10]``).
   * - ``port``
     - 75563
     - Default network port (overridden by per-node ports).

.. tip::

   When running multiple instances on the same machine via SSH, assign
   different ports: ``nodes=localhost:34739,localhost:34740``.

Debug / Utility
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Key
     - Default
     - Description
   * - ``core_dump``
     - 0
     - Enable core dumps on crash.
   * - ``kill_mode``
     - 0
     - Kill leftover processes on SSH nodes before launch.
   * - ``counter_folder``
     - ``./counters``
     - Output directory for performance counter files.
   * - ``counter_capture_interval``
     - 100
     - Capture interval for ``PERIODIC`` counters (milliseconds).
