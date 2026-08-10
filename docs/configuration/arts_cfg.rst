Runtime Configuration (``arts.cfg``)
====================================

ARTS reads its runtime configuration from ``arts.cfg`` in the current
working directory.  Override the path with the ``ARTS_CONFIG`` environment
variable:

.. code-block:: bash

   ARTS_CONFIG=/path/to/custom.cfg ./my_program

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
   * - ``worker_threads``
     - 4
     - Worker threads per node.  For multi-node runs the actual worker
       count is ``worker_threads``; sender/receiver threads are separate.
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
     - 16
     - Routing table size as power of 2 (e.g., 16 → 2\ :sup:`16`).
   * - ``auto_shutdown``
     - 0
     - Terminate when all EDTs complete.  Adds overhead from global EDT
       tracking; prefer explicit ``arts_shutdown()`` calls.

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
   * - ``gpu_lc_sync``
     - 0
     - Location Consistency sync policy: 0 = artsGetLatestGpuDb,
       1 = artsGetRandomGpuDb.
   * - ``gpu_buff_on``
     - 0
     - Enable GPU stream buffering.

Networking
----------

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Key
     - Default
     - Description
   * - ``progress_threads``
     - 1
     - Number of progress threads per node, draining the fabric completion
       queue (0 for local mode). The transport injects sends directly from
       workers, so there is no separate sender-thread key; fold any sender
       count from an older config into ``worker_threads``. The removed
       ``sender_threads`` key and the old ``receiver_threads`` name are a
       **hard error** at config-parse time, not a silent ignore, so a stale
       config fails loudly instead of quietly changing its thread budget.
   * - ``provider``
     - auto
     - libfabric provider name passed to ``fi_getinfo``'s ``prov_name`` hint
       (``tcp``, ``verbs``, ``cxi``, etc.). Unset/empty auto-selects. When
       set, this overrides the ambient ``FI_PROVIDER`` environment variable
       for the process — the config is the deliberate, versioned artifact;
       the environment variable is ambient and host-specific.
   * - ``regpool_slab_mb``
     - 64
     - Registered-memory slab pool size in MB, per NUMA node. Backs the
       libfabric memory registration the DB payload buffers draw from; the
       pool grows on demand from this floor.
   * - ``port_count``
     - (from ``ports``, else 1)
     - Parallel connections each node listens on.  The ``ports`` list must
       name exactly this many ports.
   * - ``ports``
     - (see below)
     - Network port(s) every node's listen ports are derived from.  A single
       port (``25000``), a range (``[25000-25001]``), or a comma-separated
       list (``25000,25010``).  **Required** with ``launcher=ssh``,
       ``slurm``, or ``lsf``, and **rejected** with ``launcher=local``.

Who fixes the ports depends on where the ranks land.  A remote launcher puts
one rank on each host, so every rank can listen on the same port — but each of
them resolves its peers' ports from its own copy of this config, so the config
must name them.  Pick a base below the kernel ephemeral port range
(``net.ipv4.ip_local_port_range``, typically 32768–60999): a listen port inside
it collides at random with the source port of any outgoing connection the
machine makes, and the bind dies instantly.

A ``launcher=local`` run is the mirror image.  Every rank is a process on this
one machine, so they cannot share a port and the ports must be found rather
than declared: before spawning the peers, the spawning rank bind-probes a block
of ``node_count × port_count`` ports, slides past anything already holding
them, and hands the result to the ranks it spawns.  The search is seeded from
the process id so two runs starting at the same moment do not pick the same
block.  Naming ``ports`` there is a configure error — it would only reintroduce
the collisions the search exists to avoid.

.. tip::

   To run several ranks on one machine, use ``launcher=local`` and let it
   place them.  The nodes list carries hostnames only.

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
