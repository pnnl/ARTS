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
     - 1
     - Number of parallel network connections per node pair.
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
       SLURM and LSF are auto-detected from environment variables
       (``SLURM_PROCID``, ``LSB_HOSTS``) and override this setting.
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
   * - ``default_ports``
     - 75563
     - Default network port(s).  Per-node ports in the ``nodes`` list
       override this.  Supports single port (``34739``), range
       (``[34739-34740]``), or comma-separated (``34739,34800``).

.. tip::

   When running multiple instances on the same machine via SSH, assign
   different ports per node: ``nodes=localhost:34739,localhost:34740``.

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
