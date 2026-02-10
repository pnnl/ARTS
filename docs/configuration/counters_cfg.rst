Performance Counters (``counters.cfg``)
=======================================

ARTS includes a configurable performance counter system for profiling
EDT execution, DataBlock usage, memory footprint, and network traffic.

.. contents:: On this page
   :local:
   :depth: 2

File Format
-----------

``counters.cfg`` lives in the source tree (typically
``sample_configs/counters.cfg``) and is processed at CMake configure
time.  After editing, re-run ``cmake`` to regenerate counter code.

Each line follows the format:

.. code-block:: text

   COUNTER_NAME=MODE[,LEVEL[,REDUCE]]

Parameters
~~~~~~~~~~

**Mode** (required):

- ``OFF`` — counter disabled.
- ``ONCE`` — captured once at shutdown.
- ``PERIODIC`` — captured at regular intervals (see
  ``counter_capture_interval`` in ``arts.cfg``).

**Level** (optional, default ``NODE``):

- ``THREAD`` — per-thread output, no reduction.
- ``NODE`` — per-node output, reduced across threads.
- ``CLUSTER`` — cluster-wide output, reduced across all nodes.

**Reduce** (optional, default ``SUM``):

- ``SUM`` — sum values.
- ``MAX`` — maximum value.
- ``MIN`` — minimum value.
- ``MASTER`` — master thread/node value only.

Examples
~~~~~~~~

.. code-block:: ini

   # Disabled
   EDT_RUNNING_TIME=OFF

   # Capture once at end, cluster-wide sum
   EDT_RUNNING_TIME=ONCE,CLUSTER

   # Periodic per-thread capture (no reduction)
   EDT_RUNNING_TIME=PERIODIC,THREAD

   # Periodic cluster-wide maximum
   EDT_RUNNING_TIME=PERIODIC,CLUSTER,MAX

Available Counters
------------------

EDT Counters
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``EDT_RUNNING_TIME``
     - Wall-clock time spent executing EDTs.
   * - ``NUM_EDTS_CREATED``
     - Number of EDTs created.
   * - ``NUM_EDTS_ACQUIRED``
     - Number of EDTs acquired (dequeued for execution).
   * - ``NUM_EDTS_FINISHED``
     - Number of EDTs that completed execution.

DataBlock Counters
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_DBS_CREATED``
     - Number of DataBlocks created.

Memory Counters
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``MEMORY_FOOTPRINT``
     - Current runtime memory footprint.

Network Counters
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``REMOTE_BYTES_SENT``
     - Bytes sent over the network.
   * - ``REMOTE_BYTES_RECEIVED``
     - Bytes received from the network.

Timing Counters
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``INITIALIZATION_TIME``
     - Time spent in the ARTS initialization phase.
   * - ``END_TO_END_TIME``
     - Total end-to-end execution time.

Acquire-Mode Counters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``ACQUIRE_READ_MODE``
     - Count of READ-mode acquisitions.
   * - ``ACQUIRE_WRITE_MODE``
     - Count of WRITE-mode acquisitions.
   * - ``OWNER_UPDATES_SAVED``
     - Owner updates avoided via READ override.
   * - ``OWNER_UPDATES_PERFORMED``
     - Owner updates actually performed.

ARTS ID Tracking Counters
~~~~~~~~~~~~~~~~~~~~~~~~~

These counters link compile-time metadata (from ArtsMate) with
runtime performance:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``ARTS_ID_EDT_METRICS``
     - Per-arts_id EDT aggregate metrics.
   * - ``ARTS_ID_DB_METRICS``
     - Per-arts_id DB aggregate metrics.
   * - ``ARTS_ID_EDT_CAPTURES``
     - Per-arts_id EDT detailed captures.
   * - ``ARTS_ID_DB_CAPTURES``
     - Per-arts_id DB detailed captures.

Output
------

Counter output is written to the directory specified by
``counter_folder`` in ``arts.cfg`` (default: ``./counters``).
Files are named by node/thread and contain timestamped values in a
format suitable for post-processing.
