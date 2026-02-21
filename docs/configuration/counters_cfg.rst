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

Naming Convention
~~~~~~~~~~~~~~~~~

All counter names use a type prefix:

- ``TIME_*`` — timer counters (nanoseconds, measured with START/STOP).
- ``NUM_*`` — count counters (event occurrences, measured with INCREMENT_BY).
- ``BYTES_*`` — byte counters (accumulated bytes, measured with INCREMENT/DECREMENT_BY).
- ``OBJ_*`` — per-object counters (tracked per ``arts_id`` in hash tables/traces).

Examples
~~~~~~~~

.. code-block:: ini

   # Disabled
   TIME_EDT_EXEC=OFF

   # Capture once at end, cluster-wide sum
   TIME_EDT_EXEC=ONCE,CLUSTER

   # Periodic per-thread capture (no reduction)
   TIME_EDT_EXEC=PERIODIC,THREAD

   # Periodic cluster-wide maximum
   TIME_EDT_EXEC=PERIODIC,CLUSTER,MAX

Available Counters
------------------

EDT Lifecycle — Time
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``TIME_EDT_EXEC``
     - Wall-clock time executing EDT user functions (paused during yield/wait).
   * - ``TIME_EDT_CREATE``
     - Time spent in ``arts_edt_create()`` (GUID alloc, route table insertion).
   * - ``TIME_EDT_SIGNAL``
     - Time spent in EDT dependency signaling.
   * - ``TIME_CONTEXT_SWITCH``
     - Time spent saving/restoring EDT thread-local context during preemption.

EDT Lifecycle — Count
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_EDT_CREATE``
     - Number of EDTs created.
   * - ``NUM_EDT_ACQUIRE``
     - Number of EDTs that resolved all dependencies and became ready.
   * - ``NUM_EDT_FINISH``
     - Number of EDTs that completed execution.
   * - ``NUM_EDT_SIGNAL``
     - Number of EDT dependency signals (``arts_signal_edt`` calls).
   * - ``NUM_YIELD``
     - Count of voluntary EDT yields (``arts_yield``, ``arts_wait_on_handle``).

DataBlock Lifecycle — Time
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``TIME_DB_CREATE``
     - Time spent in ``arts_db_create()`` (alloc, route table, data copy).
   * - ``TIME_DB_GET``
     - Time spent in ``arts_get_from_db()`` (remote DataBlock read).
   * - ``TIME_DB_PUT``
     - Time spent in ``arts_put_in_db()`` (remote DataBlock write).

DataBlock Lifecycle — Count
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_DB_CREATE``
     - Number of DataBlocks created.
   * - ``NUM_DB_GET``
     - Number of ``arts_get_from_db()`` calls (remote DB reads).
   * - ``NUM_DB_PUT``
     - Number of ``arts_put_in_db()`` calls (remote DB writes).
   * - ``NUM_DB_DESTROY``
     - Number of ``arts_db_destroy()`` calls.
   * - ``NUM_DB_ACQUIRE_READ``
     - Count of READ-mode DataBlock acquisitions.
   * - ``NUM_DB_ACQUIRE_WRITE``
     - Count of WRITE-mode DataBlock acquisitions.
   * - ``NUM_OWNER_UPDATE_SAVED``
     - Owner updates avoided via READ access (no writeback needed).
   * - ``NUM_OWNER_UPDATE_PERFORMED``
     - Owner updates performed via WRITE access (writeback to owner).

DataBlock Lifecycle — Bytes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``BYTES_DB_CREATE``
     - Total bytes of DataBlock data allocated via ``arts_db_create()``.
   * - ``BYTES_DB_PUT``
     - Total bytes written via ``arts_put_in_db()``.

Memory — Bytes
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``BYTES_MEMORY_FOOTPRINT``
     - Running total of allocated minus freed bytes (current heap footprint).

Network — Bytes
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``BYTES_REMOTE_SENT``
     - Total bytes sent over the network to remote ranks.
   * - ``BYTES_REMOTE_RECEIVED``
     - Total bytes received from the network.

Network — Count
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_REMOTE_SEND``
     - Number of remote messages sent (per ``arts_actual_send`` call).
   * - ``NUM_REMOTE_RECEIVE``
     - Number of remote messages received (per completed packet).

Network — Time
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``TIME_REMOTE_MOVE``
     - Time spent in ``arts_remote_memory_move()`` (sending local DB to remote).

Event — Time
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``TIME_EVENT_CREATE``
     - Time spent in ``arts_event_create()``.
   * - ``TIME_PERSISTENT_EVENT_CREATE``
     - Time spent in ``arts_persistent_event_create()``.
   * - ``TIME_EVENT_SIGNAL``
     - Time spent in ``arts_event_satisfy_slot()`` (signaling dependents).
   * - ``TIME_PERSISTENT_EVENT_SIGNAL``
     - Time spent in ``arts_persistent_event_satisfy()`` (latch reaches 0).

Event — Count
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_EVENT_CREATE``
     - Number of ``arts_event_create()`` calls.
   * - ``NUM_EVENT_SIGNAL``
     - Number of ``arts_event_satisfy_slot()`` calls.
   * - ``NUM_PERSISTENT_EVENT_CREATE``
     - Number of ``arts_persistent_event_create()`` calls.
   * - ``NUM_PERSISTENT_EVENT_SIGNAL``
     - Number of ``arts_persistent_event_satisfy()`` calls.

Scheduling
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_STEAL_ATTEMPT``
     - Number of work-stealing attempts.
   * - ``NUM_STEAL_SUCCESS``
     - Number of successful work steals (non-NULL deque pop).
   * - ``TIME_YIELD``
     - Time spent in ``arts_yield`` / ``arts_wait_on_handle`` (idle waiting).

Epoch
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_EPOCH_CREATE``
     - Number of epoch creations.

Out-of-Order
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_OO_ENQUEUE``
     - Number of OO list insertions (deferred operations for not-yet-created targets).

Object Counters
~~~~~~~~~~~~~~~

Per-object counters track performance metrics aggregated by ``arts_id``
(a compiler-assigned unique identifier for each EDT/DB type).
Output is written to separate ``object_n{id}.json`` and ``object.json``
files (not embedded in scalar counter output).

Enabling any EDT counter activates the per-thread EDT hash table (~40 KB).
Enabling any DB counter activates the per-thread DB hash table (~40 KB).

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``OBJ_NUM_EDT``
     - EDT invocation count per arts_id.
   * - ``OBJ_TIME_EDT_EXEC``
     - Total EDT execution time per arts_id (nanoseconds).
   * - ``OBJ_TIME_EDT_STALL``
     - Total EDT stall time per arts_id (nanoseconds).
   * - ``OBJ_NUM_DB``
     - DB access count per arts_id.
   * - ``OBJ_BYTES_DB_LOCAL``
     - Local bytes accessed per arts_id.
   * - ``OBJ_BYTES_DB_REMOTE``
     - Remote bytes accessed per arts_id.
   * - ``OBJ_NUM_DB_CACHE_MISS``
     - Cache misses per arts_id.
   * - ``OBJ_TRACE_EDT``
     - Detailed per-invocation EDT execution records.
   * - ``OBJ_TRACE_DB``
     - Detailed per-invocation DB access records.

Runtime Phases — Time
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``TIME_INIT``
     - Time spent in ARTS initialization (config, network, threads).
   * - ``TIME_TOTAL``
     - Total end-to-end execution time (init complete to shutdown).

Output
------

Counter output is written as JSON to the directory specified by
``counter_folder`` in ``arts.cfg`` (default: ``./counters``).

File naming depends on the counter level:

- **THREAD** — ``n{node}_t{thread}.json`` (one file per thread).
- **NODE** — ``n{node}.json`` (one file per node, reduced across threads).
- **CLUSTER** — ``cluster.json`` (single file on the master node, reduced
  across all nodes).

Object counters are written to separate ``object_n{node}.json`` and
``object.json`` files.

Each JSON file includes a ``timestamp`` (Unix epoch), a ``version``
string, and a ``counters`` object whose keys are counter names.
``PERIODIC`` counters include a ``captureHistory`` array of
``[epoch, value]`` pairs.
