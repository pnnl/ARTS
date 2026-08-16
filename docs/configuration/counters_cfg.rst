Performance Counters (``counters.cfg``)
=======================================

ARTS includes a configurable performance counter system for profiling
EDT execution, DataBlock usage, memory footprint, and network traffic.

.. contents:: On this page
   :local:
   :depth: 2

File Format
-----------

``counters.cfg`` lives in the source tree (default ``configs/counters.cfg``,
overridable via the ``ARTS_COUNTER_CONFIG`` CMake cache variable) and is
processed at CMake configure time.  After editing, re-run ``cmake`` to
regenerate counter code.

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

The list below is generated from the ``ARTS_COUNTER_LIST`` X-macro in
``libs/include/internal/arts/counter/counter.h`` — the single source of
truth the enum, the string table, and ``artsrun``'s catalog are all
derived from.  Grouping follows the header's own comments.

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
     - Number of EDT dependency signals (``arts_edt_satisfy_slot()`` calls).
   * - ``NUM_YIELD``
     - Count of voluntary EDT yields (``arts_event_wait()`` blocking).

DataBlock Lifecycle — Time
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``TIME_DB_CREATE``
     - Time spent in ``arts_db_create()`` (alloc, route table, data copy).

DataBlock Lifecycle — Count
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_DB_CREATE``
     - Number of DataBlocks created.
   * - ``NUM_DB_DESTROY``
     - Number of ``arts_db_destroy()`` calls.
   * - ``NUM_DB_ACQUIRE_READ``
     - Count of RO-mode DataBlock acquisitions.
   * - ``NUM_DB_ACQUIRE_WRITE``
     - Count of RW-mode DataBlock acquisitions.

Coherence — Acquire Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where an acquire was answered.  Their sum is the acquire population; the
hit share is what aggregation buys, and is comparable across every
coherence arm because both are counted in the shared acquire helpers
rather than in one arm's handler.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_DB_ACQUIRE_LOCAL_HIT``
     - Acquire answered from a copy already resident on the requesting rank.
   * - ``NUM_DB_ACQUIRE_REMOTE``
     - Acquire that had to fetch from a remote rank.

DataBlock Lifecycle — Bytes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``BYTES_DB_CREATE``
     - Total bytes of DataBlock data allocated via ``arts_db_create()``.

Coherence — Bytes
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``BYTES_DB_PAYLOAD_SENT``
     - Payload bytes actually shipped for a DataBlock, as opposed to
       ``BYTES_REMOTE_SENT`` which mixes control traffic in.

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
     - Number of remote messages sent.
   * - ``NUM_REMOTE_RECEIVE``
     - Number of remote messages received.

Network — Outbound Message Size Census
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Outbound message size census, bucketed by total wire size (header +
payload) at the async send entry points, before fragmentation/retry.
``NET_MSG_TOTAL`` is the message population count standing in for a
per-wire-type breakdown (no array-counter infra to key by message type
cheaply).

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NET_MSG_LE64``
     - Outbound messages with total wire size <= 64 bytes.
   * - ``NET_MSG_LE512``
     - Outbound messages with total wire size <= 512 bytes.
   * - ``NET_MSG_LE4K``
     - Outbound messages with total wire size <= 4 KB.
   * - ``NET_MSG_LE64K``
     - Outbound messages with total wire size <= 64 KB.
   * - ``NET_MSG_GT64K``
     - Outbound messages with total wire size > 64 KB.
   * - ``NET_MSG_TOTAL``
     - Total outbound message count (the population the size buckets
       partition).

Network — Time
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``TIME_REMOTE_MOVE``
     - Time spent moving a local DB's data out to a remote rank.

Event — Time
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``TIME_EVENT_CREATE``
     - Time spent in ``arts_event_create()``.
   * - ``TIME_EVENT_SIGNAL``
     - Time spent in ``arts_event_satisfy_slot()`` (signaling dependents).

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

Scheduling — Count
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_STEAL_ATTEMPT``
     - Number of work-stealing attempts.
   * - ``NUM_STEAL_SUCCESS``
     - Number of successful work steals (non-NULL deque pop).

Scheduling — Time
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``TIME_YIELD``
     - Time spent blocked in ``arts_event_wait()`` (idle, spin-polling the
       scheduler).

Out-of-Order
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_OO_ENQUEUE``
     - Number of OO list insertions (deferred operations for not-yet-created
       targets).

Validation Arm (VAL) — Snapshot Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A reply with no payload (``data_present`` 0) is one the version ledger
saved; a size-only CTS (``data_present`` 2) is the rendezvous negotiating
a landing and is counted apart, since it makes a size-unknown first touch
cost two requests rather than one.  Inert unless the build compiles the
VAL family in.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_SNAPSHOT_REQUEST``
     - Remote snapshot request issued by a VAL-arm RO acquire.
   * - ``NUM_SNAPSHOT_HEADER_ONLY``
     - Reply with no payload: the requested version was already covered by
       the requester's cached-version ledger.
   * - ``NUM_SNAPSHOT_SIZE_CTS``
     - Size-only CTS reply: the rendezvous negotiating a landing buffer for
       a size-unknown first touch.

Size-Only CTS Fallbacks / RO Combining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Size-only CTS fallbacks on the other planes: with the GUID size hint
these count only sentinel GUIDs (pre-reserved ranges, oversize), so a
nonzero steady rate is the hint NOT covering a workload.  The RO combine
counters are live only when ``ARTS_RO_REQUEST_COMBINING`` is on.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_GRANT_SIZE_CTS``
     - Size-only CTS fallback on the migrating-grant plane (VAL/INV).
   * - ``NUM_EXCL_SIZE_CTS``
     - Size-only CTS fallback on the EXCL plane.
   * - ``NUM_INV_SIZE_CTS``
     - Size-only CTS fallback on the INV plane.
   * - ``NUM_RO_COMBINE_WINDOW``
     - A combining window opened for a same-DB remote RO acquire.
   * - ``NUM_RO_COMBINE_JOINED``
     - A request that joined an open combining window instead of going to
       the wire.

Invalidation Arm (INV)
~~~~~~~~~~~~~~~~~~~~~~

One round per RW release, its multicast and the acks it blocks on.  The
time is the release-side cost the write-policy regime crossover is
attributed to.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_INVALIDATE_ROUND``
     - Number of invalidation rounds run (one per RW release under INV).
   * - ``NUM_INVALIDATE_SENT``
     - Number of INVALIDATE messages multicast to sharers.
   * - ``NUM_INVALIDATE_ACK``
     - Number of INVALIDATE acknowledgements received.
   * - ``TIME_INVALIDATE_ROUND``
     - Time an RW release spends blocked on its invalidation round.

Migrating Write Permission
~~~~~~~~~~~~~~~~~~~~~~~~~~

A grant that moved versus one a later local writer reused without
touching the wire (the sticky grant).  Live in the arms whose ownership
migrates (VAL, INV).

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_GRANT_MIGRATE``
     - An ownership grant that moved across ranks.
   * - ``NUM_GRANT_LOCAL_REUSE``
     - A local writer that reused a still-resident grant without touching
       the wire.

Exclusion Arm (EXCL)
~~~~~~~~~~~~~~~~~~~~

Turns that could not be granted on arrival and had to queue at the home:
the reader/writer serialization its philosophy pays for.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_EXCL_QUEUE_WAIT``
     - Turns that queued at the home instead of being granted on arrival.

Write-Through Publish Flight Machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These count the RARE transitions a green suite cannot prove exercised —
a campaign where one stays zero ran no coverage of that path, not a
healthy path.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_PUB_FLIGHT``
     - Payload/control flights launched.
   * - ``NUM_PUB_FLIGHT_JOIN``
     - Releases coalesced into an open flight (the write-combining win).
   * - ``NUM_PUB_FLIGHT_TRAILING``
     - Relaunches for waiters the ACK left uncovered.
   * - ``NUM_PUB_CTS_FALLBACK``
     - Credit-less announce legs (steady nonzero = the credit teachers are
       not covering a workload).
   * - ``NUM_PUB_FLIGHT_ABANDON``
     - Parked publish waiters woken by a destroy/teardown path instead of
       an ACK.

Grant Plane Rare Paths
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Counter
     - Description
   * - ``NUM_GRANT_BATON_RECLAIM``
     - A releasing claimant found the queue non-empty on its post-release
       re-check and re-claimed it (the lost-wake window the re-check loop
       exists to close).
   * - ``NUM_GRANT_HOME_DATALESS``
     - An ownership grant INTO the write-through home, which moves the
       permission with no payload (the home already holds the newest
       bytes).

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
   * - ``OBJ_NUM_DB``
     - DB access count per arts_id.
   * - ``OBJ_BYTES_DB``
     - Bytes accessed per arts_id.
   * - ``OBJ_NUM_DB_CACHE_MISS``
     - Cache misses per arts_id.
   * - ``OBJ_TRACE_EDT``
     - Detailed per-invocation EDT execution records.
   * - ``OBJ_TRACE_DB``
     - Detailed per-invocation DB access records.

.. note::

   End-to-end / init wall time is no longer a counter.  ``TIME_INIT`` and
   ``TIME_TOTAL`` were replaced by the env-gated ``[E2E] <ns>`` stderr
   marker: set ``ARTS_E2E_MARKER`` in the environment and rank 0 prints the
   span from application start to shutdown recognition on exit (see
   ``libs/src/core/system/threads.c`` / ``runtime.c``), in the same form
   the reference runtimes (xsocr, ocr-vx) use so a harness parses all
   three identically.

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
