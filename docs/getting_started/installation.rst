Installation
============

This page covers building ARTS from source.

.. contents:: On this page
   :local:
   :depth: 2

Prerequisites
-------------

- **C17 compiler**: GCC >= 7 or Clang >= 5 (other compilers are not supported)
- **CMake** >= 3.22
- **Ninja** build system (Make is not supported)
- **POSIX threads** (pthreads)

- **hwloc** (bundled as a git submodule; built from source automatically)

Optional:

- **CUDA toolkit** for GPU support

Obtaining the Source
--------------------

.. code-block:: bash

   git clone <repository-url> arts
   cd arts

Building
--------

ARTS *requires* the Ninja generator. Make is not supported.

.. code-block:: bash

   mkdir build && cd build
   cmake -GNinja .. -DCMAKE_BUILD_TYPE=Release
   ninja
   ninja install   # installs to CMAKE_INSTALL_PREFIX (default: project/install)

Debug build (add ``-DARTS_USE_SANS=ON`` for ASan/UBSan/LSan, which are OFF by default):

.. code-block:: bash

   cmake -GNinja .. -DCMAKE_BUILD_TYPE=Debug -DARTS_USE_SANS=ON

CMake Options
~~~~~~~~~~~~~

All options are set with ``-D<NAME>=<VALUE>`` on the cmake line.

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - Option
     - Default
     - Description
   * - ``ARTS_BUILD_SHARED``
     - ON
     - Build the shared library ``libarts.so`` alongside ``libarts.a``.  The
       static library is unconditional — it is the substrate the benchmark
       variants, the OCR shim and the installed archive derive from.
   * - ``ARTS_BUILD_EXAMPLES``
     - OFF (forced)
     - **Deprecated** — ``examples/`` predates the current public API and
       does not build; enabling is a configure error until the examples are
       modernized.  See ``tests/ocr/`` and ``benchmarks/`` for working usage.
   * - ``ARTS_BUILD_TESTS``
     - OFF
     - Build the test programs (registers them with ctest).  Off by
       default: tests statically link the runtime (hermetic), so a full
       test tree costs real disk and link time — turn it on for
       validation builds.
   * - ``ARTS_BUILD_BENCHMARKS``
     - ON
     - Build the benchmark apps (ARTS and the enabled reference runtimes).
   * - ``ARTS_BUILD_XSOCR``
     - ON
     - Build the XSOCR reference runtime and its ``*_xsocr`` app variants
       (needs a host MPI).
   * - ``ARTS_BUILD_OCRVX``
     - ON
     - Build the ocr-vx reference runtime and its ``*_ocrvx`` app variants
       (needs a host MPI).
   * - ``ARTS_BUILD_HPX``
     - ON
     - Source-build HPX v1.11.0 with its MPI parcelport and build the
       standalone HPX benchmark apps (needs a host MPI).
   * - ``ARTS_BUILD_BASELINES``
     - OFF
     - Build the native (non-OCR) OpenMP/MPI baseline implementations —
       outside the experiment driver's catalog, explicit opt-in.
   * - ``ARTS_BUILD_DOCS``
     - OFF
     - Build the Doxygen + Sphinx documentation.
   * - ``ARTS_USE_GPU``
     - OFF
     - Enable CUDA GPU support (requires the CUDA toolkit).
   * - ``ARTS_USE_LOCAL_CUDA_ARCHITECTURES``
     - ON
     - Auto-detect the local GPU's CUDA architecture via ``nvidia-smi`` (only
       when ``ARTS_USE_GPU=ON``; otherwise set ``CMAKE_CUDA_ARCHITECTURES``).
   * - ``ARTS_MEMORY_MODEL``
     - OCR
     - Memory model — ``OCR`` (default; implements the OCR v1.2.0 §1.6
       contract) or ``DB_WRF`` (write-race-free at DB granularity; evaluation
       only — emits a configure warning). Compile-time; all ranks must share
       one build.
   * - ``ARTS_COHERENCE_PROTOCOL``
     - VAL
     - Coherence family — ``VAL`` (default; validation: versioned snapshots,
       readers never blocked/invalidated), ``INV`` (invalidation: directory
       write-invalidate with acknowledged per-release invalidation rounds),
       or ``EXCL`` (exclusion: per-DB distributed reader-writer lock).
       Valid combos: OCR×{VAL,INV}×WT×{PURGE,RETAIN}, OCR×{VAL,INV}×WB×RETAIN,
       OCR×EXCL×WB×{PURGE,RETAIN}, DB_WRF×VAL×WT×RETAIN.
   * - ``ARTS_WRITE_POLICY``
     - WB
     - Write policy at release granularity, live in INV/VAL — ``WT``
       (write-through at release; the home holds a current copy and serves
       reads) or ``WB`` (write-back; the payload stays with the last writer
       and moves on demand).
   * - ``ARTS_RELEASE_POLICY``
     - RETAIN
     - Release policy, live in EXCL and in WT × {VAL, INV} — ``PURGE``
       (copy and permission return to the home when the last local user
       finishes) or ``RETAIN`` (keep both until another node asks; the home
       recalls on demand). ``WB`` requires ``RETAIN``.
   * - ``ARTS_NO_RO_COMBINING``
     - OFF
     - Ablation knob: compile the VAL-family read path WITHOUT
       requester-side combining. By default concurrent same-DB remote RO
       acquires share a single in-flight snapshot request per DB — part
       of the family's definition, matching the duplicate suppression
       every other message class has structurally (no effect under
       EXCL/INV, whose caches carry no combining window).
   * - ``ARTS_MALLOC``
     - mimalloc
     - General allocator — ``mimalloc`` (default; vendored static, the
       only multinode-capable choice) or ``system`` (libc malloc;
       single-node / sanitizer builds only).
   * - ``ARTS_DEFAULT_DB_KIND``
     - ARTS_DB
     - Default DB storage kind the ``ARTS_DB_DEFAULT`` macro expands to:
       ``ARTS_DB`` (regular DRAM) or ``ARTS_DB_CXL`` (CXL shared).
   * - ``ARTS_USE_CXL``
     - OFF
     - Enable CXL shared-memory DataBlocks (requires the Rapid API).
   * - ``ARTS_CXL_RAPID_INCLUDE_DIR``
     - (empty)
     - Path to the Rapid API includes (required when ``ARTS_USE_CXL=ON``).
   * - ``ARTS_CXL_LIB_DIR``
     - (empty)
     - Path to ``arts_cxl_lib`` (required when ``ARTS_USE_CXL=ON``).
   * - ``ARTS_NOHINT_EDT_PLACEMENT``
     - ROUNDROBIN
     - Where an EDT created with no placement preference (NULL hint, or
       hint rank ``ARTS_HINT_ANY_RANK``) lands — ``ROUNDROBIN`` (default)
       or ``CREATOR``.
   * - ``ARTS_NOHINT_DB_HOME``
     - CREATOR
     - Where a DB created with no placement preference (NULL hint) is
       homed — ``CREATOR`` (default; first-touch, home stays on the
       creating rank) or ``ROUNDROBIN``.  An explicit hint rank or
       pre-reserved GUID always wins.
   * - ``ARTS_SHIM_NOHINT_DB_HOME``
     - CREATOR
     - No-hint DB home policy for the OCR shim's ``ocrDbCreate`` —
       ``CREATOR`` (default; defers to ``ARTS_NOHINT_DB_HOME``) or
       ``ROUNDROBIN`` (its own counter, independent of the runtime policy).
   * - ``ARTS_SHIM_NOHINT_EDT_PLACE``
     - ROUNDROBIN
     - No-hint EDT placement policy for the OCR shim's ``ocrEdtCreate`` —
       ``ROUNDROBIN`` (default; defers to ``ARTS_NOHINT_EDT_PLACEMENT``) or
       ``CREATOR`` (pin to the calling rank).
   * - ``ARTS_LOG_LEVEL``
     - 3 / 1
     - Log verbosity (3 in Debug, 1 otherwise): 0=ERROR … 3=DEBUG.
   * - ``ARTS_USE_SANS``
     - OFF
     - ASan + UBSan + LSan in Debug builds (excludes CUDA; mutually exclusive
       with ``ARTS_USE_TSAN``).
   * - ``ARTS_USE_TSAN``
     - OFF
     - ThreadSanitizer in Debug builds (excludes CUDA; mutually exclusive with
       ``ARTS_USE_SANS``).
   * - ``ARTS_SEQUENCE_NUMBERS``
     - OFF
     - Per-message wire sequence-number drop/reorder diagnostic (per-send
       lock + per-recv check). Changes the wire header layout — every rank
       in a run must build with the same setting. Hot-path; enable only to
       debug transport ordering.
   * - ``ARTS_COUNTER_CONFIG``
     - configs/counters.cfg
     - Counter configuration file parsed into introspection macros.

To pick a faster linker, use CMake's own ``-DCMAKE_LINKER_TYPE=MOLD`` (cmake ≥ 3.29);
there is no ARTS-specific linker option.

See :ref:`coherence_protocols` for the normative definition of the protocols and their contracts.

GPU Build
~~~~~~~~~

.. code-block:: bash

   cmake -GNinja .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ROOT=$CUDAROOT

Verifying the Build
-------------------

Configure with ``-DARTS_BUILD_TESTS=ON`` and run the single-node test
suite as a smoke check (the ``examples/`` tree is deprecated and excluded
from the build — see the option table above):

.. code-block:: bash

   cmake .. -DARTS_BUILD_TESTS=ON && ninja
   ctest -L single_node --output-on-failure

Every test should report ``Passed``.
