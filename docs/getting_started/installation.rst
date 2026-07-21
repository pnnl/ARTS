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
     - Build the shared library ``libarts.so``.
   * - ``ARTS_BUILD_STATIC``
     - ON
     - Build the static library ``libarts.a``.
   * - ``ARTS_BUILD_EXAMPLES``
     - OFF
     - Build the example programs in ``examples/``.
   * - ``ARTS_BUILD_TESTS``
     - ON
     - Build the test programs (registers them with ctest).
   * - ``ARTS_BUILD_BENCHMARKS``
     - ON
     - Build the OCR benchmark apps (XSOCR + ARTS + ocrvx).
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
     - RCU
     - Coherence protocol — ``RCU`` (default; versioned snapshots, readers
       never blocked/invalidated) or ``RWLOCK`` (per-DB distributed
       reader-writer lock). Valid combos: OCR×RCU×{E,L}, OCR×RWLOCK×{E,L},
       DB_WRF×RCU×EAGER.
   * - ``ARTS_PROTOCOL_TIMING``
     - LAZY
     - Timing of consistency actions — ``LAZY`` (acquire-time, default) or
       ``EAGER`` (release-time).
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

After building, run a quick test with the Fibonacci example:

.. code-block:: bash

   cd build/examples/cpu
   ARTS_CONFIG=../../../configs/local/1n.cfg ./fib 10

Expected output shows the 10th Fibonacci number and timing info.
