Installation
============

This page covers building ARTS from source.

.. contents:: On this page
   :local:
   :depth: 2

Prerequisites
-------------

- **C17 compiler**: GCC >= 7 or Clang >= 5 (other compilers are not supported)
- **CMake** >= 3.12
- **Ninja** build system (Make is not supported)
- **POSIX threads** (pthreads)

Optional:

- **CUDA toolkit** for GPU support
- **hwloc** for hardware-topology-aware pinning

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

Debug build with sanitizers (enabled by default in Debug mode):

.. code-block:: bash

   cmake -GNinja .. -DCMAKE_BUILD_TYPE=Debug

CMake Options
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Option
     - Default
     - Description
   * - ``ARTS_USE_GPU``
     - ON
     - Enable GPU / CUDA support (requires CUDA toolkit; set to OFF to
       disable).
   * - ``ARTS_BUILD_EXAMPLES``
     - ON
     - Build example programs in ``examples/``.
   * - ``ARTS_BUILD_TESTS``
     - ON
     - Build test programs in ``tests/``.
   * - ``ARTS_USE_SANITIZERS``
     - ON
     - Enable address and undefined behavior sanitizers in Debug builds.

GPU Build
~~~~~~~~~

.. code-block:: bash

   cmake -GNinja .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ROOT=$CUDAROOT

Verifying the Build
-------------------

After building, run a quick test with the Fibonacci example:

.. code-block:: bash

   cd build/examples/cpu
   cp ../../sample_configs/arts.cfg .
   ./fib 10

Expected output shows the 10th Fibonacci number and timing info.
