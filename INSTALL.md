Table of Contents
=================

*   [Project Overview](#project-overview)
*   [Installation Guide](#installation-guide)
    *   [Environment Requirements](#environment-requirements)
    *   [Dependencies](#dependencies)
    *   [Building](#building)
    *   [Build Options](#build-options)
    *   [Memory Consistency Models](#memory-consistency-models)
*   [Running Tests](#running-tests)
*   [Configuration](#configuration)
*   [Contributors](#contributors)

Project Overview
================

**Project Name:** Abstract RunTime System (ARTS)

**Principal Investigator:** Joshua Suetterlein (Joshua.Suetterlein@pnnl.gov)

**General Area or Topic of Investigation:** Asynchronous Many-Task runtime (AMT)

**Release Number:** 2.0.0

ARTS is an asynchronous, distributed, event-driven task runtime based on OCR
concepts. Programs express work as **EDTs** (Event-Driven Tasks), data as
**DataBlocks** (DBs), and wiring as **Events** — all identified by globally
unique **GUIDs**. The runtime builds dynamic DAGs and schedules EDTs as their
dependencies are satisfied, providing a distributed global address space and a
distributed memory model with efficient synchronization for massively parallel
systems.

Installation Guide
==================

Environment Requirements
------------------------

**Languages:** C17 (runtime core), C++17, and optionally CUDA (GPU support).

**Operating System:** Linux (x86-64 or ARMv8.1+). The runtime relies on a
double-width compare-and-swap (`cmpxchg16b` on x86-64, `casp` on ARMv8.1+);
on x86-64 it is built with `-mcx16` and linked against `libatomic`.

**Compiler:** GCC >= 7 or Clang >= 5 (strict `-std=c17`, no GNU extensions).
Other compilers are not supported.

**Required Memory / Disk:** ~2 GB RAM and ~500 MB disk for a full build.

Dependencies
------------

| Name | Version | Required | Notes |
| ---- | ------- | -------- | ----- |
| CMake | >= 3.22 | Required | The build is CMake-driven. |
| Ninja | any | Required | ARTS enforces the Ninja generator; Make is **not** supported. |
| pthreads / librt / libm / libanl | system | Required | Provided by glibc on Linux. |
| libatomic | system | Required | For 16-byte atomics (DWCAS). |
| hwloc | >= 2.0 | Bundled | Built from source from the `third_party/hwloc` submodule — no system install needed. |
| CUDA Toolkit | >= 11 (tested) | Optional | Only when `ARTS_USE_GPU=ON`. |
| MPI | any | Optional | Only needed by the OCR reference benchmarks (`ARTS_BUILD_BENCHMARKS=ON`). |

Initialize submodules before the first build:

```bash
git clone <repository-url> arts
cd arts
git submodule update --init --recursive
```

Building
--------

ARTS *requires* the Ninja generator. The default install prefix is
`<project>/install`.

```bash
# Configure (Debug is the default build type)
cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release

# Build
ninja -C build

# Install (to CMAKE_INSTALL_PREFIX, default: <project>/install)
ninja -C build install
```

The public headers (`arts.h`, `arts/graph.h`, `arts/gpu.h`, `arts/array_db.h`)
and the `libarts` static/shared libraries are installed under the prefix, along
with a CMake package config so downstream projects can `find_package(ARTS)`.

Build Options
-------------

All options are set on the cmake line with `-D<NAME>=<VALUE>`. The full list
with defaults lives in [README.md](README.md#build-options); the most common are:

| Option | Default | Purpose |
| ------ | ------- | ------- |
| `CMAKE_BUILD_TYPE` | `Debug` | `Debug` or `Release`. |
| `ARTS_COHERENCE_PROTOCOL` | `MRNEW` | Coherence protocol (admission policy) — `MRNEW` (default, Multi-Reader Node-Exclusive Writer; implements the OCR v1.2.0 §1.6 contract) or `MRMW` (true multi-writer, lossy DB-DRF; evaluation only — emits a configure warning). Compile-time; all ranks must share one build. |
| `ARTS_PROTOCOL_TIMING` | `LAZY` | Timing of consistency actions (meaningful only for `MRNEW`) — `LAZY` (acquire-time, default) or `EAGER` (release-time). Ignored under `MRMW`. |
| `ARTS_USE_GPU` | `OFF` | Enable CUDA GPU support. |
| `ARTS_BUILD_TESTS` | `ON` | Build the ctest suite. |
| `ARTS_BUILD_BENCHMARKS` | `ON` | Build the OCR benchmark apps (needs MPI). |
| `ARTS_USE_SANS` | `OFF` | ASan + UBSan + LSan in Debug builds. |
| `ARTS_USE_TSAN` | `OFF` | ThreadSanitizer in Debug builds (mutually exclusive with `ARTS_USE_SANS`). |

A faster linker is selected with CMake's own `-DCMAKE_LINKER_TYPE=MOLD`
(cmake >= 3.29) — there is no ARTS-specific linker option.

Coherence Protocols
-------------------

DataBlock consistency behavior is controlled by one primary compile-time
knob and one conditional sub-knob. `ARTS_COHERENCE_PROTOCOL` selects the
**admission policy**: `MRNEW` (default, Multi-Reader Node-Exclusive Writer)
implements the OCR v1.2.0 §1.6 memory model; `MRMW` (true multi-writer,
lossy) is the weaker DB-DRF evaluation protocol that emits a configure-time
warning and can make racy-but-legal OCR programs yield wrong results.
`ARTS_PROTOCOL_TIMING` selects **when** consistency actions occur: `LAZY`
(acquire-time, default) or `EAGER` (release-time); it is meaningful only
under `MRNEW` and is ignored under `MRMW`. One binary is exactly one
configuration, and every rank in a multinode run must use the same build.
To cover all meaningful configurations:

```bash
cmake -GNinja -Bbuild_mrnew_eager -DCMAKE_BUILD_TYPE=Debug -DARTS_COHERENCE_PROTOCOL=MRNEW -DARTS_PROTOCOL_TIMING=EAGER
cmake -GNinja -Bbuild_mrnew_lazy  -DCMAKE_BUILD_TYPE=Debug -DARTS_COHERENCE_PROTOCOL=MRNEW -DARTS_PROTOCOL_TIMING=LAZY
cmake -GNinja -Bbuild_mrmw        -DCMAKE_BUILD_TYPE=Debug -DARTS_COHERENCE_PROTOCOL=MRMW
ninja -C build_mrnew_eager && ninja -C build_mrnew_lazy && ninja -C build_mrmw
```

Running Tests
=============

CTest is the test runner. Tests are grouped by ctest labels:

```bash
cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Debug
ninja -C build
( cd build && ctest -L single_node --output-on-failure )   # single-node
( cd build && ctest -L multinode  --output-on-failure )     # 2n/3n/4n localhost
( cd build && ctest -L gpu        --output-on-failure )     # GPU build only
ctest --test-dir build -R edt_create_basic                  # one test by name
```

Each test sets its own `ARTS_CONFIG` to point at the matching cfg under
`configs/local/laptop/`, so no config files are copied into the build directory.

Configuration
=============

An ARTS program reads its runtime configuration from `arts.cfg` in the working
directory (or the file named by the `ARTS_CONFIG` environment variable).
Templates live under `configs/`:

- `configs/local/laptop/{1n,2n,3n,4n,2n_io}.cfg` — localhost, 14-thread budget
- `configs/local/server/{1n,2n,4n,8n,16n,2n_io,4n_io,8n_io}.cfg` — localhost, 48-thread budget
- `configs/local/gpu/{1n,2n}.cfg` — GPU-enabled
- `configs/mpi/laptop/{1n,2n,3n,4n}.cfg` — MPI launcher (xsocr), 14-thread
- `configs/mpi/server/{1n,2n,4n,8n,16n}.cfg` — MPI launcher (xsocr), 48-thread

The values most often changed are the launcher, the worker/sender/receiver
thread counts, and the GPU count.

Contributors
============

### Main Team Members

1. Joshua Suetterlein, joshua.suetterlein@pnnl.gov
2. Joseph Manzano, joseph.manzano@pnnl.gov
3. Andres Marquez, andres.marquez@pnnl.gov

### Contributors

1. Vinay Amatya
2. Kiran Ranganath
3. Marcin Zalewski
4. Jesun Firoz
5. Vito Castellana
6. Marco Minutoli
7. Antonino Tumeo
8. John Feo
9. Andrew Lumsdaine
