# ARTS Runtime Overview

ARTS (Asynchronous Runtime System) is a distributed, event-driven runtime loosely based on the ideas pioneered in the **Open Community Runtime (OCR)**: work is expressed as Event-Driven Tasks (EDTs), data lives in datablocks identified by GUIDs, and dependencies are tracked through events instead of global synchronization. ARTS uses a CDAG-style memory model where each datablock has a canonical owner copy, remote updates flow via owner updates, and the runtime wires dependency graphs dynamically.

## What ARTS Provides

- **Event-Driven Tasks (EDTs)** – Lightweight units of work scheduled when their input dependencies are satisfied.
- **Datablocks (DBs)** – Explicit data objects with globally unique identifiers (GUIDs). They carry ownership/rendezvous information so ARTS can ship or replicate data across nodes and enforce the CDAG consistency rules.
- **Events & Dependencies** – OCR-style events connect producers/consumers. The runtime builds a dynamic DAG and triggers EDTs once all prereqs fire.
- **GUID system** – Every EDT, datablock, and event has a GUID so DAGs can be wired across nodes without global pointers.
- **Datablock lifecycle** – Applications allocate datablocks via `artsDbCreate`, pass GUIDs to EDTs, and the runtime handles acquire/release semantics (read/write modes, owner hand-offs). Reference counts and versioning live in `core/src/runtime/datablock/*`.
- **Distributed Scheduling** – A decentralized scheduler assigns EDTs to worker threads, maintains per-thread deques, supports work stealing, and cooperates with the network layer (`core/src/runtime/network`) to migrate work or data.
- **Networked DB protocol** – Messages for acquire/release/clone requests flow through configurable transports (shared-memory, MPI, or GASNet depending on build flags). The protocol keeps metadata (size, owner, access mode) alongside payloads so receivers can reconcile updates efficiently.

## Relationship to OCR

ARTS borrows heavily from OCR concepts:
- EDTs ↔ OCR tasks
- Datablocks ↔ OCR datablocks
- Events ↔ OCR events/slots
- GUIDs ↔ OCR GUIDs

However ARTS is purpose-built for this repository and trimmed to match its compiler/tooling integration: lean APIs in `core/include/arts/` and a GUID allocator tailored to cartesian DAGs.

## Dependencies

See `INSTALL.md` for detailed package lists. At a high level you need:
- A C/C++ compiler with OpenMP support (GCC or Clang)
- CMake + Ninja (preferred) or Make
- `libhwloc`, `libnuma`, pthreads
- Optional: MPI or GASNet if building networked backends

## Building

Follow `external/arts/INSTALL.md`. Typical steps:

```bash
cd external/arts
mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
ninja
```

The root CARTS build invokes this automatically when you configure `cmake` at the repository top level.

## Repository Layout (selected paths)

- `core/` – Runtime sources: task scheduler, GUID tables, datablock manager, network transports, logging.
- `cmake/` – Build helpers
- `sampleConfigs/` – Example `arts.cfg` files used by tests/benchmarks
- `example/` – Small standalone programs showing how to create EDTs/datablocks
- `benchmark/` – Runtime microbenchmarks
- `docs` – Installation and configuration notes (`INSTALL.md`)

## Learn More

- `INSTALL.md` – Build instructions and optional components
- `FULL_LICENSE.md` / `LICENSE.md` – Licensing information
- Example programs under `example/` for hands-on API references
