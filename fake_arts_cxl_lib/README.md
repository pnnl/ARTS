# fake_arts_cxl_lib

A single-node, shared-memory emulation of the ARTS CXL FAM (Fabric-Attached Memory) library. Lets you build and test ARTS applications on machines without CXL hardware by backing the FAM region with a POSIX shared-memory segment.

## Overview

Real ARTS CXL hardware exposes a FAM region at a fixed virtual address shared across all processes (and nodes) in a job. This library reproduces that contract using `shm_open` + `mmap` so that the same application code compiles and runs correctly on ordinary servers.

The shared library (`libarts_cxl_lib.so`) is a drop-in replacement: it exposes the same `SharedAlloc.h` and `MemOps.h` macros/functions, maps the FAM window to the same base address (`0x200000000000`), and optionally records every flush call to a binary trace file for offline analysis.

## Repository layout

```
inc/
  SharedAlloc.h       Public allocation API (macros + C functions)
  MemOps.h            Public flush/fence API (macros + C functions)
src/
  fake_cxl_internal.h Internal constants, struct definitions, shared state
  shm_backend.c       SHM creation, mmap at fixed address, bump-allocator core
  alloc.c             SharedAlloc.h implementation (thin wrappers over bump alloc)
  flush_track.c       Flush logging: in-memory ring buffer + on-exit binary dump
  fake_cxl.c          x86 clflushopt/clflush/sfence; compiler-barrier fallback elsewhere
test/
  smoke_test.c        Self-contained correctness test (run via CTest)
CMakeLists.txt
rapid-config.cmake    Stub CMake package that satisfies find_package(rapid CONFIG)
```

## Public API

Both headers live in `inc/` and are installed as the public include path of the CMake target.

### `SharedAlloc.h` — FAM allocation

| Macro | Description |
|---|---|
| `SHARED_CXL_MALLOC(sz)` | Allocate `sz` bytes from the FAM region |
| `GLOBAL_CXL_MALLOC(sz)` | Same as `SHARED_CXL_MALLOC` |
| `GLOBAL_CXL_MALLOC_DEV(sz, dev)` | Allocate on a specific device (single-device stub: `dev` is ignored) |
| `GLOBAL_CXL_FREE(ptr)` | No-op (bump allocator; FAM is reclaimed on process exit) |
| `SHARED_CXL_FREE(ptr)` | No-op |
| `LAST_SHARED_CXL_MALLOC(sz)` | "Last-rank" allocation — identical to `SHARED_CXL_MALLOC` on a single node |
| `SHARED_CXL_MALLOC_INITIALIZED(...)` | No-op (compatibility shim) |
| `IS_FAM_PTR(ptr)` | Returns non-zero if `ptr` falls within the FAM window |
| `GET_FAM_DEV_ID(ptr)` | Returns `0` (single device) |
| `GET_FAM_REGION_DEV_ID()` | Returns `0` |
| `GET_FAM_DEV_COUNT()` | Returns `1` |

Free is intentionally a no-op. The bump allocator advances a cursor and never reclaims individual objects; the entire region is released when the last process exits (via `shm_unlink` registered with `atexit`).

### `MemOps.h` — cache-line flush and fence

| Macro | Description |
|---|---|
| `FLUSH_FENCE_PRODUCER(ptr, size)` | `clflushopt` every cache line in `[ptr, ptr+size)` then `sfence`; records to flush log |
| `FLUSH_FENCE_CONSUMER(ptr, size)` | `clflush` every cache line in `[ptr, ptr+size)`; records to flush log |

On non-x86 platforms both macros compile to compiler barriers only (the fake region is ordinary DRAM so the CPU provides coherence).

## How it works

### Initialization (`shm_backend.c`)

A `__attribute__((constructor))` function runs when the shared library loads:

1. Reads `ARTS_FAKE_CXL_REGION_SIZE` (default 32 GiB, minimum 64 MiB).
2. Opens `/arts_fake_cxl` with `shm_open(O_CREAT|O_RDWR)` — the first process creates it; subsequent processes on the same node open the existing segment.
3. Maps the segment with `MAP_FIXED_NOREPLACE` at `0x200000000000` (the same address the real CXL hardware uses).
4. Initialises the allocator cursor in the region header with an atomic CAS so only the first process sets it to `sizeof(cxl_region_header_t)`; later processes skip the reset.
5. Registers `shm_unlink` via `atexit`.

### Bump allocator (`shm_backend.c` + `alloc.c`)

The first 64 bytes of the mapped region are a `cxl_region_header_t` holding a single `volatile uint64_t alloc_cursor`. `fake_cxl_bump_alloc` rounds the requested size up to a 64-byte cache-line boundary, then does an atomic fetch-add on the cursor. All allocation macros delegate to this function. There is no free list.

### Flush tracking (`flush_track.c`)

`flush_log_record` appends an entry to a statically-allocated ring buffer (`FLUSH_LOG_CAPACITY = 1 M entries = 32 MiB`). Each entry stores:

- `timestamp_ns` — `CLOCK_MONOTONIC` nanoseconds
- `thread_id` — `pthread_self()` cast to `uint64_t`
- `ptr`, `size` — the flushed range
- `type` — `0` = producer, `1` = consumer

A `__attribute__((destructor))` function writes the buffer to a binary file on process exit. The file path defaults to `arts_flush_trace.bin` in the working directory and can be overridden with `ARTS_FLUSH_LOG`.

Binary file format:

| Offset | Size | Content |
|---|---|---|
| 0 | 8 bytes | Magic `ARTSFCLX` (`0x4152545346434C58`) |
| 8 | 8 bytes | Entry count (uint64) |
| 16 | `count × 32` bytes | Array of `flush_entry_t` structs |

## Multi-process usage

Multiple distinct processes can attach to the same FAM region concurrently without any extra configuration.

### How processes share the region

- The first process to load the library creates the POSIX shared-memory segment `/arts_fake_cxl` and initialises the allocator cursor.
- Every subsequent process opens the same segment. `O_CREAT` is harmless when the segment already exists; the atomic CAS on the cursor ensures only the first process sets the initial value.
- All processes map the segment at the same fixed virtual address (`0x200000000000`), so FAM pointers can be passed between processes as plain integers — no translation required.
- The allocator cursor is inside the shared region, so `SHARED_CXL_MALLOC` / `GLOBAL_CXL_MALLOC` calls from different processes are serialised by the same atomic fetch-add. Each process receives a unique, non-overlapping allocation.

### Process startup ordering

Processes can start in any order. A late-joining process will see the cursor already advanced past allocations made by earlier processes, which is correct — those allocations belong to the processes that made them.

If your job manager starts all processes simultaneously, the first `shm_open` call races to create the segment. This is safe: `shm_open(O_CREAT|O_RDWR)` is atomic at the kernel level, and the CAS on the cursor handles the case where two processes both succeed in creating (or opening) the segment before either has initialised it.

### Process exit and the SHM segment lifetime

Each process registers `shm_unlink("/arts_fake_cxl")` via `atexit`. `shm_unlink` removes the name from the filesystem namespace but does not destroy the underlying memory — existing mappings in all still-running processes remain valid. The physical pages are freed only when the last mapping is unmapped (i.e. the last process holding the segment exits).

The practical consequence: **if the first process to exit is not the last process running**, the segment name is deleted and no new processes can attach. Processes already attached continue to work normally. To avoid this, ensure that a long-lived "coordinator" process is the last to exit, or start processes in a way that guarantees they all outlive the first one to finish.

### Flush logs in multi-process jobs

Each process maintains its own in-process flush ring buffer (in `flush_track.c`) and writes it to a separate file on exit. To avoid all processes writing to the same file, set `ARTS_FLUSH_LOG` to a per-process path:

```sh
ARTS_FLUSH_LOG=flush_rank_${MY_RANK}.bin ./my_arts_app
```

The files can be concatenated and sorted by `timestamp_ns` for a global trace.

## Building

```sh
cmake -B build -G Ninja
cmake --build build
```

The shared library is placed at `build/src/libarts_cxl_lib.so` (this path is expected by the ARTS build system).

Run the smoke test:

```sh
cd build && ctest --output-on-failure
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ARTS_FAKE_CXL_REGION_SIZE` | `34359738368` (32 GiB) | Size of the FAM shared-memory region in bytes. Must be at least 64 MiB. Accepts `0x`-prefixed hex. |
| `ARTS_FLUSH_LOG` | `arts_flush_trace.bin` | Path for the binary flush trace written on process exit. Set to an empty string or avoid the variable to still get the default path. |

## Integrating with the ARTS build system

ARTS detects the "Rapid" CXL SDK via `find_package(rapid CONFIG)`. The `rapid-config.cmake` stub satisfies this check and causes ARTS to define `ARTS_CXL_NATIVE`, activating the real CXL allocation code paths against this library.

Point ARTS at the fake library:

```sh
cmake -B build \
  -DARTS_CXL_RAPID_INCLUDE_DIR=/path/to/fake_arts_cxl_lib/inc \
  -DCMAKE_PREFIX_PATH=/path/to/fake_arts_cxl_lib \
  ...
```

ARTS appends `/..` to `ARTS_CXL_RAPID_INCLUDE_DIR` when calling `find_package`, so the directory above `inc/` must contain `rapid-config.cmake` — which it does in this repository.

## Limitations

- **Single node only.** The SHM segment is local to one machine. True CXL FAM is shared across nodes over fabric; that topology is not emulated.
- **No per-object free.** The bump allocator cannot reclaim individual allocations. Applications that rely on freeing FAM memory and reusing it will exhaust the region.
- **Single device.** All `dev_id` queries return `0` and `GET_FAM_DEV_COUNT()` returns `1`. Multi-device sub-region partitioning is a noted TODO in `alloc.c`.
- **Fixed virtual address.** The region must map at `0x200000000000`. If another mapping occupies that window the library will abort. Reduce `ARTS_FAKE_CXL_REGION_SIZE` or ensure the address is free.
