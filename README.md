# ARTS Runtime Overview

ARTS (Asynchronous Runtime System) is a distributed, event-driven runtime loosely based on the ideas pioneered in the **Open Community Runtime (OCR)**: work is expressed as Event-Driven Tasks (EDTs), data lives in datablocks identified by GUIDs, and dependencies are tracked through events instead of global synchronization. The runtime implements the OCR v1.2.0 §1.6 memory model: each DataBlock has a canonical owner, remote updates flow via ownership transfer, and the runtime wires dependency graphs dynamically.

## What ARTS Provides

- **Event-Driven Tasks (EDTs)** – Lightweight units of work scheduled when their input dependencies are satisfied.
- **Datablocks (DBs)** – Explicit data objects with globally unique identifiers (GUIDs). They carry ownership/rendezvous information so ARTS can ship or replicate data across nodes and enforce the OCR consistency contract.
- **Events & Dependencies** – OCR-style events connect producers/consumers. The runtime builds a dynamic DAG and triggers EDTs once all prereqs fire.
- **GUID system** – Every EDT, datablock, and event has a GUID so DAGs can be wired across nodes without global pointers.
- **Datablock lifecycle** – Applications allocate datablocks via `artsDbCreate`, pass GUIDs to EDTs, and the runtime handles acquire/release semantics (read/write modes, owner hand-offs). Reference counts and versioning live in `libs/core/src/runtime/datablock/*`.
- **Distributed Scheduling** – A decentralized scheduler assigns EDTs to worker threads, maintains per-thread deques, supports work stealing, and cooperates with the transport layer (`libs/src/core/transport/`) to migrate work or data.
- **Networked DB protocol** – Cross-rank DB acquire/release/publish and control messages ride a libfabric (OFI) RDM-endpoint transport (`libs/src/core/transport/net.c`); the provider is chosen via the config file's `provider` key (or the `FI_PROVIDER` env var), defaulting to `tcp`. DataBlock payloads are staged in a per-NUMA registered memory pool ("regpool") and delivered by one-sided RDMA into the receiver's pre-registered buffer; small control messages ride ordinary two-sided sends. Dedicated worker threads run EDTs while separate progress thread(s) drive completion. libfabric is vendored as a submodule and built in-tree, so no external networking library is required. A TCP socket mesh still exists, but only for process launch, the one-shot startup address exchange, and post-bootstrap liveness detection — no DB payload or protocol traffic crosses it.

## Relationship to OCR

ARTS borrows heavily from OCR concepts:
- EDTs ↔ OCR tasks
- Datablocks ↔ OCR datablocks
- Events ↔ OCR events/slots
- GUIDs ↔ OCR GUIDs

However ARTS is purpose-built for this repository and trimmed to match its compiler/tooling integration: lean APIs in `libs/core/include/` and a GUID allocator tailored to cartesian DAGs.

## Dependencies

See `INSTALL.md` for detailed package lists. At a high level you need:
- A C/C++ compiler with OpenMP support (GCC or Clang)
- CMake + Ninja (preferred) or Make
- `libhwloc`, `libnuma`, pthreads
- The libfabric (OFI) transport is a vendored submodule, built in-tree — no external networking library to install
- Optional: MPI, only for the xsocr/ocr-vx/baseline reference benchmark backends (not needed for ARTS itself)

## Building

Follow `external/arts/INSTALL.md`. Typical steps:

```bash
cd external/arts
mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
ninja
```

The root CARTS build invokes this automatically when you configure `cmake` at the repository top level.

### Build Options

All options are set on the cmake line with `-D<NAME>=<VALUE>`, e.g.
`cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release -DARTS_USE_GPU=ON`.

| Option | Default | Purpose |
|--------|---------|---------|
| `ARTS_BUILD_SHARED` | `ON` | Build the shared library `libarts.so`. |
| `ARTS_BUILD_STATIC` | `ON` | Build the static library `libarts.a`. |
| `ARTS_BUILD_EXAMPLES` | `OFF` (forced) | **Deprecated** — `examples/` predates the current public API and does not build; enabling is a configure error until the examples are modernized. See `tests/ocr/` and `benchmarks/` for working usage. |
| `ARTS_BUILD_TESTS` | `ON` | Build the test programs and register them with ctest. |
| `ARTS_BUILD_BENCHMARKS` | `ON` | Build the OCR benchmark apps (XSOCR + ARTS + ocrvx backends). |
| `ARTS_BUILD_XSOCR` | `ON` | Build the XSOCR reference runtime and its `*_xsocr` app variants (needs a host MPI). |
| `ARTS_BUILD_OCRVX` | `ON` | Build the ocr-vx reference runtime and its `*_ocrvx` app variants (needs a host MPI). |
| `ARTS_BUILD_BASELINES` | `OFF` | Build the native (non-OCR) OpenMP/MPI baseline implementations — outside the experiment driver's catalog, explicit opt-in. |
| `ARTS_BUILD_DOCS` | `OFF` | Build the Doxygen + Sphinx documentation. |
| `ARTS_USE_GPU` | `OFF` | Enable CUDA GPU support (builds `libarts_cuda`). |
| `ARTS_USE_LOCAL_CUDA_ARCHITECTURES` | `ON` | When GPU is on, auto-detect the local GPU's CUDA architecture via `nvidia-smi`. Only meaningful with `ARTS_USE_GPU=ON`; pair with the stock `CMAKE_CUDA_ARCHITECTURES` (e.g. `-DCMAKE_CUDA_ARCHITECTURES="80;86"`) to set SM targets by hand. |
| `ARTS_MEMORY_MODEL` | `OCR` | Memory model — `OCR` (default; implements the OCR v1.2.0 §1.6 contract) or `DB_WRF` (write-race-free at DB granularity: the program must event-order every write-write conflict on a DB; evaluation only — emits a configure warning). Compile-time; all ranks must share one build. |
| `ARTS_COHERENCE_PROTOCOL` | `VAL` | Coherence family — who keeps reader copies valid: `VAL` (default; acquire-time version validation, readers never blocked/tracked/invalidated), `INV` (release-time invalidation rounds), or `EXCL` (per-DB distributed reader-writer lock). Valid combos: OCR×{VAL,INV}×{WT,WB}, OCR×EXCL×WB×{PURGE,RETAIN}, DB_WRF×VAL×WT. |
| `ARTS_WRITE_POLICY` | `WB` | Write policy at release granularity — `WT` (write-through: payload flushed to the block's home at every release; home serves reads) or `WB` (default; write-back: payload stays with the last writer, directory forwards on demand). Live in INV/VAL; EXCL requires WB. |
| `ARTS_RELEASE_POLICY` | `RETAIN` | What a node does with its write grant when the last local user finishes — `PURGE` (hand copy and permission back to the home) or `RETAIN` (default; keep both until another node asks). Live in EXCL; INV/VAL require RETAIN. |
| `ARTS_DEFAULT_DB_KIND` | `ARTS_DB` | Default DB storage kind that the `ARTS_DB_DEFAULT` macro expands to — `ARTS_DB` (regular DRAM) or `ARTS_DB_CXL` (CXL shared). |
| `ARTS_USE_CXL` | `OFF` | Enable CXL shared-memory DataBlocks (requires the Rapid API). |
| `ARTS_CXL_RAPID_INCLUDE_DIR` | — | Path to the Rapid API include dir (required when `ARTS_USE_CXL=ON`). |
| `ARTS_CXL_LIB_DIR` | — | Path to the `arts_cxl_lib` dir (required when `ARTS_USE_CXL=ON`). |
| `ARTS_LOG_LEVEL` | `3` (Debug) / `1` (Release) | Log verbosity: `0`=ERROR, `1`=+WARN, `2`=+INFO, `3`=+DEBUG. |
| `ARTS_USE_SANS` | `OFF` | Enable ASan + UBSan + LSan in Debug builds (excludes CUDA). Mutually exclusive with `ARTS_USE_TSAN`. |
| `ARTS_USE_TSAN` | `OFF` | Enable ThreadSanitizer in Debug builds (excludes CUDA). Compiler-incompatible with `ARTS_USE_SANS`; use a separate build dir. |
| `ARTS_COUNTER_CONFIG` | `configs/counters.cfg` | Counter configuration file parsed at configure time into introspection macros. |

Standard CMake variables also apply: `CMAKE_BUILD_TYPE` (`Debug` default, or `Release`),
`CMAKE_INSTALL_PREFIX` (`./install` default), `CMAKE_CUDA_ARCHITECTURES` (see `ARTS_USE_LOCAL_CUDA_ARCHITECTURES`),
and `CMAKE_LINKER_TYPE` (cmake ≥ 3.29; e.g. `-DCMAKE_LINKER_TYPE=MOLD` to pick a faster linker like mold/lld/gold).

## Running

An ARTS program reads its runtime configuration from `arts.cfg` in the
working directory; the `ARTS_CONFIG` environment variable overrides the path:

```bash
ARTS_CONFIG=configs/local/test/1n.cfg ./my_program
```

Ready-to-run localhost shapes (single-node and 2/3/4-rank multinode) live in
`configs/local/test/`, and `configs/example.cfg` is the annotated reference
listing every recognized key. A `launcher=local` configuration spawns all
ranks on this machine and claims its own ports; the remote launchers (`ssh`,
and `slurm`/`lsf`, which are auto-detected from the scheduler's environment)
must name `ports` in the configuration. The full walk-through — writing and
linking a program, multi-node configurations, the complete key reference —
is in the Sphinx guides: `docs/getting_started/quickstart.rst` and
`docs/configuration/arts_cfg.rst`.

Benchmarks and experiment campaigns are driven by `artsrun` rather than run
by hand — see `tools/artsrun/README.md`.

## Repository Layout (selected paths)

- `libs/core/` – Runtime sources: task scheduler, GUID tables, datablock manager, network transports, logging.
- `cmake/` – Build helpers
- `configs/` – Runtime configuration files (`local/`, `test/`, `twosisters/` subdirs)
- `example/` – Small standalone programs showing how to create EDTs/datablocks
- `benchmark/` – Runtime microbenchmarks
- `docs` – Installation and configuration notes (`INSTALL.md`)

## Documentation

API reference and guides are built with Doxygen + Sphinx.

### Prerequisites

```bash
# Doxygen (if not already installed)
sudo apt install doxygen

# Python dependencies
pip install -r docs/requirements.txt
```

### Build

```bash
cmake -GNinja -Bbuild -DARTS_BUILD_DOCS=ON
ninja -C build docs
```

HTML output goes to `build/docs/sphinx/`.

### View locally

```bash
cd build/docs/sphinx/
python3 -m http.server 8000
```

Then open <http://localhost:8000> in a browser.

## Learn More

- `INSTALL.md` – Build instructions and optional components
- `FULL_LICENSE.md` / `LICENSE.md` – Licensing information
- Example programs under `example/` for hands-on API references
