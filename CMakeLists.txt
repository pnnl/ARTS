###############################################################################
## This material was prepared as an account of work sponsored by an agency
## of the United States Government.  Neither the United States Government
## nor the United States Department of Energy, nor Battelle, nor any of
## their employees, nor any jurisdiction or organization that has cooperated
## in the development of these materials, makes any warranty, express or
## implied, or assumes any legal liability or responsibility for the accuracy,
## completeness, or usefulness or any information, apparatus, product,
## software, or process disclosed, or represents that its use would not
## infringe privately owned rights.
##
## Reference herein to any specific commercial product, process, or service
## by trade name, trademark, manufacturer, or otherwise does not necessarily
## constitute or imply its endorsement, recommendation, or favoring by the
## United States Government or any agency thereof, or Battelle Memorial
## Institute. The views and opinions of authors expressed herein do not
## necessarily state or reflect those of the United States Government or
## any agency thereof.
##
##                      PACIFIC NORTHWEST NATIONAL LABORATORY
##                                  operated by
##                                    BATTELLE
##                                    for the
##                      UNITED STATES DEPARTMENT OF ENERGY
##                         under Contract DE-AC05-76RL01830
##
## Copyright 2019 Battelle Memorial Institute
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    https://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
###############################################################################

cmake_minimum_required(VERSION 3.22)
project(ARTS
    VERSION 2.0.0
    DESCRIPTION "ARTS - Abstract Runtime System"
    LANGUAGES C CXX
)

# Enforce Ninja generator
if(NOT CMAKE_GENERATOR STREQUAL "Ninja")
    message(FATAL_ERROR
        "ARTS requires the Ninja build generator.\n"
        "  Use: cmake -GNinja <source-dir>")
endif()

# C17 strict mode — no GNU extensions
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Feature test macro: strict -std=c17 hides POSIX/BSD/XSI functions.
# _DEFAULT_SOURCE enables glibc defaults (POSIX.1-2008 + XSI + BSD + extensions)
# while CMAKE_C_EXTENSIONS=OFF keeps the C17 *language* strict.
add_compile_definitions(_DEFAULT_SOURCE _GNU_SOURCE)

# 128-bit compare-and-swap, per architecture:
#  - x86-64: cmpxchg16b, enabled by -mcx16.
#  - ARM64 (AArch64): ARMv8.1-a LSE gives an inline lock-free `casp`.  Without
#    it, 16-byte atomics fall back to a libatomic lock (NOT lock-free), which
#    would silently break the DWCAS contract.  (-moutline-atomics is the
#    alternative if an ARMv8.0 baseline must be supported.)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-mcx16>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mcx16>)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-march=armv8.1-a>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=armv8.1-a>)
endif()

# DWCAS feature detection (cmpxchg16b on x86_64, casp on ARMv8.1+).
# Required by lockfree_pool.h for DWCAS.  GCC and Clang
# emit a libcall (`__atomic_compare_exchange_16`) for 16-byte atomics
# even when -mcx16 is set, so we link against libatomic.  Detection
# probes the linker pipeline (compile + link) by running CheckCSourceCompiles
# with -latomic in CMAKE_REQUIRED_LIBRARIES.
include(CheckCSourceCompiles)
set(_arts_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
set(_arts_saved_required_libs  "${CMAKE_REQUIRED_LIBRARIES}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -mcx16")
endif()
# Some distributions (e.g., Debian without libatomic-dev) ship only
# `libatomic.so.1` without the canonical `libatomic.so` symlink, which
# defeats `find_library(NAMES atomic)`.  Probe for the SONAME directly
# as a fallback.
find_library(ATOMIC_LIBRARY NAMES atomic)
if(NOT ATOMIC_LIBRARY)
    find_file(ATOMIC_LIBRARY_SONAME libatomic.so.1
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib /usr/lib64
        NO_DEFAULT_PATH)
    if(ATOMIC_LIBRARY_SONAME)
        set(ATOMIC_LIBRARY "${ATOMIC_LIBRARY_SONAME}" CACHE FILEPATH
            "Path to libatomic" FORCE)
    endif()
endif()
if(ATOMIC_LIBRARY)
    set(CMAKE_REQUIRED_LIBRARIES "${ATOMIC_LIBRARY}" ${CMAKE_REQUIRED_LIBRARIES})
endif()
check_c_source_compiles("
  #include <stdint.h>
  int main(void) {
    __int128 a __attribute__((aligned(16))) = 0;
    __int128 b __attribute__((aligned(16))) = 0;
    return __atomic_compare_exchange_n(&a, &b, (__int128)1,
                                       0, __ATOMIC_SEQ_CST,
                                       __ATOMIC_SEQ_CST);
  }
" ARTS_HAVE_DWCAS)
set(CMAKE_REQUIRED_FLAGS "${_arts_saved_required_flags}")
set(CMAKE_REQUIRED_LIBRARIES "${_arts_saved_required_libs}")
unset(_arts_saved_required_flags)
unset(_arts_saved_required_libs)
if(NOT ARTS_HAVE_DWCAS)
    message(FATAL_ERROR
        "ARTS requires a LOCK-FREE 128-bit CAS (cmpxchg16b on x86-64, casp on "
        "ARMv8.1+).  Build with -mcx16 (AMD64) or -march=armv8.1-a / +lse / "
        "-moutline-atomics (ARM64).")
endif()
add_compile_definitions(ARTS_HAVE_DWCAS)
# Make libatomic available to all targets that may emit 16-byte atomic
# libcalls (lockfree_pool.h users).
if(ATOMIC_LIBRARY)
    link_libraries(${ATOMIC_LIBRARY})
endif()

# Set global properties for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default install prefix if not specified
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Install prefix" FORCE)
    message(STATUS "Install prefix not specified, defaulting to ${CMAKE_INSTALL_PREFIX}")
else()
    message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
    message(STATUS "Build type not specified, defaulting to Debug")
endif()

# Include utilities for modern CMake
include(GNUInstallDirs)
include(CheckLanguage)

# ═══════════════════════════════════════════════════════════════════════════
# Build-time options  (override on the cmake line with -D<NAME>=<VALUE>)
# ═══════════════════════════════════════════════════════════════════════════
include(CMakeDependentOption)

# ── Library targets ────────────────────────────────────────────────────────
option(ARTS_BUILD_SHARED "Build the shared library (libarts.so)" ON)
option(ARTS_BUILD_STATIC "Build the static library (libarts.a)"  ON)

# ── Optional components ────────────────────────────────────────────────────
option(ARTS_BUILD_EXAMPLES   "Build example programs"                          OFF)
option(ARTS_BUILD_TESTS      "Build test programs (enables ctest)"             ON)
option(ARTS_BUILD_BENCHMARKS "Build OCR benchmark apps (XSOCR + ARTS + ocrvx)" ON)
option(ARTS_BUILD_DOCS       "Build documentation (requires Doxygen + Sphinx)" OFF)

# ── Transport ──────────────────────────────────────────────────────────────
option(ARTS_TRANSPORT_OFI "libfabric (OFI) transport" ON)

# ── GPU / CUDA ─────────────────────────────────────────────────────────────
# ARTS_USE_LOCAL_CUDA_ARCHITECTURES only applies when ARTS_USE_GPU=ON;
# cmake_dependent_option forces it OFF (and hides it) otherwise.  The related
# stock CMake variable CMAKE_CUDA_ARCHITECTURES picks the SM targets nvcc emits
# (e.g. "80;86"): when the option below is ON it is auto-filled from nvidia-smi,
# otherwise nvcc's default applies (or your own -DCMAKE_CUDA_ARCHITECTURES=...).
option(ARTS_USE_GPU "Enable CUDA GPU support" OFF)
cmake_dependent_option(ARTS_USE_LOCAL_CUDA_ARCHITECTURES
    "Auto-detect CUDA architectures from local GPUs using nvidia-smi" ON
    "ARTS_USE_GPU" OFF)

# ── Memory model × coherence protocol × data placement ─────────────────────
# Compile-time switches — one binary is exactly one configuration; all ranks in
# a multinode run must use the same build.
#   ARTS_MEMORY_MODEL        OCR (default) — the OCR v1.2.0 §1.6 memory model:
#                            races are legal; the runtime orders all conflicting
#                            accesses it must (word-granular race semantics).
#                            DB_WRF — write-race-free at datablock granularity:
#                            the PROGRAM must order every write-write conflict
#                            on a DB via events; read-write races keep the same
#                            word-granular (regular-register) reader floor.
#   ARTS_COHERENCE_PROTOCOL  Coherence family — who keeps reader copies valid:
#                            VAL (default) — validation: each read acquire
#                            checks its cached copy's version at the serving
#                            side and refetches only when stale; readers are
#                            never blocked, tracked, or invalidated.
#                            INV — invalidation: reader copies stay valid until
#                            a writer's release retires every standing copy in
#                            an acknowledged invalidation round.
#                            EXCL — exclusion: queue-fair distributed
#                            reader-writer lock per DB; readers and writers
#                            take turns, so no stale copy ever exists.
#   ARTS_WRITE_POLICY        WT | WB (default) — write policy of a node's
#                            data-block cache at release granularity: WT =
#                            write-through (payload flushed to the block's home
#                            at every release; home serves reads), WB =
#                            write-back (payload stays with the last writer;
#                            directory forwards on demand). Live in INV/VAL.
#   ARTS_RELEASE_POLICY      PURGE | RETAIN (default) — what a node does with
#                            its grant when the last local user finishes:
#                            PURGE = hand copy and permission back to the home;
#                            RETAIN = keep both until another node asks. Live
#                            in EXCL.
# The axes are nominally independent: 12 combinations, 6 built. Everything
# else is a configure error naming the reason:
#   OCR    × VAL  × {WT, WB} × RETAIN
#   OCR    × INV  × {WT, WB} × RETAIN
#   OCR    × EXCL × WB × {PURGE, RETAIN}
#   DB_WRF × VAL  × WT × RETAIN        (home-canonical sync publish)
# Not built: EXCL×WT (the payload rides the permission, so the release policy
# already decides both), INV/VAL×PURGE (the migrating write grant is
# metadata-only; giving it up early saves no data movement).
set(ARTS_MEMORY_MODEL "OCR" CACHE STRING
    "Memory model the build implements: OCR (default; races legal, runtime-ordered) or DB_WRF (program orders write-write conflicts per DB)")
set_property(CACHE ARTS_MEMORY_MODEL PROPERTY STRINGS OCR DB_WRF)
set(ARTS_COHERENCE_PROTOCOL "VAL" CACHE STRING
    "Coherence family: VAL (default; acquire-time version validation, readers never blocked/tracked/invalidated), INV (release-time invalidation rounds), or EXCL (per-DB distributed reader-writer lock)")
set_property(CACHE ARTS_COHERENCE_PROTOCOL PROPERTY STRINGS VAL INV EXCL)
set(ARTS_WRITE_POLICY "WB" CACHE STRING
    "Write policy at release granularity: WT (flush to the home at every release) or WB (resident with the last writer, default). Live in INV/VAL; EXCL requires WB")
set_property(CACHE ARTS_WRITE_POLICY PROPERTY STRINGS WT WB)
set(ARTS_RELEASE_POLICY "RETAIN" CACHE STRING
    "Grant policy at the last local release: PURGE (hand copy and permission back to the home) or RETAIN (keep both until asked, default). Live in EXCL; INV/VAL require RETAIN")
set_property(CACHE ARTS_RELEASE_POLICY PROPERTY STRINGS PURGE RETAIN)
option(ARTS_RO_REQUEST_COMBINING
    "Combine concurrent same-DB remote read acquires into a single in-flight snapshot request per DB (VAL pull path; no effect under EXCL)" OFF)
set(ARTS_MALLOC "mimalloc" CACHE STRING
    "General allocator: mimalloc (default, vendored static) or system (libc malloc)")
set_property(CACHE ARTS_MALLOC PROPERTY STRINGS mimalloc system)

# ── Default DB storage kind ────────────────────────────────────────────────
set(ARTS_DEFAULT_DB_KIND "ARTS_DB" CACHE STRING
    "Default DB storage kind the ARTS_DB_DEFAULT macro expands to: ARTS_DB (regular DRAM) or ARTS_DB_CXL (CXL shared)")
set_property(CACHE ARTS_DEFAULT_DB_KIND PROPERTY STRINGS ARTS_DB ARTS_DB_CXL)

# ── CXL shared memory (requires the Rapid API) ─────────────────────────────
option(ARTS_USE_CXL "Enable CXL shared-memory DB support" OFF)
set(ARTS_CXL_RAPID_INCLUDE_DIR "" CACHE PATH
    "Path to Rapid API include directory (required when ARTS_USE_CXL=ON)")
set(ARTS_CXL_LIB_DIR "" CACHE PATH
    "Path to arts_cxl_lib directory (required when ARTS_USE_CXL=ON)")

# ── No-hint placement policies (runtime-wide) ───────────────────────────────
# Select what the runtime does when the caller expresses no placement
# preference; an explicit hint always wins.  The defaults differ on purpose:
# an EDT chain seeded on one rank can never bootstrap distribution by staying
# with its creator, so no-hint EDTs round-robin; a DB created by an
# already-placed EDT inherits that distribution, so no-hint DB homes stay
# with the creator (first-touch).
set(ARTS_NOHINT_EDT_PLACEMENT "ROUNDROBIN" CACHE STRING
    "No-hint EDT placement policy (NULL hint or rank == ARTS_HINT_ANY_RANK): \
ROUNDROBIN (default) or CREATOR")
set_property(CACHE ARTS_NOHINT_EDT_PLACEMENT PROPERTY STRINGS ROUNDROBIN CREATOR)
if(ARTS_NOHINT_EDT_PLACEMENT STREQUAL "CREATOR")
  set(_ARTS_NOHINT_EDT_RR 0)
else()
  set(_ARTS_NOHINT_EDT_RR 1)
endif()

set(ARTS_NOHINT_DB_HOME "CREATOR" CACHE STRING
    "No-hint DB home policy (NULL hint; explicit hint rank or pre-reserved \
GUID always wins): CREATOR (default; first-touch — home stays on the \
creating rank) or ROUNDROBIN (distribute homes across ranks)")
set_property(CACHE ARTS_NOHINT_DB_HOME PROPERTY STRINGS CREATOR ROUNDROBIN)
if(ARTS_NOHINT_DB_HOME STREQUAL "ROUNDROBIN")
  set(_ARTS_NOHINT_DB_RR 1)
else()
  set(_ARTS_NOHINT_DB_RR 0)
endif()

# ── OCR shim (benchmarks) ───────────────────────────────────────────────────
set(ARTS_SHIM_NOHINT_DB_HOME "CREATOR" CACHE STRING
    "No-hint DB home policy for the OCR shim's ocrDbCreate (no OCR affinity \
hint given): CREATOR (default; first-touch — defers to arts_db_create's own \
no-hint policy) or ROUNDROBIN (explicit round-robin home distribution)")
set_property(CACHE ARTS_SHIM_NOHINT_DB_HOME PROPERTY STRINGS CREATOR ROUNDROBIN)
if(ARTS_SHIM_NOHINT_DB_HOME STREQUAL "ROUNDROBIN")
  set(_ARTS_SHIM_NOHINT_RR 1)
else()
  set(_ARTS_SHIM_NOHINT_RR 0)
endif()

set(ARTS_SHIM_NOHINT_EDT_PLACE "ROUNDROBIN" CACHE STRING
    "No-hint EDT placement policy for the OCR shim's ocrEdtCreate (no OCR \
affinity hint given): ROUNDROBIN (default, defers to ARTS_NOHINT_EDT_PLACEMENT) \
or CREATOR (pin to the calling rank)")
set_property(CACHE ARTS_SHIM_NOHINT_EDT_PLACE PROPERTY STRINGS ROUNDROBIN CREATOR)
if(ARTS_SHIM_NOHINT_EDT_PLACE STREQUAL "CREATOR")
  set(_ARTS_SHIM_NOHINT_EDT_RR 0)
else()
  set(_ARTS_SHIM_NOHINT_EDT_RR 1)
endif()

# ── Diagnostics ────────────────────────────────────────────────────────────
# Default log level: Debug -> 3 (DEBUG), other build types -> 1 (WARN).
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_arts_log_default 3)
else()
    set(_arts_log_default 1)
endif()
if(DEFINED ARTS_LOG_LEVEL)
    set(_arts_log_level "${ARTS_LOG_LEVEL}")
else()
    set(_arts_log_level "${_arts_log_default}")
endif()
set(ARTS_LOG_LEVEL "${_arts_log_level}" CACHE STRING
    "Log verbosity: 0=ERROR, 1=+WARN, 2=+INFO, 3=+DEBUG" FORCE)
set_property(CACHE ARTS_LOG_LEVEL PROPERTY STRINGS 0 1 2 3)
if(NOT ARTS_LOG_LEVEL MATCHES "^[0-3]$")
    message(FATAL_ERROR
        "Invalid ARTS_LOG_LEVEL='${ARTS_LOG_LEVEL}'. Expected one of: 0, 1, 2, 3")
endif()
# ASan+UBSan+LSan and TSan are compiler-mutually-exclusive (ASan and LSan both
# conflict with -fsanitize=thread); they need separate build trees.  Wiring is
# applied in the Debug compiler-flags block below.
option(ARTS_USE_SANS "Enable ASan + UBSan + LSan in Debug builds (excludes CUDA)" OFF)
option(ARTS_USE_TSAN "Enable ThreadSanitizer in Debug builds (excludes CUDA; mutually exclusive with ARTS_USE_SANS)" OFF)

# Wire sequence numbers: a per-message seq_num field + a per-send lock around
# (assign-seq, enqueue) + a per-receive wire-ordering check — a drop/reorder
# diagnostic.  This is HOT-PATH (every send and receive) and OFF by default;
# turn it on only to debug transport ordering.  It changes the wire packet
# header layout, so every rank in a run must use the same setting.
option(ARTS_SEQUENCE_NUMBERS "Enable per-message wire sequence-number drop/reorder diagnostic (per-send lock + per-recv check; OFF for performance)" OFF)
if(ARTS_SEQUENCE_NUMBERS)
  add_compile_definitions(SEQUENCENUMBERS=1)
endif()

# ── Build tooling ──────────────────────────────────────────────────────────
# (To pick a faster linker, use CMake's own -DCMAKE_LINKER_TYPE=MOLD/LLD/GOLD
#  (cmake >= 3.29) or -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=mold — no ARTS option.)
set(ARTS_COUNTER_CONFIG
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/counters.cfg"
    CACHE FILEPATH "ARTS counter configuration file (parsed to generate introspection macros)")

# ── Memory-model × family × second-axes validation + arm derivation ───────
# The user-facing axes reduce to one protocol ARM — the TU family that gets
# compiled and the ARTS_PROTOCOL_* macro it defines:
#   (OCR,    VAL,  WT/WB)  -> arm VAL      (coherence/val/,     ARTS_PROTOCOL_VAL)
#   (OCR,    INV,  WT/WB)  -> arm INV      (coherence/inv/,     ARTS_PROTOCOL_INV)
#   (OCR,    EXCL, P/R)    -> arm EXCL     (coherence/excl/,    ARTS_PROTOCOL_EXCL)
#   (DB_WRF, VAL,  WT)     -> arm WRF_VAL  (coherence/wrf_val/, ARTS_PROTOCOL_WRF_VAL)
# ARTS_COHERENCE_ARM is internal — downstream CMake (TU selection, tests)
# branches on the arm, never on the raw axes.
# Migration guards: pre-rename axis values.
if(ARTS_COHERENCE_PROTOCOL STREQUAL "MRNEW" OR ARTS_COHERENCE_PROTOCOL STREQUAL "MRMW" OR
   ARTS_COHERENCE_PROTOCOL STREQUAL "MRSW" OR ARTS_COHERENCE_PROTOCOL STREQUAL "LOCK" OR
   ARTS_COHERENCE_PROTOCOL STREQUAL "RCU" OR ARTS_COHERENCE_PROTOCOL STREQUAL "RWLOCK" OR
   ARTS_COHERENCE_PROTOCOL STREQUAL "MSI")
  message(FATAL_ERROR
          "ARTS_COHERENCE_PROTOCOL=${ARTS_COHERENCE_PROTOCOL} was renamed "
          "(2026-07-31 terminology pass). Use the family names: "
          "RCU -> VAL, MSI -> INV, RWLOCK -> EXCL. "
          "(Older single-axis values: MRNEW -> VAL, LOCK -> EXCL, "
          "MRMW -> ARTS_MEMORY_MODEL=DB_WRF ARTS_COHERENCE_PROTOCOL=VAL "
          "ARTS_WRITE_POLICY=WT. MRSW was retired, archive/coherence-mrsw/.)")
endif()
if(DEFINED ARTS_DATA_PLACEMENT)
  message(FATAL_ERROR
          "ARTS_DATA_PLACEMENT was split (2026-07-31 terminology pass) into "
          "ARTS_WRITE_POLICY=WT|WB (live in INV/VAL; old HOME -> WT, "
          "OWNER -> WB) and ARTS_RELEASE_POLICY=PURGE|RETAIN (live in EXCL; "
          "old HOME -> PURGE, OWNER -> RETAIN).")
endif()
if(NOT ARTS_MEMORY_MODEL STREQUAL "OCR" AND NOT ARTS_MEMORY_MODEL STREQUAL "DB_WRF")
  message(FATAL_ERROR
          "ARTS_MEMORY_MODEL must be OCR or DB_WRF, got: ${ARTS_MEMORY_MODEL}")
endif()
if(NOT ARTS_COHERENCE_PROTOCOL STREQUAL "VAL" AND
   NOT ARTS_COHERENCE_PROTOCOL STREQUAL "INV" AND
   NOT ARTS_COHERENCE_PROTOCOL STREQUAL "EXCL")
  message(FATAL_ERROR
          "ARTS_COHERENCE_PROTOCOL must be VAL, INV or EXCL, got: ${ARTS_COHERENCE_PROTOCOL}")
endif()
if(NOT ARTS_WRITE_POLICY STREQUAL "WT" AND NOT ARTS_WRITE_POLICY STREQUAL "WB")
  message(FATAL_ERROR
          "ARTS_WRITE_POLICY must be WT or WB, got: ${ARTS_WRITE_POLICY}")
endif()
if(NOT ARTS_RELEASE_POLICY STREQUAL "PURGE" AND NOT ARTS_RELEASE_POLICY STREQUAL "RETAIN")
  message(FATAL_ERROR
          "ARTS_RELEASE_POLICY must be PURGE or RETAIN, got: ${ARTS_RELEASE_POLICY}")
endif()
# Per-family live axis: the dead axis must sit at its required value.
if(ARTS_COHERENCE_PROTOCOL STREQUAL "EXCL" AND ARTS_WRITE_POLICY STREQUAL "WT")
  message(FATAL_ERROR
          "EXCL x WT is not built: under exclusion no copy outlives a write "
          "turn, so the payload rides the permission and the release policy "
          "(ARTS_RELEASE_POLICY=PURGE|RETAIN) already decides both. "
          "Use ARTS_WRITE_POLICY=WB.")
endif()
if((ARTS_COHERENCE_PROTOCOL STREQUAL "INV" OR ARTS_COHERENCE_PROTOCOL STREQUAL "VAL")
   AND ARTS_RELEASE_POLICY STREQUAL "PURGE")
  message(FATAL_ERROR
          "${ARTS_COHERENCE_PROTOCOL} x PURGE is not built: the migrating "
          "write grant is metadata-only in the sharer-bearing families, so "
          "returning it early saves no data movement and forfeits the "
          "message-free re-acquisition of a repeated local writer. "
          "Use ARTS_RELEASE_POLICY=RETAIN.")
endif()
if(ARTS_COHERENCE_PROTOCOL STREQUAL "INV" AND ARTS_MEMORY_MODEL STREQUAL "DB_WRF")
  message(FATAL_ERROR
          "DB_WRF x INV is not a meaningful configuration: INV already "
          "orders writers at runtime (per-release rounds), so the DB-WRF "
          "program obligation is redundant — use ARTS_MEMORY_MODEL=OCR.")
endif()
if(ARTS_MEMORY_MODEL STREQUAL "DB_WRF")
  if(ARTS_COHERENCE_PROTOCOL STREQUAL "EXCL")
    message(FATAL_ERROR
            "DB_WRF x EXCL is not a meaningful configuration: EXCL already "
            "serializes writer-writer at runtime, so the DB-WRF program "
            "obligation is redundant — use ARTS_MEMORY_MODEL=OCR "
            "ARTS_COHERENCE_PROTOCOL=EXCL.")
  endif()
  if(ARTS_WRITE_POLICY STREQUAL "WB")
    message(FATAL_ERROR
            "DB_WRF x VAL x WB is designed but unimplemented (owner-held "
            "canonical copy, control-only release). The implemented DB_WRF "
            "configuration is ARTS_WRITE_POLICY=WT (home-canonical sync "
            "publish at release).")
  endif()
  set(ARTS_COHERENCE_ARM "WRF_VAL")
  message(WARNING
          "ARTS_MEMORY_MODEL=DB_WRF: write-write conflicts on a DB must be "
          "event-ordered BY THE PROGRAM (whole-DB granularity — unordered "
          "writers to disjoint regions of one DB are also outside the "
          "contract). Programs valid under the OCR memory model may produce "
          "wrong results here. Evaluation builds only — never a correctness "
          "baseline.")
  message(STATUS "ARTS: model DB_WRF, family VAL (arm WRF_VAL), write policy WT")
elseif(ARTS_COHERENCE_PROTOCOL STREQUAL "INV")
  # INV (invalidation): readers are never blocked, tracked only as a
  # node-granular copy roster, and every RW release retires every valid copy
  # in an invalidation round whose acks it collects before returning.  WT
  # additionally publishes each release to the home (home-canonical); WB
  # keeps the canonical copy at the owner and moves it owner-to-owner by
  # migration (home is a pure directory that redirects reads), so it drops the
  # publish and its ack — never the invalidate ack.
  set(ARTS_COHERENCE_ARM "INV")
  message(STATUS "ARTS: model OCR, family INV, write policy ${ARTS_WRITE_POLICY}")
elseif(ARTS_COHERENCE_PROTOCOL STREQUAL "EXCL")
  # EXCL (exclusion, entry-consistency style): a DB is held in one access mode
  # at a time (RO-shared or RW-exclusive); reader/writer are time-exclusive;
  # writers are passive (park, no invalidate push); all inter-node transitions
  # serialize through the GUID home's single lock_state word.  The release
  # policy decides what the zero edge does: PURGE returns grant and payload to
  # the home, RETAIN keeps both until the home recalls them.
  set(ARTS_COHERENCE_ARM "EXCL")
  message(STATUS "ARTS: model OCR, family EXCL, release policy ${ARTS_RELEASE_POLICY}")
else()
  set(ARTS_COHERENCE_ARM "VAL")
  message(STATUS "ARTS: model OCR, family VAL, write policy ${ARTS_WRITE_POLICY}")
endif()
if(ARTS_RO_REQUEST_COMBINING)
  # Requester-side combining of the MVCC-family remote read path: while one
  # snapshot request per DB is in flight, later same-rank read acquires park
  # and share its response instead of issuing their own round trips.  Pure
  # pull — no reader state at the home, no writer-side invalidation.
  add_compile_definitions(ARTS_RO_REQUEST_COMBINING=1)
  message(STATUS "ARTS RO request combining: ON (one in-flight snapshot request per DB)")
endif()
if(ARTS_MALLOC STREQUAL "mimalloc")
  add_compile_definitions(ARTS_MALLOC_MIMALLOC=1)
  message(STATUS "ARTS allocator: mimalloc (vendored static)")
elseif(ARTS_MALLOC STREQUAL "system")
  message(STATUS "ARTS allocator: system (libc malloc)")
else()
  message(FATAL_ERROR "ARTS_MALLOC must be 'mimalloc' or 'system', got '${ARTS_MALLOC}'")
endif()
if(NOT ARTS_DEFAULT_DB_KIND STREQUAL "ARTS_DB" AND
   NOT ARTS_DEFAULT_DB_KIND STREQUAL "ARTS_DB_CXL")
  message(FATAL_ERROR
          "ARTS_DEFAULT_DB_KIND must be ARTS_DB or ARTS_DB_CXL "
          "(node-pinned / GPU storage classes are not valid global defaults), "
          "got: ${ARTS_DEFAULT_DB_KIND}")
endif()
add_compile_definitions(ARTS_DB_DEFAULT=${ARTS_DEFAULT_DB_KIND})

# ── Mutually-exclusive sanitizer guard ─────────────────────────────────────
if(ARTS_USE_SANS AND ARTS_USE_TSAN)
  message(FATAL_ERROR
    "ARTS_USE_SANS (ASan/UBSan/LSan) and ARTS_USE_TSAN are "
    "compiler-incompatible — use separate build directories.")
endif()

# Compiler-specific settings
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Configuring Debug build")

    # Common debug flags
    add_compile_options(
        -O0 -g3 -Wall -Wextra
        -Wno-unused -Wno-unused-parameter -Wno-unused-variable
        -Wno-unused-function -Wno-unused-value -Wno-sign-compare
        -Wno-implicit-fallthrough
    )

    # Compiler-specific flags
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-Wno-stringop-truncation)
    endif()

    # Sanitizers (C/CXX only — CUDA excluded to avoid ASan shadow memory
    # conflicting with the CUDA driver's virtual address space mapping)
    if(ARTS_USE_SANS)
        add_compile_options(
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fsanitize=address>
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fsanitize=undefined>
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fsanitize=leak>
        )
        add_link_options(-fsanitize=address -fsanitize=undefined -fsanitize=leak)
        message(STATUS "Enabled sanitizers for Debug build (C/CXX only, excludes CUDA)")
    elseif(ARTS_USE_TSAN)
        add_compile_options(
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fsanitize=thread>
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fsanitize=undefined>
        )
        add_link_options(-fsanitize=thread -fsanitize=undefined)
        message(STATUS "Enabled ThreadSanitizer for Debug build (C/CXX only, excludes CUDA)")
    endif()
else()
    message(STATUS "Configuring Release build")
    add_compile_options(-march=native -mtune=native)
endif()

# CUDA support detection (only when GPU support is requested)
set(CUDA_AVAILABLE OFF)
if(ARTS_USE_GPU)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

        message(STATUS "CUDA compiler detected: ${CMAKE_CUDA_COMPILER}")

        # Try to find CUDA toolkit - required for full CUDA support
        # Fix for CUDA 12.0+ where nvToolsExt is no longer a library but header-only
        if(NOT TARGET CUDA::nvToolsExt)
            add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        endif()

        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA toolkit found: ${CUDAToolkit_VERSION}")
            set(CUDA_AVAILABLE ON)
        else()
            message(STATUS "CUDA compiler available but CUDAToolkit not found")
            message(STATUS "  CUDA support disabled - toolkit required for CUDA library")
            set(CUDA_AVAILABLE OFF)
        endif()

        # Set CUDA architectures - auto-detect if explicitly requested
        if(ARTS_USE_LOCAL_CUDA_ARCHITECTURES)
            # Try to auto-detect GPU compute capability using nvidia-smi
            find_program(NVIDIA_SMI nvidia-smi)
            if(NVIDIA_SMI)
                execute_process(
                    COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader,nounits
                    OUTPUT_VARIABLE NVIDIA_SMI_OUTPUT
                    ERROR_QUIET
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                )

                if(NVIDIA_SMI_OUTPUT)
                    # Parse compute capabilities for all GPUs (e.g., "8.6\n7.5" -> "86;75")
                    string(REPLACE "\n" ";" GPU_CAPS ${NVIDIA_SMI_OUTPUT})
                    string(REPLACE "." "" CUDA_ARCH_NUMS "${GPU_CAPS}")

                    # Remove duplicates and sort
                    list(REMOVE_DUPLICATES CUDA_ARCH_NUMS)
                    list(SORT CUDA_ARCH_NUMS)

                    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_NUMS})
                    message(STATUS "ARTS_USE_LOCAL_CUDA_ARCHITECTURES enabled - detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
                else()
                    message(WARNING "ARTS_USE_LOCAL_CUDA_ARCHITECTURES enabled but nvidia-smi detection failed - keeping default CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
                endif()
            else()
                message(WARNING "ARTS_USE_LOCAL_CUDA_ARCHITECTURES enabled but nvidia-smi not found - keeping default CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
            endif()
        else()
            message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()

        # CUDA compiler flags
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)
        endif()

        # Ensure NVCC uses the same host compiler and flags as the C++ compiler
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
        if(CMAKE_CXX_FLAGS)
            string(REPLACE " " "," CUDA_HOST_FLAGS "${CMAKE_CXX_FLAGS}")
            add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${CUDA_HOST_FLAGS}>)
        endif()
    else()
        message(STATUS "CUDA compiler not found - CUDA support unavailable")
    endif()
else()
    message(STATUS "GPU support disabled (ARTS_USE_GPU=OFF)")
endif()

# Determine which libraries to build
set(BUILD_CUDA_LIBRARY OFF)
if(ARTS_USE_GPU AND CUDA_AVAILABLE)
    # GPU support enabled and CUDA is available
    set(BUILD_CUDA_LIBRARY ON)
    message(STATUS "Will build both standard ARTS library and CUDA-enabled ARTS library")
else()
    message(STATUS "Will build only standard ARTS library (GPU support disabled)")
endif()

# Find required system libraries
find_package(Threads REQUIRED)
if(NOT APPLE)
    find_library(RT_LIBRARY rt REQUIRED)
    find_library(M_LIBRARY m REQUIRED)
    find_library(ANL_LIBRARY anl REQUIRED)
else()
    set(RT_LIBRARY "")
    set(M_LIBRARY "")
    set(ANL_LIBRARY "")
endif()

# ── CXL shared-memory support ─────────────────────────────────────────────
if(ARTS_USE_CXL)
    # Validate compiler support
    if(NOT (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang"))
        set(ARTS_USE_CXL OFF)
        message(STATUS "CXL: ${CMAKE_C_COMPILER_ID} not supported, disabling")
    endif()
endif()
if(ARTS_USE_CXL)
    # Validate required paths
    if(NOT ARTS_CXL_RAPID_INCLUDE_DIR)
        message(FATAL_ERROR
            "ARTS_USE_CXL is ON but ARTS_CXL_RAPID_INCLUDE_DIR is not set.\n"
            "  -DARTS_CXL_RAPID_INCLUDE_DIR=/path/to/rapid/include")
    endif()
    if(NOT ARTS_CXL_LIB_DIR)
        message(FATAL_ERROR
            "ARTS_USE_CXL is ON but ARTS_CXL_LIB_DIR is not set.\n"
            "  -DARTS_CXL_LIB_DIR=/path/to/arts_cxl_lib")
    endif()

    # Derive library and include paths
    set(ARTS_CXL_LIB_PATH "${ARTS_CXL_LIB_DIR}/build/src/libarts_cxl_lib.so")
    set(ARTS_CXL_INCLUDE_DIR "${ARTS_CXL_LIB_DIR}/inc")

    # Verify paths exist
    if(NOT EXISTS "${ARTS_CXL_RAPID_INCLUDE_DIR}")
        message(FATAL_ERROR "ARTS_CXL_RAPID_INCLUDE_DIR not found: ${ARTS_CXL_RAPID_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${ARTS_CXL_LIB_PATH}")
        message(FATAL_ERROR "CXL ARTS library not found: ${ARTS_CXL_LIB_PATH}")
    endif()
    if(NOT EXISTS "${ARTS_CXL_INCLUDE_DIR}")
        message(FATAL_ERROR "CXL ARTS include dir not found: ${ARTS_CXL_INCLUDE_DIR}")
    endif()

    # Try to find Rapid API
    list(APPEND CMAKE_PREFIX_PATH "${ARTS_CXL_RAPID_INCLUDE_DIR}/..")
    find_package(rapid CONFIG QUIET)
    if(rapid_FOUND)
        message(STATUS "CXL: Enabled - ARTS_USE_CXL + ARTS_CXL_NATIVE defined")
        add_compile_definitions(ARTS_USE_CXL ARTS_CXL_NATIVE)
        message(STATUS "  Rapid include:    ${ARTS_CXL_RAPID_INCLUDE_DIR}")
        message(STATUS "  ARTS CXL lib dir: ${ARTS_CXL_LIB_DIR}")
        message(STATUS "  ARTS CXL library: ${ARTS_CXL_LIB_PATH}")
        message(STATUS "  ARTS CXL include: ${ARTS_CXL_INCLUDE_DIR}")
    else()
        set(ARTS_USE_CXL OFF)
        message(STATUS "CXL: Rapid API not found at ${ARTS_CXL_RAPID_INCLUDE_DIR}, disabling")
    endif()
endif()
if(ARTS_USE_CXL)
    add_library(arts_cxl_lib SHARED IMPORTED)
    set_target_properties(arts_cxl_lib PROPERTIES
        IMPORTED_LOCATION "${ARTS_CXL_LIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${ARTS_CXL_INCLUDE_DIR};${ARTS_CXL_RAPID_INCLUDE_DIR}"
    )

    find_package(rapid CONFIG REQUIRED
        PATHS "${ARTS_CXL_RAPID_INCLUDE_DIR}/..")
        set_target_properties(arts_cxl_lib PROPERTIES
        IMPORTED_LOCATION "${ARTS_CXL_LIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${ARTS_CXL_INCLUDE_DIR};${ARTS_CXL_RAPID_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "rapid::rapid"
    )
endif()

# Add third-party dependencies
add_subdirectory(third_party)

# Hardware locality (hwloc) — always built from source, v2+ required
if(HWLOC_VERSION VERSION_GREATER_EQUAL "2.0")
    message(STATUS "Hardware topology: hwloc v${HWLOC_VERSION} (source build)")
else()
    message(FATAL_ERROR "hwloc v${HWLOC_VERSION} < 2.0 — v2+ required")
endif()

add_compile_definitions(ARTS_LOG_LEVEL=${ARTS_LOG_LEVEL})
message(STATUS "Log level: ${ARTS_LOG_LEVEL} (0=ERROR, 1=+WARN, 2=+INFO, 3=+DEBUG)")

set(COUNTER_CONFIG_FILE "${ARTS_COUNTER_CONFIG}")

# Create the ARTS library targets (core + extensions)
add_subdirectory(libs)

# Optional components
if(ARTS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(ARTS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ARTS_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

add_subdirectory(configs)

# Documentation (Doxygen XML → Breathe → Sphinx HTML)
if(ARTS_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    find_program(SPHINX_EXECUTABLE sphinx-build
        HINTS "$ENV{HOME}/.local/bin" "${CMAKE_SOURCE_DIR}/.venv/bin"
        DOC "Sphinx documentation generator"
    )
    if(NOT SPHINX_EXECUTABLE)
        message(FATAL_ERROR
            "sphinx-build not found. Install with:\n"
            "  pip install -r docs/requirements.txt")
    endif()

    # Paths for Doxyfile.in @VAR@ substitution
    set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen")
    set(DOXYGEN_INPUT_DIRS
        "${CMAKE_SOURCE_DIR}/libs/include/public/arts.h \
         ${CMAKE_SOURCE_DIR}/libs/include/public/arts/graph.h \
         ${CMAKE_SOURCE_DIR}/libs/include/internal/arts/runtime/rt.h \
         ${CMAKE_SOURCE_DIR}/libs/include/internal/arts/gas/guid.h \
         ${CMAKE_SOURCE_DIR}/libs/include/internal/arts/defs.h"
    )
    configure_file(
        "${CMAKE_SOURCE_DIR}/docs/Doxyfile.in"
        "${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile"
        @ONLY
    )

    # Target: ninja doxygen  (XML only)
    add_custom_target(doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Generating Doxygen XML..."
        VERBATIM
    )

    # Target: ninja docs  (Sphinx HTML, depends on doxygen)
    set(SPHINX_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx")
    add_custom_target(docs
        COMMAND ${CMAKE_COMMAND} -E env
        "BREATHE_DOXYGEN_XML_DIR=${DOXYGEN_OUTPUT_DIR}/xml"
        ${SPHINX_EXECUTABLE}
        -b html
        -c "${CMAKE_SOURCE_DIR}/docs"
        "${CMAKE_SOURCE_DIR}/docs"
        "${SPHINX_BUILD_DIR}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Building Sphinx HTML documentation..."
        DEPENDS doxygen
        VERBATIM
    )

    message(STATUS "Documentation: enabled (ninja doxygen | ninja docs)")
endif()

# Installation configuration
# Install public headers — all other headers are internal.
# External users: #include "arts.h", #include "arts/graph.h"
install(FILES libs/include/public/arts.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY libs/include/public/arts/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/arts
    FILES_MATCHING PATTERN "*.h"
)

# Install library targets
if(ARTS_BUILD_STATIC)
    install(TARGETS arts_static
        EXPORT ARTSTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    if(BUILD_CUDA_LIBRARY)
        install(TARGETS arts_cuda_static
            EXPORT ARTSTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()
endif()

if(ARTS_BUILD_SHARED)
    install(TARGETS arts_shared
        EXPORT ARTSTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    if(BUILD_CUDA_LIBRARY)
        install(TARGETS arts_cuda_shared
            EXPORT ARTSTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()
endif()

# Install the interface target
install(TARGETS arts_interface
    EXPORT ARTSTargets
)

# Generate and install CMake config files for find_package support
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ARTSConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ARTSConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ARTS
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ARTSConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ARTSConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ARTSConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ARTS
)

# Export targets (only export the main libraries, not object libraries)
install(EXPORT ARTSTargets
    FILE ARTSTargets.cmake
    NAMESPACE ARTS::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ARTS
)

# Print configuration summary
message(STATUS "")
message(STATUS "=================== ARTS Configuration Summary ===================")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "C Compiler:        ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "CXX Compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA Compiler:     ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "")
message(STATUS "Features:")
message(STATUS "  Standard Library: ON (always built)")
if(BUILD_CUDA_LIBRARY)
    message(STATUS "  CUDA Library:     ON (libarts_cuda)")
else()
    message(STATUS "  CUDA Library:     OFF")
endif()
if(ARTS_TRANSPORT_OFI)
    message(STATUS "  Network:          libfabric (OFI) / TCP bootstrap")
else()
    message(STATUS "  Network:          TCP")
endif()
message(STATUS "  Hardware Topo:    hwloc v${HWLOC_VERSION} (source)")
message(STATUS "")
message(STATUS "Build options:")
if(ARTS_BUILD_SHARED)
    message(STATUS "  Shared library:  libarts.so (standard)")
    if(BUILD_CUDA_LIBRARY)
        message(STATUS "                   libarts_cuda.so (CUDA-enabled)")
    endif()
else()
    message(STATUS "  Shared library:  OFF")
endif()
if(ARTS_BUILD_STATIC)
    message(STATUS "  Static library:  libarts.a (standard)")
    if(BUILD_CUDA_LIBRARY)
        message(STATUS "                   libarts_cuda.a (CUDA-enabled)")
    endif()
else()
    message(STATUS "  Static library:  OFF")
endif()
message(STATUS "  Examples:        ${ARTS_BUILD_EXAMPLES}")
message(STATUS "  Tests:           ${ARTS_BUILD_TESTS}")
message(STATUS "  Benchmarks:      ${ARTS_BUILD_BENCHMARKS}")
message(STATUS "")
message(STATUS "Installation paths:")
message(STATUS "  Libraries:       ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
message(STATUS "  Headers:         ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "  CMake config:    ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/ARTS")
message(STATUS "=================================================================")
message(STATUS "")
message(STATUS "To specify a custom install prefix, use:")
message(STATUS "  cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ..")
message(STATUS "=================================================================")

