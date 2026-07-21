# XSOCR (reference OCR x86 implementation) built as a CMake library
#
# All 147 source files are globbed; ocr-config.h #ifdef guards ensure
# only the x86-relevant modules compile to non-empty objects.

set(OCR_ROOT "${CMAKE_SOURCE_DIR}/third_party/ocr/ocr")

file(GLOB_RECURSE XSOCR_SRCS "${OCR_ROOT}/src/*.c")

# Build as OBJECT library first, then assemble into static
add_library(ocr_xsocr_objects OBJECT ${XSOCR_SRCS})
# Debug info in every build flavor: no codegen impact, and scheduler-level
# faults in the reference runtime are otherwise undebuggable.
target_compile_options(ocr_xsocr_objects PRIVATE -g)

# XSOCR uses GNU extensions (VLAs, typeof, __thread, etc.)
set_target_properties(ocr_xsocr_objects PROPERTIES
    C_STANDARD 11
    C_EXTENSIONS ON
)

target_compile_options(ocr_xsocr_objects PRIVATE
    ${BENCH_C_OPT}
    -Wall
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-unused-function
    -Wno-sign-compare
    -Wno-maybe-uninitialized
)

# ocr-config.h lives in this directory
target_include_directories(ocr_xsocr_objects PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}     # ocr_config_x86.h (renamed to ocr-config.h via wrapper)
    ${OCR_ROOT}/src
    ${OCR_ROOT}/src/inc
)

# Public OCR headers for consumers
target_include_directories(ocr_xsocr_objects PUBLIC
    ${OCR_ROOT}/inc
)

# The XSOCR source expects #include "ocr-config.h" but our file is
# named ocr_config_x86.h.  Map it via a thin wrapper.
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/ocr-config.h")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ocr-config.h"
         "#include \"${CMAKE_CURRENT_SOURCE_DIR}/ocr_config_x86.h\"\n")
endif()
target_include_directories(ocr_xsocr_objects PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}     # generated ocr-config.h wrapper
)

# Coherence configuration: the lockable datablock factory's per-DB hints
# OCR_HINT_DB_EAGER (push a read-only copy to a remote consumer at satisfy) and
# OCR_HINT_DB_LAZY (retain a DB at its current owner and migrate it on the next
# acquire) are gated by ALLOW_EAGER_DB / ENABLE_LAZY_DB.  Both are left
# undefined so every datablock uses the factory's default home-centric protocol
# (fetch-on-acquire, writeback-to-home-on-release) uniformly; the advisory
# hints, which only some apps set, are then inert.  This keeps the runtime's
# coherence behaviour single and uniform across all blocks.  The
# channel/collective extension chain below is unrelated and stays enabled.
# OCR_NOHINT_DB_HOME_ROUNDROBIN: no-hint DB home placement policy.
# 1 = round-robin the home across all locations (default); 0 = pin to the
# creator (legacy behavior).
# INIT_DEQUE_CAPACITY: the workstealing deque does not grow; a full deque's
# push would silently wrap and corrupt entries when asserts are compiled out.
# Size it for large single-rank EDT fanouts, and keep the overflow assert
# active in release builds (OCR_ASSERT_CRITICAL) so exhaustion aborts loudly
# instead of corrupting the scheduler.
target_compile_definitions(ocr_xsocr_objects PRIVATE
    "OCR_TYPE_H=x86.h"
    CACHE_LINE_SZB=64
    GUID_PROVIDER_LOCID_SIZE=10
    OCR_NOHINT_DB_HOME_ROUNDROBIN=1
    INIT_DEQUE_CAPACITY=2097152
    OCR_ASSERT_CRITICAL=1
)

# NOTE: the collective-event / distributed-labeled extension chain
# (ENABLE_EXTENSION_COLLECTIVE_EVT, ENABLE_EXTENSION_MULTI_OUTPUT_SLOT,
# ENABLE_EXTENSION_DISTRIBUTED_LABELED, REG_ASYNC_SGL) is enabled in the
# libocr build via ocr_config_x86.h (the config header included by every
# libocr translation unit through the generated ocr-config.h wrapper), not
# here on the command line.  Apps that exercise these primitives pass the
# matching app-side extension defines via their per-app EXTRA_DEFINES (e.g.
# hpcg_intel_Eager_Collective adds COLLECTIVE_EVT + MULTI_OUTPUT_SLOT) so
# the app and runtime headers agree.  Enabling the chain on the libocr build
# alone (config header) is enough for the runtime to handle the collective
# event cross-rank; it does NOT, however, implement remote metadata cloning
# for reserved-GUID (labeled) datablocks — that path is stubbed in the
# labeled-GUID provider ("Labeled Limitation").

# Static library
add_library(ocr_xsocr_static STATIC $<TARGET_OBJECTS:ocr_xsocr_objects>)
target_include_directories(ocr_xsocr_static PUBLIC ${OCR_ROOT}/inc)
target_link_libraries(ocr_xsocr_static PUBLIC pthread)

# MPI support for distributed (HC_DIST) policy domain
find_package(MPI COMPONENTS C)
if(MPI_C_FOUND)
    target_include_directories(ocr_xsocr_objects PRIVATE ${MPI_C_INCLUDE_DIRS})
    target_link_libraries(ocr_xsocr_static PUBLIC MPI::MPI_C)
    message(STATUS "XSOCR: MPI found — distributed (HC_DIST) comm-platform enabled")
else()
    message(WARNING "XSOCR: MPI not found — distributed comm-platform will not link")
endif()

# libnuma for distributed memory-platform (ENABLE_MEM_PLATFORM_NUMA_ALLOC)
find_library(NUMA_LIB numa)
if(NUMA_LIB)
    target_link_libraries(ocr_xsocr_static PUBLIC ${NUMA_LIB})
endif()

# Check for librt (needed on older glibc)
include(CheckLibraryExists)
check_library_exists(rt clock_gettime "" HAVE_LIBRT)
if(HAVE_LIBRT)
    target_link_libraries(ocr_xsocr_static PUBLIC rt)
endif()
