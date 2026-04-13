# XSOCR (reference OCR x86 implementation) built as a CMake library
#
# All 147 source files are globbed; ocr-config.h #ifdef guards ensure
# only the x86-relevant modules compile to non-empty objects.

set(OCR_ROOT "${CMAKE_SOURCE_DIR}/third_party/ocr/ocr")

file(GLOB_RECURSE XSOCR_SRCS "${OCR_ROOT}/src/*.c")

# Build as OBJECT library first, then assemble into static
add_library(ocr_xsocr_objects OBJECT ${XSOCR_SRCS})

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

target_compile_definitions(ocr_xsocr_objects PRIVATE
    "OCR_TYPE_H=x86.h"
    CACHE_LINE_SZB=64
    GUID_PROVIDER_LOCID_SIZE=10
    ALLOW_EAGER_DB
    ENABLE_LAZY_DB
    OCR_MAX_MULTI_SLOT=32
)

# NOTE: we do NOT enable ENABLE_EXTENSION_COLLECTIVE_EVT or
# ENABLE_EXTENSION_MULTI_OUTPUT_SLOT here.  Both are implemented in
# xsocr (see third_party/ocr/ocr/src/event/hc/hc-event.c and
# ocr-edt.c), but turning them on pulls in a chain of HC policy-domain
# features (MD_ALLOC, REG_ASYNC_SGL, ...) that require several more
# defines to compile.  The only app that needs this chain is
# hpcg_intel_Eager_Collective, which runs cleanly on arts; we handle
# it as arts_only in the correctness harness rather than expanding
# the xsocr build surface here.

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
