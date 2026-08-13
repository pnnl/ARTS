# OCR-VX distributed MPI runtime (ocr_vdm_mpi)
# Only built when ARTS_BUILD_BENCHMARKS=ON.
#
# MPI is host infrastructure, like the compiler: never vendored, never
# required for ARTS itself.  ocr-vx is a *reference* runtime that happens to
# be built on MPI, so probe for it before touching anything else in this
# file (including the vendored-TBB setup below) and skip the whole reference
# cleanly when the host has none.
find_package(MPI COMPONENTS CXX)
if(NOT MPI_CXX_FOUND)
    message(STATUS "Skipping the ocr-vx reference runtime (ocr_vdm_mpi) and every *_ocrvx app variant: no MPI found on this host. Provide an MPI with mpicc/mpicxx (e.g. on CMAKE_PREFIX_PATH) to enable them.")
    return()
endif()

# Past the probe, so a host that skips this reference never fetches the two
# submodules only it consumes: this runtime's own sources, and the TBB its
# task layer is built on.
arts_init_submodule(third_party/ocr-vx build_one/CMakeLists.txt)
arts_init_submodule(third_party/tbb CMakeLists.txt)

# oneTBB: build from the bundled submodule, as static archives so no consumer
# carries a libtbb/libtbbmalloc soname that a host copy on the loader path
# could answer for.  BUILD_SHARED_LIBS is set in a function scope: the value
# reaches the subdirectory and nothing else, leaving the enclosing project's
# own shared/static choice alone.
# Clear TBB_DIR to avoid accidentally picking up a system TBB.
# Disable hwloc (tbbbind) — its detection module pkg-configs the host, and it
# generates build-dir INTERFACE_INCLUDE_DIRECTORIES that violate CMake policy
# CMP0070; we don't need NUMA-aware task pinning.
function(arts_add_vendored_tbb)
    set(BUILD_SHARED_LIBS OFF)
    set(TBB_DIR "" CACHE PATH "" FORCE)
    set(TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ON CACHE BOOL "" FORCE)
    set(TBBMALLOC_BUILD ON CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/tbb
                     ${CMAKE_CURRENT_BINARY_DIR}/tbb_build
                     EXCLUDE_FROM_ALL)
endfunction()
arts_add_vendored_tbb()

set(OCRVX_ROOT "${CMAKE_SOURCE_DIR}/third_party/ocr-vx")

set(OCRVX_SRCS
    ${OCRVX_ROOT}/src/distributed/ocr_distributed.cpp
    ${OCRVX_ROOT}/src/distributed/ocr_distributed_main.cpp
    ${OCRVX_ROOT}/src/distributed/command_processing.cpp
    ${OCRVX_ROOT}/src/distributed/communication.cpp
    ${OCRVX_ROOT}/src/distributed/ocr_api_impl.cpp
    ${OCRVX_ROOT}/src/ocr_log.cpp
    ${OCRVX_ROOT}/src/threadqueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ocr-app-stubs.cpp
)

add_library(ocr_vdm_mpi STATIC ${OCRVX_SRCS})

target_compile_definitions(ocr_vdm_mpi PUBLIC
    OCR_USE_MPI
    OCR_ENABLE_EDT_NAMING
    "OCR_TYPE_H=vdm.h"
)

target_include_directories(ocr_vdm_mpi PUBLIC
    ${OCRVX_ROOT}/include/ocr
    ${OCRVX_ROOT}/include/common
    ${OCRVX_ROOT}/include/distributed
)

target_link_libraries(ocr_vdm_mpi PUBLIC
    TBB::tbb
    TBB::tbbmalloc
    MPI::MPI_CXX
    # This runtime reaches a header that includes <numa.h> through the shared
    # OCR source tree; the vendored libnuma supplies it on every host.
    arts::numa
    pthread
)

set_target_properties(ocr_vdm_mpi PROPERTIES
    CXX_STANDARD 14
    CXX_EXTENSIONS ON
)

# Export paths for benchmarks/apps
set(OCRVX_ROOT "${OCRVX_ROOT}" PARENT_SCOPE)
set(OCRVX_OCR_INC "${OCRVX_ROOT}/include/ocr" PARENT_SCOPE)
set(OCRVX_COMPAT_H "${CMAKE_CURRENT_SOURCE_DIR}/ocr-app-compat.h" PARENT_SCOPE)
