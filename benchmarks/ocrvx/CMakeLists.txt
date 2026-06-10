# OCR-VX distributed MPI runtime (ocr_vdm_mpi)
# Only built when ARTS_BUILD_BENCHMARKS=ON.

# oneTBB: build from bundled submodule.
# Clear TBB_DIR to avoid accidentally picking up a system TBB.
# Disable hwloc (tbbbind) — it generates build-dir INTERFACE_INCLUDE_DIRECTORIES
# that violate CMake policy CMP0070, and we don't need NUMA-aware task pinning.
set(TBB_DIR "" CACHE PATH "" FORCE)
set(TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ON CACHE BOOL "" FORCE)
set(TBBMALLOC_BUILD ON CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/tbb
                 ${CMAKE_CURRENT_BINARY_DIR}/tbb_build
                 EXCLUDE_FROM_ALL)

find_package(MPI COMPONENTS CXX REQUIRED)

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

target_compile_options(ocr_vdm_mpi PRIVATE
    -w   # suppress warnings from third-party code
)

target_link_libraries(ocr_vdm_mpi PUBLIC
    TBB::tbb
    TBB::tbbmalloc
    MPI::MPI_CXX
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
