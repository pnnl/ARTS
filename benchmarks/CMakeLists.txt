# Benchmarks: OCR apps running on XSOCR (reference) and ARTS (via shim)
#
# Directory structure:
#   xsocr/     - Native XSOCR library (reference OCR x86 implementation)
#   ocr_shim/  - ARTS-backed OCR compatibility shim
#   apps/      - OCR benchmark apps built against both backends
#   hpx/       - Standalone HPX reference apps

# Benchmark optimization flags — controlled by CMAKE_BUILD_TYPE
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(BENCH_C_OPT -O0 -g)
else()
    set(BENCH_C_OPT -O3)
endif()

# Suppress the "#pragma message: OCR_TYPE_H not defined" note from ocr.h.
# Defining OCR_TYPE_H causes ocr.h to #include "ocr-options_<value>.h",
# so we generate a no-op header in the build directory.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ocr-options_arts.h"
     "/* No-op OCR options for ARTS benchmark builds. */\n")
set(BENCH_OCR_TYPE_DEFINE "OCR_TYPE_H=arts.h")
set(BENCH_OCR_OPTIONS_DIR "${CMAKE_CURRENT_BINARY_DIR}")

if(ARTS_BUILD_XSOCR)
    add_subdirectory(xsocr)
endif()
if(ARTS_BUILD_OCRVX)
    add_subdirectory(ocrvx)
endif()
if(ARTS_BUILD_HPX AND ARTS_HPX_AVAILABLE)
    include(ExternalProject)

    set(ARTS_HPX_APPS_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/hpx-build")
    set(ARTS_HPX_APPS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/hpx")
    ExternalProject_Add(hpx_apps_ep
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hpx"
        BINARY_DIR "${ARTS_HPX_APPS_BUILD_DIR}"
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        CMAKE_GENERATOR "${CMAKE_GENERATOR}"
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            # Same MPI wrapper the HPX build compiled with: the static HPX
            # package expects MPI from the toolchain, not from a resolved
            # library list.
            -DCMAKE_CXX_COMPILER=${MPI_CXX_COMPILER}
            -DCMAKE_PREFIX_PATH=${ARTS_HPX_INSTALL_DIR}
            # The consumer package re-finds hwloc, and a find through
            # NAMES/paths can land on a host copy; the exact archive the
            # HPX build embedded is the only correct answer.
            -DHwloc_INCLUDE_DIR=${HPX_HWLOC_INSTALL_DIR}/include
            -DHwloc_LIBRARY=${HPX_HWLOC_INSTALL_DIR}/lib/libhwloc.a
            -DHPX_APPS_RUNTIME_OUTPUT_DIRECTORY=${ARTS_HPX_APPS_OUTPUT_DIR}
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
            --parallel ${CMAKE_BUILD_PARALLEL_LEVEL}
        INSTALL_COMMAND ""
        # The solver core is plain C++ shared by every target here; running
        # its unit check as the test step keeps a broken core from ever
        # reaching a benchmark binary unnoticed.
        TEST_COMMAND "${ARTS_HPX_APPS_BUILD_DIR}/nqueens_core_test"
        BUILD_ALWAYS TRUE
        BUILD_BYPRODUCTS
            "${ARTS_HPX_APPS_OUTPUT_DIR}/nqueens_hpx"
        DEPENDS hpx_ep
        LOG_CONFIGURE 1
        LOG_BUILD 1
    )
    add_custom_target(nqueens_hpx DEPENDS hpx_apps_ep)
endif()
add_subdirectory(ocr_shim)
add_subdirectory(apps)
if(ARTS_BUILD_BASELINES)
    add_subdirectory(baseline)
endif()
