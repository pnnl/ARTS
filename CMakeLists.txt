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

# x86-64: enable 128-bit compare-and-swap (cmpxchg16b)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-mcx16>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mcx16>)
endif()

# DWCAS feature detection (cmpxchg16b on x86_64, casp on ARMv8.1+).
# Required by Task 4m / 4m-2 (lockfree_pool, tiered_pool).  GCC and Clang
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
        "ARTS requires DWCAS (cmpxchg16b/casp).  "
        "Compile with -mcx16 (x86_64) or ARMv8.1+.")
endif()
# Make libatomic available to all targets that may emit 16-byte atomic
# libcalls (lockfree_pool.h / tiered_pool.h users).
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

# Build options
option(ARTS_BUILD_SHARED "Build shared library" ON)
option(ARTS_BUILD_STATIC "Build static library" ON)
option(ARTS_BUILD_EXAMPLES "Build example programs" ON)
option(ARTS_BUILD_TESTS "Build test programs" ON)
option(ARTS_BUILD_BENCHMARKS "Build OCR benchmark applications (XSOCR + ARTS backends)" OFF)

# Feature options (can be set via command line)
# Default log level: Debug->3 (DEBUG), Release/etc->1 (WARN).
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
option(ARTS_USE_GPU "Enable GPU support (default: ON, disable when OFF)" ON)
option(ARTS_USE_CXL "Enable CXL shared-memory support (default: OFF)" OFF)

# Host memory consistency model for ARTS_DB.  LRC (default) uses Lazy
# Release Consistency (no synchronous home write-back, owner-direct data
# flow) — it avoids the per-release write-back/ACK round-trip that makes
# the eager RC protocol scale poorly when one rank produces many remotely
# homed datablocks.  RC is the eager Release Consistency protocol; LC is
# the weak Location Consistency model.  Compile-time switch — one binary is
# exactly one mode.  Both ranks in a multinode run must use the same build
# (mode-mismatch is caught at runtime via ARTS_FATAL stub handlers for
# opposite-mode wire messages).
set(ARTS_MEMORY_MODEL "LRC" CACHE STRING
    "Host memory consistency model for ARTS_DB: LRC (default), RC, or LC")
set_property(CACHE ARTS_MEMORY_MODEL PROPERTY STRINGS RC LRC LC)

if(NOT ARTS_MEMORY_MODEL STREQUAL "RC" AND
   NOT ARTS_MEMORY_MODEL STREQUAL "LRC" AND
   NOT ARTS_MEMORY_MODEL STREQUAL "LC")
  message(FATAL_ERROR
          "ARTS_MEMORY_MODEL must be RC, LRC, or LC, got: ${ARTS_MEMORY_MODEL}")
endif()

if(ARTS_MEMORY_MODEL STREQUAL "LRC")
  add_compile_definitions(ARTS_MEMORY_MODEL_LRC=1)
  message(STATUS "ARTS memory model: LRC (Lazy Release Consistency, default)")
elseif(ARTS_MEMORY_MODEL STREQUAL "LC")
  add_compile_definitions(ARTS_MEMORY_MODEL_LC=1)
  message(STATUS "ARTS memory model: LC (Location Consistency)")
else()
  # Define RC explicitly (not just the absence of LRC/LC) so the spec's
  # `#if defined(ARTS_MEMORY_MODEL_RC) || defined(ARTS_MEMORY_MODEL_LRC)`
  # field guards select correctly.  Legacy `#else` RC branches still work.
  add_compile_definitions(ARTS_MEMORY_MODEL_RC=1)
  message(STATUS "ARTS memory model: RC (Release Consistency)")
endif()

# Default DB storage subtype that benchmarks select via ARTS_DB_DEFAULT.
# Flip this one variable to retarget every benchmark in one shot
# (e.g. -DARTS_DB_DEFAULT=ARTS_DB_CXL) — no source edits needed.
set(ARTS_DB_DEFAULT "ARTS_DB" CACHE STRING
    "DB subtype that ARTS_DB_DEFAULT expands to (ARTS_DB / ARTS_DB_PIN / ARTS_DB_GPU_PIN / ARTS_DB_GPU / ARTS_DB_CXL)")
set_property(CACHE ARTS_DB_DEFAULT PROPERTY STRINGS
    ARTS_DB ARTS_DB_PIN ARTS_DB_GPU_PIN ARTS_DB_GPU ARTS_DB_CXL)
add_compile_definitions(ARTS_DB_DEFAULT=${ARTS_DB_DEFAULT})

# Default EDT kind (mirrors ARTS_DB_DEFAULT pattern).  Flip this one
# variable to retarget the EDT default in one shot.
set(ARTS_EDT_DEFAULT_KIND "ARTS_EDT_CPU" CACHE STRING
    "EDT kind that ARTS_EDT_DEFAULT expands to (ARTS_EDT_CPU / ARTS_EDT_GPU)")
set_property(CACHE ARTS_EDT_DEFAULT_KIND PROPERTY STRINGS
    ARTS_EDT_CPU ARTS_EDT_GPU)
add_compile_definitions(ARTS_EDT_DEFAULT=${ARTS_EDT_DEFAULT_KIND})

set(ARTS_CXL_RAPID_INCLUDE_DIR "" CACHE PATH
    "Path to Rapid API include directory (required when ARTS_USE_CXL=ON)")
set(ARTS_CXL_LIB_DIR "" CACHE PATH
    "Path to arts_cxl_lib directory (required when ARTS_USE_CXL=ON)")
option(ARTS_USE_SANITIZERS "Enable address and undefined behavior sanitizers in Debug builds" OFF)
option(ARTS_USE_LOCAL_CUDA_ARCHITECTURES "Auto-detect CUDA architectures from local GPUs using nvidia-smi" ON)

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
    if(ARTS_USE_SANITIZERS)
        add_compile_options(
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fsanitize=address>
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fsanitize=undefined>
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fsanitize=leak>
        )
        add_link_options(-fsanitize=address -fsanitize=undefined -fsanitize=leak)
        message(STATUS "Enabled sanitizers for Debug build (C/CXX only, excludes CUDA)")
    endif()
else()
    message(STATUS "Configuring Release build")
    add_compile_options(-march=native -mtune=native)
endif()

# Linker configuration
option(ARTS_USE_LINKER "Specify linker to use for ARTS builds" "")
if(ARTS_USE_LINKER)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --ld-path=${ARTS_USE_LINKER} -w")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} --ld-path=${ARTS_USE_LINKER} -w")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --ld-path=${ARTS_USE_LINKER} -w")
    message(STATUS "Using custom linker: ${ARTS_USE_LINKER}")
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

set(COUNTER_CONFIG_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/counters.cfg"
    CACHE FILEPATH "ARTS counter configuration file (parsed to generate introspection macros)")
set(COUNTER_CONFIG_FILE "${COUNTER_CONFIG_PATH}")

# Create the ARTS library targets (core + extensions)
add_subdirectory(libs)

# Optional components
# NOTE: examples/ is temporarily relocated out of the tree during refactoring.
# Re-enable this block once the directory is restored.
# if(ARTS_BUILD_EXAMPLES)
#     add_subdirectory(examples)
# endif()

if(ARTS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ARTS_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# LRC microbenchmarks are always built (independent of ARTS_BUILD_BENCHMARKS).
add_subdirectory(benchmarks/lrc_microbench)

add_subdirectory(configs)

# Documentation (Doxygen XML → Breathe → Sphinx HTML)
option(ARTS_BUILD_DOCS "Build documentation (requires Doxygen + Sphinx)" OFF)
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
message(STATUS "  Network:          TCP")
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

