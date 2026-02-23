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

# Feature options (can be set via command line)
set(ARTS_LOG_LEVEL "1" CACHE STRING "Log verbosity: 0=ERROR, 1=+WARN (default), 2=+INFO, 3=+DEBUG")
option(ARTS_USE_GPU "Enable GPU support (default: ON, disable when OFF)" ON)
option(ARTS_USE_RDMA "Use RDMA RSockets networking (default: ON, use TCP when OFF)" OFF)
option(ARTS_USE_SANITIZERS "Enable address and undefined behavior sanitizers in Debug builds" ON)
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

    # Sanitizers
    if(ARTS_USE_SANITIZERS)
        add_compile_options(-fsanitize=address -fsanitize=undefined -fsanitize=leak)
        add_link_options(-fsanitize=address -fsanitize=undefined -fsanitize=leak)
        message(STATUS "Enabled sanitizers for Debug build")
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

# Network library detection and configuration
if(NOT ARTS_USE_RDMA)
    # User explicitly disabled RDMA, use TCP
    message(STATUS "Network: Using TCP (RDMA disabled)")
    set(RDMACM_LIBRARIES "")
    set(RDMACM_INCLUDE_DIRS "")
else()
    # Try to use RDMA
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(RDMACM QUIET librdmacm)
    endif()

    if(RDMACM_FOUND)
        add_compile_definitions(ARTS_USE_RDMA)
        message(STATUS "Network: Using RDMA RSockets - ARTS_USE_RDMA defined")
    else()
        message(STATUS "Network: RDMA requested but not available, falling back to TCP")
        set(RDMACM_LIBRARIES "")
        set(RDMACM_INCLUDE_DIRS "")
    endif()
endif()

# Add third-party dependencies
add_subdirectory(third_party)

# Hardware locality (hwloc) — mandatory dependency, v2+ required
if(HWLOC_FOUND)
    if(HWLOC_VERSION VERSION_GREATER_EQUAL "2.0")
        message(STATUS "Hardware topology: hwloc v${HWLOC_VERSION}")
    else()
        message(FATAL_ERROR "hwloc v${HWLOC_VERSION} < 2.0 — v2+ required")
    endif()
else()
    message(FATAL_ERROR "hwloc not found — required dependency")
endif()

add_compile_definitions(ARTS_LOG_LEVEL=${ARTS_LOG_LEVEL})
message(STATUS "Log level: ${ARTS_LOG_LEVEL} (0=ERROR, 1=+WARN, 2=+INFO, 3=+DEBUG)")

set(COUNTER_CONFIG_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_configs/counters.cfg"
    CACHE FILEPATH "ARTS counter configuration file (parsed to generate introspection macros)")
set(COUNTER_CONFIG_FILE "${COUNTER_CONFIG_PATH}")

# Create the ARTS library targets (core + extensions)
add_subdirectory(libs)

# Optional components
if(ARTS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(ARTS_BUILD_TESTS)
    add_subdirectory(tests)
endif()

add_subdirectory(sample_configs)

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
         ${CMAKE_SOURCE_DIR}/libs/include/public/arts/array_db.h \
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
# External users: #include "arts.h", #include "arts/graph.h", #include "arts/array_db.h"
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
if(ARTS_USE_RDMA AND RDMACM_FOUND)
    message(STATUS "  Network:          RDMA RSockets")
else()
    message(STATUS "  Network:          TCP (default)")
endif()
message(STATUS "  Hardware Topo:    hwloc v${HWLOC_VERSION}")
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
