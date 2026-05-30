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

# Create the main ARTS library interface target
add_library(arts_interface INTERFACE)
target_include_directories(arts_interface INTERFACE
    $<BUILD_INTERFACE:${ARTS_PUBLIC_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${ARTS_INTERNAL_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${ARTS_BUILD_INTERNAL_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# enable RC coherence-message OoO defer.  Activates the
# OOO_DB_* dispatcher arms in out_of_order.c and lets coherence_handlers.c
# use the new home_lookup_or_defer path (instead of the old NOTIFY-only
# helper).  Build-integration of coherence_*.c is gated separately via
# libs/src/core/memory/CMakeLists.txt.
target_compile_definitions(arts_interface INTERFACE ARTS_COHERENCE_INTEGRATED)

if(ARTS_USE_CXL)
    target_link_libraries(arts_interface INTERFACE arts_cxl_lib)
endif()

# Module subdirectories — each defines OBJECT library targets
add_subdirectory(core)
add_subdirectory(graph)

# Main source file (main.c lives in core/; scheduler.c is an OBJECT library)
set(ARTS_MAIN_SOURCES "core/main.c")

# Some platforms do not provide a standalone libatomic.
# ARTS uses compiler atomic builtins, so link libatomic only when present.
# Reuse the top-level ATOMIC_LIBRARY (probed for DWCAS support); fall
# back to a local find_library if it wasn't set (e.g., older configurations).
if(ATOMIC_LIBRARY)
    set(ARTS_ATOMIC_LIBRARY "${ATOMIC_LIBRARY}" CACHE FILEPATH
        "libatomic path (shared with top-level DWCAS probe)" FORCE)
else()
    find_library(ARTS_ATOMIC_LIBRARY atomic)
endif()
if(ARTS_ATOMIC_LIBRARY)
    message(STATUS "Using libatomic: ${ARTS_ATOMIC_LIBRARY}")
else()
    message(STATUS "libatomic not found; relying on compiler/runtime atomics")
endif()

#
# Standard ARTS Library (always built, no CUDA)
#

# Create static library if requested
if(ARTS_BUILD_STATIC)
    add_library(arts_static STATIC ${ARTS_MAIN_SOURCES})

    target_include_directories(arts_static BEFORE PRIVATE
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
        ${ARTS_PUBLIC_INCLUDE_DIR}
        ${ARTS_INTERNAL_INCLUDE_DIR}
    )

    # Add hwloc include directories if available
    if(HWLOC_INCLUDE_DIRS)
        target_include_directories(arts_static PRIVATE ${HWLOC_INCLUDE_DIRS})
    endif()

    # Add object library contents directly to avoid export issues
    target_sources(arts_static PRIVATE
        $<TARGET_OBJECTS:arts_gas>
        $<TARGET_OBJECTS:arts_transport>
        $<TARGET_OBJECTS:arts_counter>
        $<TARGET_OBJECTS:arts_scheduler>
        $<TARGET_OBJECTS:arts_compute>
        $<TARGET_OBJECTS:arts_memory>
        $<TARGET_OBJECTS:arts_sync>
        $<TARGET_OBJECTS:arts_remote>
        $<TARGET_OBJECTS:arts_system>
        $<TARGET_OBJECTS:arts_utils>
        $<TARGET_OBJECTS:arts_graph_datastructures>
        $<TARGET_OBJECTS:arts_graph_distribution>
    )

    target_link_libraries(arts_static PUBLIC arts_interface)

    # Link external dependencies statically
    target_link_libraries(arts_static PRIVATE
        Threads::Threads
        ${RT_LIBRARY}
        ${M_LIBRARY}
        ${ANL_LIBRARY}
        ${RDMACM_LIBRARIES}
        $<BUILD_INTERFACE:${HWLOC_LIBRARIES}>
        ${HWLOC_PRIVATE_LIBS}
    )

    if(ARTS_ATOMIC_LIBRARY)
        target_link_libraries(arts_static PRIVATE ${ARTS_ATOMIC_LIBRARY})
    endif()

    # Add macOS framework dependencies for hwloc
    if(APPLE AND HWLOC_FOUND)
        target_link_libraries(arts_static PRIVATE "-framework CoreFoundation" "-framework IOKit" "-framework OpenCL")
    endif()

    if(ARTS_USE_CXL)
        target_link_libraries(arts_static PUBLIC arts_cxl_lib)
    endif()

    # Set library properties
    set_target_properties(arts_static PROPERTIES
        OUTPUT_NAME arts
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        EXPORT_NAME static
    )
endif()

# Create shared library if requested
if(ARTS_BUILD_SHARED)
    add_library(arts_shared SHARED ${ARTS_MAIN_SOURCES})

    target_include_directories(arts_shared BEFORE PRIVATE
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
        ${ARTS_PUBLIC_INCLUDE_DIR}
        ${ARTS_INTERNAL_INCLUDE_DIR}
    )

    # Add hwloc include directories if available
    if(HWLOC_INCLUDE_DIRS)
        target_include_directories(arts_shared PRIVATE ${HWLOC_INCLUDE_DIRS})
    endif()

    # Add object library contents directly to avoid export issues
    target_sources(arts_shared PRIVATE
        $<TARGET_OBJECTS:arts_gas>
        $<TARGET_OBJECTS:arts_transport>
        $<TARGET_OBJECTS:arts_counter>
        $<TARGET_OBJECTS:arts_scheduler>
        $<TARGET_OBJECTS:arts_compute>
        $<TARGET_OBJECTS:arts_memory>
        $<TARGET_OBJECTS:arts_sync>
        $<TARGET_OBJECTS:arts_remote>
        $<TARGET_OBJECTS:arts_system>
        $<TARGET_OBJECTS:arts_utils>
        $<TARGET_OBJECTS:arts_graph_datastructures>
        $<TARGET_OBJECTS:arts_graph_distribution>
    )

    target_link_libraries(arts_shared PUBLIC arts_interface)

    # Link external dependencies statically into the shared library
    target_link_libraries(arts_shared PRIVATE
        Threads::Threads
        ${RT_LIBRARY}
        ${M_LIBRARY}
        ${ANL_LIBRARY}
        ${RDMACM_LIBRARIES}
        $<BUILD_INTERFACE:${HWLOC_LIBRARIES}>
        ${HWLOC_PRIVATE_LIBS}
    )

    # Hide hwloc symbols from shared library exports (GNU ld/lld)
    if(HWLOC_FOUND AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(arts_shared PRIVATE
            "LINKER:--exclude-libs,libhwloc.a"
        )
    endif()

    if(ARTS_ATOMIC_LIBRARY)
        target_link_libraries(arts_shared PRIVATE ${ARTS_ATOMIC_LIBRARY})
    endif()

    # Add macOS framework dependencies for hwloc
    if(APPLE AND HWLOC_FOUND)
        target_link_libraries(arts_shared PRIVATE "-framework CoreFoundation" "-framework IOKit" "-framework OpenCL")
    endif()

    if(ARTS_USE_CXL)
        target_link_libraries(arts_shared PUBLIC arts_cxl_lib)
    endif()

    # Set library properties
    set_target_properties(arts_shared PROPERTIES
        OUTPUT_NAME arts
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        EXPORT_NAME shared
    )
endif()

#
# CUDA-enabled ARTS Library (built when CUDA available)
#

if(BUILD_CUDA_LIBRARY)
    # Create static CUDA library if requested
    if(ARTS_BUILD_STATIC)
        add_library(arts_cuda_static STATIC ${ARTS_MAIN_SOURCES})

        target_include_directories(arts_cuda_static BEFORE PRIVATE
            ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
            ${ARTS_PUBLIC_INCLUDE_DIR}
            ${ARTS_INTERNAL_INCLUDE_DIR}
        )

        # Add hwloc include directories if available
        if(HWLOC_INCLUDE_DIRS)
            target_include_directories(arts_cuda_static PRIVATE ${HWLOC_INCLUDE_DIRS})
        endif()

        # Add object library contents directly to avoid export issues
        target_sources(arts_cuda_static PRIVATE
            $<TARGET_OBJECTS:arts_gas>
            $<TARGET_OBJECTS:arts_transport>
            $<TARGET_OBJECTS:arts_counter>
            $<TARGET_OBJECTS:arts_scheduler_gpu>
            $<TARGET_OBJECTS:arts_compute_gpu>
            $<TARGET_OBJECTS:arts_memory_gpu>
            $<TARGET_OBJECTS:arts_sync>
            $<TARGET_OBJECTS:arts_remote>
            $<TARGET_OBJECTS:arts_system>
            $<TARGET_OBJECTS:arts_utils>
            $<TARGET_OBJECTS:arts_gpu>
            $<TARGET_OBJECTS:arts_graph_datastructures>
            $<TARGET_OBJECTS:arts_graph_distribution>
        )

        target_link_libraries(arts_cuda_static PUBLIC arts_interface)

        # Add CUDA compile definition for this target
        target_compile_definitions(arts_cuda_static PRIVATE ARTS_USE_GPU)

        # Link external dependencies and CUDA libraries
        target_link_libraries(arts_cuda_static PRIVATE
            Threads::Threads
            ${RT_LIBRARY}
            ${M_LIBRARY}
            ${ANL_LIBRARY}
            ${RDMACM_LIBRARIES}
            $<BUILD_INTERFACE:${HWLOC_LIBRARIES}>
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::cublas
        )

        if(ARTS_ATOMIC_LIBRARY)
            target_link_libraries(arts_cuda_static PRIVATE ${ARTS_ATOMIC_LIBRARY})
        endif()

        # Add macOS framework dependencies for hwloc
        if(APPLE AND HWLOC_FOUND)
            target_link_libraries(arts_cuda_static PRIVATE "-framework CoreFoundation" "-framework IOKit" "-framework OpenCL")
        endif()

        # Set library properties
        set_target_properties(arts_cuda_static PROPERTIES
            OUTPUT_NAME arts_cuda
            VERSION ${PROJECT_VERSION}
            SOVERSION ${PROJECT_VERSION_MAJOR}
            EXPORT_NAME cuda_static
        )
    endif()

    # Create shared CUDA library if requested
    if(ARTS_BUILD_SHARED)
        add_library(arts_cuda_shared SHARED ${ARTS_MAIN_SOURCES})

        target_include_directories(arts_cuda_shared BEFORE PRIVATE
            ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
            ${ARTS_PUBLIC_INCLUDE_DIR}
            ${ARTS_INTERNAL_INCLUDE_DIR}
        )

        # Add hwloc include directories if available
        if(HWLOC_INCLUDE_DIRS)
            target_include_directories(arts_cuda_shared PRIVATE ${HWLOC_INCLUDE_DIRS})
        endif()

        # Add object library contents directly to avoid export issues
        target_sources(arts_cuda_shared PRIVATE
            $<TARGET_OBJECTS:arts_gas>
            $<TARGET_OBJECTS:arts_transport>
            $<TARGET_OBJECTS:arts_counter>
            $<TARGET_OBJECTS:arts_scheduler_gpu>
            $<TARGET_OBJECTS:arts_compute_gpu>
            $<TARGET_OBJECTS:arts_memory_gpu>
            $<TARGET_OBJECTS:arts_sync>
            $<TARGET_OBJECTS:arts_remote>
            $<TARGET_OBJECTS:arts_system>
            $<TARGET_OBJECTS:arts_utils>
            $<TARGET_OBJECTS:arts_gpu>
            $<TARGET_OBJECTS:arts_graph_datastructures>
            $<TARGET_OBJECTS:arts_graph_distribution>
        )

        target_link_libraries(arts_cuda_shared PUBLIC arts_interface)

        # Add CUDA compile definition for this target
        target_compile_definitions(arts_cuda_shared PRIVATE ARTS_USE_GPU)

        # Link external dependencies and CUDA libraries
        target_link_libraries(arts_cuda_shared PRIVATE
            Threads::Threads
            ${RT_LIBRARY}
            ${M_LIBRARY}
            ${ANL_LIBRARY}
            ${RDMACM_LIBRARIES}
            $<BUILD_INTERFACE:${HWLOC_LIBRARIES}>
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::cublas
        )

        # Hide hwloc symbols from shared library exports (GNU ld/lld)
        if(HWLOC_FOUND AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
            target_link_options(arts_cuda_shared PRIVATE
                "LINKER:--exclude-libs,libhwloc.a"
            )
        endif()

        if(ARTS_ATOMIC_LIBRARY)
            target_link_libraries(arts_cuda_shared PRIVATE ${ARTS_ATOMIC_LIBRARY})
        endif()

        # Add macOS framework dependencies for hwloc
        if(APPLE AND HWLOC_FOUND)
            target_link_libraries(arts_cuda_shared PRIVATE "-framework CoreFoundation" "-framework IOKit" "-framework OpenCL")
        endif()

        # Set library properties
        set_target_properties(arts_cuda_shared PROPERTIES
            OUTPUT_NAME arts_cuda
            VERSION ${PROJECT_VERSION}
            SOVERSION ${PROJECT_VERSION_MAJOR}
            EXPORT_NAME cuda_shared
        )
    endif()
endif()

# Create alias targets for convenience (prefer shared over static)
if(ARTS_BUILD_SHARED)
    add_library(ARTS::arts ALIAS arts_shared)
    add_library(ARTS::shared ALIAS arts_shared)
    if(BUILD_CUDA_LIBRARY)
        add_library(ARTS::arts_cuda ALIAS arts_cuda_shared)
        add_library(ARTS::cuda_shared ALIAS arts_cuda_shared)
    endif()
endif()

if(ARTS_BUILD_STATIC)
    if(NOT TARGET ARTS::arts)
        add_library(ARTS::arts ALIAS arts_static)
    endif()
    add_library(ARTS::static ALIAS arts_static)
    if(BUILD_CUDA_LIBRARY)
        if(NOT TARGET ARTS::arts_cuda)
            add_library(ARTS::arts_cuda ALIAS arts_cuda_static)
        endif()
        add_library(ARTS::cuda_static ALIAS arts_cuda_static)
    endif()
endif()

# Ensure at least one library type is built
if(NOT ARTS_BUILD_STATIC AND NOT ARTS_BUILD_SHARED)
    message(FATAL_ERROR "At least one of ARTS_BUILD_STATIC or ARTS_BUILD_SHARED must be enabled")
endif()
