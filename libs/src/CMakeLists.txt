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

if(ARTS_USE_CXL)
    target_link_libraries(arts_interface INTERFACE arts_cxl_lib)
endif()

# --- Coherence-protocol helpers (shared by main lib + benchmark variants) -----
# Sets ${out} (caller scope) to the coherence model TU paths for (proto,timing),
# RELATIVE to libs/src/core/. Mirrors libs/src/core/CMakeLists.txt selection.
function(arts_coherence_model_tus out proto timing)
    if(proto STREQUAL "MRMW")
        set(tus coherence/mrmw/mrmw.c)
    elseif(proto STREQUAL "LOCK")
        if(timing STREQUAL "LAZY")
            set(tus coherence/lock/home.c coherence/lock/lazy.c)
        else()
            set(tus coherence/lock/home.c coherence/lock/eager.c)
        endif()
    elseif(proto STREQUAL "MRSW")
        if(timing STREQUAL "EAGER")
            set(tus coherence/mrsw/eager.c coherence/mrsw/ownership.c coherence/mrsw/home.c coherence/mrsw/waiter_queue.c)
        else()
            set(tus coherence/mrsw/lazy.c coherence/mrsw/ownership.c coherence/mrsw/home.c coherence/mrsw/waiter_queue.c)
        endif()
    else() # MRNEW
        if(timing STREQUAL "EAGER")
            set(tus coherence/mrnew/eager.c coherence/mrnew/ownership.c coherence/mrnew/home.c)
        else()
            set(tus coherence/mrnew/lazy.c coherence/mrnew/ownership.c coherence/mrnew/home.c)
        endif()
    endif()
    set(${out} ${tus} PARENT_SCOPE)
endfunction()

# Applies the protocol/timing compile definitions to <target> (PRIVATE).
function(arts_apply_protocol target proto timing)
    if(proto STREQUAL "MRMW")
        target_compile_definitions(${target} PRIVATE ARTS_PROTOCOL_MRMW=1)
    elseif(proto STREQUAL "LOCK")
        target_compile_definitions(${target} PRIVATE ARTS_PROTOCOL_LOCK=1)
        if(timing STREQUAL "EAGER")
            target_compile_definitions(${target} PRIVATE ARTS_TIMING_EAGER=1)
        else()
            target_compile_definitions(${target} PRIVATE ARTS_TIMING_LAZY=1)
        endif()
    elseif(proto STREQUAL "MRSW")
        target_compile_definitions(${target} PRIVATE ARTS_PROTOCOL_MRSW=1)
        if(timing STREQUAL "EAGER")
            target_compile_definitions(${target} PRIVATE ARTS_TIMING_EAGER=1)
        else()
            target_compile_definitions(${target} PRIVATE ARTS_TIMING_LAZY=1)
        endif()
    else() # MRNEW
        target_compile_definitions(${target} PRIVATE ARTS_PROTOCOL_MRNEW=1)
        if(timing STREQUAL "EAGER")
            target_compile_definitions(${target} PRIVATE ARTS_TIMING_EAGER=1)
        else()
            target_compile_definitions(${target} PRIVATE ARTS_TIMING_LAZY=1)
        endif()
    endif()
endfunction()

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
        $<TARGET_OBJECTS:arts_ooo>
        $<TARGET_OBJECTS:arts_system>
        $<TARGET_OBJECTS:arts_utils>
        $<TARGET_OBJECTS:arts_graph>
    )

    target_link_libraries(arts_static PUBLIC arts_interface)

    # Link external dependencies statically
    target_link_libraries(arts_static PRIVATE
        Threads::Threads
        ${RT_LIBRARY}
        ${M_LIBRARY}
        ${ANL_LIBRARY}
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

    arts_apply_protocol(arts_static ${ARTS_COHERENCE_PROTOCOL} ${ARTS_PROTOCOL_TIMING})

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
        $<TARGET_OBJECTS:arts_ooo>
        $<TARGET_OBJECTS:arts_system>
        $<TARGET_OBJECTS:arts_utils>
        $<TARGET_OBJECTS:arts_graph>
    )

    target_link_libraries(arts_shared PUBLIC arts_interface)

    # Link external dependencies statically into the shared library
    target_link_libraries(arts_shared PRIVATE
        Threads::Threads
        ${RT_LIBRARY}
        ${M_LIBRARY}
        ${ANL_LIBRARY}
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

    arts_apply_protocol(arts_shared ${ARTS_COHERENCE_PROTOCOL} ${ARTS_PROTOCOL_TIMING})

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
            $<TARGET_OBJECTS:arts_ooo>
            $<TARGET_OBJECTS:arts_system>
            $<TARGET_OBJECTS:arts_utils>
            $<TARGET_OBJECTS:arts_gpu>
            $<TARGET_OBJECTS:arts_graph>
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

        arts_apply_protocol(arts_cuda_static ${ARTS_COHERENCE_PROTOCOL} ${ARTS_PROTOCOL_TIMING})

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
            $<TARGET_OBJECTS:arts_ooo>
            $<TARGET_OBJECTS:arts_system>
            $<TARGET_OBJECTS:arts_utils>
            $<TARGET_OBJECTS:arts_gpu>
            $<TARGET_OBJECTS:arts_graph>
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

        arts_apply_protocol(arts_cuda_shared ${ARTS_COHERENCE_PROTOCOL} ${ARTS_PROTOCOL_TIMING})

        # Set library properties
        set_target_properties(arts_cuda_shared PROPERTIES
            OUTPUT_NAME arts_cuda
            VERSION ${PROJECT_VERSION}
            SOVERSION ${PROJECT_VERSION_MAJOR}
            EXPORT_NAME cuda_shared
        )
    endif()
endif()

# --- Per-protocol variant libraries (benchmarks/harness only) -----------------
# Build a full arts static lib for one (proto,timing). The protocol-independent
# sources are gathered from the CPU object-lib targets (their compiled objects
# cannot be reused — each variant needs its own protocol defines), the coherence
# MODEL TUs are swapped for this variant's, and the defines are applied per-target.
if(ARTS_BUILD_BENCHMARKS AND ARTS_BUILD_STATIC)
    set(_ARTS_VARIANT_OBJLIBS
        arts_compute arts_memory arts_sync arts_ooo arts_scheduler
        arts_transport arts_gas arts_system arts_utils arts_counter arts_graph)

    function(arts_variant suffix proto timing)
        # Gather all protocol-independent sources (absolute paths) from the
        # object libs, dropping any coherence MODEL TU (path contains
        # /coherence/<model>/), which is variant-specific.
        set(srcs "${CMAKE_CURRENT_SOURCE_DIR}/core/main.c")
        foreach(lib ${_ARTS_VARIANT_OBJLIBS})
            get_target_property(lsrcs ${lib} SOURCES)
            get_target_property(ldir  ${lib} SOURCE_DIR)
            foreach(s ${lsrcs})
                if(IS_ABSOLUTE "${s}")
                    set(abs "${s}")
                else()
                    set(abs "${ldir}/${s}")
                endif()
                # skip coherence model TUs (mrnew/mrsw/lock/mrmw subdirs)
                if(NOT abs MATCHES "/coherence/(mrnew|mrsw|lock|mrmw)/")
                    list(APPEND srcs "${abs}")
                endif()
            endforeach()
        endforeach()
        # add this variant's coherence model TUs
        arts_coherence_model_tus(model "${proto}" "${timing}")
        foreach(tu ${model})
            list(APPEND srcs "${CMAKE_CURRENT_SOURCE_DIR}/core/${tu}")
        endforeach()

        add_library(arts_static_${suffix} STATIC ${srcs})
        target_include_directories(arts_static_${suffix} BEFORE PRIVATE
            ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
            ${ARTS_PUBLIC_INCLUDE_DIR}
            ${ARTS_INTERNAL_INCLUDE_DIR})
        if(HWLOC_INCLUDE_DIRS)
            target_include_directories(arts_static_${suffix} PRIVATE ${HWLOC_INCLUDE_DIRS})
        endif()
        target_link_libraries(arts_static_${suffix} PUBLIC arts_interface)
        target_link_libraries(arts_static_${suffix} PRIVATE
            Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY}
            $<BUILD_INTERFACE:${HWLOC_LIBRARIES}> ${HWLOC_PRIVATE_LIBS})
        if(ARTS_ATOMIC_LIBRARY)
            target_link_libraries(arts_static_${suffix} PRIVATE ${ARTS_ATOMIC_LIBRARY})
        endif()
        if(ARTS_MALLOC STREQUAL "mimalloc")
            target_link_libraries(arts_static_${suffix} PRIVATE mimalloc-static)
        endif()
        arts_apply_protocol(arts_static_${suffix} "${proto}" "${timing}")
    endfunction()

    arts_variant(mrnew_eager MRNEW EAGER)
    arts_variant(mrnew_lazy  MRNEW LAZY)
    arts_variant(mrsw_eager  MRSW  EAGER)
    arts_variant(mrsw_lazy   MRSW  LAZY)
    arts_variant(mrmw        MRMW  LAZY)
    arts_variant(lock_eager  LOCK  EAGER)
    arts_variant(lock_lazy   LOCK  LAZY)
endif()

# Link mimalloc-static when selected.  The OBJECT library arts_utils needs
# the include dir at compile time (for malloc.c's #include <mimalloc.h>);
# target_link_libraries on an OBJECT target propagates INTERFACE properties
# (include dirs) to its compilation unit without introducing a link dep.
if(ARTS_MALLOC STREQUAL "mimalloc")
    target_link_libraries(arts_utils PRIVATE mimalloc-static)
    if(ARTS_BUILD_STATIC)
        target_link_libraries(arts_static PRIVATE mimalloc-static)
    endif()
    if(ARTS_BUILD_SHARED)
        target_link_libraries(arts_shared PRIVATE mimalloc-static)
    endif()
    if(BUILD_CUDA_LIBRARY)
        if(ARTS_BUILD_STATIC)
            target_link_libraries(arts_cuda_static PRIVATE mimalloc-static)
        endif()
        if(ARTS_BUILD_SHARED)
            target_link_libraries(arts_cuda_shared PRIVATE mimalloc-static)
        endif()
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
