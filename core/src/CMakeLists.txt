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
    $<BUILD_INTERFACE:${ARTS_CORE_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${ARTS_CORE_BUILD_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_subdirectory(system)
add_subdirectory(utils)
add_subdirectory(network)
add_subdirectory(runtime)
add_subdirectory(gas)
add_subdirectory(introspection)

# GPU module is only built for CUDA library
if(BUILD_CUDA_LIBRARY)
    add_subdirectory(gpu)
endif()

# Main source files
file(GLOB ARTS_MAIN_SOURCES "*.c")

#
# Standard ARTS Library (always built, no CUDA)
#

# Create static library if requested
if(ARTS_BUILD_STATIC)
    add_library(arts_static STATIC ${ARTS_MAIN_SOURCES})

    # Add hwloc include directories if available
    if(HWLOC_INCLUDE_DIRS)
        target_include_directories(arts_static PRIVATE ${HWLOC_INCLUDE_DIRS})
    endif()

    # Add object library contents directly to avoid export issues
    target_sources(arts_static PRIVATE
        $<TARGET_OBJECTS:arts_gas>
        $<TARGET_OBJECTS:arts_network>
        $<TARGET_OBJECTS:arts_introspection>
        $<TARGET_OBJECTS:arts_runtime>
        $<TARGET_OBJECTS:arts_runtime_compute>
        $<TARGET_OBJECTS:arts_runtime_memory>
        $<TARGET_OBJECTS:arts_runtime_sync>
        $<TARGET_OBJECTS:arts_runtime_network>
        $<TARGET_OBJECTS:arts_system>
        $<TARGET_OBJECTS:arts_utils>
    )

    target_link_libraries(arts_static PUBLIC arts_interface)

    # Link external dependencies statically
    target_link_libraries(arts_static PRIVATE
        Threads::Threads
        ${RT_LIBRARY}
        ${M_LIBRARY}
        ${ANL_LIBRARY}
        ${RDMACM_LIBRARIES}
        ${HWLOC_LIBRARIES}
    )

    # Add macOS framework dependencies for hwloc
    if(APPLE AND HWLOC_FOUND)
        target_link_libraries(arts_static PRIVATE "-framework CoreFoundation" "-framework IOKit" "-framework OpenCL")
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

    # Add hwloc include directories if available
    if(HWLOC_INCLUDE_DIRS)
        target_include_directories(arts_shared PRIVATE ${HWLOC_INCLUDE_DIRS})
    endif()

    # Add object library contents directly to avoid export issues
    target_sources(arts_shared PRIVATE
        $<TARGET_OBJECTS:arts_gas>
        $<TARGET_OBJECTS:arts_network>
        $<TARGET_OBJECTS:arts_introspection>
        $<TARGET_OBJECTS:arts_runtime>
        $<TARGET_OBJECTS:arts_runtime_compute>
        $<TARGET_OBJECTS:arts_runtime_memory>
        $<TARGET_OBJECTS:arts_runtime_sync>
        $<TARGET_OBJECTS:arts_runtime_network>
        $<TARGET_OBJECTS:arts_system>
        $<TARGET_OBJECTS:arts_utils>
    )

    target_link_libraries(arts_shared PUBLIC arts_interface)

    # Link external dependencies statically into the shared library
    target_link_libraries(arts_shared PRIVATE
        Threads::Threads
        ${RT_LIBRARY}
        ${M_LIBRARY}
        ${ANL_LIBRARY}
        ${RDMACM_LIBRARIES}
        ${HWLOC_LIBRARIES}
    )

    # Add macOS framework dependencies for hwloc
    if(APPLE AND HWLOC_FOUND)
        target_link_libraries(arts_shared PRIVATE "-framework CoreFoundation" "-framework IOKit" "-framework OpenCL")
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

        # Add hwloc include directories if available
        if(HWLOC_INCLUDE_DIRS)
            target_include_directories(arts_cuda_static PRIVATE ${HWLOC_INCLUDE_DIRS})
        endif()

        # Add object library contents directly to avoid export issues
        target_sources(arts_cuda_static PRIVATE
            $<TARGET_OBJECTS:arts_gas>
            $<TARGET_OBJECTS:arts_network>
            $<TARGET_OBJECTS:arts_introspection>
            $<TARGET_OBJECTS:arts_runtime_gpu>
            $<TARGET_OBJECTS:arts_runtime_compute_gpu>
            $<TARGET_OBJECTS:arts_runtime_memory_gpu>
            $<TARGET_OBJECTS:arts_runtime_sync>
            $<TARGET_OBJECTS:arts_runtime_network>
            $<TARGET_OBJECTS:arts_system>
            $<TARGET_OBJECTS:arts_utils>
            $<TARGET_OBJECTS:arts_gpu>
        )

        target_link_libraries(arts_cuda_static PUBLIC arts_interface)

        # Add CUDA compile definition for this target
        target_compile_definitions(arts_cuda_static PRIVATE USE_GPU)

        # Link external dependencies and CUDA libraries
        target_link_libraries(arts_cuda_static PRIVATE
            Threads::Threads
            ${RT_LIBRARY}
            ${M_LIBRARY}
            ${ANL_LIBRARY}
            ${RDMACM_LIBRARIES}
            ${HWLOC_LIBRARIES}
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::cublas
        )

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

        # Add hwloc include directories if available
        if(HWLOC_INCLUDE_DIRS)
            target_include_directories(arts_cuda_shared PRIVATE ${HWLOC_INCLUDE_DIRS})
        endif()

        # Add object library contents directly to avoid export issues
        target_sources(arts_cuda_shared PRIVATE
            $<TARGET_OBJECTS:arts_gas>
            $<TARGET_OBJECTS:arts_network>
            $<TARGET_OBJECTS:arts_introspection>
            $<TARGET_OBJECTS:arts_runtime_gpu>
            $<TARGET_OBJECTS:arts_runtime_compute_gpu>
            $<TARGET_OBJECTS:arts_runtime_memory_gpu>
            $<TARGET_OBJECTS:arts_runtime_sync>
            $<TARGET_OBJECTS:arts_runtime_network>
            $<TARGET_OBJECTS:arts_system>
            $<TARGET_OBJECTS:arts_utils>
            $<TARGET_OBJECTS:arts_gpu>
        )

        target_link_libraries(arts_cuda_shared PUBLIC arts_interface)

        # Add CUDA compile definition for this target
        target_compile_definitions(arts_cuda_shared PRIVATE USE_GPU)

        # Link external dependencies and CUDA libraries
        target_link_libraries(arts_cuda_shared PRIVATE
            Threads::Threads
            ${RT_LIBRARY}
            ${M_LIBRARY}
            ${ANL_LIBRARY}
            ${RDMACM_LIBRARIES}
            ${HWLOC_LIBRARIES}
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::cublas
        )

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
