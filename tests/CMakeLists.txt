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

# Define test programs to build
set(TEST_PROGRAMS
    # --- Original tests ---
    edt_epoch_wait
    route_table
    route_table_remote_guid
    cdag
    cdag_ew_chain
    cdag_ew_ordering
    out_of_order_list
    pin_db
    get_from_db
    put_in_db
    arts_send
    active_message_db
    active_message_buffer
    termination_detection
    rec_epoch
    epoch_pool
    epoch_basic
    epoch_deferred_start
    epoch_finish_edt
    epoch_shad_adapter
    buffer
    buffer_basic
    remote_db_event
    shad_lock
    route_table_iter
    route_table_iter_destroy
    acquire_mode
    arts_id
    # --- New unit tests ---
    edt_create_basic
    edt_chain
    edt_fan_out
    edt_signal
    edt_dep_variants
    edt_destroy
    stress_edt
    paramv_memcpy
    db_create
    db_create_with_data
    db_dependence
    db_destroy
    db_local
    db_local_create
    db_put_get
    db_put_get_at
    db_remote
    db_rename
    event_basic
    event_chain
    persistent_event
    persistent_event_advanced
    guid_basic
    guid_range
    hint_routing
    node_query
    record_dep_at
    utility_api
    multinode_db
    multinode_edt
    multinode_epoch
    multinode_event
)

# Programs that require arts_graph library
set(GRAPH_TEST_PROGRAMS
    csr
    distribution
)

# Function to create a test executable
function(add_arts_test test_name)
    add_executable(${test_name} ${test_name}.c)

    # Set include directories
    target_include_directories(${test_name} PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})

    # Match CARTS launcher expectations: non-PIE binaries avoid pointer-auth
    # failures when passing raw function pointers across nodes.
    set_property(TARGET ${test_name} PROPERTY POSITION_INDEPENDENT_CODE OFF)
    target_compile_options(${test_name} PRIVATE -fno-pie -fno-PIE)
    target_link_options(${test_name} PRIVATE -no-pie -fno-pie -fno-PIE)

    # Link with runtime libraries
    target_link_libraries(${test_name} PRIVATE Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})

    # Use the appropriate ARTS library based on availability
    if(TARGET arts_shared)
        target_link_libraries(${test_name} PRIVATE arts_shared)
    elseif(TARGET arts_static)
        target_link_libraries(${test_name} PRIVATE arts_static)
    else()
        message(FATAL_ERROR "No ARTS library target found")
    endif()
endfunction()

# Create standard test executables
foreach(test_program ${TEST_PROGRAMS})
    add_arts_test(${test_program})
endforeach()

# Function to create a test executable that uses arts_graph
function(add_arts_graph_test test_name)
    add_executable(${test_name} ${test_name}.c)

    # Set include directories
    target_include_directories(${test_name} PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR} ${ARTS_GRAPH_INCLUDE_DIR})

    set_property(TARGET ${test_name} PROPERTY POSITION_INDEPENDENT_CODE OFF)
    target_compile_options(${test_name} PRIVATE -fno-pie -fno-PIE)
    target_link_options(${test_name} PRIVATE -no-pie -fno-pie -fno-PIE)

    # Link with arts_graph and runtime libraries
    target_link_libraries(${test_name} PRIVATE arts_graph Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})

    # Use the appropriate ARTS library based on availability
    if(TARGET arts_shared)
        target_link_libraries(${test_name} PRIVATE arts_shared)
    elseif(TARGET arts_static)
        target_link_libraries(${test_name} PRIVATE arts_static)
    else()
        message(FATAL_ERROR "No ARTS library target found")
    endif()
endfunction()

# Create graph test executables
foreach(test_program ${GRAPH_TEST_PROGRAMS})
    add_arts_graph_test(${test_program})
endforeach()

# Programs that require arts_array_db library
set(ARRAY_DB_TEST_PROGRAMS
    array_db
    array_db_advanced
    array_db_for_each
    array_db_with_guid
    local_array_db
    for_each
    termination_detection_array_db
    put_array_db_epoch
    put_array_db_epoch_direct
    gather_array_db_epoch
)

# Function to create a test executable that uses arts_array_db
function(add_arts_array_db_test test_name)
    add_executable(${test_name} ${test_name}.c)

    # Set include directories
    target_include_directories(${test_name} PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR} ${ARTS_ARRAY_DB_INCLUDE_DIR})

    set_property(TARGET ${test_name} PROPERTY POSITION_INDEPENDENT_CODE OFF)
    target_compile_options(${test_name} PRIVATE -fno-pie -fno-PIE)
    target_link_options(${test_name} PRIVATE -no-pie -fno-pie -fno-PIE)

    # Link with arts_array_db and runtime libraries
    target_link_libraries(${test_name} PRIVATE arts_array_db Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})

    # Use the appropriate ARTS library based on availability
    if(TARGET arts_shared)
        target_link_libraries(${test_name} PRIVATE arts_shared)
    elseif(TARGET arts_static)
        target_link_libraries(${test_name} PRIVATE arts_static)
    else()
        message(FATAL_ERROR "No ARTS library target found")
    endif()
endfunction()

# Create array_db test executables
foreach(test_program ${ARRAY_DB_TEST_PROGRAMS})
    add_arts_array_db_test(${test_program})
endforeach()

# CUDA test programs (only built if CUDA is available)
if(BUILD_CUDA_LIBRARY)
    set(CUDA_TEST_PROGRAMS
        gpu_stream
        gpu_for_all
        lc_sync
        gpu_lc_red
        gpu_lib
        gpu_ring
    )

    # Function to create a CUDA test executable
    function(add_cuda_test test_name)
        add_executable(${test_name} ${test_name}.cu)

        # Set include directories
        target_include_directories(${test_name} PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

        set_property(TARGET ${test_name} PROPERTY POSITION_INDEPENDENT_CODE OFF)
        target_compile_options(${test_name} PRIVATE -fno-pie -fno-PIE)
        target_link_options(${test_name} PRIVATE -no-pie -fno-pie -fno-PIE)

        # Link with CUDA libraries
        target_link_libraries(${test_name} PRIVATE
            Threads::Threads
            ${RT_LIBRARY}
            ${M_LIBRARY}
            ${ANL_LIBRARY}
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::cublas
        )

        # Use the CUDA-enabled ARTS library
        if(TARGET arts_cuda_shared)
            target_link_libraries(${test_name} PRIVATE arts_cuda_shared)
        elseif(TARGET arts_cuda_static)
            target_link_libraries(${test_name} PRIVATE arts_cuda_static)
        else()
            message(FATAL_ERROR "No CUDA ARTS library target found")
        endif()
    endfunction()

    # Create CUDA test executables
    foreach(cuda_test ${CUDA_TEST_PROGRAMS})
        add_cuda_test(${cuda_test})
    endforeach()

    message(STATUS "CUDA test programs will be built")
else()
    message(STATUS "CUDA test programs will not be built (CUDA not available)")
endif()
