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
    coherence_basic
    coherence_rw_chain
    coherence_rw_ordering
    coherence_rw_ro_rw
    coherence_rw_rw_chain_seal
    coherence_ro_accumulation
    coherence_mixed_local_remote
    out_of_order_list
    coherence_ooo_concurrent_drain
    coherence_lock_req_before_create
    coherence_destroy_during_acquire
    coherence_stress_single_node
    coherence_stress_dist
    coherence_multi_writer_same_addr
    coherence_ro_acquire_stress
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
    multinode_coherence
    multinode_event_types
    multinode_array_db
    multinode_db_advanced
    # --- Graph tests ---
    csr
    distribution
    # --- Array DB tests ---
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

# Create test executables
foreach(test_program ${TEST_PROGRAMS})
    add_arts_test(${test_program})
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
        gpu_edt_basic
        gpu_edt_dep
        gpu_edt_with_guid
        gpu_edt_passthrough
        gpu_memory
        gpu_memset
        gpu_db
        gpu_lc_sync_basic
        gpu_multi_kernel
    )

    # Function to create a CUDA test executable
    function(add_cuda_test test_name)
        add_executable(${test_name} ${test_name}.cu)

        # Set include directories
        target_include_directories(${test_name} PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})

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

###############################################################################
# CTest registration
#
# ARTS runtime occupies the entire process (binds ports, spawns worker threads).
# RESOURCE_LOCK ensures only one test runs at a time, even with ctest -j.
###############################################################################

# --- Helper functions ---

function(register_single_node_test test_name)
    cmake_parse_arguments(ARG "" "TIMEOUT" "ARGS" ${ARGN})
    if(NOT ARG_TIMEOUT)
        set(ARG_TIMEOUT 10)
    endif()
    add_test(
        NAME ${test_name}
        COMMAND ${test_name} ${ARG_ARGS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(${test_name} PROPERTIES
        LABELS "single_node"
        TIMEOUT ${ARG_TIMEOUT}
        RESOURCE_LOCK "arts_runtime"
    )
endfunction()

function(register_multinode_test test_name)
    cmake_parse_arguments(ARG "" "TIMEOUT" "ARGS" ${ARGN})
    if(NOT ARG_TIMEOUT)
        set(ARG_TIMEOUT 30)
    endif()
    add_test(
        NAME ${test_name}
        COMMAND ${test_name} ${ARG_ARGS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(${test_name} PROPERTIES
        LABELS "multinode"
        TIMEOUT ${ARG_TIMEOUT}
        RESOURCE_LOCK "arts_runtime"
        ENVIRONMENT "ARTS_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/arts_multinode.cfg"
    )
endfunction()

function(register_gpu_test test_name)
    cmake_parse_arguments(ARG "" "TIMEOUT" "ARGS" ${ARGN})
    if(NOT ARG_TIMEOUT)
        set(ARG_TIMEOUT 10)
    endif()
    add_test(
        NAME ${test_name}
        COMMAND ${test_name} ${ARG_ARGS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(${test_name} PROPERTIES
        LABELS "gpu"
        TIMEOUT ${ARG_TIMEOUT}
        RESOURCE_LOCK "arts_runtime"
        ENVIRONMENT "ARTS_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/arts_gpu.cfg;ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0:alloc_dealloc_mismatch=0"
    )
endfunction()

# --- Single-node tests ---

# EDT tests
register_single_node_test(edt_create_basic)
register_single_node_test(edt_chain)
register_single_node_test(edt_fan_out)
register_single_node_test(edt_signal)
register_single_node_test(edt_dep_variants)
register_single_node_test(edt_destroy)
register_single_node_test(edt_epoch_wait)
register_single_node_test(stress_edt TIMEOUT 30)
register_single_node_test(paramv_memcpy)

# DB tests
register_single_node_test(db_create)
register_single_node_test(db_create_with_data)
register_single_node_test(db_dependence)
register_single_node_test(db_destroy)
register_single_node_test(db_local)
register_single_node_test(db_local_create)
register_single_node_test(db_put_get)
register_single_node_test(db_rename)
register_single_node_test(pin_db ARGS 0)
register_single_node_test(get_from_db ARGS 16 1)
register_single_node_test(put_in_db ARGS 16 1)

# Array DB tests
register_single_node_test(array_db)
register_single_node_test(array_db_advanced)
register_single_node_test(array_db_for_each)
register_single_node_test(array_db_with_guid)
register_single_node_test(local_array_db)
register_single_node_test(for_each)
register_single_node_test(put_array_db_epoch ARGS 16)
register_single_node_test(put_array_db_epoch_direct ARGS 16)
register_single_node_test(gather_array_db_epoch)
register_single_node_test(termination_detection_array_db ARGS 16)

# Epoch / sync tests
register_single_node_test(epoch_basic)
register_single_node_test(epoch_deferred_start)
register_single_node_test(epoch_finish_edt)
register_single_node_test(epoch_pool ARGS 3)
register_single_node_test(epoch_shad_adapter ARGS 3)
register_single_node_test(rec_epoch ARGS 3)
register_single_node_test(termination_detection ARGS 3)

# Event tests
register_single_node_test(event_basic)
register_single_node_test(event_chain)
register_single_node_test(persistent_event)
register_single_node_test(persistent_event_advanced)

# GUID tests
register_single_node_test(guid_basic)
register_single_node_test(guid_range)

# Routing / misc
register_single_node_test(hint_routing)
register_single_node_test(node_query)
register_single_node_test(record_dep_at)
register_single_node_test(utility_api)
register_single_node_test(route_table)
register_single_node_test(route_table_iter)
register_single_node_test(route_table_iter_destroy)
register_single_node_test(acquire_mode)
register_single_node_test(arts_id)
register_single_node_test(out_of_order_list)
register_single_node_test(coherence_ooo_concurrent_drain TIMEOUT 60)
set_property(TEST coherence_ooo_concurrent_drain APPEND PROPERTY
    LABELS "coherence_stress")
register_single_node_test(buffer)
register_single_node_test(buffer_basic)
register_single_node_test(shad_lock)

# Coherence tests (formerly CDAG — names track the v3 RC protocol naming)
register_single_node_test(coherence_basic ARGS 32 0 0 0)
register_single_node_test(coherence_rw_chain ARGS 4)
register_single_node_test(coherence_rw_ordering)
register_single_node_test(coherence_rw_ro_rw)
register_single_node_test(coherence_rw_rw_chain_seal)
register_single_node_test(coherence_ro_accumulation)

# Graph tests
register_single_node_test(csr)
register_single_node_test(distribution)

# --- Multi-node tests ---

register_multinode_test(multinode_db)
register_multinode_test(multinode_edt)
register_multinode_test(multinode_epoch)
register_multinode_test(multinode_event)
register_multinode_test(active_message_db ARGS 64)
register_multinode_test(active_message_buffer ARGS 64)
register_multinode_test(arts_send ARGS 16)
register_multinode_test(route_table_remote_guid)
register_multinode_test(db_put_get_at)
register_multinode_test(db_remote)
register_multinode_test(remote_db_event)
register_multinode_test(multinode_coherence)
register_multinode_test(multinode_event_types)
register_multinode_test(multinode_array_db)
register_multinode_test(multinode_db_advanced)
register_multinode_test(coherence_mixed_local_remote)

# 3-rank coherence stress: B.3 LOCK_REQ-before-DB_CREATE race regression.
# Uses arts_4node.cfg (the test SKIPs cleanly if rank_count < 3).  The 4-node
# config is copied into the tests build dir by sample_configs/CMakeLists.txt.
add_test(
    NAME coherence_lock_req_before_create
    COMMAND coherence_lock_req_before_create
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(coherence_lock_req_before_create PROPERTIES
    LABELS "coherence_stress;multinode"
    TIMEOUT 120
    RESOURCE_LOCK "arts_runtime"
    ENVIRONMENT "ARTS_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/arts_4node.cfg"
)

# Coherence stress test suite (Spec section 6 B.1/B.4/B.5/B.6 single-node).
# Each adds the "coherence_stress" label on top of "single_node" so the suite
# can be filtered with `ctest -L coherence_stress`.
register_single_node_test(coherence_destroy_during_acquire TIMEOUT 60)
set_property(TEST coherence_destroy_during_acquire APPEND PROPERTY
    LABELS "coherence_stress")
register_single_node_test(coherence_stress_single_node TIMEOUT 120)
set_property(TEST coherence_stress_single_node APPEND PROPERTY
    LABELS "coherence_stress")
register_single_node_test(coherence_multi_writer_same_addr TIMEOUT 60)
set_property(TEST coherence_multi_writer_same_addr APPEND PROPERTY
    LABELS "coherence_stress")
register_single_node_test(coherence_ro_acquire_stress TIMEOUT 60)
set_property(TEST coherence_ro_acquire_stress APPEND PROPERTY
    LABELS "coherence_stress")

# Distributed coherence stress: B.2.  Uses arts_4node.cfg (the test SKIPs
# cleanly if rank_count < 2).  Same scaffolding as B.3 above.
add_test(
    NAME coherence_stress_dist
    COMMAND coherence_stress_dist
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(coherence_stress_dist PROPERTIES
    LABELS "coherence_stress;multinode"
    TIMEOUT 120
    RESOURCE_LOCK "arts_runtime"
    ENVIRONMENT "ARTS_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/arts_4node.cfg"
)

# --- GPU tests ---

if(BUILD_CUDA_LIBRARY)
    register_gpu_test(gpu_edt_basic)
    register_gpu_test(gpu_edt_dep)
    register_gpu_test(gpu_edt_with_guid)
    register_gpu_test(gpu_edt_passthrough)
    register_gpu_test(gpu_memory)
    register_gpu_test(gpu_memset)
    register_gpu_test(gpu_db)
    register_gpu_test(gpu_lc_sync_basic)
    register_gpu_test(lc_sync)
    register_gpu_test(gpu_multi_kernel)
    register_gpu_test(gpu_for_all)
    register_gpu_test(gpu_lib)
endif()
