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
    lockfree_stack_stress
    lockfree_pool_stress
    tiered_pool_stress
    shared_t_stress
    route_table_install_race
    event_once_storm
    event_sticky_late_bind
    event_channel_fifo
    event_destroy_race
    pin_db
    get_from_db
    put_in_db
    arts_send
    active_message_db
    termination_detection
    rec_epoch
    epoch_pool
    epoch_basic
    epoch_deferred_start
    epoch_finish_edt
    remote_db_event
    route_table_iter
    route_table_iter_destroy
    acquire_mode
    arts_id
    # --- New unit tests ---
    edt_create_basic
    edt_chain
    edt_fan_out
    edt_signal
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
    event_create_destroy_basic
    event_satisfy_once
    event_add_dep_idem_fastpath
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
    # --- Coherence (coh_* prefix = newer minimal API tests) ---
    coh_basic
    coh_rw_chain
    coh_rw_ordering
    coh_rw_ro_rw
    coh_rw_rw_chain_seal
    coh_ro_accumulation
    coh_mixed_local_remote
    # --- DB type tests ---
    db_pin
    # --- Termination tests ---
    term_shutdown
    term_abort
    term_sigfpe
    term_sigsegv
    # --- Multinode (built here, registered below) ---
    multinode_stdout_forward
    multinode_coh
    # --- Pure unit tests (no ARTS runtime) ---
    lockfree_stack_test
    lockfree_marked_list_test
    coherence_buffer_test
    # --- New unit tests (Tasks 6-12) ---
    counter_smoke
    edt_dep_variants
    event_hint_presets
    db_labeled_guid
    edt_epoch_api
    multinode_labeled_guid
    multinode_event_hint
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
        set(ARG_TIMEOUT 60)
    endif()
    foreach(variant IN ITEMS 2n 4n 5n)
        add_test(
            NAME ${test_name}_${variant}
            COMMAND ${test_name} ${ARG_ARGS}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        set_tests_properties(${test_name}_${variant} PROPERTIES
            LABELS "multinode"
            TIMEOUT ${ARG_TIMEOUT}
            RESOURCE_LOCK "arts_runtime"
            ENVIRONMENT "ARTS_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/arts_${variant}ode.cfg"
        )
    endforeach()
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

# Pure unit tests: no ARTS runtime started, use separate resource lock.
function(register_pure_unit_test test_name)
    cmake_parse_arguments(ARG "" "TIMEOUT" "" ${ARGN})
    if(NOT ARG_TIMEOUT)
        set(ARG_TIMEOUT 30)
    endif()
    add_test(
        NAME ${test_name}
        COMMAND ${test_name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(${test_name} PROPERTIES
        LABELS "single_node"
        TIMEOUT ${ARG_TIMEOUT}
        RESOURCE_LOCK "pure_unit_test"
    )
endfunction()

# --- Single-node tests ---

# EDT tests
register_single_node_test(edt_create_basic)
register_single_node_test(edt_chain)
register_single_node_test(edt_fan_out)
register_single_node_test(edt_signal)
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
register_single_node_test(rec_epoch ARGS 3)
register_single_node_test(termination_detection ARGS 3)

# Event tests
register_single_node_test(event_basic)
register_single_node_test(event_chain)
register_single_node_test(event_create_destroy_basic)
register_single_node_test(event_satisfy_once)
register_single_node_test(event_add_dep_idem_fastpath)
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

# D1 lifecycle stress
register_single_node_test(shared_t_stress TIMEOUT 60)

# Strict lock-free reserve_or_lookup stress
register_single_node_test(route_table_install_race TIMEOUT 120)

# Event redesign stress tests
register_single_node_test(event_once_storm TIMEOUT 60)
register_single_node_test(event_sticky_late_bind TIMEOUT 60)
register_single_node_test(event_channel_fifo TIMEOUT 60)
register_single_node_test(event_destroy_race TIMEOUT 60)

# Coherence tests (formerly CDAG — names track the RC protocol naming)
register_single_node_test(coherence_basic ARGS 32 0 0 0)
register_single_node_test(coherence_rw_chain ARGS 4)
register_single_node_test(coherence_rw_ordering)
register_single_node_test(coherence_rw_ro_rw)
register_single_node_test(coherence_rw_rw_chain_seal)
register_single_node_test(coherence_ro_accumulation)

# Graph tests
register_single_node_test(csr)
register_single_node_test(distribution)

# --- coh_* coherence tests (minimal API variants) ---
register_single_node_test(coh_basic)
register_single_node_test(coh_rw_chain ARGS 4)
register_single_node_test(coh_rw_ordering)
register_single_node_test(coh_rw_ro_rw)
register_single_node_test(coh_rw_rw_chain_seal)
register_single_node_test(coh_ro_accumulation)
register_single_node_test(coh_mixed_local_remote)

# --- DB type tests ---
register_single_node_test(db_pin)

# --- Termination tests ---
register_single_node_test(term_shutdown)
set_tests_properties(term_shutdown PROPERTIES
    PASS_REGULAR_EXPRESSION "TERM_TEST_EXIT")

register_single_node_test(term_abort ARGS 0 42)
set_tests_properties(term_abort PROPERTIES
    PASS_REGULAR_EXPRESSION "TERM_ABORT rc=42")

# term_sigfpe / term_sigsegv crash via signal — CTest Exception short-circuits
# PASS_REGULAR_EXPRESSION before evaluating it.  Use a bash wrapper so the
# shell catches the signal exit and CTest sees exit-code 0; the regex then
# matches [ARTS] Crashed: SIG* in the captured output.
add_test(
    NAME term_sigfpe
    COMMAND bash -c "ASAN_OPTIONS=handle_sigfpe=0:handle_segv=0 ${CMAKE_CURRENT_BINARY_DIR}/term_sigfpe 2>&1; true"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(term_sigfpe PROPERTIES
    LABELS "single_node"
    TIMEOUT 10
    RESOURCE_LOCK "arts_runtime"
    PASS_REGULAR_EXPRESSION "\\[ARTS\\] Crashed: SIGFPE")

add_test(
    NAME term_sigsegv
    COMMAND bash -c "ASAN_OPTIONS=handle_segv=0 ${CMAKE_CURRENT_BINARY_DIR}/term_sigsegv 2>&1; true"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(term_sigsegv PROPERTIES
    LABELS "single_node"
    TIMEOUT 10
    RESOURCE_LOCK "arts_runtime"
    PASS_REGULAR_EXPRESSION "\\[ARTS\\] Crashed: SIGSEGV")

# --- Pure unit tests (no ARTS runtime) ---
register_pure_unit_test(lockfree_stack_test)
register_pure_unit_test(lockfree_marked_list_test)
register_pure_unit_test(coherence_buffer_test TIMEOUT 60)

# Counter smoke test: verifies counter output files are written.
register_single_node_test(counter_smoke TIMEOUT 30)
set_tests_properties(counter_smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "COUNTER_SMOKE_PASS")

register_single_node_test(edt_dep_variants)

register_single_node_test(event_hint_presets)

register_single_node_test(db_labeled_guid)

register_single_node_test(edt_epoch_api)

# --- Multi-node tests ---

register_multinode_test(multinode_db)
register_multinode_test(multinode_edt)
register_multinode_test(multinode_epoch)
register_multinode_test(multinode_event)
register_multinode_test(active_message_db ARGS 64)
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

register_multinode_test(multinode_stdout_forward)
set_tests_properties(multinode_stdout_forward_2n PROPERTIES
    PASS_REGULAR_EXPRESSION "MULTINODE_STDOUT_FORWARD_SENTINEL 42")
set_tests_properties(multinode_stdout_forward_4n PROPERTIES
    PASS_REGULAR_EXPRESSION "MULTINODE_STDOUT_FORWARD_SENTINEL 42")
set_tests_properties(multinode_stdout_forward_5n PROPERTIES
    PASS_REGULAR_EXPRESSION "MULTINODE_STDOUT_FORWARD_SENTINEL 42")

register_multinode_test(multinode_coh TIMEOUT 60)
register_multinode_test(multinode_labeled_guid TIMEOUT 30)
register_multinode_test(multinode_event_hint TIMEOUT 30)

# 3-rank coherence stress: B.3 LOCK_REQ-before-DB_CREATE race regression.
# Uses arts_2node.cfg; the test SKIPs cleanly when rank_count < 3, so on a
# 2-node-only setup this becomes a no-op.  Re-enable on a 3+ rank cfg to
# actually exercise the race.
add_test(
    NAME coherence_lock_req_before_create
    COMMAND coherence_lock_req_before_create
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(coherence_lock_req_before_create PROPERTIES
    LABELS "coherence_stress;multinode"
    TIMEOUT 120
    RESOURCE_LOCK "arts_runtime"
    ENVIRONMENT "ARTS_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/arts_2node.cfg"
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

# Lock-free stack stress (Task 4h): 8 producers x 1M nodes, single
# consumer reverse_drain.  Pure unit test — does not start the ARTS
# runtime, but RESOURCE_LOCK is harmless and keeps it serialized with
# other tests under ctest -j.
register_single_node_test(lockfree_stack_stress TIMEOUT 60)

# Lock-free pool stress (Task 4m): DWCAS tagged-head Treiber stack ABA
# scenario + batch fetch / release contention.  Pure unit test, no
# runtime needed.
register_single_node_test(lockfree_pool_stress TIMEOUT 60)

# Tiered pool stress (Task 4m-2): 3-tier (tcache + NUMA + global)
# symmetric + asymmetric workloads.  Pure unit test, no runtime needed.
register_single_node_test(tiered_pool_stress TIMEOUT 60)

# Distributed coherence stress: B.2.  Uses arts_2node.cfg (the test SKIPs
# cleanly if rank_count < 2).
add_test(
    NAME coherence_stress_dist
    COMMAND coherence_stress_dist
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(coherence_stress_dist PROPERTIES
    LABELS "coherence_stress;multinode"
    TIMEOUT 120
    RESOURCE_LOCK "arts_runtime"
    ENVIRONMENT "ARTS_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/arts_2node.cfg"
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
