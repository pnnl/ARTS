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
    coherence_lock_req_before_create
    coherence_destroy_during_acquire
    coherence_stress_single_node
    coherence_stress_dist
    coherence_multi_writer_same_addr
    coherence_ro_acquire_stress
    coherence_home_producer_ro
    coherence_zero_size_xfer
    coherence_rw_multihop
    lockfree_stack_stress
    lockfree_pool_stress
    tiered_pool_stress
    route_table_install_race
    event_once_storm
    event_sticky_late_bind
    event_channel_fifo
    event_destroy_race
    pin_db
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
    event_basic
    event_chain
    event_create_destroy_basic
    event_satisfy_once
    event_add_dep_idem_fastpath
    event_channel_lifetime
    event_idem_silent_oversatisfy
    persistent_event
    persistent_event_advanced
    guid_basic
    guid_range
    hint_routing
    node_query
    record_dep_at
    utility_api
    multinode_edt
    multinode_epoch
    multinode_event
    multinode_coherence
    multinode_event_types
    # --- Graph tests ---
    csr
    distribution
    # --- Array DB tests ---
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
    stdio_forward_scale_test
    # --- New unit tests (Tasks 6-12) ---
    counter_smoke
    edt_dep_variants
    event_hint_presets
    db_labeled_guid
    edt_epoch_api
    multinode_labeled_guid
    multinode_event_hint
    # --- Finish-EDT mechanism tests ---
    finish_event_basic
    finish_event_inherit
    finish_event_multinode
    finish_event_stress
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
    cmake_parse_arguments(ARG "" "TIMEOUT" "ARGS;VARIANTS" ${ARGN})
    if(NOT ARG_TIMEOUT)
        set(ARG_TIMEOUT 60)
    endif()
    if(NOT ARG_VARIANTS)
        # All four local multinode flavours.  Names map 1:1 to cfg files
        # in configs/local/ which run_tests.sh copies into TEST_BIN_DIR.
        #   2n     — 2 nodes, 5w/1s/1r per node (worker-heavy default)
        #   3n     — 3 nodes, 2w/1s/1r per node
        #   4n     — 4 nodes, 1w/1s/1r per node
        #   2n_io  — 2 nodes, 3w/2s/2r per node (network-heavy variant:
        #            multiple sender/receiver threads break per-peer message
        #            FIFO, exercising concurrent cross-rank handler dispatch)
        set(ARG_VARIANTS 2n 3n 4n 2n_io)
    endif()
    foreach(variant IN LISTS ARG_VARIANTS)
        add_test(
            NAME ${test_name}_${variant}
            COMMAND ${test_name} ${ARG_ARGS}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        set_tests_properties(${test_name}_${variant} PROPERTIES
            LABELS "multinode"
            TIMEOUT ${ARG_TIMEOUT}
            RESOURCE_LOCK "arts_runtime"
            ENVIRONMENT "ARTS_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/arts_${variant}.cfg"
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
register_single_node_test(pin_db ARGS 0)

# Array DB tests

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
register_single_node_test(event_channel_lifetime)
register_single_node_test(event_idem_silent_oversatisfy)
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
register_pure_unit_test(stdio_forward_scale_test)

# --- Foundation concurrency primitives (standalone — no ARTS runtime) ---
# Exercise arts_shared_ptr_t (shared.c) and the header-only Vyukov
# arts_mpsc_t in isolation.  These deliberately do NOT link the ARTS
# library: the shared-ptr tests provide their own libc-backed
# arts_calloc/arts_free shims, and the mpsc queue is header-only.  Keeping
# them library-independent lets the foundation be validated regardless of
# the runtime's build state.  libatomic + -mcx16 come from the global
# top-level setup.
set(SYNC_SHARED_TESTS shared_basic shared_atomic_race shared_aba)
set(SYNC_MPSC_TESTS mpsc_concurrent mpsc_drain_race)
foreach(t ${SYNC_SHARED_TESTS})
    add_executable(${t} sync/${t}.c
        ${CMAKE_SOURCE_DIR}/libs/src/core/utils/shared.c)
    target_include_directories(${t} PRIVATE
        ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
    target_link_libraries(${t} PRIVATE Threads::Threads)
    register_pure_unit_test(${t})
endforeach()
foreach(t ${SYNC_MPSC_TESTS})
    add_executable(${t} sync/${t}.c)
    target_include_directories(${t} PRIVATE
        ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
    target_link_libraries(${t} PRIVATE Threads::Threads)
    register_pure_unit_test(${t})
endforeach()

# Counter smoke test: verifies counter output files are written.
register_single_node_test(counter_smoke TIMEOUT 30)
set_tests_properties(counter_smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "COUNTER_SMOKE_PASS")

register_single_node_test(edt_dep_variants)

register_single_node_test(event_hint_presets)

register_single_node_test(db_labeled_guid)

register_single_node_test(edt_epoch_api)

# --- Finish-EDT mechanism tests ---
register_single_node_test(finish_event_basic)
register_single_node_test(finish_event_inherit)
register_multinode_test(finish_event_multinode TIMEOUT 30)
set_tests_properties(finish_event_multinode_2n PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS: cross-node finish-scope drained")

register_multinode_test(finish_event_stress TIMEOUT 300)
set_tests_properties(finish_event_stress_2n PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS: 100 iterations of recursive cross-node finish-EDT")

# --- Multi-node tests ---

register_multinode_test(multinode_edt)
register_multinode_test(multinode_epoch)
register_multinode_test(multinode_event)
register_multinode_test(active_message_db ARGS 64)
register_multinode_test(route_table_remote_guid)
register_multinode_test(remote_db_event)
register_multinode_test(multinode_coherence)
register_multinode_test(multinode_event_types)
register_multinode_test(coherence_mixed_local_remote)
register_multinode_test(coherence_home_producer_ro TIMEOUT 30)
register_multinode_test(coherence_zero_size_xfer TIMEOUT 30)
register_multinode_test(coherence_rw_multihop TIMEOUT 30)

register_multinode_test(multinode_stdout_forward)
set_tests_properties(multinode_stdout_forward_2n PROPERTIES
    PASS_REGULAR_EXPRESSION "MULTINODE_STDOUT_FORWARD_SENTINEL 42")

register_multinode_test(multinode_coh TIMEOUT 60)
register_multinode_test(multinode_labeled_guid TIMEOUT 30)
register_multinode_test(multinode_event_hint TIMEOUT 30)

# 3-rank coherence stress: B.3 LOCK_REQ-before-DB_CREATE race regression.
# Uses arts_2n.cfg; the test SKIPs cleanly when rank_count < 3, so on a
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
    ENVIRONMENT "ARTS_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/arts_2n.cfg"
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

# Distributed coherence stress: B.2.  Uses arts_2n.cfg (the test SKIPs
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
    ENVIRONMENT "ARTS_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/arts_2n.cfg"
)

# --- LC-specific test binaries ---
# These tests target the LC memory model.  They are built unconditionally
# so the binary exists in all three build trees; CTest entries are only
# registered when ARTS_MEMORY_MODEL is LC.

add_arts_test(lc_multi_writer_concurrent)
add_arts_test(lc_drf_invariant)
add_arts_test(lc_local_hit_dedup)
add_arts_test(lc_counter_race)

if(ARTS_MEMORY_MODEL STREQUAL "LC")
    # lc_drf_invariant (single-node): home-rank writer and home-rank reader;
    # no WRITEBACK round-trip.  Validates epoch happens-before ordering.
    add_test(
        NAME lc_drf_invariant
        COMMAND lc_drf_invariant
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(lc_drf_invariant PROPERTIES
        LABELS "single_node"
        TIMEOUT 30
        RESOURCE_LOCK "arts_runtime"
        PASS_REGULAR_EXPRESSION "LC_DRF: PASS"
    )

    # lc_drf_invariant (2-node): writer on rank 1, reader on rank 0 (home).
    # Exercises the cross-rank RW WRITEBACK path: writer_count bump on
    # acquire ensures release_rw sends WRITEBACK; home delivers updated
    # data to the RO reader.
    add_test(
        NAME lc_drf_invariant_2n
        COMMAND lc_drf_invariant
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(lc_drf_invariant_2n PROPERTIES
        LABELS "multinode"
        TIMEOUT 60
        RESOURCE_LOCK "arts_runtime"
        ENVIRONMENT "ARTS_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/arts_2n.cfg"
        PASS_REGULAR_EXPRESSION "LC_DRF: PASS"
    )

    # lc_local_hit_dedup: N concurrent same-rank RO acquires on a DB homed
    # at a different rank — exercises the GET_DATA / STILL_VALID dedup
    # code path under LC.  Requires 2-node config.
    add_test(
        NAME lc_local_hit_dedup_2n
        COMMAND lc_local_hit_dedup
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(lc_local_hit_dedup_2n PROPERTIES
        LABELS "multinode"
        TIMEOUT 60
        RESOURCE_LOCK "arts_runtime"
        ENVIRONMENT "ARTS_CONFIG=${CMAKE_CURRENT_BINARY_DIR}/arts_2n.cfg"
        PASS_REGULAR_EXPRESSION "LC_LOCAL_HIT_DEDUP: PASS"
    )

    # lc_multi_writer_concurrent: manual test only — non-deterministic
    # last-writer-wins outcome cannot be expressed as a fixed regex.
    # Run directly with: build_lc/tests/lc_multi_writer_concurrent
endif()

# --- Mode-mismatch detection binary (no CTest entry — driven by run_mode_mismatch.sh) ---
add_executable(coherence_mode_mismatch coherence_mode_mismatch.c)
target_include_directories(coherence_mode_mismatch PRIVATE
    ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
    ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
set_property(TARGET coherence_mode_mismatch PROPERTY POSITION_INDEPENDENT_CODE OFF)
target_compile_options(coherence_mode_mismatch PRIVATE -fno-pie -fno-PIE)
target_link_options(coherence_mode_mismatch PRIVATE -no-pie -fno-pie -fno-PIE)
target_link_libraries(coherence_mode_mismatch PRIVATE
    Threads::Threads ${RT_LIBRARY} ${M_LIBRARY} ${ANL_LIBRARY})
if(TARGET arts_shared)
    target_link_libraries(coherence_mode_mismatch PRIVATE arts_shared)
elseif(TARGET arts_static)
    target_link_libraries(coherence_mode_mismatch PRIVATE arts_static)
endif()

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
