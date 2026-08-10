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
    route_table
    route_table_remote_guid
    coherence_basic
    coherence_rw_chain
    coherence_rwro_phase
    coherence_rw_ordering
    coherence_rw_ro_rw
    coherence_rw_rw_chain_seal
    coherence_ro_accumulation
    coherence_mixed_local_remote
    coherence_destroy_during_acquire
    coherence_stress_single_node
    coherence_stress_dist
    coherence_payload_large
    coherence_satisfy_before_release
    coherence_multi_writer_same_addr
    coherence_multi_writer_multi_db
    coherence_ro_acquire_stress
    coherence_home_producer_ro
    coherence_zero_size_xfer
    coherence_rw_multihop
    coherence_rw_pipeline
    coherence_owner_ownership_reorder
    coherence_owner_confirm_gate
    lockfree_stack_stress
    lockfree_pool_stress
    route_table_install_race
    event_once_storm
    event_sticky_late_bind
    event_channel_fifo
    event_destroy_race
    pin_db
    active_message_db
    termination_detection
    finish_event_recursive
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
    event_channel_advanced
    guid_basic
    guid_range
    hint_routing
    node_query
    record_dep_at
    utility_api
    multinode_edt
    finish_event_mn_termination
    multinode_event
    multinode_coherence
    multinode_event_types
    # --- Graph tests ---
    csr
    distribution
    # --- Array DB tests ---
    # --- DB type tests ---
    db_pin
    # --- Termination tests ---
    term_shutdown
    term_abort
    term_sigfpe
    term_sigsegv
    # --- Multinode (built here, registered below) ---
    multinode_stdout_forward
    # --- Pure unit tests (no ARTS runtime) ---
    lockfree_stack_test
    coherence_buffer_test
    config_ports_contract
    port_block_relocation
    stdio_forward_scale_test
    regpool_basic
    regpool_guard
    # --- New unit tests (Tasks 6-12) ---
    counter_smoke
    edt_dep_variants
    event_hint_presets
    db_labeled_guid
    multinode_labeled_guid
    multinode_event_hint
    per_db_buf_pool
    # --- Finish-EDT mechanism tests ---
    finish_event_basic
    finish_event_inherit
    finish_event_multinode
    finish_event_stress
    finish_event_wait
    finish_event_nested
    coherence_multi_writer_dist
    coherence_writer_ro_overlap
)

# Function to create a test executable
function(add_arts_test test_name)
    # Sources live under one of the ocr/ unit/ subdirectories.
    #   ocr/  — portable OCR programs (arts_rt main, no globals, no protocol
    #           #if; valid under any coherence protocol)
    #   unit/ — whitebox tests (internal headers, global state, or protocol-#if
    #           guards; exercise ARTS-internal mechanisms)
    set(_src "")
    foreach(_sub ocr unit)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_sub}/${test_name}.c)
            set(_src ${_sub}/${test_name}.c)
        endif()
    endforeach()
    add_executable(${test_name} ${_src})

    # Set include directories
    target_include_directories(${test_name} PRIVATE ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR} ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})

    # Whitebox tests include protocol-sensitive internal headers (ooo.h,
    # coherence/types.h, protocol.h) that require exactly one compile-time
    # coherence-protocol selection.  The selection is applied per-target (not
    # globally) so it cannot collide with the per-variant benchmark libraries;
    # apply this build's option-selected protocol to the test the same way the
    # ARTS library targets receive it.
    arts_apply_protocol(${test_name} ${ARTS_COHERENCE_ARM} ${ARTS_WRITE_POLICY} ${ARTS_RELEASE_POLICY})

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

        # Protocol-sensitive internal headers need the compile-time coherence
        # selection applied per-target (see add_arts_test).
        arts_apply_protocol(${test_name} ${ARTS_COHERENCE_ARM} ${ARTS_WRITE_POLICY} ${ARTS_RELEASE_POLICY})

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

# A local multinode run claims its own port block: before spawning the peers
# the runtime bind-probes a pid-seeded block, slides past whatever holds it, and
# hands the result down.  So the suites no longer hand out bases by hand, and a
# straggler, an orphan, or an unrelated process squatting a port is absorbed by
# the runtime rather than poisoning the next test.  Do not reintroduce a fixed
# base here: a fixed base is exactly the shared state that made one test's
# leftovers everyone else's problem.

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
        ENVIRONMENT "ARTS_CONFIG=${CMAKE_SOURCE_DIR}/configs/local/test/1n.cfg"
    )
endfunction()

function(register_multinode_test test_name)
    cmake_parse_arguments(ARG "" "TIMEOUT" "ARGS;VARIANTS" ${ARGN})
    if(NOT ARG_TIMEOUT)
        set(ARG_TIMEOUT 60)
    endif()
    if(NOT ARG_VARIANTS)
        # All four local multinode flavours.  Names map 1:1 to the cfg files in
        # configs/local/; each test's ARTS_CONFIG env points straight at the
        # source cfg (no copy), so ctest runs standalone.
        #   2n     — 2 nodes, 6w/1p per node (worker-heavy default)
        #   3n     — 3 nodes, 3w/1p per node
        #   4n     — 4 nodes, 2w/1p per node
        #   2n_io  — 2 nodes, 5w/2p per node (network-heavy variant:
        #            multiple progress threads break per-peer message FIFO,
        #            exercising concurrent cross-rank handler dispatch)
        set(ARG_VARIANTS 2n 3n 4n 2n_io)
    endif()
    foreach(variant IN LISTS ARG_VARIANTS)
        if(variant STREQUAL "2n_io")
            set(_variant_port_count 2)
        else()
            set(_variant_port_count 1)
        endif()
        add_test(
            NAME ${test_name}_${variant}
            COMMAND ${test_name} ${ARG_ARGS}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        set_tests_properties(${test_name}_${variant} PROPERTIES
            LABELS "multinode"
            TIMEOUT ${ARG_TIMEOUT}
            RESOURCE_LOCK "arts_runtime"
            ENVIRONMENT "ARTS_CONFIG=${CMAKE_SOURCE_DIR}/configs/local/test/${variant}.cfg"
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
        ENVIRONMENT "ARTS_CONFIG=${CMAKE_SOURCE_DIR}/configs/local/gpu/1n.cfg;ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0:alloc_dealloc_mismatch=0"
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

# Finish-event / sync tests
register_single_node_test(finish_event_recursive ARGS 3)
register_single_node_test(termination_detection ARGS 3)

# Event tests
register_single_node_test(event_basic)
register_single_node_test(event_chain)
register_single_node_test(event_create_destroy_basic)
register_single_node_test(event_satisfy_once)
register_single_node_test(event_add_dep_idem_fastpath)
register_single_node_test(event_channel_lifetime)
register_single_node_test(event_idem_silent_oversatisfy)
register_single_node_test(event_channel_advanced)

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

# Coherence tests — exercise the OCR-model ownership protocol paths
register_single_node_test(coherence_basic ARGS 32 0 0 0)
register_single_node_test(coherence_rw_chain)
register_single_node_test(coherence_rwro_phase)
register_single_node_test(coherence_rw_ordering)
register_single_node_test(coherence_rw_ro_rw)
register_single_node_test(coherence_rw_rw_chain_seal)
register_single_node_test(coherence_ro_accumulation)

# Graph tests
register_single_node_test(csr)
register_single_node_test(distribution)

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
# matches [ARTS] Crashed: SIG* in the captured output.  Hand-registered, so
# ARTS_CONFIG has to be set here too: without it the runtime falls back to
# ./arts.cfg in the working directory, which the build tree does not contain.
add_test(
    NAME term_sigfpe
    COMMAND bash -c "ASAN_OPTIONS=handle_sigfpe=0:handle_segv=0 ${CMAKE_CURRENT_BINARY_DIR}/term_sigfpe 2>&1; true"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(term_sigfpe PROPERTIES
    LABELS "single_node"
    TIMEOUT 10
    RESOURCE_LOCK "arts_runtime"
    ENVIRONMENT "ARTS_CONFIG=${CMAKE_SOURCE_DIR}/configs/local/test/1n.cfg"
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
    ENVIRONMENT "ARTS_CONFIG=${CMAKE_SOURCE_DIR}/configs/local/test/1n.cfg"
    PASS_REGULAR_EXPRESSION "\\[ARTS\\] Crashed: SIGSEGV")

# --- Pure unit tests (no ARTS runtime) ---
register_pure_unit_test(lockfree_stack_test)
register_pure_unit_test(coherence_buffer_test TIMEOUT 60)
# Forks to probe the ARTS_ERROR death paths of the port contract; loads cfgs
# only, so it binds nothing and needs no runtime.
register_pure_unit_test(config_ports_contract TIMEOUT 30)
set_tests_properties(config_ports_contract PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS config_ports_contract")
# Binds real sockets to occupy a block, then checks the selector moves off it.
register_pure_unit_test(port_block_relocation TIMEOUT 30)
set_tests_properties(port_block_relocation PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS port_block_relocation|SKIP port_block_relocation")
register_pure_unit_test(stdio_forward_scale_test)

# regpool_basic/regpool_guard assert the confinement invariant (every
# returned pointer resolves through arts_regpool_lookup to a registered
# slab), which only the arena-allocator implementation provides.  The
# ARTS_MALLOC=system fallback has no arena/slab concept — it is a plain
# pass-through allocator, so arts_regpool_lookup always misses there by
# design (see regpool.c's non-mimalloc branch).  Still built under every
# ARTS_MALLOC config for compile coverage; only run where the invariant
# they check actually holds.
if(ARTS_MALLOC STREQUAL "mimalloc")
    register_pure_unit_test(regpool_basic TIMEOUT 60)
    register_pure_unit_test(regpool_guard TIMEOUT 60)
endif()

# net_selfsend — whitebox self-send exercise for the libfabric transport core
# (transport/net.c).  It #includes net.c directly (statics under test) with a
# stubbed dispatch symbol, so it must NOT link the ARTS library (that would
# duplicate net.o / the dispatcher).  It links the real registered-pool object
# (bounce-buffer path needs registered memory) + mimalloc + libfabric, and
# builds only when the OFI transport is on AND the arena allocator is selected
# (registered slabs require it).  No ARTS runtime, no ports -> a pure unit test.
if(ARTS_TRANSPORT_OFI AND ARTS_MALLOC STREQUAL "mimalloc")
    add_executable(net_selfsend unit/net_selfsend.c $<TARGET_OBJECTS:arts_regpool>)
    target_include_directories(net_selfsend PRIVATE
        ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
    target_compile_definitions(net_selfsend PRIVATE ARTS_TRANSPORT_OFI=1 ARTS_LOG_LEVEL=2)
    target_link_libraries(net_selfsend PRIVATE
        Threads::Threads mimalloc-static arts::libfabric ${RT_LIBRARY})
    register_pure_unit_test(net_selfsend TIMEOUT 60)
    set_tests_properties(net_selfsend PROPERTIES ENVIRONMENT "FI_PROVIDER=tcp")

    # net_rendezvous — whitebox one-sided data plane (fi_writedata PUT into a
    # pre-registered landing buffer + txid immediate pairing).  Same standalone
    # harness shape as net_selfsend (includes net.c; must not link libarts).
    add_executable(net_rendezvous unit/net_rendezvous.c $<TARGET_OBJECTS:arts_regpool>)
    target_include_directories(net_rendezvous PRIVATE
        ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
    target_compile_definitions(net_rendezvous PRIVATE ARTS_TRANSPORT_OFI=1 ARTS_LOG_LEVEL=2)
    target_link_libraries(net_rendezvous PRIVATE
        Threads::Threads mimalloc-static arts::libfabric ${RT_LIBRARY})
    register_pure_unit_test(net_rendezvous TIMEOUT 60)
    set_tests_properties(net_rendezvous PROPERTIES ENVIRONMENT "FI_PROVIDER=tcp")

    # net_src_bind — whitebox for arts_net_init's steering hints: net_interface
    # source-bind (loopback), fabric_domain pin, and the fatal missing-interface
    # contract.  Same standalone harness shape as net_selfsend.
    add_executable(net_src_bind unit/net_src_bind.c $<TARGET_OBJECTS:arts_regpool>)
    target_include_directories(net_src_bind PRIVATE
        ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
    target_compile_definitions(net_src_bind PRIVATE ARTS_TRANSPORT_OFI=1 ARTS_LOG_LEVEL=2)
    target_link_libraries(net_src_bind PRIVATE
        Threads::Threads mimalloc-static arts::libfabric ${RT_LIBRARY})
    register_pure_unit_test(net_src_bind TIMEOUT 60)
endif()

# net_addr_frame_bound — whitebox for the bootstrap address-frame length bound
# (arts_net_addr_frame_ok in net.h).  Header-only: it exercises the pure
# predicate, so it needs the internal include dir but no libfabric/runtime link.
# Compiles in every build; the OFI predicate is under test only when the OFI
# transport is on (otherwise the test reports SKIP).
add_executable(net_addr_frame_bound unit/net_addr_frame_bound.c)
target_include_directories(net_addr_frame_bound PRIVATE
    ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
    ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
if(ARTS_TRANSPORT_OFI)
    target_compile_definitions(net_addr_frame_bound PRIVATE ARTS_TRANSPORT_OFI=1)
endif()
target_link_libraries(net_addr_frame_bound PRIVATE Threads::Threads)
register_pure_unit_test(net_addr_frame_bound)

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
    add_executable(${t} unit/${t}.c
        ${CMAKE_SOURCE_DIR}/libs/src/core/utils/shared.c)
    target_include_directories(${t} PRIVATE
        ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
    target_link_libraries(${t} PRIVATE Threads::Threads)
    register_pure_unit_test(${t})
endforeach()
foreach(t ${SYNC_MPSC_TESTS})
    add_executable(${t} unit/${t}.c)
    target_include_directories(${t} PRIVATE
        ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
    target_link_libraries(${t} PRIVATE Threads::Threads)
    register_pure_unit_test(${t})
endforeach()

# --- Work-stealing deque (deque.c) + array_list (array_list.c) ---
# Standalone like the SYNC_* tests above: each compiles the target source TU
# directly and provides its own libc-backed arts_calloc_align/arts_free
# (deque) or arts_malloc/arts_free (array_list) + an arts_atomic_cswap_u64
# shim (deque), so no ARTS library link is needed.  The deque RACE tests
# (chaselev/tiebreak/grow) validate the exactly-once Chase-Lev contract;
# they are functionally green under ASan but DELIBERATELY trip ThreadSanitizer
# because deque.c uses the volatile + __sync_synchronize idiom rather than C11
# atomics (suspected bug B052 / weak-mem ordering) — that TSan report is a true
# property of the runtime, not a test defect.
set(DEQUE_UNIT_TESTS
    deque_single_thread deque_chaselev_race deque_last_element_tiebreak
    deque_grow_during_steal)
foreach(t ${DEQUE_UNIT_TESTS})
    add_executable(${t} unit/${t}.c
        ${CMAKE_SOURCE_DIR}/libs/src/core/utils/deque.c)
    target_include_directories(${t} PRIVATE
        ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
    target_link_libraries(${t} PRIVATE Threads::Threads)
endforeach()
register_pure_unit_test(deque_single_thread)
register_pure_unit_test(deque_chaselev_race TIMEOUT 60)
register_pure_unit_test(deque_last_element_tiebreak TIMEOUT 60)
register_pure_unit_test(deque_grow_during_steal TIMEOUT 60)

add_executable(array_list_basic unit/array_list_basic.c
    ${CMAKE_SOURCE_DIR}/libs/src/core/utils/array_list.c)
target_include_directories(array_list_basic PRIVATE
    ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
    ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
target_link_libraries(array_list_basic PRIVATE Threads::Threads)
register_pure_unit_test(array_list_basic)

# --- OoO defer/replay engine (ooo.c) ---
# Standalone like the SYNC_*/deque tests: each test TU #includes the runtime's
# ooo.c so the file-static g_ooo_table + engine bodies are compiled in, links
# shared.c for the cb shared-ptr slot, and supplies counting handler stubs +
# a single-slot arts_route_table_reserve_or_lookup + libc malloc shims (no ARTS
# runtime).  The per-config protocol macros (ARTS_PROTOCOL_*/ARTS_WRITE_POLICY_*/ARTS_RELEASE_*) are
# inherited from the top-level add_compile_definitions, so the g_ooo_table
# preprocessor arms + the tests' matching handler-stub arms select the same
# kind set for whichever protocol this build dir is configured for.  ooo.c pulls
# the generated counter Preamble.h, hence ARTS_BUILD_INTERNAL_INCLUDE_DIR.
set(OOO_UNIT_TESTS
    ooo_toctou_rescue ooo_drain_repush ooo_table_completeness)
foreach(t ${OOO_UNIT_TESTS})
    add_executable(${t} unit/${t}.c
        ${CMAKE_SOURCE_DIR}/libs/src/core/utils/shared.c)
    target_include_directories(${t} PRIVATE
        ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
    # ooo.c (included directly by each test TU) pulls arts/ooo.h which requires
    # exactly one compile-time protocol selection — same reason add_arts_test
    # calls arts_apply_protocol.
    arts_apply_protocol(${t} ${ARTS_COHERENCE_ARM} ${ARTS_WRITE_POLICY} ${ARTS_RELEASE_POLICY})
    target_link_libraries(${t} PRIVATE Threads::Threads)
endforeach()
register_pure_unit_test(ooo_toctou_rescue TIMEOUT 60)
register_pure_unit_test(ooo_drain_repush TIMEOUT 60)
register_pure_unit_test(ooo_table_completeness)

# --- atomics.c primitives + spin locks (C23p) ---
# Standalone like the SYNC_*/deque tests: link the runtime's atomics.c TU
# directly (libc-only dependency), no ARTS library.  atomics_rmw_conventions
# pins the per-function pre/post return conventions; atomics_rmw_contention
# proves the fetch_add tickets form a gap-free partition (linearizability);
# atomics_locks proves mutual-exclusion (no lost updates), the try_lock state
# machine, and the reader/writer no-overlap invariant, and documents B130
# (writer_try_lock blocks on reader drain).  atomics_locks is functionally
# green under ASan but DELIBERATELY trips ThreadSanitizer: atomics.c's RW lock
# uses plain `volatile` spin-hint reads of *write_lock/*read_lock racing with
# the barrier'd atomic RMWs on the same word — a true C-UB data race property
# of the runtime (benign on x86 TSO), not a test defect.
set(ATOMICS_UNIT_TESTS
    atomics_rmw_conventions atomics_rmw_contention atomics_locks)
foreach(t ${ATOMICS_UNIT_TESTS})
    add_executable(${t} unit/${t}.c
        ${CMAKE_SOURCE_DIR}/libs/src/core/utils/atomics.c)
    target_include_directories(${t} PRIVATE
        ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
    target_link_libraries(${t} PRIVATE Threads::Threads)
endforeach()
register_pure_unit_test(atomics_rmw_conventions)
register_pure_unit_test(atomics_rmw_contention TIMEOUT 60)
register_pure_unit_test(atomics_locks TIMEOUT 60)

# --- malloc.c allocator + footprint, diag.c timestamp/printf (C23p) ---
# Each test #includes the target runtime .c TU directly (with the heavy header
# include guards pre-defined + minimal ARTS_ERROR/ARTS_ALIGNED/footprint stubs
# inside the test), so no extra source and no ARTS library link is needed.
# malloc_alignment is the ASan target (heap safety + a-aligned user pointers);
# malloc_footprint_balance DELIBERATELY fails (B132): realloc shrink+free leaks
# (old-new) bytes from BYTES_MEMORY_FOOTPRINT because free subtracts only the
# shrunken header size — a true runtime accounting bug, left unmasked.
# diag_timestamp pins best-effort per-thread monotonicity + fold sanity and
# documents B133 (arts_printf prefix+body are two non-atomic stdio calls).
set(MISC_UTIL_UNIT_TESTS
    malloc_footprint_balance malloc_alignment diag_timestamp)
foreach(t ${MISC_UTIL_UNIT_TESTS})
    add_executable(${t} unit/${t}.c)
    target_include_directories(${t} PRIVATE
        ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR})
    target_link_libraries(${t} PRIVATE Threads::Threads)
endforeach()
# These tests #include malloc.c directly; when mimalloc is selected the TU
# needs <mimalloc.h> on its include path, inherited from mimalloc-static.
if(ARTS_MALLOC STREQUAL "mimalloc")
    foreach(t malloc_footprint_balance malloc_alignment)
        target_link_libraries(${t} PRIVATE mimalloc-static)
    endforeach()
endif()
register_pure_unit_test(malloc_footprint_balance TIMEOUT 60)
register_pure_unit_test(malloc_alignment)
register_pure_unit_test(diag_timestamp TIMEOUT 60)

# Counter smoke test: verifies counter output files are written.
register_single_node_test(counter_smoke TIMEOUT 30)
set_tests_properties(counter_smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "COUNTER_SMOKE_PASS")

register_single_node_test(edt_dep_variants)

register_single_node_test(event_hint_presets)

register_single_node_test(db_labeled_guid)

register_pure_unit_test(per_db_buf_pool)

# --- Finish-EDT mechanism tests ---
register_single_node_test(finish_event_basic)
register_single_node_test(finish_event_inherit)
register_single_node_test(finish_event_wait TIMEOUT 30)
set_tests_properties(finish_event_wait PROPERTIES
    PASS_REGULAR_EXPRESSION "OK wait-style")
register_single_node_test(finish_event_nested)
set_tests_properties(finish_event_nested PROPERTIES
    PASS_REGULAR_EXPRESSION "OK nested")
register_multinode_test(finish_event_multinode TIMEOUT 30)
set_tests_properties(finish_event_multinode_2n PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS: cross-node finish-scope drained")

register_multinode_test(finish_event_stress TIMEOUT 300)
set_tests_properties(finish_event_stress_2n PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS: 100 iterations of recursive cross-node finish-EDT")

# --- Multi-node tests ---

register_multinode_test(multinode_edt)
register_multinode_test(finish_event_mn_termination)
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
register_multinode_test(coherence_rw_pipeline TIMEOUT 40)
register_multinode_test(coherence_rw_chain TIMEOUT 30)
register_multinode_test(coherence_rwro_phase TIMEOUT 30)

register_multinode_test(multinode_stdout_forward)
set_tests_properties(multinode_stdout_forward_2n PROPERTIES
    PASS_REGULAR_EXPRESSION "MULTINODE_STDOUT_FORWARD_SENTINEL 42")

register_multinode_test(multinode_labeled_guid TIMEOUT 30)
register_multinode_test(multinode_event_hint TIMEOUT 30)

# B.3 LOCK_REQ-before-DB_CREATE race regression is registered at 3n/4n (where
# the race is actually exercised) as `excl_req_before_create` in
# generated_tests.cmake.  The former 2n single-entry `coherence_lock_req_before_create`
# here was a no-op (SKIPs at rank<3) over byte-identical source, so it was removed.

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
register_single_node_test(coherence_multi_writer_multi_db TIMEOUT 60)
set_property(TEST coherence_multi_writer_multi_db APPEND PROPERTY
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
    ENVIRONMENT "ARTS_CONFIG=${CMAKE_SOURCE_DIR}/configs/local/test/2n.cfg"
)

# Large-payload cross-rank coherence transfer: a ~256 KiB DB is stamped by a
# home-rank writer and RW-verified by workers on other ranks, so every hop ships
# the whole payload over the wire.  This is the guard for the transport's
# message framing — a payload split across frames the connectionless RX cannot
# reassemble corrupts silently (checksum mismatch -> FAIL) or mis-dispatches
# (spurious shutdown -> TIMEOUT); the sizeof(int) DBs elsewhere never reach it.
# Standard MN variants (2n/3n/4n/2n_io); SKIPs cleanly at rank_count < 2.
register_multinode_test(coherence_payload_large TIMEOUT 120)
foreach(_v 2n 3n 4n 2n_io)
    set_property(TEST coherence_payload_large_${_v} APPEND PROPERTY
        LABELS "coherence_stress")
    # A framing bug can mis-dispatch the raw payload frame -> spurious shutdown,
    # ending the run cleanly (exit 0) BEFORE any worker verifies.  Require the
    # end-of-run PASS line so that "verification never completed" fails, and
    # fail on any worker's corruption abort.
    set_tests_properties(coherence_payload_large_${_v} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: 4 DBs"
        FAIL_REGULAR_EXPRESSION "FAIL:")
endforeach()

# Add-dependence-then-release hand-off: consumers whose slots satisfied BEFORE
# the producer's release must observe the release-published bytes.  A DB whose
# home is neither the creator nor pre-seeded exposes a serve-early bug as a
# consumer NULL/zero read (FAIL/abort — rank death still exits 0 through the
# survivor's clean shutdown, so require the PASS line) or as a never-woken
# parked acquire (missing PASS -> timeout).
# NOT registered under WRF_VAL: the release-gated hand-off is an OCR-model
# (exclusive-RW serialization) guarantee.  WRF_VAL's DB-WRF admission has no
# ownership round — a reader satisfied mid-hold legally reads the current
# canonical snapshot, so this assertion is outside that contract.
if(NOT ARTS_COHERENCE_ARM STREQUAL "WRF_VAL")
register_multinode_test(coherence_satisfy_before_release TIMEOUT 60)
foreach(_v 2n 3n 4n 2n_io)
    set_property(TEST coherence_satisfy_before_release_${_v} APPEND PROPERTY
        LABELS "coherence_stress")
    set_tests_properties(coherence_satisfy_before_release_${_v} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: 4 DBs observed|SKIP"
        FAIL_REGULAR_EXPRESSION "FAIL:")
endforeach()
endif()

# Lazy-protocol ownership-transfer TRANSFER/INVALIDATE two-wire reorder stress.
# MUST use
# a progress_threads>=2 config (arts_2n_io.cfg = 2 nodes, 5w/2p) so the
# TRANSFER_OWNERSHIP(+1) and follow-up INVALIDATE(-1) to the same just-granted
# owner can be dispatched out of order by different progress threads — the
# window the commutative-signed writer_count fix closes.  Drives a few thousand
# ownership transfers and PASSES on chain completion (no transfer lost = no
# stranded RW acquirer); a lost transfer would strand a queued acquirer and hang
# (the regression), caught by the test's watchdog under this TIMEOUT.  The exact
# end-sum is a diagnostic only, not a gate (a cross-rank publish can be dropped
# by a separate FIFO-ordering defect orthogonal to writer_count — see the test
# file header).  SKIPs cleanly when rank_count < 2 or under the RELAXED model
# (no exclusive-RW chain).
add_test(
    NAME coherence_owner_ownership_reorder
    COMMAND coherence_owner_ownership_reorder
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
set_tests_properties(coherence_owner_ownership_reorder PROPERTIES
    LABELS "coherence_stress;multinode"
    TIMEOUT 120
    RESOURCE_LOCK "arts_runtime"
    ENVIRONMENT "ARTS_CONFIG=${CMAKE_SOURCE_DIR}/configs/local/test/2n_io.cfg"
)

# OWNER-placement stale-RO ownership-confirm gate: RO acquire causally ordered after a
# remote RW writer must observe the writer's value, not a stale snapshot from
# the old owner.  The race window is home's rw_holder lag behind INSTALL_ACK.
# The 2n_io config (multiple progress threads -> wire reorder) exposes it;
# 1n passes trivially (no ownership transfer).  RELAXED SKIPs cleanly.
register_single_node_test(coherence_owner_confirm_gate TIMEOUT 120)
register_multinode_test(coherence_owner_confirm_gate TIMEOUT 120)

# coherence_multi_writer_dist: multinode cross-rank RW same-address.  The
# requester is NOT the home rank, so increments cross the wire (LOCK_REQUEST/
# GRANT or each protocol's equivalent) and the final RO read must observe every
# writer's atomic update.  An ordinary OCR program — registered under every
# protocol; at >=3 ranks a DB homed on a non-zero rank has two distinct remote
# writer ranks, exercising the cross-rank publish handoff that 2n cannot.
register_multinode_test(coherence_multi_writer_dist TIMEOUT 120 VARIANTS 2n 3n 4n)
foreach(_v 2n 3n 4n)
    set_property(TEST coherence_multi_writer_dist_${_v} APPEND PROPERTY
        LABELS "coherence_stress")
endforeach()

# coherence_writer_ro_overlap: cross-rank RW write write windows OVERLAPPED by
# concurrent RO readers on other ranks — the one regime the rest of the suite
# leaves uncovered (coherence_stress_dist is RW-only across ranks; the
# large-payload and rwro_phase tests phase-separate writers from readers).  Runs
# of same-rank RW EDTs hold a write window open across several releases while readers
# stream in; the run boundaries migrate the write window so redirects race live
# hand-offs.  An ordinary OCR program: the header has a single event-ordered
# writer that every acquire must observe verbatim, and a round's writers all
# stamp the churn region with the same byte, so the only legal observations are
# this round's and the previous round's stamps.  Both assertions survive a
# whole-DB last-writer-wins publish, so it is registered under every arm
# including WRF_VAL.  Single-node registration covers the all-local paths; the
# multinode variants carry the cross-rank ones.
register_single_node_test(coherence_writer_ro_overlap TIMEOUT 120)
register_multinode_test(coherence_writer_ro_overlap TIMEOUT 180)
set(_writer_ro_cases
    coherence_writer_ro_overlap
    coherence_writer_ro_overlap_2n
    coherence_writer_ro_overlap_3n
    coherence_writer_ro_overlap_4n
    coherence_writer_ro_overlap_2n_io)
foreach(_t IN LISTS _writer_ro_cases)
    set_property(TEST ${_t} APPEND PROPERTY LABELS "coherence_stress")
    # A protocol that ends the run early (spurious shutdown) would exit 0 with
    # nothing verified — require the end-of-run scalar, and fail on any
    # acquirer's violation report.
    set_tests_properties(${_t} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: rw-write window/ro-overlap rounds=16 dbs=8 reads=1024 writes=1152"
        FAIL_REGULAR_EXPRESSION "FAIL:")
endforeach()

# --- DB-WRF contract tests (portable OCR programs, sources under ocr/) ---
# These OCR programs exercise the DB-WRF contract (defined results only when
# accesses are race-free at DB granularity).  Their last-writer / racy outcomes
# are only well-defined under the WRF_VAL protocol, so the CTest entries are
# registered only when ARTS_COHERENCE_ARM is WRF_VAL; the binaries build in
# every tree.

add_arts_test(db_wrf_multi_writer_concurrent)
add_arts_test(db_wrf_invariant)
add_arts_test(db_wrf_local_hit_dedup)
add_arts_test(db_wrf_counter_race)

if(ARTS_COHERENCE_ARM STREQUAL "WRF_VAL")
    # db_wrf_invariant (single-node): home-rank writer and home-rank
    # reader; no PUBLISH round-trip.  Validates finish-event happens-before
    # ordering.
    add_test(
        NAME db_wrf_invariant
        COMMAND db_wrf_invariant
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(db_wrf_invariant PROPERTIES
        LABELS "single_node"
        TIMEOUT 30
        RESOURCE_LOCK "arts_runtime"
        ENVIRONMENT "ARTS_CONFIG=${CMAKE_SOURCE_DIR}/configs/local/test/1n.cfg"
        PASS_REGULAR_EXPRESSION "DB_WRF: PASS"
    )

    # db_wrf_invariant (2-node): writer on rank 1, reader on rank 0
    # (home).  Exercises the cross-rank RW PUBLISH path: writer_count bump
    # on acquire ensures release_rw sends PUBLISH; home delivers updated
    # data to the RO reader.
    add_test(
        NAME db_wrf_invariant_2n
        COMMAND db_wrf_invariant
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(db_wrf_invariant_2n PROPERTIES
        LABELS "multinode"
        TIMEOUT 60
        RESOURCE_LOCK "arts_runtime"
        ENVIRONMENT "ARTS_CONFIG=${CMAKE_SOURCE_DIR}/configs/local/test/2n.cfg"
        PASS_REGULAR_EXPRESSION "DB_WRF: PASS"
    )

    # db_wrf_local_hit_dedup: N concurrent same-rank RO acquires on a DB
    # homed at a different rank — exercises the GET_DATA / STILL_VALID dedup
    # code path under the WRF_VAL protocol.  Requires 2-node config.
    add_test(
        NAME db_wrf_local_hit_dedup_2n
        COMMAND db_wrf_local_hit_dedup
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(db_wrf_local_hit_dedup_2n PROPERTIES
        LABELS "multinode"
        TIMEOUT 60
        RESOURCE_LOCK "arts_runtime"
        ENVIRONMENT "ARTS_CONFIG=${CMAKE_SOURCE_DIR}/configs/local/test/2n.cfg"
        PASS_REGULAR_EXPRESSION "DB_WRF_LOCAL_HIT_DEDUP: PASS"
    )

    # db_wrf_multi_writer_concurrent: manual test only — non-deterministic
    # last-writer-wins outcome cannot be expressed as a fixed regex.
    # Run directly with: build_wrf_val_wt/tests/db_wrf_multi_writer_concurrent
endif()

# --- Mode-mismatch detection binary (no CTest entry — driven by run_mode_mismatch.sh) ---
add_executable(coherence_mode_mismatch ocr/coherence_mode_mismatch.c)
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

include(${CMAKE_CURRENT_SOURCE_DIR}/generated_tests.cmake)

# ============================================================================
# Cross-rank failure channel.
#
# ctest only ever sees the MASTER rank's exit status: the local launcher reaps
# its children with waitpid but discards their status, so a rank that aborts or
# exits non-zero is indistinguishable from one that finished cleanly.  For a
# multinode test the only verdict that leaves a non-master rank is what it
# PRINTED, which the launcher forwards to the master's stdout — hence this gate.
#
# It doubles as a second net single-node: a check that runs on a worker thread
# and only prints would otherwise pass on main()'s unconditional return 0.  The
# tests themselves carry their verdict in the exit status (see
# tests/test_failure_status.h); this catches what the status cannot reach.
#
# Conservative form (as in the T294 sweep above): the token matched is one these
# tests print ONLY on a failure branch — a clean run never emits it, and the
# runtime's own [ERROR] log aborts, so it cannot appear either.  Each registered
# variant needs its own entry: a multinode test has no bare-name entry.
# ============================================================================
set_tests_properties(await_publish_ack_shutdown PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(await_publish_ack_shutdown_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(await_publish_ack_shutdown_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(await_publish_ack_shutdown_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(await_publish_ack_shutdown_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(cat_c_ref_balance PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(cat_c_ref_balance_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(cat_c_ref_balance_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(cat_c_ref_balance_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(cat_c_ref_balance_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_install_version_monotone PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_install_version_monotone_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_install_version_monotone_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_install_version_monotone_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_install_version_monotone_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_owner_ownership_reorder PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rw_chain PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rw_chain_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rw_chain_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rw_chain_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rw_chain_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rwro_phase PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rwro_phase_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rwro_phase_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rwro_phase_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rwro_phase_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_stress_dist PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_zero_size_xfer_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_zero_size_xfer_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_zero_size_xfer_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_zero_size_xfer_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(create_coalesce_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(create_coalesce_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(create_coalesce_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(create_coalesce_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(csr_empty_partition PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(csr_load_args PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(csr_load_csr_format PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(csr_load_edgelist PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(ctx_crossrank_proxy_finish_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(ctx_crossrank_proxy_finish_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(ctx_crossrank_proxy_finish_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(ctx_crossrank_proxy_finish_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_acquire_all_bias_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_acquire_all_bias_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_acquire_all_bias_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_acquire_all_bias_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_create_install_race_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_create_install_race_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_create_install_race_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_create_install_race_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_release_alias_slot PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_rw_secure_double_fire PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_rw_secure_double_fire_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_rw_secure_double_fire_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_rw_secure_double_fire_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_rw_secure_double_fire_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_wrf_promote_manual_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_wrf_promote_manual_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_wrf_promote_manual_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_wrf_promote_manual_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(destroy_before_create_defer_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(destroy_before_create_defer_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(destroy_before_create_defer_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(destroy_before_create_defer_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(dist_args_oob PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_crossrank_ptr_payload PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_crossrank_ptr_payload_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_crossrank_ptr_payload_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_crossrank_ptr_payload_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_crossrank_ptr_payload_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_destroy_crossrank_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_destroy_crossrank_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_destroy_crossrank_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_destroy_crossrank_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_output_event PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_output_event_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_output_event_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_output_event_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_output_event_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_satisfy_out_of_range PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(event_destroy_before_create PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(event_destroy_before_create_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(event_destroy_before_create_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(event_destroy_before_create_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(event_destroy_before_create_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_multiconsumer_queue PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_multiconsumer_queue_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_multiconsumer_queue_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_multiconsumer_queue_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_multiconsumer_queue_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_purge_grant_d6_d7 PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_purge_grant_d6_d7_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_purge_grant_d6_d7_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_purge_grant_d6_d7_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_purge_grant_d6_d7_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_samerank_rw_grant_race PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_samerank_rw_grant_race_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_samerank_rw_grant_race_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_samerank_rw_grant_race_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(excl_samerank_rw_grant_race_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(mark_edt_ready_idempotent PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(publish_ack_post_on_miss PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(publish_ack_post_on_miss_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(publish_ack_post_on_miss_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(publish_ack_post_on_miss_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(publish_ack_post_on_miss_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(random_thread_safe PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(self_send_vs_dispatcher_parity PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(self_send_vs_dispatcher_parity_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(self_send_vs_dispatcher_parity_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(self_send_vs_dispatcher_parity_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(self_send_vs_dispatcher_parity_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(snapshot_response_3case PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(snapshot_response_3case_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(snapshot_response_3case_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(snapshot_response_3case_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(snapshot_response_3case_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_baton_invalidate_recheck PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_baton_invalidate_recheck_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_baton_invalidate_recheck_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_baton_invalidate_recheck_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_baton_invalidate_recheck_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_no_acquire PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_no_acquire_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_no_acquire_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_no_acquire_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_no_acquire_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_publish_before_decrement PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_publish_before_decrement_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_publish_before_decrement_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_publish_before_decrement_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_publish_before_decrement_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_rw_drain_count PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_rw_drain_count_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_rw_drain_count_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_rw_drain_count_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_rw_drain_count_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_snapshot_request PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_snapshot_request_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_snapshot_request_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_snapshot_request_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_snapshot_request_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wb_confirm_ack_round PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wb_confirm_ack_round_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wb_confirm_ack_round_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wb_confirm_ack_round_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wb_confirm_ack_round_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_writer_count_nonneg PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_writer_count_nonneg_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_writer_count_nonneg_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_writer_count_nonneg_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_writer_count_nonneg_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wt_publish_ack PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wt_publish_ack_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wt_publish_ack_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wt_publish_ack_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wt_publish_ack_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wt_vs_wb_divergence PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wt_vs_wb_divergence_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wt_vs_wb_divergence_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wt_vs_wb_divergence_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(val_wt_vs_wb_divergence_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(wrf_val_home_vs_nonhome_writer_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(wrf_val_home_vs_nonhome_writer_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(wrf_val_home_vs_nonhome_writer_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(wrf_val_home_vs_nonhome_writer_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(wrf_val_sentinel_snapshot_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(wrf_val_sentinel_snapshot_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(wrf_val_sentinel_snapshot_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(wrf_val_sentinel_snapshot_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
