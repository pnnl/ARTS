# ============================================================================
# Auto-generated Wave-A pure_unit test registrations.
#
# Generated from research/test-census/waveA-result.json. Do NOT edit by hand;
# regenerate from the census instead.
#
# Protocol selection (ARTS_PROTOCOL_* / ARTS_WRITE_POLICY_* / ARTS_RELEASE_*)
# is supplied GLOBALLY by the build directory's coherence configuration --
# it is deliberately NOT hardcoded here, so each generated test compiles under
# whatever protocol the enclosing build dir selected.
#
# Tests that EXPOSE a runtime bug and are designed to fail/crash are registered
# WITHOUT a PASS_REGULAR_EXPRESSION so they show up as failing (documenting the
# defect). They are NOT masked with WILL_FAIL.
# ============================================================================

# Helper: build a standalone pure_unit test that compiles specific libs/src .c
# TUs directly (it does NOT link libarts) with the canonical include dirs +
# Threads. libatomic + -mcx16 come from the top-level setup (not re-added here).
function(add_pure_unit_src name)
    cmake_parse_arguments(PU "" "TIMEOUT;PASS_REGEX" "SOURCES;DEFINES;LIBS" ${ARGN})
    add_executable(${name} unit/${name}.c ${PU_SOURCES})
    target_include_directories(${name} PRIVATE
        ${ARTS_PUBLIC_INCLUDE_DIR} ${ARTS_INTERNAL_INCLUDE_DIR}
        ${ARTS_BUILD_INTERNAL_INCLUDE_DIR}
        # Some pure_unit TUs '#include' a runtime .c by a path relative to a
        # source root (e.g. "transport/launcher.c", "counter.c") to pull in
        # file-static symbols without de-static-ing the runtime.  Expose the
        # source roots so those includes resolve.
        ${CMAKE_SOURCE_DIR}/libs/src
        ${CMAKE_SOURCE_DIR}/libs/src/core
        ${CMAKE_SOURCE_DIR}/libs/src/core/counter)
    if(PU_DEFINES)
        target_compile_definitions(${name} PRIVATE ${PU_DEFINES})
    endif()
    # Some pure_unit TUs pull arts/ooo.h (directly or via route_table.c) which
    # requires exactly one compile-time coherence-protocol selection.  Apply it
    # unconditionally — tests that do not include protocol-sensitive headers
    # simply leave the macros unused, so this is harmless for all others.
    arts_apply_protocol(${name} ${ARTS_COHERENCE_ARM} ${ARTS_WRITE_POLICY} ${ARTS_RELEASE_POLICY})
    set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE OFF)
    target_compile_options(${name} PRIVATE -fno-pie -fno-PIE)
    target_link_options(${name} PRIVATE -no-pie -fno-pie -fno-PIE)
    target_link_libraries(${name} PRIVATE Threads::Threads ${PU_LIBS})
    if(NOT PU_TIMEOUT)
        set(PU_TIMEOUT 60)
    endif()
    register_pure_unit_test(${name} TIMEOUT ${PU_TIMEOUT})
    if(PU_PASS_REGEX)
        set_tests_properties(${name} PROPERTIES
            PASS_REGULAR_EXPRESSION "${PU_PASS_REGEX}")
    endif()
endfunction()

add_pure_unit_src(lf_lifo_pop_one PASS_REGEX "PASS lf_lifo_pop_one" TIMEOUT 60)
add_pure_unit_src(lf_lifo_drain PASS_REGEX "PASS lf_lifo_drain" TIMEOUT 60)
add_pure_unit_src(lf_lifo_layout PASS_REGEX "PASS lf_lifo_layout" TIMEOUT 60)
add_pure_unit_src(lf_pool_dwcas PASS_REGEX "PASS lf_pool_dwcas" TIMEOUT 60)
add_pure_unit_src(lf_pool_batch PASS_REGEX "PASS lf_pool_batch" TIMEOUT 60)
add_pure_unit_src(mpsc_drain_remaining PASS_REGEX "PASS mpsc_drain_remaining" TIMEOUT 60)
add_pure_unit_src(mpsc_rethread_stub PASS_REGEX "PASS mpsc_rethread_stub" TIMEOUT 60)
add_pure_unit_src(mpsc_transient_empty PASS_REGEX "PASS mpsc_transient_empty" TIMEOUT 60)
add_pure_unit_src(link_list_mpsc SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/utils/link_list.c PASS_REGEX "PASS link_list_mpsc" TIMEOUT 60)
# T016 EXPOSES B118: expected to FAIL/crash under sanitizer (documents runtime bug, do not mask)
add_pure_unit_src(link_list_lifecycle SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/utils/link_list.c TIMEOUT 60)
add_pure_unit_src(shared_compare_exchange SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/utils/shared.c PASS_REGEX "PASS shared_compare_exchange" TIMEOUT 60)
add_pure_unit_src(shared_lifecycle_race SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/utils/shared.c PASS_REGEX "PASS shared_lifecycle_race" TIMEOUT 60)
add_pure_unit_src(route_table_install_replace PASS_REGEX "PASS route_table_install_replace" TIMEOUT 60)
add_pure_unit_src(route_table_install_if_absent PASS_REGEX "PASS route_table_install_if_absent" TIMEOUT 60)
add_pure_unit_src(route_table_move_item PASS_REGEX "PASS route_table_move_item" TIMEOUT 60)
add_pure_unit_src(route_table_segment_grow PASS_REGEX "PASS route_table_segment_grow" TIMEOUT 60)
add_pure_unit_src(route_item_mirror_bridge PASS_REGEX "PASS route_item_mirror_bridge" TIMEOUT 60)
add_pure_unit_src(guid_encoding_roundtrip PASS_REGEX "PASS guid_encoding_roundtrip" TIMEOUT 60)
add_pure_unit_src(guid_from_index_overflow PASS_REGEX "PASS guid_from_index_overflow" TIMEOUT 60)
add_pure_unit_src(guid_reserve_range PASS_REGEX "PASS guid_reserve_range" TIMEOUT 60)
add_pure_unit_src(guid_hash_key_divzero PASS_REGEX "PASS guid_hash_key_divzero" TIMEOUT 60)
add_pure_unit_src(guid_db_seq_alloc_stress PASS_REGEX "PASS guid_db_seq_alloc_stress" TIMEOUT 120)
add_pure_unit_src(route_table_db_shard PASS_REGEX "PASS route_table_db_shard" TIMEOUT 60)
add_pure_unit_src(db_cache_layout PASS_REGEX "PASS db_cache_layout" TIMEOUT 60)
add_pure_unit_src(buffer_payload_roundtrip SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/coherence/buffer.c ${CMAKE_SOURCE_DIR}/libs/src/core/utils/shared.c PASS_REGEX "PASS buffer_payload_roundtrip" TIMEOUT 60)
add_pure_unit_src(buffer_zero_size SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/coherence/buffer.c ${CMAKE_SOURCE_DIR}/libs/src/core/utils/shared.c PASS_REGEX "PASS buffer_zero_size" TIMEOUT 60)
add_pure_unit_src(buffer_version_guard SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/coherence/buffer.c ${CMAKE_SOURCE_DIR}/libs/src/core/utils/shared.c PASS_REGEX "PASS buffer_version_guard" TIMEOUT 60)
add_pure_unit_src(buffer_stub_db_size_learn SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/coherence/buffer.c ${CMAKE_SOURCE_DIR}/libs/src/core/utils/shared.c PASS_REGEX "PASS buffer_stub_db_size_learn" TIMEOUT 60)
add_pure_unit_src(buffer_destroy_vs_acquire SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/coherence/buffer.c ${CMAKE_SOURCE_DIR}/libs/src/core/utils/shared.c PASS_REGEX "PASS buffer_destroy_vs_acquire" TIMEOUT 60)
add_pure_unit_src(rank_u64_map_roundtrip SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/coherence/rank_u64_map.c PASS_REGEX "PASS rank_u64_map_roundtrip" TIMEOUT 60)
add_pure_unit_src(rank_u64_map_advance SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/coherence/rank_u64_map.c PASS_REGEX "PASS rank_u64_map_advance" TIMEOUT 60)
# T056 rank_bitset: the arts_rank_bitset_* symbols live in the protocol-specific
# coherence directory.c, so the source/link selection is keyed on the build dir's
# ${ARTS_COHERENCE_ARM}:
#   VAL -> compile val/directory.c standalone (ARTS_UNIT_STANDALONE_SHIMS shims)
#   WRF_VAL  -> the rank bit-set is not compiled under WRF_VAL (the protocol directory.c
#            bodies do not even build there, as their node/waiter structs are
#            #if'd out), and the test self-skips (prints PASS).  So link NO
#            protocol directory.c — the self-skipping main needs no home symbols.
#   EXCL  -> excl/directory.c pulls transport/edt deps, so link the full libarts
#            (shims auto-compiled-out via the ARTS_UNIT_STANDALONE_SHIMS gate)
if(ARTS_COHERENCE_ARM STREQUAL "EXCL")
    add_arts_test(rank_bitset)
    register_pure_unit_test(rank_bitset TIMEOUT 60)
    set_tests_properties(rank_bitset PROPERTIES PASS_REGULAR_EXPRESSION "PASS rank_bitset")
elseif(ARTS_COHERENCE_ARM STREQUAL "WRF_VAL")
    # No protocol directory.c source: the test body self-skips under WRF_VAL.
    add_pure_unit_src(rank_bitset
        DEFINES ARTS_UNIT_STANDALONE_SHIMS=1 PASS_REGEX "PASS rank_bitset" TIMEOUT 60)
elseif(ARTS_COHERENCE_ARM STREQUAL "INV")
    set(_rank_bitset_dir ${CMAKE_SOURCE_DIR}/libs/src/core/coherence/inv/directory.c)
    add_pure_unit_src(rank_bitset SOURCES ${_rank_bitset_dir}
        DEFINES ARTS_UNIT_STANDALONE_SHIMS=1 PASS_REGEX "PASS rank_bitset" TIMEOUT 60)
else()
    set(_rank_bitset_dir ${CMAKE_SOURCE_DIR}/libs/src/core/coherence/val/directory.c)
    add_pure_unit_src(rank_bitset SOURCES ${_rank_bitset_dir}
        DEFINES ARTS_UNIT_STANDALONE_SHIMS=1 PASS_REGEX "PASS rank_bitset" TIMEOUT 60)
endif()
# T169 EXPOSES B-set-ip-null: expected to FAIL/crash under sanitizer (documents runtime bug, do not mask)
add_pure_unit_src(socket_set_ip_null_ifa LIBS ${CMAKE_DL_LIBS} TIMEOUT 60)
add_pure_unit_src(socket_helpers LIBS ${CMAKE_DL_LIBS} PASS_REGEX "PASS socket_helpers" TIMEOUT 60)
# T180 EXPOSES B071: expected to FAIL/crash under sanitizer (documents runtime bug, do not mask)
# (SEQUENCENUMBERS is an optional SEQ-header ABI variant supplied by the build dir
#  when enabled; not forced here so the default wire layout is exercised.)
add_pure_unit_src(protocol_abi_asserts TIMEOUT 60)
# Real-hardware Dekker litmus for the grant-baton release/re-check pair: the
# fenced shape must never lose an item; a weakened fence turns the runtime's
# 0.2% wedge back into a deterministic failure here.
add_pure_unit_src(grant_baton_dekker_litmus PASS_REGEX "PASS: grant_baton_dekker_litmus" TIMEOUT 120)
# T184 VERIFIES B070 fix: dispatcher now floors header.size >= sizeof(packet)
add_pure_unit_src(dispatcher_payload_size_underflow TIMEOUT 60)
add_pure_unit_src(launcher_shell_quote PASS_REGEX "PASS launcher_shell_quote" TIMEOUT 60)
add_pure_unit_src(launcher_argv_build PASS_REGEX "PASS launcher_argv_build" TIMEOUT 60)
# T192 VERIFIES B110 fix: launcher command builder is overflow-safe (arts_cmd_appendf bounded-append)
add_pure_unit_src(launcher_ssh_command_build DEFINES ARTS_T192_TRIGGER_OOB=1 TIMEOUT 60)
add_pure_unit_src(stdio_forward_fidelity PASS_REGEX "PASS stdio_forward_fidelity" TIMEOUT 60)
add_pure_unit_src(stdio_forward_partial_eintr PASS_REGEX "PASS stdio_forward_partial_eintr" TIMEOUT 60)
add_pure_unit_src(stdio_forward_make_pipe_fail PASS_REGEX "PASS stdio_forward_make_pipe_fail" TIMEOUT 60)
add_pure_unit_src(stdio_forward_fileno_negative PASS_REGEX "PASS stdio_forward_fileno_negative" TIMEOUT 60)
add_pure_unit_src(stdio_forward_shutdown_idempotent PASS_REGEX "PASS stdio_forward_shutdown_idempotent" TIMEOUT 60)
add_pure_unit_src(stdio_forward_concurrent_push PASS_REGEX "PASS stdio_forward_concurrent_push" TIMEOUT 60)
add_pure_unit_src(stdio_forward_stderr PASS_REGEX "PASS stdio_forward_stderr" TIMEOUT 60)
add_pure_unit_src(stdio_forward_fflush_no_deadlock PASS_REGEX "PASS stdio_forward_fflush_no_deadlock" TIMEOUT 60)
add_pure_unit_src(stdio_forward_multi_source PASS_REGEX "PASS stdio_forward_multi_source" TIMEOUT 60)
# T203 EXPOSES B097: expected to FAIL/crash under sanitizer (documents runtime bug, do not mask)
add_pure_unit_src(config_routing_table_overflow TIMEOUT 60)
# T204 EXPOSES B098: expected to FAIL/crash under sanitizer (documents runtime bug, do not mask)
add_pure_unit_src(config_lsf_single_host_oob TIMEOUT 60)
# T205 EXPOSES B100: expected to FAIL/crash under sanitizer (documents runtime bug, do not mask)
add_pure_unit_src(config_find_variable TIMEOUT 60)
add_pure_unit_src(config_parse_port_spec PASS_REGEX "PASS config_parse_port_spec" TIMEOUT 60)
add_pure_unit_src(config_count_nodes PASS_REGEX "PASS config_count_nodes" TIMEOUT 60)
add_pure_unit_src(config_get_variables_oob PASS_REGEX "PASS config_get_variables_oob" TIMEOUT 60)
add_pure_unit_src(config_routing_table_ssh_bracket PASS_REGEX "PASS config_routing_table_ssh_bracket" TIMEOUT 60)
add_pure_unit_src(config_routing_table_slurm_hostlist PASS_REGEX "PASS config_routing_table_slurm_hostlist" TIMEOUT 60)
add_pure_unit_src(placement_invariants SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/system/placement.c PASS_REGEX "PASS placement_invariants" TIMEOUT 60)
add_pure_unit_src(config_route_table_size_shift PASS_REGEX "PASS config_route_table_size_shift" TIMEOUT 60)
add_pure_unit_src(config_thread_count_underflow PASS_REGEX "PASS config_thread_count_underflow" TIMEOUT 60)
add_pure_unit_src(config_launcher_env_precedence PASS_REGEX "PASS config_launcher_env_precedence" TIMEOUT 60)
add_pure_unit_src(config_destroy_leak PASS_REGEX "PASS config_destroy_leak" TIMEOUT 60)
# T217: needs_full_build -- compiles only inside an ARTS_USE_CXL build
#   (config.c pulls cxl/deque.h -> rapid API). Built against libarts so the
#   real path links; self-skips (prints SKIP, exit 0) in non-CXL builds.
# Only buildable when the CXL backend (rapid API) is actually configured:
# defining ARTS_USE_CXL forces config.c down the cxl/deque.h path, which needs
# the rapid allocator symbols (GLOBAL_MALLOC_DEV). Guard so non-CXL build dirs
# simply skip the target.
if(ARTS_USE_CXL)
    add_arts_test(config_cxl_alloc_strategy)
    target_compile_definitions(config_cxl_alloc_strategy PRIVATE ARTS_USE_CXL)
    register_pure_unit_test(config_cxl_alloc_strategy TIMEOUT 60)
    set_tests_properties(config_cxl_alloc_strategy PROPERTIES PASS_REGULAR_EXPRESSION "config_cxl_alloc_strategy")
endif()

add_pure_unit_src(runtime_edt_event_layout PASS_REGEX "PASS runtime_edt_event_layout" TIMEOUT 60)
# topology.c #includes <hwloc.h> and calls hwloc_* — the vendored hwloc
# target carries the headers and the archive together; a host hwloc must
# never satisfy either.
add_pure_unit_src(topology_thread_mask SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/system/placement.c LIBS arts::hwloc PASS_REGEX "PASS topology_thread_mask" TIMEOUT 60)
add_pure_unit_src(threads_worker_underflow PASS_REGEX "PASS threads_worker_underflow" TIMEOUT 60)
add_pure_unit_src(signals_formatters PASS_REGEX "PASS signals_formatters" TIMEOUT 60)
# T245 EXPOSES B132: expected to FAIL/crash under sanitizer (documents runtime bug, do not mask)
# T250 EXPOSES B138: expected to FAIL/crash under sanitizer (documents runtime bug, do not mask)
add_pure_unit_src(json_writer SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/counter/json.c TIMEOUT 60)
# T251 EXPOSES B134: expected to FAIL/crash under sanitizer (documents runtime bug, do not mask)
add_pure_unit_src(counter_json_parser SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/counter/json.c TIMEOUT 60)
add_pure_unit_src(edge_vector PASS_REGEX "PASS edge_vector" TIMEOUT 60)
# T261 EXPOSES B140: expected to FAIL/crash under sanitizer (documents runtime bug, do not mask)
add_pure_unit_src(block_dist_query TIMEOUT 60)
# T272 EXPOSES B151: expected to FAIL/crash under sanitizer (documents runtime bug, do not mask)
add_pure_unit_src(csr_free_null TIMEOUT 60)
add_pure_unit_src(gpu_reduction_tree PASS_REGEX "PASS gpu_reduction_tree" TIMEOUT 60)
# T274 EXPOSES B-fit-mask-and: expected to FAIL/crash under sanitizer (documents runtime bug, do not mask)
add_pure_unit_src(gpu_fit_schemes TIMEOUT 60)
# T275 EXPOSES B-all-or-nothing: expected to FAIL/crash under sanitizer (documents runtime bug, do not mask)
add_pure_unit_src(gpu_locality_schemes TIMEOUT 60)

# ============================================================================
# Wave C/D runtime + config_specific tests
#
# Generated from research/test-census/waveCD-result.json (clusters C02r..C27, C12r..C18r,
# C19r/C21r/C22r/C23r/C24r/C25r) plus the C13 EDT cluster (scanned from tests/ocr/
# directly — its census metadata was lost when the authoring agent died).
#
# Harness mapping:
#   runtime_single / config_specific -> register_single_node_test
#   runtime_multinode                -> register_multinode_test (_2n/_3n/_4n/_2n_io)
#   "both" notes                     -> single AND multinode
# Protocol/placement selection is supplied GLOBALLY by the build dir's coherence
# configuration; the test bodies self-skip (print SKIP, exit 0) in non-target
# configs, so config_specific PASS regexes accept the SKIP line too.
#
# Bug-exposing tests (exposes_runtime_bug=true with empty pass_regex) are
# registered WITHOUT a PASS_REGULAR_EXPRESSION so they visibly FAIL/hang in
# their target config. They are NOT masked with WILL_FAIL. Do not weaken.
# ============================================================================

# --- C12: DB lifecycle & acquire/release accounting ---
add_arts_test(db_alias_dedup)
register_single_node_test(db_alias_dedup TIMEOUT 30)
set_tests_properties(db_alias_dedup PROPERTIES PASS_REGULAR_EXPRESSION "PASS: db_alias_dedup|SKIP db_alias_dedup")

add_arts_test(db_no_acquire)
register_single_node_test(db_no_acquire TIMEOUT 30)
set_tests_properties(db_no_acquire PROPERTIES PASS_REGULAR_EXPRESSION "PASS: db_no_acquire|SKIP db_no_acquire")

# config_specific: EXCL-only real body, self-skips elsewhere
add_arts_test(db_excl_creator_skip_hold)
register_single_node_test(db_excl_creator_skip_hold TIMEOUT 30)
set_tests_properties(db_excl_creator_skip_hold PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS: db_excl_creator_skip_hold|SKIP db_excl_creator_skip_hold")

add_arts_test(db_copy_to_new_type_race)
register_single_node_test(db_copy_to_new_type_race TIMEOUT 30)
set_tests_properties(db_copy_to_new_type_race PROPERTIES PASS_REGULAR_EXPRESSION "PASS: db_copy_to_new_type_race|SKIP db_copy_to_new_type_race")

# T121 EXPOSES B-create-install-uaf: expected to FAIL (UAF/underflow crash or hang) in its
# target config (non-EXCL, >=3 ranks); self-skips (exit 0) under EXCL or with <3 ranks. Do not mask.
add_arts_test(db_create_install_race)
register_multinode_test(db_create_install_race TIMEOUT 60)

add_arts_test(db_acquire_replay_local)
register_single_node_test(db_acquire_replay_local TIMEOUT 30)
set_tests_properties(db_acquire_replay_local PROPERTIES PASS_REGULAR_EXPRESSION "PASS: db_acquire_replay_local|SKIP db_acquire_replay_local")

# T123 EXPOSES B-release-alias-underflow: expected to FAIL (double-decrement) in its target
# config (ownership/EXCL); self-skips (exit 0) under WRF_VAL. Do not mask.
add_arts_test(db_release_alias_slot)
register_single_node_test(db_release_alias_slot TIMEOUT 30)

# both single + multinode meaningful (remote homes force the GRANT_REQUEST/GRANT double-fire path)
add_arts_test(db_rw_secure_double_fire)
register_single_node_test(db_rw_secure_double_fire TIMEOUT 60)
register_multinode_test(db_rw_secure_double_fire TIMEOUT 60)
set_tests_properties(db_rw_secure_double_fire PROPERTIES PASS_REGULAR_EXPRESSION "PASS: db_rw_secure_double_fire|SKIP db_rw_secure_double_fire")
set_tests_properties(db_rw_secure_double_fire_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS: db_rw_secure_double_fire|SKIP db_rw_secure_double_fire")

add_arts_test(db_acquire_all_bias)
register_multinode_test(db_acquire_all_bias TIMEOUT 60)
set_tests_properties(db_acquire_all_bias_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS: db_acquire_all_bias|SKIP db_acquire_all_bias")

add_arts_test(db_destroy_implicit_release)
register_single_node_test(db_destroy_implicit_release TIMEOUT 30)
set_tests_properties(db_destroy_implicit_release PROPERTIES PASS_REGULAR_EXPRESSION "PASS: db_destroy_implicit_release|SKIP db_destroy_implicit_release")

add_arts_test(db_user_ptr_uaf)
register_single_node_test(db_user_ptr_uaf TIMEOUT 30)
set_tests_properties(db_user_ptr_uaf PROPERTIES PASS_REGULAR_EXPRESSION "PASS: db_user_ptr_uaf|SKIP db_user_ptr_uaf")

# dbcreate_matrix: probe of the DB-create capability matrix --
# (creator-acquisition) x (home-determination) x (locality).  Exercises the
# ARTS core create path directly (arts.h, no OCR shim), so it is a coverage
# regression guard rather than a bug-exposing test: each cell prints its result
# independently, so require only that the scalar appears, not a specific count.
# Cross-runtime capability parity (shim vs the reference runtimes) is compared
# separately through the correctness harness, not here.
add_arts_test(dbcreate_matrix)
register_single_node_test(dbcreate_matrix TIMEOUT 60)
register_multinode_test(dbcreate_matrix TIMEOUT 60)
set_tests_properties(dbcreate_matrix PROPERTIES PASS_REGULAR_EXPRESSION "CELLS_OK=[0-9]+/12")
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(dbcreate_matrix_${_v} PROPERTIES PASS_REGULAR_EXPRESSION "CELLS_OK=[0-9]+/12")
endforeach()

# db_cxl_paths: CXL real body, builds+links against standard libarts via the public-API
# SKIP path in non-CXL builds; the ARTS_USE_CXL define selects the real body.
if(ARTS_USE_CXL)
    add_arts_test(db_cxl_paths)
    target_compile_definitions(db_cxl_paths PRIVATE ARTS_USE_CXL)
    register_single_node_test(db_cxl_paths TIMEOUT 30)
    set_tests_properties(db_cxl_paths PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: db_cxl_paths|SKIP db_cxl_paths")
endif()

# --- C14: EDT context save/restore + created-DB / owned-finish cleanup ---
add_arts_test(ctx_save_restore_nested)
register_single_node_test(ctx_save_restore_nested TIMEOUT 30)
set_tests_properties(ctx_save_restore_nested PROPERTIES PASS_REGULAR_EXPRESSION "PASS ctx_save_restore_nested|SKIP ctx_save_restore_nested")

add_arts_test(ctx_created_db_release_order)
register_single_node_test(ctx_created_db_release_order TIMEOUT 30)
set_tests_properties(ctx_created_db_release_order PROPERTIES PASS_REGULAR_EXPRESSION "PASS ctx_created_db_release_order|SKIP ctx_created_db_release_order")

add_arts_test(ctx_owned_finish_cleanup)
register_single_node_test(ctx_owned_finish_cleanup TIMEOUT 30)
set_tests_properties(ctx_owned_finish_cleanup PROPERTIES PASS_REGULAR_EXPRESSION "PASS ctx_owned_finish_cleanup|SKIP ctx_owned_finish_cleanup")

add_arts_test(ctx_null_guards)
register_single_node_test(ctx_null_guards TIMEOUT 30)
set_tests_properties(ctx_null_guards PROPERTIES PASS_REGULAR_EXPRESSION "PASS ctx_null_guards|SKIP ctx_null_guards")

add_arts_test(ctx_crossrank_proxy_finish)
register_multinode_test(ctx_crossrank_proxy_finish TIMEOUT 60)
set_tests_properties(ctx_crossrank_proxy_finish_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS ctx_crossrank_proxy_finish|SKIP ctx_crossrank_proxy_finish")

# ctx_gpu_lib_edt: GPU-only real body (ARTS_TEST_GPU); SKIP stub links CPU libarts otherwise.
if(BUILD_CUDA_LIBRARY)
    add_arts_test(ctx_gpu_lib_edt)
    target_compile_definitions(ctx_gpu_lib_edt PRIVATE ARTS_TEST_GPU=1)
    register_gpu_test(ctx_gpu_lib_edt TIMEOUT 30)
    set_tests_properties(ctx_gpu_lib_edt PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS ctx_gpu_lib_edt|SKIP ctx_gpu_lib_edt")
endif()

add_arts_test(ctx_cleanup_tls_leak)
register_single_node_test(ctx_cleanup_tls_leak TIMEOUT 30)
set_tests_properties(ctx_cleanup_tls_leak PROPERTIES PASS_REGULAR_EXPRESSION "PASS ctx_cleanup_tls_leak|SKIP ctx_cleanup_tls_leak")

# --- C15: event channel / satisfy / destroy / create-collision ---
add_arts_test(event_channel_transient_null)
register_single_node_test(event_channel_transient_null TIMEOUT 60)
set_tests_properties(event_channel_transient_null PROPERTIES PASS_REGULAR_EXPRESSION "event_channel_transient_null:.*PASS|SKIP event_channel_transient_null")

add_arts_test(event_channel_many_producers)
register_single_node_test(event_channel_many_producers TIMEOUT 60)
set_tests_properties(event_channel_many_producers PROPERTIES PASS_REGULAR_EXPRESSION "event_channel_many_producers:.*PASS|SKIP event_channel_many_producers")

add_arts_test(event_satisfy_adddep_window)
register_single_node_test(event_satisfy_adddep_window TIMEOUT 60)
set_tests_properties(event_satisfy_adddep_window PROPERTIES PASS_REGULAR_EXPRESSION "event_satisfy_adddep_window:.*PASS|SKIP event_satisfy_adddep_window")


add_arts_test(event_finish_latch_rearm)
register_single_node_test(event_finish_latch_rearm TIMEOUT 60)
set_tests_properties(event_finish_latch_rearm PROPERTIES PASS_REGULAR_EXPRESSION "event_finish_latch_rearm:.*PASS|SKIP event_finish_latch_rearm")

add_arts_test(event_error_paths)
register_single_node_test(event_error_paths TIMEOUT 30)
set_tests_properties(event_error_paths PROPERTIES PASS_REGULAR_EXPRESSION "CHANNEL: only DECR|SKIP event_error_paths")

add_arts_test(event_check_collision)
register_single_node_test(event_check_collision TIMEOUT 30)
set_tests_properties(event_check_collision PROPERTIES PASS_REGULAR_EXPRESSION "event_check_collision:.*PASS|SKIP event_check_collision")

add_arts_test(event_remote_create_race)
register_multinode_test(event_remote_create_race TIMEOUT 90)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(event_remote_create_race_${_v} PROPERTIES
        PASS_REGULAR_EXPRESSION "event_remote_create_race:.*PASS|SKIP: event_remote_create_race")
endforeach()

# both single + multinode (OoO defer is home-rank-local but census asks for multinode too)
# COUNTED delivery + reclamation, and that an undeclared ONCE still lingers.
# Whitebox (reads the route table), single-node: the property is local.
add_arts_test(event_counted_reclaim)
register_single_node_test(event_counted_reclaim TIMEOUT 60)

add_arts_test(event_destroy_before_create)
register_single_node_test(event_destroy_before_create TIMEOUT 30)
register_multinode_test(event_destroy_before_create TIMEOUT 60)
set_tests_properties(event_destroy_before_create PROPERTIES PASS_REGULAR_EXPRESSION "event_destroy_before_create:.*PASS|SKIP event_destroy_before_create")
set_tests_properties(event_destroy_before_create_2n PROPERTIES PASS_REGULAR_EXPRESSION "event_destroy_before_create:.*PASS|SKIP event_destroy_before_create")

# event_gpu_force_defer: GPU-only real body (ARTS_USE_GPU); SKIP stub otherwise.
if(BUILD_CUDA_LIBRARY)
    add_arts_test(event_gpu_force_defer)
    target_compile_definitions(event_gpu_force_defer PRIVATE ARTS_USE_GPU)
    register_gpu_test(event_gpu_force_defer TIMEOUT 30)
    set_tests_properties(event_gpu_force_defer PROPERTIES
        PASS_REGULAR_EXPRESSION "event_gpu_force_defer:.*PASS|SKIP event_gpu_force_defer")
endif()

# --- C07: VAL protocol/placement-specific ---
# All VAL config_specific tests self-skip (no-op main printing SKIP) under non-VAL.
add_arts_test(val_wt_vs_wb_divergence)
register_single_node_test(val_wt_vs_wb_divergence TIMEOUT 60)
register_multinode_test(val_wt_vs_wb_divergence TIMEOUT 60)
set_tests_properties(val_wt_vs_wb_divergence PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_wt_vs_wb_divergence|SKIP val_wt_vs_wb_divergence")
set_tests_properties(val_wt_vs_wb_divergence_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_wt_vs_wb_divergence|SKIP val_wt_vs_wb_divergence")

add_arts_test(val_writer_count_nonneg)
register_single_node_test(val_writer_count_nonneg TIMEOUT 120)
register_multinode_test(val_writer_count_nonneg TIMEOUT 120)
set_tests_properties(val_writer_count_nonneg PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_writer_count_nonneg|SKIP val_writer_count_nonneg")
set_tests_properties(val_writer_count_nonneg_2n_io PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_writer_count_nonneg|SKIP val_writer_count_nonneg")

add_arts_test(val_publish_before_decrement)
register_single_node_test(val_publish_before_decrement TIMEOUT 120)
register_multinode_test(val_publish_before_decrement TIMEOUT 120)
set_tests_properties(val_publish_before_decrement PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_publish_before_decrement|SKIP val_publish_before_decrement")
set_tests_properties(val_publish_before_decrement_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_publish_before_decrement|SKIP val_publish_before_decrement")

add_arts_test(val_wt_publish_ack)
register_single_node_test(val_wt_publish_ack TIMEOUT 120)
register_multinode_test(val_wt_publish_ack TIMEOUT 120)
set_tests_properties(val_wt_publish_ack PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_wt_publish_ack|SKIP val_wt_publish_ack")
set_tests_properties(val_wt_publish_ack_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_wt_publish_ack|SKIP val_wt_publish_ack")

add_arts_test(val_wb_confirm_ack_round)
register_single_node_test(val_wb_confirm_ack_round TIMEOUT 120)
register_multinode_test(val_wb_confirm_ack_round TIMEOUT 120)
set_tests_properties(val_wb_confirm_ack_round PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_wb_confirm_ack_round|SKIP val_wb_confirm_ack_round")
set_tests_properties(val_wb_confirm_ack_round_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_wb_confirm_ack_round|SKIP val_wb_confirm_ack_round")

add_arts_test(val_rw_drain_count)
register_single_node_test(val_rw_drain_count TIMEOUT 120)
register_multinode_test(val_rw_drain_count TIMEOUT 120)
set_tests_properties(val_rw_drain_count PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_rw_drain_count|SKIP val_rw_drain_count")
set_tests_properties(val_rw_drain_count_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_rw_drain_count|SKIP val_rw_drain_count")

add_arts_test(val_snapshot_request)
register_single_node_test(val_snapshot_request TIMEOUT 120)
register_multinode_test(val_snapshot_request TIMEOUT 120)
set_tests_properties(val_snapshot_request PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_snapshot_request|SKIP val_snapshot_request")
set_tests_properties(val_snapshot_request_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_snapshot_request|SKIP val_snapshot_request")

add_arts_test(val_baton_invalidate_recheck)
register_single_node_test(val_baton_invalidate_recheck TIMEOUT 120)
register_multinode_test(val_baton_invalidate_recheck TIMEOUT 120)
set_tests_properties(val_baton_invalidate_recheck PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_baton_invalidate_recheck|SKIP val_baton_invalidate_recheck")
set_tests_properties(val_baton_invalidate_recheck_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_baton_invalidate_recheck|SKIP val_baton_invalidate_recheck")

# grant_sticky: asserts the grant outlives its writers (single-rank only —
# a second rank could revoke it, which is what the test must exclude).
add_arts_test(grant_sticky)
register_single_node_test(grant_sticky TIMEOUT 60)
set_tests_properties(grant_sticky PROPERTIES PASS_REGULAR_EXPRESSION "PASS grant_sticky|SKIP grant_sticky")

# grant_ex_holder_sharer: after a grant moves, the ex-holder must be retired
# by the new owner's rounds.  2+ ranks (the grant has to leave the reader).
add_arts_test(grant_ex_holder_sharer)
register_multinode_test(grant_ex_holder_sharer TIMEOUT 120)
set_tests_properties(grant_ex_holder_sharer_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS grant_ex_holder_sharer|SKIP grant_ex_holder_sharer")
set_tests_properties(grant_ex_holder_sharer_3n PROPERTIES PASS_REGULAR_EXPRESSION "PASS grant_ex_holder_sharer|SKIP grant_ex_holder_sharer")
set_tests_properties(grant_ex_holder_sharer_4n PROPERTIES PASS_REGULAR_EXPRESSION "PASS grant_ex_holder_sharer|SKIP grant_ex_holder_sharer")
set_tests_properties(grant_ex_holder_sharer_2n_io PROPERTIES PASS_REGULAR_EXPRESSION "PASS grant_ex_holder_sharer|SKIP grant_ex_holder_sharer")

add_arts_test(val_no_acquire)
register_single_node_test(val_no_acquire TIMEOUT 120)
register_multinode_test(val_no_acquire TIMEOUT 120)
set_tests_properties(val_no_acquire PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_no_acquire|SKIP val_no_acquire")
set_tests_properties(val_no_acquire_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS val_no_acquire|SKIP val_no_acquire")

# --- C09: EXCL protocol-specific ---
# T093 EXPOSES B-lock-samerank-rw-grant-loss: expected to FAIL (hang) under EXCL; self-skips
# elsewhere. Do not mask.
add_arts_test(excl_samerank_rw_grant_race)
register_single_node_test(excl_samerank_rw_grant_race TIMEOUT 60)
register_multinode_test(excl_samerank_rw_grant_race TIMEOUT 60)

# T094 EXPOSES B-lock-multiconsumer-queue: expected to FAIL (lost GRANT / stranded writer)
# under EXCL; valid RW-churn correctness check (no skip) elsewhere. Do not mask.
add_arts_test(excl_multiconsumer_queue)
register_single_node_test(excl_multiconsumer_queue TIMEOUT 60)
register_multinode_test(excl_multiconsumer_queue TIMEOUT 60)

add_arts_test(excl_request_coalesce)
register_single_node_test(excl_request_coalesce TIMEOUT 60)
set_tests_properties(excl_request_coalesce PROPERTIES PASS_REGULAR_EXPRESSION "excl_request_coalesce: .* — PASS|SKIP excl_request_coalesce")

# T096 EXPOSES B-lock-destroy-during-acquire: expected to FAIL (unbalanced counter / stranded
# waiter) under EXCL; self-skips elsewhere. Do not mask.
add_arts_test(excl_destroy_during_acquire)
register_single_node_test(excl_destroy_during_acquire TIMEOUT 60)

add_arts_test(excl_purge_grant_d6_d7)
register_single_node_test(excl_purge_grant_d6_d7 TIMEOUT 60)
register_multinode_test(excl_purge_grant_d6_d7 TIMEOUT 60)
set_tests_properties(excl_purge_grant_d6_d7 PROPERTIES PASS_REGULAR_EXPRESSION "excl_purge_grant_d6_d7 D7: .* — PASS|SKIP excl_purge_grant_d6_d7")
set_tests_properties(excl_purge_grant_d6_d7_2n PROPERTIES PASS_REGULAR_EXPRESSION "excl_purge_grant_d6_d7 D7: .* — PASS|SKIP excl_purge_grant_d6_d7")

# T098 EXPOSES B-lock-grant-destroyed: expected to FAIL (dropped GRANT / stall) under EXCL at

# excl_req_before_create: protocol-agnostic, needs 3+ ranks (verbatim copy of the old
# coherence_lock_req_before_create into the planned filename); register at 3n/4n.
add_arts_test(excl_req_before_create)
register_multinode_test(excl_req_before_create TIMEOUT 90 VARIANTS 3n 4n)
set_tests_properties(excl_req_before_create_3n PROPERTIES PASS_REGULAR_EXPRESSION "PASS: [0-9]+ iterations completed|SKIP excl_req_before_create")
set_tests_properties(excl_req_before_create_4n PROPERTIES PASS_REGULAR_EXPRESSION "PASS: [0-9]+ iterations completed|SKIP excl_req_before_create")

# excl_mode_mismatch_fatal: standalone driver paired by run_mode_mismatch.sh — NOT a plain
# ctest (like coherence_mode_mismatch). Build the binary only; no add_test registration.
add_arts_test(excl_mode_mismatch_fatal)

# --- C10: WRF_VAL protocol-specific ---
# wrf_val_is_serialized: pure_unit, WRF_VAL-only (self-skips else); links the real per-config libarts
# symbol, so it uses add_arts_test (not add_pure_unit_src) + register_pure_unit_test.
add_arts_test(wrf_val_is_serialized)
register_pure_unit_test(wrf_val_is_serialized TIMEOUT 30)
set_tests_properties(wrf_val_is_serialized PROPERTIES PASS_REGULAR_EXPRESSION "PASS wrf_val_is_serialized|SKIP wrf_val_is_serialized")

# T102 EXPOSES B-wrf_val-writer-count-leak: expected to FAIL (stranded parked acquire) under WRF_VAL

# T103 EXPOSES update_cached_version_max watermark hazard (latent): see status_note. WRF_VAL, >=2 ranks.
add_arts_test(wrf_val_getdata_dedup)
register_multinode_test(wrf_val_getdata_dedup TIMEOUT 120)

# T104 EXPOSES in-place buf->version-vs-install hazard. WRF_VAL, >=2 ranks.
add_arts_test(wrf_val_home_vs_nonhome_writer)
register_multinode_test(wrf_val_home_vs_nonhome_writer TIMEOUT 120)

# T105 EXPOSES snapshot_request master==NULL/version=0 hazard. WRF_VAL, >=2 ranks.
add_arts_test(wrf_val_sentinel_snapshot)
register_multinode_test(wrf_val_sentinel_snapshot TIMEOUT 120)


add_arts_test(db_wrf_promote_manual)
register_multinode_test(db_wrf_promote_manual TIMEOUT 120)
set_tests_properties(db_wrf_promote_manual_2n PROPERTIES PASS_REGULAR_EXPRESSION "db_wrf_promote_manual: PASS|SKIP db_wrf_promote_manual")

# --- C11: snapshot / publish-ack / destroy-notify / dispatcher-parity (config_specific,
# but census asks for multinode variants to expose the wire reorder; register both) ---
# T108 EXPOSES B014/B028. Non-EXCL (snapshot-bearing); self-skips under EXCL.
add_arts_test(snapshot_response_3case)
register_single_node_test(snapshot_response_3case TIMEOUT 120)
register_multinode_test(snapshot_response_3case TIMEOUT 120)

# T109 EXPOSES B017/B018. WT + WRF_VAL; self-skips under WB/EXCL.
add_arts_test(publish_ack_post_on_miss)
register_single_node_test(publish_ack_post_on_miss TIMEOUT 120)
register_multinode_test(publish_ack_post_on_miss TIMEOUT 120)

# T110 EXPOSES B018 (cv-guarded releaser wake). EXCL only; self-skips elsewhere.
add_arts_test(excl_release_wake_on_miss)
register_single_node_test(excl_release_wake_on_miss TIMEOUT 120)
register_multinode_test(excl_release_wake_on_miss TIMEOUT 120)


# T112 EXPOSES B023 (self-send vs dispatcher parity). All protocols; needs 1n AND multinode.
add_arts_test(self_send_vs_dispatcher_parity)
register_single_node_test(self_send_vs_dispatcher_parity TIMEOUT 120)
register_multinode_test(self_send_vs_dispatcher_parity TIMEOUT 120)

# T113 EXPOSES B015/B016 (create coalesce). runtime_multinode (needs >=2 ranks).
add_arts_test(create_coalesce)
register_multinode_test(create_coalesce TIMEOUT 120)

# T114 EXPOSES B019 (destroy-before-create defer). config_specific (non-EXCL), needs >=2 ranks.
add_arts_test(destroy_before_create_defer)
register_multinode_test(destroy_before_create_defer TIMEOUT 120)

# T115 EXPOSES B024 (Cat-C ref balance) — normally PASSES (regression guard). 1n + multinode.
add_arts_test(cat_c_ref_balance)
register_single_node_test(cat_c_ref_balance TIMEOUT 120)
register_multinode_test(cat_c_ref_balance TIMEOUT 120)
set_tests_properties(cat_c_ref_balance PROPERTIES PASS_REGULAR_EXPRESSION "PASS: cat_c_ref_balance|SKIP cat_c_ref_balance")
set_tests_properties(cat_c_ref_balance_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS: cat_c_ref_balance|SKIP cat_c_ref_balance")

# T116 EXPOSES B017 (await_publish_ack under shutdown). WT + WRF_VAL; self-skips under WB/EXCL.
# Normally PASSES (prints token before shutdown). 1n + multinode.
add_arts_test(await_publish_ack_shutdown)
register_single_node_test(await_publish_ack_shutdown TIMEOUT 120)
register_multinode_test(await_publish_ack_shutdown TIMEOUT 120)
set_tests_properties(await_publish_ack_shutdown PROPERTIES PASS_REGULAR_EXPRESSION "PASS: await_publish_ack_shutdown reached shutdown|SKIP await_publish_ack_shutdown")
set_tests_properties(await_publish_ack_shutdown_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS: await_publish_ack_shutdown reached shutdown|SKIP await_publish_ack_shutdown")

# --- C03r: route-table / GUID determinism ---

add_arts_test(guid_crossrank_determinism)
register_multinode_test(guid_crossrank_determinism TIMEOUT 120)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(guid_crossrank_determinism_${_v} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: rank .* recovered identical|SKIP: guid_crossrank_determinism")
endforeach()

add_arts_test(guid_keygen_table_coupling)
register_multinode_test(guid_keygen_table_coupling TIMEOUT 120)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(guid_keygen_table_coupling_${_v} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: rank .* all .* reserved GUIDs unique|SKIP guid_keygen_table_coupling_${_v}")
endforeach()

add_arts_test(round_robin_home_distribution)
register_multinode_test(round_robin_home_distribution TIMEOUT 120)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(round_robin_home_distribution_${_v} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: rank .* home == idx|SKIP: round_robin_home_distribution")
endforeach()

add_arts_test(guid_index_from_mismatch)
register_single_node_test(guid_index_from_mismatch TIMEOUT 60)
set_tests_properties(guid_index_from_mismatch PROPERTIES PASS_REGULAR_EXPRESSION "guid_index_from_mismatch: ALL PASSED|SKIP guid_index_from_mismatch")

# --- C04r: OoO defer/replay engine ---
add_arts_test(ooo_concurrent_multidrain)
register_single_node_test(ooo_concurrent_multidrain TIMEOUT 120)
set_tests_properties(ooo_concurrent_multidrain PROPERTIES PASS_REGULAR_EXPRESSION "PASS: ooo_concurrent_multidrain|SKIP ooo_concurrent_multidrain")

add_arts_test(ooo_force_push_drain)
register_single_node_test(ooo_force_push_drain TIMEOUT 120)
set_tests_properties(ooo_force_push_drain PROPERTIES PASS_REGULAR_EXPRESSION "PASS: ooo_force_push_drain deferred dep replayed|SKIP ooo_force_push_drain")

add_arts_test(ooo_hit_ref_pin)
register_single_node_test(ooo_hit_ref_pin TIMEOUT 120)
set_tests_properties(ooo_hit_ref_pin PROPERTIES PASS_REGULAR_EXPRESSION "PASS: ooo_hit_ref_pin|SKIP ooo_hit_ref_pin")

# ooo_gen_crossgen_drop is NOT registered.  It drives destroy + re-create of one
# labeled GUID and asserts the new generation is not corrupted by an operation
# still in flight for the old one.  ARTS does not implement that: a create onto
# an occupied slot REPLACES, which is the OCR standard's unchecked default (the
# standard calls the outcome undefined), and the runtime carries no generation
# stamp that would let a late message tell the two apart.  The source is kept
# under tests/ocr/ as the executable statement of the limitation — see
# docs/programming_model/guids.rst — so the day the semantics are implemented
# the test is one line away from running again.

# --- C05r: DB create-with-data / coherence install monotone / snapshot drain / stub install ---
add_arts_test(db_create_with_data_bytes)
register_single_node_test(db_create_with_data_bytes TIMEOUT 60)
set_tests_properties(db_create_with_data_bytes PROPERTIES PASS_REGULAR_EXPRESSION "PASS: db_create_with_data_bytes|SKIP db_create_with_data_bytes")

add_arts_test(coherence_install_version_monotone)
register_single_node_test(coherence_install_version_monotone TIMEOUT 120)
register_multinode_test(coherence_install_version_monotone TIMEOUT 120)
set_tests_properties(coherence_install_version_monotone PROPERTIES PASS_REGULAR_EXPRESSION "PASS: coherence_install_version_monotone|SKIP coherence_install_version_monotone")
set_tests_properties(coherence_install_version_monotone_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS: coherence_install_version_monotone|SKIP coherence_install_version_monotone")

# T059 EXPOSES B-mark-double-dec: expected to FAIL (delta==2 not 1). runtime_single (boots
# arts_rt for route-table/EDT alloc). Lives in tests/unit/. Do not mask.
add_arts_test(mark_edt_ready_idempotent)
register_single_node_test(mark_edt_ready_idempotent TIMEOUT 60)

add_arts_test(snapshot_drain_case3)
register_multinode_test(snapshot_drain_case3 TIMEOUT 180)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(snapshot_drain_case3_${_v} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: snapshot_drain_case3|SKIP snapshot_drain_case3")
endforeach()

add_arts_test(stub_install_winner)
register_multinode_test(stub_install_winner TIMEOUT 180)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(stub_install_winner_${_v} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: stub_install_winner|SKIP stub_install_winner")
endforeach()


# --- C02r: scheduler ---
# T024 EXPOSES suspected non-owner-push deque[0] race — normally PASSES (guard). runtime_single.
add_arts_test(scheduler_async_nonowner_push)
register_single_node_test(scheduler_async_nonowner_push TIMEOUT 120)
set_tests_properties(scheduler_async_nonowner_push PROPERTIES PASS_REGULAR_EXPRESSION "scheduler_async_nonowner_push: .* each ran once — PASS|SKIP scheduler_async_nonowner_push")

add_arts_test(scheduler_acquire_bias)
register_single_node_test(scheduler_acquire_bias TIMEOUT 120)
set_tests_properties(scheduler_acquire_bias PROPERTIES PASS_REGULAR_EXPRESSION "scheduler_acquire_bias: .* — PASS|SKIP scheduler_acquire_bias")

add_arts_test(scheduler_loop_variants)
register_single_node_test(scheduler_loop_variants TIMEOUT 60)
set_tests_properties(scheduler_loop_variants PROPERTIES PASS_REGULAR_EXPRESSION "scheduler_loop_variants: .* — PASS|SKIP scheduler_loop_variants")

# --- C16r: transport socket (bootstrap-only after the fabric cutover) ---
add_arts_test(socket_connect_retry_abort)
register_multinode_test(socket_connect_retry_abort TIMEOUT 120)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(socket_connect_retry_abort_${_v} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: socket_connect_retry_abort connected|SKIP socket_connect_retry_abort_${_v}")
endforeach()

# T172 EXPOSES B-remote-alive-nonatomic — normally PASSES. runtime_multinode.
add_arts_test(socket_remote_alive_owner)
register_multinode_test(socket_remote_alive_owner TIMEOUT 120)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(socket_remote_alive_owner_${_v} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: socket_remote_alive_owner|SKIP socket_remote_alive_owner_${_v}")
endforeach()

# --- C18r: dispatcher ---
# dispatcher_mode_mismatch_matrix: standalone pairing driver (run_mode_mismatch.sh) — NOT a
# plain ctest. Build the binary only; no add_test registration.
add_arts_test(dispatcher_mode_mismatch_matrix)

add_arts_test(dispatcher_miss_drop_refpin)
register_multinode_test(dispatcher_miss_drop_refpin TIMEOUT 120)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(dispatcher_miss_drop_refpin_${_v} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: dispatcher_miss_drop_refpin|SKIP dispatcher_miss_drop_refpin_${_v}")
endforeach()

# dispatcher_redirect_miss_reflect: WB-only; self-skips elsewhere. multinode.
add_arts_test(dispatcher_redirect_miss_reflect)
register_multinode_test(dispatcher_redirect_miss_reflect TIMEOUT 120)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(dispatcher_redirect_miss_reflect_${_v} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS: dispatcher_redirect_miss_reflect|SKIP dispatcher_redirect_miss_reflect")
endforeach()

add_arts_test(dispatcher_default_fatal)
register_single_node_test(dispatcher_default_fatal TIMEOUT 60)
set_tests_properties(dispatcher_default_fatal PROPERTIES PASS_REGULAR_EXPRESSION "PASS: dispatcher_default_fatal|SKIP dispatcher_default_fatal")

# --- C19r: launcher ---
add_arts_test(launcher_no_respawn)
register_multinode_test(launcher_no_respawn TIMEOUT 60)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(launcher_no_respawn_${_v} PROPERTIES PASS_REGULAR_EXPRESSION "PASS: launcher_no_respawn|SKIP launcher_no_respawn_${_v}")
endforeach()

add_arts_test(launcher_orphan_prevention)
register_multinode_test(launcher_orphan_prevention TIMEOUT 60)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(launcher_orphan_prevention_${_v} PROPERTIES PASS_REGULAR_EXPRESSION "PASS: launcher_orphan_prevention|SKIP launcher_orphan_prevention_${_v}")
endforeach()

add_arts_test(launcher_partial_spawn_cleanup)
register_multinode_test(launcher_partial_spawn_cleanup TIMEOUT 60)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(launcher_partial_spawn_cleanup_${_v} PROPERTIES PASS_REGULAR_EXPRESSION "PASS: launcher_partial_spawn_cleanup|SKIP launcher_partial_spawn_cleanup_${_v}")
endforeach()

add_arts_test(launcher_cleanup_escalation)
register_multinode_test(launcher_cleanup_escalation TIMEOUT 60)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(launcher_cleanup_escalation_${_v} PROPERTIES PASS_REGULAR_EXPRESSION "PASS: launcher_cleanup_escalation|SKIP launcher_cleanup_escalation_${_v}")
endforeach()

# launcher_kill_mode: SSH-only (ARTS_TEST_SSH_KILL_MODE). Default CI build is the SKIP stub.
add_arts_test(launcher_kill_mode)
register_single_node_test(launcher_kill_mode TIMEOUT 10)
set_tests_properties(launcher_kill_mode PROPERTIES PASS_REGULAR_EXPRESSION "SKIP launcher_kill_mode")

# --- C21r: config loading ---
add_arts_test(config_load_multinode_port_offset)
register_multinode_test(config_load_multinode_port_offset TIMEOUT 60)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(config_load_multinode_port_offset_${_v} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS config_load_multinode_port_offset|SKIP config_load_multinode_port_offset_${_v}")
endforeach()

add_arts_test(config_load_single_reclaim)
register_single_node_test(config_load_single_reclaim TIMEOUT 30)
set_tests_properties(config_load_single_reclaim PROPERTIES PASS_REGULAR_EXPRESSION "PASS config_load_single_reclaim|SKIP config_load_single_reclaim")

add_arts_test(config_port_count_agreement)
register_multinode_test(config_port_count_agreement TIMEOUT 60 VARIANTS 2n)
set_tests_properties(config_port_count_agreement_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS config_port_count_agreement|SKIP config_port_count_agreement")

# config_override_embedded: pure_unit (links libarts, starts no runtime).
add_arts_test(config_override_embedded)
register_pure_unit_test(config_override_embedded TIMEOUT 30)
set_tests_properties(config_override_embedded PROPERTIES PASS_REGULAR_EXPRESSION "PASS config_override_embedded|SKIP config_override_embedded")

# config_removed_keys_reject: pure_unit (links libarts, starts no runtime;
# forks to probe the sender_threads/receiver_threads ARTS_ERROR death paths).
add_arts_test(config_removed_keys_reject)
register_pure_unit_test(config_removed_keys_reject TIMEOUT 30)
set_tests_properties(config_removed_keys_reject PROPERTIES PASS_REGULAR_EXPRESSION "PASS config_removed_keys_reject|SKIP config_removed_keys_reject")

# config_provider_regpool_parse: pure_unit (links libarts, starts no runtime).
add_arts_test(config_provider_regpool_parse)
register_pure_unit_test(config_provider_regpool_parse TIMEOUT 30)
set_tests_properties(config_provider_regpool_parse PROPERTIES PASS_REGULAR_EXPRESSION "PASS config_provider_regpool_parse|SKIP config_provider_regpool_parse")

# --- C22r: runtime startup/shutdown/threads/signals ---
add_arts_test(runtime_barrier_counts)
register_single_node_test(runtime_barrier_counts TIMEOUT 60)
set_tests_properties(runtime_barrier_counts PROPERTIES PASS_REGULAR_EXPRESSION "PASS runtime_barrier_counts|SKIP runtime_barrier_counts")

# T221 EXPOSES B-shutdown-state-dualpath — normally green baseline (documented, not deterministic).
add_arts_test(runtime_concurrent_shutdown)
register_single_node_test(runtime_concurrent_shutdown TIMEOUT 60)
set_tests_properties(runtime_concurrent_shutdown PROPERTIES PASS_REGULAR_EXPRESSION "SHUTDOWN_STATE_OK|SKIP runtime_concurrent_shutdown")

add_arts_test(runtime_stop_by_role_spin)
register_single_node_test(runtime_stop_by_role_spin TIMEOUT 60)
set_tests_properties(runtime_stop_by_role_spin PROPERTIES PASS_REGULAR_EXPRESSION "STOP_BY_ROLE_DONE|SKIP runtime_stop_by_role_spin")

# T223 EXPOSES B-local-spin-ordering — TSan-only; on non-TSan a clean smoke (has PASS token).
add_arts_test(runtime_local_spin_publish)
register_single_node_test(runtime_local_spin_publish TIMEOUT 60)
set_tests_properties(runtime_local_spin_publish PROPERTIES PASS_REGULAR_EXPRESSION "LOCAL_SPIN_PUBLISH_DONE|SKIP runtime_local_spin_publish")

add_arts_test(runtime_teardown_leak)
register_single_node_test(runtime_teardown_leak TIMEOUT 60)
set_tests_properties(runtime_teardown_leak PROPERTIES PASS_REGULAR_EXPRESSION "TEARDOWN_LEAK_DONE|SKIP runtime_teardown_leak")

# runtime_gpu_scheduler_promotion: GPU-only (ARTS_USE_GPU); SKIP stub otherwise.
if(BUILD_CUDA_LIBRARY)
    add_arts_test(runtime_gpu_scheduler_promotion)
    target_compile_definitions(runtime_gpu_scheduler_promotion PRIVATE ARTS_USE_GPU)
    register_gpu_test(runtime_gpu_scheduler_promotion TIMEOUT 120)
    set_tests_properties(runtime_gpu_scheduler_promotion PROPERTIES
        PASS_REGULAR_EXPRESSION "GPU_SCHED_PROMOTION_DONE|SKIP runtime_gpu_scheduler_promotion")
endif()

# runtime_cxl_arena: CXL-only (ARTS_USE_CXL); SKIP stub otherwise.
if(ARTS_USE_CXL)
    add_arts_test(runtime_cxl_arena)
    target_compile_definitions(runtime_cxl_arena PRIVATE ARTS_USE_CXL)
    register_single_node_test(runtime_cxl_arena TIMEOUT 120)
    set_tests_properties(runtime_cxl_arena PROPERTIES PASS_REGULAR_EXPRESSION "PASS runtime_cxl_arena|SKIP runtime_cxl_arena")
endif()

add_arts_test(threads_shutdown_cancel)
register_single_node_test(threads_shutdown_cancel TIMEOUT 60)
set_tests_properties(threads_shutdown_cancel PROPERTIES PASS_REGULAR_EXPRESSION "SHUTDOWN_CANCEL_DONE|SKIP threads_shutdown_cancel")

# T231 EXPOSES B-counter-save-publish — TSan-only; on non-TSan a clean smoke (has PASS token).
add_arts_test(threads_counter_save_publish)
register_single_node_test(threads_counter_save_publish TIMEOUT 60)
set_tests_properties(threads_counter_save_publish PROPERTIES PASS_REGULAR_EXPRESSION "COUNTER_SAVE_PUBLISH_DONE|SKIP threads_counter_save_publish")

add_arts_test(threads_pin_affinity)
register_single_node_test(threads_pin_affinity TIMEOUT 60)
set_tests_properties(threads_pin_affinity PROPERTIES PASS_REGULAR_EXPRESSION "PASS threads_pin_affinity|SKIP threads_pin_affinity")

add_arts_test(signals_sigterm_graceful)
register_single_node_test(signals_sigterm_graceful TIMEOUT 60)
set_tests_properties(signals_sigterm_graceful PROPERTIES PASS_REGULAR_EXPRESSION "SIGTERM_GRACEFUL_OK|SKIP signals_sigterm_graceful")

# signals_double_sigterm: success criterion is exit code 143 (= 128+SIGTERM), not a stdout
# token; pass_regex empty per the census. The DOUBLE_SIGTERM_UNEXPECTED_CLEAN_EXIT token marks
# a regression. Exit-code based: WILL_FAIL accepts the non-zero (143) escalation exit; the
# FAIL_REGULAR_EXPRESSION catches a regression that exits 0 after printing the clean token.
add_arts_test(signals_double_sigterm)
register_single_node_test(signals_double_sigterm TIMEOUT 60)
set_tests_properties(signals_double_sigterm PROPERTIES
    WILL_FAIL TRUE
    FAIL_REGULAR_EXPRESSION "DOUBLE_SIGTERM_UNEXPECTED_CLEAN_EXIT")

add_arts_test(signals_concurrent_shutdown_edts)
register_single_node_test(signals_concurrent_shutdown_edts TIMEOUT 60)
set_tests_properties(signals_concurrent_shutdown_edts PROPERTIES PASS_REGULAR_EXPRESSION "CONCURRENT_SHUTDOWN_EDTS_OK|SKIP signals_concurrent_shutdown_edts")

# main_rank_overflow_abort: success is abort with exit 1; exit-code based.  WILL_FAIL accepts
# the non-zero (abort) exit; the FAIL_REGULAR_EXPRESSION catches a regression that returns 0
# after printing RANK_OVERFLOW_NO_ABORT (the guard failed to fire).
add_arts_test(main_rank_overflow_abort)
register_single_node_test(main_rank_overflow_abort TIMEOUT 60)
set_tests_properties(main_rank_overflow_abort PROPERTIES
    WILL_FAIL TRUE
    FAIL_REGULAR_EXPRESSION "RANK_OVERFLOW_NO_ABORT")

add_arts_test(shutdown_peer_broadcast)
register_multinode_test(shutdown_peer_broadcast TIMEOUT 120)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(shutdown_peer_broadcast_${_v} PROPERTIES PASS_REGULAR_EXPRESSION "PEER_BROADCAST_EXIT|SKIP shutdown_peer_broadcast_${_v}")
endforeach()

add_arts_test(shutdown_abort_eof)
register_multinode_test(shutdown_abort_eof TIMEOUT 120)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(shutdown_abort_eof_${_v} PROPERTIES PASS_REGULAR_EXPRESSION "ABORT_EOF_EXIT|SKIP shutdown_abort_eof_${_v}")
endforeach()

add_arts_test(shutdown_idempotent_concurrent_init)
register_multinode_test(shutdown_idempotent_concurrent_init TIMEOUT 120)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(shutdown_idempotent_concurrent_init_${_v} PROPERTIES PASS_REGULAR_EXPRESSION "IDEMPOTENT_INIT_EXIT|SKIP shutdown_idempotent_concurrent_init_${_v}")
endforeach()

# --- C23r: random / system-info ---
# T244 EXPOSES B129 (sign-extended jrand48): expected to FAIL (high-bits set). Do not mask.
add_arts_test(random_thread_safe)
register_single_node_test(random_thread_safe TIMEOUT 60)

add_arts_test(system_info_per_rank)
register_multinode_test(system_info_per_rank TIMEOUT 120)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(system_info_per_rank_${_v} PROPERTIES PASS_REGULAR_EXPRESSION "PASS system_info_per_rank|SKIP system_info_per_rank_${_v}")
endforeach()

add_arts_test(system_info_in_edt)
register_single_node_test(system_info_in_edt TIMEOUT 60)
set_tests_properties(system_info_in_edt PROPERTIES PASS_REGULAR_EXPRESSION "PASS system_info_in_edt|SKIP system_info_in_edt")

# --- C24r: counters ---
# T252 EXPOSES B135 (k-way merge heap-buffer-overflow): registered WITHOUT a PASS regex so an
# ASan abort surfaces as failure. Do not mask.
add_arts_test(counter_kway_merge)
register_single_node_test(counter_kway_merge TIMEOUT 60)

# T253 — cluster-reduce node coverage: the master's cluster.json must reduce
# every node's counter file (asserts NUM_EDT_FINISH >= ranks+1; SUM short of
# that means a node's file was dropped).  Originally targeted B136 via the
# ONCE,CLUSTER,MASTER TIME_TOTAL counter; that counter was retired from the
# counter configs (e2e time is the [E2E] stderr marker now), so the test
# asserts on a counter the stock configs still capture.  The single shared
# binary cannot know which node-config variant invoked it, so the SKIP arm is
# unsuffixed (matching the test's actual output). runtime_multinode.
add_arts_test(counter_master_reduce)
register_multinode_test(counter_master_reduce TIMEOUT 30)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(counter_master_reduce_${_v} PROPERTIES PASS_REGULAR_EXPRESSION "PASS counter_master_reduce|SKIP counter_master_reduce")
endforeach()

# T254 EXPOSES B137 (counter start-field race) — TSan-only; on non-TSan deterministic green.
add_arts_test(counter_periodic_race)
register_single_node_test(counter_periodic_race TIMEOUT 60)
set_tests_properties(counter_periodic_race PROPERTIES PASS_REGULAR_EXPRESSION "PASS counter_periodic_race|SKIP counter_periodic_race")

add_arts_test(counter_capture_epoch_monotonic)
register_single_node_test(counter_capture_epoch_monotonic TIMEOUT 60)
set_tests_properties(counter_capture_epoch_monotonic PROPERTIES PASS_REGULAR_EXPRESSION "PASS counter_capture_epoch_monotonic|SKIP counter_capture_epoch_monotonic")

add_arts_test(counter_smoke_value)
register_single_node_test(counter_smoke_value TIMEOUT 30)
set_tests_properties(counter_smoke_value PROPERTIES PASS_REGULAR_EXPRESSION "PASS counter_smoke_value|SKIP counter_smoke_value")

add_arts_test(counter_timer_balance)
register_single_node_test(counter_timer_balance TIMEOUT 30)
set_tests_properties(counter_timer_balance PROPERTIES PASS_REGULAR_EXPRESSION "PASS counter_timer_balance|SKIP counter_timer_balance")

add_arts_test(counter_time_sync)
register_multinode_test(counter_time_sync TIMEOUT 30)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(counter_time_sync_${_v} PROPERTIES PASS_REGULAR_EXPRESSION "PASS counter_time_sync|SKIP counter_time_sync_${_v}")
endforeach()

add_arts_test(object_counter_full)
register_single_node_test(object_counter_full TIMEOUT 30)
set_tests_properties(object_counter_full PROPERTIES PASS_REGULAR_EXPRESSION "PASS object_counter_full|SKIP object_counter_full")

# --- C25r: block dist / CSR ---
# T262 EXPOSES B-args-oob: expected to FAIL (ASan OOB). runtime_single (needs arts_rt). Do not mask.
add_arts_test(dist_args_oob)
register_single_node_test(dist_args_oob TIMEOUT 10)

add_arts_test(block_dist_roundrobin)
register_multinode_test(block_dist_roundrobin TIMEOUT 60)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(block_dist_roundrobin_${_v} PROPERTIES PASS_REGULAR_EXPRESSION "PASS block_dist_roundrobin|SKIP block_dist_roundrobin_${_v}")
endforeach()

# T264 EXPOSES B-csr-empty-ub: expected to FAIL (NULL deref). Do not mask.
add_arts_test(csr_empty_partition)
register_single_node_test(csr_empty_partition TIMEOUT 10)

add_arts_test(csr_wellformed)
register_single_node_test(csr_wellformed TIMEOUT 10)
set_tests_properties(csr_wellformed PROPERTIES PASS_REGULAR_EXPRESSION "PASS csr_wellformed|SKIP csr_wellformed")

add_arts_test(csr_from_guid_refcount)
register_single_node_test(csr_from_guid_refcount TIMEOUT 10)
set_tests_properties(csr_from_guid_refcount PROPERTIES PASS_REGULAR_EXPRESSION "PASS csr_from_guid_refcount|SKIP csr_from_guid_refcount")

add_arts_test(csr_get_neighbors)
register_single_node_test(csr_get_neighbors TIMEOUT 10)
set_tests_properties(csr_get_neighbors PROPERTIES PASS_REGULAR_EXPRESSION "PASS csr_get_neighbors|SKIP csr_get_neighbors")

# T268 EXPOSES B-csr-leak: expected to FAIL (LSan leak). Do not mask.
add_arts_test(csr_load_edgelist)
register_single_node_test(csr_load_edgelist TIMEOUT 10)

# T269 EXPOSES B-csr-leak + B-csr-token-zero: expected to FAIL (LSan leak). Do not mask.
add_arts_test(csr_load_csr_format)
register_single_node_test(csr_load_csr_format TIMEOUT 10)

# T270 EXPOSES B-args-oob (+ B-csr-leak): expected to FAIL (ASan OOB). Do not mask.
add_arts_test(csr_load_args)
register_single_node_test(csr_load_args TIMEOUT 10)

add_arts_test(csr_remote_null)
register_multinode_test(csr_remote_null TIMEOUT 60)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(csr_remote_null_${_v} PROPERTIES PASS_REGULAR_EXPRESSION "PASS csr_remote_null|SKIP csr_remote_null_${_v}")
endforeach()

# --- C27: public-API surface ---
add_arts_test(api_db_check_rendezvous)
register_single_node_test(api_db_check_rendezvous TIMEOUT 10)
set_tests_properties(api_db_check_rendezvous PROPERTIES PASS_REGULAR_EXPRESSION "PASS api_db_check_rendezvous|SKIP api_db_check_rendezvous")

add_arts_test(api_init_per_node_worker)
register_single_node_test(api_init_per_node_worker TIMEOUT 10)
set_tests_properties(api_init_per_node_worker PROPERTIES PASS_REGULAR_EXPRESSION "PASS api_init_per_node_worker|SKIP api_init_per_node_worker")

add_arts_test(api_current_finish_event_null)
register_single_node_test(api_current_finish_event_null TIMEOUT 10)
set_tests_properties(api_current_finish_event_null PROPERTIES PASS_REGULAR_EXPRESSION "PASS api_current_finish_event_null|SKIP api_current_finish_event_null")

add_arts_test(api_latch_incr_slot)
register_single_node_test(api_latch_incr_slot TIMEOUT 10)
set_tests_properties(api_latch_incr_slot PROPERTIES PASS_REGULAR_EXPRESSION "PASS api_latch_incr_slot|SKIP api_latch_incr_slot")

# api_gpu_context_accessors: GPU-only real body (ARTS_TEST_GPU); SKIP stub otherwise.
if(BUILD_CUDA_LIBRARY)
    add_arts_test(api_gpu_context_accessors)
    target_compile_definitions(api_gpu_context_accessors PRIVATE ARTS_TEST_GPU=1)
    register_gpu_test(api_gpu_context_accessors TIMEOUT 10)
    set_tests_properties(api_gpu_context_accessors PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS api_gpu_context_accessors|SKIP api_gpu_context_accessors")
endif()

# api_edt_id_profiling: full_counters (OBJ EDT counter) build; SKIP stub in other counter
# builds. Gated on the header-derived ARTS_OBJECT_EDT_TABLE_ENABLED macro the body checks, so
# it builds+links in any build dir and self-selects body vs SKIP.
add_arts_test(api_edt_id_profiling)
register_single_node_test(api_edt_id_profiling TIMEOUT 10)
set_tests_properties(api_edt_id_profiling PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS api_edt_id_profiling|SKIP api_edt_id_profiling")

# --- C13: EDT semantics (census metadata lost; classified by scanning tests/ocr/) ---
add_arts_test(edt_sentinel_single_fire)
register_single_node_test(edt_sentinel_single_fire TIMEOUT 30)
set_tests_properties(edt_sentinel_single_fire PROPERTIES PASS_REGULAR_EXPRESSION "PASS edt_sentinel_single_fire|SKIP edt_sentinel_single_fire")

# edt_satisfy_out_of_range EXPOSES a satisfy-out-of-range bug: prints no PASS token (FAIL+abort
# or hang). Registered WITHOUT a PASS_REGULAR_EXPRESSION. Do not mask.
add_arts_test(edt_satisfy_out_of_range)
register_single_node_test(edt_satisfy_out_of_range TIMEOUT 30)

add_arts_test(edt_destroy_vs_satisfy)
register_single_node_test(edt_destroy_vs_satisfy TIMEOUT 30)
set_tests_properties(edt_destroy_vs_satisfy PROPERTIES PASS_REGULAR_EXPRESSION "PASS edt_destroy_vs_satisfy|SKIP edt_destroy_vs_satisfy")

# edt_output_event: picks consumer_rank 1 when available; runs on 1n too. single + multinode.
add_arts_test(edt_output_event)
register_single_node_test(edt_output_event TIMEOUT 30)
register_multinode_test(edt_output_event TIMEOUT 60)
set_tests_properties(edt_output_event PROPERTIES PASS_REGULAR_EXPRESSION "PASS edt_output_event|SKIP edt_output_event")
set_tests_properties(edt_output_event_2n PROPERTIES PASS_REGULAR_EXPRESSION "PASS edt_output_event|SKIP edt_output_event")

# edt_finish_scope_balance: round-robins members across all ranks; runs on 1n too. single + multinode.
add_arts_test(edt_finish_scope_balance)
register_single_node_test(edt_finish_scope_balance TIMEOUT 30)
register_multinode_test(edt_finish_scope_balance TIMEOUT 60)
foreach(_v "" _2n _3n _4n _2n_io)
    set_tests_properties(edt_finish_scope_balance${_v} PROPERTIES PASS_REGULAR_EXPRESSION "PASS edt_finish_scope_balance|SKIP edt_finish_scope_balance")
endforeach()

# edt_remote_create_race: needs 2+ ranks; SKIPs cleanly on 1n.
add_arts_test(edt_remote_create_race)
register_multinode_test(edt_remote_create_race TIMEOUT 60)
foreach(_v 2n 3n 4n 2n_io)
    set_tests_properties(edt_remote_create_race_${_v} PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS edt_remote_create_race|SKIP edt_remote_create_race")
endforeach()

add_arts_test(edt_signal_alias)
register_single_node_test(edt_signal_alias TIMEOUT 30)
set_tests_properties(edt_signal_alias PROPERTIES PASS_REGULAR_EXPRESSION "PASS edt_signal_alias|SKIP edt_signal_alias")

# edt_size_offsets: pure_unit, but calls libarts symbols (arts_get_depv, check_out_edts) so it
# links libarts via add_arts_test + register_pure_unit_test (starts no runtime).
add_arts_test(edt_size_offsets)
register_pure_unit_test(edt_size_offsets TIMEOUT 30)
set_tests_properties(edt_size_offsets PROPERTIES PASS_REGULAR_EXPRESSION "PASS edt_size_offsets|SKIP edt_size_offsets")

# --- Wave C/D pure_unit tests (C14 T150 / C15 T162) ---
# array_list_ctx_semantics: compiles array_list.c directly (libc arts_malloc/arts_free shims),
# no ARTS runtime/library link.
add_pure_unit_src(array_list_ctx_semantics SOURCES ${CMAKE_SOURCE_DIR}/libs/src/core/utils/array_list.c PASS_REGEX "PASS array_list_ctx_semantics" TIMEOUT 30)
# vector_basic: #includes vector.c directly (libc shims); pins the contracts
# the per-worker EDT-context lists lean on (lazy alloc, doubling, swap-remove,
# clear-keeps-block, free-keeps-sentinel).
add_pure_unit_src(vector_basic PASS_REGEX "PASS vector_basic" TIMEOUT 30)
# event_drain_simple_idempotent: header-only (arts/utils/lockfree_lifo.h); no library link.
add_pure_unit_src(event_drain_simple_idempotent PASS_REGEX "PASS event_drain_simple_idempotent" TIMEOUT 30)

# ============================================================================
# C06 per-protocol pure_unit tests
#
# These compile specific protocol-coherence source TUs directly (NOT linking
# libarts) via add_pure_unit_src with ARTS_UNIT_STANDALONE_SHIMS, and self-adapt
# to the build dir's protocol macro (self-skip / print SKIP in non-target
# configs).
# ============================================================================

# directory_grantreq_queue: VAL/EXCL (the .c #if-guards select the protocol directory.c; WRF_VAL self-skips).
add_pure_unit_src(directory_grantreq_queue DEFINES ARTS_UNIT_STANDALONE_SHIMS PASS_REGEX "PASS directory_grantreq_queue:|SKIP" TIMEOUT 60)

# pending_rw_treiber: VAL only (self-skips else).
add_pure_unit_src(pending_rw_treiber DEFINES ARTS_UNIT_STANDALONE_SHIMS PASS_REGEX "PASS pending_rw_treiber:|SKIP" TIMEOUT 60)

# excl_compute_next: EXCL only (self-skips else).
add_pure_unit_src(excl_compute_next DEFINES ARTS_UNIT_STANDALONE_SHIMS PASS_REGEX "PASS excl_compute_next:|SKIP" TIMEOUT 60)

# inv_compute_next: INV only (self-skips else).  One truth table covers both
# placements — the arbiters are placement-independent.
add_pure_unit_src(inv_compute_next DEFINES ARTS_UNIT_STANDALONE_SHIMS PASS_REGEX "PASS inv_compute_next:|SKIP" TIMEOUT 60)


# acquire_is_serialized (T068): needs_full_build — it does NOT #include a coherence .c; it
# links the real arts_db_acquire_is_serialized symbol out of the per-config static libarts and
# never starts the runtime. So it uses add_arts_test (links libarts) + register_pure_unit_test.
# Runs (and asserts the protocol-correct answer) under all 6 configs.
add_arts_test(acquire_is_serialized)
register_pure_unit_test(acquire_is_serialized TIMEOUT 30)
set_tests_properties(acquire_is_serialized PROPERTIES PASS_REGULAR_EXPRESSION "PASS acquire_is_serialized|SKIP acquire_is_serialized")

# ============================================================================
# C13 — EDT lifecycle, satisfy, finish-scope (T139-T144)
#
# The six previously-missing C13 EDT tests.  Each links libarts and starts the
# runtime (except the GPU/CXL config_specific ones, which self-skip out of
# config).  Single-node ones gate on "PASS <name>|SKIP"; the multinode one is
# registered for all four flavours; the GPU one is wrapped in BUILD_CUDA_LIBRARY
# with the ARTS_USE_GPU define so its real body activates only in GPU builds;
# the CXL one is wrapped in ARTS_USE_CXL.
# ============================================================================

# T139 — arts_edt_register_cb_deleter constructor-order vs route_table.
add_arts_test(edt_register_cb_deleter)
register_single_node_test(edt_register_cb_deleter TIMEOUT 30)
set_tests_properties(edt_register_cb_deleter PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS edt_register_cb_deleter|SKIP")

# T143 — DB_MODE_VAL raw-uint64 dep delivered to depv[slot].
add_arts_test(edt_db_mode_val)
register_single_node_test(edt_db_mode_val TIMEOUT 30)
set_tests_properties(edt_db_mode_val PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS edt_db_mode_val|SKIP")

# T141 — cross-rank arts_edt_destroy + destroy-vs-satisfy (multinode; self-skips
# at node_count<2).
add_arts_test(edt_destroy_crossrank)
register_multinode_test(edt_destroy_crossrank TIMEOUT 60)

# T140 — GPU LC drain force-defer (config_specific: GPU). The .c self-skips when
# ARTS_USE_GPU is undefined; in a CUDA build the define activates the real body.
if(BUILD_CUDA_LIBRARY)
    add_arts_test(edt_gpu_lc_force_defer)
    target_compile_definitions(edt_gpu_lc_force_defer PRIVATE ARTS_USE_GPU)
    register_gpu_test(edt_gpu_lc_force_defer TIMEOUT 30)
    set_tests_properties(edt_gpu_lc_force_defer PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS edt_gpu_lc_force_defer|SKIP")
endif()

# T144 — CXL ptr-surfacing in arts_edt_satisfy_slot (config_specific: CXL).
if(ARTS_USE_CXL)
    add_arts_test(edt_cxl_ptr_surface)
    target_compile_definitions(edt_cxl_ptr_surface PRIVATE ARTS_USE_CXL)
    register_single_node_test(edt_cxl_ptr_surface TIMEOUT 30)
    set_tests_properties(edt_cxl_ptr_surface PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS edt_cxl_ptr_surface|SKIP")
endif()

# ============================================================================
# T294 — PASS-gate sweep: add FAIL_REGULAR_EXPRESSION to pre-existing OCR/unit
# tests that print a failure token (FAIL/ERROR/MISMATCH) on an internal failure
# but still exit 0, so a silent failure would pass ctest unnoticed.  This is the
# CONSERVATIVE form: a FAIL_REGULAR_EXPRESSION only catches the test's actual
# printed failure token, so it never false-fails a clean run (unlike a PASS
# regex, which would need an exact, always-present success token).  Every token
# matched here is printed ONLY on a failure branch (the runtime's own [ERROR]
# log always aborts, so it never appears on a clean run either).  Tests that
# rely solely on a nonzero exit code, the term_* signal tests, GPU .cu tests,
# and the Wave-A/C/D generated tests (already gated or intentional bug-exposers)
# are intentionally NOT included here.
# ============================================================================
set_tests_properties(acquire_mode PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL|ERROR")
set_tests_properties(array_list_basic PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(arts_id PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL|ERROR")
set_tests_properties(atomics_locks PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(atomics_rmw_contention PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(atomics_rmw_conventions PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_basic PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_home_producer_ro_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_home_producer_ro_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_home_producer_ro_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_home_producer_ro_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_owner_confirm_gate PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_owner_confirm_gate_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_owner_confirm_gate_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_owner_confirm_gate_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_owner_confirm_gate_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_mixed_local_remote_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_mixed_local_remote_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_mixed_local_remote_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_mixed_local_remote_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
foreach(_v 2n 3n 4n)
    set_tests_properties(coherence_multi_writer_dist_${_v} PROPERTIES
        FAIL_REGULAR_EXPRESSION "FAIL"
        PASS_REGULAR_EXPRESSION "PASS: EXCL distributed arbitration|SKIP coherence_multi_writer_dist")
endforeach()
set_tests_properties(coherence_multi_writer_multi_db PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_multi_writer_same_addr PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_ro_accumulation PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_ro_acquire_stress PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rw_multihop_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rw_multihop_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rw_multihop_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rw_multihop_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rw_ordering PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rw_pipeline_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rw_pipeline_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rw_pipeline_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_rw_pipeline_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
# NOTE: coherence_rw_ro_rw legitimately prints "MISMATCH (racy, expected)" for
# its documented RW/RO reorder race, so MISMATCH must NOT gate it; only a hard
# "FAIL" indicates a real defect.
set_tests_properties(coherence_rw_ro_rw PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
# Intentionally racy: same-DB RW writers wired with no event ordering may observe
# a stale value and print "MISMATCH (racy, expected)" — that is a legal outcome
# under the app-ordered (DB-WRF) memory model, so it must NOT fail the test.  Only
# a hard "FAIL" (NULL ptr / hang) is a real failure.
set_tests_properties(coherence_rw_rw_chain_seal PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(coherence_stress_single_node PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_create PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_create_with_data PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_dependence PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_destroy PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_labeled_guid PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_local PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_local_create PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(db_pin PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(deque_chaselev_race PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(deque_grow_during_steal PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(deque_last_element_tiebreak PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(deque_single_thread PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(diag_timestamp PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_chain PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_create_basic PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_dep_variants PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_destroy PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_fan_out PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(edt_signal PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(event_chain PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(event_channel_fifo PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(event_channel_lifetime PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(event_destroy_race PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(event_hint_presets PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(event_idem_silent_oversatisfy PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(event_once_storm PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(event_sticky_late_bind PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(finish_event_mn_termination_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(finish_event_mn_termination_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(finish_event_mn_termination_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(finish_event_mn_termination_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(guid_basic PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(guid_range PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(hint_routing PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(malloc_alignment PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL|ERROR")
set_tests_properties(malloc_footprint_balance PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL|ERROR")
set_tests_properties(multinode_coherence_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(multinode_coherence_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(multinode_coherence_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(multinode_coherence_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(multinode_edt_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(multinode_edt_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(multinode_edt_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(multinode_edt_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(multinode_labeled_guid_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(multinode_labeled_guid_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(multinode_labeled_guid_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(multinode_labeled_guid_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(node_query PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(ooo_drain_repush PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(ooo_table_completeness PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(ooo_toctou_rescue PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(paramv_memcpy PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(event_channel_advanced PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(record_dep_at PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(route_table_install_race PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(route_table_remote_guid_2n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(route_table_remote_guid_3n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(route_table_remote_guid_4n PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(route_table_remote_guid_2n_io PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(stdio_forward_scale_test PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(stress_edt PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
set_tests_properties(utility_api PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")
