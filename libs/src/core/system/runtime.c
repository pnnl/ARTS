/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/
#include "arts/runtime_state.h"
#ifndef __cplusplus
/* tiered_pool.h relies on C11 _Atomic and is C-only; pull these on the C
 * side and guard the corresponding init/destroy calls below with the same
 * macro. */
#include "arts/event.h"             /* struct arts_event_dep_s */
#include "arts/utils/tiered_pool.h" /* arts_tiered_pool_init / destroy */
#endif
#include "arts/utils/malloc.h"

#include <assert.h>
#include <stdlib.h>

#include "arts/counter/Preamble.h"
#include "arts/counter/counter.h"
#include "arts/counter/object_counter.h"
#include "arts/db.h"
#include "arts/defs.h"
#include "arts/edt.h"
#include "arts/edt_context.h" /* arts_owned_finish_cleanup, ctx tls */
#include "arts/gas/guid.h"
#include "arts/gas/route_table.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/system/topology.h"
#include "arts/transport/dispatcher.h"
#include "arts/transport/outbox.h"
#include "arts/transport/protocol.h"
#include "arts/transport/socket.h"
#include "arts/utils/array_list.h"
#include "arts/utils/atomics.h"
#include "arts/utils/deque.h"

#ifdef ARTS_USE_GPU
#include "arts/gpu/gpu_internal.h"
#include "arts/gpu/gpu_stream.h"
#endif

#ifdef ARTS_USE_CXL
#include "arts/cxl/deque.h"
#endif

#define PACKET_SIZE 4096

static int arts_runtime_argc = 0;
static char **arts_runtime_argv = NULL;

ARTS_WEAK void init_per_node(unsigned int node_id, int argc, char **argv) {
  (void)node_id;
  (void)argc;
  (void)argv;
}

ARTS_WEAK void init_per_worker(unsigned int node_id, unsigned int worker_id,
                               int argc, char **argv) {
  (void)node_id;
  (void)worker_id;
  (void)argc;
  (void)argv;
}

ARTS_WEAK void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
}

struct arts_runtime_shared_s arts_node_info;

void arts_runtime_node_init(struct arts_config_s *config) {
  unsigned int tc = config->thread_count;

  /* Scheduler */
#ifdef ARTS_USE_GPU
  /* GPU EDTs are pushed onto per-worker GPU deques that only a GPU scheduler
   * loop drains; under the default (CPU-only) scheduler they are never popped,
   * so any program that spawns a GPU EDT hangs.  When GPU support is enabled
   * but the scheduler is still at its default selection, promote to the GPU
   * scheduler loop.  An explicit non-default choice (e.g. a GPU backoff/demand
   * variant) is respected.  Index 3 is arts_gpu_scheduler_loop in the GPU
   * build's scheduler_loop[] dispatch table. */
  if (config->gpu > 0 && config->scheduler == 0) {
    config->scheduler = 3;
  }
#endif
  arts_node_info.scheduler = scheduler_loop[config->scheduler];

  /* Deque implementation selection (0=simple, 1=priority) */
  arts_deque_select(config->deque_type);

  /* Per-thread indexed arrays */
  arts_node_info.deque =
      (struct arts_deque_s **)arts_malloc(sizeof(struct arts_deque_s *) * tc);
  arts_node_info.receiver_deque =
      config->receiver_thread_count
          ? (struct arts_deque_s **)arts_malloc(sizeof(struct arts_deque_s *) *
                                                config->receiver_thread_count)
          : NULL;
  arts_node_info.gpu_deque =
      (struct arts_deque_s **)arts_malloc(sizeof(struct arts_deque_s *) * tc);
  arts_node_info.route_table =
      (arts_route_table_t **)arts_calloc(tc, sizeof(arts_route_table_t *));
  arts_node_info.gpu_route_table =
      config->gpu ? (arts_route_table_t **)arts_calloc(
                        config->gpu, sizeof(arts_route_table_t *))
                  : NULL;
  {
    unsigned int shard_entries =
        config->route_table_entries / ARTS_REMOTE_ROUTE_SHARDS;
    if (shard_entries < 1) {
      shard_entries = 1;
    }
    unsigned int shard_shift = (config->route_table_size >= 3)
                                   ? config->route_table_size - 3
                                   : config->route_table_size;
    for (int s = 0; s < ARTS_REMOTE_ROUTE_SHARDS; s++) {
      arts_node_info.remote_route_table[s] =
          arts_new_route_table(shard_entries, shard_shift);
    }
  }
  arts_node_info.local_spin = (volatile bool **)arts_calloc(tc, sizeof(bool *));
  arts_node_info.thread_roles =
      (unsigned int *)arts_calloc(tc, sizeof(unsigned int));

  /* Thread counts */
  arts_node_info.worker_thread_count = config->worker_thread_count;
  arts_node_info.sender_thread_count = config->sender_thread_count;
  arts_node_info.receiver_thread_count = config->receiver_thread_count;
  arts_node_info.total_thread_count = tc;

  /* Synchronization barriers */
  arts_node_info.ready_to_push = tc;
  arts_node_info.ready_to_parallel_start = tc;
  arts_node_info.ready_to_inspect = tc;
  arts_node_info.ready_to_execute = tc;
  arts_node_info.ready_to_clean = tc;

  /* Locks and shutdown coordination */
  arts_node_info.steal_request_lock = 1U;
  arts_node_info.shutdown_state = 0U;
  arts_node_info.outbox_pending = 0U;
  /* Seed at our own rank so different ranks pick different first targets;
   * across ranks the round-robin then walks the cluster evenly instead of
   * hammering rank 0. */
  arts_node_info.db_rr_route = arts_global_rank_id;

  /* Network buffer */
  arts_node_info.buf = (char *)arts_malloc(PACKET_SIZE);
  arts_node_info.packet_size = PACKET_SIZE;

  /* GPU config */
  arts_node_info.gpu = config->gpu;
  arts_node_info.gpu_route_table_size = config->gpu_route_table_size;
  arts_node_info.gpu_route_table_entries = config->gpu_route_table_entries;
  arts_node_info.gpu_locality = config->gpu_locality;
  arts_node_info.gpu_fit = config->gpu_fit;
  arts_node_info.gpu_lc_sync = config->gpu_lc_sync;
  arts_node_info.gpu_max_edts = config->gpu_max_edts;
  arts_node_info.gpu_max_memory = config->gpu_max_memory;
  arts_node_info.gpu_p2p = config->gpu_p2p;
  arts_node_info.gpu_buff_on = config->gpu_buff_on;
  arts_node_info.free_db_after_gpu_run = config->free_db_after_gpu_run;
  arts_node_info.run_gpu_gc_idle = config->run_gpu_gc_idle;
  arts_node_info.run_gpu_gc_pre_edt = config->run_gpu_gc_pre_edt;
  arts_node_info.delete_zeros_gpu_gc = config->delete_zeros_gpu_gc;

#ifdef ARTS_USE_CXL
  /* CXL shared-memory deque and DB arenas */
  uint64_t cxl_total_devs = GET_CXL_DEV_COUNT();
  arts_printf("CXL FAM device count: %lu\n", cxl_total_devs);

  if (config->cxl_db_allocation_strategy == ARTS_CXL_DB_ALLOC_ROUND_ROBIN) {
    /* Round-robin: allocate one arena per available device. */
    unsigned int dev_count = (unsigned int)cxl_total_devs;
    if (dev_count == 0) {
      dev_count = 1; /* Fallback: at least one arena on device 0. */
    }
    if (dev_count > ARTS_CXL_MAX_DEVICES) {
      dev_count = ARTS_CXL_MAX_DEVICES;
    }
    uint64_t dev_ids[ARTS_CXL_MAX_DEVICES];
    for (unsigned int i = 0; i < dev_count; i++) {
      dev_ids[i] = (uint64_t)i;
    }
    arts_printf("CXL DB allocation: round_robin across %u device(s)\n",
                dev_count);
#ifdef ARTS_CXL_NATIVE
    if (!arts_global_rank_id) {
      arts_node_info.cxl_deque =
          arts_cxl_deque_create_with_arenas(dev_ids, dev_count);
    } else {
      arts_node_info.cxl_deque = arts_cxl_deque_get();
    }
#else
    arts_node_info.cxl_deque =
        arts_cxl_deque_create_with_arenas(dev_ids, dev_count);
#endif
    arts_node_info.cxl_db_dev_count = dev_count;
  } else {
    /* Static: allocate a single arena on the configured device. */
    uint64_t dev_id = (uint64_t)config->cxl_db_allocation_device;
    arts_printf("CXL DB allocation: static on device %lu\n", dev_id);
    uint64_t dev_ids[1] = {dev_id};
#ifdef ARTS_CXL_NATIVE
    if (!arts_global_rank_id) {
      arts_node_info.cxl_deque = arts_cxl_deque_create_with_arenas(dev_ids, 1);
    } else {
      arts_node_info.cxl_deque = arts_cxl_deque_get();
    }
#else
    arts_node_info.cxl_deque = arts_cxl_deque_create_with_arenas(dev_ids, 1);
#endif
    arts_node_info.cxl_db_dev_count = 1;
    arts_node_info.cxl_db_static_device = (unsigned int)dev_id;
  }

  arts_printf("CXL FAM device ID: %lu\n",
              GET_CXL_DEV_ID(arts_node_info.cxl_deque));
  arts_printf("CXL FAM region device ID: %lu\n", GET_CXL_REGION_DEV_ID());
  pthread_mutex_init(&arts_node_info.cxl_local_lock, NULL);
  arts_node_info.cxl_db_rr_idx = 0;
  assert(arts_cxl_deque_get_db_arena_range(arts_node_info.cxl_deque,
                                           &arts_node_info.cxl_db_arena_start,
                                           &arts_node_info.cxl_db_arena_end) &&
         "CXL DB arena pointers must be valid");
#endif

  /* GUID generation */
  arts_node_info.keys = (uint64_t **)arts_calloc(tc, sizeof(uint64_t *));
  arts_node_info.global_guid_thread_id =
      (uint64_t *)arts_calloc(tc, sizeof(uint64_t));

  /* Performance counters */
  arts_node_info.counter_folder = config->counter_folder;
  arts_node_info.counter_capture_interval = config->counter_capture_interval;

  arts_node_info.live_counters =
      (arts_counter_t **)arts_calloc(tc, sizeof(arts_counter_t *));

  arts_node_info.saved_counters =
      (arts_counter_t **)arts_calloc(tc, sizeof(arts_counter_t *));
  for (unsigned int t = 0; t < tc; t++) {
    arts_node_info.saved_counters[t] = (arts_counter_t *)arts_calloc(
        NUM_COUNTER_TYPES, sizeof(arts_counter_t));
  }

  arts_node_info.capture_arrays =
      (arts_array_list_t ***)arts_calloc(tc, sizeof(arts_array_list_t **));
  for (unsigned int t = 0; t < tc; t++) {
    arts_node_info.capture_arrays[t] = (arts_array_list_t **)arts_calloc(
        NUM_COUNTER_TYPES, sizeof(arts_array_list_t *));
    for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
      if (arts_counter_mode_array[i] == ARTS_COUNTER_MODE_PERIODIC) {
        arts_node_info.capture_arrays[t][i] =
            arts_new_array_list(sizeof(arts_counter_capture_t), 16);
      }
    }
  }

  /* Object counter storage (per-arts_id tracking) */
  arts_object_alloc_node_storage(tc);

#ifndef __cplusplus
  /* Per-rank pool of arts_event_dep_s nodes (Task 4o).  Allocate the
   * tiered_pool struct on the heap because runtime_state.h forward-declares
   * arts_tiered_pool_t (full definition lives in arts/utils/tiered_pool.h
   * which cannot be included from runtime_state.h without a circular
   * dependency).  Future work will make event.c consume this pool. */
  arts_node_info.event_dep_pool =
      (arts_tiered_pool_t *)arts_calloc(1, sizeof(arts_tiered_pool_t));
  arts_tiered_pool_cfg_t event_dep_cfg = {
      .H_local = 128, .B_local = 64, .H_numa = 1024, .B_numa = 256};
  arts_tiered_pool_init(arts_node_info.event_dep_pool,
                        sizeof(struct arts_event_dep_s), event_dep_cfg);
#endif

#ifdef ARTS_USE_GPU
  if (arts_node_info.gpu) {
    arts_node_init_gpus();
  }
#endif
}

void arts_runtime_global_cleanup() {
  arts_counter_capture_stop();
  // Write all counter outputs (thread, node, and cluster levels)
  unsigned int tc = arts_node_info.total_thread_count;
  for (unsigned int t = 0; t < tc; t++) {
    arts_counter_write(arts_node_info.counter_folder, arts_global_rank_id, t);
  }
  // Write object counter output (per-arts_id tracking)
  arts_object_write_node(arts_node_info.counter_folder, arts_global_rank_id,
                         tc);
  arts_clean_up_dbs();

  /* Counter cleanup (reverse of arts_runtime_node_init allocation) */
  for (unsigned int t = 0; t < tc; t++) {
    if (arts_node_info.capture_arrays && arts_node_info.capture_arrays[t]) {
      for (unsigned int i = 0; i < NUM_COUNTER_TYPES; i++) {
        if (arts_node_info.capture_arrays[t][i]) {
          arts_delete_array_list(arts_node_info.capture_arrays[t][i]);
        }
      }
      arts_free(arts_node_info.capture_arrays[t]);
    }
    if (arts_node_info.saved_counters) {
      arts_free(arts_node_info.saved_counters[t]);
    }
  }
  arts_free(arts_node_info.capture_arrays);
  arts_free(arts_node_info.saved_counters);
  arts_free(arts_node_info.live_counters);

  /* Object counter cleanup */
  arts_object_cleanup_node_storage(tc);

#ifdef ARTS_USE_CXL
  arts_cxl_deque_free(arts_node_info.cxl_deque);
  pthread_mutex_destroy(&arts_node_info.cxl_local_lock);
#endif

#ifdef ARTS_USE_GPU
  /* GPU cleanup must run BEFORE route tables are freed — free_gpu_item()
     calls arts_route_table_lookup_db_safe() for LC DB host-side metadata. */
  if (arts_node_info.gpu) {
    arts_cleanup_gpus();
  }
#endif

  /* Route table cleanup (after entries cleaned by arts_clean_up_dbs) */
  for (unsigned int i = 0; i < tc; i++) {
    arts_delete_route_table(arts_node_info.route_table[i]);
  }
  arts_free(arts_node_info.route_table);
  for (int s = 0; s < ARTS_REMOTE_ROUTE_SHARDS; s++) {
    arts_delete_route_table(arts_node_info.remote_route_table[s]);
  }

  /* Per-thread indexed arrays */
  arts_free(arts_node_info.deque);
  arts_free(arts_node_info.receiver_deque);
  arts_free(arts_node_info.gpu_deque);
  arts_free(arts_node_info.gpu_route_table);
  arts_free((void *)arts_node_info.local_spin);
  arts_free(arts_node_info.thread_roles);
  arts_free(arts_node_info.buf);
  for (unsigned int i = 0; i < tc; i++) {
    arts_free(arts_node_info.keys[i]);
  }
  arts_free(arts_node_info.keys);
  arts_free(arts_node_info.global_guid_thread_id);

#ifndef __cplusplus
  /* Tear down event_dep_pool (paired with init in arts_runtime_node_init). */
  if (arts_node_info.event_dep_pool) {
    arts_tiered_pool_destroy(arts_node_info.event_dep_pool);
    arts_free(arts_node_info.event_dep_pool);
    arts_node_info.event_dep_pool = NULL;
  }
#endif

  /* Network outbound queues and sequence tracking arrays */
  arts_transport_cleanup();

  /* Socket server global arrays (safe to call even for single-node) */
  arts_socket_cleanup();
}

/*
 * arts_thread_zero_node_start — Thread 0 (master) startup sequence.
 *
 * After all threads have registered (ready_to_push barrier), thread 0:
 *   1. Enables global GUID generation.
 *   2. Schedules main_edt on rank 0 (if defined by the application).
 *   3. Waits for all threads through a series of barriers before entering
 *      the main scheduler loop.
 */
void arts_thread_zero_node_start(int argc, char **argv) {
  ARTS_INFO("Thread 0: starting node initialization");
  arts_runtime_argc = argc;
  arts_runtime_argv = argv;
  set_global_guid_on();

  // Note: Counter capture starts AFTER barriers below, when receiver threads
  // are running. This ensures time sync messages can be processed.
  TIME_INIT_STOP();
  TIME_TOTAL_START();

  if (init_per_node) {
    init_per_node(arts_global_rank_id, argc, argv);
  }

#ifdef ARTS_USE_GPU
  arts_init_per_gpu_wrapper(argc, argv);
#endif
  set_guid_generator_after_parallel_start();

  arts_atomic_sub(&arts_node_info.ready_to_parallel_start, 1U);
  while (arts_node_info.ready_to_parallel_start) {
  }
  if (init_per_worker && arts_thread_info.role == ARTS_ROLE_WORKER) {
    init_per_worker(arts_global_rank_id, arts_thread_info.group_pos, argc,
                    argv);
  }
  if (!arts_global_rank_id) {
    ARTS_INFO("Thread 0: scheduling main_edt on rank 0 (argc=%d)", argc);
    uint64_t main_args[2] = {(uint64_t)argc, (uint64_t)argv};
    arts_edt_hint_t main_hint = ARTS_EDT_HINT_DEFAULTS;
    arts_edt_create(main_edt, 2, main_args, 0, &main_hint);
  }

  arts_owned_finish_cleanup();

  arts_atomic_sub(&arts_node_info.ready_to_inspect, 1U);
  while (arts_node_info.ready_to_inspect) {
  }
  arts_atomic_sub(&arts_node_info.ready_to_execute, 1U);
  while (arts_node_info.ready_to_execute) {
  }

  // Start counter capture AFTER all barriers, when receiver threads are in
  // their runtime loops. This ensures time sync requests can be processed.
  arts_counter_capture_start();
}

void arts_runtime_private_init(struct thread_mask_s *thread,
                               struct arts_config_s *config) {
  arts_node_info.deque[thread->id] = arts_thread_info.my_deque =
      arts_deque_new(config->deque_size);
  arts_node_info.gpu_deque[thread->id] = arts_thread_info.my_gpu_deque =
      (config->gpu && thread->role == ARTS_ROLE_WORKER)
          ? arts_deque_new(config->deque_size)
          : NULL;
  /* Every thread owns a local route table at its own slot: any thread (workers
   * AND network threads — multiple receivers run arts_handler_edt_create
   * concurrently) mints GUIDs into a disjoint key partition keyed by
   * thread->id, and arts_get_route_table resolves those local GUIDs to
   * route_table[id]. */
  arts_node_info.route_table[thread->id] = arts_new_route_table(
      config->route_table_entries, config->route_table_size);
  if (thread->role == ARTS_ROLE_WORKER) {
#ifdef ARTS_USE_GPU
    if (config->gpu) {
      arts_worker_init_gpus();
    }
#endif
  }

  if (thread->role == ARTS_ROLE_SENDER || thread->role == ARTS_ROLE_RECEIVER) {
    if (thread->role == ARTS_ROLE_SENDER) {
      unsigned int size = arts_global_rank_count * config->port_count /
                          arts_node_info.sender_thread_count;
      unsigned int rem = arts_global_rank_count * config->port_count %
                         arts_node_info.sender_thread_count;
      unsigned int start;
      if (thread->group_pos < rem) {
        start = thread->group_pos * (size + 1);
        arts_remote_set_thread_outbound_queues(start, start + size + 1);
      } else {
        start = (rem * (size + 1)) + ((thread->group_pos - rem) * size);
        arts_remote_set_thread_outbound_queues(start, start + size);
      }
    }
    if (thread->role == ARTS_ROLE_RECEIVER) {
      arts_node_info.receiver_deque[thread->group_pos] =
          arts_node_info.deque[thread->id];
      unsigned int size = (arts_global_rank_count - 1) * config->port_count /
                          arts_node_info.receiver_thread_count;
      unsigned int rem = (arts_global_rank_count - 1) * config->port_count %
                         arts_node_info.receiver_thread_count;
      unsigned int start;
      if (thread->group_pos < rem) {
        start = thread->group_pos * (size + 1);
        arts_remote_set_thread_inbound_queues(start, start + size + 1);
      } else {
        start = (rem * (size + 1)) + ((thread->group_pos - rem) * size);
        arts_remote_set_thread_inbound_queues(start, start + size);
      }
    }
  }
  arts_node_info.local_spin[thread->id] = &arts_thread_info.alive;
  arts_node_info.thread_roles[thread->id] = (unsigned int)thread->role;
  arts_thread_info.alive = true;
  arts_thread_info.pu_id = thread->pu_id;
  arts_thread_info.thread_id = thread->id;
  arts_thread_info.group_pos = thread->group_pos;
  arts_thread_info.numa_domain_id = thread->numa_domain_id;
  arts_thread_info.role = thread->role;
  arts_thread_info.current_edt_guid = 0;

  // Register thread-local counter storage with nodeInfo
  arts_node_info.live_counters[thread->id] = arts_thread_local_counters;
#if ENABLE_ARTS_ID_EDT_METRICS || ENABLE_ARTS_ID_DB_METRICS
  arts_id_init_hash_table(&arts_thread_local_arts_id_metrics);
#endif
#if ENABLE_ARTS_ID_EDT_CAPTURES
  arts_thread_local_edt_capture_list =
      arts_new_array_list(sizeof(arts_id_edt_capture_t), 1024);
#endif
#if ENABLE_ARTS_ID_DB_CAPTURES
  arts_thread_local_db_capture_list =
      arts_new_array_list(sizeof(arts_id_db_capture_t), 1024);
#endif

  arts_guid_key_generator_init();

  arts_atomic_sub(&arts_node_info.ready_to_push, 1U);
  while (arts_node_info.ready_to_push) {
  };
  if (thread->id) {
    arts_atomic_sub(&arts_node_info.ready_to_parallel_start, 1U);
    while (arts_node_info.ready_to_parallel_start) {
    };

    if (arts_thread_info.role == ARTS_ROLE_WORKER) {
      if (init_per_worker) {
        init_per_worker(arts_global_rank_id, arts_thread_info.group_pos,
                        arts_runtime_argc, arts_runtime_argv);
      }
      arts_owned_finish_cleanup();
    }

    arts_atomic_sub(&arts_node_info.ready_to_inspect, 1U);
    while (arts_node_info.ready_to_inspect) {
    };
    arts_atomic_sub(&arts_node_info.ready_to_execute, 1U);
    while (arts_node_info.ready_to_execute) {
    };
  }
  arts_thread_info.drand_buf[0] = 1202107158 + (thread->id * 1999);
  arts_thread_info.drand_buf[1] = 0;
  arts_thread_info.drand_buf[2] = 0;
}

void arts_runtime_private_cleanup() {
  arts_atomic_sub(&arts_node_info.ready_to_clean, 1U);
  while (arts_node_info.ready_to_clean) {
  };
  arts_remote_thread_outbound_queues_cleanup();
  arts_remote_thread_inbound_queues_cleanup();
  if (arts_thread_info.my_deque) {
    arts_deque_delete(arts_thread_info.my_deque);
  }
  if (arts_thread_info.my_gpu_deque) {
    arts_deque_delete(arts_thread_info.my_gpu_deque);
  }
  arts_cleanup_edt_tls();
}

/*
 * arts_runtime_stop — Stop all worker/network threads.
 *
 * Called from arts_shutdown() (single-node) or from the network send thread
 * after the shutdown timeout (multi-node).
 *
 * Protocol:
 *   1. Wait for each thread to register its local_spin pointer (non-NULL
 *      means the thread has finished arts_runtime_private_init).
 *   2. Set *local_spin[i] = false, which clears arts_thread_info.alive for
 *      that thread, causing it to exit its scheduler/network loop.
 */
/*
 * Helper: walk the thread table and clear alive=false for every thread
 * whose role matches `role_mask`. The registration spin is bounded to
 * avoid an infinite busy-wait if a thread crashed during init.
 */
static void arts_runtime_stop_by_role(unsigned int role_mask,
                                      const char *role_label) {
  const unsigned int MAX_SPIN = 10000000; /* ~sub-second upper bound */
  unsigned int i;
  for (i = 0; i < arts_node_info.total_thread_count; i++) {
    /* Skip threads whose role is not in the mask. Thread 0 is the main
     * thread and always has WORKER role. */
    if ((1U << arts_node_info.thread_roles[i]) & role_mask) {
      unsigned int spin = 0;
      while (!arts_node_info.local_spin[i]) {
        if (++spin >= MAX_SPIN) {
          ARTS_WARN("arts_runtime_stop_%s: thread %u never registered "
                    "local_spin — giving up (may leak)",
                    role_label, i);
          goto next;
        }
      }
      (*arts_node_info.local_spin[i]) = false;
      ARTS_DEBUG("arts_runtime_stop_%s: thread %u signaled to stop", role_label,
                 i);
    }
  next:
    continue;
  }
}

/*
 * arts_runtime_stop_workers — clear alive on worker threads only.
 * Network threads (senders/receivers) stay alive so they can keep
 * handling shutdown-related traffic.
 */
void arts_runtime_stop_workers() {
  arts_node_info.shutdown_state = 1U;
  ARTS_INFO("arts_runtime_stop_workers");
  arts_runtime_stop_by_role(1U << ARTS_ROLE_WORKER, "workers");
}

/*
 * arts_runtime_stop_network — clear alive on sender and receiver threads.
 * Call AFTER arts_runtime_stop_workers and the associated worker loop
 * exits, once no more network traffic is expected.
 */
void arts_runtime_stop_network() {
  ARTS_INFO("arts_runtime_stop_network");
  arts_runtime_stop_by_role(
      (1U << ARTS_ROLE_RECEIVER) | (1U << ARTS_ROLE_SENDER), "network");
}

void arts_runtime_stop() {
  /* Legacy entry point: stop everything in one call. Preserved for the
   * few call sites (e.g. receiver-thread EOF handling in socket.c) that
   * the shutdown protocol refactor has not yet migrated to the cleaner
   * arts_enter_shutdown_state(false) path. */
  arts_node_info.shutdown_state = 1U;
  ARTS_INFO("arts_runtime_stop: stopping %u threads",
            arts_node_info.total_thread_count);
  arts_runtime_stop_by_role(0xFFFFFFFFU, "all");
  ARTS_INFO("arts_runtime_stop: all threads signaled");
}

/*
 * arts_runtime_loop — Main per-thread dispatch loop.
 *
 * Each thread enters exactly one of three roles:
 *   - network_receive: Polls for incoming messages (multi-node only).
 *   - network_send:    Drains outbound queues; triggers arts_runtime_stop()
 *                      when shutdown timeout elapses.
 *   - worker:          Runs the selected scheduler loop until alive==false.
 *
 * On single-node configurations, all threads are workers (no network threads).
 * The loop exits when arts_runtime_stop() sets alive=false for this thread.
 */
int arts_runtime_loop() {
  ARTS_DEBUG("Thread %u entering runtime_loop (role=%d)",
             arts_thread_info.thread_id, arts_thread_info.role);
  switch (arts_thread_info.role) {
  case ARTS_ROLE_RECEIVER:
    while (arts_thread_info.alive) {
      arts_transport_receive();
    }
    break;
  case ARTS_ROLE_SENDER:
    while (arts_thread_info.alive) {
      arts_remote_async_send();
    }
    break;
  case ARTS_ROLE_WORKER:
    while (arts_thread_info.alive) {
      arts_node_info.scheduler();
    }
    break;
  default:
    break;
  }
  ARTS_DEBUG("Thread %u exiting runtime_loop", arts_thread_info.thread_id);
  return 0;
}
