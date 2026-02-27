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
#include "arts/utils/malloc.h"

#include <stdlib.h>
#include <time.h>

#include "arts/counter/Preamble.h"
#include "arts/counter/counter.h"
#include "arts/counter/object_counter.h"
#include "arts/defs.h"
#include "arts/gas/guid.h"
#include "arts/gas/route_table.h"
#include "arts/transport/protocol.h"
#include "arts/transport/dispatcher.h"
#include "arts/transport/socket.h"
#include "arts/compute/edt.h"
#include "arts/memory/db.h"
#include "arts/sync/termination.h"
#include "arts/system/topology.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/array_list.h"
#include "arts/utils/atomics.h"
#include "arts/utils/deque.h"

#ifdef ARTS_USE_GPU
#include "arts/gpu/gpu_internal.h"
#include "arts/gpu/gpu_stream.h"
#endif

#define PACKET_SIZE 4096
#define NETWORK_BACKOFF_INCREMENT 0

extern unsigned int num_numa_domains;

ARTS_WEAK void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
}

struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

typedef bool (*scheduler_t)(void);
#ifdef ARTS_USE_GPU
scheduler_t scheduler_loop[] = {
    (scheduler_t)arts_default_scheduler_loop,
    (scheduler_t)arts_network_before_steal_scheduler_loop,
    (scheduler_t)arts_network_first_scheduler_loop,
    (scheduler_t)arts_gpu_scheduler_loop,
    (scheduler_t)arts_gpu_scheduler_backoff_loop,
    (scheduler_t)arts_gpu_scheduler_demand_loop};
#else
scheduler_t scheduler_loop[] = {
    (scheduler_t)arts_default_scheduler_loop,
    (scheduler_t)arts_network_before_steal_scheduler_loop,
    (scheduler_t)arts_network_first_scheduler_loop};
#endif

void arts_runtime_node_init(struct arts_config_s *config) {
  unsigned int tc = config->thread_count;

  /* Scheduler */
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
  arts_node_info.remote_route_table = arts_new_route_table(
      config->route_table_entries, config->route_table_size);
  arts_node_info.local_spin = (volatile bool **)arts_calloc(tc, sizeof(bool *));
  arts_node_info.memory_moves =
      (unsigned int **)arts_calloc(tc, sizeof(unsigned int *));
  arts_node_info.atomic_waits =
      (struct atomic_create_barrier_info_s **)arts_calloc(
          tc, sizeof(struct atomic_create_barrier_info_s *));

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
  arts_node_info.send_lock = 0U;
  arts_node_info.recv_lock = 0U;
  arts_node_info.steal_request_lock = 1U;
  arts_node_info.shutdown_count = arts_global_rank_count - 1;
  arts_node_info.ready_to_shutdown = arts_global_rank_count - 1;
  arts_node_info.auto_shutdown_guid = config->auto_shutdown ? 1 : NULL_GUID;

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

#ifdef ARTS_USE_GPU
  /* GPU cleanup must run BEFORE route tables are freed — free_gpu_item()
     calls arts_route_table_lookup_db() for LC DB host-side metadata. */
  if (arts_node_info.gpu) {
    arts_cleanup_gpus();
  }
#endif

  /* Route table cleanup (after entries cleaned by arts_clean_up_dbs) */
  for (unsigned int i = 0; i < tc; i++) {
    arts_delete_route_table(arts_node_info.route_table[i]);
  }
  arts_free(arts_node_info.route_table);
  arts_delete_route_table(arts_node_info.remote_route_table);

  /* Per-thread indexed arrays */
  arts_free(arts_node_info.deque);
  arts_free(arts_node_info.receiver_deque);
  arts_free(arts_node_info.gpu_deque);
  arts_free(arts_node_info.gpu_route_table);
  arts_free((void *)arts_node_info.local_spin);
  arts_free(arts_node_info.memory_moves);
  arts_free(arts_node_info.atomic_waits);
  arts_free(arts_node_info.buf);
  for (unsigned int i = 0; i < tc; i++) {
    arts_free(arts_node_info.keys[i]);
  }
  arts_free(arts_node_info.keys);
  arts_free(arts_node_info.global_guid_thread_id);

  /* Network outbound queues and sequence tracking arrays */
  arts_server_cleanup();

  /* Socket server global arrays (safe to call even for single-node) */
  arts_ll_server_cleanup();
}

/*
 * arts_thread_zero_node_start — Thread 0 (master) startup sequence.
 *
 * After all threads have registered (ready_to_push barrier), thread 0:
 *   1. Enables global GUID generation.
 *   2. Creates the shutdown epoch (termination detection).
 *   3. Schedules main_edt on rank 0 (if defined by the application).
 *   4. Waits for all threads through a series of barriers before entering
 *      the main scheduler loop.
 */
void arts_thread_zero_node_start(int argc, char **argv) {
  ARTS_INFO("Thread 0: starting node initialization");
  set_global_guid_on();
  arts_shutdown_epoch_create();

  // Note: Counter capture starts AFTER barriers below, when receiver threads
  // are running. This ensures time sync messages can be processed.
  TIME_INIT_STOP();
  TIME_TOTAL_START();

#ifdef ARTS_USE_GPU
  arts_init_per_gpu_wrapper(argc, argv);
#endif
  set_guid_generator_after_parallel_start();

  arts_atomic_sub(&arts_node_info.ready_to_parallel_start, 1U);
  while (arts_node_info.ready_to_parallel_start) {
  }
  if (!arts_global_rank_id) {
    ARTS_INFO("Thread 0: scheduling main_edt on rank 0 (argc=%d)", argc);
    uint64_t main_args[2] = {(uint64_t)argc, (uint64_t)argv};
    arts_hint_t main_hint = {0, 0};
    arts_edt_create(main_edt, 2, main_args, 0, &main_hint);
  }

  arts_increment_finished_epoch_list();

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
  if (thread->role == ARTS_ROLE_WORKER) {
    arts_node_info.route_table[thread->id] = arts_new_route_table(
        config->route_table_entries, config->route_table_size);
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
  arts_thread_info.alive = true;
  arts_node_info.memory_moves[thread->id] =
      (unsigned int *)&arts_thread_info.outstanding_memory_moves;
  arts_node_info.atomic_waits[thread->id] = &arts_thread_info.atomic_wait;
  arts_thread_info.atomic_wait.wait = true;
  arts_thread_info.outstanding_memory_moves = 0;
  arts_thread_info.pu_id = thread->pu_id;
  arts_thread_info.thread_id = thread->id;
  arts_thread_info.group_pos = thread->group_pos;
  arts_thread_info.numa_domain_id = thread->numa_domain_id;
  arts_thread_info.role = thread->role;
  arts_thread_info.back_off = 1;
  arts_thread_info.current_edt_guid = 0;
  arts_thread_info.local_counting = 1;
  arts_thread_info.shad_lock = 0;

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
      arts_increment_finished_epoch_list();
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
  if (arts_thread_info.my_node_deque) {
    arts_deque_delete(arts_thread_info.my_node_deque);
  }
  if (arts_thread_info.my_gpu_deque) {
    arts_deque_delete(arts_thread_info.my_gpu_deque);
  }
  arts_cleanup_epoch_pools();
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
void arts_runtime_stop() {
  ARTS_INFO("arts_runtime_stop: stopping %u threads",
            arts_node_info.total_thread_count);
  unsigned int i;
  for (i = 0; i < arts_node_info.total_thread_count; i++) {
    ARTS_DEBUG("arts_runtime_stop: waiting for thread %u to register", i);
    while (!arts_node_info.local_spin[i]) {
      ;
    }
    (*arts_node_info.local_spin[i]) = false;
    ARTS_DEBUG("arts_runtime_stop: thread %u signaled to stop", i);
  }
  ARTS_INFO("arts_runtime_stop: all threads signaled");
}

void arts_handle_remote_stolen_edt(struct arts_edt_s *edt) {
  ARTS_DEBUG("Processing stolen EDT[Id:%lu, Guid:%lu] on PU %u", edt->arts_id,
             edt->current_edt, arts_thread_info.pu_id);
  increment_queue_epoch(edt->epoch_guid);
  arts_shutdown_epoch_inc_queue();
#ifdef ARTS_USE_GPU
  if (arts_node_info.gpu &&
      (!arts_thread_info.my_deque || !arts_thread_info.my_gpu_deque))
    arts_store_new_edts(edt);
  else
#endif
  {
    if (edt->edt_type == ARTS_EDT_GPU) {
      arts_deque_push_front(arts_thread_info.my_gpu_deque, edt, 0);
    } else {
      arts_deque_push_front(arts_thread_info.my_deque, edt, 0);
    }
  }
}

/*
 * arts_handle_ready_edt — Transition an EDT from "all deps signaled" to
 *                         "queued for execution".
 *
 * Called when depc_needed reaches 0 after the last signal or after the
 * sentinel is removed during creation.
 *
 * Two phases:
 *   Phase 1 (acquire_dbs): Re-initialize depc_needed = depc + 1 (sentinel)
 *     and attempt to acquire each DB dependency locally.  If a DB is not
 *     available, an OOO request is issued; when it resolves later, it will
 *     decrement depc_needed and potentially push the EDT to the deque.
 *   Phase 2 (sentinel removal): Atomically decrement the sentinel.  If all
 *     DBs were acquired synchronously, depc_needed hits 0 here and the EDT
 *     is pushed to the worker deque for execution.
 */
void arts_handle_ready_edt(struct arts_edt_s *edt) {
  ARTS_INFO("EDT[Guid:%lu, Id:%lu] ready — entering acquire_dbs "
            "(depc=%u)",
            edt->current_edt, edt->arts_id, edt->depc);
  acquire_dbs(edt);
  unsigned int remaining = arts_atomic_sub(&edt->depc_needed, 1U);
  ARTS_INFO("EDT[Guid:%lu] acquire_dbs done, sentinel removed: "
            "depc_needed=%u",
            edt->current_edt, remaining);
  if (remaining == 0) {
    INCREMENT_NUM_EDT_ACQUIRE_BY(1);
    increment_queue_epoch(edt->epoch_guid);
    arts_shutdown_epoch_inc_queue();
#ifdef ARTS_USE_GPU
    if (arts_node_info.gpu &&
        (!arts_thread_info.my_deque || !arts_thread_info.my_gpu_deque)) {
      if (!arts_thread_info.my_deque) {
        /* CUDA callback thread: new_edts/new_edt_lock set from closure */
        arts_store_new_edts(edt);
      } else {
        /* Non-worker thread (sender/receiver): push to worker 0's deque */
        if (edt->edt_type == ARTS_EDT_GPU) {
          arts_deque_push_front(arts_node_info.gpu_deque[0], edt, 0);
        } else {
          arts_deque_push_front(arts_node_info.deque[0], edt, 0);
        }
      }
    } else
#endif
    {
      if (edt->edt_type == ARTS_EDT_GPU) {
        ARTS_INFO("EDT[Guid:%lu] pushed to GPU deque", edt->current_edt);
        arts_deque_push_front(arts_thread_info.my_gpu_deque, edt, 0);
      } else {
        ARTS_INFO("EDT[Guid:%lu] pushed to worker deque", edt->current_edt);
        arts_deque_push_front(arts_thread_info.my_deque, edt, 0);
      }
    }
  } else {
    ARTS_DEBUG("EDT[Guid:%lu] waiting for %u more DB acquisitions",
               edt->current_edt, remaining);
  }
}

void arts_run_edt(struct arts_edt_s *edt) {
  uint32_t depc = edt->depc;
  arts_edt_dep_t *depv =
      (arts_edt_dep_t *)(((uint64_t *)(edt + 1)) + edt->paramc);

  arts_edt_t func = edt->func_ptr;
  uint32_t paramc = edt->paramc;
  const uint64_t *paramv = (uint64_t *)(edt + 1);

  ARTS_INFO("Running EDT[Id:%lu, Guid:%lu, Deps: %u, Params: %u, "
            "DepvPtr: %p]",
            edt->arts_id, edt->current_edt, depc, paramc, depv);
  prep_dbs(depc, depv, false);

  arts_set_thread_local_edt_info(edt);

  TIME_EDT_EXEC_START();
  struct timespec start_time;
  struct timespec end_time;
  (void)clock_gettime(CLOCK_MONOTONIC, &start_time);
  func(paramc, paramv, depc, depv);
  (void)clock_gettime(CLOCK_MONOTONIC, &end_time);
  TIME_EDT_EXEC_STOP();

  // Record per-object EDT metrics
  uint64_t exec_ns = ((end_time.tv_sec - start_time.tv_sec) * 1000000000ULL) +
                     (end_time.tv_nsec - start_time.tv_nsec);
  arts_object_record_edt(edt->arts_id, exec_ns, 0);
  arts_object_trace_edt(edt->arts_id, exec_ns, 0);

  INCREMENT_NUM_EDT_FINISH_BY(1);

  arts_unset_thread_local_edt_info();

  // This is for a synchronous path
  if (edt->output_buffer != NULL_GUID) {
    arts_set_buffer(edt->output_buffer, arts_calloc(1, sizeof(unsigned int)),
                    sizeof(unsigned int));
  }

  ARTS_INFO("EDT[Guid:%lu, Id:%lu] finished (exec_ns=%lu)", edt->current_edt,
            edt->arts_id, exec_ns);
  release_dbs(depc, depv, false);
  arts_release_created_dbs();
  arts_edt_delete(edt);
  DEC_OUTSTANDING_EDTS(1);
  ARTS_DEBUG("EDT completed, outstanding_edts decremented");
}

inline struct arts_edt_s *arts_runtime_steal_from_network() {
  struct arts_edt_s *edt = NULL;
  if (arts_global_rank_count > 1) {
    unsigned int index = arts_thread_info.thread_id;
    for (unsigned int i = 0; i < arts_node_info.receiver_thread_count; i++) {
      index = (index + 1) % arts_node_info.receiver_thread_count;
      if ((edt = (struct arts_edt_s *)arts_deque_pop_back(
               arts_node_info.receiver_deque[index])) != NULL) {
        break;
      }
    }
  }
  return edt;
}

inline struct arts_edt_s *arts_runtime_steal_from_worker() {
  struct arts_edt_s *edt = NULL;
  if (arts_node_info.total_thread_count > 1) {
    INCREMENT_NUM_STEAL_ATTEMPT_BY(1);
    long unsigned int steal_loc;
    do {
      steal_loc = jrand48(arts_thread_info.drand_buf);
      steal_loc = steal_loc % arts_node_info.total_thread_count;
    } while (steal_loc == arts_thread_info.thread_id);
    edt = (struct arts_edt_s *)arts_deque_pop_back(
        arts_node_info.deque[steal_loc]);
    if (edt) {
      INCREMENT_NUM_STEAL_SUCCESS_BY(1);
    }
  }
  return edt;
}

bool arts_network_first_scheduler_loop() {
  struct arts_edt_s *edt_found;
  if (!(edt_found = arts_runtime_steal_from_network())) {
    if (!(edt_found = (struct arts_edt_s *)arts_deque_pop_front(
              arts_thread_info.my_node_deque))) {
      if (!(edt_found = (struct arts_edt_s *)arts_deque_pop_front(
                arts_thread_info.my_deque))) {
        edt_found = arts_runtime_steal_from_worker();
      }
    }
  }
  if (edt_found) {
    arts_run_edt(edt_found);
    return true;
  }
  return false;
}

bool arts_network_before_steal_scheduler_loop() {
  struct arts_edt_s *edt_found;
  if (!(edt_found = (struct arts_edt_s *)arts_deque_pop_front(
            arts_thread_info.my_node_deque))) {
    if (!(edt_found = (struct arts_edt_s *)arts_deque_pop_front(
              arts_thread_info.my_deque))) {
      if (!(edt_found = arts_runtime_steal_from_network())) {
        edt_found = arts_runtime_steal_from_worker();
      }
    }
  }

  if (edt_found) {
    arts_run_edt(edt_found);
    return true;
  }
  return false;
}

struct arts_edt_s *arts_find_edt() {
  struct arts_edt_s *edt_found = NULL;
  if (!(edt_found = (struct arts_edt_s *)arts_deque_pop_front(
            arts_thread_info.my_deque))) {
    if (!edt_found) {
      if (!(edt_found = arts_runtime_steal_from_worker())) {
        edt_found = arts_runtime_steal_from_network();
      }
    }
  }
  return edt_found;
}

bool arts_default_scheduler_loop() {
  struct arts_edt_s *edt_found = NULL;
  if (!(edt_found = (struct arts_edt_s *)arts_deque_pop_front(
            arts_thread_info.my_deque))) {
    if (!edt_found) {
      if (!(edt_found = arts_runtime_steal_from_worker())) {
        edt_found = arts_runtime_steal_from_network();
      }
    }
  }

  if (edt_found) {
    arts_run_edt(edt_found);
    // arts_wake_up_context();
    return true;
  }
  CHECK_OUTSTANDING_EDTS(10000000);
  return false;
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
      arts_server_try_to_receive(&arts_node_info.buf,
                                 &arts_node_info.packet_size,
                                 &arts_node_info.steal_request_lock);
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
