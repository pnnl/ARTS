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
/* tiered_pool.h relies on C11 _Atomic and is C-only; scheduler.c is also
 * compiled as scheduler_gpu.cu (C++) — only pull these on the C side and
 * guard the corresponding init/destroy calls below with the same macro. */
#include "arts/event.h"             /* struct arts_event_dep_s */
#include "arts/utils/tiered_pool.h" /* arts_tiered_pool_init / destroy */
#endif
#include "arts/utils/malloc.h"

#include <assert.h>
#include <stdlib.h>
#include <time.h>

#include "arts/counter/Preamble.h"
#include "arts/counter/counter.h"
#include "arts/counter/object_counter.h"
#include "arts/db.h"
#include "arts/defs.h"
#include "arts/edt.h"
#include "arts/edt_context.h" /* arts_set/unset_thread_local_edt_info */
#include "arts/gas/guid.h"
#include "arts/gas/route_table.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/system/topology.h"
#include "arts/transport/dispatcher.h"
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
bool arts_cxl_scheduler_loop(void);
#endif

extern unsigned int num_numa_domains;

static inline void arts_runtime_idle_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
  __asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ __volatile__("yield" ::: "memory");
#else
  __asm__ __volatile__("" ::: "memory");
#endif
}

ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

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
#ifdef ARTS_USE_CXL
    (scheduler_t)arts_cxl_scheduler_loop,
#endif
    (scheduler_t)arts_network_before_steal_scheduler_loop,
    (scheduler_t)arts_network_first_scheduler_loop};
#endif

/* Schedule a fully DB-acquired EDT onto a work-stealing deque.  Reached when
 * the strict sequential acquire walk in arts_db_acquire_all completes
 * (rw_cursor
 * == depc) — either synchronously (initial arts_handle_ready_edt on a worker)
 * or asynchronously (a coherence wake / OoO drain resumes the walk on a
 * receiver or drain thread).  Hence the deque[0] fallback: the completing
 * thread may have no my_deque. */
void arts_schedule_ready_edt(struct arts_edt_s *edt) {
  INCREMENT_NUM_EDT_ACQUIRE_BY(1);
#ifdef ARTS_USE_GPU
  if (arts_node_info.gpu && !arts_thread_info.my_gpu_deque &&
      edt->edt_type == ARTS_EDT_GPU) {
    /* CUDA callback thread, GPU EDT: deferred via new_edts. */
    arts_store_new_edts(edt);
  } else
#endif
      if (edt->edt_type == ARTS_EDT_GPU) {
    struct arts_deque_s *q = arts_thread_info.my_gpu_deque
                                 ? arts_thread_info.my_gpu_deque
                                 : arts_node_info.gpu_deque[0];
    ARTS_INFO("EDT[Guid:%lu] pushed to GPU deque", edt->guid);
    arts_deque_push_front(q, edt, 0);
  } else {
    struct arts_deque_s *q = arts_thread_info.my_deque
                                 ? arts_thread_info.my_deque
                                 : arts_node_info.deque[0];
    ARTS_INFO("EDT[Guid:%lu] pushed to worker deque", edt->guid);
    arts_deque_push_front(q, edt, 0);
  }
}

/*
 * arts_handle_ready_edt — Transition an EDT from "all deps signaled" to DB
 * acquisition.  Called when depc_needed reaches 0 (the satisfy phase
 * completes).  Seeds the two acquire counters (rw_cursor = 0, the readiness
 * counter acquire_remaining = 1 as a +1 bias) and enters arts_db_acquire_all,
 * which fires all non-serialized (RO) deps at once and walks the serialized
 * (RW) deps by GUID-ordered cursor.  The EDT is scheduled (via
 * arts_schedule_ready_edt) when acquire_remaining reaches 0 — i.e. when every
 * DB dep's data has resolved at this rank.
 */
void arts_handle_ready_edt(struct arts_edt_s *edt) {
  ARTS_INFO("EDT[Guid:%lu, Id:%lu] ready — entering sequential DB acquire "
            "(depc=%u)",
            edt->guid, edt->arts_id, edt->depc);
#ifdef ARTS_USE_CXL
  if (arts_node_info.scheduler == (void *)arts_cxl_scheduler_loop &&
      arts_deque_full(arts_thread_info.my_deque)) {
    while (!arts_cxl_deque_push(arts_node_info.cxl_deque,
                                &arts_node_info.cxl_local_lock,
                                arts_edt_total_size(edt), edt)) {
    }
    return;
  }
#endif
  edt->rw_cursor = 0;
  edt->acquire_remaining =
      1; /* +1 bias; arts_db_acquire_all adds the dep count */
  arts_db_acquire_all(edt);
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
            edt->arts_id, edt->guid, depc, paramc, depv);
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

  /* Release DBs before signaling finish-event completion: any RC writeback
   * messages (WRITEBACK) are queued to the sender thread before the finish
   * DECR message, so TCP FIFO ordering guarantees data arrives at home first.
   */
  release_dbs(depc, depv, false);
  arts_release_created_dbs();

  arts_unset_thread_local_edt_info();

  ARTS_INFO("EDT[Guid:%lu, Id:%lu] finished (exec_ns=%lu)", edt->guid,
            edt->arts_id, exec_ns);
#ifdef ARTS_USE_CXL
  if (!IS_CXL_PTR(edt)) {
    arts_edt_delete(edt);
  }
#else
  arts_edt_delete(edt);
#endif
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
              arts_thread_info.my_deque))) {
      edt_found = arts_runtime_steal_from_worker();
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
            arts_thread_info.my_deque))) {
    if (!(edt_found = arts_runtime_steal_from_network())) {
      edt_found = arts_runtime_steal_from_worker();
    }
  }

  if (edt_found) {
    arts_run_edt(edt_found);
    return true;
  }
  return false;
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
  arts_runtime_idle_pause();
  return false;
}

#ifdef ARTS_USE_CXL
bool arts_cxl_scheduler_loop() {
  struct arts_edt_s *edt_found = NULL;
  if (!(edt_found = (struct arts_edt_s *)arts_deque_pop_front(
            arts_thread_info.my_deque))) {
    if (!(edt_found = arts_runtime_steal_from_worker())) {
      arts_cxl_deque_pop(arts_node_info.cxl_deque,
                         &arts_node_info.cxl_local_lock, (void **)&edt_found);
    }
  }
  if (edt_found) {
    arts_run_edt(edt_found);
    return true;
  }
  CHECK_OUTSTANDING_EDTS(10000000);
  arts_runtime_idle_pause();
  return false;
}
#endif /* ARTS_USE_CXL */
