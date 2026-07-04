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
#ifndef ARTS_RUNTIME_STATE_H
#define ARTS_RUNTIME_STATE_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts/counter/counter.h"
#include "arts/counter/object_counter.h"
#include "arts/defs.h"
#include "arts/gas/route_table.h"
#include "arts/runtime_types.h"
#include "arts/system/topology.h"
#ifdef ARTS_USE_CXL
#include "arts/cxl/deque.h"
#include <pthread.h>
#endif

struct arts_runtime_shared_s {
  volatile unsigned int steal_request_lock;
  char pad3[56];
  bool (*scheduler)();
  struct arts_deque_s **deque;
  struct arts_deque_s **progress_deque;
  struct arts_deque_s **gpu_deque;
#ifdef ARTS_USE_CXL
  arts_cxl_deque_t *cxl_deque;
  pthread_mutex_t cxl_local_lock;
  void *cxl_db_arena_start;
  void *cxl_db_arena_end;
  unsigned int cxl_db_dev_count; /**< Number of CXL DB arenas (devices). */
  volatile unsigned int cxl_db_rr_idx; /**< Round-robin device index counter. */
  unsigned int cxl_db_static_device; /**< Device index for static allocation. */
#endif
  struct arts_route_table_s **route_table;
  struct arts_route_table_s **gpu_route_table;
  struct arts_route_table_s *remote_route_table[ARTS_REMOTE_ROUTE_SHARDS];
  volatile bool **local_spin;
  /* Per-thread role, indexed by thread_id (0..total_thread_count-1).
   * Populated during arts_runtime_private_init so that shutdown paths
   * can filter threads by role (workers vs. network). */
  unsigned int *thread_roles;
  unsigned int worker_thread_count;
  unsigned int progress_thread_count;
  unsigned int total_thread_count;
  volatile unsigned int ready_to_push;
  volatile unsigned int ready_to_parallel_start;
  volatile unsigned int ready_to_inspect;
  volatile unsigned int ready_to_execute;
  volatile unsigned int ready_to_clean;
  /* Global shutdown flag. 0 = running normally, 1 = shutting down.
   * Set by arts_runtime_stop() and checked by long-running loops
   * (e.g. arts_transport_connect retry) so they can bail out promptly. */
  volatile unsigned int shutdown_state;
  /* End-to-end wall-clock marker (rank 0 only), independent of the counter
   * subsystem.  e2e_start_stamp is taken when the main application EDT
   * becomes runnable (init complete — so application init callbacks run
   * inside the span); e2e_end_stamp is taken at shutdown recognition (the
   * enter-shutdown CAS, before any teardown/drain).  When e2e_marker_enabled
   * (set once from $ARTS_E2E_MARKER at startup) their difference is printed
   * as "[E2E] <ns>" on rank 0 at exit.  Gated so a normal run is never
   * perturbed. */
  uint64_t e2e_start_stamp;
  uint64_t e2e_end_stamp;
  int e2e_marker_enabled;
  /* Round-robin counter used by arts_db_create when the caller passes
   * NULL hint (no node preference) — distributes home rank across all
   * nodes so DBs aren't all pinned to the creator. */
  volatile unsigned int db_rr_route;
  /* Round-robin counter used by arts_edt_create when the caller passes
   * NULL hint (no node preference) — distributes execution rank across
   * all nodes so hint-less work isn't all pinned to the creator. */
  volatile unsigned int edt_rr_route;
  char *buf;
  int packet_size;
  unsigned int gpu;
  unsigned int gpu_locality;
  unsigned int gpu_fit;
  unsigned int gpu_lc_sync;
  unsigned int gpu_max_edts;
  unsigned int gpu_p2p;
  unsigned int gpu_route_table_size;
  unsigned int gpu_route_table_entries;
  uint64_t gpu_max_memory;
  bool free_db_after_gpu_run;
  bool run_gpu_gc_idle;
  bool run_gpu_gc_pre_edt;
  bool delete_zeros_gpu_gc;
  bool gpu_buff_on;
  uint64_t **keys;
  uint64_t *global_guid_thread_id;
  const char *counter_folder;
  arts_counter_t *
      *live_counters; // [thread_id] -> pointer to thread's __thread counters
  arts_counter_t *
      *saved_counters; // [thread_id][counter_index] - final counter values
  arts_array_list_t ***capture_arrays; // [thread_id][counter_index] - capture
                                       // history (PERIODIC)
  uint64_t counter_capture_interval;
  // Object counter storage (per-thread saved data for per-arts_id tracking)
  arts_object_table_t **object_tables;   // [thread_id]
  arts_array_list_t **object_edt_traces; // [thread_id]
  arts_array_list_t **object_db_traces;  // [thread_id]
} ARTS_ALIGNED(64);

struct arts_runtime_private_s {
  struct arts_deque_s *my_deque;
  struct arts_deque_s *my_gpu_deque;
  unsigned int pu_id;
  unsigned int thread_id;
  unsigned int group_pos;
  unsigned int numa_domain_id;
  volatile bool alive;
  enum arts_thread_role role;
  arts_guid_t current_edt_guid;
  int edt_free;
  unsigned short drand_buf[3];
};

extern struct arts_runtime_shared_s arts_node_info;
extern ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

#define ARTS_LOOK_UP_CONFIG(name) arts_node_info.name

#ifdef __cplusplus
}
#endif

/* Back-compat: the runtime function prototypes (and the scheduler_t typedef /
 * scheduler_loop extern) used to live here.  They now live in arts/runtime.h.
 * Include it at the end so every existing runtime_state.h includer still
 * transitively sees them without source churn.  This header must be included
 * AFTER the struct layouts above so the include direction stays one-way
 * (runtime_state.h -> runtime.h); runtime.h itself only forward-declares the
 * struct types it references, never including this header back. */
#include "arts/runtime.h"

#endif
