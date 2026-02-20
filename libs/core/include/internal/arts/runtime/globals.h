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
#ifndef ARTS_RUNTIME_GLOBALS_H
#define ARTS_RUNTIME_GLOBALS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/introspection/arts_id_counter.h"
#include "arts/introspection/counter.h"
#include "arts/runtime/rt.h"

struct atomic_create_barrier_info_s {
  volatile unsigned int wait;
  volatile unsigned int result;
};

struct arts_runtime_shared_s {
  volatile unsigned int send_lock;
  char pad1[56];
  volatile unsigned int recv_lock;
  char pad2[56];
  volatile unsigned int steal_request_lock;
  char pad3[56];
  bool (*scheduler)();
  struct arts_deque_s **deque;
  struct arts_deque_s **receiver_deque;
  struct arts_deque_s **gpu_deque;
  struct arts_route_table_s **route_table;
  struct arts_route_table_s **gpu_route_table;
  struct arts_route_table_s *remote_route_table;
  volatile bool **local_spin;
  unsigned int **memory_moves;
  struct atomic_create_barrier_info_s **atomic_waits;
  unsigned int worker_thread_count;
  unsigned int sender_thread_count;
  unsigned int receiver_thread_count;
  unsigned int remote_stealing_thread_count;
  unsigned int total_thread_count;
  volatile unsigned int ready_to_push;
  volatile unsigned int ready_to_parallel_start;
  volatile unsigned int ready_to_inspect;
  volatile unsigned int ready_to_execute;
  volatile unsigned int ready_to_clean;
  volatile unsigned int ready_to_shutdown;
  char *buf;
  int packet_size;
  bool shutdown_started;
  volatile unsigned int shutdown_count;
  uint64_t shutdown_timeout;
  uint64_t shutdown_force_timeout;
  unsigned int print_node_stats;
  arts_guid_t shutdown_epoch;
  unsigned int shad_loop_stride;
  bool tmt;
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
  unsigned int pin_threads;
  uint64_t **keys;
  uint64_t *global_guid_thread_id;
  const char *counter_folder;
  arts_counter_t **live_counters;           // [thread_id] -> pointer to thread's __thread counters
  arts_counter_t **saved_counters;          // [thread_id][counter_index] - final counter values
  arts_array_list_t ***capture_arrays;       // [thread_id][counter_index] - capture history (PERIODIC)
  uint64_t counter_capture_interval;
  // arts_id reduced metrics computed at output time (not stored during runtime)
} __attribute__((aligned(64)));

struct arts_runtime_private_s {
  struct arts_deque_s *my_deque;
  struct arts_deque_s *my_node_deque;
  struct arts_deque_s *my_gpu_deque;
  unsigned int core_id;
  unsigned int thread_id;
  unsigned int group_id;
  unsigned int numa_domain_id;
  unsigned int back_off;
  volatile unsigned int outstanding_memory_moves;
  struct atomic_create_barrier_info_s atomic_wait;
  volatile bool alive;
  volatile bool worker;
  volatile bool network_send;
  volatile bool network_receive;
  volatile bool status_send;
  arts_guid_t current_edt_guid;
  int malloc_type;
  int malloc_trace;
  int edt_free;
  int local_counting;
  unsigned int shad_lock;
  unsigned short drand_buf[3];
  // Thread's counter storage accessed via artsThreadLocalCounterCaptures
};

extern struct arts_runtime_shared_s arts_node_info;
extern __thread struct arts_runtime_private_s arts_thread_info;

extern unsigned int arts_global_rank_id;
extern unsigned int arts_global_rank_count;
extern unsigned int arts_global_master_rank_id;
extern bool arts_global_i_will_print;
extern uint64_t arts_guid_min;
extern uint64_t arts_guid_max;

#define MASTER_PRINTF(...)                                                     \
  do {                                                                         \
    if (arts_global_rank_id == arts_global_master_rank_id)                      \
      arts_printf(__VA_ARGS__);                                                \
  } while (0)
#define ONCE_PRINTF(...)                                                       \
  do {                                                                         \
    if (arts_global_i_will_print == true)                                       \
      arts_printf(__VA_ARGS__);                                                \
  } while (0)

#define ARTS_LOOK_UP_CONFIG(name) arts_node_info.name

#define ARTS_TYPE_NAME                                                           \
  const char *const arts_type_name[] = {"ARTS_NULL",                            \
                                       "ARTS_EDT",                             \
                                       "ARTS_GPU_EDT",                         \
                                       "ARTS_EVENT",                           \
                                       "ARTS_PERSISTENT_EVENT",                \
                                       "ARTS_EPOCH",                           \
                                       "ARTS_CALLBACK",                        \
                                       "ARTS_BUFFER",                          \
                                       "ARTS_DB",                              \
                                       "ARTS_DB_READ",                         \
                                       "ARTS_DB_WRITE",                        \
                                       "ARTS_DB_PIN",                          \
                                       "ARTS_DB_ONCE",                         \
                                       "ARTS_DB_ONCE_LOCAL",                   \
                                       "ARTS_DB_GPU_READ",                     \
                                       "ARTS_DB_GPU_WRITE",                    \
                                       "ARTS_DB_LC",                           \
                                       "ARTS_LAST_TYPE",                       \
                                       "ARTS_SINGLE_VALUE",                    \
                                       "ARTS_PTR",                             \
                                       "ARTS_DB_LC_SYNC",                      \
                                       "ARTS_DB_LC_NO_COPY",                   \
                                       "ARTS_DB_GPU_MEMSET"}

#define GET_TYPE_NAME(x) arts_type_name[x]

extern const char *const arts_type_name[];

extern volatile uint64_t outstanding_edts;
void check_out_edts(uint64_t threshold);

// #ifdef CHECK_NO_EDT
#define INC_OUTSTANDING_EDTS(num_edts)                                            \
  arts_atomic_fetch_add_u64(&outstanding_edts, num_edts)
#define DEC_OUTSTANDING_EDTS(num_edts)                                            \
  arts_atomic_fetch_sub_u64(&outstanding_edts, num_edts)
#define CHECK_OUTSTANDING_EDTS(threshold) check_out_edts(threshold)
// #else
// #define INC_OUTSTANDING_EDTS(num_edts)
// #define DEC_OUTSTANDING_EDTS(num_edts)
// #define CHECK_OUTSTANDING_EDTS(threshold)
// #endif

#ifdef __cplusplus
}
#endif

#endif
