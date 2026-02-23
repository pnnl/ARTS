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
#ifndef ARTS_RUNTIME_SYNC_RUNTIME_H
#define ARTS_RUNTIME_SYNC_RUNTIME_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts/counter/counter.h"
#include "arts/counter/object_counter.h"
#include "arts/defs.h"
#include "arts/runtime/rt.h"
#include "arts/system/abstract_machine_model.h"

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
  arts_guid_t auto_shutdown_guid;
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
  struct arts_deque_s *my_node_deque;
  struct arts_deque_s *my_gpu_deque;
  unsigned int pu_id;
  unsigned int thread_id;
  unsigned int group_pos;
  unsigned int numa_domain_id;
  unsigned int back_off;
  volatile unsigned int outstanding_memory_moves;
  struct atomic_create_barrier_info_s atomic_wait;
  volatile bool alive;
  enum arts_thread_role role;
  arts_guid_t current_edt_guid;
  int edt_free;
  int local_counting;
  unsigned int shad_lock;
  unsigned short drand_buf[3];
};

extern struct arts_runtime_shared_s arts_node_info;
extern ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

#define ARTS_LOOK_UP_CONFIG(name) arts_node_info.name

void arts_runtime_node_init(struct arts_config_s *config);
void arts_runtime_global_cleanup();
void arts_runtime_private_cleanup();
void arts_runtime_stop();
void arts_handle_ready_edt(struct arts_edt_s *edt);
void arts_rehandle_ready_edt(struct arts_edt_s *edt);
void arts_run_edt(struct arts_edt_s *edt);
void arts_handle_remote_stolen_edt(struct arts_edt_s *edt);
bool arts_runtime_scheduler_loop();
void arts_thread_zero_node_start(int argc, char **argv);
void arts_runtime_private_init(struct thread_mask_s *thread,
                               struct arts_config_s *config);
int arts_runtime_loop();
int arts_runtime_scheduler_loop_wait(volatile bool *wait_for_me);
bool arts_default_scheduler_loop();
struct arts_edt_s *arts_find_edt();

bool arts_runtime_edt_lock_db(arts_guid_t db_guid, struct arts_db_s *db,
                              void *edt_packet, bool shared);
void arts_runtime_edt_lock_db_signal_next(struct arts_db_s *db,
                                          arts_guid_t db_guid, bool remote);
struct arts_edt_s *arts_runtime_steal_from_worker();
struct arts_edt_s *arts_runtime_steal_from_network();
void arts_db_unlock(struct arts_db_s *db, arts_guid_t db_guid, bool write);
bool arts_db_lock_all_dbs(struct arts_edt_s *edt);
bool arts_db_lock(arts_guid_t db_guid, void *edt_packet, unsigned int rank,
                  bool shared);

bool arts_network_first_scheduler_loop();
bool arts_network_before_steal_scheduler_loop();
bool arts_gpu_scheduler_backoff_loop();
bool arts_gpu_scheduler_demand_loop();

#ifdef __cplusplus
}
#endif

#endif
