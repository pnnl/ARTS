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
#include "arts/runtime/rt.h"
#include "arts/system/abstract_machine_model.h"

#define NODEDEQUESIZE 8

enum artsInitType {
  ARTS_WORKER_THREAD,
  ARTS_RECEIVER_THREAD,
  ARTS_REMOTE_STEAL_THREAD,
  ARTS_COUNTER_THREAD,
  ARTS_OTHER_THREAD
};

void arts_runtime_node_init(unsigned int worker_threads,
                         unsigned int receiving_threads,
                         unsigned int sender_threads,
                         unsigned int receiver_threads,
                         unsigned int total_threads, bool remote_stealing_on,
                         struct arts_config_s *config);
void arts_runtime_global_cleanup();
void arts_runtime_private_cleanup();
void arts_runtime_stop();
void arts_handle_ready_edt(struct arts_edt_s *edt);
void arts_rehandle_ready_edt(struct arts_edt_s *edt);
void arts_run_edt(struct arts_edt_s *edt);
void arts_handle_remote_stolen_edt(struct arts_edt_s *edt);
bool arts_runtime_scheduler_loop();
void arts_thread_zero_node_start();
void arts_thread_zero_private_init(struct thread_mask_s *unit,
                               struct arts_config_s *config);
void arts_runtime_private_init(struct thread_mask_s *unit, struct arts_config_s *config);
int arts_runtime_loop();
int arts_runtime_scheduler_loop_wait(volatile bool *wait_for_me);
bool arts_default_scheduler_loop();
struct arts_edt_s *arts_find_edt();

bool arts_runtime_edt_lock_db(arts_guid_t db_guid, struct arts_db_s *db, void *edt_packet,
                          bool shared);
void arts_runtime_edt_lock_db_signal_next(struct arts_db_s *db, arts_guid_t db_guid,
                                    bool remote);
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
