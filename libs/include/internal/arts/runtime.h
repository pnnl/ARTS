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
#ifndef ARTS_RUNTIME_H
#define ARTS_RUNTIME_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

/* Forward declarations of the types referenced (by pointer) in the runtime
 * function signatures below.  The concrete struct layouts live in
 * arts/runtime_state.h; the runtime prototypes only ever take/return pointers
 * to these, so forward declarations suffice and avoid a circular include
 * (arts/runtime_state.h includes this header for back-compat). */
struct arts_config_s;
struct thread_mask_s;
struct arts_edt_s;

void arts_runtime_node_init(struct arts_config_s *config);
void arts_runtime_global_cleanup();
void arts_runtime_private_cleanup();
void arts_runtime_stop();
void arts_runtime_stop_workers();
void arts_runtime_stop_network();
void arts_handle_ready_edt(struct arts_edt_s *edt);
void arts_run_edt(struct arts_edt_s *edt);
/* Schedule a fully DB-acquired EDT onto a work-stealing deque (deque[0]
 * fallback for non-worker completing threads).  Reached when the sequential
 * acquire walk in arts_db_acquire_all completes (resume_k == depc). */
void arts_schedule_ready_edt(struct arts_edt_s *edt);
void arts_thread_zero_node_start(int argc, char **argv);
void arts_runtime_private_init(struct thread_mask_s *thread,
                               struct arts_config_s *config);
int arts_runtime_loop();
bool arts_default_scheduler_loop();

struct arts_edt_s *arts_runtime_steal_from_worker();
struct arts_edt_s *arts_runtime_steal_from_network();

bool arts_network_first_scheduler_loop();
bool arts_network_before_steal_scheduler_loop();
bool arts_gpu_scheduler_backoff_loop();
bool arts_gpu_scheduler_demand_loop();

/* Scheduler-loop dispatch table. Defined alongside the scheduler loop bodies
 * (their definitions must share that translation unit). The runtime bring-up
 * code selects an entry by config index, so the table is declared here for
 * cross-module visibility. */
typedef bool (*scheduler_t)(void);
extern scheduler_t scheduler_loop[];

#ifdef __cplusplus
}
#endif

#endif
