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

#ifndef ARTS_TERMINATION_DETECTION_H
#define ARTS_TERMINATION_DETECTION_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/runtime/rt.h"

arts_epoch_t *create_epoch(arts_guid_t *guid, arts_guid_t edt_guid,
                           unsigned int slot);
void increment_queue_epoch(arts_guid_t epoch_guid);
void increment_active_epoch(arts_guid_t epoch_guid);
void increment_finished_epoch(arts_guid_t epoch_guid);
void send_epoch(arts_guid_t epoch_guid, unsigned int source, unsigned int dest);
void broadcast_epoch_request(arts_guid_t epoch_guid);
bool check_epoch(arts_epoch_t *epoch, unsigned int total_active,
                 unsigned int total_finish);
void reduce_epoch(arts_guid_t epoch_guid, unsigned int active,
                  unsigned int finish);
void delete_epoch(arts_guid_t epoch_guid, arts_epoch_t *epoch);

typedef struct arts_epoch_pool_s {
  struct arts_epoch_pool_s *next;
  unsigned int size;
  unsigned int index;
  volatile unsigned int outstanding;
  arts_epoch_t pool[];
} arts_epoch_pool_t;

arts_epoch_pool_t *create_epoch_pool(arts_guid_t *epoch_pool_guid,
                                     unsigned int pool_size,
                                     arts_guid_t *start_guid);
void arts_link_epoch_pool_to_tls(arts_epoch_pool_t *pool);
arts_epoch_t *get_pool_epoch(arts_guid_t edt_guid, unsigned int slot);
void arts_cleanup_epoch_pools(void);

void arts_shutdown_epoch_inc_active();
void arts_shutdown_epoch_inc_queue();
void arts_shutdown_epoch_inc_finished();
bool arts_shutdown_epoch_create();

#ifdef __cplusplus
}
#endif
#endif
