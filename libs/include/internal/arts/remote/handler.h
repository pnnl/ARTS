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
#ifndef ARTS_REMOTE_HANDLER_H
#define ARTS_REMOTE_HANDLER_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/runtime_types.h"
#include "arts/transport/protocol.h"

void arts_send_event_add_dependence(arts_guid_t source, arts_guid_t destination,
                                    uint32_t slot, unsigned int rank,
                                    arts_db_access_mode_t mode);

void arts_send_memory_move(unsigned int rank, arts_guid_t guid, void *ptr,
                           unsigned int mem_size, unsigned message_type,
                           void (*free_method)(void *));
void arts_handler_edt_create(void *ptr);
void arts_handler_event_create(void *ptr);
void arts_send_edt_satisfy_slot(arts_guid_t edt, arts_guid_t db, uint32_t slot,
                                arts_db_access_mode_t mode, void *ptr,
                                unsigned int size);
void arts_send_event_satisfy_slot(arts_guid_t event_guid, arts_guid_t data_guid,
                                  uint32_t slot);
/* Cross-rank arts_event_destroy: forwarder + handler.
 * Forwarder serializes the GUID into MSG_EVENT_DESTROY;
 * handler runs arts_route_table_mark_delete on the home rank.  mark_delete
 * is idempotent (DELETE is sticky), so duplicate messages are safe. */
void arts_send_event_destroy(arts_guid_t guid);
void arts_handler_event_destroy(void *ptr);
void arts_db_request_callback(struct arts_edt_s *edt, unsigned int slot,
                              struct arts_db_s *db_res);
void arts_send_epoch_create(unsigned int rank, arts_guid_t epoch_guid,
                            arts_guid_t edt_guid, unsigned int slot);
void arts_handler_epoch_create(void *pack);
void arts_send_epoch_init_pool(unsigned int rank, unsigned int pool_size,
                               arts_guid_t start_guid, arts_guid_t pool_guid);
void arts_handler_epoch_init_pool(void *pack);
/* Wire TX for the epoch reduction protocol.  RX is inline-decoded in the
 * dispatcher, which calls the entries send_epoch / reduce_epoch (sync/epoch.h);
 * those route through arts_ooo_dispatch_or_defer_guid to the pure cores
 * arts_handler_epoch_request / arts_handler_epoch_send. */
void arts_send_epoch_request(unsigned int rank, arts_guid_t guid);
void arts_send_epoch_send(unsigned int rank, arts_guid_t guid,
                          unsigned int active, unsigned int finish);
void arts_send_epoch_delete(unsigned int rank, arts_guid_t epoch_guid);
void arts_handler_epoch_delete(void *pack);

// RTT-based time synchronization for precise epoch alignment
// Worker initiates sync request, master responds, worker calculates offset
void arts_send_time_sync_request(void); // Worker sends request to master
void arts_handler_time_sync_request(void *pack);  // Master handles request
void arts_handler_time_sync_response(void *pack); // Worker handles response

#ifdef __cplusplus
}
#endif

#endif
