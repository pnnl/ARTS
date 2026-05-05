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

void arts_remote_send(unsigned int rank, send_handler_t fun_ptr, void *args,
                      unsigned int size, bool free);

void arts_remote_add_dependence(arts_guid_t source, arts_guid_t destination,
                                uint32_t slot, unsigned int rank,
                                arts_db_access_mode_t mode);
void arts_remote_add_dependence_with_hints(arts_guid_t source,
                                           arts_guid_t destination,
                                           uint32_t slot, unsigned int rank,
                                           arts_db_access_mode_t mode);

void arts_remote_memory_move(unsigned int route, arts_guid_t guid, void *ptr,
                             unsigned int mem_size, unsigned message_type,
                             void (*free_method)(void *));
void arts_remote_memory_move_no_free(unsigned int route, arts_guid_t guid,
                                     void *ptr, unsigned int mem_size,
                                     unsigned message_type);
void arts_remote_handle_edt_move(void *ptr);
void arts_remote_handle_db_move(void *ptr);
void arts_remote_handle_event_move(void *ptr);
void arts_remote_signal_edt(arts_guid_t edt, arts_guid_t db, uint32_t slot,
                            arts_db_access_mode_t mode);
void arts_remote_set_dep_mode(arts_guid_t edt_guid, uint32_t slot,
                              arts_db_access_mode_t mode);
void arts_remote_event_satisfy_slot(arts_guid_t event_guid,
                                    arts_guid_t data_guid, uint32_t slot);
void arts_db_request_callback(struct arts_edt_s *edt, unsigned int slot,
                              struct arts_db_s *db_res);
void arts_remote_get_from_db(arts_guid_t edt_guid, arts_guid_t db_guid,
                             unsigned int slot, unsigned int offset,
                             unsigned int len, unsigned int rank);
void arts_remote_handle_get_from_db(void *pack);
void arts_remote_put_in_db(void *ptr, arts_guid_t edt_guid, arts_guid_t db_guid,
                           unsigned int slot, unsigned int offset,
                           unsigned int len, arts_guid_t epoch_guid,
                           unsigned int rank);
void arts_remote_handle_put_in_db(void *pack);
void arts_remote_signal_edt_with_ptr(arts_guid_t edt_guid, arts_guid_t db_guid,
                                     void *ptr, unsigned int size,
                                     unsigned int slot);
void arts_remote_handle_signal_edt_with_ptr(void *pack);
void arts_remote_handle_send(void *pack);
void arts_remote_epoch_init_send(unsigned int rank, arts_guid_t epoch_guid,
                                 arts_guid_t edt_guid, unsigned int slot);
void arts_remote_handle_epoch_init_send(void *pack);
void arts_remote_epoch_init_pool_send(unsigned int rank, unsigned int pool_size,
                                      arts_guid_t start_guid,
                                      arts_guid_t pool_guid);
void arts_remote_handle_epoch_init_pool_send(void *pack);
void arts_remote_epoch_req(unsigned int rank, arts_guid_t guid);
void arts_remote_handle_epoch_req(void *pack);
void arts_remote_epoch_send(unsigned int rank, arts_guid_t guid,
                            unsigned int active, unsigned int finish);
void arts_remote_handle_epoch_send(void *pack);
void arts_remote_epoch_delete(unsigned int rank, arts_guid_t epoch_guid);
void arts_remote_handle_epoch_delete(void *pack);
void arts_remote_handle_buffer_send(void *pack);
void arts_remote_db_rename(arts_guid_t new_guid, arts_guid_t old_guid);
void arts_remote_handle_db_rename(void *pack);

// RTT-based time synchronization for precise epoch alignment
// Worker initiates sync request, master responds, worker calculates offset
void arts_remote_time_sync_request(void); // Worker sends request to master
void arts_remote_handle_time_sync_req(void *pack);  // Master handles request
void arts_remote_handle_time_sync_resp(void *pack); // Worker handles response

#ifdef __cplusplus
}
#endif

#endif
