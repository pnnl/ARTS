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
#ifndef ARTS_GAS_OUT_OF_ORDER_H
#define ARTS_GAS_OUT_OF_ORDER_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/gas/out_of_order_list.h"
#include "arts/runtime_types.h"

enum arts_out_of_order_type {
  OO_SIGNAL_EDT,
  OO_EVENT_SATISFY_SLOT,
  OO_ADD_DEPENDENCE,
  OO_HANDLE_READY_EDT,
  OO_REMOTE_DB_SEND,
  OO_DB_REQUEST_SATISFY,
  OO_DB_FULL_SEND,
  OO_GET_FROM_DB,
  OO_SIGNAL_EDT_PTR,
  OO_PUT_IN_DB,
  OO_EPOCH_ACTIVE,
  OO_EPOCH_FINISH,
  OO_EPOCH_SEND,
  OO_EPOCH_INC_QUEUE
};

void arts_out_of_order_signal_edt(arts_guid_t wait_on, arts_guid_t edt_packet,
                                  arts_guid_t data_guid, uint32_t slot,
                                  arts_db_access_mode_t mode, bool force);
void arts_out_of_order_event_satisfy(arts_guid_t wait_on,
                                     arts_guid_t event_guid,
                                     arts_guid_t data_guid);
void arts_out_of_order_event_satisfy_slot(arts_guid_t wait_on,
                                          arts_guid_t event_guid,
                                          arts_guid_t data_guid, uint32_t slot,
                                          bool force);
void arts_out_of_order_add_dependence(arts_guid_t source,
                                      arts_guid_t destination, uint32_t slot,
                                      arts_db_access_mode_t mode,
                                      arts_guid_t wait_on);
void arts_out_of_order_handle_ready_edt(arts_guid_t trigger_guid,
                                        struct arts_edt_s *edt);
void arts_out_of_order_handle_remote_db_send(int rank, arts_guid_t db_guid,
                                             arts_db_access_mode_t mode);
void arts_out_of_order_handle_db_request_with_oo_list(
    struct arts_out_of_order_list_s *add_to_me, void **data,
    struct arts_edt_s *edt, unsigned int slot);
void arts_out_of_order_handle_db_request(arts_guid_t db_guid,
                                         struct arts_edt_s *edt,
                                         unsigned int slot, bool inc);
void arts_out_of_order_handle_remote_db_full_send(arts_guid_t db_guid, int rank,
                                                  arts_guid_t edt_guid,
                                                  unsigned int slot,
                                                  arts_db_access_mode_t mode);
void arts_out_of_order_get_from_db(arts_guid_t edt_guid, arts_guid_t db_guid,
                                   unsigned int slot, unsigned int offset,
                                   unsigned int size);
void arts_out_of_order_signal_edt_with_ptr(arts_guid_t edt_guid,
                                           arts_guid_t db_guid, void *ptr,
                                           unsigned int size,
                                           unsigned int slot);
void arts_out_of_order_put_in_db(void *ptr, arts_guid_t edt_guid,
                                 arts_guid_t db_guid, unsigned int slot,
                                 unsigned int offset, unsigned int size,
                                 arts_guid_t epoch_guid);
void arts_out_of_order_inc_active_epoch(arts_guid_t epoch_guid);
void arts_out_of_order_inc_finished_epoch(arts_guid_t epoch_guid);
void arts_out_of_order_send_epoch(arts_guid_t epoch_guid, unsigned int source,
                                  unsigned int dest);
void arts_out_of_order_inc_queue_epoch(arts_guid_t epoch_guid);

void arts_out_of_order_handler(void *handle_me, void *memory_ptr);

#ifdef __cplusplus
}
#endif

#endif
