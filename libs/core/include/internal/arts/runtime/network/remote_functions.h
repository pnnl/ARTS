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
#ifndef ARTS_RUNTIME_NETWORK_REMOTE_FUNCTIONS_H
#define ARTS_RUNTIME_NETWORK_REMOTE_FUNCTIONS_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/network/remote_protocol.h"
#include "arts/runtime/rt.h"

void arts_remote_send(unsigned int rank, send_handler_t fun_ptr, void *args,
                      unsigned int size, bool free);

void arts_remote_add_dependence(arts_guid_t source, arts_guid_t destination,
                                uint32_t slot, unsigned int rank);
void arts_remote_add_dependence_with_hints(arts_guid_t source,
                                           arts_guid_t destination,
                                           uint32_t slot, unsigned int rank,
                                           arts_db_mode_t mode);
void arts_remote_add_dependence_to_persistent_event(arts_guid_t source,
                                                    arts_guid_t destination,
                                                    uint32_t slot,
                                                    unsigned int rank);
void arts_remote_add_dependence_to_persistent_event_with_hints(
    arts_guid_t source, arts_guid_t destination, uint32_t slot,
    unsigned int rank, arts_db_mode_t mode);
void arts_remote_add_dependence_to_persistent_event_with_byte_offset(
    arts_guid_t source, arts_guid_t destination, uint32_t slot,
    unsigned int rank, arts_db_mode_t mode, uint64_t byte_offset, uint64_t len);
void arts_remote_update_route_table(arts_guid_t guid, unsigned int rank);
void arts_remote_handle_update_db_guid(void *ptr);
void arts_remote_handle_invalidate_db(void *ptr);
void arts_remote_db_destroy(arts_guid_t guid, unsigned int origin_rank,
                            bool clean);
void arts_remote_handle_db_destroy_forward(void *ptr);
void arts_remote_handle_db_clean_forward(void *ptr);
void arts_remote_handle_db_destroy(void *ptr);
void arts_remote_update_db(arts_guid_t guid, bool send_db);
void arts_remote_handle_update_db(void *ptr);

struct artsDiffList;
void arts_remote_partial_update_db(arts_guid_t guid, struct artsDiffList *diffs,
                                   void *working);
void arts_remote_handle_partial_update(void *ptr);

void arts_remote_memory_move(unsigned int route, arts_guid_t guid, void *ptr,
                             unsigned int mem_size, unsigned message_type,
                             void (*free_method)(void *));
void arts_remote_memory_move_no_free(unsigned int route, arts_guid_t guid,
                                     void *ptr, unsigned int mem_size,
                                     unsigned message_type);
void arts_remote_handle_edt_move(void *ptr);
void arts_remote_handle_db_move(void *ptr);
void arts_remote_handle_event_move(void *ptr);
void arts_remote_handle_persistent_event_move(void *ptr);
void arts_remote_signal_edt(arts_guid_t edt, arts_guid_t db, uint32_t slot,
                            arts_db_mode_t mode);
void arts_remote_event_satisfy_slot(arts_guid_t event_guid,
                                    arts_guid_t data_guid, uint32_t slot);
void arts_remote_persistent_event_satisfy_slot(arts_guid_t event_guid,
                                               uint32_t action, bool lock);
void arts_remote_db_add_dependence(arts_guid_t db_src, arts_guid_t edt_dest,
                                   uint32_t edt_slot);
void arts_remote_db_add_dependence_with_hints(arts_guid_t db_src,
                                              arts_guid_t edt_dest,
                                              uint32_t edt_slot,
                                              arts_db_mode_t mode);
void arts_remote_db_add_dependence_with_byte_offset(
    arts_guid_t db_src, arts_guid_t edt_dest, uint32_t edt_slot,
    arts_db_mode_t mode, uint64_t byte_offset, uint64_t len);
void arts_remote_handle_db_add_dependence_with_byte_offset(void *ptr);
void arts_remote_db_increment_latch(arts_guid_t db);
void arts_remote_db_decrement_latch(arts_guid_t db);
void arts_db_request_callback(struct arts_edt_s *edt, unsigned int slot,
                              struct arts_db_s *db_res);
bool arts_remote_db_request(arts_guid_t data_guid, int rank,
                            struct arts_edt_s *edt, int pos,
                            arts_db_mode_t mode, bool agg_request);
void arts_remote_db_forward(int dest_rank, int source_rank,
                            arts_guid_t data_guid, arts_db_mode_t mode);
void arts_remote_db_send_now(int rank, struct arts_db_s *db);
void arts_remote_db_send_check(int rank, struct arts_db_s *db,
                               arts_db_mode_t mode);
void arts_remote_db_send(struct arts_remote_db_request_packet_s *pack);
void arts_remote_handle_db_received(
    struct arts_remote_db_send_packet_s *packet);
void arts_remote_db_full_request(arts_guid_t data_guid, int rank,
                                 arts_guid_t edt_guid, int pos,
                                 arts_db_mode_t mode);
void arts_remote_db_forward_full(int dest_rank, int source_rank,
                                 arts_guid_t data_guid, arts_guid_t edt_guid,
                                 int pos, arts_db_mode_t mode);
void arts_remote_db_full_send_now(int rank, struct arts_db_s *db,
                                  arts_guid_t edt_guid, unsigned int slot,
                                  arts_db_mode_t mode);
void arts_remote_db_full_send_check(int rank, struct arts_db_s *db,
                                    arts_guid_t edt_guid, unsigned int slot,
                                    arts_db_mode_t mode);
void arts_remote_db_full_send(
    struct arts_remote_db_full_request_packet_s *pack);
void arts_remote_handle_db_full_recieved(
    struct arts_remote_db_full_send_packet_s *packet);
void arts_remote_send_already_local(int rank, arts_guid_t guid,
                                    arts_guid_t edt_guid, unsigned int slot,
                                    arts_db_mode_t mode);
void arts_remote_handle_send_already_local(void *pack);
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
void arts_remote_metric_update(int rank, int type, int level,
                               uint64_t time_stamp, uint64_t to_add, bool sub);
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
