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
#include "arts/network/server.h"

#include <unistd.h>

#include "arts.h"
#include "arts/utils/malloc.h"
#include "arts/introspection/metrics.h"
#include "arts/network/remote.h"
#include "arts/network/remote_protocol.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/runtime.h"
#include "arts/runtime/compute/edt_functions.h"
#include "arts/runtime/memory/db_functions.h"
#include "arts/runtime/memory/array_db.h"
#include "arts/runtime/network/remote_functions.h"
#include "arts/runtime/sync/event_functions.h"
#include "arts/system/arts_print.h"

#define EDT_MUG_SIZE 32

extern bool server_end;

#ifdef SEQUENCENUMBERS
uint64_t *rec_seq_numbers;
#endif

void arts_remote_shutdown() { arts_ll_server_shutdown(); }

void arts_server_setup(struct arts_config_s *config) {
  // ASYNC Message Deque Init
  arts_ll_server_setup(config);
  out_init(arts_global_rank_count * config->port_count);
#ifdef SEQUENCENUMBERS
  rec_seq_numbers = (uint64_t *)arts_calloc(arts_global_rank_count, sizeof(uint64_t));
#endif
}

void arts_server_process_packet(struct arts_remote_packet_s *packet) {
  if (packet->message_type != ARTS_REMOTE_METRIC_UPDATE_MSG &&
      packet->message_type != ARTS_REMOTE_SHUTDOWN_MSG) {
    ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_NETWORK_RECIEVE_BW, ARTS_METRIC_THREAD, packet->size);
    ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_FREE_BW + packet->message_type, ARTS_METRIC_THREAD,
                            packet->size);
    ARTS_METRICS_UPDATE_PACKET_INFO(packet->size);
  }
#ifdef SEQUENCENUMBERS
  uint64_t exp_seq_number =
      __sync_fetch_and_add(&rec_seq_numbers[packet->seq_rank], 1U);
  if (exp_seq_number != packet->seq_num) {
    ARTS_DEBUG(
        "MESSAGE RECIEVED OUT OF ORDER exp: %lu rec: %lu source: %u type: %d",
        exp_seq_number, packet->seq_num, packet->rank, packet->message_type);
  }
//    else
//        ARTS_INFO("Recv: %lu -> %lu = %lu", packet->seq_rank, arts_global_rank_id,
//        packet->seq_num);
#endif

  switch (packet->message_type) {
  case ARTS_REMOTE_SHUTDOWN_MSG: {
    ARTS_INFO("Node %u: Received shutdown message from node %u",
              arts_global_rank_id, packet->rank);
    arts_runtime_stop();
    break;
  }
  case ARTS_REMOTE_EDT_SIGNAL_MSG: {
    struct arts_remote_edt_signal_packet_s *pack =
        (struct arts_remote_edt_signal_packet_s *)(packet);
    internal_signal_edt_with_mode(pack->edt, pack->slot, pack->db, pack->mode);
    break;
  }
  case ARTS_REMOTE_EVENT_SATISFY_SLOT_MSG: {
    struct arts_remote_event_satisfy_slot_packet_s *pack =
        (struct arts_remote_event_satisfy_slot_packet_s *)(packet);
    arts_event_satisfy_slot(pack->event, pack->db, pack->slot);
    break;
  }
  case ARTS_REMOTE_PERSISTENT_EVENT_SATISFY_SLOT_MSG: {
    struct arts_remote_persistent_event_satisfy_slot_packet_s *pack =
        (struct arts_remote_persistent_event_satisfy_slot_packet_s *)(packet);

    arts_persistent_event_satisfy(pack->event, pack->action, pack->lock);
    break;
  }
  case ARTS_REMOTE_DB_INCREMENT_LATCH_MSG: {
    struct arts_remote_guid_only_packet_s *pack =
        (struct arts_remote_guid_only_packet_s *)(packet);
    arts_db_increment_latch(pack->guid);
    break;
  }
  case ARTS_REMOTE_DB_DECREMENT_LATCH_MSG: {
    struct arts_remote_guid_only_packet_s *pack =
        (struct arts_remote_guid_only_packet_s *)(packet);
    arts_db_decrement_latch(pack->guid);
    break;
  }
  case ARTS_REMOTE_DB_ADD_DEPENDENCE_MSG: {
    struct arts_remote_db_add_dependence_packet_s *pack =
        (struct arts_remote_db_add_dependence_packet_s *)(packet);
    arts_db_add_dependence_with_mode_and_diff(pack->db_src, pack->edt_dest,
                                       pack->edt_slot, pack->mode);
    break;
  }
  case ARTS_REMOTE_DB_ADD_DEPENDENCE_WITH_BYTE_OFFSET_MSG: {
    arts_remote_handle_db_add_dependence_with_byte_offset(packet);
    break;
  }
  case ARTS_REMOTE_DB_REQUEST_MSG: {
    struct arts_remote_db_request_packet_s *pack =
        (struct arts_remote_db_request_packet_s *)(packet);
    if (packet->size != sizeof(*pack)) {
      ARTS_INFO("Error dbpacket insanity");
    }
    arts_remote_db_send(pack);
    break;
  }
  case ARTS_REMOTE_DB_SEND_MSG: {
    ARTS_DEBUG("Remote Db Received");
    struct arts_remote_db_send_packet_s *pack =
        (struct arts_remote_db_send_packet_s *)(packet);
    arts_remote_handle_db_received(pack);
    break;
  }
  case ARTS_REMOTE_ADD_DEPENDENCE_MSG: {
    ARTS_DEBUG("Dependence Received");
    struct arts_remote_add_dependence_packet_s *pack =
        (struct arts_remote_add_dependence_packet_s *)(packet);
    arts_add_dependence(pack->source, pack->destination, pack->slot);
    break;
  }
  case ARTS_REMOTE_ADD_DEPENDENCE_TO_PERSISTENT_EVENT_MSG: {
    ARTS_DEBUG("Persistent Dependence Received");
    struct arts_remote_add_dependence_packet_s *pack =
        (struct arts_remote_add_dependence_packet_s *)(packet);
    arts_add_dependence_to_persistent_event_with_mode_and_diff(
        pack->source, pack->destination, pack->slot, pack->mode);
    break;
  }
  case ARTS_REMOTE_ADD_DEPENDENCE_TO_PERSISTENT_EVENT_WITH_BYTE_OFFSET_MSG: {
    ARTS_DEBUG("Persistent Dependence with ByteOffset Received");
    struct arts_remote_add_dependence_with_byte_offset_packet_s *pack =
        (struct arts_remote_add_dependence_with_byte_offset_packet_s *)(packet);
    arts_add_dependence_to_persistent_event_with_byte_offset(
        pack->source, pack->destination, pack->slot, pack->mode,
        pack->byte_offset, pack->size);
    break;
  }
  case ARTS_REMOTE_INVALIDATE_DB_MSG: {
    ARTS_DEBUG("DB Invalidate Received");
    arts_remote_handle_invalidate_db(packet);
    break;
  }
  case ARTS_REMOTE_DB_FULL_REQUEST_MSG: {
    struct arts_remote_db_full_request_packet_s *pack =
        (struct arts_remote_db_full_request_packet_s *)(packet);
    arts_remote_db_full_send(pack);
    break;
  }
  case ARTS_REMOTE_DB_FULL_SEND_MSG: {
    ARTS_DEBUG("DB Full Send Received");
    struct arts_remote_db_full_send_packet_s *pack =
        (struct arts_remote_db_full_send_packet_s *)(packet);
    arts_remote_handle_db_full_recieved(pack);
    break;
  }
  case ARTS_REMOTE_DB_FULL_SEND_ALREADY_LOCAL_MSG: {
    ARTS_DEBUG("DB Full Send Already Local Received");
    arts_remote_handle_send_already_local(packet);
    break;
  }
  case ARTS_REMOTE_DB_DESTROY_MSG: {
    ARTS_DEBUG("DB Destroy Received");
    arts_remote_handle_db_destroy(packet);
    break;
  }
  case ARTS_REMOTE_DB_DESTROY_FORWARD_MSG: {
    ARTS_DEBUG("DB Destroy Forward Received");
    arts_remote_handle_db_destroy_forward(packet);
    break;
  }
  case ARTS_REMOTE_DB_CLEAN_FORWARD_MSG: {
    ARTS_DEBUG("DB Clean Forward Received");
    arts_remote_handle_db_clean_forward(packet);
    break;
  }
  case ARTS_REMOTE_DB_UPDATE_GUID_MSG: {
    ARTS_DEBUG("DB Guid Update Received");
    arts_remote_handle_update_db_guid(packet);
    break;
  }
  case ARTS_REMOTE_EDT_MOVE_MSG: {
    ARTS_DEBUG("EDT Move Received");
    arts_remote_handle_edt_move(packet);
    break;
  }
  case ARTS_REMOTE_DB_MOVE_MSG: {
    ARTS_DEBUG("DB Move Received");
    arts_remote_handle_db_move(packet);
    break;
  }
  case ARTS_REMOTE_DB_UPDATE_MSG: {
    ARTS_DEBUG("DB Update Received");
    arts_remote_handle_update_db(packet);
    break;
  }
  case ARTS_REMOTE_DB_PARTIAL_UPDATE_MSG: {
    ARTS_DEBUG("DB Partial Update Received");
    arts_remote_handle_partial_update(packet);
    break;
  }
  case ARTS_REMOTE_EVENT_MOVE_MSG: {
    ARTS_DEBUG("Event Move Received");
    arts_remote_handle_event_move(packet);
    break;
  }
  case ARTS_REMOTE_PERSISTENT_EVENT_MOVE_MSG: {
    ARTS_DEBUG("Persistent Event Move Received");
    arts_remote_handle_persistent_event_move(packet);
    break;
  }
  case ARTS_REMOTE_METRIC_UPDATE_MSG: {

    struct arts_remote_metric_update_s *pack =
        (struct arts_remote_metric_update_s *)(packet);
    ARTS_DEBUG("Metric update Received %u -> %d %ld", arts_global_rank_id,
               pack->type, pack->to_add);
    ARTS_METRICS_HANDLE_REMOTE_UPDATE(pack->type, ARTS_METRIC_SYSTEM, pack->to_add,
                                  pack->sub);
    break;
  }
  case ARTS_REMOTE_GET_FROM_DB_MSG: {
    ARTS_DEBUG("Get From DB Received");
    arts_remote_handle_get_from_db(packet);
    break;
  }
  case ARTS_REMOTE_PUT_IN_DB_MSG: {
    ARTS_DEBUG("Put In DB Received");
    arts_remote_handle_put_in_db(packet);
    break;
  }
  case ARTS_REMOTE_SIGNAL_EDT_WITH_PTR_MSG: {
    ARTS_DEBUG("Signal EDT With Ptr Received");
    arts_remote_handle_signal_edt_with_ptr(packet);
    break;
  }
  case ARTS_REMOTE_SEND_MSG: {
    ARTS_DEBUG("Send Received");
    arts_remote_handle_send(packet);
    break;
  }
  case ARTS_EPOCH_INIT_MSG: {
    ARTS_DEBUG("Epoch Init Received");
    arts_remote_handle_epoch_init_send(packet);
    break;
  }
  case ARTS_EPOCH_REQ_MSG: {
    ARTS_DEBUG("Epoch Req Received");
    arts_remote_handle_epoch_req(packet);
    break;
  }
  case ARTS_EPOCH_SEND_MSG: {
    ARTS_DEBUG("Epoch Send Received");
    arts_remote_handle_epoch_send(packet);
    break;
  }
  case ARTS_ATOMIC_ADD_ARRAYDB_MSG: {
    ARTS_DEBUG("Atomic Add ArrayDB Received");
    arts_remote_handle_atomic_add_in_array_db(packet);
    break;
  }
  case ARTS_ATOMIC_CAS_ARRAYDB_MSG: {
    ARTS_DEBUG("Atomic Compare And Swap ArrayDB Received");
    arts_remote_handle_atomic_compare_and_swap_in_array_db(packet);
    break;
  }
  case ARTS_EPOCH_INIT_POOL_MSG: {
    ARTS_DEBUG("Epoch Init Pool Received");
    arts_remote_handle_epoch_init_pool_send(packet);
    break;
  }
  case ARTS_EPOCH_DELETE_MSG: {
    ARTS_DEBUG("Epoch Delete Received");
    arts_remote_handle_epoch_delete(packet);
    break;
  }
  case ARTS_REMOTE_BUFFER_SEND_MSG: {
    ARTS_DEBUG("Buffer Send Received");
    arts_remote_handle_buffer_send(packet);
    break;
  }
  case ARTS_REMOTE_DB_MOVE_REQ_MSG: {
    ARTS_DEBUG("DB Move Request Received");
    arts_db_move_request_handle(packet);
    break;
  }
  case ARTS_REMOTE_DB_RENAME_MSG: {
    ARTS_DEBUG("DB Rename Received");
    arts_remote_handle_db_rename(packet);
    break;
  }
  case ARTS_REMOTE_TIME_SYNC_REQ_MSG: {
    ARTS_DEBUG("Time Sync Request Received");
    arts_remote_handle_time_sync_req(packet);
    break;
  }
  case ARTS_REMOTE_TIME_SYNC_RESP_MSG: {
    ARTS_DEBUG("Time Sync Response Received");
    arts_remote_handle_time_sync_resp(packet);
    break;
  }
  default: {
    ARTS_INFO("Unknown Packet %d %d %d", packet->message_type, packet->size,
              packet->rank);
    arts_shutdown();
    arts_runtime_stop();
  }
  }
}
