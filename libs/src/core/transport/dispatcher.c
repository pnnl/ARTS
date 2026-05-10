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
#include "arts/transport/dispatcher.h"

#include <unistd.h>

#include "arts.h"
#include "arts/compute/edt.h"
#include "arts/memory/coherence_handlers.h"
#include "arts/memory/db.h"
#include "arts/remote/handler.h"
#include "arts/runtime_state.h"
#include "arts/sync/event.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/transport/protocol.h"
#include "arts/utils/malloc.h"

#define EDT_MUG_SIZE 32

extern bool server_end;

#ifdef SEQUENCENUMBERS
uint64_t *rec_seq_numbers;
#endif

/*
 * arts_remote_send_shutdown_broadcast — First step of the shutdown protocol.
 *
 * Enqueue a header-only ARTS_REMOTE_SHUTDOWN_MSG to every other rank.
 * The sender thread drains the outbox; the caller should then wait for
 * arts_node_info.outbox_pending to reach zero (see wait_for_outbox_drain
 * in threads.c) before proceeding to local shutdown.
 */
void arts_remote_send_shutdown_broadcast(void) {
  if (arts_global_rank_count <= 1) {
    return;
  }
  for (unsigned int r = 0; r < arts_global_rank_count; r++) {
    if (r == arts_global_rank_id) {
      continue;
    }
    struct arts_remote_packet_s packet;
    arts_fill_packet_header(&packet, sizeof(packet), ARTS_REMOTE_SHUTDOWN_MSG);
    arts_remote_send_request_async((int)r, (char *)&packet, sizeof(packet));
  }
}

void arts_server_cleanup(void) {
  out_cleanup();
#ifdef SEQUENCENUMBERS
  arts_free(rec_seq_numbers);
  rec_seq_numbers = NULL;
#endif
}

void arts_server_setup(struct arts_config_s *config) {
  // ASYNC Message Queue Init
  arts_ll_server_setup(config);
  out_init(arts_global_rank_count * config->port_count);
#ifdef SEQUENCENUMBERS
  rec_seq_numbers =
      (uint64_t *)arts_calloc(arts_global_rank_count, sizeof(uint64_t));
#endif
}

void arts_server_process_packet(struct arts_remote_packet_s *packet) {
#ifdef SEQUENCENUMBERS
  uint64_t exp_seq_number =
      __sync_fetch_and_add(&rec_seq_numbers[packet->seq_rank], 1U);
  if (exp_seq_number != packet->seq_num) {
    ARTS_DEBUG(
        "MESSAGE RECIEVED OUT OF ORDER exp: %lu rec: %lu source: %u type: %d",
        exp_seq_number, packet->seq_num, packet->rank, packet->message_type);
  }
//    else
//        ARTS_INFO("Recv: %lu -> %lu = %lu", packet->seq_rank,
//        arts_global_rank_id, packet->seq_num);
#endif

  /* Catch-all dispatcher entry trace (multinode-receive diagnostic) */
  fprintf(stderr, "[DBG-RX rank %u] dispatch type=%d from_rank=%d\n",
          arts_global_rank_id, (int)packet->message_type, (int)packet->rank);
  fflush(stderr);

  switch (packet->message_type) {
  case ARTS_REMOTE_SHUTDOWN_MSG: {
    ARTS_INFO("Node %u: Received shutdown message from node %u",
              arts_global_rank_id, packet->rank);
    /* Passive shutdown entry — we received SHUTDOWN_MSG from another
     * rank, so we enter SHUTTING_DOWN locally (idempotent CAS, no
     * re-broadcast). The main thread will handle network stop and
     * bounded join after the worker loop exits. */
    arts_enter_shutdown_state(/* initiator = */ false);
    break;
  }
  case ARTS_REMOTE_EDT_SIGNAL_MSG: {
    struct arts_remote_edt_signal_packet_s *pack =
        (struct arts_remote_edt_signal_packet_s *)(packet);
    fprintf(stderr,
            "[DBG-RX rank %u] EDT_SIGNAL: edt=%lu slot=%u db=%lu mode=%d\n",
            arts_global_rank_id, (uint64_t)pack->edt, pack->slot,
            (uint64_t)pack->db, pack->mode);
    fflush(stderr);
    internal_signal_edt_with_mode(pack->edt, pack->slot, pack->db, pack->mode);
    break;
  }
  case ARTS_REMOTE_EVENT_SATISFY_SLOT_MSG: {
    arts_remote_handle_event_satisfy_slot(packet);
    break;
  }
  case ARTS_REMOTE_ADD_DEPENDENCE_MSG: {
    ARTS_DEBUG("Dependence Received");
    struct arts_remote_add_dependence_packet_s *pack =
        (struct arts_remote_add_dependence_packet_s *)(packet);
    arts_add_dependence(pack->source, pack->destination, pack->slot,
                        pack->mode);
    break;
  }
  case ARTS_REMOTE_EDT_MOVE_MSG: {
    ARTS_DEBUG("EDT Move Received");
    fprintf(stderr, "[DBG-RX rank %u] EDT_MOVE received\n",
            arts_global_rank_id);
    fflush(stderr);
    arts_remote_handle_edt_move(packet);
    break;
  }
  case ARTS_REMOTE_DB_MOVE_MSG: {
    ARTS_DEBUG("DB Move Received");
    arts_remote_handle_db_move(packet);
    break;
  }
  case ARTS_REMOTE_EVENT_MOVE_MSG: {
    ARTS_DEBUG("Event Move Received");
    arts_remote_handle_event_move(packet);
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
  case ARTS_REMOTE_SET_DEP_MODE_MSG: {
    ARTS_DEBUG("Set Dep Mode Received");
    struct arts_remote_set_dep_mode_packet_s *pack =
        (struct arts_remote_set_dep_mode_packet_s *)(packet);
    arts_set_dep_mode(pack->edt, pack->slot, pack->mode);
    break;
  }
  /* ===== coherence wire-message dispatch =============
   * Three handlers (GRANT / WRITEBACK / DATA_RESPONSE) carry trailing
   * payload right after sizeof(struct ...); pass that pointer + size as
   * the data/data_size arguments.
   *
   * Shared messages compile in both RC and LRC builds — Phase 6 will add
   * #ifdef branches inside the individual handlers as LRC diverges.
   * RC-only messages fatal in LRC builds to catch binary mode mismatch. */
  case ARTS_REMOTE_LOCK_REQ_MSG: {
    ARTS_DEBUG("Coh LOCK_REQ Received");
    struct arts_remote_lock_req_packet_s *pack =
        (struct arts_remote_lock_req_packet_s *)(packet);
    arts_coh_handle_lock_req(pack);
    break;
  }
  case ARTS_REMOTE_INVALIDATE_NOTICE_MSG: {
    ARTS_DEBUG("Coh INVALIDATE_NOTICE Received");
    arts_coh_handle_invalidate_notice(
        (struct arts_remote_invalidate_notice_packet_s *)(packet));
    break;
  }
  case ARTS_REMOTE_RELEASE_OWNERSHIP_MSG: {
    ARTS_DEBUG("Coh RELEASE_OWNERSHIP Received");
    arts_coh_handle_release_ownership(
        (struct arts_remote_release_ownership_packet_s *)(packet));
    break;
  }
  case ARTS_REMOTE_GET_DATA_MSG: {
    ARTS_DEBUG("Coh GET_DATA Received");
    arts_coh_handle_get_data((struct arts_remote_get_data_packet_s *)(packet));
    break;
  }
  case ARTS_REMOTE_DATA_RESPONSE_MSG: {
    ARTS_DEBUG("Coh DATA_RESPONSE Received");
    struct arts_remote_data_response_packet_s *pack =
        (struct arts_remote_data_response_packet_s *)(packet);
    const void *data = (const char *)pack + sizeof(*pack);
    uint64_t data_size = pack->header.size - sizeof(*pack);
    arts_coh_handle_data_response(pack, data_size > 0 ? data : NULL, data_size);
    break;
  }
  case ARTS_REMOTE_DB_CREATE_COHERENT_MSG: {
    ARTS_DEBUG("Coh DB_CREATE_COHERENT Received");
    arts_coh_handle_db_create_coherent(
        (struct arts_remote_db_create_coherent_packet_s *)(packet));
    break;
  }
  case ARTS_REMOTE_DESTROY_REQ_MSG: {
    ARTS_DEBUG("Coh DESTROY_REQ Received");
    arts_coh_handle_destroy_req(
        (struct arts_remote_destroy_req_packet_s *)(packet));
    break;
  }
  case ARTS_REMOTE_DESTROY_NOTIFY_MSG: {
    ARTS_DEBUG("Coh DESTROY_NOTIFY Received");
    arts_coh_handle_destroy_notify(
        (struct arts_remote_destroy_notify_packet_s *)(packet));
    break;
  }
  /* RC-only messages — fatal in LRC builds to catch binary mode mismatch */
#ifndef ARTS_MEMORY_MODEL_LRC
  case ARTS_REMOTE_GRANT_MSG: {
    ARTS_DEBUG("Coh GRANT Received");
    struct arts_remote_grant_packet_s *pack =
        (struct arts_remote_grant_packet_s *)(packet);
    const void *data = (const char *)pack + sizeof(*pack);
    uint64_t data_size = pack->header.size - sizeof(*pack);
    arts_coh_handle_grant(pack, data_size > 0 ? data : NULL, data_size);
    break;
  }
  case ARTS_REMOTE_WRITEBACK_MSG: {
    ARTS_DEBUG("Coh WRITEBACK Received");
    struct arts_remote_writeback_packet_s *pack =
        (struct arts_remote_writeback_packet_s *)(packet);
    const void *data = (const char *)pack + sizeof(*pack);
    uint64_t data_size = pack->header.size - sizeof(*pack);
    arts_coh_handle_writeback(pack, data_size > 0 ? data : NULL, data_size);
    break;
  }
  case ARTS_REMOTE_WRITEBACK_ACK_MSG: {
    ARTS_DEBUG("Coh WRITEBACK_ACK Received");
    arts_coh_handle_writeback_ack(
        (struct arts_remote_writeback_ack_packet_s *)(packet));
    break;
  }
#else  /* ARTS_MEMORY_MODEL_LRC */
  case ARTS_REMOTE_GRANT_MSG:
  case ARTS_REMOTE_WRITEBACK_MSG:
  case ARTS_REMOTE_WRITEBACK_ACK_MSG: {
    ARTS_ERROR("LRC build received RC-only message type %d from rank %u — "
               "binary mode mismatch?",
               packet->message_type, packet->rank);
    break;
  }
#endif /* ARTS_MEMORY_MODEL_LRC */
  case ARTS_REMOTE_EVENT_DESTROY_MSG: {
    ARTS_DEBUG("Event Destroy Received");
    arts_remote_handle_event_destroy(packet);
    break;
  }
  /* ===== LRC-only message dispatch
   * ============================================ These slots are only sent
   * between ranks compiled with ARTS_MEMORY_MODEL=LRC.  Real handlers are wired
   * in later tasks; for now the LRC build accepts them as stubs, and the RC
   * build fatals immediately to catch a binary mode mismatch between ranks. */
#ifdef ARTS_MEMORY_MODEL_LRC
  case ARTS_REMOTE_REDIRECT_RO_MSG: {
    ARTS_DEBUG("LRC REDIRECT_RO Received");
    arts_coh_handle_redirect_ro(
        (struct arts_remote_redirect_ro_packet_s *)(packet));
    break;
  }
  case ARTS_REMOTE_TRANSFER_OWNERSHIP_MSG: {
    ARTS_DEBUG("LRC TRANSFER_OWNERSHIP Received");
    /* Payload immediately follows the header in the contiguous wire buffer. */
    arts_coh_handle_transfer_ownership((void *)packet,
                                       (size_t)packet->size);
    break;
  }
  case ARTS_REMOTE_INSTALL_ACK_MSG: {
    ARTS_DEBUG("LRC INSTALL_ACK Received");
    arts_coh_handle_install_ack(
        (struct arts_remote_install_ack_packet_s *)(packet));
    break;
  }
#else  /* !ARTS_MEMORY_MODEL_LRC */
  case ARTS_REMOTE_REDIRECT_RO_MSG:
  case ARTS_REMOTE_TRANSFER_OWNERSHIP_MSG:
  case ARTS_REMOTE_INSTALL_ACK_MSG: {
    ARTS_ERROR("RC build received LRC-only message type %d from rank %u — "
               "binary mode mismatch?",
               packet->message_type, packet->rank);
    break;
  }
#endif /* ARTS_MEMORY_MODEL_LRC */
  default: {
    ARTS_INFO("Unknown Packet %d %d %d", packet->message_type, packet->size,
              packet->rank);
    arts_shutdown();
    arts_runtime_stop();
  }
  }
}
