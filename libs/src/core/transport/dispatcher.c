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
#include "arts/sync/epoch.h"
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
 * Enqueue a header-only MSG_SHUTDOWN to every other rank.
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
    arts_fill_packet_header(&packet, sizeof(packet), MSG_SHUTDOWN);
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

  switch (packet->message_type) {
  case MSG_SHUTDOWN: {
    ARTS_INFO("Node %u: Received shutdown message from node %u",
              arts_global_rank_id, packet->rank);
    /* Passive shutdown entry — we received SHUTDOWN_MSG from another
     * rank, so we enter SHUTTING_DOWN locally (idempotent CAS, no
     * re-broadcast). The main thread will handle network stop and
     * bounded join after the worker loop exits. */
    arts_enter_shutdown_state(/* initiator = */ false);
    break;
  }
  case MSG_EDT_SATISFY_SLOT: {
    struct arts_remote_edt_signal_packet_s *pack =
        (struct arts_remote_edt_signal_packet_s *)(packet);
    /* DB_MODE_PTR carries an inline payload right after the header; other
     * modes deliver a GUID/value reference only (size == 0). */
    void *source = pack->size > 0 ? (void *)(pack + 1) : NULL;
    arts_edt_satisfy_slot(pack->edt, pack->slot, pack->db, pack->mode, source,
                          pack->size);
    break;
  }
  case MSG_EVENT_SATISFY_SLOT: {
    struct arts_remote_event_satisfy_slot_packet_s *pack =
        (struct arts_remote_event_satisfy_slot_packet_s *)(packet);
    arts_event_satisfy_slot(pack->event, pack->db, pack->slot);
    break;
  }
  case MSG_EVENT_ADD_DEPENDENCE: {
    ARTS_DEBUG("Dependence Received");
    struct arts_remote_add_dependence_packet_s *pack =
        (struct arts_remote_add_dependence_packet_s *)(packet);
    arts_add_dependence(pack->source, pack->destination, pack->slot,
                        pack->mode);
    break;
  }
  case MSG_EDT_CREATE: {
    ARTS_DEBUG("EDT Create Received");
    arts_handler_edt_create(packet);
    break;
  }
  case MSG_EVENT_CREATE: {
    ARTS_DEBUG("Event Move Received");
    arts_handler_event_create(packet);
    break;
  }
  case MSG_EPOCH_CREATE: {
    ARTS_DEBUG("Epoch Init Received");
    arts_handler_epoch_create(packet);
    break;
  }
  case MSG_EPOCH_REQUEST: {
    ARTS_DEBUG("Epoch Req Received");
    struct arts_remote_guid_only_packet_s *pack =
        (struct arts_remote_guid_only_packet_s *)(packet);
    /* source and dest are the requester rank (the query origin). */
    send_epoch(pack->guid, pack->header.rank, pack->header.rank);
    break;
  }
  case MSG_EPOCH_SEND: {
    ARTS_DEBUG("Epoch Send Received");
    struct arts_remote_epoch_send_packet_s *pack =
        (struct arts_remote_epoch_send_packet_s *)(packet);
    reduce_epoch(pack->epoch_guid, pack->active, pack->finish);
    break;
  }
  case MSG_EPOCH_INIT_POOL: {
    ARTS_DEBUG("Epoch Init Pool Received");
    arts_handler_epoch_init_pool(packet);
    break;
  }
  case MSG_EPOCH_DELETE: {
    ARTS_DEBUG("Epoch Delete Received");
    arts_handler_epoch_delete(packet);
    break;
  }
  case MSG_TIME_SYNC_REQUEST: {
    ARTS_DEBUG("Time Sync Request Received");
    arts_handler_time_sync_request(packet);
    break;
  }
  case MSG_TIME_SYNC_RESPONSE: {
    ARTS_DEBUG("Time Sync Response Received");
    arts_handler_time_sync_response(packet);
    break;
  }
  /* ===== coherence wire-message dispatch =============
   * Three handlers (GRANT / WRITEBACK / DATA_RESPONSE) carry trailing
   * payload right after sizeof(struct ...); pass that pointer + size as
   * the data/data_size arguments.
   *
   * LOCK_REQ / INVALIDATE_NOTICE / RELEASE_OWNERSHIP: shared between RC and
   * LRC (both use per-DB exclusive ownership), but LC has no such concept.
   * Fatal in LC builds to catch binary mode mismatch. */
#if defined(ARTS_MEMORY_MODEL_LC)
  case MSG_DB_OWNERSHIP_REQUEST:
  case MSG_DB_OWNERSHIP_INVALIDATE:
  case MSG_DB_OWNERSHIP_RETURN: {
    ARTS_ERROR("LC build received exclusivity message type %d from rank %u "
               "— LC has no LOCK_REQ / INVALIDATE / RELEASE_OWNERSHIP; "
               "binary mode mismatch?",
               packet->message_type, packet->rank);
    break;
  }
#else  /* RC and LRC: full handlers */
  case MSG_DB_OWNERSHIP_REQUEST: {
    ARTS_DEBUG("Coh LOCK_REQ Received");
    struct arts_remote_lock_req_packet_s *pack =
        (struct arts_remote_lock_req_packet_s *)(packet);
    arts_handler_db_ownership_request(pack);
    break;
  }
  case MSG_DB_OWNERSHIP_INVALIDATE: {
    ARTS_DEBUG("Coh INVALIDATE_NOTICE Received");
    arts_handler_db_ownership_invalidate(
        (struct arts_remote_invalidate_notice_packet_s *)(packet));
    break;
  }
  case MSG_DB_OWNERSHIP_RETURN: {
    ARTS_DEBUG("Coh RELEASE_OWNERSHIP Received");
    arts_handler_db_ownership_return(
        (struct arts_remote_release_ownership_packet_s *)(packet));
    break;
  }
#endif /* ARTS_MEMORY_MODEL_LC */
  case MSG_DB_SNAPSHOT_REQUEST: {
    ARTS_DEBUG("Coh GET_DATA Received");
    arts_handler_db_snapshot_request(
        (struct arts_remote_get_data_packet_s *)(packet));
    break;
  }
  case MSG_DB_SNAPSHOT_RESPONSE: {
    ARTS_DEBUG("Coh DATA_RESPONSE Received");
    struct arts_remote_data_response_packet_s *pack =
        (struct arts_remote_data_response_packet_s *)(packet);
    const void *data = (const char *)pack + sizeof(*pack);
    uint64_t data_size = pack->header.size - sizeof(*pack);
    arts_handler_db_snapshot_response(pack, data_size > 0 ? data : NULL,
                                      data_size);
    break;
  }
  case MSG_DB_CREATE: {
    ARTS_DEBUG("Coh DB_CREATE_COHERENT Received");
    arts_handler_db_create_coherent(
        (struct arts_remote_db_create_coherent_packet_s *)(packet));
    break;
  }
  case MSG_DB_DESTROY: {
    ARTS_DEBUG("Coh DESTROY_REQ Received");
    arts_handler_db_destroy(
        (struct arts_remote_destroy_req_packet_s *)(packet));
    break;
  }
  case MSG_DB_CACHE_DESTROY: {
    ARTS_DEBUG("Coh DESTROY_NOTIFY Received");
    arts_handler_db_cache_destroy(
        (struct arts_remote_destroy_notify_packet_s *)(packet));
    break;
  }
  /* GRANT: RC-only — fatal in LRC and LC builds to catch binary mode mismatch.
   */
#if defined(ARTS_MEMORY_MODEL_LRC) || defined(ARTS_MEMORY_MODEL_LC)
  case MSG_DB_OWNERSHIP_RESPONSE: {
    ARTS_ERROR("Non-RC build received RC-only GRANT message from rank %u — "
               "binary mode mismatch?",
               packet->rank);
    break;
  }
#else  /* RC build */
  case MSG_DB_OWNERSHIP_RESPONSE: {
    ARTS_DEBUG("Coh GRANT Received");
    struct arts_remote_grant_packet_s *pack =
        (struct arts_remote_grant_packet_s *)(packet);
    const void *data = (const char *)pack + sizeof(*pack);
    uint64_t data_size = pack->header.size - sizeof(*pack);
    arts_handler_db_ownership_response(pack, data_size > 0 ? data : NULL,
                                       data_size);
    break;
  }
#endif /* ARTS_MEMORY_MODEL_LRC || ARTS_MEMORY_MODEL_LC */
  /* WRITEBACK + WRITEBACK_ACK: used by RC and LC (sync release writeback).
   * Fatal in LRC only — LRC uses async transfer, not synchronous writeback. */
#if defined(ARTS_MEMORY_MODEL_LRC)
  case MSG_DB_WRITEBACK:
  case MSG_DB_WRITEBACK_ACK: {
    ARTS_ERROR("LRC build received writeback message type %d from rank %u — "
               "LRC has no synchronous writeback; binary mode mismatch?",
               packet->message_type, packet->rank);
    break;
  }
#else  /* RC and LC: full handlers */
  case MSG_DB_WRITEBACK: {
    ARTS_DEBUG("Coh WRITEBACK Received");
    struct arts_remote_writeback_packet_s *pack =
        (struct arts_remote_writeback_packet_s *)(packet);
    const void *data = (const char *)pack + sizeof(*pack);
    uint64_t data_size = pack->header.size - sizeof(*pack);
    arts_handler_db_writeback(pack, data_size > 0 ? data : NULL, data_size);
    break;
  }
  case MSG_DB_WRITEBACK_ACK: {
    ARTS_DEBUG("Coh WRITEBACK_ACK Received");
    arts_handler_db_writeback_ack(
        (struct arts_remote_writeback_ack_packet_s *)(packet));
    break;
  }
#endif /* ARTS_MEMORY_MODEL_LRC */
  case MSG_EVENT_DESTROY: {
    ARTS_DEBUG("Event Destroy Received");
    arts_handler_event_destroy(packet);
    break;
  }
  /* ===== LRC-only message dispatch
   * ============================================ These slots are only sent
   * between ranks compiled with ARTS_MEMORY_MODEL=LRC.  Real handlers are wired
   * in later tasks; for now the LRC build accepts them as stubs, and the RC
   * build fatals immediately to catch a binary mode mismatch between ranks. */
#ifdef ARTS_MEMORY_MODEL_LRC
  case MSG_DB_SNAPSHOT_REDIRECT: {
    ARTS_DEBUG("LRC REDIRECT_RO Received");
    arts_handler_db_snapshot_redirect(
        (struct arts_remote_redirect_ro_packet_s *)(packet));
    break;
  }
  case MSG_DB_OWNERSHIP_RESPONSE_LRC: {
    ARTS_DEBUG("LRC TRANSFER_OWNERSHIP Received");
    /* Payload immediately follows the header in the contiguous wire buffer. */
    arts_handler_db_ownership_response_lrc((void *)packet,
                                           (size_t)packet->size);
    break;
  }
  case MSG_DB_OWNERSHIP_RESPONSE_ACK: {
    ARTS_DEBUG("LRC INSTALL_ACK Received");
    arts_handler_db_ownership_response_ack(
        (struct arts_remote_install_ack_packet_s *)(packet));
    break;
  }
#else  /* !ARTS_MEMORY_MODEL_LRC */
  case MSG_DB_SNAPSHOT_REDIRECT:
  case MSG_DB_OWNERSHIP_RESPONSE_LRC:
  case MSG_DB_OWNERSHIP_RESPONSE_ACK: {
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
