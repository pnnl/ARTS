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
#include "arts/remote/handler.h"

#include <string.h>

#include "arts.h"
#include "arts/compute/edt.h"
#include "arts/gas/out_of_order.h"
#include "arts/gas/route_table.h"
#include "arts/memory/db.h"
#include "arts/runtime_state.h"
#include "arts/sync/epoch.h"
#include "arts/sync/event.h"  /* arts_event_free_internal */
#include "arts/sync/shared.h" /* arts_shared_init */
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

static void send_remote_add_dependence_packet(unsigned int message_type,
                                              arts_guid_t source,
                                              arts_guid_t destination,
                                              uint32_t slot, unsigned int rank,
                                              arts_db_access_mode_t mode) {
  struct arts_remote_add_dependence_packet_s packet;
  packet.source = source;
  packet.destination = destination;
  packet.slot = slot;
  packet.mode = mode;
  arts_fill_packet_header(&packet.header, sizeof(packet), message_type);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_remote_add_dependence(arts_guid_t source, arts_guid_t destination,
                                uint32_t slot, unsigned int rank,
                                arts_db_access_mode_t mode) {
  ARTS_DEBUG("Remote Add dependence sent %d", rank);
  send_remote_add_dependence_packet(ARTS_REMOTE_ADD_DEPENDENCE_MSG, source,
                                    destination, slot, rank, mode);
}

void arts_remote_add_dependence_with_hints(arts_guid_t source,
                                           arts_guid_t destination,
                                           uint32_t slot, unsigned int rank,
                                           arts_db_access_mode_t mode) {
  ARTS_DEBUG("Remote Add dependence (mode=%u) sent %d", mode, rank);
  send_remote_add_dependence_packet(ARTS_REMOTE_ADD_DEPENDENCE_MSG, source,
                                    destination, slot, rank, mode);
}

void arts_remote_set_dep_mode(arts_guid_t edt_guid, uint32_t slot,
                              arts_db_access_mode_t mode) {
  unsigned int rank = arts_guid_get_rank(edt_guid);
  struct arts_remote_set_dep_mode_packet_s packet;
  packet.edt = edt_guid;
  packet.slot = slot;
  packet.mode = mode;
  arts_fill_packet_header(&packet.header, sizeof(packet),
                          ARTS_REMOTE_SET_DEP_MODE_MSG);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_remote_memory_move(unsigned int rank, arts_guid_t guid, void *ptr,
                             unsigned int mem_size, unsigned message_type,
                             void (*free_method)(void *)) {
  TIME_REMOTE_MOVE_START();
  struct arts_remote_guid_only_packet_s packet;
  arts_fill_packet_header(&packet.header, sizeof(packet) + mem_size,
                          message_type);
  packet.guid = guid;
  arts_remote_send_request_payload_async_free((int)rank, (char *)&packet,
                                              sizeof(packet), (char *)ptr, 0,
                                              mem_size, free_method);
  /* route_table slot now persists; Lifecycle redesign is follow-up work. */
  (void)guid;
  TIME_REMOTE_MOVE_STOP();
}

void arts_remote_memory_move_no_free(unsigned int rank, arts_guid_t guid,
                                     void *ptr, unsigned int mem_size,
                                     unsigned message_type) {
  struct arts_remote_guid_only_packet_s packet;
  arts_fill_packet_header(&packet.header, sizeof(packet) + mem_size,
                          message_type);
  packet.guid = guid;
  arts_remote_send_request_payload_async((int)rank, (char *)&packet,
                                         sizeof(packet), (char *)ptr, mem_size);
}

void arts_remote_handle_edt_move(void *ptr) {
  struct arts_remote_guid_only_packet_s *packet =
      (struct arts_remote_guid_only_packet_s *)ptr;
  uint64_t size =
      packet->header.size - sizeof(struct arts_remote_guid_only_packet_s);
  struct arts_edt_s *edt = (struct arts_edt_s *)arts_malloc_align(size, 16);

  memcpy(edt, packet + 1, size);
  /* re-stamp the deleter pointer.  The wire image was copied
   * verbatim from the sender's process, so edt->shared.deleter contains
   * the sender-side function pointer — non-PIE binaries make that valid
   * here, but reinstalling defensively keeps the lifecycle ownership
   * symmetric with edt_create_internal. */
  arts_shared_init(&edt->shared, arts_edt_get_deleter());
  /* finish-scope chain: if the EDT arrived with a non-NULL finish_event,
   * the field currently holds the *parent* finish_event GUID (which lives
   * on the source rank).  Allocate a local proxy LATCH and rewrite the
   * field so this EDT's finish_event is local-home.  Register a dep so
   * that proxy fire emits DECR on the remote parent.
   *
   * The matching INCR on the remote parent was already emitted on the
   * source rank inside arts_edt_create_internal before the EDT was
   * shipped — race-free under source-rank local sync ordering. */
  if (edt->finish_event != NULL_GUID) {
    arts_guid_t parent_fe = edt->finish_event;
    arts_event_hint_t latch_hint = ARTS_EVENT_HINT_LATCH(1);
    arts_guid_t proxy = arts_event_create(&latch_hint);
    /* add_dependence registers the proxy in its own waiter list (local op).
     * The cross-node satisfy-on-fire is emitted automatically by the
     * LATCH fire path when proxy.counter reaches 0. */
    arts_add_dependence(proxy, parent_fe, ARTS_EVENT_LATCH_DECR_SLOT,
                        DB_MODE_NULL);
    edt->finish_event = proxy;
  }
  /* add_item_race installs the EDT under the route_table lock.  On
   * rejection (another thread won the install race) free the freshly
   * unmarshaled buffer through the deleter — mirrors event_move's
   * race-loser cleanup pattern. */
  if (!arts_route_table_add_item_race(edt, packet->guid, arts_global_rank_id,
                                      false)) {
    /* race-loser cleanup: if we allocated a proxy LATCH for the
     * finish-scope chain, drain it.  proxy.counter == 1 (self-alive
     * token, just allocated above).  DECR drives counter to 0 → fire,
     * which emits the cross-node DECR to the remote parent_fe via the
     * dep we just registered.  This cancels the source-rank INCR that
     * was emitted before this EDT was shipped, keeping the parent
     * finish-scope balanced.  proxy itself self-destroys on fire (LATCH
     * auto_destroy semantics). */
    if (edt->finish_event != NULL_GUID) {
      arts_event_satisfy_slot(edt->finish_event, NULL_GUID,
                              ARTS_EVENT_LATCH_DECR_SLOT);
    }
    arts_edt_get_deleter()(edt);
    return;
  }
  ARTS_INFO("EDT[Guid:%lu] Moved to Rank: %d", packet->guid,
            arts_global_rank_id);
  if (edt->depc_needed == 0) {
    arts_handle_ready_edt(edt);
  } else {
    arts_route_table_fire_oo(packet->guid, arts_out_of_order_handler);
  }
}

void arts_remote_handle_db_move(void *ptr) {
  struct arts_remote_guid_only_packet_s *packet =
      (struct arts_remote_guid_only_packet_s *)ptr;
  uint64_t size =
      packet->header.size - sizeof(struct arts_remote_guid_only_packet_s);

  struct arts_db_s db_header_buf;
  memcpy(&db_header_buf, (packet + 1), sizeof(struct arts_db_s));
  uint64_t db_size = db_header_buf.header.size;

  struct arts_header_s *mem_packet =
      (struct arts_header_s *)arts_malloc_align(db_size, 16);

  if (size == db_size) {
    memcpy(mem_packet, packet + 1, size);
  } else {
    mem_packet->type = (unsigned int)arts_guid_get_kind(packet->guid);
    mem_packet->size = db_size;
  }
  /* DB-level coherence (RC) is owned by ARTS_DB's coherence_cache,
   * not by db_list.  After the move, the moved DB does not carry an
   * RC cache (state is rebuilt lazily on first acquire); other subtypes
   * have no DB-level coherence at all.  Just clear db_list. */
  {
    struct arts_db_s *new_db = (struct arts_db_s *)mem_packet;
    new_db->db_list = NULL;
  }

  ARTS_INFO("DB[Guid:%lu] Moved to Rank: %d", packet->guid,
            arts_global_rank_id);
  if (arts_route_table_add_item_race(mem_packet, packet->guid,
                                     arts_global_rank_id, true)) {
    arts_route_table_fire_oo(packet->guid, arts_out_of_order_handler);
  }
}

void arts_remote_handle_event_move(void *ptr) {
  struct arts_remote_guid_only_packet_s *packet =
      (struct arts_remote_guid_only_packet_s *)ptr;
  uint64_t size =
      packet->header.size - sizeof(struct arts_remote_guid_only_packet_s);

  struct arts_event_s *mem_packet =
      (struct arts_event_s *)arts_malloc_align(size, 16);

  memcpy(mem_packet, packet + 1, size);
  /* add_item_race installs the event under the route_table lock; on
   * success it also fires OoO replay internally, so no extra fire_oo
   * is required.  On rejection (another rank won the install race),
   * release the freshly-unmarshaled buffer through event_deleter (via
   * arts_event_free_internal) — raw arts_free would skip the dep-stack
   * drain.  In practice the dep stack is empty at this point (nothing
   * has been pushed locally yet), but using the proper deleter keeps
   * lifecycle ownership symmetric with event_alloc. */
  if (!arts_route_table_add_item_race(mem_packet, packet->guid,
                                      arts_global_rank_id, false)) {
    arts_event_free_internal(mem_packet);
  }
}

void arts_remote_handle_event_satisfy_slot(void *ptr) {
  struct arts_remote_event_satisfy_slot_packet_s *pack =
      (struct arts_remote_event_satisfy_slot_packet_s *)ptr;
  /* Delegate to the local entry point.  arts_event_satisfy_slot itself
   * already performs the safe-lookup pair (lookup_event_safe + release)
   * and runs event_drain on DECR.  The event GUID's rank equals the
   * local rank by construction (the sender forwarded here precisely
   * because arts_guid_get_rank == us), so the `rank != arts_global_rank_id`
   * branch never re-enters arts_remote_event_satisfy_slot — no loop.
   * The OoO miss path is the same as the local one (event not yet
   * installed → enqueue via arts_out_of_order_event_satisfy_slot). */
  arts_event_satisfy_slot(pack->event, pack->db, pack->slot);
}

void arts_remote_event_destroy(arts_guid_t guid) {
  unsigned int rank = arts_guid_get_rank(guid);
  struct arts_remote_guid_only_packet_s packet;
  packet.guid = guid;
  arts_fill_packet_header(&packet.header, sizeof(packet),
                          ARTS_REMOTE_EVENT_DESTROY_MSG);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_remote_handle_event_destroy(void *ptr) {
  struct arts_remote_guid_only_packet_s *packet =
      (struct arts_remote_guid_only_packet_s *)ptr;
  /* mark_delete is idempotent (DELETE bit sticky); receiving the message
   * twice is safe.  free_item runs via the route table once count==0. */
  arts_route_table_mark_delete(packet->guid);
}

static void send_remote_edt_signal_packet(arts_guid_t edt, arts_guid_t db,
                                          uint32_t slot,
                                          arts_db_access_mode_t mode) {
  struct arts_remote_edt_signal_packet_s packet;
  unsigned int rank = arts_guid_get_rank(edt);

  if (rank == arts_global_rank_id) {
    rank = arts_route_table_lookup_rank(edt);
  }

  packet.db = db;
  packet.edt = edt;
  packet.slot = slot;
  packet.mode = mode;
  packet.db_route = arts_guid_get_rank(db);
  arts_fill_packet_header(&packet.header, sizeof(packet),
                          ARTS_REMOTE_EDT_SIGNAL_MSG);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_remote_signal_edt(arts_guid_t edt, arts_guid_t db, uint32_t slot,
                            arts_db_access_mode_t mode) {
  ARTS_INFO("Remote Signal from DB[Guid:%lu] to EDT[Guid:%lu, Slot:%d, Rank: "
            "%d]",
            db, edt, slot, arts_guid_get_rank(edt));
  send_remote_edt_signal_packet(edt, db, slot, mode);
}

void arts_remote_event_satisfy_slot(arts_guid_t event_guid,
                                    arts_guid_t data_guid, uint32_t slot) {
  struct arts_remote_event_satisfy_slot_packet_s packet;
  packet.event = event_guid;
  packet.db = data_guid;
  packet.slot = slot;
  arts_fill_packet_header(&packet.header, sizeof(packet),
                          ARTS_REMOTE_EVENT_SATISFY_SLOT_MSG);
  arts_remote_send_request_async((int)arts_guid_get_rank(event_guid),
                                 (char *)&packet, sizeof(packet));
}

/*
 * arts_db_request_callback — Used by the OoO replay path
 * (arts_out_of_order_handle_db_request) when a local DB referenced by
 * an EDT dependency arrives in the route table after the EDT was
 * registered.  Fills the EDT's dep slot with the freshly-installed
 * DB pointer and drops one depc_needed.  For ARTS_DB types the
 * RC acquire path replaces this; for non-RC pinned types the OoO
 * replay covers the local-create-after-consumer race.
 */
void arts_db_request_callback(struct arts_edt_s *edt, unsigned int slot,
                              struct arts_db_s *db_res) {
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
  if (db_res) {
    /* legacy "no-op acquire to balance return_db" is gone.
     * The slot ptr is filled directly; ownership of the DB lifetime is
     * driven by the route_table lifecycle (mark_delete + free_item). */
    depv[slot].ptr = db_res + 1;
  } else {
    /* DB was destroyed between the OO check and the lookup (DELETE_ITEM
     * race).  Treat this slot as a NULL dependency — the data is gone. */
    ARTS_WARN("arts_db_request_callback: db_res is NULL for EDT[Guid:%lu] "
              "slot=%u (DB destroyed during OO resolution)",
              edt->current_edt, slot);
    depv[slot].guid = NULL_GUID;
    depv[slot].ptr = NULL;
  }
  unsigned int temp = arts_atomic_sub(&edt->depc_needed, 1U);
  if (temp == 0) {
    arts_handle_remote_stolen_edt(edt);
  }
}

void arts_remote_get_from_db(arts_guid_t edt_guid, arts_guid_t db_guid,
                             unsigned int slot, unsigned int offset,
                             unsigned int len, unsigned int rank) {
  struct arts_remote_get_put_packet_s packet;
  packet.edt_guid = edt_guid;
  packet.db_guid = db_guid;
  packet.slot = slot;
  packet.offset = offset;
  packet.size = len;
  arts_fill_packet_header(&packet.header, sizeof(packet),
                          ARTS_REMOTE_GET_FROM_DB_MSG);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_remote_handle_get_from_db(void *pack) {
  struct arts_remote_get_put_packet_s *packet =
      (struct arts_remote_get_put_packet_s *)pack;
  arts_db_get(
      packet->edt_guid, packet->db_guid, packet->slot, packet->offset,
      packet->size,
      &(arts_db_op_hint_t){.rank = arts_global_rank_id, .epoch = NULL_GUID});
}

void arts_remote_put_in_db(void *ptr, arts_guid_t edt_guid, arts_guid_t db_guid,
                           unsigned int slot, unsigned int offset,
                           unsigned int len, arts_guid_t epoch_guid,
                           unsigned int rank) {
  struct arts_remote_get_put_packet_s packet;
  packet.edt_guid = edt_guid;
  packet.db_guid = db_guid;
  packet.epoch_guid = epoch_guid;
  packet.slot = slot;
  packet.offset = offset;
  packet.size = len;
  uint64_t total_size = sizeof(struct arts_remote_get_put_packet_s) + len;
  arts_fill_packet_header(&packet.header, total_size,
                          ARTS_REMOTE_PUT_IN_DB_MSG);
  //    arts_remote_send_request_payload_async(rank, (char *)&packet,
  //    sizeof(packet), (char *)ptr, len);
  arts_remote_send_request_payload_async_free((int)rank, (char *)&packet,
                                              sizeof(packet), (char *)ptr, 0,
                                              len, arts_free);
}

void arts_remote_handle_put_in_db(void *pack) {
  struct arts_remote_get_put_packet_s *packet =
      (struct arts_remote_get_put_packet_s *)pack;
  void *data = (void *)(packet + 1);
  internal_put_in_db(data, packet->edt_guid, packet->db_guid, packet->slot,
                     packet->offset, packet->size, packet->epoch_guid,
                     arts_global_rank_id);
}

void arts_remote_signal_edt_with_ptr(arts_guid_t edt_guid, arts_guid_t db_guid,
                                     void *ptr, unsigned int size,
                                     unsigned int slot) {
  unsigned int rank = arts_guid_get_rank(edt_guid);
  ARTS_DEBUG("SEND NOW: %u -> %u", arts_global_rank_id, rank);
  uint64_t total_size =
      sizeof(struct arts_remote_signal_edt_with_ptr_packet_s) + size;
  char *buf = (char *)arts_malloc((size_t)total_size);
  struct arts_remote_signal_edt_with_ptr_packet_s *packet =
      (struct arts_remote_signal_edt_with_ptr_packet_s *)buf;
  packet->edt_guid = edt_guid;
  packet->db_guid = db_guid;
  packet->size = size;
  packet->slot = slot;
  arts_fill_packet_header(&packet->header, total_size,
                          ARTS_REMOTE_SIGNAL_EDT_WITH_PTR_MSG);
  memcpy(buf + sizeof(*packet), ptr, size);
  arts_remote_send_request_async((int)rank, buf, (unsigned int)total_size);
  arts_free(buf);
}

void arts_remote_handle_signal_edt_with_ptr(void *pack) {
  struct arts_remote_signal_edt_with_ptr_packet_s *packet =
      (struct arts_remote_signal_edt_with_ptr_packet_s *)pack;
  void *source = (void *)(packet + 1);
  internal_signal_edt(packet->edt_guid, packet->slot, packet->db_guid,
                      DB_MODE_PTR, source, packet->size);
}

void arts_remote_send(unsigned int rank, send_handler_t fun_ptr, void *args,
                      unsigned int size, bool free) {
  if (rank == arts_global_rank_id) {
    fun_ptr(args);
    if (free) {
      arts_free(args);
    }
    return;
  }
  struct arts_remote_send_s packet;
  packet.fun_ptr = fun_ptr;
  int total_size = (int)(sizeof(struct arts_remote_send_s) + size);
  arts_fill_packet_header(&packet.header, total_size, ARTS_REMOTE_SEND_MSG);

  if (free) {
    arts_remote_send_request_payload_async_free((int)rank, (char *)&packet,
                                                sizeof(packet), (char *)args, 0,
                                                size, arts_free);
  } else {
    arts_remote_send_request_payload_async((int)rank, (char *)&packet,
                                           sizeof(packet), (char *)args, size);
  }
}

void arts_remote_handle_send(void *pack) {
  struct arts_remote_send_s *packet = (struct arts_remote_send_s *)pack;
  void *args = (void *)(packet + 1);
  packet->fun_ptr(args);
}

void arts_remote_epoch_init_send(unsigned int rank, arts_guid_t epoch_guid,
                                 arts_guid_t edt_guid, unsigned int slot) {
  struct arts_remote_epoch_init_packet_s packet;
  packet.epoch_guid = epoch_guid;
  packet.edt_guid = edt_guid;
  packet.slot = slot;
  arts_fill_packet_header(&packet.header, sizeof(packet), ARTS_EPOCH_INIT_MSG);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_remote_handle_epoch_init_send(void *pack) {
  ARTS_DEBUG("Net Epoch Init Rec");
  struct arts_remote_epoch_init_packet_s *packet =
      (struct arts_remote_epoch_init_packet_s *)pack;
  arts_guid_t local_epoch_guid = packet->epoch_guid;
  create_epoch(&local_epoch_guid, packet->edt_guid, packet->slot);
  packet->epoch_guid = local_epoch_guid;
}

void arts_remote_epoch_init_pool_send(unsigned int rank, unsigned int pool_size,
                                      arts_guid_t start_guid,
                                      arts_guid_t pool_guid) {
  //    ARTS_INFO("Net Epoch Init Pool Send: %u %lu %lu", rank, start_guid,
  //    pool_guid);
  struct arts_remote_epoch_init_pool_packet_s packet;
  packet.pool_size = pool_size;
  packet.start_guid = start_guid;
  packet.pool_guid = pool_guid;
  arts_fill_packet_header(&packet.header, sizeof(packet),
                          ARTS_EPOCH_INIT_POOL_MSG);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_remote_handle_epoch_init_pool_send(void *pack) {
  struct arts_remote_epoch_init_pool_packet_s *packet =
      (struct arts_remote_epoch_init_pool_packet_s *)pack;
  arts_guid_t local_pool_guid = packet->pool_guid;
  arts_guid_t local_start_guid = packet->start_guid;
  arts_epoch_pool_t *pool =
      create_epoch_pool(&local_pool_guid, packet->pool_size, &local_start_guid);
  arts_link_epoch_pool_to_tls(pool);
  packet->pool_guid = local_pool_guid;
  packet->start_guid = local_start_guid;
}

void arts_remote_epoch_req(unsigned int rank, arts_guid_t guid) {
  struct arts_remote_guid_only_packet_s packet;
  packet.guid = guid;
  arts_fill_packet_header(&packet.header, sizeof(packet), ARTS_EPOCH_REQ_MSG);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_remote_handle_epoch_req(void *pack) {
  struct arts_remote_guid_only_packet_s *packet =
      (struct arts_remote_guid_only_packet_s *)pack;
  // For now the source and dest are the same...
  send_epoch(packet->guid, packet->header.rank, packet->header.rank);
}

void arts_remote_epoch_send(unsigned int rank, arts_guid_t guid,
                            unsigned int active, unsigned int finish) {
  struct arts_remote_epoch_send_packet_s packet;
  packet.epoch_guid = guid;
  packet.active = active;
  packet.finish = finish;
  arts_fill_packet_header(&packet.header, sizeof(packet), ARTS_EPOCH_SEND_MSG);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_remote_handle_epoch_send(void *pack) {
  struct arts_remote_epoch_send_packet_s *packet =
      (struct arts_remote_epoch_send_packet_s *)pack;
  reduce_epoch(packet->epoch_guid, packet->active, packet->finish);
}

void arts_remote_epoch_delete(unsigned int rank, arts_guid_t epoch_guid) {
  struct arts_remote_guid_only_packet_s packet;
  packet.guid = epoch_guid;
  arts_fill_packet_header(&packet.header, sizeof(packet),
                          ARTS_EPOCH_DELETE_MSG);
  arts_remote_send_request_async((int)rank, (char *)&packet, sizeof(packet));
}

void arts_remote_handle_epoch_delete(void *pack) {
  struct arts_remote_guid_only_packet_s *packet =
      (struct arts_remote_guid_only_packet_s *)pack;
  delete_epoch(packet->guid, NULL);
}

void arts_remote_db_rename(arts_guid_t new_guid, arts_guid_t old_guid) {
  unsigned int dest_rank = arts_guid_get_rank(old_guid);
  struct arts_remote_db_rename_s packet;
  packet.old_guid = old_guid;
  packet.new_guid = new_guid;
  packet.header.size = sizeof(packet);
  packet.header.message_type = ARTS_REMOTE_DB_RENAME_MSG;
  packet.header.rank = dest_rank;
  arts_remote_send_request_async((int)dest_rank, (char *)&packet,
                                 sizeof(packet));
}

void arts_remote_handle_db_rename(void *pack) {
  struct arts_remote_db_rename_s *packet =
      (struct arts_remote_db_rename_s *)pack;
  arts_db_rename_with_guid(packet->new_guid, packet->old_guid);
}

// RTT-based time synchronization for counter capture alignment
// External declarations for time sync state (defined in Counter.c)
extern volatile int64_t arts_counter_time_offset;
extern volatile bool arts_counter_time_sync_received;

// Worker sends sync request to master with its current timestamp (T1)
void arts_remote_time_sync_request(void) {
  struct arts_remote_time_sync_req_packet_s packet;
  packet.worker_send_time = arts_get_time_stamp(); // T1
  arts_fill_packet_header(&packet.header, sizeof(packet),
                          ARTS_REMOTE_TIME_SYNC_REQ_MSG);

  // Send to master
  arts_remote_send_request_async((int)arts_global_master_rank_id,
                                 (char *)&packet, sizeof(packet));
  ARTS_INFO("Time sync: Worker %u sent request to master %u at T1=%lu",
            arts_global_rank_id, arts_global_master_rank_id,
            packet.worker_send_time);
}

// Master handles sync request: records T2 and sends response with T1, T2
void arts_remote_handle_time_sync_req(void *pack) {
  struct arts_remote_time_sync_req_packet_s *req =
      (struct arts_remote_time_sync_req_packet_s *)pack;
  uint64_t master_recv_time = arts_get_time_stamp(); // T2

  struct arts_remote_time_sync_resp_packet_s resp;
  resp.worker_send_time = req->worker_send_time; // Echo T1
  resp.master_recv_time = master_recv_time;      // T2
  arts_fill_packet_header(&resp.header, sizeof(resp),
                          ARTS_REMOTE_TIME_SYNC_RESP_MSG);

  // Send response back to the requesting worker
  arts_remote_send_request_async((int)req->header.rank, (char *)&resp,
                                 sizeof(resp));
  ARTS_INFO("Time sync: Master received request from rank %u, T1=%lu, T2=%lu",
            req->header.rank, req->worker_send_time, master_recv_time);
}

// Worker handles sync response: calculates offset using RTT
void arts_remote_handle_time_sync_resp(void *pack) {
  struct arts_remote_time_sync_resp_packet_s *resp =
      (struct arts_remote_time_sync_resp_packet_s *)pack;
  uint64_t worker_recv_time = arts_get_time_stamp(); // T3

  uint64_t ntp_t1 = resp->worker_send_time;
  uint64_t ntp_t2 = resp->master_recv_time;
  uint64_t ntp_t3 = worker_recv_time;

  // RTT = T3 - T1 (round-trip time in worker's clock)
  // One-way delay estimate = RTT / 2 (assuming symmetric network)
  // At T2 (master clock), worker clock was approximately T1 + RTT/2
  // offset = workerTime - masterTime = (T1 + RTT/2) - T2 = (T1 + T3)/2 - T2
  int64_t offset = (int64_t)((ntp_t1 + ntp_t3) / 2) - (int64_t)ntp_t2;

  __atomic_store_n(&arts_counter_time_offset, offset, __ATOMIC_RELAXED);
  __atomic_store_n(&arts_counter_time_sync_received, true, __ATOMIC_RELEASE);

  uint64_t rtt = ntp_t3 - ntp_t1;
  ARTS_INFO("Time sync: Worker %u received response, T1=%lu, T2=%lu, T3=%lu, "
            "RTT=%lu ns (%.3f ms), offset=%ld ns (%.3f ms)",
            arts_global_rank_id, ntp_t1, ntp_t2, ntp_t3, rtt,
            (double)rtt / 1000000.0, offset, (double)offset / 1000000.0);
}
