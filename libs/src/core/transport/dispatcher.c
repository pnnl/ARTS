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

#include <assert.h>    /* lazy INVALIDATE direct-call invariant assert */
#include <semaphore.h> /* sem_post (LOCK_RELEASE_ACK inline wake) */
#include <string.h> /* memcpy (WRITEBACK inline-payload copy into OoO args) */
#include <unistd.h>

#include "arts.h"
#include "arts/coherence/coherence.h" /* arts_handler_db_ownership_* bodies */
#include "arts/coherence/handlers.h"
#include "arts/counter/counter.h" /* arts_handler_time_sync_* */
#include "arts/db.h"
#include "arts/edt.h"
#include "arts/event.h"
#include "arts/gas/route_table.h" /* arts_route_table_lookup_db (Cat-C lookup-acquire) */
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/transport/outbox.h"
#include "arts/transport/protocol.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h" /* arts_shared_get / arts_shared_release */

#ifdef SEQUENCENUMBERS
uint64_t *rec_seq_numbers;
#endif

/*
 * arts_transport_broadcast_shutdown — First step of the shutdown protocol.
 *
 * Enqueue a header-only MSG_SHUTDOWN to every other rank.
 * The sender thread drains the outbox; the caller should then wait for
 * arts_node_info.outbox_pending to reach zero (see wait_for_outbox_drain
 * in threads.c) before proceeding to local shutdown.
 */
void arts_transport_broadcast_shutdown(void) {
  if (arts_global_rank_count <= 1) {
    return;
  }
  for (unsigned int r = 0; r < arts_global_rank_count; r++) {
    if (r == arts_global_rank_id) {
      continue;
    }
    struct arts_msg_header_s packet;
    arts_fill_packet_header(&packet, sizeof(packet), MSG_SHUTDOWN);
    arts_transport_send_async((int)r, (char *)&packet, sizeof(packet));
  }
}

void arts_transport_cleanup(void) {
  arts_outbox_cleanup();
#ifdef SEQUENCENUMBERS
  arts_free(rec_seq_numbers);
  rec_seq_numbers = NULL;
#endif
}

void arts_transport_setup(struct arts_config_s *config) {
  // ASYNC Message Queue Init
  arts_socket_setup(config);
  arts_outbox_init(arts_global_rank_count * config->port_count);
#ifdef SEQUENCENUMBERS
  rec_seq_numbers =
      (uint64_t *)arts_calloc(arts_global_rank_count, sizeof(uint64_t));
#endif
}

void arts_transport_dispatch_packet(struct arts_msg_header_s *packet) {
#ifdef SEQUENCENUMBERS
  /* Wire-ordering check — wire RX entry only.  A self-loopback packet never
   * traverses the wire and carries no per-sender sequence number (it bypasses
   * the outbox that stamps them), so it enters via arts_transport_dispatch_body
   * directly and is exempt from this check. */
  uint64_t exp_seq_number =
      __sync_fetch_and_add(&rec_seq_numbers[packet->seq_rank], 1U);
  if (exp_seq_number != packet->seq_num) {
    ARTS_DEBUG(
        "MESSAGE RECIEVED OUT OF ORDER exp: %lu rec: %lu source: %u type: %d",
        exp_seq_number, packet->seq_num, packet->rank, packet->message_type);
  }
#endif
  arts_transport_dispatch_body(packet);
}

void arts_transport_dispatch_body(struct arts_msg_header_s *packet) {
  switch (packet->message_type) {
  case MSG_SHUTDOWN: {
    ARTS_INFO("Node %u: Received shutdown message from node %u",
              arts_global_rank_id, packet->rank);
    /* Passive shutdown entry (spec Cat E): we received SHUTDOWN_MSG from
     * another rank, so the dispatcher calls the lightweight RX handler
     * directly — idempotent CAS gate + worker stop, no re-broadcast, no
     * drain-wait.  The main thread handles network stop and bounded join
     * after the worker loop exits. */
    arts_handler_shutdown();
    break;
  }
  case MSG_EDT_SATISFY_SLOT: {
    struct arts_msg_edt_satisfy_slot_packet_s *pack =
        (struct arts_msg_edt_satisfy_slot_packet_s *)(packet);
    /* RX is at the EDT's home — route straight into the OoO engine, same as
     * arts_edt_satisfy_slot's home==self branch.  DB_MODE_PTR carries an inline
     * payload right after the header (size > 0); other modes deliver a
     * GUID/value reference only (size == 0).  The single mode-discriminated
     * OOO_EDT_SATISFY_SLOT kind lays the PTR inline payload immediately after
     * the args struct so the deferred payload reconstructs it; the handler
     * branches on mode to locate it. */
    uint32_t payload = (pack->mode == DB_MODE_PTR) ? pack->size : 0u;
    uint32_t asz =
        (uint32_t)sizeof(struct arts_ooo_args_edt_satisfy_s) + payload;
    char *buf = (char *)arts_malloc(asz);
    struct arts_ooo_args_edt_satisfy_s *a =
        (struct arts_ooo_args_edt_satisfy_s *)buf;
    a->edt_guid = pack->edt;
    a->data_guid = pack->db;
    a->slot = pack->slot;
    a->mode = pack->mode;
    a->size = payload;
    if (payload > 0) {
      memcpy(buf + sizeof(*a), (void *)(pack + 1), payload);
    }
    arts_ooo_dispatch_or_defer_guid(pack->edt, OOO_EDT_SATISFY_SLOT, buf, asz);
    arts_free(buf);
    break;
  }
  case MSG_EVENT_SATISFY_SLOT: {
    struct arts_msg_event_satisfy_slot_packet_s *pack =
        (struct arts_msg_event_satisfy_slot_packet_s *)(packet);
    /* RX is at the event's home — route straight into the OoO engine, same as
     * arts_event_satisfy_slot's home==self branch. */
    struct arts_ooo_args_event_satisfy_s a = {
        .event_guid = pack->event, .data_guid = pack->db, .slot = pack->slot};
    arts_ooo_dispatch_or_defer_guid(pack->event, OOO_EVENT_SATISFY_SLOT, &a,
                                    sizeof(a));
    break;
  }
  case MSG_EVENT_ADD_DEPENDENCE: {
    ARTS_DEBUG("Dependence Received");
    struct arts_msg_add_dependence_packet_s *pack =
        (struct arts_msg_add_dependence_packet_s *)(packet);
    /* The wire message means source == event (an EDT/DB source satisfies
     * immediately and is never shipped as ADD_DEPENDENCE), and RX is at the
     * source event's home — route straight into the OoO engine, same as
     * arts_event_add_dependence's home==self branch.  No generic src-kind
     * re-derivation. */
    struct arts_ooo_args_event_add_dep_s a = {.source = pack->source,
                                              .destination = pack->destination,
                                              .slot = pack->slot,
                                              .mode = pack->mode};
    arts_ooo_dispatch_or_defer_guid(pack->source, OOO_EVENT_ADD_DEPENDENCE, &a,
                                    sizeof(a));
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
   * OWNERSHIP_REQUEST / RELEASE_OWNERSHIP: shared between the eager and lazy
   * protocols (both use per-DB exclusive ownership), but MRMW has
   * no such concept.  Fatal in MRMW builds to catch binary mode mismatch.
   * INVALIDATE_NOTICE is handled in its own three-model block below (eager =
   * Cat-B defer; lazy = direct, never deferred).
   */
#if defined(ARTS_PROTOCOL_MRMW) || defined(ARTS_PROTOCOL_LOCK)
  case MSG_DB_OWNERSHIP_REQUEST: {
    ARTS_ERROR("MRMW/LOCK build received exclusivity message type %d from rank "
               "%u — protocol has no OWNERSHIP_REQUEST; binary mode mismatch?",
               packet->message_type, packet->rank);
    break;
  }
#else  /* MRNEW/MRSW eager and lazy: full handlers */
  case MSG_DB_OWNERSHIP_REQUEST: {
    ARTS_DEBUG("Coh OWNERSHIP_REQUEST Received");
    struct arts_msg_ownership_request_packet_s *pack =
        (struct arts_msg_ownership_request_packet_s *)(packet);
    struct arts_ooo_args_db_ownership_request_s args = {
        .requester = pack->header.rank,
        .db_guid = pack->db_guid,
    };
    arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_OWNERSHIP_REQUEST,
                                    &args, sizeof(args));
    break;
  }
#endif /* ARTS_PROTOCOL_MRMW */
  /* INVALIDATE_NOTICE — protocol-split.
   *   eager/lazy : NOT deferred.  Home publishes the invalidate target
   *           (rw_holder) only after that rank's cache install — the CONFIRM
   *           owner-swap (post-install in both timings) or the DB_CREATE on the
   *           creator — so the target's cache is provably already installed
   * when INVALIDATE arrives.  Call the pure handler body directly with the
   *           looked-up cache.  (The before-install GRANT/INVALIDATE reorder
   *           that once forced eager through the OoO engine is gone: EAGER no
   *           longer flips rw_holder before install.)
   *   MRMW: no ownership transfer (caught by the fatal group above). */
#if defined(ARTS_PROTOCOL_MRMW) || defined(ARTS_PROTOCOL_LOCK)
  case MSG_DB_OWNERSHIP_INVALIDATE: {
    ARTS_ERROR("MRMW/LOCK build received INVALIDATE from rank %u — "
               "protocol has no ownership invalidate; binary mode mismatch?",
               packet->rank);
    break;
  }
#else  /* MRNEW/MRSW: eager + lazy share the direct-call body */
  case MSG_DB_OWNERSHIP_INVALIDATE: {
    ARTS_DEBUG("Coh INVALIDATE_NOTICE Received");
    struct arts_msg_ownership_invalidate_packet_s *pack =
        (struct arts_msg_ownership_invalidate_packet_s *)(packet);
    struct arts_ooo_args_db_ownership_invalidate_s args = {
        .db_guid = pack->db_guid,
        .new_owner_rank = pack->new_owner_rank,
    };
    /* Pin the db_s for the handler's duration (the embedded cache is its FIRST
     * member, offset 0) so a concurrent DESTROY on another receiver thread
     * cannot free it mid-handler.  The home publishes the rw_holder target only
     * after that rank's cache install, so the db_s is normally present.  But a
     * destroy may have NULLed the route slot between that publish and this
     * INVALIDATE arriving (destroy-during-acquire); the destroy drains every
     * waiter itself, so an orphaned INVALIDATE for a torn-down DB is simply
     * dropped. */
    arts_shared_ptr_t db_h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
    if (db != NULL) {
      arts_handler_db_ownership_invalidate(db, &args);
    }
    arts_shared_release(&db_h);
    break;
  }
#endif /* model dispatch for MSG_DB_OWNERSHIP_INVALIDATE */
#if defined(ARTS_PROTOCOL_LOCK)
  case MSG_DB_SNAPSHOT_REQUEST:
  case MSG_DB_SNAPSHOT_RESPONSE: {
    ARTS_ERROR("LOCK build received snapshot message type %d from rank %u — "
               "LOCK has no RO snapshot protocol; binary mode mismatch?",
               packet->message_type, packet->rank);
    break;
  }
#else  /* MRNEW/MRSW/MRMW: snapshot handlers */
  case MSG_DB_SNAPSHOT_REQUEST: {
    ARTS_DEBUG("Coh GET_DATA Received");
    struct arts_msg_snapshot_request_packet_s *pack =
        (struct arts_msg_snapshot_request_packet_s *)(packet);
    struct arts_ooo_args_db_snapshot_request_s args = {
        .requester = pack->header.rank,
        .db_guid = pack->db_guid,
        .edt_guid = pack->edt_guid,
        .slot = pack->slot,
    };
    arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_SNAPSHOT_REQUEST,
                                    &args, sizeof(args));
    break;
  }
  case MSG_DB_SNAPSHOT_RESPONSE: {
    ARTS_DEBUG("Coh DATA_RESPONSE Received");
    struct arts_msg_snapshot_response_packet_s *pack =
        (struct arts_msg_snapshot_response_packet_s *)(packet);
    /* header.size is the peer-supplied total on-wire byte count; a malformed
     * value below the fixed struct size would underflow the unsigned payload
     * length and drive an OOB copy/allocation. Drop such packets. */
    if (pack->header.size < sizeof(*pack)) {
      break;
    }
    const void *data = (const char *)pack + sizeof(*pack);
    uint64_t data_size = pack->header.size - sizeof(*pack);
    /* Cat-C lookup-acquire-or-drop: HIT runs the pure body against the
     * ref-pinned home db_s; MISS (DB destroyed / slot NULL-stored) silently
     * drops — the parked EDT this 1:1 response would resume was torn down. */
    struct arts_db_snapshot_response_args_s args = {
        .edt_guid = pack->edt_guid,
        .slot = pack->slot,
        .data_present = pack->data_present,
        .version = pack->version,
        .data = data_size > 0 ? data : NULL,
        .data_size = data_size,
    };
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_snapshot_response(db, &args);
    }
    arts_shared_release(&h);
    break;
  }
#endif /* ARTS_PROTOCOL_LOCK */
  case MSG_DB_CREATE: {
    ARTS_DEBUG("Coh DB_CREATE_COHERENT Received");
    arts_handler_db_create(
        (struct arts_msg_db_create_coherent_packet_s *)(packet));
    break;
  }
  case MSG_DB_DESTROY: {
    ARTS_DEBUG("Coh DESTROY_REQ Received");
    struct arts_msg_destroy_packet_s *pack =
        (struct arts_msg_destroy_packet_s *)(packet);
    struct arts_ooo_args_db_destroy_s args = {
        .requester = pack->header.rank,
        .db_guid = pack->db_guid,
    };
    arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_DESTROY, &args,
                                    sizeof(args));
    break;
  }
  case MSG_DB_CACHE_DESTROY: {
    ARTS_DEBUG("Coh DESTROY_NOTIFY Received");
    struct arts_msg_cache_destroy_packet_s *pack =
        (struct arts_msg_cache_destroy_packet_s *)(packet);
    /* Cat-C lookup-acquire-or-drop: HIT wakes parked waiters + detaches the
     * cb; MISS (already torn down on this rank) silently drops (idempotent). */
    struct arts_db_cache_destroy_args_s args = {.db_guid = pack->db_guid};
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_cache_destroy(db, &args);
    }
    arts_shared_release(&h);
    break;
  }
  /* OWNERSHIP_RESPONSE: the single ownership-transfer wire message.  eager =
   * GRANT (buffer payload); lazy = TRANSFER_OWNERSHIP (map + buffer); MRMW
   * has no ownership transfer and fatals to catch a binary mode mismatch. */
#if defined(ARTS_PROTOCOL_MRMW) || defined(ARTS_PROTOCOL_LOCK)
  case MSG_DB_OWNERSHIP_RESPONSE: {
    ARTS_ERROR("MRMW/LOCK build received OWNERSHIP_RESPONSE from rank %u — "
               "protocol has no ownership transfer; binary mode mismatch?",
               packet->rank);
    break;
  }
#else  /* MRNEW/MRSW eager and lazy: one converged layout */
  case MSG_DB_OWNERSHIP_RESPONSE: {
    ARTS_DEBUG("Coh OWNERSHIP_RESPONSE Received");
    /* Payload (map + data) immediately follows the header in the contiguous
     * wire buffer; the handler parses it from the full packet.  Both timings
     * share the lazy-style layout (EAGER carries map_entry_count=0). */
    arts_handler_db_ownership_response((void *)packet, (size_t)packet->size);
    break;
  }
#endif /* model dispatch for MSG_DB_OWNERSHIP_RESPONSE */
  /* WRITEBACK + WRITEBACK_ACK: used by the eager protocol and MRMW
   * (sync release writeback).  Fatal in the lazy protocol — lazy uses
   * async transfer, not synchronous writeback. */
#if defined(ARTS_TIMING_LAZY) || defined(ARTS_PROTOCOL_LOCK)
  case MSG_DB_WRITEBACK:
  case MSG_DB_WRITEBACK_ACK: {
    ARTS_ERROR("lazy/LOCK build received writeback message type %d from rank "
               "%u — protocol has no synchronous WRITEBACK; binary mode "
               "mismatch?",
               packet->message_type, packet->rank);
    break;
  }
#else  /* MRNEW/MRSW eager and MRMW: full handlers */
  case MSG_DB_WRITEBACK: {
    ARTS_DEBUG("Coh WRITEBACK Received");
    struct arts_msg_writeback_packet_s *pack =
        (struct arts_msg_writeback_packet_s *)(packet);
    /* header.size is the peer-supplied total on-wire byte count; a malformed
     * value below the fixed struct size would underflow the unsigned payload
     * length and drive an OOB copy/allocation. Drop such packets. */
    if (pack->header.size < sizeof(*pack)) {
      break;
    }
    const void *data = (const char *)pack + sizeof(*pack);
    uint64_t data_size = pack->header.size - sizeof(*pack);
    /* WRITEBACK carries an inline data payload: lay it immediately after the
     * args struct so the deferred OoO payload reconstructs it, and pass
     * sizeof(struct) + data_size as the args size.  The pure body reads the
     * payload back from (char *)args + sizeof(struct). */
    uint32_t asz =
        (uint32_t)(sizeof(struct arts_ooo_args_db_writeback_s) + data_size);
    char *abuf = (char *)arts_malloc(asz);
    struct arts_ooo_args_db_writeback_s *args =
        (struct arts_ooo_args_db_writeback_s *)abuf;
    args->releaser = pack->header.rank;
    args->db_guid = pack->db_guid;
    args->version = pack->version;
    args->cv = pack->cv;
    args->data_size = data_size;
    if (data_size > 0) {
      memcpy(abuf + sizeof(*args), data, data_size);
    }
    arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_WRITEBACK, abuf, asz);
    arts_free(abuf);
    break;
  }
  case MSG_DB_WRITEBACK_ACK: {
    ARTS_DEBUG("Coh WRITEBACK_ACK Received");
    struct arts_msg_writeback_ack_packet_s *pack =
        (struct arts_msg_writeback_ack_packet_s *)(packet);
    /* Cat-C SPECIAL — sem-post on BOTH HIT and MISS.  The wake is a
     * cache-independent pointer-identity sem-post on cv (the releaser's
     * stack-local sem_t); a torn-down home cache must NOT drop the ACK or the
     * blocked releaser hangs.  The body ignores item_v (the post needs only
     * cv), so call it unconditionally — db may be NULL on a MISS and the body
     * never dereferences it.  Still take the ref handle so the lookup-acquire
     * pattern is uniform with the other Cat-C handlers. */
    struct arts_db_writeback_ack_args_s args = {.cv = pack->cv};
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    arts_handler_db_writeback_ack(db, &args);
    arts_shared_release(&h);
    break;
  }
#endif /* ARTS_TIMING_LAZY */
  case MSG_EVENT_DESTROY: {
    ARTS_DEBUG("Event Destroy Received");
    /* Decode the GUID and route into the OoO engine.  A DESTROY that races
     * ahead of the event's CREATE (before-create wire reorder) defers on the
     * slot and replays on the create handler's drain; otherwise the pure body
     * (arts_handler_event_destroy) runs inline on the live event. */
    struct arts_msg_guid_only_packet_s *pack =
        (struct arts_msg_guid_only_packet_s *)(packet);
    struct arts_ooo_args_event_destroy_s args = {.guid = pack->guid};
    arts_ooo_dispatch_or_defer_guid(pack->guid, OOO_EVENT_DESTROY, &args,
                                    sizeof(args));
    break;
  }
  case MSG_EDT_DESTROY: {
    ARTS_DEBUG("EDT Destroy Received");
    /* Decode the GUID and route into the OoO engine (symmetric with
     * MSG_EVENT_DESTROY).  Before-create reorder defers; otherwise the pure
     * body (arts_handler_edt_destroy) runs inline on the live EDT. */
    struct arts_msg_guid_only_packet_s *pack =
        (struct arts_msg_guid_only_packet_s *)(packet);
    struct arts_ooo_args_edt_destroy_s args = {.guid = pack->guid};
    arts_ooo_dispatch_or_defer_guid(pack->guid, OOO_EDT_DESTROY, &args,
                                    sizeof(args));
    break;
  }
  /* ===== lazy-only message dispatch ========================================
   * These slots are only sent between ranks compiled with the lazy coherence
   * protocol (MRNEW/MRSW).  The LOCK protocol has its own lazy messages
   * (FORWARD/DELIVER/CONFIRM/RORET) dispatched in the ARTS_PROTOCOL_LOCK block
   * below.  The eager build fatals immediately to catch a binary mode mismatch.
   */
#if defined(ARTS_TIMING_LAZY) && !defined(ARTS_PROTOCOL_LOCK)
  case MSG_DB_SNAPSHOT_REDIRECT: {
    ARTS_DEBUG("Lazy REDIRECT_RO Received");
    struct arts_msg_snapshot_redirect_packet_s *pack =
        (struct arts_msg_snapshot_redirect_packet_s *)(packet);
    /* Cat-C lookup-acquire-or-{DESTROY_NOTIFY}: HIT serves DATA_RESPONSE from
     * the ref-pinned owner-side db_s; MISS (DB destroyed / not yet installed on
     * this rank) sends DESTROY_NOTIFY to the requester so its parked RO waiter
     * wakes and observes DB_DESTROYED rather than hanging. */
    struct arts_db_snapshot_redirect_args_s args = {
        .db_guid = pack->db_guid,
        .edt_guid = pack->edt_guid,
        .requester_rank = pack->requester_rank,
        .slot = pack->slot,
    };
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_snapshot_redirect(db, &args);
    } else {
      arts_send_db_cache_destroy(pack->requester_rank, pack->db_guid);
    }
    arts_shared_release(&h);
    break;
  }
  case MSG_DB_OWNERSHIP_CONFIRM_ACK: {
    ARTS_DEBUG("Lazy CONFIRM_ACK Received");
    struct arts_msg_ownership_confirm_ack_packet_s *pack =
        (struct arts_msg_ownership_confirm_ack_packet_s *)(packet);
    /* Cat-C lookup-acquire-or-drop: HIT runs the confirm_ack body on the
     * ref-pinned db_s.  MISS (DB destroyed): the gated waiters are woken by the
     * destroy fan-out, so a miss silently drops. */
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_ownership_confirm_ack(db, pack);
    }
    arts_shared_release(&h);
    break;
  }
#else  /* eager build or LOCK (LOCK has its own lazy messages below) */
  case MSG_DB_SNAPSHOT_REDIRECT:
  case MSG_DB_OWNERSHIP_CONFIRM_ACK: {
    ARTS_ERROR(
        "non-MRNEW/MRSW-lazy build received MRNEW/MRSW-lazy message type %d "
        "from rank %u — binary mode mismatch?",
        packet->message_type, packet->rank);
    break;
  }
#endif /* ARTS_TIMING_LAZY && !ARTS_PROTOCOL_LOCK */
  /* OWNERSHIP_CONFIRM: both timings (new owner C → home A flips rw_holder +
   * advances the round).  LAZY additionally replies with CONFIRM_ACK; EAGER's
   * home handler does not (the new owner already drained at
   * OWNERSHIP_RESPONSE). MRMW has no ownership transfer and fatals. */
#if defined(ARTS_PROTOCOL_MRMW) || defined(ARTS_PROTOCOL_LOCK)
  case MSG_DB_OWNERSHIP_CONFIRM: {
    ARTS_ERROR("MRMW/LOCK build received OWNERSHIP_CONFIRM from rank %u — "
               "protocol has no ownership transfer; binary mode mismatch?",
               packet->rank);
    break;
  }
#else /* MRNEW/MRSW */
  case MSG_DB_OWNERSHIP_CONFIRM: {
    ARTS_DEBUG("Coh CONFIRM Received");
    struct arts_msg_ownership_confirm_packet_s *pack =
        (struct arts_msg_ownership_confirm_packet_s *)(packet);
    /* Cat-C lookup-acquire-or-drop: HIT advances the transfer round on the
     * ref-pinned home db_s; MISS (DB destroyed) silently drops.  EAGER's
     * handler reads pending_install_owner and ignores args, so pass NULL. */
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
#ifdef ARTS_TIMING_LAZY
      struct arts_db_ownership_response_ack_args_s args = {
          .db_guid = pack->db_guid,
          .version = pack->version,
      };
      arts_handler_db_ownership_confirm(db, &args);
#else
      arts_handler_db_ownership_confirm(db, NULL);
#endif
    }
    arts_shared_release(&h);
    break;
  }
#endif /* OWNERSHIP_CONFIRM model dispatch */
  /* LOCK has its own REQUEST wire (both timings) and timing-specific grant/
   * release messages.  The legacy coherence cases are excluded from LOCK builds
   * (each is already guarded above). */
#ifdef ARTS_PROTOCOL_LOCK
  /* MSG_DB_LOCK_REQUEST is shared: both EAGER and LAZY home-dispatch via OoO
   * (a REQUEST can arrive before the home db_s is installed on a remote-create
   * lazy-install path). */
  case MSG_DB_LOCK_REQUEST: {
    ARTS_DEBUG("Coh LOCK_REQUEST Received");
    struct arts_msg_lock_request_packet_s *pack =
        (struct arts_msg_lock_request_packet_s *)(packet);
    struct arts_ooo_args_db_lock_request_s args = {
        .requester = pack->header.rank,
        .db_guid = pack->db_guid,
        .mode = pack->mode,
    };
    arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_LOCK_REQUEST, &args,
                                    sizeof(args));
    break;
  }
#ifdef ARTS_TIMING_EAGER
  /* EAGER-only: synchronous grant/release/release-ack round-trip. */
  case MSG_DB_LOCK_GRANT: {
    ARTS_DEBUG("Coh LOCK_GRANT Received");
    /* Cat-C: the grant receiver always sent its own REQUEST first, so its cache
     * exists.  Payload (data) follows the header in the contiguous buffer; the
     * handler parses it from the full packet. */
    arts_handler_db_lock_grant((void *)packet, (size_t)packet->size);
    break;
  }
  case MSG_DB_LOCK_RELEASE: {
    ARTS_DEBUG("Coh LOCK_RELEASE Received");
    struct arts_msg_lock_release_packet_s *pack =
        (struct arts_msg_lock_release_packet_s *)(packet);
    /* header.size is the peer-supplied total on-wire byte count; a malformed
     * value below the fixed struct size would underflow the unsigned payload
     * length and drive an OOB copy/allocation. Drop such packets. */
    if (pack->header.size < sizeof(*pack)) {
      break;
    }
    const void *data = (const char *)pack + sizeof(*pack);
    uint64_t data_size = pack->header.size - sizeof(*pack);
    /* RW release carries an inline writeback payload; lay it after the args
     * struct so the deferred OoO payload reconstructs it, pass total size.
     * Also decode cv (ACK token) and version for monotone buf_install. */
    uint32_t asz =
        (uint32_t)(sizeof(struct arts_ooo_args_db_lock_release_s) + data_size);
    char *abuf = (char *)arts_malloc(asz);
    struct arts_ooo_args_db_lock_release_s *args =
        (struct arts_ooo_args_db_lock_release_s *)abuf;
    args->releaser = pack->header.rank;
    args->db_guid = pack->db_guid;
    args->mode = pack->mode;
    args->data_size = data_size;
    args->cv = pack->cv;
    args->version = pack->version;
    if (data_size > 0) {
      memcpy(abuf + sizeof(*args), data, data_size);
    }
    arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_LOCK_RELEASE, abuf,
                                    asz);
    arts_free(abuf);
    break;
  }
  case MSG_DB_LOCK_RELEASE_ACK: {
    ARTS_DEBUG("Coh LOCK_RELEASE_ACK Received");
    struct arts_msg_lock_release_ack_packet_s *pack =
        (struct arts_msg_lock_release_ack_packet_s *)(packet);
    /* Cat-C SPECIAL — pointer-identity sem_post on cv directly.  The wake
     * is cache-independent: a torn-down home cache must NOT drop the ACK or
     * the blocked releaser hangs (await_writeback_ack would spin forever).
     * arts_handler_db_writeback_ack is not linked in the LOCK build (it
     * lives in MRNEW/MRSW/MRMW TUs), so inline the sem_post here. */
    if (pack->cv != 0) {
      sem_post((sem_t *)(uintptr_t)pack->cv);
    }
    break;
  }
#endif /* ARTS_TIMING_EAGER */
#ifdef ARTS_TIMING_LAZY
  /* LAZY-only: async migration/serve protocol.
   * All four are Cat-C (direct): their target is always installed by the time
   * the message arrives — FORWARD/DELIVER targets a cache built at acquire;
   * CONFIRM/RORET reach home only after home sent FORWARD, so home db_s exists.
   */
  case MSG_DB_LOCK_FORWARD: {
    ARTS_DEBUG("Coh LOCK_FORWARD Received");
    arts_handler_db_lock_forward((void *)packet);
    break;
  }
  case MSG_DB_LOCK_DELIVER: {
    ARTS_DEBUG("Coh LOCK_DELIVER Received");
    arts_handler_db_lock_deliver((void *)packet, (size_t)packet->size);
    break;
  }
  case MSG_DB_LOCK_CONFIRM: {
    ARTS_DEBUG("Coh LOCK_CONFIRM Received");
    arts_handler_db_lock_confirm((void *)packet);
    break;
  }
  case MSG_DB_LOCK_RORET: {
    ARTS_DEBUG("Coh LOCK_RORET Received");
    arts_handler_db_lock_roret((void *)packet);
    break;
  }
#endif /* ARTS_TIMING_LAZY */
#endif /* ARTS_PROTOCOL_LOCK */
  default: {
    ARTS_INFO("Unknown Packet %d %d %d", packet->message_type, packet->size,
              packet->rank);
    arts_shutdown();
    arts_runtime_stop();
  }
  }
}
