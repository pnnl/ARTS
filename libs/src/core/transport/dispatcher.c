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

#include <assert.h>    /* WB-write-policy INVALIDATE direct-call invariant assert */
#include <semaphore.h> /* sem_post (EXCL_RELEASE_ACK inline wake) */
#include <string.h> /* memcpy (PUBLISH inline-payload copy into OoO args) */
#include <unistd.h>

#include "arts.h"
#include "arts/coherence/coherence.h" /* arts_handler_db_grant_* bodies */
#include "arts/coherence/handlers.h"
#include "arts/counter/counter.h" /* arts_handler_time_sync_* */
#include "arts/db.h"
#include "arts/edt.h"
#include "arts/event.h"
#include "arts/gas/route_table.h" /* arts_route_table_lookup_db (Cat-C lookup-acquire) */
#include "arts/memory/regpool.h" /* push-rendezvous landing alloc */
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/transport/net.h"
#include "arts/transport/protocol.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h" /* arts_shared_get / arts_shared_release */

#ifdef SEQUENCENUMBERS
uint64_t *rec_seq_numbers;
#endif

/*
 * arts_transport_broadcast_shutdown — First step of the shutdown protocol.
 *
 * Inject a header-only MSG_SHUTDOWN onto the fabric to every other rank (same
 * path as every other message).  The caller then waits (bounded) on
 * arts_net_drain_outstanding for those sends to complete before proceeding to
 * local shutdown.
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
  arts_loopback_cleanup();
#ifdef SEQUENCENUMBERS
  arts_free(rec_seq_numbers);
  rec_seq_numbers = NULL;
#endif
}

void arts_transport_setup(struct arts_config_s *config) {
  /* Establish the bootstrap TCP mesh.  The outbound queue the outbox once
   * carved here is gone — each producing thread injects its own sends directly
   * onto the fabric. */
  arts_socket_setup(config);
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
     * arts_edt_satisfy_slot's home==self branch. */
    struct arts_ooo_args_edt_satisfy_s a = {.edt_guid = pack->edt,
                                            .data_guid = pack->db,
                                            .slot = pack->slot,
                                            .mode = pack->mode};
    arts_ooo_dispatch_or_defer_guid(pack->edt, OOO_EDT_SATISFY_SLOT, &a,
                                    sizeof(a));
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
   * Three handlers (GRANT / PUBLISH / SNAPSHOT_RESPONSE) carry trailing
   * payload right after sizeof(struct ...); pass that pointer + size as
   * the data/data_size arguments.
   *
   * GRANT_REQUEST: shared between the WT and WB write policies (both use
   * the migrating grant), but WRF_VAL has no such concept.  Fatal in
   * WRF_VAL builds to catch binary mode mismatch.
   * GRANT_INVALIDATE is handled in its own three-model block below (WT =
   * Cat-B defer; WB = direct, never deferred).
   */
#if defined(ARTS_PROTOCOL_WRF_VAL) || defined(ARTS_PROTOCOL_EXCL)
  case MSG_DB_GRANT_REQUEST: {
    ARTS_ERROR("WRF_VAL/EXCL build received exclusivity message type %d from rank "
               "%u — this protocol has no migrating grant; binary mode mismatch?",
               packet->message_type, packet->rank);
    break;
  }
#else  /* VAL WT and WB: full handlers */
  case MSG_DB_GRANT_REQUEST: {
    ARTS_DEBUG("Coh GRANT_REQUEST Received");
    struct arts_msg_grant_request_packet_s *pack =
        (struct arts_msg_grant_request_packet_s *)(packet);
    struct arts_ooo_args_db_grant_request_s args = {
        .requester = pack->header.rank,
        .db_guid = pack->db_guid,
        .rdzv = {.addr = pack->rdzv.addr,
                 .key = pack->rdzv.key,
                 .txid = pack->rdzv.txid,
                 .cookie = pack->rdzv.cookie},
    };
    arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_GRANT_REQUEST,
                                    &args, sizeof(args));
    break;
  }
#endif /* ARTS_PROTOCOL_WRF_VAL */
  /* GRANT_INVALIDATE — protocol-split.
   *   WT/WB: NOT deferred.  Home publishes the invalidate target
   *           (rw_holder) only after that rank's cache install — the CONFIRM
   *           owner-swap (post-install in both write policies) or the DB_CREATE on the
   *           creator — so the target's cache is provably already installed
   * when INVALIDATE arrives.  Call the pure handler body directly with the
   *           looked-up cache.  (The before-install GRANT/INVALIDATE reorder
   *           that once forced WT through the OoO engine is gone: WT no
   *           longer flips rw_holder before install.)
   *   WRF_VAL: no ownership transfer (caught by the fatal group above). */
#if defined(ARTS_PROTOCOL_WRF_VAL) || defined(ARTS_PROTOCOL_EXCL)
  case MSG_DB_GRANT_INVALIDATE: {
    ARTS_ERROR("WRF_VAL/EXCL build received INVALIDATE from rank %u — "
               "this protocol has no migrating grant; binary mode mismatch?",
               packet->rank);
    break;
  }
#else /* every grant-bearing arm (VAL, INV) shares the direct-call body */
  case MSG_DB_GRANT_INVALIDATE: {
    ARTS_DEBUG("Coh GRANT_INVALIDATE Received");
    struct arts_msg_grant_invalidate_packet_s *pack =
        (struct arts_msg_grant_invalidate_packet_s *)(packet);
    struct arts_ooo_args_db_grant_invalidate_s args = {
        .db_guid = pack->db_guid,
        .new_owner_rank = pack->new_owner_rank,
        .new_owner_rdzv = {.addr = pack->new_owner_rdzv.addr,
                           .key = pack->new_owner_rdzv.key,
                           .txid = pack->new_owner_rdzv.txid,
                           .cookie = pack->new_owner_rdzv.cookie},
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
      arts_handler_db_grant_invalidate(db, &args);
    }
    arts_shared_release(&db_h);
    break;
  }
#endif /* model dispatch for MSG_DB_GRANT_INVALIDATE */
#if defined(ARTS_PROTOCOL_EXCL) || defined(ARTS_PROTOCOL_INV)
  case MSG_DB_SNAPSHOT_REQUEST:
  case MSG_DB_SNAPSHOT_RESPONSE: {
    ARTS_ERROR("EXCL/INV build received snapshot message type %d from rank %u — "
               "this protocol has no RO snapshot round; binary mode mismatch?",
               packet->message_type, packet->rank);
    break;
  }
#else  /* VAL/WRF_VAL: snapshot handlers */
  case MSG_DB_SNAPSHOT_REQUEST: {
    ARTS_DEBUG("Coh SNAPSHOT_REQUEST Received");
    struct arts_msg_snapshot_request_packet_s *pack =
        (struct arts_msg_snapshot_request_packet_s *)(packet);
    struct arts_ooo_args_db_snapshot_request_s args = {
        .requester = pack->header.rank,
        .db_guid = pack->db_guid,
        .edt_guid = pack->edt_guid,
        .slot = pack->slot,
        .rdzv = {.addr = pack->rdzv.addr,
                 .key = pack->rdzv.key,
                 .txid = pack->rdzv.txid,
                 .cookie = pack->rdzv.cookie},
    };
    arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_SNAPSHOT_REQUEST,
                                    &args, sizeof(args));
    break;
  }
  case MSG_DB_SNAPSHOT_RESPONSE: {
    ARTS_DEBUG("Coh SNAPSHOT_RESPONSE Received");
    struct arts_msg_snapshot_response_packet_s *pack =
        (struct arts_msg_snapshot_response_packet_s *)(packet);
    /* No inline payload rides the wire anymore (the snapshot payload travels
     * one-sided and pairs by rdzv_txid inside the handler); a data-bearing
     * response is metadata-only.  Cat-C lookup-acquire-or-drop: HIT runs the
     * pure body against the ref-pinned db_s; MISS (DB destroyed / slot
     * NULL-stored) silently drops — the parked EDT this 1:1 response would
     * resume was torn down. */
    struct arts_db_snapshot_response_args_s args = {
        .edt_guid = pack->edt_guid,
        .slot = pack->slot,
        .data_present = pack->data_present,
        .version = pack->version,
        .data = NULL,
        .data_size = 0,
        .db_size = pack->db_size,
        .rdzv_txid = pack->rdzv_txid,
        .rdzv_cookie = pack->rdzv_cookie,
    };
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_snapshot_response(db, &args);
    } else {
      /* MISS with a one-sided payload in flight: still consume the txid so
       * the pairing table stays leak-free (the landing frees on arrival). */
      arts_db_rdzv_discard_landing(pack->rdzv_txid, pack->rdzv_cookie);
    }
    arts_shared_release(&h);
    break;
  }
#endif /* ARTS_PROTOCOL_EXCL */
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
  /* GRANT_RESPONSE: the single ownership-transfer wire message.  One
   * converged layout carries the map (WT: map_entry_count=0) plus buffer;
   * WRF_VAL has no ownership transfer and fatals to catch a binary mode
   * mismatch. */
#if defined(ARTS_PROTOCOL_WRF_VAL) || defined(ARTS_PROTOCOL_EXCL)
  case MSG_DB_GRANT_RESPONSE:
  case MSG_DB_GRANT_CTS: {
    ARTS_ERROR("WRF_VAL/EXCL build received ownership-transfer message type %d "
               "from rank %u — this protocol has no migrating grant; binary "
               "mode mismatch?",
               packet->message_type, packet->rank);
    break;
  }
#else  /* VAL WT and WB: one converged layout */
  case MSG_DB_GRANT_RESPONSE: {
    ARTS_DEBUG("Coh GRANT_RESPONSE Received");
    /* The (small) serialized map immediately follows the header; the buffer
     * payload travels one-sided and pairs by rdzv_txid inside the handler.
     * Both write policies share the WB-style layout (WT: map_entry_count=0). */
    arts_handler_db_grant_response((void *)packet, (size_t)packet->size);
    break;
  }
  case MSG_DB_GRANT_CTS: {
    ARTS_DEBUG("Coh GRANT_CTS Received");
    struct arts_msg_grant_cts_packet_s *pack =
        (struct arts_msg_grant_cts_packet_s *)(packet);
    /* Cat-C lookup-acquire-or-drop: HIT learns db_size + re-issues the
     * in-flight request with a landing; MISS (DB destroyed) silently drops
     * (the requester's parked waiters are woken by the destroy fan-out). */
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_grant_cts(db, pack);
    }
    arts_shared_release(&h);
    break;
  }
#endif /* model dispatch for MSG_DB_GRANT_RESPONSE */
  /* PUBLISH + PUBLISH_ACK: used by the WT write policy and WRF_VAL
   * (sync release publish).  Fatal in the WB write policy — WB uses
   * async transfer, not synchronous publish. */
/* The CTS leg exists only when a publish carries payload — i.e. under WT
 * (and EXCL's purge policy).  INV's WB publish is control-only, so it never
 * announces. */
#if (defined(ARTS_PROTOCOL_EXCL) && defined(ARTS_RELEASE_RETAIN)) ||          \
    (!defined(ARTS_PROTOCOL_EXCL) && defined(ARTS_WRITE_POLICY_WB))
  case MSG_DB_PUBLISH_CTS: {
    ARTS_ERROR("WB-write-policy build received PUBLISH_CTS from rank %u — no "
               "synchronous publish exists; binary mode mismatch?",
               packet->rank);
    break;
  }
#else
  /* PUBLISH_CTS is valid under EVERY non-WB write policy: the ownership
   * protocols' and the lossy multi-writer protocol's dirty-publish
   * announce leg, and the exclusive-lock protocol's landing-less RW release
   * (a creator-seeded hold that never received a grant). */
  case MSG_DB_PUBLISH_CTS: {
    ARTS_DEBUG("Coh PUBLISH_CTS Received");
    struct arts_msg_publish_cts_packet_s *pack =
        (struct arts_msg_publish_cts_packet_s *)(packet);
    /* Cat-C SPECIAL — pointer-identity wake of the blocked releaser's stack
     * rendezvous (sem first member).  Cache-independent: write the landing
     * fields, THEN post (sem_post is the release/acquire edge), even if the
     * DB is being torn down — a stranded releaser must never hang. */
    struct arts_db_pub_rendezvous_s *wr =
        (struct arts_db_pub_rendezvous_s *)(uintptr_t)pack->cv;
    if (wr != NULL) {
      wr->landing.addr = pack->landing.addr;
      wr->landing.key = pack->landing.key;
      wr->landing.txid = pack->landing.txid;
      wr->landing.cookie = pack->landing.cookie;
      sem_post(&wr->sem);
    }
    break;
  }
#endif /* PUBLISH_CTS placement dispatch */
/* Every arm that publishes at a release: WT write policies (payload write-through)
 * and INV under either write policy (its release always asks the home for an
 * invalidation round, carrying payload only under WT). */
#if defined(ARTS_PROTOCOL_EXCL) ||                                           \
    (defined(ARTS_WRITE_POLICY_WB) && !defined(ARTS_PROTOCOL_INV))
  case MSG_DB_PUBLISH:
  case MSG_DB_PUBLISH_ACK:
  case MSG_DB_CREATE_RETURN: {
    ARTS_ERROR("this build received publish message type %d from rank %u — "
               "its releases publish nothing; binary mode mismatch?",
               packet->message_type, packet->rank);
    break;
  }
#else  /* full handlers */
  case MSG_DB_PUBLISH: {
    ARTS_DEBUG("Coh PUBLISH Received");
    struct arts_msg_publish_packet_s *pack =
        (struct arts_msg_publish_packet_s *)(packet);
    /* Control-only in every phase (announce / commit / data-less) — the dirty
     * payload travels one-sided and pairs by rdzv_txid inside the handler. */
    struct arts_ooo_args_db_publish_s args = {
        .releaser = pack->header.rank,
        .db_guid = pack->db_guid,
        .version = pack->version,
        .cv = pack->cv,
        .data_size = pack->data_size,
        .rdzv_txid = pack->rdzv_txid,
        .rdzv_cookie = pack->rdzv_cookie,
        .data_inline = 0,
    };
    arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_PUBLISH, &args,
                                    sizeof(args));
    break;
  }
  case MSG_DB_PUBLISH_ACK: {
    ARTS_DEBUG("Coh PUBLISH_ACK Received");
    struct arts_msg_publish_ack_packet_s *pack =
        (struct arts_msg_publish_ack_packet_s *)(packet);
    /* Cat-C SPECIAL — call the body on BOTH HIT and MISS.  A nonzero cv is a
     * cache-independent pointer-identity sem-post (the releaser's heap
     * rendezvous); a torn-down cache must NOT drop that ACK or the blocked
     * releaser hangs.  cv == 0 is flight completion, which needs the live
     * releaser-side cache — the body tolerates a MISS (destroy while a
     * publish is outstanding is an ordering violation by the program; the
     * escaped waiters unblock through the shutdown path). */
    struct arts_db_publish_ack_args_s args = {.cv = pack->cv,
                                              .version = pack->version,
                                              .credit_addr = pack->credit_addr,
                                              .credit_rkey = pack->credit_rkey,
                                              .credit_txid = pack->credit_txid};
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    arts_handler_db_publish_ack(db, &args);
    arts_shared_release(&h);
    break;
  }
  case MSG_DB_CREATE_RETURN: {
    ARTS_DEBUG("Coh CREATE_RETURN Received");
    struct arts_msg_db_create_return_packet_s *pack =
        (struct arts_msg_db_create_return_packet_s *)(packet);
    /* Cat-C lookup-or-drop: record the creator's first publish credit.  The
     * creator always installed its descriptor before sending the create this
     * answers, so a MISS means the DB is already destroyed — the credit dies
     * with it (a credit is a hint, never a precondition). */
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL && pack->credit_txid != 0) {
      struct arts_db_cache_s *cache = &db->cache;
      __atomic_store_n(&cache->home_pub_addr, pack->credit_addr,
                       __ATOMIC_RELAXED);
      __atomic_store_n(&cache->home_pub_rkey, pack->credit_rkey,
                       __ATOMIC_RELAXED);
      __atomic_store_n(&cache->home_pub_txid, pack->credit_txid,
                       __ATOMIC_RELEASE);
    }
    arts_shared_release(&h);
    break;
  }
#endif /* ARTS_WRITE_POLICY_WB */
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
  /* The versioned-snapshot redirect: VAL's read path under WB, where the home
 * holds no bytes and forwards the read to the owner.  INV has its own reader
 * plane (INV_REDIRECT) and EXCL its own FORWARD, so both are excluded. */
#if defined(ARTS_WRITE_POLICY_WB) && !defined(ARTS_PROTOCOL_EXCL) &&         \
    !defined(ARTS_PROTOCOL_INV)
  case MSG_DB_SNAPSHOT_REDIRECT: {
    ARTS_DEBUG("Coh SNAPSHOT_REDIRECT Received");
    struct arts_msg_snapshot_redirect_packet_s *pack =
        (struct arts_msg_snapshot_redirect_packet_s *)(packet);
    /* Cat-C lookup-acquire-or-{DESTROY_NOTIFY}: HIT serves SNAPSHOT_RESPONSE from
     * the ref-pinned owner-side db_s; MISS (DB destroyed / not yet installed on
     * this rank) sends DESTROY_NOTIFY to the requester so its parked RO waiter
     * wakes and observes DB_DESTROYED rather than hanging. */
    struct arts_db_snapshot_redirect_args_s args = {
        .db_guid = pack->db_guid,
        .edt_guid = pack->edt_guid,
        .requester_rank = pack->requester_rank,
        .slot = pack->slot,
        .rdzv = {.addr = pack->rdzv.addr,
                 .key = pack->rdzv.key,
                 .txid = pack->rdzv.txid,
                 .cookie = pack->rdzv.cookie},
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
#else
  case MSG_DB_SNAPSHOT_REDIRECT: {
    ARTS_ERROR("this build received a snapshot redirect from rank %u — it has "
               "no versioned-snapshot read path; binary mode mismatch?",
               packet->rank);
    break;
  }
#endif /* VAL + WB snapshot redirect */

/* CONFIRM_ACK is the WB write policy's confirm gate: a new owner may not run
 * until the home has published the directory flip.  Every grant-bearing arm
 * needs it under WB — the gate belongs to the write policy, not the protocol. */
#if defined(ARTS_WRITE_POLICY_WB) && !defined(ARTS_PROTOCOL_EXCL) &&         \
    !defined(ARTS_PROTOCOL_WRF_VAL)
  case MSG_DB_GRANT_CONFIRM_ACK: {
    ARTS_DEBUG("Coh GRANT_CONFIRM_ACK Received");
    struct arts_msg_grant_confirm_ack_packet_s *pack =
        (struct arts_msg_grant_confirm_ack_packet_s *)(packet);
    /* Cat-C lookup-acquire-or-drop: HIT runs the confirm_ack body on the
     * ref-pinned db_s.  MISS (DB destroyed): the gated waiters are woken by the
     * destroy fan-out, so a miss silently drops. */
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_grant_confirm_ack(db, pack);
    }
    arts_shared_release(&h);
    break;
  }
#else
  case MSG_DB_GRANT_CONFIRM_ACK: {
    ARTS_ERROR("this build received GRANT_CONFIRM_ACK from rank %u — it "
               "has no WB-write-policy confirm gate; binary mode mismatch?",
               packet->rank);
    break;
  }
#endif /* WB-write-policy confirm gate */
  /* GRANT_CONFIRM: both write policies (new owner C → home A flips rw_holder +
   * advances the round).  WB additionally replies with CONFIRM_ACK; WT's
   * home handler does not (the new owner already drained at
   * GRANT_RESPONSE). WRF_VAL has no ownership transfer and fatals. */
#if defined(ARTS_PROTOCOL_WRF_VAL) || defined(ARTS_PROTOCOL_EXCL)
  case MSG_DB_GRANT_CONFIRM: {
    ARTS_ERROR("WRF_VAL/EXCL build received GRANT_CONFIRM from rank %u — "
               "this protocol has no migrating grant; binary mode mismatch?",
               packet->rank);
    break;
  }
#else /* VAL */
  case MSG_DB_GRANT_CONFIRM: {
    ARTS_DEBUG("Coh GRANT_CONFIRM Received");
    struct arts_msg_grant_confirm_packet_s *pack =
        (struct arts_msg_grant_confirm_packet_s *)(packet);
    /* Cat-C lookup-acquire-or-drop: HIT advances the transfer round on the
     * ref-pinned home db_s; MISS (DB destroyed) silently drops.  WT's
     * handler reads pending_install_owner and ignores args, so pass NULL. */
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
#ifdef ARTS_WRITE_POLICY_WB
      struct arts_db_grant_response_ack_args_s args = {
          .db_guid = pack->db_guid,
          .version = pack->version,
      };
      arts_handler_db_grant_confirm(db, &args);
#else
      arts_handler_db_grant_confirm(db, NULL);
#endif
    }
    arts_shared_release(&h);
    break;
  }
#endif /* GRANT_CONFIRM model dispatch */
  /* EXCL has its own REQUEST wire (both release policies) and policy-specific grant/
   * release messages.  The legacy coherence cases are excluded from EXCL builds
   * (each is already guarded above). */
#ifdef ARTS_PROTOCOL_EXCL
  /* MSG_DB_EXCL_REQUEST is shared: both PURGE and RETAIN home-dispatch vian OoO
   * (a REQUEST can arrive before the home db_s is installed on a remote-create
   * stub-install path). */
  case MSG_DB_EXCL_REQUEST: {
    ARTS_DEBUG("Coh EXCL_REQUEST Received");
    struct arts_msg_excl_request_packet_s *pack =
        (struct arts_msg_excl_request_packet_s *)(packet);
    struct arts_ooo_args_db_excl_request_s args = {
        .requester = pack->header.rank,
        .db_guid = pack->db_guid,
        .mode = pack->mode,
        .rdzv = {.addr = pack->rdzv.addr,
                 .key = pack->rdzv.key,
                 .txid = pack->rdzv.txid,
                 .cookie = pack->rdzv.cookie},
    };
    arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_EXCL_REQUEST, &args,
                                    sizeof(args));
    break;
  }
  case MSG_DB_EXCL_CTS: {
    ARTS_DEBUG("Coh EXCL_CTS Received");
    struct arts_msg_excl_cts_packet_s *pack =
        (struct arts_msg_excl_cts_packet_s *)(packet);
    /* Cat-C lookup-acquire-or-drop: HIT learns db_size + re-issues the
     * request with a landing; MISS (DB destroyed) silently drops. */
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_excl_cts(db, pack);
    }
    arts_shared_release(&h);
    break;
  }
#ifdef ARTS_RELEASE_PURGE
  /* PURGE-only: synchronous grant/release/release-ack round-trip. */
  case MSG_DB_EXCL_GRANT: {
    ARTS_DEBUG("Coh EXCL_GRANT Received");
    /* Cat-C: the grant receiver always sent its own REQUEST first, so its cache
     * exists.  Payload (data) follows the header in the contiguous buffer; the
     * handler parses it from the full packet. */
    arts_handler_db_excl_grant((void *)packet, (size_t)packet->size);
    break;
  }
  case MSG_DB_EXCL_RELEASE: {
    ARTS_DEBUG("Coh EXCL_RELEASE Received");
    struct arts_msg_excl_release_packet_s *pack =
        (struct arts_msg_excl_release_packet_s *)(packet);
    /* Control-only: a dirty RW release PUT its bytes into the grant's home
     * landing and pairs by rdzv_txid inside the handler. */
    struct arts_ooo_args_db_excl_release_s args = {
        .releaser = pack->header.rank,
        .db_guid = pack->db_guid,
        .mode = pack->mode,
        .data_size = pack->data_size,
        .cv = pack->cv,
        .version = pack->version,
        .rdzv_txid = pack->rdzv_txid,
        .rdzv_cookie = pack->rdzv_cookie,
        .data_inline = 0,
    };
    /* A home slot can be absent for two distinct reasons, and an EXCL_RELEASE
     * must treat them oppositely:
     *   - post-destroy (gen > 0): a concurrent (legal) destroy detached the slot
     *     while this holder still owed its release.  The DB is gone, so there is
     *     no publish target and no next grantee — deferring on the OoO list
     *     would wait for an install that never comes and strand the remote
     *     releaser in await_publish_ack.  ACK it directly (a torn-down home
     *     must never drop the ACK) and discard any paired one-sided landing.
     *   - pre-create (gen == 0): a creator-remote seeded RW hold releases via
     *     the announce leg, ordered only after its own DB_CREATE_COHERENT; with
     *     >= 2 progress threads the create/release can dispatch out of per-peer
     *     order, so the release may reach home before the home db_s is
     *     installed.  This one MUST keep deferring so the install-time OoO drain
     *     replays it.
     * When the slot is present, run the release inline on the pinned db_s. */
    arts_shared_ptr_t db_h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
    if (db != NULL) {
      arts_handler_db_excl_release(db, &args);
      arts_shared_release(&db_h);
    } else {
      arts_shared_release(&db_h);
      if (arts_route_table_was_destroyed(pack->db_guid)) {
        if (pack->rdzv_txid != 0) {
          arts_db_rdzv_discard_landing(pack->rdzv_txid, pack->rdzv_cookie);
        }
        if ((arts_db_access_mode_t)pack->mode == DB_MODE_RW && pack->cv != 0) {
          arts_send_db_excl_release_ack(pack->header.rank, pack->db_guid,
                                        pack->cv);
        }
      } else {
        arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_EXCL_RELEASE,
                                        &args, sizeof(args));
      }
    }
    break;
  }
  case MSG_DB_EXCL_RELEASE_ACK: {
    ARTS_DEBUG("Coh EXCL_RELEASE_ACK Received");
    struct arts_msg_excl_release_ack_packet_s *pack =
        (struct arts_msg_excl_release_ack_packet_s *)(packet);
    /* Cat-C SPECIAL — pointer-identity sem_post on cv directly.  The wake
     * is cache-independent: a torn-down home cache must NOT drop the ACK or
     * the blocked releaser hangs (await_publish_ack would spin forever).
     * arts_handler_db_publish_ack is not linked in the exclusion build (it
     * is compiled only in the publishing arms), so inline the sem_post
     * here. */
    if (pack->cv != 0) {
      sem_post((sem_t *)(uintptr_t)pack->cv);
    }
    break;
  }
#endif /* ARTS_RELEASE_PURGE */
#ifdef ARTS_RELEASE_RETAIN
  /* RETAIN-only: async migration/serve protocol.
   * All four are Cat-C (direct): their target is always installed by the time
   * the message arrives — FORWARD/DELIVER targets a cache built at acquire;
   * CONFIRM/RORET reach home only after home sent FORWARD, so home db_s exists.
   */
  case MSG_DB_EXCL_FORWARD: {
    ARTS_DEBUG("Coh EXCL_FORWARD Received");
    arts_handler_db_excl_forward((void *)packet);
    break;
  }
  case MSG_DB_EXCL_DELIVER: {
    ARTS_DEBUG("Coh EXCL_DELIVER Received");
    arts_handler_db_excl_deliver((void *)packet, (size_t)packet->size);
    break;
  }
  case MSG_DB_EXCL_CONFIRM: {
    ARTS_DEBUG("Coh EXCL_CONFIRM Received");
    arts_handler_db_excl_confirm((void *)packet);
    break;
  }
  case MSG_DB_EXCL_RORET: {
    ARTS_DEBUG("Coh EXCL_RORET Received");
    arts_handler_db_excl_roret((void *)packet);
    break;
  }
#endif /* ARTS_RELEASE_RETAIN */
#endif /* ARTS_PROTOCOL_EXCL */
#ifdef ARTS_PROTOCOL_INV
  /* INV wire arm.  REQUEST is Cat-B (a message can reach home before the home
   * db_s installs on a remote-create stub-install path); the rest are Cat-C
   * with per-message MISS actions that must never strand a round, a blocked
   * releaser, or a parked requester. */
  case MSG_DB_INV_REQUEST: {
    ARTS_DEBUG("Coh INV_REQUEST Received");
    struct arts_msg_inv_request_packet_s *pack =
        (struct arts_msg_inv_request_packet_s *)(packet);
    struct arts_ooo_args_db_inv_request_s args = {
#ifdef ARTS_WRITE_POLICY_WB
        /* The subject, not the sender: a holder that cannot serve re-sends a
         * read request on the reader's behalf. */
        .requester = pack->requester,
#else
        .requester = pack->header.rank,
#endif
        .db_guid = pack->db_guid,
        .mode = (arts_db_access_mode_t)pack->mode,
        .rdzv = {.addr = pack->rdzv.addr,
                 .key = pack->rdzv.key,
                 .txid = pack->rdzv.txid,
                 .cookie = pack->rdzv.cookie},
    };
    arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_INV_REQUEST, &args,
                                    sizeof(args));
    break;
  }
  case MSG_DB_INV_CTS: {
    ARTS_DEBUG("Coh INV_CTS Received");
    struct arts_msg_inv_cts_packet_s *pack =
        (struct arts_msg_inv_cts_packet_s *)(packet);
    /* Cat-C lookup-acquire-or-drop: HIT learns db_size + re-issues the
     * request with a landing; MISS (DB destroyed) silently drops. */
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_inv_cts(db, pack);
    }
    arts_shared_release(&h);
    break;
  }
  case MSG_DB_INV_DELIVER: {
    ARTS_DEBUG("Coh INV_DELIVER Received");
    /* Cat-C: the receiver sent its own REQUEST first, so its cache exists;
     * a destroyed cache discards the landing inside the handler. */
    arts_handler_db_inv_deliver((void *)packet, (size_t)packet->size);
    break;
  }
  case MSG_DB_INV_INVALIDATE: {
    ARTS_DEBUG("Coh INV_INVALIDATE Received");
    struct arts_msg_inv_invalidate_packet_s *pack =
        (struct arts_msg_inv_invalidate_packet_s *)(packet);
    /* Cat-C; MISS (cache already destroyed) MUST still ack — the round's
     * close is gated on every roster member answering. */
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_inv_invalidate(db);
      arts_shared_release(&h);
    } else {
      arts_shared_release(&h);
      arts_send_db_inv_invalidate_ack(pack->header.rank, pack->db_guid);
    }
    break;
  }
  case MSG_DB_INV_INVALIDATE_ACK: {
    ARTS_DEBUG("Coh INV_INVALIDATE_ACK Received");
    struct arts_msg_inv_invalidate_ack_packet_s *pack =
        (struct arts_msg_inv_invalidate_ack_packet_s *)(packet);
    /* Cat-C at home; a MISS means the DB was destroyed with a round open —
     * outside the destroy contract — drop. */
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_inv_invalidate_ack(db, pack->header.rank);
    }
    arts_shared_release(&h);
    break;
  }
#ifdef ARTS_WRITE_POLICY_WB
  /* The home holds no bytes, so a read request it cannot answer is forwarded
   * to the rank the directory names.  Cat-B: the forward can outrun the
   * holder's own DB_CREATE on a remote-create path. */
  case MSG_DB_INV_REDIRECT: {
    ARTS_DEBUG("Coh INV_REDIRECT Received");
    struct arts_msg_inv_redirect_packet_s *pack =
        (struct arts_msg_inv_redirect_packet_s *)(packet);
    struct arts_ooo_args_db_inv_redirect_s args = {
        .requester = pack->requester,
        .db_guid = pack->db_guid,
        .rdzv = {.addr = pack->rdzv.addr,
                 .key = pack->rdzv.key,
                 .txid = pack->rdzv.txid,
                 .cookie = pack->rdzv.cookie},
    };
    arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_INV_REDIRECT, &args,
                                    sizeof(args));
    break;
  }
#else
  case MSG_DB_INV_REDIRECT: {
    ARTS_ERROR("write-through build received a redirect from rank %u — the "
               "home serves reads itself under this policy; binary mode "
               "mismatch",
               ((struct arts_msg_header_s *)packet)->rank);
    break;
  }
#endif /* ARTS_WRITE_POLICY_WB */
#endif /* ARTS_PROTOCOL_INV */
  default: {
    ARTS_INFO("Unknown Packet %d %d %d", packet->message_type, packet->size,
              packet->rank);
    arts_shutdown();
    arts_runtime_stop();
  }
  }
}
