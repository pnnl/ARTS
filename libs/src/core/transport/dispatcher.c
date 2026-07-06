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

/* ===== Generic push rendezvous ==============================================
 * Sender side: a bulk payload the receiver did not ask for (EDT/event moves,
 * DB_MODE_PTR satisfies) whose wire total would breach the control ceiling.
 * RTS announces the size; the target allocates a plain registered landing and
 * replies CTS; the sender PUTs, patches {rdzv_txid, rdzv_cookie(, rdzv_size)}
 * into the retained control packet by message type, and sends it; the target
 * pairs {packet, write completion} and re-enters the normal handler with the
 * landed bytes. */

struct rdzv_push_ctx_s {
  int rank;
  char *payload;
  uint64_t size;
  void (*free_method)(void *);
  unsigned int packet_len;
  /* packet bytes follow inline */
};

/* PUT local-completion hook: run the caller's completion-gated free. */
static void rdzv_push_src_done(void *arg) {
  struct rdzv_push_ctx_s *ctx = (struct rdzv_push_ctx_s *)arg;
  if (ctx->free_method != NULL) {
    ctx->free_method(ctx->payload);
  }
  arts_free(ctx);
}

void arts_transport_send_pushed_payload(int rank,
                                        const struct arts_msg_header_s *packet,
                                        unsigned int packet_len, char *payload,
                                        uint64_t size,
                                        void (*free_method)(void *)) {
  /* Mirror the public wrappers' self/out-of-range warn-drop BEFORE retaining
   * state: a dropped RTS would otherwise strand the ctx and the payload (the
   * completion-gated free only runs on the CTS round-trip). */
  if ((unsigned int)rank == arts_global_rank_id ||
      (unsigned int)rank >= arts_global_rank_count) {
    ARTS_WARN("Cannot push to rank %u (self=%u, total=%u)", (unsigned int)rank,
              arts_global_rank_id, arts_global_rank_count);
    if (free_method != NULL) {
      free_method(payload);
    }
    return;
  }
  struct rdzv_push_ctx_s *ctx = (struct rdzv_push_ctx_s *)arts_malloc(
      sizeof(struct rdzv_push_ctx_s) + packet_len);
  ctx->rank = rank;
  ctx->payload = payload;
  ctx->size = size;
  ctx->free_method = free_method;
  ctx->packet_len = packet_len;
  memcpy(ctx + 1, packet, packet_len);
  struct arts_msg_rdzv_push_rts_packet_s rts;
  arts_fill_packet_header(&rts.header, sizeof(rts), MSG_RDZV_PUSH_RTS);
  rts.size = size;
  rts.push_cookie = (uint64_t)(uintptr_t)ctx;
  arts_transport_send_async(rank, (char *)&rts, sizeof(rts));
}

/* Target-side continuation: the pushed bytes fully landed; rebuild the
 * contiguous (packet + payload) image the normal handlers expect, dispatch
 * it, and free the landing. */
struct rdzv_push_landed_ctx_s {
  char *landing;
  uint64_t size;
  unsigned int packet_len;
  /* control-packet bytes follow inline */
};

static void rdzv_push_landed_cb(void *arg) {
  struct rdzv_push_landed_ctx_s *ctx = (struct rdzv_push_landed_ctx_s *)arg;
  struct arts_msg_header_s *hdr = (struct arts_msg_header_s *)(ctx + 1);
  uint64_t total = (uint64_t)ctx->packet_len + ctx->size;
  char *rebuilt = (char *)arts_malloc((size_t)total);
  memcpy(rebuilt, hdr, ctx->packet_len);
  memcpy(rebuilt + ctx->packet_len, ctx->landing, (size_t)ctx->size);
  struct arts_msg_header_s *rh = (struct arts_msg_header_s *)rebuilt;
  rh->size = total; /* the handlers derive the blob size from header.size */
  /* Strip the pairing marks so the re-entry takes the inline arm. */
  switch (rh->message_type) {
  case MSG_EDT_SATISFY_SLOT: {
    struct arts_msg_edt_satisfy_slot_packet_s *sp =
        (struct arts_msg_edt_satisfy_slot_packet_s *)rebuilt;
    sp->rdzv_txid = 0;
    sp->rdzv_cookie = 0;
    break;
  }
  case MSG_EDT_CREATE:
  case MSG_EVENT_CREATE: {
    struct arts_msg_memory_move_packet_s *mp =
        (struct arts_msg_memory_move_packet_s *)rebuilt;
    mp->rdzv_txid = 0;
    mp->rdzv_cookie = 0;
    mp->rdzv_size = 0;
    break;
  }
  default:
    ARTS_ERROR("push rendezvous: unsupported pushed message type %u",
               rh->message_type);
  }
  arts_transport_dispatch_body(rh);
  arts_free(rebuilt);
  arts_regpool_free(ctx->landing);
  arts_free(ctx);
}

/* Register the pairing for a pushed message's payload; the packet bytes are
 * retained in the ctx (the wire buffer is freed after dispatch). */
static void rdzv_push_expect(const struct arts_msg_header_s *packet,
                             unsigned int packet_len, uint64_t txid,
                             uint64_t cookie, uint64_t size) {
  struct rdzv_push_landed_ctx_s *ctx = (struct rdzv_push_landed_ctx_s *)
      arts_malloc(sizeof(struct rdzv_push_landed_ctx_s) + packet_len);
  ctx->landing = (char *)(uintptr_t)cookie;
  ctx->size = size;
  ctx->packet_len = packet_len;
  memcpy(ctx + 1, packet, packet_len);
  arts_net_rdzv_expect(txid, rdzv_push_landed_cb, ctx);
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
    if (pack->rdzv_txid != 0) {
      /* Oversized DB_MODE_PTR payload traveling by push rendezvous: pair this
       * packet with the write completion, then re-enter with the landed bytes
       * rebuilt inline. */
      rdzv_push_expect(packet, (unsigned int)sizeof(*pack), pack->rdzv_txid,
                       pack->rdzv_cookie, pack->size);
      break;
    }
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
    struct arts_msg_memory_move_packet_s *pack =
        (struct arts_msg_memory_move_packet_s *)(packet);
    if (pack->rdzv_txid != 0) {
      rdzv_push_expect(packet, (unsigned int)sizeof(*pack), pack->rdzv_txid,
                       pack->rdzv_cookie, pack->rdzv_size);
      break;
    }
    arts_handler_edt_create(packet);
    break;
  }
  case MSG_EVENT_CREATE: {
    ARTS_DEBUG("Event Move Received");
    struct arts_msg_memory_move_packet_s *pack =
        (struct arts_msg_memory_move_packet_s *)(packet);
    if (pack->rdzv_txid != 0) {
      rdzv_push_expect(packet, (unsigned int)sizeof(*pack), pack->rdzv_txid,
                       pack->rdzv_cookie, pack->rdzv_size);
      break;
    }
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
        .rdzv = {.addr = pack->rdzv.addr,
                 .key = pack->rdzv.key,
                 .txid = pack->rdzv.txid,
                 .cookie = pack->rdzv.cookie},
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
    ARTS_DEBUG("Coh DATA_RESPONSE Received");
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
  case MSG_DB_OWNERSHIP_RESPONSE:
  case MSG_DB_OWNERSHIP_CTS: {
    ARTS_ERROR("MRMW/LOCK build received ownership-transfer message type %d "
               "from rank %u — protocol has no ownership transfer; binary "
               "mode mismatch?",
               packet->message_type, packet->rank);
    break;
  }
#else  /* MRNEW/MRSW eager and lazy: one converged layout */
  case MSG_DB_OWNERSHIP_RESPONSE: {
    ARTS_DEBUG("Coh OWNERSHIP_RESPONSE Received");
    /* The (small) serialized map immediately follows the header; the buffer
     * payload travels one-sided and pairs by rdzv_txid inside the handler.
     * Both timings share the lazy-style layout (EAGER: map_entry_count=0). */
    arts_handler_db_ownership_response((void *)packet, (size_t)packet->size);
    break;
  }
  case MSG_DB_OWNERSHIP_CTS: {
    ARTS_DEBUG("Coh OWNERSHIP_CTS Received");
    struct arts_msg_ownership_cts_packet_s *pack =
        (struct arts_msg_ownership_cts_packet_s *)(packet);
    /* Cat-C lookup-acquire-or-drop: HIT learns db_size + re-issues the
     * in-flight request with a landing; MISS (DB destroyed) silently drops
     * (the requester's parked waiters are woken by the destroy fan-out). */
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_ownership_cts(db, pack);
    }
    arts_shared_release(&h);
    break;
  }
#endif /* model dispatch for MSG_DB_OWNERSHIP_RESPONSE */
  /* WRITEBACK + WRITEBACK_ACK: used by the eager protocol and MRMW
   * (sync release writeback).  Fatal in the lazy protocol — lazy uses
   * async transfer, not synchronous writeback. */
#if defined(ARTS_TIMING_LAZY)
  case MSG_DB_WRITEBACK_CTS: {
    ARTS_ERROR("lazy build received WRITEBACK_CTS from rank %u — no "
               "synchronous writeback exists; binary mode mismatch?",
               packet->rank);
    break;
  }
#else
  /* WRITEBACK_CTS is valid under EVERY non-lazy timing: the ownership
   * protocols' and the lossy multi-writer protocol's dirty-writeback
   * announce leg, and the exclusive-lock protocol's landing-less RW release
   * (a creator-seeded hold that never received a grant). */
  case MSG_DB_WRITEBACK_CTS: {
    ARTS_DEBUG("Coh WRITEBACK_CTS Received");
    struct arts_msg_writeback_cts_packet_s *pack =
        (struct arts_msg_writeback_cts_packet_s *)(packet);
    /* Cat-C SPECIAL — pointer-identity wake of the blocked releaser's stack
     * rendezvous (sem first member).  Cache-independent: write the landing
     * fields, THEN post (sem_post is the release/acquire edge), even if the
     * DB is being torn down — a stranded releaser must never hang. */
    struct arts_db_wb_rendezvous_s *wr =
        (struct arts_db_wb_rendezvous_s *)(uintptr_t)pack->cv;
    if (wr != NULL) {
      wr->landing.addr = pack->landing.addr;
      wr->landing.key = pack->landing.key;
      wr->landing.txid = pack->landing.txid;
      wr->landing.cookie = pack->landing.cookie;
      sem_post(&wr->sem);
    }
    break;
  }
#endif /* WRITEBACK_CTS timing dispatch */
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
    /* Control-only in every phase (announce / commit / data-less) — the dirty
     * payload travels one-sided and pairs by rdzv_txid inside the handler. */
    struct arts_ooo_args_db_writeback_s args = {
        .releaser = pack->header.rank,
        .db_guid = pack->db_guid,
        .version = pack->version,
        .cv = pack->cv,
        .data_size = pack->data_size,
        .rdzv_txid = pack->rdzv_txid,
        .rdzv_cookie = pack->rdzv_cookie,
        .data_inline = 0,
    };
    arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_WRITEBACK, &args,
                                    sizeof(args));
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
        .rdzv = {.addr = pack->rdzv.addr,
                 .key = pack->rdzv.key,
                 .txid = pack->rdzv.txid,
                 .cookie = pack->rdzv.cookie},
    };
    arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_LOCK_REQUEST, &args,
                                    sizeof(args));
    break;
  }
  case MSG_DB_LOCK_CTS: {
    ARTS_DEBUG("Coh LOCK_CTS Received");
    struct arts_msg_lock_cts_packet_s *pack =
        (struct arts_msg_lock_cts_packet_s *)(packet);
    /* Cat-C lookup-acquire-or-drop: HIT learns db_size + re-issues the
     * request with a landing; MISS (DB destroyed) silently drops. */
    arts_shared_ptr_t h = arts_route_table_lookup_db(pack->db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_lock_cts(db, pack);
    }
    arts_shared_release(&h);
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
    /* Control-only: a dirty RW release PUT its bytes into the grant's home
     * landing and pairs by rdzv_txid inside the handler. */
    struct arts_ooo_args_db_lock_release_s args = {
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
    /* A home slot can be absent for two distinct reasons, and a LOCK_RELEASE
     * must treat them oppositely:
     *   - post-destroy (gen > 0): a concurrent (legal) destroy detached the slot
     *     while this holder still owed its release.  The DB is gone, so there is
     *     no writeback target and no next grantee — deferring on the OoO list
     *     would wait for an install that never comes and strand the remote
     *     releaser in await_writeback_ack.  ACK it directly (a torn-down home
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
      arts_handler_db_lock_release(db, &args);
      arts_shared_release(&db_h);
    } else {
      arts_shared_release(&db_h);
      if (arts_route_table_was_destroyed(pack->db_guid)) {
        if (pack->rdzv_txid != 0) {
          arts_db_rdzv_discard_landing(pack->rdzv_txid, pack->rdzv_cookie);
        }
        if ((arts_db_access_mode_t)pack->mode == DB_MODE_RW && pack->cv != 0) {
          arts_send_db_lock_release_ack(pack->header.rank, pack->db_guid,
                                        pack->cv);
        }
      } else {
        arts_ooo_dispatch_or_defer_guid(pack->db_guid, OOO_DB_LOCK_RELEASE,
                                        &args, sizeof(args));
      }
    }
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
  case MSG_RDZV_PUSH_RTS: {
    ARTS_DEBUG("RDZV_PUSH_RTS Received");
    struct arts_msg_rdzv_push_rts_packet_s *pack =
        (struct arts_msg_rdzv_push_rts_packet_s *)(packet);
    /* Allocate a plain registered landing for the incoming push and hand it
     * back.  Fail-loud if it cannot be advertised (one-sided delivery needs
     * the registered pool). */
    char *landing =
        (char *)arts_regpool_alloc_aligned((size_t)pack->size, 64);
    struct arts_msg_rdzv_push_cts_packet_s cts;
    arts_fill_packet_header(&cts.header, sizeof(cts), MSG_RDZV_PUSH_CTS);
    cts.push_cookie = pack->push_cookie;
    if (!arts_net_rdzv_local(landing, pack->size, &cts.landing.addr,
                             &cts.landing.key)) {
      ARTS_ERROR("push rendezvous: landing is not fabric-registered — "
                 "one-sided payloads require the registered pool");
    }
    cts.landing.txid = arts_net_rdzv_txid_next();
    cts.landing.cookie = (uint64_t)(uintptr_t)landing;
    arts_transport_send_async((int)packet->rank, (char *)&cts, sizeof(cts));
    break;
  }
  case MSG_RDZV_PUSH_CTS: {
    ARTS_DEBUG("RDZV_PUSH_CTS Received");
    struct arts_msg_rdzv_push_cts_packet_s *pack =
        (struct arts_msg_rdzv_push_cts_packet_s *)(packet);
    struct rdzv_push_ctx_s *ctx =
        (struct rdzv_push_ctx_s *)(uintptr_t)pack->push_cookie;
    /* PUT the retained payload into the granted landing, then send the
     * retained control packet with the pairing marks patched in by message
     * type.  The payload's completion-gated free rides the PUT's local
     * completion. */
    struct arts_msg_header_s *hdr = (struct arts_msg_header_s *)(ctx + 1);
    switch (hdr->message_type) {
    case MSG_EDT_SATISFY_SLOT: {
      struct arts_msg_edt_satisfy_slot_packet_s *sp =
          (struct arts_msg_edt_satisfy_slot_packet_s *)hdr;
      sp->rdzv_txid = pack->landing.txid;
      sp->rdzv_cookie = pack->landing.cookie;
      break;
    }
    case MSG_EDT_CREATE:
    case MSG_EVENT_CREATE: {
      struct arts_msg_memory_move_packet_s *mp =
          (struct arts_msg_memory_move_packet_s *)hdr;
      mp->rdzv_txid = pack->landing.txid;
      mp->rdzv_cookie = pack->landing.cookie;
      mp->rdzv_size = ctx->size;
      break;
    }
    default:
      ARTS_ERROR("push rendezvous: unsupported pushed message type %u",
                 hdr->message_type);
    }
    /* Send the control packet BEFORE posting the PUT: send_async copies the
     * bytes synchronously, while the PUT's local completion — which frees ctx
     * (and hdr inside it) — can fire as early as the submit's own
     * backpressure reap.  Packet/completion ordering is irrelevant (the
     * target pairs by txid in either order). */
    int target_rank = ctx->rank;
    arts_transport_send_async(target_rank, (char *)hdr, ctx->packet_len);
    arts_net_put_payload(target_rank, pack->landing.addr, pack->landing.key,
                         pack->landing.txid, ctx->payload, ctx->size,
                         rdzv_push_src_done, ctx);
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
