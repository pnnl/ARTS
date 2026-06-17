/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence protocol wire-message SENDERS.
 *
 * Each arts_send_db_* helper fills a wire packet (header + body) and either
 * dispatches the matching handler inline (when the destination is the local
 * rank — arts_transport_send_async drops self-sends, and the eager protocol
 * uses uniform "send to home" semantics including home == self) or enqueues
 * the packet on the outbox for the transport layer.
 *
 * The receive-side bodies (arts_handler_db_*) and the home-side dedup /
 * transfer helpers live in coherence_handlers.c.
 *
 * Single-node note: arts_transport_send_async drops messages whose
 * destination is the local rank (self_send_check rejects).  The eager protocol
 * uses uniform "send to home" semantics including home == self, so we dispatch
 * handlers directly when rank == self instead of going over the network.
 */

#include "arts/coherence/handlers.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "arts/coherence/coherence.h" /* mark_edt_ready_by_guid (MRSW destroy-notify wake) */
#include "arts/db.h" /* struct arts_db_s (Cat-C self-send lookup-acquire) */
#include "arts/gas/route_table.h" /* arts_route_table_lookup_db (Cat-C self-send) */
#include "arts/ooo.h" /* arts_ooo_dispatch_or_defer_guid (self-send replay) */
#include "arts/system/threads.h"
#include "arts/transport/outbox.h" /* outbound send helpers */
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h" /* arts_shared_get / arts_shared_release */

/* ===== Sender helpers ============================================== */

/* arts_send_db_ownership_request / _return / _invalidate and the
 * OWNERSHIP_RESPONSE sender live in the protocol TUs (eager+lazy only): the
 * request / return / invalidate senders in coherence/ownership.c, the
 * OWNERSHIP_RESPONSE sender in coherence/eager.c (GRANT) and coherence/lazy.c
 * (TRANSFER_OWNERSHIP).  MRMW has no exclusive-ownership wire messages. */

void arts_send_db_writeback(unsigned int home_rank, arts_guid_t db_guid,
                            uint64_t version, uint64_t cv, const void *data,
                            uint64_t data_size) {
  struct arts_msg_writeback_packet_s p;
  uint64_t total = sizeof(p) + data_size;
  arts_fill_packet_header(&p.header, total, MSG_DB_WRITEBACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  p.cv = cv;
  memset(p.pad, 0, sizeof(p.pad));
#if !defined(ARTS_TIMING_LAZY)
  /* Self-send (home == self) — eager/MRMW only.  The lazy protocol has no
   * synchronous writeback at all (its OOO_DB_WRITEBACK kind does not exist), so
   * this whole sender is statically excluded under the lazy build. */
  if (home_rank == arts_global_rank_id) {
    /* Route through the OoO engine exactly as the wire RX dispatcher does —
     * HIT runs the writeback body inline, MISS defers the args (trailing data
     * preserved) and replays once the home db_s is installed + drained.
     * WRITEBACK carries an inline payload, so lay it immediately after the args
     * struct; the body reads it back from (char *)args + sizeof(struct). */
    uint32_t asz =
        (uint32_t)(sizeof(struct arts_ooo_args_db_writeback_s) + data_size);
    char *abuf = (char *)arts_malloc(asz);
    struct arts_ooo_args_db_writeback_s *args =
        (struct arts_ooo_args_db_writeback_s *)abuf;
    args->releaser = p.header.rank;
    args->db_guid = db_guid;
    args->version = version;
    args->cv = cv;
    args->data_size = data_size;
    if (data_size > 0 && data != NULL) {
      memcpy(abuf + sizeof(*args), data, data_size);
    }
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_WRITEBACK, abuf, asz);
    arts_free(abuf);
    return;
  }
#endif
  /* Sentinel DBs (db_size==0) still need WRITEBACK for ownership
   * transfer / R3 ordering, but the payload-async path errors on
   * zero-size payload — route via the no-payload async send. */
  if (data == NULL || data_size == 0) {
    arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
    return;
  }
  arts_transport_send_payload_async((int)home_rank, (char *)&p, sizeof(p),
                                    (char *)data, data_size);
}

/* WRITEBACK_ACK is the reply to a synchronous WRITEBACK round, which only the
 * eager and MRMW protocols use (the lazy protocol transfers ownership
 * owner→owner without a synchronous writeback, so it never sends or receives
 * WRITEBACK_ACK and its dispatcher fatals on the wire message). */
#if !defined(ARTS_TIMING_LAZY)
void arts_send_db_writeback_ack(unsigned int releaser_rank, arts_guid_t db_guid,
                                uint64_t cv) {
  struct arts_msg_writeback_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_WRITEBACK_ACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.cv = cv;
  if (releaser_rank == arts_global_rank_id) {
    /* Self-send: mirror the wire RX dispatcher's Cat-C lookup-acquire.  The
     * wake is a cache-independent pointer-identity sem-post on cv; the body
     * ignores item_v, so call it unconditionally (db may be NULL — a missing
     * home cache must still post the sem, else the blocked releaser hangs). */
    struct arts_db_writeback_ack_args_s args = {.cv = cv};
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    arts_handler_db_writeback_ack(db, &args);
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)releaser_rank, (char *)&p, sizeof(p));
}
#endif /* !ARTS_TIMING_LAZY */

void arts_send_db_snapshot_request(unsigned int home_rank, arts_guid_t db_guid,
                                   arts_guid_t edt_guid, uint32_t slot) {
  struct arts_msg_snapshot_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_SNAPSHOT_REQUEST);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.edt_guid = edt_guid;
  p.slot = slot;
  memset(p.pad, 0, sizeof(p.pad));
  if (home_rank == arts_global_rank_id) {
    /* Self-send: route through the OoO engine exactly as the wire RX
     * dispatcher does — HIT serves the snapshot inline, MISS defers the args
     * and replays once the home db_s is installed + drained.  (The handler is
     * now a pure (item, args) body; it no longer does its own lookup-or-defer,
     * so the inline shortcut must enter through dispatch_or_defer.) */
    struct arts_ooo_args_db_snapshot_request_s args = {
        .requester = p.header.rank,
        .db_guid = db_guid,
        .edt_guid = edt_guid,
        .slot = slot,
    };
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_SNAPSHOT_REQUEST, &args,
                                    sizeof(args));
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_snapshot_response(unsigned int requester_rank,
                                    arts_guid_t db_guid, uint64_t version,
                                    arts_guid_t edt_guid, uint32_t slot,
                                    const void *data, uint64_t data_size) {
  struct arts_msg_snapshot_response_packet_s p;
  uint64_t total = sizeof(p) + (data ? data_size : 0);
  arts_fill_packet_header(&p.header, total, MSG_DB_SNAPSHOT_RESPONSE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  p.edt_guid = edt_guid;
  p.slot = slot;
  p.data_present = data ? 1u : 0u;
  if (requester_rank == arts_global_rank_id) {
    /* Self-send: mirror the wire RX dispatcher's Cat-C lookup-acquire-or-drop.
     * HIT runs the pure body against the ref-pinned db_s; MISS (DB destroyed)
     * silently drops — the parked EDT this response resumes was torn down. */
    struct arts_db_snapshot_response_args_s args = {
        .edt_guid = edt_guid,
        .slot = slot,
        .data_present = data ? 1u : 0u,
        .version = version,
        .data = data && data_size > 0 ? data : NULL,
        .data_size = data ? data_size : 0,
    };
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_snapshot_response(db, &args);
    }
    arts_shared_release(&h);
    return;
  }
  if (data && data_size > 0) {
    arts_transport_send_payload_async((int)requester_rank, (char *)&p,
                                      sizeof(p), (char *)data, data_size);
  } else {
    arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
  }
}

void arts_send_db_create_coherent(unsigned int home_rank, arts_guid_t db_guid,
                                  uint64_t db_size, uint16_t flags,
                                  uint16_t db_type) {
  struct arts_msg_db_create_coherent_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_CREATE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.db_size = db_size;
  p.flags = flags;
  p.db_type = db_type;
  memset(p.pad, 0, sizeof(p.pad));
  if (home_rank == arts_global_rank_id) {
    arts_handler_db_create(&p);
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_destroy(unsigned int home_rank, arts_guid_t db_guid) {
  struct arts_msg_destroy_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_DESTROY);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    /* Self-send: route through the OoO engine exactly as the wire RX
     * dispatcher does — HIT runs the destroy body inline, MISS defers the args
     * and replays once the home db_s is installed + drained.  (The handler is
     * now a pure (item, args) body; it no longer does its own lookup-or-defer,
     * so the inline shortcut must enter through dispatch_or_defer.) */
    struct arts_ooo_args_db_destroy_s args = {
        .requester = p.header.rank,
        .db_guid = db_guid,
    };
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_DESTROY, &args,
                                    sizeof(args));
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_cache_destroy(unsigned int sharer_rank, arts_guid_t db_guid) {
  struct arts_msg_cache_destroy_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_CACHE_DESTROY);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (sharer_rank == arts_global_rank_id) {
    /* Self-send: mirror the wire RX dispatcher's Cat-C lookup-acquire-or-drop.
     * HIT wakes parked waiters + detaches the cb; MISS (already torn down on
     * this rank) silently drops (idempotent). */
    struct arts_db_cache_destroy_args_s args = {.db_guid = db_guid};
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_cache_destroy(db, &args);
    }
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)sharer_rank, (char *)&p, sizeof(p));
}

/* The LAZY-only senders (CONFIRM, CONFIRM_ACK, REDIRECT_RO) live in
 * coherence/lazy.c alongside their handlers; the MRNEW OWNERSHIP_RESPONSE
 * senders live in coherence/eager.c / coherence/lazy.c. */
