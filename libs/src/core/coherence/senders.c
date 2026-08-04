/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence protocol wire-message SENDERS.
 *
 * Each arts_send_db_* helper fills a wire packet (header + body) and either
 * dispatches the matching handler inline (when the destination is the local
 * rank — arts_transport_send_async drops self-sends, and the HOME placement
 * uses uniform "send to home" semantics including home == self) or hands the
 * packet to the transport layer; bulk payloads travel one-sided into
 * receiver-advertised rendezvous landings, never on the control plane.
 *
 * The receive-side bodies (arts_handler_db_*) and the home-side dedup /
 * transfer helpers live in coherence_handlers.c.
 *
 * Single-node note: arts_transport_send_async drops messages whose
 * destination is the local rank (self_send_check rejects).  The HOME protocol
 * uses uniform "send to home" semantics including home == self, so we dispatch
 * handlers directly when rank == self instead of going over the network.
 */

#include "arts/coherence/handlers.h"

#include "arts/coherence/buffer.h" /* landing alloc / ref-release cb */

#include <semaphore.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "arts/coherence/coherence.h" /* mark_edt_ready_by_guid */
#include "arts/db.h" /* struct arts_db_s (Cat-C self-send lookup-acquire) */
#include "arts/gas/route_table.h" /* arts_route_table_lookup_db (Cat-C self-send) */
#include "arts/ooo.h" /* arts_ooo_dispatch_or_defer_guid (self-send replay) */
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/transport/net.h" /* outbound send helpers */
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h" /* arts_shared_get / arts_shared_release */

/* ===== Sender helpers ============================================== */

/* arts_send_db_grant_request / _return / _invalidate and the
 * OWNERSHIP_RESPONSE sender live in the protocol TUs (HOME and OWNER only): the
 * request / return / invalidate senders in coherence/grant.c, the
 * OWNERSHIP_RESPONSE sender in coherence/home.c (GRANT) and coherence/owner.c
 * (TRANSFER_OWNERSHIP).  WRF_RCU has no exclusive-ownership wire messages. */

#if !defined(ARTS_PROTOCOL_EXCL)
void arts_send_db_publish(unsigned int home_rank, arts_guid_t db_guid,
                            uint64_t version, uint64_t cv, const void *data,
                            uint64_t data_size, uint64_t rdzv_txid,
                            uint64_t rdzv_cookie) {
  struct arts_msg_publish_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_PUBLISH);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  p.cv = cv;
  p.data_size = data_size;
  p.rdzv_txid = rdzv_txid;
  p.rdzv_cookie = rdzv_cookie;
/* The OoO kind exists only in arms that publish at a release; RCU under OWNER
 * publishes nothing, so its local-hit branch is compiled out with it. */
#if !defined(ARTS_WRITE_POLICY_WB) || defined(ARTS_PROTOCOL_INV)
  /* Local hit: the home is this rank, so there is nothing to put on the wire.
   * Route through the OoO engine exactly as the wire RX dispatcher does — a
   * HIT runs the publish body inline, a MISS defers the args (trailing payload
   * preserved) and replays once the home db_s installs and drains.  Skipping
   * the engine here would lose the before-create reorder handling that the
   * remote path gets for free. */
  if (home_rank == arts_global_rank_id) {
    /* Route through the OoO engine exactly as the wire RX dispatcher does —
     * HIT runs the publish body inline, MISS defers the args (trailing data
     * preserved) and replays once the home db_s is installed + drained.  A
     * same-rank publish carries its payload inline after the args struct
     * (data_inline=1); no rendezvous round exists for it. */
    uint64_t inline_size = (data != NULL) ? data_size : 0;
    uint32_t asz =
        (uint32_t)(sizeof(struct arts_ooo_args_db_publish_s) + inline_size);
    char *abuf = (char *)arts_malloc(asz);
    struct arts_ooo_args_db_publish_s *args =
        (struct arts_ooo_args_db_publish_s *)abuf;
    args->releaser = p.header.rank;
    args->db_guid = db_guid;
    args->version = version;
    args->cv = cv;
    args->data_size = data_size;
    args->rdzv_txid = 0;
    args->rdzv_cookie = 0;
    args->data_inline = (inline_size > 0) ? 1u : 0u;
    if (inline_size > 0) {
      memcpy(abuf + sizeof(*args), data, inline_size);
    }
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_PUBLISH, abuf, asz);
    arts_free(abuf);
    return;
  }
#endif
  /* Remote: the packet is control-only in every phase — announce
   * (data_size>0, txid 0), commit (txid set), or a data-less ordering round
   * (data_size 0).  The dirty payload itself travels one-sided
   * (arts_db_publish_sync PUTs it between announce and commit). */
  (void)data;
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}

#endif /* !ARTS_PROTOCOL_EXCL */

/* PUBLISH_CTS — home → releaser: a home landing for an announced dirty
 * publish (a fresh buffer under the ownership/multi-writer protocols; the
 * stable buffer under the exclusive-lock protocol's landing-less release).
 * Never a self-send (a same-rank publish is inline).  Compiled for every
 * protocol with a synchronous publish leg. */
void arts_send_db_publish_cts(unsigned int releaser_rank, arts_guid_t db_guid,
                                const struct arts_rdzv_landing_s *landing,
                                uint64_t cv) {
  struct arts_msg_publish_cts_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_PUBLISH_CTS);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.landing.addr = landing->addr;
  p.landing.key = landing->key;
  p.landing.txid = landing->txid;
  p.landing.cookie = landing->cookie;
  p.cv = cv;
  arts_transport_send_async((int)releaser_rank, (char *)&p, sizeof(p));
}

/* PUBLISH_ACK is the reply to a synchronous PUBLISH round, which only the
 * the HOME placement and WRF_RCU use (the OWNER placement transfers ownership
 * owner→owner without a synchronous publish, so it never sends or receives
 * PUBLISH_ACK and its dispatcher fatals on the wire message). */
#if !defined(ARTS_PROTOCOL_EXCL) &&                                          \
    (!defined(ARTS_WRITE_POLICY_WB) || defined(ARTS_PROTOCOL_INV))
void arts_send_db_publish_ack(unsigned int releaser_rank, arts_guid_t db_guid,
                                uint64_t cv) {
  struct arts_msg_publish_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_PUBLISH_ACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.cv = cv;
  if (releaser_rank == arts_global_rank_id) {
    /* Self-send: mirror the wire RX dispatcher's Cat-C lookup-acquire.  The
     * wake is a cache-independent pointer-identity sem-post on cv; the body
     * ignores item_v, so call it unconditionally (db may be NULL — a missing
     * home cache must still post the sem, else the blocked releaser hangs). */
    struct arts_db_publish_ack_args_s args = {.cv = cv};
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    arts_handler_db_publish_ack(db, &args);
    arts_shared_release(&h);
    return;
  }
  arts_transport_send_async((int)releaser_rank, (char *)&p, sizeof(p));
}
#endif /* !ARTS_WRITE_POLICY_WB && !ARTS_PROTOCOL_EXCL */

/* The versioned-snapshot read path: RCU and WRF_RCU only.  MSI's readers hold
 * durable copies and fetch with MSI_REQUEST instead. */
#if !defined(ARTS_PROTOCOL_EXCL) && !defined(ARTS_PROTOCOL_INV)
void arts_send_db_snapshot_request(struct arts_db_cache_s *cache,
                                   arts_guid_t edt_guid, uint32_t slot) {
  arts_guid_t db_guid = cache->db_guid;
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  /* Advertise a fresh snapshot landing when the size is known; a size-unknown
   * first touch sends landing-less (txid 0) and the server answers a
   * size-only CTS response, whose handler re-enters this sender.  The
   * rendezvous plane exists only when a peer could PUT (multi-rank run). */
  struct arts_rdzv_landing_s rdzv = {0, 0, 0, 0};
  if (cache->db_size > 0 && arts_global_rank_count > 1) {
    (void)arts_db_buf_landing_alloc(cache, cache->db_size, &rdzv);
  }
  struct arts_msg_snapshot_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_SNAPSHOT_REQUEST);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.edt_guid = edt_guid;
  p.slot = slot;
  memset(p.pad, 0, sizeof(p.pad));
  p.rdzv.addr = rdzv.addr;
  p.rdzv.key = rdzv.key;
  p.rdzv.txid = rdzv.txid;
  p.rdzv.cookie = rdzv.cookie;
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
        .rdzv = rdzv,
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
                                    uint32_t kind, uint64_t db_size,
                                    const struct arts_rdzv_landing_s *landing,
                                    arts_shared_ptr_t src_h) {
  struct arts_db_buffer_s *src =
      (struct arts_db_buffer_s *)arts_shared_get(src_h);
  struct arts_msg_snapshot_response_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_SNAPSHOT_RESPONSE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  p.edt_guid = edt_guid;
  p.slot = slot;
  p.data_present = kind;
  p.db_size = db_size;
  p.rdzv_txid = 0;
  p.rdzv_cookie = (landing != NULL) ? landing->cookie : 0;
  if (requester_rank == arts_global_rank_id) {
    /* Self-serve (an OWNER REDIRECT round-trip can land back on the requester
     * rank): no RDMA — the bytes ride inline through the args while src_h
     * pins the buffer; the handler recycles our own unused landing (cookie).
     * Mirror the wire RX dispatcher's Cat-C lookup-acquire-or-drop. */
    struct arts_db_snapshot_response_args_s args = {
        .edt_guid = edt_guid,
        .slot = slot,
        .data_present = kind,
        .version = version,
        .data = (kind == 1 && src != NULL) ? src->data : NULL,
        .data_size = (kind == 1 && src != NULL) ? db_size : 0,
        .db_size = db_size,
        .rdzv_txid = 0,
        .rdzv_cookie = (landing != NULL) ? landing->cookie : 0,
    };
    arts_shared_ptr_t h = arts_route_table_lookup_db(db_guid);
    struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(h);
    if (db != NULL) {
      arts_handler_db_snapshot_response(db, &args);
    }
    arts_shared_release(&h);
    if (src != NULL) {
      arts_db_buf_release(&src_h);
    }
    return;
  }
  if (kind == 1) {
    /* One-sided serve: PUT straight from the live buffer into the
     * requester's landing — zero copy at the source.  The strong buffer ref
     * transfers to the PUT's local completion, keeping the bytes valid until
     * the fabric no longer reads them. */
    if (src == NULL || landing == NULL || landing->txid == 0) {
      ARTS_ERROR("coherence: data-bearing snapshot response without a source "
                 "buffer or landing (txid=%llx)",
                 (unsigned long long)(landing != NULL ? landing->txid : 0));
    }
    p.rdzv_txid = landing->txid;
    arts_net_put_payload((int)requester_rank, landing->addr, landing->key,
                         landing->txid, src->data, db_size,
                         arts_db_buf_ref_release_cb, (void *)src_h);
  } else if (src != NULL) {
    arts_db_buf_release(&src_h);
  }
  arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
}
#endif /* !ARTS_PROTOCOL_EXCL && !ARTS_PROTOCOL_INV */

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

#ifdef ARTS_PROTOCOL_EXCL
/* arts_send_db_excl_release_ack — LOCK_RELEASE_ACK: home → RW releaser.
 *
 * Mirrors arts_send_db_publish_ack (RCU HOME): forwards cv verbatim so
 * the releaser's await_publish_ack unblocks by pointer-identity sem_post.
 *
 * Cat-C SPECIAL self-send: posts the sem even when db==NULL (home cache
 * torn down concurrently) so the blocked releaser is never stranded.
 *
 * The sem_post inline (rather than calling arts_handler_db_publish_ack)
 * avoids a cross-protocol link dependency: arts_handler_db_publish_ack is
 * defined only in RCU/WRF_RCU TUs, not in the RWLOCK build. */
void arts_send_db_excl_release_ack(unsigned int releaser_rank,
                                   arts_guid_t db_guid, uint64_t cv) {
  struct arts_msg_excl_release_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_EXCL_RELEASE_ACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.cv = cv;
  if (releaser_rank == arts_global_rank_id) {
    /* Self-send: pointer-identity sem_post directly (no lookup needed —
     * the body ignores item_v and only uses cv; unconditional post matches
     * the Cat-C SPECIAL dispatcher pattern). */
    if (cv != 0) {
      sem_post((sem_t *)(uintptr_t)cv);
    }
    return;
  }
  arts_transport_send_async((int)releaser_rank, (char *)&p, sizeof(p));
}
#endif /* ARTS_PROTOCOL_EXCL */

/* The OWNER-only senders (CONFIRM, CONFIRM_ACK, REDIRECT_RO) live in
 * coherence/owner.c alongside their handlers; the RCU OWNERSHIP_RESPONSE
 * senders live in coherence/home.c / coherence/owner.c. */
