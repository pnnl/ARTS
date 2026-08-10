/* SPDX-License-Identifier: Apache-2.0
 *
 * MSI protocol, OWNER placement.
 *
 * OWNER means the canonical payload stays with whoever wrote it last and moves
 * only on demand, so the home holds no bytes at all.  That single fact is this
 * file: an RO acquire never reads a local home buffer, a release publishes
 * control only (the round still runs — the invalidation IS what makes this
 * MSI), and a read request at the home is redirected to the current grant
 * holder, which serves the requester directly.
 *
 * Everything else — the reader plane, the invalidation round, the cache and
 * directory lifecycles, the senders — is in msi/directory.c, and write
 * ownership is the shared migrating grant (coherence/grant.c).
 *
 * The redirect is a retry, never a park: a rank that no longer holds the bytes
 * bounces the request back through the home, and each pass re-resolves against
 * a later ownership generation.  Nothing queues and nothing sleeps, which is
 * what keeps a reader from ever waiting behind a writer.
 */
#include "arts/counter/object_counter.h"
#include "arts/coherence/inv/types.h"

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/directory.h"
#include "arts/coherence/handlers.h"
#include "arts/db.h"
#include "arts/edt.h"
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/system/identity.h"
#include "arts/utils/atomics.h"
#include "arts/ooo.h"
#include "arts/transport/net.h"
#include "arts/transport/protocol.h"
#include "arts/utils/malloc.h"
#include "arts/counter/Preamble.h"

/* ===== acquire ==========================================================
 * The two planes are answered independently, because they are independent:
 * a read never waits on a writer and a write never waits on a reader.
 *
 *   RO — no rank can serve a read from a home buffer here (there is none), so
 *        a rank without a covering copy always fetches: it joins or opens the
 *        single in-flight fetch, and the home redirects that fetch to the
 *        current holder.
 *   RW — the migrating sentinel grant, verbatim from the shared plane.
 */

bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  /* A write acquire can wait on another rank's release (cross-DB cyclic
   * hazard: acquire-all must issue it in strict slot order).  A read acquire
   * waits only on the serve — never on another EDT's release — so it carries
   * no cycle risk and may issue in parallel. */
  return mode == DB_MODE_RW;
}

void arts_handler_db_acquire(void *item, void *args) {
  struct arts_db_s *db = (struct arts_db_s *)item;
  struct arts_ooo_args_db_acquire_s *a =
      (struct arts_ooo_args_db_acquire_s *)args;
  struct arts_edt_s *edt = a->edt;
  unsigned int slot = a->slot;
  struct arts_db_cache_s *cache = &db->cache;
  arts_edt_dep_t *dep = &((arts_edt_dep_t *)arts_get_depv(edt))[slot];
  arts_db_access_mode_t mode = dep->mode;

  if (mode != DB_MODE_RO) {
    /* RW: the shared grant.  writer_count > 0 IS "this rank holds it" — but an
     * install whose directory flip the home has not yet published may NOT run:
     * its stores would be observable while the directory still names the
     * previous owner, and a reader registered in that window is redirected to
     * that rank's retained (now stale) copy.  The CONFIRM_ACK opens the gate
     * and drains whatever parked behind it. */
    if (arts_atomic_read(&cache->grant_unconfirmed) == 0 &&
        arts_db_acquire_rw_local_fast(cache, dep)) {
      arts_db_acquire_resolved(edt, slot);
      return;
    }
    arts_db_acquire_remote_rw(cache, edt->guid, slot); /* parks */
    return;
  }

  /* Covering-copy fast path: pure loads, no CAS, no node.  A rank holding a
   * CONFIRMED grant covers reads too — it has the newest bytes by definition.
   * An unconfirmed install does not: until the home publishes the flip, the
   * bytes here are not yet the ones the directory points readers at. */
  uint64_t peek =
      atomic_load_explicit(&cache->cache_state, memory_order_acquire);
  if (MSI_CACHE_RO(peek) == MSI_RO_VALID ||
      ((int)arts_atomic_read(&cache->writer_count) > 0 &&
       arts_atomic_read(&cache->grant_unconfirmed) == 0)) {
    /* A durable copy answers with no message and no CAS — still an
     * acquire served locally, so it belongs in the same census the
     * other arms feed through arts_db_acquire_local. */
    INCREMENT_NUM_DB_ACQUIRE_LOCAL_HIT_BY(1);
    arts_object_acquire(false);
    mark_edt_ready_by_guid(edt->guid, slot);
    return;
  }

  /* Park-or-open: the decision CAS links our node as the chain head in the
   * same atom (node fields written before every attempt). */
  uint32_t idx = inv_waiter_alloc(cache);
  struct arts_db_inv_waiter_s *node = inv_waiter_ptr(cache, idx);
  node->edt_guid = edt->guid;
  node->slot = slot;

  uint32_t act;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&cache->cache_state, memory_order_acquire);
    node->next = MSI_CACHE_HEAD_RO(cur);
    next = inv_cache_compute_next(cur, MSI_CACHE_OP_ACQ_RO, idx, &act);
    if (next == cur) {
      break; /* copy turned valid mid-retry: no word write */
    }
  } while (!atomic_compare_exchange_weak_explicit(&cache->cache_state, &cur,
                                                  next, memory_order_acq_rel,
                                                  memory_order_acquire));

  switch (act) {
  case MSI_CACHE_ACT_SELF_SERVE:
    /* The copy turned valid while we were deciding — served from here. */
    INCREMENT_NUM_DB_ACQUIRE_LOCAL_HIT_BY(1);
    arts_object_acquire(false);
    inv_waiter_free(cache, idx);
    mark_edt_ready_by_guid(edt->guid, slot);
    break;
  case MSI_CACHE_ACT_SEND_RO:
    INCREMENT_NUM_DB_ACQUIRE_REMOTE_BY(1);
    arts_object_acquire(true);
    arts_send_db_inv_request(cache, DB_MODE_RO);
    break;
  default: /* PARK: the fetch's committer will serve us */
    /* Parked behind an open fetch: still an acquire this rank could not
     * answer, so it counts with the one that issued the fetch. */
    INCREMENT_NUM_DB_ACQUIRE_REMOTE_BY(1);
    arts_object_acquire(true);
    break;
  }
}

/* ===== release_rw (OWNER placement) =====================================
 * The HOME arm with one argument changed: the payload does not travel to the
 * home, because the home never serves it.  The round still runs — that is what
 * makes this MSI and not RCU — so the release still returns only once every
 * stale copy is dead, and the count is still dropped only after the round
 * acks, which is what orders any ownership transfer behind it.
 */
void arts_db_release_rw(struct arts_db_cache_s *cache) {
  /* Defensive: writer_count==0 means our acquire never bumped the grant (e.g.
   * a cache already torn down by a destroy fan-out); decrementing would
   * underflow. */
  if (arts_atomic_read(&cache->writer_count) == 0) {
    return;
  }
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  uint64_t new_version = 0;
  if (buf != NULL) {
    arts_atomic_add_u64(&buf->version, 1);
    new_version = arts_atomic_read_u64(&buf->version);
    arts_db_buf_release(&buf_h);
  }
  /* Control-only publish: ask the home for this release's invalidation round
   * and block until every ack is in.  No bytes move. */
  TIME_INVALIDATE_ROUND_START();
  arts_db_publish_sync(cache, new_version, /*data=*/NULL, /*data_size=*/0);
  TIME_INVALIDATE_ROUND_STOP();

  int rest = (int)arts_atomic_sub(&cache->writer_count, 1); /* post value */
  if (rest == 0 && cache->incoming_new_owner != ARTS_NO_PENDING_OWNER) {
    /* An INVALIDATE named a transfer target while writers were live; this
     * (last) releaser is the unique actor that ships the owner->owner
     * transfer. */
    arts_db_send_grant_response(cache);
  }
}

/* ===== read request at the home =========================================
 * The home holds no bytes, so it cannot answer.  It registers the requester in
 * the roster — BEFORE the redirect departs, so a copy that is about to exist is
 * already a target of the next round — and points the requester at the rank the
 * directory currently names.
 */
void arts_handler_db_inv_request(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_ooo_args_db_inv_request_s *a =
      (struct arts_ooo_args_db_inv_request_s *)args_v;
  struct arts_db_cache_s *cache = &db->cache;
  (void)arts_rank_bitset_set(&db->cached_ranks, a->requester);
  if (a->rdzv.txid == 0 && cache->db_size != 0 &&
      a->requester != arts_global_rank_id) {
    /* First touch: the request carried no landing (db_size unknown at the
     * requester).  Answer the size and let it re-issue with one. */
    arts_send_db_inv_cts(a->requester, cache->db_guid, cache->db_size, a->mode);
    return;
  }
  if (a->mode != DB_MODE_RO) {
    /* There is no write branch: RW acquires go through the shared grant's
     * OWNERSHIP_REQUEST.  This handler serves readers only. */
    return;
  }
  /* Roster bit BEFORE the serve/redirect departs. */
  (void)arts_rank_bitset_set(&db->roster, a->requester);
  unsigned int holder =
      atomic_load_explicit(&db->rw_holder, memory_order_acquire);
  if (holder == arts_global_rank_id) {
    /* The home happens to hold the grant: serve from here, no redirect. */
    arts_send_db_inv_deliver(a->requester, db, &a->rdzv);
    return;
  }
  arts_send_db_inv_redirect(holder, cache->db_guid, a->requester, &a->rdzv);
}

/* ===== redirect (home -> current holder) ================================
 * The holder serves the named requester from its own buffer.  A MISS — the
 * bytes left this rank between the home's directory read and this arrival —
 * bounces the request back to the home rather than queueing: nothing here may
 * block a reader, and the next pass resolves against a later generation.
 */
void arts_handler_db_inv_redirect(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_ooo_args_db_inv_redirect_s *a =
      (struct arts_ooo_args_db_inv_redirect_s *)args_v;
  struct arts_db_cache_s *cache = &db->cache;
  /* "Do I still hold the bytes?" is a question only a DB that HAS bytes can be
   * asked.  A sentinel DB (db_size == 0) never installs a buffer — NULL is its
   * defined value — so testing for one would bounce every read back to the
   * home forever.  Serve it unconditionally; the reply is header-only. */
  if (cache->db_size != 0) {
    arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
    bool have = (arts_shared_get(buf_h) != NULL);
    arts_db_buf_release(&buf_h);
    if (!have) {
      /* The bytes moved on between the home's directory read and this
       * arrival.  Bounce rather than queue — a reader must never wait here —
       * and let the next pass resolve against a later generation. */
      arts_send_db_inv_request_for(cache->db_guid, a->requester, &a->rdzv);
      return;
    }
  }
  arts_send_db_inv_deliver(a->requester, db, &a->rdzv);
}


/* ===== OWNER-only senders =============================================== */

/* home → the current grant holder: serve this reader.  The subject rides in
 * the packet because the holder answers it directly, so the wire sender and
 * the requester are different ranks. */
void arts_send_db_inv_redirect(unsigned int holder_rank, arts_guid_t db_guid,
                               unsigned int requester,
                               const struct arts_rdzv_landing_s *rdzv) {
  struct arts_msg_inv_redirect_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_INV_REDIRECT);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.requester = requester;
  p.pad = 0;
  p.rdzv.addr = rdzv->addr;
  p.rdzv.key = rdzv->key;
  p.rdzv.txid = rdzv->txid;
  p.rdzv.cookie = rdzv->cookie;
  if (holder_rank == arts_global_rank_id) {
    struct arts_ooo_args_db_inv_redirect_s args = {
        .requester = requester, .db_guid = db_guid, .rdzv = *rdzv};
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_INV_REDIRECT, &args,
                                    sizeof(args));
    return;
  }
  arts_transport_send_async((int)holder_rank, (char *)&p, sizeof(p));
}

/* The bounce: a holder that no longer has the bytes re-sends the read request
 * to the home on the reader's behalf.  A retry, not a park — the next pass
 * resolves against a later ownership generation. */
void arts_send_db_inv_request_for(arts_guid_t db_guid, unsigned int requester,
                                  const struct arts_rdzv_landing_s *rdzv) {
  unsigned int home_rank = arts_guid_get_rank(db_guid);
  struct arts_msg_inv_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_INV_REQUEST);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.mode = (uint32_t)DB_MODE_RO;
  p.requester = requester;
  p.rdzv.addr = rdzv->addr;
  p.rdzv.key = rdzv->key;
  p.rdzv.txid = rdzv->txid;
  p.rdzv.cookie = rdzv->cookie;
  if (home_rank == arts_global_rank_id) {
    struct arts_ooo_args_db_inv_request_s args = {.requester = requester,
                                                  .db_guid = db_guid,
                                                  .mode = DB_MODE_RO,
                                                  .rdzv = *rdzv};
    arts_ooo_dispatch_or_defer_guid(db_guid, OOO_DB_INV_REQUEST, &args,
                                    sizeof(args));
    return;
  }
  arts_transport_send_async((int)home_rank, (char *)&p, sizeof(p));
}
