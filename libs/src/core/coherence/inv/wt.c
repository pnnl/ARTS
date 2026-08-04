/* SPDX-License-Identifier: Apache-2.0
 *
 * MSI protocol, HOME placement.
 *
 * HOME means the canonical payload comes back to the data block's home rank at
 * every release, so the home can serve readers from it.  That single fact is
 * this file: an RO acquire on the home rank reads the home buffer directly, a
 * release publishes its payload along with the round request, and a read
 * request at the home is answered from the home's own bytes.
 *
 * Everything else — the reader plane, the invalidation round, the cache and
 * directory lifecycles, the senders — is in msi/directory.c, and write
 * ownership is the shared migrating grant.
 */
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
#include "arts/utils/malloc.h"

/* ===== acquire ==========================================================
 * The two planes are answered independently, because they are independent:
 * a read never waits on a writer and a write never waits on a reader.
 *
 *   RO — served from the home (HOME placement puts current bytes there).  A
 *        rank holding a VALID copy short-circuits to a pure load; otherwise it
 *        joins or opens the single in-flight fetch.
 *   RW — the migrating sentinel grant, verbatim from the shared plane: bump
 *        the count if this rank already holds the grant, else park on
 *        pending_rw and let the coalescing flag decide who requests.
 */

bool arts_db_acquire_is_serialized(arts_db_access_mode_t mode) {
  /* A write acquire can wait on another rank's release (cross-DB cyclic
   * hazard: acquire-all must issue it in strict slot order).  A read acquire
   * waits only on the home's reply — never on another EDT's release — so it
   * carries no cycle risk and may issue in parallel. */
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
  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);

  if (mode != DB_MODE_RO) {
    /* RW: the shared grant.  writer_count > 0 IS "this rank holds it". */
    if (arts_db_acquire_rw_local_fast(cache, dep)) {
      arts_db_acquire_resolved(edt, slot);
      return;
    }
    arts_db_acquire_remote_rw(cache, edt->guid, slot); /* parks */
    return;
  }

  if (is_home) {
    /* The home's installed buffer is the canonical copy — serve it.  Before
     * the creator's first publish there is nothing to serve, so hold on the
     * snapshot reorder buffer and let the install's drain resume us. */
    if (cache->db_size != 0) {
      arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
      bool published = (arts_shared_get(buf_h) != NULL);
      arts_db_buf_release(&buf_h);
      if (!published) {
        struct arts_db_snapshot_waiter_s *w =
            (struct arts_db_snapshot_waiter_s *)arts_malloc(sizeof(*w));
        w->edt_guid = edt->guid;
        w->slot = slot;
        w->target_version = 0;
        w->serve = NULL; /* local waiter: the drain resumes the EDT */
        w->requester = arts_global_rank_id;
        w->rdzv = (struct arts_rdzv_landing_s){0, 0, 0, 0};
        arts_lf_stack_push(&cache->pending_snapshot, &w->link);
        /* Race recovery: an install may have landed between the read and the
         * push — drain (our own node included) so nobody parks forever. */
        buf_h = arts_db_buf_acquire(cache);
        published = (arts_shared_get(buf_h) != NULL);
        arts_db_buf_release(&buf_h);
        if (published) {
          arts_db_drain_pending_snapshot(cache);
        }
        return;
      }
    }
    mark_edt_ready_by_guid(edt->guid, slot);
    return;
  }

  /* Covering-copy fast path: pure loads, no CAS, no node.  This is the whole
   * point of a write-invalidate sharer plane — a reader that holds a valid
   * copy costs nothing until a writer kills it.  A rank holding the grant
   * covers reads too: it has the newest bytes by definition. */
  uint64_t peek =
      atomic_load_explicit(&cache->cache_state, memory_order_acquire);
  if (MSI_CACHE_RO(peek) == MSI_RO_VALID ||
      (int)arts_atomic_read(&cache->writer_count) > 0) {
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
    inv_waiter_free(cache, idx);
    mark_edt_ready_by_guid(edt->guid, slot);
    break;
  case MSI_CACHE_ACT_SEND_RO:
    arts_send_db_inv_request(cache, DB_MODE_RO);
    break;
  default: /* PARK: the fetch's committer will serve us */
    break;
  }
}

/* ===== release_rw (HOME placement) ======================================
 * Identical in shape to every other grant-bearing arm: bump the version,
 * publish, drop the count, and on the 0-edge ship the grant onward.  Two
 * things distinguish this arm, and only these two:
 *
 *   - HOME placement means the publish carries the payload, so the home holds
 *     current bytes and can serve readers from them;
 *   - MSI means the home does not acknowledge that publish until it has run an
 *     invalidation round over the sharer roster and collected every ack.  The
 *     release therefore returns only when no stale copy of this data block
 *     exists anywhere.
 *
 * Both live behind the ONE blocking call below.  Everything after it is the
 * shared grant discipline: the count is dropped AFTER the round acks, so the
 * 0-edge — and therefore any ownership transfer — is structurally ordered
 * behind this release's round.  An INVALIDATE arriving mid-round is harmless:
 * it withdraws the sentinel and names the next owner, but this releaser still
 * holds its own count, so the ship happens here, at its decrement.
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
  }
  bool is_home = (arts_guid_get_rank(cache->db_guid) == arts_global_rank_id);
  /* A home owner's buffer IS the home buffer, so no bytes move — but the round
   * still runs: it is what retires the remote copies. */
  if (buf != NULL) {
    arts_db_publish_sync(cache, new_version, is_home ? NULL : buf->data,
                         is_home ? 0u : cache->db_size);
  }
  if (buf != NULL) {
    arts_db_buf_release(&buf_h);
  }

  int rest = (int)arts_atomic_sub(&cache->writer_count, 1); /* post value */
  if (rest == 0 && cache->incoming_new_owner != ARTS_NO_PENDING_OWNER) {
    /* An INVALIDATE named a transfer target while writers were live; this
     * (last) releaser is the unique actor that ships the owner->owner
     * transfer. */
    arts_db_send_grant_response(cache);
  }
}

/* arts_db_release_ro: the shared no-op body in coherence.c applies — an MSI
 * read release touches no protocol state (a valid copy persists; the EDT's
 * buffer ref is dropped by the dep-release path). */

/* ===== arts_handler_db_destroy =========================================
 * OOO_DB_DESTROY Cat-B body.  Fan-out DESTROY_NOTIFY to every rank in
 * cached_ranks, then detach the route-table slot. */

/* Deferred pre-publication read serve: re-issued by the pending_snapshot
 * drain once the first install lands.  Same 3-step as the direct serve
 * (roster bit before the send). */
static void inv_pending_ro_serve(struct arts_db_cache_s *cache,
                                 struct arts_db_snapshot_waiter_s *w) {
  struct arts_db_s *db = (struct arts_db_s *)cache; /* cache is FIRST member */
  (void)arts_rank_bitset_set(&db->roster, w->requester);
  arts_send_db_inv_deliver(w->requester, db, &w->rdzv);
}

void arts_handler_db_inv_request(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_ooo_args_db_inv_request_s *a =
      (struct arts_ooo_args_db_inv_request_s *)args_v;
  struct arts_db_cache_s *cache = &db->cache;
  (void)arts_rank_bitset_set(&db->cached_ranks, a->requester);
  if (a->rdzv.txid == 0 && cache->db_size != 0 &&
      a->requester != arts_global_rank_id) {
    /* First touch: the request carried no landing (db_size unknown at the
     * requester).  It was NOT queued/served — answer the size and let the
     * requester re-issue with a landing. */
    arts_send_db_inv_cts(a->requester, cache->db_guid, cache->db_size,
                         a->mode);
    return;
  }
  if (a->mode == DB_MODE_RO) {
    if (cache->db_size != 0) {
      arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
      bool published = (arts_shared_get(buf_h) != NULL);
      arts_db_buf_release(&buf_h);
      if (!published) {
        /* Pre-publication hold: no release has published this DB yet, and a
         * read that is event-ordered after the creator's writes must observe
         * the release-published bytes — serving now would hand out
         * never-published state.  Push-then-recheck: a concurrent first
         * install either sees our node in its drain or we drain ourselves. */
        struct arts_db_snapshot_waiter_s *w =
            (struct arts_db_snapshot_waiter_s *)arts_malloc(sizeof(*w));
        w->edt_guid = NULL_GUID;
        w->slot = 0;
        w->target_version = 0;
        w->serve = inv_pending_ro_serve;
        w->requester = a->requester;
        w->rdzv = a->rdzv;
        arts_lf_stack_push(&cache->pending_snapshot, &w->link);
        buf_h = arts_db_buf_acquire(cache);
        published = (arts_shared_get(buf_h) != NULL);
        arts_db_buf_release(&buf_h);
        if (published) {
          arts_db_drain_pending_snapshot(cache);
        }
        return;
      }
    }
    /* Read serve: roster bit BEFORE the send — a serve whose copy departs
     * must already be a target of the next round. */
    (void)arts_rank_bitset_set(&db->roster, a->requester);
    arts_send_db_inv_deliver(a->requester, db, &a->rdzv);
    return;
  }
  /* There is no write branch: RW acquires go through the shared grant's
   * OWNERSHIP_REQUEST, which the home answers by naming the next holder and
   * revoking the current one.  This handler serves readers only. */
}

