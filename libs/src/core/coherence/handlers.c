/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence protocol wire-message handlers (receive side) + home-side
 * dedup / ownership-transfer helpers.
 *
 * The matching arts_send_db_* SENDERS (packet-fill + outbox enqueue) live
 * in coherence_senders.c.
 *
 * Lookup discipline.  Two handler categories, both lookup-then-operate but
 * differing on the MISS action:
 *   - Cat-B (deferrable home-side: OWNERSHIP_REQUEST / GET_DATA / WRITEBACK / DESTROY):
 *     the wire dispatcher routes through the OoO engine, which acquires the
 *     home db_s (ref-pinned) and hands a pure (item, args) body the live cache
 *     on a HIT, or DEFERS the args and replays them once DB_CREATE installs.
 *   - Cat-C (non-deferrable: DATA_RESPONSE / DESTROY_NOTIFY / WRITEBACK_ACK /
 *     RELEASE_OWNERSHIP / REDIRECT_RO / INSTALL_ACK): the wire dispatcher
 *     (and the matching self-send shortcut) does the ref-pinned
 *     lookup-acquire; on a HIT it calls the pure (item, args) body, and on a
 *     MISS it applies that handler's exact miss-action (silent drop,
 *     DESTROY_NOTIFY reply, or the WRITEBACK_ACK sem-post — see each body).
 * Either way the handler body itself performs NO route-table lookup; it
 * operates on the already-acquired, ref-pinned cache (the FIRST member of the
 * db_s the caller hands it).
 *
 * Memory ordering: ARTS atomics are __sync_*-based (full fence) and
 * the per-rank network thread (S1) is the sole writer of home.*
 * state, so the trickier orderings are confined to:
 *   - cache.writer_count (worker ↔ network handler)
 *   - cache.buffer       (worker ↔ network handler installs)
 *   - cache.pending_count (worker push ↔ marker mark)
 * All accessed via arts_atomic_*, all matching the algorithm in the
 * coherence design plan.
 */

#include "arts/coherence/handlers.h"

#include <assert.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h" /* ARTS_DB_PROP_NO_ACQUIRE */
#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/home.h"
#include "arts/db.h"
#include "arts/gas/route_table.h"
#include "arts/ooo.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"

/* ===== Home-side handlers ========================================== */

/* arts_handler_db_ownership_request lives in coherence/release.c (RC+LRC
 * only — LC has no OWNERSHIP_REQUEST / GRANT round). */

/* arts_handler_db_snapshot_request (GET_DATA) is model-specific — RC/LC serve
 * from home's canonical buffer (dedup), LRC records the sharer + REDIRECTs to
 * the owner — so its whole body lives in coherence/{rc,lrc,lc}.c. */

/* arts_handler_db_writeback (+_ack) is model-specific — RC/LC install + ACK
 * (RC additionally advances the ownership chain on WB_AND_TRANSFER), LRC has no
 * synchronous writeback (no-op fillers preserve the OoO-table / link parity) —
 * so their whole bodies live in coherence/{rc,lrc,lc}.c. */

/* arts_handler_db_ownership_return lives in coherence/release.c (RC+LRC
 * only — LC has no ownership chain). */

/* arts_handler_db_destroy is model-specific — the roster fan-out source differs
 * (RC/LC walk home->last_sent_version; LRC walks rw_holder + cached_ranks +
 * pending_rw) — so its whole body lives in coherence/{rc,lrc,lc}.c.  All
 * three skeletons run fan-out + arts_db_fail_trigger_pending FIRST, then
 * arts_route_table_set_destroyed LAST. */

void arts_handler_db_create(struct arts_msg_db_create_coherent_packet_s *p) {
  /* Home-side init for non-home creator.  Per coherence design plan
   * §968-988: install zero-init buffer, home struct with rw_holder =
   * creator_rank, writer_count = 0 (home is non-owner). */
  unsigned int creator_rank = p->header.rank;
  arts_guid_t db_guid = p->db_guid;
  uint64_t db_size = p->db_size;
  /* NO_ACQUIRE: the creator neither acquires nor writes back, so home is the
   * sole idle owner (not a non-owner awaiting a creator writeback). */
  bool no_acquire = (p->flags & ARTS_DB_PROP_NO_ACQUIRE) != 0;

  /* This handler installs/initializes the home directory, so it is only ever
   * dispatched to the GUID's home rank — where the descriptor is always a
   * full allocation.  A cache-only stub (which omits the home-arm fields)
   * is only ever installed on non-home ranks, so the home-field writes below
   * (arts_db_home_init / rw_holder) stay in bounds. */
  assert((unsigned int)arts_guid_get_rank(db_guid) == arts_global_rank_id &&
         "home-directory init must run on the GUID home rank");

  /* Race against lazy_install or another path that already set up an
   * empty cache_s on this rank — coalesce by promoting the existing
   * lazy entry rather than allocating a duplicate. */
  arts_shared_ptr_t existing_h = arts_route_table_lookup_db(db_guid);
  struct arts_db_s *existing = (struct arts_db_s *)arts_shared_get(existing_h);
  if (existing != NULL && existing->db_type == ARTS_DB) {
    struct arts_db_cache_s *cache = &existing->cache;
    struct arts_db_s *db = existing;
    if (arts_db_buf_peek(cache) == NULL && db_size > 0) {
      arts_db_buf_install(cache, 1, NULL, db_size);
    }
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
    if (!db->home_initialized) {
      arts_db_home_init(db, creator_rank, arts_global_rank_count);
      db->home_initialized = true;
    } else {
      arts_db_create_publish_holder(db, creator_rank);
    }
    arts_shared_release(&existing_h);
    return;
  }
  if (existing != NULL) {
    /* existing non-ARTS_DB entry (no coherence cache) — drop the ref and
     * proceed to the install/coalesce branch below. */
    arts_shared_release(&existing_h);
  }

  /* No existing entry -- allocate the db_s stub (cache embedded), install in
   * route_table.
   *
   * Lazy buffer install (OCR pattern): HOME_RECV does NOT install a
   * buffer here.  cache->buffer stays NULL with version 0 -- "metadata
   * only" state.  The first WRITEBACK from the creator's release_rw
   * installs the buffer at home (version 1+, with the creator's
   * payload).  Cross-rank GET_DATA before that point is served as a
   * no-payload DATA_RESPONSE (handle_get_data); the requesting rank
   * sees ptr=NULL (per OCR spec ch2:832-839 "value of the created data
   * block is undefined" -- ARTS interprets this as "before any writer
   * has published, no data exists; reading is application's
   * responsibility"). */
  struct arts_db_s *stub =
      (struct arts_db_s *)arts_malloc_align(sizeof(struct arts_db_s), 16);
  memset(stub, 0, sizeof(struct arts_db_s));
  stub->db_type = (arts_db_types_t)p->db_type;
  if (no_acquire) {
    /* Home is the idle RW owner from creation — identical to a locally created
     * DB (CREATOR_HOME: rw_holder = self, writer_count > 0).  Install a
     * zero-init buffer so the first consumer's RW acquire is granted locally
     * with a writable payload.  HOME_RECV (rw_holder = creator) would route
     * that acquire's INVALIDATE_NOTICE to a creator that holds no cache — a
     * phantom holder — and the acquire would stall forever. */
    arts_db_cache_init(&stub->cache, db_guid, db_size,
                       ARTS_DB_INIT_CREATOR_HOME, creator_rank);
    if (db_size > 0) {
      arts_db_buf_install(&stub->cache, /*new_version=*/1,
                          /*data_payload=*/NULL, db_size);
    }
  } else {
    arts_db_cache_init(&stub->cache, db_guid, db_size, ARTS_DB_INIT_HOME_RECV,
                       creator_rank);
    /* Case-D leaf: LC installs a version-1 zero buffer now (home is canonical,
     * no creator writeback to wait for); RC/LRC keep the lazy OCR install
     * (no-op). */
    arts_db_create_install_home_buffer(&stub->cache, db_size);
  }

  if (arts_route_table_install_if_absent(stub, db_guid, arts_global_rank_id,
                                         /*used=*/true)) {
    arts_ooo_drain_guid(db_guid);
    return;
  }

  /* Lost race — free our stub and coalesce into the existing entry. */
  arts_db_free(stub);
  arts_shared_ptr_t winner_h = arts_route_table_lookup_db(db_guid);
  struct arts_db_s *winner = (struct arts_db_s *)arts_shared_get(winner_h);
  if (winner != NULL && winner->db_type == ARTS_DB) {
    struct arts_db_cache_s *cache = &winner->cache;
    struct arts_db_s *db = winner;
    if (arts_db_buf_peek(cache) == NULL && db_size > 0) {
      arts_db_buf_install(cache, 1, NULL, db_size);
    }
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
    if (!db->home_initialized) {
      arts_db_home_init(db, creator_rank, arts_global_rank_count);
      db->home_initialized = true;
    } else {
      arts_db_create_publish_holder(db, creator_rank);
    }
  }
  if (winner != NULL) {
    arts_shared_release(&winner_h);
  }
}

/* ===== Sharer-side response handlers =============================== */

/* The RC GRANT handler arts_handler_db_ownership_response lives in
 * coherence/rc.c; LRC's TRANSFER_OWNERSHIP overload lives in
 * coherence/lrc.c; LC has no ownership transfer (dispatcher fatals). */

/* Cat-C pure body (DATA_RESPONSE).  The wire dispatcher / self-send shortcut
 * has already looked the home db_s up with a held ref and passes it as item_v
 * (cache is its FIRST member, offset 0, so item_v IS the cache).  No
 * lookup/NULL-check here — the dispatcher's MISS branch SILENTLY DROPS (this
 * 1:1 response resumes a parked EDT; a missing cache means it was torn down).
 *
 * No acquire-time list registration: this 1:1 response resumes the parked EDT
 * (a->edt_guid, a->slot) directly.  Three cases over a monotonic version:
 *   1. a->version <= buf->version : nothing newer to install — resume self
 *      against the live buffer.
 *   2. data + a->version > buf->version : install + drain-all the reorder
 *      buffer + resume self.
 *   3. NO_DATA + a->version > buf->version : the with-data reply was
 *      reordered behind us — push self onto pending_snapshot (a future
 *      case-2 install drains us) + re-check (race recovery).
 * Shared verbatim by RC/LRC/LC (LC routes RW through here too). */
void arts_handler_db_snapshot_response(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_db_snapshot_response_args_s *a =
      (struct arts_db_snapshot_response_args_s *)args_v;
  arts_guid_t edt_guid = a->edt_guid;
  uint32_t slot = a->slot;

  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  uint64_t buf_v = buf ? buf->version : 0;
  if (buf != NULL) {
    arts_db_buf_release(&buf_h);
  }

  if (a->version <= buf_v) {
    /* Case 1: nothing newer to install — resume self. */
    mark_edt_ready_by_guid(edt_guid, slot);
    return;
  }
  if (a->data_present) {
    /* Case 2: install (version-conditional publish inside install_buffer;
     * stale installs retreat) + drain-all + resume self. */
    arts_db_buf_install(cache, a->version, a->data, a->data_size);
    arts_db_drain_pending_snapshot(cache);
    mark_edt_ready_by_guid(edt_guid, slot);
    return;
  }
  /* Case 3: NO_DATA arrived ahead of the with-data reply.  Park a
   * reorder-buffer node; a later case-2 install drains it. */
  struct arts_db_snapshot_waiter_s *w =
      (struct arts_db_snapshot_waiter_s *)arts_malloc(sizeof(*w));
  w->edt_guid = edt_guid;
  w->slot = slot;
  w->target_version = a->version;
  arts_lf_stack_push(&cache->pending_snapshot, &w->link);
  /* Race recovery: a concurrent case-2 install may have published the buffer
   * between our version read and the push.  If so, drain (our own node
   * included) so we don't park forever.  The atomic_exchange drain is the
   * single-actor primitive — a concurrent installer's drain and ours cannot
   * both claim the same node. */
  arts_shared_ptr_t rch = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *rbuf =
      (struct arts_db_buffer_s *)arts_shared_get(rch);
  uint64_t rv = rbuf ? rbuf->version : 0;
  if (rbuf != NULL) {
    arts_db_buf_release(&rch);
  }
  if (rv >= a->version) {
    arts_db_drain_pending_snapshot(cache);
  }
}

/* arts_handler_db_ownership_invalidate (INVALIDATE_NOTICE) lives per model:
 * coherence/rc.c (commutative signed counter) and coherence/lrc.c
 * (publish-target-then-withdraw).  LC never sends INVALIDATE (dispatcher
 * fatals). */

/* arts_handler_db_writeback_ack is model-specific — RC/LC post the releaser's
 * stack-local sem_t (pointer identity), LRC has no synchronous writeback (no-op
 * filler for OoO-table / link parity) — so its whole body lives in
 * coherence/{rc,lrc,lc}.c. */

/* Cat-C pure body (DESTROY_NOTIFY).  The wire dispatcher / self-send shortcut
 * has already looked the cache up with a held ref and passes the db_s as item_v
 * (cache is its FIRST member, offset 0).  No lookup/NULL-check here — the
 * dispatcher's MISS branch SILENTLY DROPS (already torn down on this rank;
 * cb-NULL = idempotent, a second DESTROY_NOTIFY is a no-op).
 *
 * cb-model destroy (single-actor): no destroy_state gate.  fail_trigger FIRST
 * so the cache is alive while we wake waiters (mark_delete drops only the
 * install ref; the cb deleter frees once outstanding lookup refs drain). */
void arts_handler_db_cache_destroy(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_db_cache_destroy_args_s *a =
      (struct arts_db_cache_destroy_args_s *)args_v;
  arts_db_fail_trigger_pending(cache);
  (void)arts_route_table_set_destroyed(a->db_guid);
}

/* The LRC REDIRECT_RO handler arts_handler_db_snapshot_redirect lives in
 * coherence/lrc.c (owner-side, REDIRECT only exists under LRC). */
