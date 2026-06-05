/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence protocol wire-message handlers (receive side) + home-side
 * dedup / ownership-transfer helpers.
 *
 * The matching arts_send_db_* SENDERS (packet-fill + outbox enqueue) live
 * in coherence_senders.c.
 *
 * The protocol's drop-discipline (lookup -> destroy_state precheck ->
 * either OoO defer or explicit wake-up reply on failure) is
 * implemented via the `arts_coh_home_lookup_or_defer` helper at the top of this
 * file; every home-side handler entry funnels through it.  Sharer-side
 * response handlers do their own lookup + cache.destroy_state check
 * inline (no requester to notify back -- the message *is* the
 * requester's own context).
 *
 * Memory ordering: ARTS atomics are __sync_*-based (full fence) and
 * the per-rank network thread (S1) is the sole writer of home.*
 * state, so the trickier orderings are confined to:
 *   - cache.writer_count (worker ↔ network handler)
 *   - cache.buffer       (worker ↔ network handler installs)
 *   - cache.pending_count (worker push ↔ marker mark)
 *   - cache.destroy_state (forward-only tri-state)
 * All accessed via arts_atomic_*, all matching the algorithm in the
 * coherence design plan.
 */

#include "arts/db_coherence_handlers.h"

#include <assert.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "arts.h" /* ARTS_DB_PROP_NO_ACQUIRE */
#include "arts/db.h"
#include "arts/db_coherence.h"
#include "arts/db_coherence_buffer.h"
#include "arts/db_coherence_home.h"
#include "arts/db_coherence_model.h" /* per-model handler-body hooks */
#include "arts/gas/route_table.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"

/* ===== Forward decls: handler wiring with B4/B5/B11 ================= */

/* These are implemented in coherence.c (acquire/release/destroy) and
 * called from the response handlers below.  Forward-declared here to
 * avoid a circular include with coherence.c (which itself includes this
 * header for the sender helpers).  (arts_coh_drain_pending_rw_after_grant and
 * mark_edt_ready_by_guid are declared in db_coherence_model.h.) */
void arts_coh_drain_pending_snapshot(struct arts_db_cache_s *cache);
void arts_coh_fail_trigger_pending(struct arts_db_cache_s *cache);

/* ===== home_lookup_or_defer helper ============ */

/* Look up cache by guid; if not yet installed, defer the wire message
 * via the OoO list so it re-issues once DB_CREATE arrives.  Spec §4.9.
 *
 * Behaviour matrix:
 *   cache != NULL                           -> return cache (caller body)
 *   cache == NULL, packet_for_oo == NULL    -> reply per reply_kind, NULL
 *   cache == NULL, ENQUEUED                 -> NULL (fire_oo will retry)
 *   cache == NULL, FIRED_BY_DRAIN           -> NULL (handler already ran via
 *                                              the drain triggered inside
 *                                              add_oo_ex; payload freed)
 *   cache == NULL, AVAILABLE_NOW (race)     -> re-lookup; same precheck
 *
 * The OoO payload is a heap copy of `packet_for_oo` whose first field is
 * forced to `oo_type` (every OOO_DB_* payload begins with oo_type_t).  Declared
 * in db_coherence_model.h so db_coherence_release.c can call it. */
struct arts_db_cache_s *
arts_coh_home_lookup_or_defer(arts_guid_t guid, unsigned int requester,
                              ooo_kind_t oo_type, void *packet_for_oo,
                              size_t packet_size, coh_reply_kind_t reply_kind) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(guid);

  if (cache != NULL) {
    return cache;
  }

  /* cache == NULL — defer via OoO or reply immediately. */
  if (packet_for_oo == NULL) {
    if (reply_kind == COH_REPLY_DESTROY_NOTIFY) {
      arts_send_db_cache_destroy(requester, guid);
    }
    return NULL;
  }

  /* Defer the replay.  dispatch_or_defer copies `packet_for_oo` (the kind's
   * args struct) into the OoO payload; on a concurrent install it re-issues
   * this handler inline (which re-enters here, now finds the cache, and runs
   * the core).  Either way the caller does nothing further. */
  (void)requester;
  arts_ooo_dispatch_or_defer_guid(guid, oo_type, packet_for_oo,
                                  (uint32_t)packet_size);
  return NULL;
}

/* ===== Home-side handlers ========================================== */

/* arts_handler_db_ownership_request lives in db_coherence_release.c (RC+LRC
 * only — LC has no LOCK_REQ / GRANT round). */

void arts_handler_db_snapshot_request(
    struct arts_remote_snapshot_request_packet_s *p) {
  unsigned int requester = p->header.rank;

  struct arts_ooo_args_db_snapshot_request_s oo_payload = {
      .requester = requester,
      .db_guid = p->db_guid,
      .edt_guid = p->edt_guid,
      .slot = p->slot,
  };

  struct arts_db_cache_s *cache = arts_coh_home_lookup_or_defer(
      p->db_guid, requester, OOO_DB_SNAPSHOT_REQUEST, &oo_payload,
      sizeof(oo_payload), COH_REPLY_DESTROY_NOTIFY);
  if (cache == NULL) {
    return;
  }
  arts_coh_model_snapshot_request_serve(cache, requester, p->edt_guid, p->slot);
}

void arts_handler_db_writeback(struct arts_remote_writeback_packet_s *p,
                               const void *data, uint64_t data_size) {
  unsigned int releaser = p->header.rank;

  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);

  /* defer-on-no-cache.  WRITEBACK can race ahead of DB_CREATE
   * on the home rank when the producer EDT releases very early; we must
   * (a) preserve the trailing data payload in the OoO entry and (b) ACK
   * the releaser immediately so its await_writeback_ack returns instead
   * of stalling.  When DB_CREATE finally arrives, fire_oo re-issues this
   * handler with the deferred buffer. */
  if (cache == NULL) {
    /* WRITEBACK raced ahead of DB_CREATE on the home rank (the producer
     * released before the home db_s installed — common once sender/receiver
     * threads reorder the wire).  Per the master plan this is a pure Cat-B OoO
     * defer: push the writeback (trailing data preserved in the payload) and
     * return WITHOUT acking.  The single WRITEBACK_ACK is sent exactly once
     * when the handler re-runs after DB_CREATE installs the cache (the
     * install's drain re-issues it).  Acking here too would post the
     * releaser's stack-local cv twice — the second post lands on a sem the
     * releaser already sem_destroy'd → glibc futex_fatal_error / SIGABRT.  If
     * the home db_s is never created (shutdown), await_writeback_ack's
     * shutdown re-check returns the worker, so it cannot stall forever. */
    uint32_t asz =
        (uint32_t)sizeof(struct arts_ooo_args_db_writeback_s) + data_size;
    char *buf = (char *)arts_malloc(asz);
    struct arts_ooo_args_db_writeback_s *a =
        (struct arts_ooo_args_db_writeback_s *)buf;
    a->releaser = releaser;
    a->db_guid = p->db_guid;
    a->version = p->version;
    a->cv = p->cv;
    a->flag = p->flag;
    a->data_size = data_size;
    if (data_size > 0 && data != NULL) {
      memcpy(buf + sizeof(*a), data, data_size);
    }
    arts_ooo_dispatch_or_defer_guid(p->db_guid, OOO_DB_WRITEBACK, buf, asz);
    arts_free(buf);
    return;
  }

  arts_coh_install_buffer(cache, p->version, data, data_size);
  /* cv==0 marks a fire-and-forget writeback (the INVALIDATE-driven
   * WB_AND_TRANSFER, sent from a network handler that cannot block on an ACK):
   * install the data but send no ACK — there is no semaphore waiting, and
   * posting to a null cv would be a wild write. */
  if (p->cv != 0) {
    arts_send_db_writeback_ack(releaser, p->db_guid, p->cv);
  }
  /* No snapshot drain here: home's RO acquires hit case 1 (resume self) and
   * never park.  Foreign ROs are served by GET_DATA, not by drain. */

  /* WB_AND_TRANSFER: ownership-chain relay.  RC advances the transfer chain;
   * LRC (no sync writeback) and LC (no exclusive owner) make this a no-op. */
  if (p->flag == ARTS_WB_AND_TRANSFER) {
    arts_coh_model_writeback_transfer(cache);
  }
}

/* arts_handler_db_ownership_return lives in db_coherence_release.c (RC+LRC
 * only — LC has no ownership chain). */

void arts_handler_db_destroy(struct arts_remote_destroy_packet_s *p) {
  /* Spec §4.11-4.12: home-side DESTROY_REQ.
   *
   * Symmetric with LOCK_REQ / GET_DATA / WRITEBACK: when the cache_s
   * has not yet been installed on home (DB_CREATE_COHERENT raced behind
   * the destroy), we must defer via the OoO list rather than treating
   * the NULL data slot as "already destroyed".  fire_oo on
   * DB_CREATE_COHERENT arrival re-issues this handler with the now-
   * installed cache.
   *
   * Fast path: cache is installed and reachable through item->data.  We
   * proceed with the legacy three-phase destroy (xchg data->NULL, drop
   * ooList, run the cache_s self-destroy protocol).  Order matters —
   *   [1] xchg item->data to NULL FIRST so new lookups can no longer
   *       enter (they observe NULL -> enqueue to ooList).
   *   [2] drop the OoO list (silent free; user-error path).
   *   [3] PIN/CXL DBs have no embedded cache -> step 1 + arts_db_free.
   *   [4] For coherent DBs, run the cache_s self-destroy protocol:
   *       fail_trigger_pending wakes parked waiters;
   *       try_finalize_destroy single-flights the buffer detach. */
  unsigned int requester = p->header.rank;
  struct arts_ooo_args_db_destroy_s oo_payload = {
      .requester = requester,
      .db_guid = p->db_guid,
  };
  struct arts_db_cache_s *cache = arts_coh_home_lookup_or_defer(
      p->db_guid, requester, OOO_DB_DESTROY, &oo_payload, sizeof(oo_payload),
      COH_REPLY_NONE);
  if (cache == NULL) {
    /* Either cache had destroy_state != NONE (already torn down -
     * destroy is idempotent, nothing else to do) or the request was
     * deferred via OoO (will replay after DB_CREATE_COHERENT). */
    return;
  }

  /* Cache is live — perform the three-phase destroy.  Re-derive the db_s
   * from the cache (cache is the first member of db_s) so the xchg sees the
   * same descriptor the lookup observed. */
  struct arts_db_s *db = arts_db_of_cache(cache);
  if (db == NULL) {
    return;
  }

  /* cb-model destroy (single-actor on home + cb deferred-free).  No separate
   * destroy gate: mark_delete's atomic_exchange of the slot cb IS the
   * single-flight, and the handler single-actor invariant serializes destroy
   * with snapshot/ownership/writeback on this cache.  Run
   * the waiter-wake + fan-out FIRST (the cache stays alive — only the install
   * ref is dropped, by mark_delete at the very end); the cb deleter
   * (arts_db_deleter → arts_db_free → cache_destructor) then frees the cache
   * once the last outstanding lookup ref drains.  A second DESTROY_REQ finds
   * the slot absent (lookup → NULL) and is a no-op. */

  /* Any OoO replays still queued on this slot are preserved across destroy:
   * destroy is invoked only after explicit event synchronization, so a
   * straggler that arrives later legitimately waits for a labeled-GUID
   * reinstall (route-table teardown frees anything never replayed). */

  /* Notify ranks with cached copies + queued ownership requesters so remote
   * sharers wake and observe DB_DESTROYED.  Single-actor keeps the roster
   * stable across this scan.  The roster source is model-specific (RC/LC walk
   * home->last_sent_version; LRC walks rw_holder + cached_ranks + pending_rw),
   * so the fan-out is delegated to the per-model hook. */
  unsigned int self = arts_global_rank_id;
  (void)db;
  arts_coh_model_destroy_fanout(cache, self);

  arts_coh_fail_trigger_pending(cache);
  /* mark_delete LAST: detach the slot cb + drop the install ref.  The cache
   * stayed alive through the fan-out above (single-actor + install ref); its
   * cb deleter (arts_db_deleter → cache_destructor) frees it once outstanding
   * lookup refs drain. */
  (void)arts_route_table_mark_delete(p->db_guid);
}

void arts_handler_db_create(struct arts_remote_db_create_coherent_packet_s *p) {
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
    if (arts_coh_buffer_peek(cache) == NULL && db_size > 0) {
      arts_coh_install_buffer(cache, 1, NULL, db_size);
    }
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
    if (!db->home_initialized) {
      arts_db_home_init(db, creator_rank, arts_global_rank_count);
      db->home_initialized = true;
    } else {
      arts_coh_model_db_create_set_holder(db, creator_rank);
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
    arts_coh_init_cache_s(&stub->cache, db_guid, db_size,
                          ARTS_COH_INIT_CREATOR_HOME, creator_rank);
    if (db_size > 0) {
      arts_coh_install_buffer(&stub->cache, /*new_version=*/1,
                              /*data_payload=*/NULL, db_size);
    }
  } else {
    arts_coh_init_cache_s(&stub->cache, db_guid, db_size,
                          ARTS_COH_INIT_HOME_RECV, creator_rank);
    /* LC: home is canonical with no creator writeback to wait for, so install a
     * zero-init buffer now (RC/LRC keep the lazy OCR install -- no-op). */
    arts_coh_model_create_home_buffer(&stub->cache, db_size);
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
    if (arts_coh_buffer_peek(cache) == NULL && db_size > 0) {
      arts_coh_install_buffer(cache, 1, NULL, db_size);
    }
    if (cache->db_size == 0) {
      cache->db_size = db_size;
    }
    if (!db->home_initialized) {
      arts_db_home_init(db, creator_rank, arts_global_rank_count);
      db->home_initialized = true;
    } else {
      arts_coh_model_db_create_set_holder(db, creator_rank);
    }
  }
  if (winner != NULL) {
    arts_shared_release(&winner_h);
  }
}

/* ===== Sharer-side response handlers =============================== */

/* The RC GRANT handler arts_handler_db_ownership_response lives in
 * db_coherence_rc.c; LRC's TRANSFER_OWNERSHIP overload lives in
 * db_coherence_lrc.c; LC has no ownership transfer (dispatcher fatals). */

void arts_handler_db_snapshot_response(
    struct arts_remote_snapshot_response_packet_s *p, const void *data,
    uint64_t data_size) {
  /* No acquire-time list registration: this 1:1 response resumes the parked
   * EDT (p->edt_guid, p->slot) directly.  Three cases over a monotonic version:
   *   1. p->version <= buf->version : nothing newer to install — resume self
   *      against the live buffer.
   *   2. data + p->version > buf->version : install + drain-all the reorder
   *      buffer + resume self.
   *   3. NO_DATA + p->version > buf->version : the with-data reply was
   *      reordered behind us — push self onto pending_snapshot (a future
   *      case-2 install drains us) + re-check (race recovery).
   * Shared verbatim by RC/LRC/LC (LC routes RW through here too). */
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    /* destroyed (route_item NULL-stored) — drop. */
    return;
  }
  arts_guid_t edt_guid = p->edt_guid;
  uint32_t slot = p->slot;

  arts_shared_ptr_t buf_h = arts_coh_acquire_buf(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  uint64_t buf_v = buf ? (uint64_t)buf->version : 0;
  if (buf != NULL) {
    arts_coh_release_buf(&buf_h);
  }

  if (p->version <= buf_v) {
    /* Case 1: nothing newer to install — resume self. */
    mark_edt_ready_by_guid(edt_guid, slot);
    return;
  }
  if (p->data_present) {
    /* Case 2: install (version-conditional publish inside install_buffer;
     * stale installs retreat) + drain-all + resume self. */
    arts_coh_install_buffer(cache, p->version, data, data_size);
    arts_coh_drain_pending_snapshot(cache);
    mark_edt_ready_by_guid(edt_guid, slot);
    return;
  }
  /* Case 3: NO_DATA arrived ahead of the with-data reply.  Park a
   * reorder-buffer node; a later case-2 install drains it. */
  struct arts_db_snapshot_waiter_s *w =
      (struct arts_db_snapshot_waiter_s *)arts_malloc(sizeof(*w));
  w->edt_guid = edt_guid;
  w->slot = slot;
  w->target_version = p->version;
  arts_lf_stack_push(&cache->pending_snapshot, &w->link);
  /* Race recovery: a concurrent case-2 install may have published the buffer
   * between our version read and the push.  If so, drain (our own node
   * included) so we don't park forever.  The atomic_exchange drain is the
   * single-actor primitive — a concurrent installer's drain and ours cannot
   * both claim the same node. */
  arts_shared_ptr_t rch = arts_coh_acquire_buf(cache);
  struct arts_db_buffer_s *rbuf =
      (struct arts_db_buffer_s *)arts_shared_get(rch);
  uint64_t rv = rbuf ? (uint64_t)rbuf->version : 0;
  if (rbuf != NULL) {
    arts_coh_release_buf(&rch);
  }
  if (rv >= p->version) {
    arts_coh_drain_pending_snapshot(cache);
  }
}

/* arts_handler_db_ownership_invalidate (INVALIDATE_NOTICE) lives per model:
 * db_coherence_rc.c (commutative signed counter) and db_coherence_lrc.c
 * (publish-target-then-withdraw).  LC never sends INVALIDATE (dispatcher
 * fatals). */

void arts_handler_db_writeback_ack(
    struct arts_remote_writeback_ack_packet_s *p) {
  /* RC/LC: pointer-identity wakeup of the releaser's stack-local sem_t.
   * LRC has no synchronous writeback (no-op).  Delegated to the per-model
   * hook. */
  arts_coh_model_writeback_ack(p->cv);
}

void arts_handler_db_cache_destroy(
    struct arts_remote_cache_destroy_packet_s *p) {
  struct arts_db_cache_s *cache = arts_coh_route_table_lookup_cache(p->db_guid);
  if (cache == NULL) {
    return; /* already torn down on this rank (cb-NULL = idempotent). */
  }
  /* cb-model destroy (single-actor): no destroy_state gate — a second
   * DESTROY_NOTIFY finds the slot absent (cache == NULL above).  fail_trigger
   * FIRST so the cache is alive while we wake waiters (mark_delete drops only
   * the install ref; the cb deleter frees once outstanding lookup refs
   * drain). */
  arts_coh_fail_trigger_pending(cache);
  (void)arts_route_table_mark_delete(p->db_guid);
}

/* The LRC REDIRECT_RO handler arts_handler_db_snapshot_redirect lives in
 * db_coherence_lrc.c (owner-side, REDIRECT only exists under LRC). */
