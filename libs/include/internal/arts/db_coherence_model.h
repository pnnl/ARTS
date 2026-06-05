/* SPDX-License-Identifier: Apache-2.0
 *
 * Per-consistency-model coherence hooks.
 *
 * The shared coherence translation unit (db_coherence.c) declares these hooks;
 * exactly one model translation unit (db_coherence_rc.c / db_coherence_lrc.c /
 * db_coherence_lc.c) defines them. CMake links the single TU selected by
 * ARTS_MEMORY_MODEL. Because only one model is ever compiled, selection happens
 * at link time with plain extern symbols -- no function-pointer vtable, no
 * runtime indirection. No consistency-model preprocessor logic lives below this
 * seam; it survives only on the wire side (transport/protocol.h message enum
 * and transport/dispatcher.c type->handler routing).
 */
#ifndef ARTS_DB_COHERENCE_MODEL_H
#define ARTS_DB_COHERENCE_MODEL_H

#include <semaphore.h>

#include "arts/db_coherence.h"
#include "arts/db_coherence_buffer.h" /* arts_shared_ptr_t, acquire/release_buf */
#include "arts/db_coherence_handlers.h" /* arts_send_db_writeback, lrc ownership send */
#include "arts/gas/route_table.h" /* ooo_kind_t (home_lookup_or_defer) */
#include "arts/utils/atomics.h"   /* arts_atomic_* (used by model TUs) */

/* ===== Shared coherence services (defined in db_coherence.c) =================
 * Model TUs call back into these model-agnostic helpers. */

/* Block on a stack-local semaphore until the matching WRITEBACK_ACK posts it
 * (pointer identity); returns early if teardown begins.  Used by the RC and LC
 * release-tail hooks. */
void await_writeback_ack(sem_t *cv);

/* Home-side local ownership hand-off to the next queued waiter (or sentinel
 * restore when none).  Invoked by the RC release-tail hook. */
void arts_coh_local_transfer_now(struct arts_db_cache_s *cache);

/* Take the EDT's strong buffer ref and return buf->data (NULL when no buffer is
 * installed).  Used by the per-model acquire dispatch hooks. */
void *arts_coh_acquire_local(struct arts_db_cache_s *cache);

/* Case 7: fire SNAPSHOT_REQUEST (edt_guid + slot) to home and PARK.  Shared by
 * the RC/LRC and LC acquire dispatch hooks. */
arts_db_acquire_result_t
arts_coh_acquire_remote_ro(struct arts_db_cache_s *cache, arts_guid_t edt_guid,
                           unsigned int slot);

/* Wake a parked EDT's dep slot (re-derives dep->ptr from the installed buffer).
 * Defined in db_coherence_handlers.c; used by the drain paths. */
void mark_edt_ready_by_guid(arts_guid_t edt_guid, unsigned int slot);

/* Reply kind selector for home_lookup_or_defer.  Mirrors the design spec's
 * REPLY_DESTROY_NOTIFY / REPLY_WB_ACK / REPLY_NONE. */
typedef enum {
  COH_REPLY_NONE = 0,
  COH_REPLY_DESTROY_NOTIFY,
  COH_REPLY_WB_ACK,
} coh_reply_kind_t;

/* Look up cache by guid; if not yet installed, defer the wire message via the
 * OoO list (re-issues once DB_CREATE arrives) or reply per reply_kind.  Returns
 * the live cache or NULL when deferred / replied.  Defined in
 * db_coherence_handlers.c; called by home-side handlers in the shared TU and in
 * db_coherence_release.c. */
struct arts_db_cache_s *
arts_coh_home_lookup_or_defer(arts_guid_t guid, unsigned int requester,
                              ooo_kind_t oo_type, void *packet_for_oo,
                              size_t packet_size, coh_reply_kind_t reply_kind);

/* ===== Per-model hooks (defined by exactly one db_coherence_<model>.c)
 * =======*/

/* arts_handler_db_acquire 8-case body: RC/LRC run the single-owner LOCK_REQ /
 * GRANT path (db_coherence_release.c), LC runs the unified home-canonical path
 * (db_coherence_lc.c).  Caller computed mode / is_home / is_owner. */
arts_db_acquire_result_t arts_coh_model_acquire_dispatch(
    struct arts_db_cache_s *cache, arts_edt_dep_t *dep, arts_guid_t edt_guid,
    unsigned int slot, arts_db_access_mode_t mode, bool is_home, bool is_owner);

/* RO-path predicate distinguishing RC (home always canonical:
 * is_home||is_owner) from LRC (only the current owner holds data: is_owner).
 * Defined in db_coherence_rc.c / db_coherence_lrc.c; called by the RC/LRC
 * acquire dispatch in db_coherence_release.c. */
bool arts_coh_model_ro_has_local_data(bool is_home, bool is_owner);

/* RC/LRC GRANT drain: pop pending_rw FIFO, bump writer_count per waiter, wake
 * each parked EDT.  Defined in db_coherence_release.c; called from the GRANT
 * handler in db_coherence_handlers.c. */
void arts_coh_drain_pending_rw_after_grant(struct arts_db_cache_s *cache,
                                           uint64_t version, bool has_next);

/* Home-side ownership-transfer trigger: home advances the chain locally;
 * non-home ships its buffer to home (WB_AND_TRANSFER) or, for a sentinel DB, a
 * data-less ownership_return.  RC/LRC only — defined in db_coherence_release.c,
 * called from the ownership-invalidate handler. */
void arts_coh_invalidate_transfer(struct arts_db_cache_s *cache);

/* Destroy/fail fan-out of the RC/LRC pending_rw queue (wake each waiter with a
 * NULL ptr).  Defined in db_coherence_release.c (drains pending_rw) and
 * db_coherence_lc.c (no-op — LC has no pending_rw). */
void arts_coh_model_fail_trigger_pending_rw(struct arts_db_cache_s *cache);

/* Model-specific cache_s field init: RC/LRC initialize the pending_rw Vyukov
 * MPSC queue; LRC additionally arms its owner-side dedup map + transfer
 * sentinel.  LC is a no-op.  Defined per model TU. */
void arts_coh_model_init_cache_s(struct arts_db_cache_s *c);

/* Model-specific cache_s teardown: RC/LRC destroy the pending_rw queue; LC is a
 * no-op.  Defined per model TU. */
void arts_coh_model_cache_destructor(struct arts_db_cache_s *c);

/* ===== Per-model hooks (defined by exactly one db_coherence_<model>.c)
 * =======*/

/* arts_coh_release_rw seam 1: runs after the version bump, BEFORE writer_count
 * is decremented.  A model may drop its buffer ref early (LRC closes the
 * teardown window here); RC/LC are no-ops.  The model may set *buf = NULL to
 * signal the ref was already released. */
void arts_coh_model_release_rw_pre_decrement(struct arts_db_cache_s *cache,
                                             arts_shared_ptr_t *buf_h,
                                             struct arts_db_buffer_s **buf);

/* arts_coh_release_rw seam 2: runs after the writer_count decrement with the
 * post-decrement value `rest`.  Owns the transfer/writeback decision and the
 * final release of the release-scoped buffer ref. */
void arts_coh_model_release_rw_tail(struct arts_db_cache_s *cache,
                                    arts_shared_ptr_t *buf_h,
                                    struct arts_db_buffer_s *buf,
                                    uint64_t new_version, unsigned int rest,
                                    bool is_home);

/* ===== Per-model wire-handler body hooks ============================== */

/* arts_handler_db_snapshot_request (GET_DATA) body, post cache-lookup.  RC/LC
 * serve from home's canonical buffer (last_sent_version dedup); LRC records the
 * RO sharer + REDIRECTs to the current owner.  Defined per model TU. */
void arts_coh_model_snapshot_request_serve(struct arts_db_cache_s *cache,
                                           unsigned int requester,
                                           arts_guid_t edt_guid, uint32_t slot);

/* arts_handler_db_writeback WB_AND_TRANSFER tail.  RC advances the ownership
 * chain; LRC (no sync writeback) + LC (no exclusive owner) are no-ops.  Invoked
 * only when p->flag == ARTS_WB_AND_TRANSFER.  Defined per model TU. */
void arts_coh_model_writeback_transfer(struct arts_db_cache_s *cache);

/* arts_handler_db_writeback_ack body.  RC/LC post the releaser's stack-local
 * sem_t by pointer identity; LRC has no synchronous writeback (no-op).  Defined
 * per model TU. */
void arts_coh_model_writeback_ack(uint64_t cv);

/* arts_handler_db_destroy roster fan-out: send DESTROY_NOTIFY to every rank
 * that holds a cached copy / queued ownership request, then drain pending_rw.
 * `self` is arts_global_rank_id.  RC/LC walk last_sent_version; LRC walks
 * rw_holder + cached_ranks + pending_rw.  Defined per model TU. */
void arts_coh_model_destroy_fanout(struct arts_db_cache_s *cache,
                                   unsigned int self);

/* arts_handler_db_create: publish creator_rank as the home rw_holder when the
 * home directory was already initialized (coalesce path).  RC/LRC store
 * rw_holder; LC has no exclusive owner (no-op).  Defined per model TU. */
void arts_coh_model_db_create_set_holder(struct arts_db_s *db,
                                         unsigned int creator_rank);

/* arts_handler_db_create: install the home buffer at cross-rank-create time
 * (HOME_RECV).  LC's home is always canonical and there is no creator writeback
 * to wait for, so it installs a version-1 zero buffer immediately — otherwise
 * the first home RW acquire (acquire_local) returns a NULL payload.  RC/LRC
 * keep the lazy OCR install (the creator's release_rw WRITEBACK / GRANT path
 * publishes the buffer), so their hook is a no-op.  Defined per model TU. */
void arts_coh_model_create_home_buffer(struct arts_db_cache_s *cache,
                                       uint64_t db_size);

/* ===== Ownership-round seams (RC+LRC; defined in rc.c / lrc.c) ========
 * Called from the shared ownership_request / ownership_return handlers in
 * db_coherence_release.c.  LC compiles neither (no exclusive ownership). */

/* arts_handler_db_ownership_request tail after the LOCK_REQ is queued and the
 * invalidate_in_flight baton has been claimed (CAS 0->1 won).  RC INVALIDATEs
 * the current rw_holder; LRC pops the FIFO transfer target, publishes
 * pending_install_owner, and starts the invalidate round. */
void arts_coh_model_start_ownership_round(struct arts_db_cache_s *cache,
                                          struct arts_db_s *db,
                                          unsigned int requester);

/* arts_handler_db_ownership_return (RELEASE_OWNERSHIP) body after the cache is
 * resolved.  RC advances the transfer chain; LRC never receives this message
 * (no-op). */
void arts_coh_model_ownership_return(struct arts_db_cache_s *cache);

#endif /* ARTS_DB_COHERENCE_MODEL_H */
