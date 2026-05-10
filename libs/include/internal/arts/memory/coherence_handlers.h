/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence protocol wire-message handlers.
 *
 * The 11 wire messages defined in arts/transport/protocol.h each
 * land in their own handler here.  Handlers are dispatched from the
 * single per-rank network handler thread (one of S1's invariants),
 * so home-side state — home.pending_rw, home.last_sent_version,
 * home.rw_holder — is touched by exactly one writer.  Worker threads
 * concurrently read/write cache.* via the atomic primitives wired up
 * in the data structures (writer_count, buffer, pending_count,
 * destroy_state, the marked-list head/tail, the buffer pool).
 *
 * Drop discipline (every handler):
 *   1. arts_route_table_lookup_db_safe — bumps the route_table lock-field
 *      ref so the cache cannot be torn down underneath us.
 *   2. cache.destroy_state precheck.
 *   3. If lookup returned NULL, or destroy_state != NONE: send the
 *      appropriate wake-up reply (DESTROY_NOTIFY for LOCK_REQ /
 *      GET_DATA, WRITEBACK_ACK for WRITEBACK), drop the route_table
 *      ref, return.  (Silent drop would deadlock the requester.)
 *   4. Body.
 *   5. arts_route_table_release.
 *
 * Handlers receive the full wire packet; size and rank are accessed
 * via the embedded arts_remote_packet_s header. */

#ifndef ARTS_MEMORY_COHERENCE_HANDLERS_H
#define ARTS_MEMORY_COHERENCE_HANDLERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arts/transport/protocol.h"

/* Forward declaration for LRC handler parameters. */
struct arts_db_cache_s;

/* ===== Home-side handlers ============================================ */

void arts_coh_handle_lock_req(struct arts_remote_lock_req_packet_s *p);
void arts_coh_handle_get_data(struct arts_remote_get_data_packet_s *p);
void arts_coh_handle_writeback(struct arts_remote_writeback_packet_s *p,
                               const void *data, uint64_t data_size);
void arts_coh_handle_release_ownership(
    struct arts_remote_release_ownership_packet_s *p);
void arts_coh_handle_destroy_req(struct arts_remote_destroy_req_packet_s *p);
void arts_coh_handle_db_create_coherent(
    struct arts_remote_db_create_coherent_packet_s *p);

/* ===== Sharer-side (response) handlers =============================== */

void arts_coh_handle_grant(struct arts_remote_grant_packet_s *p,
                           const void *data, uint64_t data_size);
void arts_coh_handle_data_response(struct arts_remote_data_response_packet_s *p,
                                   const void *data, uint64_t data_size);
void arts_coh_handle_invalidate_notice(
    struct arts_remote_invalidate_notice_packet_s *p);
void arts_coh_handle_writeback_ack(
    struct arts_remote_writeback_ack_packet_s *p);
void arts_coh_handle_destroy_notify(
    struct arts_remote_destroy_notify_packet_s *p);

/* ===== LRC-only handlers ============================================== */

#ifdef ARTS_MEMORY_MODEL_LRC
/* Handles REDIRECT_RO at the current owner: look up the local cache and
 * send DATA_RESPONSE directly to the requester. */
void arts_coh_handle_redirect_ro(
    struct arts_remote_redirect_ro_packet_s *p);

/* Drain all deferred GET_DATA requests from the home-side RO forward
 * queue.  Called by the INSTALL_ACK handler once the new owner's
 * writer_count sentinel is published and invalidate_in_flight is
 * cleared.  No other consumer runs concurrently. */
void arts_coh_drain_pending_ro_forwards(struct arts_db_cache_s *cache);

/* Handles TRANSFER_OWNERSHIP at new owner C.  payload points to the
 * full contiguous wire buffer (header + map + data); size is total bytes. */
void arts_coh_handle_transfer_ownership(void *payload, size_t size);

/* Handles INSTALL_ACK at home A: records the new rw_holder and drains
 * pending_ro_forwards; then starts the next transfer round or releases
 * the invalidate_in_flight baton. */
void arts_coh_handle_install_ack(
    struct arts_remote_install_ack_packet_s *p);
#endif /* ARTS_MEMORY_MODEL_LRC */

/* ===== Sender helpers ================================================ */

/* Send a coherence wire packet of the given type with optional
 * trailing payload (data + data_size).  data == NULL ⇒ no payload.
 * Used by handlers that emit replies and by acquire/release in B4/B5. */
void arts_coh_send_lock_req(unsigned int home_rank, arts_guid_t db_guid);
void arts_coh_send_grant(unsigned int requester_rank, arts_guid_t db_guid,
                         uint64_t version, bool has_next, const void *data,
                         uint64_t data_size);
void arts_coh_send_writeback(unsigned int home_rank, arts_guid_t db_guid,
                             uint64_t version, uint64_t seq,
                             arts_writeback_flag_t flag, const void *data,
                             uint64_t data_size);
void arts_coh_send_writeback_ack(unsigned int releaser_rank,
                                 arts_guid_t db_guid, uint64_t seq);
/* new_owner_rank: rank that home has selected as next owner.  RC builds pass
 * 0 (receiver ignores it); LRC builds embed it in the packet so the holder
 * knows where to ship TRANSFER_OWNERSHIP without a home round-trip. */
void arts_coh_send_invalidate_notice(unsigned int owner_rank,
                                     arts_guid_t db_guid,
                                     unsigned int new_owner_rank);
void arts_coh_send_release_ownership(unsigned int home_rank,
                                     arts_guid_t db_guid);
void arts_coh_send_get_data(unsigned int home_rank, arts_guid_t db_guid,
                            void *waiter_addr);
void arts_coh_send_data_response(unsigned int requester_rank,
                                 arts_guid_t db_guid, uint64_t version,
                                 void *waiter_addr, const void *data,
                                 uint64_t data_size);
void arts_coh_send_db_create_coherent(unsigned int home_rank,
                                      arts_guid_t db_guid, uint64_t db_size,
                                      uint16_t flags, uint16_t db_type);
void arts_coh_send_destroy_req(unsigned int home_rank, arts_guid_t db_guid);
void arts_coh_send_destroy_notify(unsigned int sharer_rank,
                                  arts_guid_t db_guid);

#ifdef ARTS_MEMORY_MODEL_LRC
/* Send REDIRECT_RO from home to the current owner, asking the owner to
 * serve DATA_RESPONSE or a no-data response directly to requester_rank. */
void arts_coh_send_redirect_ro(unsigned int owner_rank, arts_guid_t db_guid,
                               unsigned int requester_rank, void *waiter_addr);

/* Send TRANSFER_OWNERSHIP from current owner B to new owner C.
 * map_buf/map_size: serialized last_sent_version map (may be NULL/0 for
 * sentinel DBs).  data/data_size: current buffer payload (may be NULL/0). */
void arts_coh_send_transfer_ownership(unsigned int new_owner_rank,
                                      arts_guid_t db_guid, uint64_t version,
                                      const void *map_buf, size_t map_size,
                                      const void *data, size_t data_size);

/* Send INSTALL_ACK from new owner C back to home A once the ownership
 * transfer is complete. */
void arts_coh_send_install_ack(unsigned int home_rank, arts_guid_t db_guid,
                               uint64_t version);

/* Ship TRANSFER_OWNERSHIP to cache->incoming_new_owner.  Called either
 * from the INVALIDATE_NOTICE handler (when writer_count reaches 0
 * immediately) or from release_rw (when transfer_pending was set). */
void arts_coh_lrc_ship_transfer(struct arts_db_cache_s *cache);

/* Kick a new INVALIDATE_NOTICE round: read rw_holder, send notice to
 * holder carrying new_owner as the TRANSFER_OWNERSHIP target. */
void arts_coh_lrc_start_invalidate_round(struct arts_db_cache_s *cache,
                                          unsigned int new_owner);
#endif /* ARTS_MEMORY_MODEL_LRC */

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_HANDLERS_H */
