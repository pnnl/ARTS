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
void arts_coh_send_invalidate_notice(unsigned int owner_rank,
                                     arts_guid_t db_guid);
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

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_HANDLERS_H */
