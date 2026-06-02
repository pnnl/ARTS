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

void arts_handler_db_ownership_request(
    struct arts_remote_ownership_request_packet_s *p);
void arts_handler_db_snapshot_request(
    struct arts_remote_snapshot_request_packet_s *p);
void arts_handler_db_writeback(struct arts_remote_writeback_packet_s *p,
                               const void *data, uint64_t data_size);
void arts_handler_db_ownership_return(
    struct arts_remote_ownership_return_packet_s *p);
void arts_handler_db_destroy(struct arts_remote_destroy_packet_s *p);
void arts_handler_db_create(
    struct arts_remote_db_create_coherent_packet_s *p);

/* ===== Sharer-side (response) handlers =============================== */

#ifdef ARTS_MEMORY_MODEL_LRC
/* LRC TRANSFER_OWNERSHIP at new owner C: payload = full contiguous wire buffer
 * (header + map + data); size is total bytes. */
void arts_handler_db_ownership_response(void *payload, size_t size);
#else
void arts_handler_db_ownership_response(
    struct arts_remote_ownership_response_packet_s *p, const void *data,
    uint64_t data_size);
#endif
void arts_handler_db_snapshot_response(
    struct arts_remote_snapshot_response_packet_s *p, const void *data,
    uint64_t data_size);
void arts_handler_db_ownership_invalidate(
    struct arts_remote_ownership_invalidate_packet_s *p);
void arts_handler_db_writeback_ack(
    struct arts_remote_writeback_ack_packet_s *p);
void arts_handler_db_cache_destroy(
    struct arts_remote_cache_destroy_packet_s *p);

/* ===== LRC-only handlers ============================================== */

#ifdef ARTS_MEMORY_MODEL_LRC
/* Handles REDIRECT_RO at the current owner: look up the local cache and
 * send DATA_RESPONSE directly to the requester. */
void arts_handler_db_snapshot_redirect(
    struct arts_remote_snapshot_redirect_packet_s *p);

/* Handles INSTALL_ACK at home A: records the new rw_holder, then starts
 * the next transfer round or releases the invalidate_in_flight baton. */
void arts_handler_db_ownership_response_ack(
    struct arts_remote_install_ack_packet_s *p);
#endif /* ARTS_MEMORY_MODEL_LRC */

/* ===== Sender helpers ================================================ */

/* Send a coherence wire packet of the given type with optional
 * trailing payload (data + data_size).  data == NULL ⇒ no payload.
 * Used by handlers that emit replies and by acquire/release in B4/B5. */
void arts_send_db_ownership_request(unsigned int home_rank,
                                    arts_guid_t db_guid);
#ifdef ARTS_MEMORY_MODEL_LRC
/* LRC OWNERSHIP_RESPONSE = TRANSFER_OWNERSHIP: carries the serialized
 * last_sent_version map + buffer payload. */
void arts_send_db_ownership_response(unsigned int new_owner_rank,
                                     arts_guid_t db_guid, uint64_t version,
                                     const void *map_buf, size_t map_size,
                                     const void *data, size_t data_size);
#else
void arts_send_db_ownership_response(unsigned int requester_rank,
                                     arts_guid_t db_guid, uint64_t version,
                                     bool has_next, const void *data,
                                     uint64_t data_size);
#endif
/* cv: address of the releaser's stack-local sem_t (as uint64_t), forwarded
 * verbatim to the home and echoed back in the ACK for pointer-identity wakeup.
 */
void arts_send_db_writeback(unsigned int home_rank, arts_guid_t db_guid,
                            uint64_t version, uint64_t cv,
                            arts_writeback_flag_t flag, const void *data,
                            uint64_t data_size);
void arts_send_db_writeback_ack(unsigned int releaser_rank, arts_guid_t db_guid,
                                uint64_t cv);
/* new_owner_rank: rank that home has selected as next owner.  RC builds pass
 * 0 (receiver ignores it); LRC builds embed it in the packet so the holder
 * knows where to ship TRANSFER_OWNERSHIP without a home round-trip. */
void arts_send_db_ownership_invalidate(unsigned int owner_rank,
                                       arts_guid_t db_guid,
                                       unsigned int new_owner_rank);
void arts_send_db_ownership_return(unsigned int home_rank, arts_guid_t db_guid);
void arts_send_db_snapshot_request(unsigned int home_rank, arts_guid_t db_guid,
                                   arts_guid_t edt_guid, uint32_t slot);
void arts_send_db_snapshot_response(unsigned int requester_rank,
                                    arts_guid_t db_guid, uint64_t version,
                                    arts_guid_t edt_guid, uint32_t slot,
                                    const void *data, uint64_t data_size);
void arts_send_db_create_coherent(unsigned int home_rank, arts_guid_t db_guid,
                                  uint64_t db_size, uint16_t flags,
                                  uint16_t db_type);
void arts_send_db_destroy(unsigned int home_rank, arts_guid_t db_guid);
void arts_send_db_cache_destroy(unsigned int sharer_rank, arts_guid_t db_guid);

#ifdef ARTS_MEMORY_MODEL_LRC
/* Send REDIRECT_RO from home to the current owner, asking the owner to
 * serve DATA_RESPONSE or a no-data response directly to requester_rank. */
void arts_send_db_snapshot_redirect(unsigned int owner_rank,
                                    arts_guid_t db_guid,
                                    unsigned int requester_rank,
                                    arts_guid_t edt_guid, uint32_t slot);

/* Send INSTALL_ACK from new owner C back to home A once the ownership
 * transfer is complete. */
void arts_send_db_ownership_response_ack(unsigned int home_rank,
                                         arts_guid_t db_guid, uint64_t version);

/* Send the LRC OWNERSHIP_RESPONSE (TRANSFER_OWNERSHIP) to
 * cache->incoming_new_owner: serialize last_sent_version + buffer and fire.
 * Called inline from the INVALIDATE_NOTICE handler and release_rw rest==0. */
void arts_coh_lrc_send_ownership_response(struct arts_db_cache_s *cache);

/* Kick a new INVALIDATE_NOTICE round: read rw_holder, send notice to
 * holder carrying new_owner as the TRANSFER_OWNERSHIP target. */
void arts_coh_lrc_start_invalidate_round(struct arts_db_cache_s *cache,
                                         unsigned int new_owner);
#endif /* ARTS_MEMORY_MODEL_LRC */

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_HANDLERS_H */
