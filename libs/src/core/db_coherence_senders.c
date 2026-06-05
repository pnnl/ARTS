/* SPDX-License-Identifier: Apache-2.0
 *
 * Coherence protocol wire-message SENDERS.
 *
 * Each arts_send_db_* helper fills a wire packet (header + body) and either
 * dispatches the matching handler inline (when the destination is the local
 * rank — arts_remote_send_request_async drops self-sends, and the RC uses
 * uniform "send to home" semantics including home == self) or enqueues the
 * packet on the outbox for the transport layer.
 *
 * The receive-side bodies (arts_handler_db_*) and the home-side dedup /
 * transfer helpers live in coherence_handlers.c.
 *
 * Single-node note: arts_remote_send_request_async drops messages whose
 * destination is the local rank (self_send_check rejects).  The RC uses
 * uniform "send to home" semantics including home == self, so we dispatch
 * handlers directly when rank == self instead of going over the network.
 */

#include "arts/db_coherence_handlers.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "arts/system/threads.h"
#include "arts/transport/outbox.h" /* outbound send helpers */
#include "arts/utils/malloc.h"

/* ===== Sender helpers ============================================== */

/* arts_send_db_ownership_request / _return / _invalidate and the
 * OWNERSHIP_RESPONSE sender live in the model TUs (RC+LRC only): the request /
 * return / invalidate senders in db_coherence_release.c, the OWNERSHIP_RESPONSE
 * sender in db_coherence_rc.c (GRANT) and db_coherence_lrc.c
 * (TRANSFER_OWNERSHIP).  LC has no exclusive-ownership wire messages. */

void arts_send_db_writeback(unsigned int home_rank, arts_guid_t db_guid,
                            uint64_t version, uint64_t cv,
                            arts_writeback_flag_t flag, const void *data,
                            uint64_t data_size) {
  struct arts_remote_writeback_packet_s p;
  uint64_t total = sizeof(p) + data_size;
  arts_fill_packet_header(&p.header, total, MSG_DB_WRITEBACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  p.cv = cv;
  p.flag = (uint8_t)flag;
  memset(p.pad, 0, sizeof(p.pad));
  if (home_rank == arts_global_rank_id) {
    arts_handler_db_writeback(&p, data, data_size);
    return;
  }
  /* Sentinel DBs (db_size==0) still need WRITEBACK for ownership
   * transfer / R3 ordering, but the payload-async path errors on
   * zero-size payload — route via the no-payload async send. */
  if (data == NULL || data_size == 0) {
    arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
    return;
  }
  arts_remote_send_request_payload_async((int)home_rank, (char *)&p, sizeof(p),
                                         (char *)data, data_size);
}

void arts_send_db_writeback_ack(unsigned int releaser_rank, arts_guid_t db_guid,
                                uint64_t cv) {
  struct arts_remote_writeback_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_WRITEBACK_ACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.cv = cv;
  if (releaser_rank == arts_global_rank_id) {
    arts_handler_db_writeback_ack(&p);
    return;
  }
  arts_remote_send_request_async((int)releaser_rank, (char *)&p, sizeof(p));
}

void arts_send_db_snapshot_request(unsigned int home_rank, arts_guid_t db_guid,
                                   arts_guid_t edt_guid, uint32_t slot) {
  struct arts_remote_snapshot_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_SNAPSHOT_REQUEST);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.edt_guid = edt_guid;
  p.slot = slot;
  memset(p.pad, 0, sizeof(p.pad));
  if (home_rank == arts_global_rank_id) {
    arts_handler_db_snapshot_request(&p);
    return;
  }
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_snapshot_response(unsigned int requester_rank,
                                    arts_guid_t db_guid, uint64_t version,
                                    arts_guid_t edt_guid, uint32_t slot,
                                    const void *data, uint64_t data_size) {
  struct arts_remote_snapshot_response_packet_s p;
  uint64_t total = sizeof(p) + (data ? data_size : 0);
  arts_fill_packet_header(&p.header, total, MSG_DB_SNAPSHOT_RESPONSE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  p.edt_guid = edt_guid;
  p.slot = slot;
  p.data_present = data ? 1u : 0u;
  if (requester_rank == arts_global_rank_id) {
    arts_handler_db_snapshot_response(&p, data, data ? data_size : 0);
    return;
  }
  if (data && data_size > 0) {
    arts_remote_send_request_payload_async((int)requester_rank, (char *)&p,
                                           sizeof(p), (char *)data, data_size);
  } else {
    arts_remote_send_request_async((int)requester_rank, (char *)&p, sizeof(p));
  }
}

void arts_send_db_create_coherent(unsigned int home_rank, arts_guid_t db_guid,
                                  uint64_t db_size, uint16_t flags,
                                  uint16_t db_type) {
  struct arts_remote_db_create_coherent_packet_s p;
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
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_destroy(unsigned int home_rank, arts_guid_t db_guid) {
  struct arts_remote_destroy_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_DESTROY);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    arts_handler_db_destroy(&p);
    return;
  }
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_cache_destroy(unsigned int sharer_rank, arts_guid_t db_guid) {
  struct arts_remote_cache_destroy_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_CACHE_DESTROY);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (sharer_rank == arts_global_rank_id) {
    arts_handler_db_cache_destroy(&p);
    return;
  }
  arts_remote_send_request_async((int)sharer_rank, (char *)&p, sizeof(p));
}

/* The LRC-only senders (INSTALL_ACK, REDIRECT_RO) live in db_coherence_lrc.c
 * alongside their handlers; the RC/LRC OWNERSHIP_RESPONSE senders live in
 * db_coherence_rc.c / db_coherence_lrc.c. */
