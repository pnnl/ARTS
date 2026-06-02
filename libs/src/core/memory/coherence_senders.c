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

#include "arts/memory/coherence_handlers.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "arts/system/threads.h"
#include "arts/transport/outbox.h" /* outbound send helpers */
#include "arts/utils/malloc.h"

/* ===== Sender helpers ============================================== */

void arts_send_db_ownership_request(unsigned int home_rank,
                                    arts_guid_t db_guid) {
  struct arts_remote_ownership_request_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_REQUEST);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    arts_handler_db_ownership_request(&p);
    return;
  }
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
}

#ifndef ARTS_MEMORY_MODEL_LRC
void arts_send_db_ownership_response(unsigned int requester_rank,
                                     arts_guid_t db_guid, uint64_t version,
                                     bool has_next, const void *data,
                                     uint64_t data_size) {
  struct arts_remote_ownership_response_packet_s p;
  uint64_t total = sizeof(p) + (data ? data_size : 0);
  arts_fill_packet_header(&p.header, total, MSG_DB_OWNERSHIP_RESPONSE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  p.has_next = has_next ? 1u : 0u;
  p.data_present = data ? 1u : 0u;
  memset(p.pad, 0, sizeof(p.pad));
  if (requester_rank == arts_global_rank_id) {
    arts_handler_db_ownership_response(&p, data, data ? data_size : 0);
    return;
  }
  if (data && data_size > 0) {
    arts_remote_send_request_payload_async((int)requester_rank, (char *)&p,
                                           sizeof(p), (char *)data, data_size);
  } else {
    arts_remote_send_request_async((int)requester_rank, (char *)&p, sizeof(p));
  }
}
#else  /* ARTS_MEMORY_MODEL_LRC: OWNERSHIP_RESPONSE = TRANSFER_OWNERSHIP,      \
        * carries  serialized last_sent_version map + buffer payload. */
void arts_send_db_ownership_response(unsigned int new_owner_rank,
                                     arts_guid_t db_guid, uint64_t version,
                                     const void *map_buf, size_t map_size,
                                     const void *data, size_t data_size) {
  struct arts_remote_ownership_response_packet_s hdr;
  uint32_t entry_count = (map_buf != NULL && map_size >= sizeof(uint32_t) * 2)
                             ? ((const uint32_t *)map_buf)[0]
                             : 0u;
  uint64_t total =
      (uint64_t)sizeof(hdr) + (uint64_t)map_size + (uint64_t)data_size;
  arts_fill_packet_header(&hdr.header, total, MSG_DB_OWNERSHIP_RESPONSE);
  hdr.header.rank = arts_global_rank_id;
  hdr.db_guid = db_guid;
  hdr.version = version;
  hdr.map_entry_count = entry_count;
  hdr.pad = 0;
  if (new_owner_rank == arts_global_rank_id) {
    /* Self-transfer: construct a contiguous buffer and call handler inline. */
    void *buf = arts_malloc((size_t)total);
    memcpy(buf, &hdr, sizeof(hdr));
    if (map_buf != NULL && map_size > 0) {
      memcpy((char *)buf + sizeof(hdr), map_buf, map_size);
    }
    if (data != NULL && data_size > 0) {
      memcpy((char *)buf + sizeof(hdr) + map_size, data, data_size);
    }
    arts_handler_db_ownership_response(buf, (size_t)total);
    arts_free(buf);
    return;
  }
  if ((map_size > 0 || data_size > 0) && (map_buf != NULL || data != NULL)) {
    size_t payload_size = map_size + data_size;
    void *payload = arts_malloc(payload_size);
    if (map_buf != NULL && map_size > 0) {
      memcpy(payload, map_buf, map_size);
    }
    if (data != NULL && data_size > 0) {
      memcpy((char *)payload + map_size, data, data_size);
    }
    arts_remote_send_request_payload_async_free(
        (int)new_owner_rank, (char *)&hdr, sizeof(hdr), (char *)payload,
        /*offset=*/0, (uint64_t)payload_size, arts_free);
  } else {
    arts_remote_send_request_async((int)new_owner_rank, (char *)&hdr,
                                   sizeof(hdr));
  }
}
#endif /* ARTS_MEMORY_MODEL_LRC */

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

void arts_send_db_ownership_invalidate(unsigned int owner_rank,
                                       arts_guid_t db_guid,
                                       unsigned int new_owner_rank) {
  struct arts_remote_ownership_invalidate_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_INVALIDATE);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.new_owner_rank = new_owner_rank;
  memset(p.pad, 0, sizeof(p.pad));
  if (owner_rank == arts_global_rank_id) {
    arts_handler_db_ownership_invalidate(&p);
    return;
  }
  arts_remote_send_request_async((int)owner_rank, (char *)&p, sizeof(p));
}

void arts_send_db_ownership_return(unsigned int home_rank,
                                   arts_guid_t db_guid) {
  struct arts_remote_ownership_return_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_RETURN);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  if (home_rank == arts_global_rank_id) {
    arts_handler_db_ownership_return(&p);
    return;
  }
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
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

#ifdef ARTS_MEMORY_MODEL_LRC

/* ===== LRC sender helpers (INSTALL_ACK, REDIRECT_RO) ================== */

void arts_send_db_ownership_response_ack(unsigned int home_rank,
                                         arts_guid_t db_guid,
                                         uint64_t version) {
  struct arts_remote_install_ack_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_OWNERSHIP_RESPONSE_ACK);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.version = version;
  if (home_rank == arts_global_rank_id) {
    arts_handler_db_ownership_response_ack(&p);
    return;
  }
  arts_remote_send_request_async((int)home_rank, (char *)&p, sizeof(p));
}

void arts_send_db_snapshot_redirect(unsigned int owner_rank,
                                    arts_guid_t db_guid,
                                    unsigned int requester_rank,
                                    arts_guid_t edt_guid, uint32_t slot) {
  struct arts_remote_snapshot_redirect_packet_s p;
  arts_fill_packet_header(&p.header, sizeof(p), MSG_DB_SNAPSHOT_REDIRECT);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.edt_guid = edt_guid;
  p.requester_rank = requester_rank;
  p.slot = slot;
  if (owner_rank == arts_global_rank_id) {
    arts_handler_db_snapshot_redirect(&p);
    return;
  }
  arts_remote_send_request_async((int)owner_rank, (char *)&p, sizeof(p));
}
#endif /* ARTS_MEMORY_MODEL_LRC */
