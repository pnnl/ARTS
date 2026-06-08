/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/
#ifndef ARTS_TRANSPORT_PROTOCOL_H
#define ARTS_TRANSPORT_PROTOCOL_H
#ifdef __cplusplus
extern "C" {
#endif
#include "arts/runtime_types.h"
#define SEQUENCENUMBERS 1

enum arts_msg_type {
  MSG_SHUTDOWN,
  MSG_EDT_SATISFY_SLOT,
  MSG_EVENT_SATISFY_SLOT,
  MSG_EVENT_ADD_DEPENDENCE,
  MSG_EDT_CREATE,
  MSG_EVENT_CREATE,
  MSG_TIME_SYNC_REQUEST,
  MSG_TIME_SYNC_RESPONSE,
  /* coherence protocol messages.  The dispatcher routes these to
   * arts_handler_db_* in libs/src/core/memory/coherence_handlers.c.
   * Sequential append (CLAUDE.md rule: NO gaps in this enum). */
  MSG_DB_OWNERSHIP_REQUEST,
  MSG_DB_OWNERSHIP_RESPONSE,
  MSG_DB_WRITEBACK,
  MSG_DB_WRITEBACK_ACK,
  MSG_DB_OWNERSHIP_INVALIDATE,
  MSG_DB_OWNERSHIP_RETURN,
  MSG_DB_SNAPSHOT_REQUEST,
  MSG_DB_SNAPSHOT_RESPONSE,
  MSG_DB_CREATE,
  MSG_DB_DESTROY,
  MSG_DB_CACHE_DESTROY,
  /* Event subsystem rewrite: cross-rank
   * arts_event_destroy → mark_delete on the home rank.  Sequential append,
   * no gaps. */
  MSG_EVENT_DESTROY,
  /* Cross-rank arts_edt_destroy → home-rank destroy (symmetric with
   * MSG_EVENT_DESTROY / MSG_DB_DESTROY; OoO-deferred on before-create
   * reorder).  Sequential append, no gaps. */
  MSG_EDT_DESTROY,
  /* LRC (Location-Consistency) coherence messages.  Only sent/received when
   * both ranks are compiled with ARTS_MEMORY_MODEL=LRC.  Sequential append,
   * no gaps. */
  MSG_DB_SNAPSHOT_REDIRECT,
  MSG_DB_OWNERSHIP_RESPONSE_ACK,
  /* RW-acquire pipelining: home → designated next owner. "You are the secured
   * next owner of this DB; advance your RW acquire cursor." Sent alongside the
   * ownership INVALIDATE. RC/LRC only. Sequential append, no gaps. */
  MSG_DB_OWNERSHIP_PROCEED,

  MSG_COUNT, /* sentinel — keep last; used for array sizing */
};

/* WRITEBACK packet flag — selects normal write-back vs. write-back +
 * ownership transfer.  Coherence release path uses these constants
 * (libs/src/core/memory/coherence_release.c lines 166, 174) and the
 * home-side WRITEBACK handler dispatches on the value.  uint8_t in the
 * wire packet to keep the struct layout tight. */
typedef enum {
  ARTS_WB_NORMAL = 0,
  ARTS_WB_AND_TRANSFER = 1,
} arts_writeback_flag_t;

// Header
struct ARTS_PACKED arts_msg_header_s {
  unsigned int message_type;
  uint64_t size;
  unsigned int rank;
#ifdef SEQUENCENUMBERS
  unsigned int seq_rank;
  uint64_t seq_num;
#endif
};

struct ARTS_PACKED arts_msg_guid_only_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t guid;
};

struct ARTS_PACKED arts_msg_add_dependence_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t source;
  arts_guid_t destination;
  uint32_t slot;
  arts_db_access_mode_t mode;
};

struct ARTS_PACKED arts_msg_edt_satisfy_slot_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t edt;
  arts_guid_t db;
  uint32_t slot;
  arts_db_access_mode_t mode;
  /* Inline payload byte count following the header (DB_MODE_PTR delivery);
   * zero when the satisfy carries only a GUID/value reference. */
  unsigned int size;
};

struct ARTS_PACKED arts_msg_event_satisfy_slot_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t event;
  arts_guid_t db;
  uint32_t slot;
};

// Time synchronization packets for RTT-based clock sync
// Worker sends request with its send time T1
struct ARTS_PACKED arts_msg_time_sync_req_packet_s {
  struct arts_msg_header_s header;
  uint64_t worker_send_time; // T1: worker's local time when sending request
};

// Master responds with T1 (echoed) and T2 (master's receive time)
struct ARTS_PACKED arts_msg_time_sync_resp_packet_s {
  struct arts_msg_header_s header;
  uint64_t worker_send_time; // T1: echoed back
  uint64_t master_recv_time; // T2: master's local time when receiving request
};

/* ===== coherence wire packets =======================
 * Pad-fields exist to keep the trailing payload (when present) on an
 * 8-byte boundary; senders call arts_transport_send_payload_async right
 * after sizeof(packet_struct) bytes, so the payload
 * starts at sizeof() — that offset must be 8-aligned.  Header is packed
 * (44 bytes), so each struct's body fields determine the pad. */

struct ARTS_PACKED arts_msg_ownership_request_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
};

/* OWNERSHIP_RESPONSE — the single ownership-transfer wire message.
 * RC carries an optional trailing buffer payload (data_present == 1).
 * LRC carries a serialized last_sent_version map + buffer payload (the old
 * model-prefixed MSG_DB_OWNERSHIP_RESPONSE_LRC collapsed into this). */
#ifdef ARTS_MEMORY_MODEL_LRC
struct ARTS_PACKED arts_msg_ownership_response_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t version;
  uint32_t map_entry_count; /* arts_msg_rank_version_pair_s entries that
                               follow the header */
  uint32_t pad;
  /* followed by:
   *   arts_msg_rank_version_pair_s pairs[map_entry_count];
   *   uint8_t data[db_size];
   */
};
#else /* RC */
struct ARTS_PACKED arts_msg_ownership_response_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t version;
  uint32_t has_next;
  uint32_t data_present;
  uint8_t pad[4];
};
#endif

/* WRITEBACK carries optional trailing buffer payload.
 * Body: db_guid(8) + version(8) + cv(8) + flag(1) = 25,
 * total = 44 + 25 = 69; pad[3] -> 72 (8-aligned).
 * cv: opaque address of the releaser's stack-local sem_t, valid only at the
 * releaser rank; the home forwards it verbatim in the ACK so the releaser
 * matches by pointer identity (no seq tracking). */
struct ARTS_PACKED arts_msg_writeback_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t version;
  uint64_t cv;
  uint8_t flag;
  uint8_t pad[3];
};

struct ARTS_PACKED arts_msg_writeback_ack_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t cv; /* releaser's sem_t address, forwarded verbatim from WRITEBACK */
};

/* INVALIDATE_NOTICE: body = db_guid(8) + new_owner_rank(4) + pad(4) = 16.
 * Total = 44 + 16 = 60 (not 8-aligned; add pad4[] → 64).
 * RC builds send new_owner_rank=0 (ignored by the handler).
 * LRC builds set new_owner_rank so the current holder knows where to send
 * TRANSFER_OWNERSHIP without a round-trip to home. */
struct ARTS_PACKED arts_msg_ownership_invalidate_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t new_owner_rank;
  uint8_t pad[4];
};

struct ARTS_PACKED arts_msg_ownership_return_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
};

struct ARTS_PACKED arts_msg_ownership_proceed_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
};

struct ARTS_PACKED arts_msg_snapshot_request_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  arts_guid_t edt_guid; /* parked EDT to resume on the requester rank */
  uint32_t slot;        /* dep slot index in the parked EDT */
  uint8_t pad[4];
};

/* DATA_RESPONSE carries optional trailing buffer payload.  Echoes the parked
 * EDT (edt_guid + slot) back so the requester's response handler resumes it
 * directly — no acquire-time list registration (the reorder-buffer design). */
struct ARTS_PACKED arts_msg_snapshot_response_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t version;
  arts_guid_t edt_guid; /* parked EDT to resume (echoed from request) */
  uint32_t slot;        /* dep slot index (echoed from request) */
  uint32_t data_present;
};

/* DB_CREATE_COHERENT — no trailing payload (zero-init buffer at home).
 * Body: db_guid(8) + db_size(8) + flags(2) + db_type(2) = 20,
 * total = 44 + 20 = 64 (already 8-aligned).  Keep pad[4] anyway so any
 * future ABI growth has space without changing on-wire size. */
struct ARTS_PACKED arts_msg_db_create_coherent_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t db_size;
  uint16_t flags;
  uint16_t db_type;
  uint8_t pad[4];
};

struct ARTS_PACKED arts_msg_destroy_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
};

struct ARTS_PACKED arts_msg_cache_destroy_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
};

/* ===== LRC (Location-Consistency) wire packets ==============================
 * Sent only between ranks compiled with ARTS_MEMORY_MODEL=LRC.
 * A rank compiled with the opposite mode that receives these messages fatals
 * immediately (see dispatcher.c). */

/* REDIRECT_RO — home forwards an RO grant request to the current owner.
 * The owner will send data directly to requester_rank using INSTALL_ACK. */
struct ARTS_PACKED arts_msg_snapshot_redirect_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  arts_guid_t edt_guid; /* parked EDT at requester_rank to resume */
  uint32_t requester_rank;
  uint32_t slot; /* dep slot index in the parked EDT */
};

/* TRANSFER_OWNERSHIP — owner sends data + version + reader-map to new owner.
 * Followed by:
 *   arts_msg_rank_version_pair_s pairs[map_entry_count];
 *   uint8_t                         data[db_size];
 */
struct ARTS_PACKED arts_msg_rank_version_pair_s {
  uint32_t rank;
  uint32_t pad;
  uint64_t version;
};

/* INSTALL_ACK — new owner (or forwarder) confirms installation to requester.
 * Carries a fresh version number so the requester can track its RO snapshot. */
struct ARTS_PACKED arts_msg_install_ack_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t version;
};

#include "arts/system/threads.h"

static inline void arts_fill_packet_header(struct arts_msg_header_s *header,
                                           uint64_t size,
                                           unsigned int message_type) {
  header->size = size;
  header->message_type = message_type;
  header->rank = arts_global_rank_id;
}

#ifdef __cplusplus
}
#endif

#endif
