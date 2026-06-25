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
/* SEQUENCENUMBERS is a performance-sensitive transport drop/reorder diagnostic
 * (per-send lock + per-receive ordering check).  It is gated by the
 * ARTS_SEQUENCE_NUMBERS CMake option (default OFF); do NOT hardcode it here. */

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
   * arts_handler_db_* in libs/src/core/coherence/handlers.c.
   * Sequential append (CLAUDE.md rule: NO gaps in this enum). */
  MSG_DB_OWNERSHIP_REQUEST,
  MSG_DB_OWNERSHIP_RESPONSE,
  MSG_DB_WRITEBACK,
  MSG_DB_WRITEBACK_ACK,
  MSG_DB_OWNERSHIP_INVALIDATE,
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
  /* Lazy-protocol-only coherence messages.  Only sent/received when both
   * ranks are compiled with the lazy coherence protocol.  Sequential append,
   * no gaps. */
  MSG_DB_SNAPSHOT_REDIRECT,
  /* Lazy-protocol-only: new owner C → home A, "I have installed the transferred
   * DB." Home reacts by flipping rw_holder to C and replying with CONFIRM_ACK.
   * Sequential append, no gaps. */
  MSG_DB_OWNERSHIP_CONFIRM,
  /* Lazy-protocol-only: home A → new owner C, "the directory now names you; you
   * may run your RW EDT." Acknowledges that home has flipped rw_holder to C (in
   * reaction to C's CONFIRM). Gates C's RW execution so its write becomes
   * observable only after the directory reflects C (no stale-RO window).
   * Sequential append, no gaps. */
  MSG_DB_OWNERSHIP_CONFIRM_ACK,

  /* LOCK-protocol coherence messages (REQUEST / GRANT / RELEASE / RELEASE_ACK).
   * Only sent/received in ARTS_COHERENCE_PROTOCOL=LOCK builds; the
   * dispatcher's LOCK cases are #ifdef-guarded.  Sequential append, no gaps. */
  MSG_DB_LOCK_REQUEST,
  MSG_DB_LOCK_GRANT,
  MSG_DB_LOCK_RELEASE,
  /* Synchronous writeback ACK: home → RW releaser after installing the
   * writeback payload.  Unblocks the releaser's await in arts_db_release_rw
   * so lock_home_grant cannot run before the new data is at home.  RO
   * releases are fire-and-forget and never send this message. */
  MSG_DB_LOCK_RELEASE_ACK,
  /* LOCK-LAZY-only messages (FORWARD / DELIVER / CONFIRM / RORET).
   * Used only in ARTS_COHERENCE_PROTOCOL=LOCK + ARTS_PROTOCOL_TIMING=LAZY
   * builds.  Sequential append, no gaps. */
  MSG_DB_LOCK_FORWARD, /* home → current owner: serve RO reader or migrate RW */
  MSG_DB_LOCK_DELIVER, /* owner → target: data + mode (no version field) */
  MSG_DB_LOCK_CONFIRM, /* new owner → home: migration complete */
  MSG_DB_LOCK_RORET,   /* reader → home: RO release (data-less) */

  MSG_COUNT, /* sentinel — keep last; used for array sizing */
};

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
  /* Pad so sizeof() (where the trailing inline payload begins) is 8-aligned:
   * 16-byte header + 8 + 8 + 4 + 4 + 4 = 44, +4 -> 48.  Keeps the payload's
   * wire offset 8-aligned. */
  uint32_t pad;
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

/* OWNERSHIP_RESPONSE — the single ownership-transfer wire message, ONE layout
 * for both timings: a serialized last_sent_version map (map_entry_count pairs)
 * followed by the buffer payload.  The owner→owner transfer ships this for both
 * EAGER and LAZY; EAGER sends map_entry_count=0 (it dedups RO via home's
 * last_sent_version, not an owner-side map), LAZY serializes its owner-side
 * map.
 */
struct ARTS_PACKED arts_msg_ownership_response_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t version;
  uint32_t map_entry_count; /* arts_msg_rank_version_pair_s entries that
                               follow the header */
  uint32_t pad;
  /* MRSW note: the TRANSFER_OWNERSHIP carries NO edt — it moves only the
   * buffer + dedup map; the one EDT this round serves rides the subsequent home
   * CONFIRM packet (which the new owner acts on once the directory names it).
   */
  /* followed by:
   *   arts_msg_rank_version_pair_s pairs[map_entry_count];
   *   uint8_t data[db_size];
   */
};

/* WRITEBACK carries optional trailing buffer payload.
 * Body: db_guid(8) + version(8) + cv(8) = 24; with the 16-byte header the total
 * is 40, already 8-aligned, so the trailing payload begins on an 8-byte
 * boundary with no pad field.
 * cv: opaque address of the releaser's stack-local sem_t, valid only at the
 * releaser rank; the home forwards it verbatim in the ACK so the releaser
 * matches by pointer identity (no seq tracking). */
struct ARTS_PACKED arts_msg_writeback_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t version;
  uint64_t cv;
};

struct ARTS_PACKED arts_msg_writeback_ack_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t cv; /* releaser's sem_t address, forwarded verbatim from WRITEBACK */
};

/* INVALIDATE_NOTICE: body = db_guid(8) + new_owner_rank(4) + pad(4) = 16.
 * Total = 44 + 16 = 60 (not 8-aligned; add pad4[] → 64).
 * Both timings set new_owner_rank so the current holder knows where to ship the
 * owner→owner OWNERSHIP_RESPONSE without a round-trip to home. */
struct ARTS_PACKED arts_msg_ownership_invalidate_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t new_owner_rank;
  uint8_t pad[4];
};

/* OWNERSHIP_CONFIRM_ACK: home → new owner C. Body = db_guid(8). No version: C
 * already holds its installed version. Lazy-only by use (eager never sends it);
 * the struct is unconditional. */
struct ARTS_PACKED arts_msg_ownership_confirm_ack_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  /* Lazy: the CONFIRM_ACK that advances the round piggybacks the next
   * transfer target so the new owner's handler applies the INVALIDATE effect
   * (publish incoming_new_owner + withdraw the sentinel) in the same message,
   * eliminating the separate INVALIDATE and its CONFIRM_ACK↔INVALIDATE reorder
   * window.  ARTS_LAZY_NO_PENDING_OWNER ⇒ plain ack, no piggybacked invalidate.
   * Mirrors INVALIDATE_NOTICE's new_owner_rank field. */
  uint32_t new_owner_rank;
  uint8_t pad[4];
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

/* ===== Lazy-protocol-only wire packets ======================================
 * Sent only between ranks compiled with the lazy coherence protocol.
 * A rank compiled with a different protocol that receives these messages fatals
 * immediately (see dispatcher.c). */

/* REDIRECT_RO — home forwards an RO grant request to the current owner.
 * The owner will send data directly to requester_rank using DATA_RESPONSE. */
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

/* OWNERSHIP_CONFIRM — new owner C confirms installation of the transferred DB
 * to home A. Carries a fresh version number so home can track the RW round. */
struct ARTS_PACKED arts_msg_ownership_confirm_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t version;
};

#ifdef ARTS_PROTOCOL_LOCK
struct ARTS_PACKED arts_msg_lock_request_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t mode; /* arts_db_access_mode_t: DB_MODE_RO or DB_MODE_RW */
  uint32_t pad;
  /* requester rank = header.rank; no edt_guid/slot (rank-granular). */
};

struct ARTS_PACKED arts_msg_lock_grant_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t mode;
  uint32_t pad;
  uint64_t version; /* monotone round counter; buf_install rejects stale */
  /* followed by: uint8_t data[db_size]; */
};

struct ARTS_PACKED arts_msg_lock_release_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t mode;
  uint32_t pad;
  uint64_t
      version; /* monotone version bumped by releaser; stale overwrite guard */
  uint64_t cv; /* RW only: releaser's stack-local sem_t address for ACK */
  /* followed by: uint8_t data[db_size]; only when mode==DB_MODE_RW. */
};

/* LOCK_RELEASE_ACK: home → RW releaser after installing writeback.  Carries
 * cv verbatim from the LOCK_RELEASE so the releaser wakes by pointer identity.
 */
struct ARTS_PACKED arts_msg_lock_release_ack_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t cv; /* releaser's sem_t address, forwarded verbatim from RELEASE */
};

/* ===== LOCK-LAZY-only wire packets =========================================
 * Sent only between ranks compiled with LOCK+LAZY.  Members unconditional;
 * structs inside the ARTS_PROTOCOL_LOCK guard so they share the LOCK types. */
#ifdef ARTS_TIMING_LAZY
/* LOCK_FORWARD: home → current owner.  mode=DB_MODE_RW → migrate ownership to
 * target; mode=DB_MODE_RO → serve one RO reader at target. */
struct ARTS_PACKED arts_msg_lock_forward_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t mode;   /* DB_MODE_RW = migrate, DB_MODE_RO = serve reader */
  uint32_t target; /* migrate: new-owner rank; serve: reader rank */
};

/* LOCK_DELIVER: owner → target.  Carries the DB data inline.  NO version
 * field — LOCK uses versionless buffer-install (exclusive-lock serialization
 * guarantees no stale write can race; mirrors the EAGER GRANT layout). */
struct ARTS_PACKED arts_msg_lock_deliver_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t mode; /* DB_MODE_RW = new owner, DB_MODE_RO = reader copy */
  uint32_t pad;
  /* followed by: uint8_t data[db_size] */
};

/* LOCK_CONFIRM / LOCK_RORET share one struct: both are data-less
 * db_guid-only messages (new-owner→home confirm, and reader→home RO release).
 */
struct ARTS_PACKED arts_msg_lock_confirm_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
};
#endif /* ARTS_TIMING_LAZY */
#endif /* ARTS_PROTOCOL_LOCK */

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
