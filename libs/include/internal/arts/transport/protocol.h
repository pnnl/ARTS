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
  MSG_DB_GRANT_REQUEST,
  MSG_DB_GRANT_RESPONSE,
  MSG_DB_PUBLISH,
  MSG_DB_PUBLISH_ACK,
  MSG_DB_GRANT_INVALIDATE,
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
  /* WB-write-policy-only coherence messages.  Only sent/received when both
   * ranks are compiled with the WB write policy.  Sequential append,
   * no gaps. */
  MSG_DB_SNAPSHOT_REDIRECT,
  /* WB-write-policy-only: new owner C → home A, "I have installed the transferred
   * DB." Home reacts by flipping rw_holder to C and replying with CONFIRM_ACK.
   * Sequential append, no gaps. */
  MSG_DB_GRANT_CONFIRM,
  /* WB-write-policy-only: home A → new owner C, "the directory now names you; you
   * may run your RW EDT." Acknowledges that home has flipped rw_holder to C (in
   * reaction to C's CONFIRM). Gates C's RW execution so its write becomes
   * observable only after the directory reflects C (no stale-RO window).
   * Sequential append, no gaps. */
  MSG_DB_GRANT_CONFIRM_ACK,

  /* EXCL-protocol coherence messages (REQUEST / GRANT / RELEASE / RELEASE_ACK).
   * Only sent/received in ARTS_COHERENCE_PROTOCOL=EXCL builds; the
   * dispatcher's EXCL cases are #ifdef-guarded.  Sequential append, no gaps. */
  MSG_DB_EXCL_REQUEST,
  MSG_DB_EXCL_GRANT,
  MSG_DB_EXCL_RELEASE,
  /* Synchronous publish ACK: home → RW releaser after installing the
   * publish payload.  Unblocks the releaser's await in arts_db_release_rw
   * so lock_home_grant cannot run before the new data is at home.  RO
   * releases are fire-and-forget and never send this message. */
  MSG_DB_EXCL_RELEASE_ACK,
  /* EXCL RETAIN-only messages (FORWARD / DELIVER / CONFIRM / RORET).
   * Used only in ARTS_COHERENCE_PROTOCOL=EXCL + ARTS_RELEASE_POLICY=RETAIN
   * builds.  Sequential append, no gaps. */
  MSG_DB_EXCL_FORWARD, /* home → current owner: serve RO reader or migrate RW */
  MSG_DB_EXCL_DELIVER, /* owner → target: data + mode (no version field) */
  MSG_DB_EXCL_CONFIRM, /* new owner → home: migration complete */
  MSG_DB_EXCL_RORET,   /* reader → home: RO release (data-less) */

  /* Rendezvous data-plane control messages.  Bulk payloads move one-sided
   * (fi_writedata PUT into a receiver-advertised landing buffer); these small
   * messages carry the size/landing legs of that handshake when a side cannot
   * know them up front.  Sequential append, no gaps. */
  MSG_DB_GRANT_CTS, /* home → RW requester: db_size for a first-touch
                           request that carried no landing; the requester
                           allocates a landing and re-issues the request. */
  MSG_DB_PUBLISH_CTS, /* home → releaser: a fresh home landing for the dirty
                           publish the releaser announced (PUBLISH with
                           data_size>0, txid==0); the releaser PUTs then sends
                           the final PUBLISH carrying the txid. */
  MSG_DB_EXCL_CTS,      /* EXCL home → requester: db_size for a first-touch
                           EXCL_REQUEST that carried no landing. */

  /* INV-protocol coherence messages.  Only sent/received in
   * ARTS_COHERENCE_PROTOCOL=INV builds; the dispatcher's INV cases are
   * #ifdef-guarded.  Sequential append, no gaps. */
  MSG_DB_INV_REQUEST, /* requester → home: RO fetch (write acquires use the
                         shared GRANT_REQUEST) */
  MSG_DB_INV_CTS,     /* home → requester: db_size for a first-touch REQUEST
                         that carried no landing */
  MSG_DB_INV_DELIVER, /* server → requester: RO copy (payload by PUT).  The
                         version it carries arbitrates the two asynchronous
                         install lanes and nothing else — the sharer plane is
                         versionless: only an INVALIDATE retires a copy. */
  MSG_DB_INV_INVALIDATE,     /* home → sharer: retire the copy ({guid} only) */
  MSG_DB_INV_INVALIDATE_ACK, /* sharer → home: round ack ({guid} only) */
  MSG_DB_INV_REDIRECT, /* home → current grant holder (WB write policy only):
                          serve this reader from your bytes, or bounce the
                          request back if you no longer have them */

  MSG_DB_CREATE_RETURN, /* home -> remote creator, after the home-side
                         * install: the DB's stable home buffer as a durable
                         * publish credit.  Pure hint (Cat-C lookup-or-drop
                         * at the creator). */
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

/* Object blob move (MSG_EDT_CREATE / MSG_EVENT_CREATE): the serialized object
 * trails the header inline, and the receiver derives its length from
 * header.size.  Control messages are two-sided at every size — the provider
 * tiers them internally — so there is no landing to name here. */
struct ARTS_PACKED arts_msg_object_blob_packet_s {
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

/* A satisfy carries a GUID/value reference only — fixed size, no trailing
 * payload, single-shot at the receiver. */
struct ARTS_PACKED arts_msg_edt_satisfy_slot_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t edt;
  arts_guid_t db;
  uint32_t slot;
  arts_db_access_mode_t mode;
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

/* Rendezvous landing advertisement — embedded in request-class messages so the
 * payload sender can fi_writedata straight into the advertiser's registered
 * buffer.  `addr` follows the advertiser's negotiated mr_mode (virtual address
 * under FI_MR_VIRT_ADDR, else offset from the registered base); `key` is the
 * MR protection key; `txid` pairs the write completion with the metadata
 * packet (0 = no landing advertised / no payload moves); `cookie` is an opaque
 * advertiser-local handle (its landing-buffer pointer) echoed VERBATIM back in
 * the metadata packet — the same trust model as the publish `cv` semaphore
 * address, valid only on the advertiser rank. */
struct ARTS_PACKED arts_msg_rdzv_landing_s {
  uint64_t addr;
  uint64_t key;
  uint64_t txid;
  uint64_t cookie;
};

struct ARTS_PACKED arts_msg_grant_request_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  /* Requester's landing for the incoming owner→owner transfer payload.
   * txid==0 = the requester does not yet know db_size (first touch): home
   * answers GRANT_CTS instead of enqueueing, unless the DB itself is a
   * sentinel (db_size==0 at home), which transfers data-less. */
  struct arts_msg_rdzv_landing_s rdzv;
};

/* GRANT_CTS — home → requester: the db_size a first-touch RW requester
 * needs to allocate its landing; the requester re-issues GRANT_REQUEST
 * with the landing attached.  The original landing-less request was NOT
 * enqueued (home only ever queues requests that carry a landing or target a
 * sentinel DB). */
struct ARTS_PACKED arts_msg_grant_cts_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t db_size;
};

/* GRANT_RESPONSE — the single ownership-transfer wire message, ONE layout
 * for both write policies: a serialized cached_version map (map_entry_count pairs)
 * rides INLINE after the header; the buffer payload does NOT ride the wire —
 * it travels one-sided (the old owner PUTs it into the landing the requester
 * advertised in its GRANT_REQUEST) and this packet pairs with that write
 * completion by rdzv_txid.  WT sends map_entry_count=0 (it dedups RO via
 * home's cached_version, not an owner-side map), WB serializes its
 * owner-side map.
 *
 *   rdzv_txid != 0 : `data_size` payload bytes were PUT into the requester's
 *                    landing (named by rdzv_cookie, the requester's own
 *                    echoed handle); install on {packet, txid} pairing.
 *   rdzv_txid == 0 : no payload moved (sentinel DB / pre-publication empty
 *                    transfer, or a same-rank transfer where the data rides
 *                    inline after the map — self-dispatch only, never wire).
 *                    A nonzero rdzv_cookie still echoes the requester's unused
 *                    landing so it can be recycled. */
struct ARTS_PACKED arts_msg_grant_response_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t version;
  uint32_t map_entry_count; /* arts_msg_rank_version_pair_s entries that
                               follow the header */
  uint32_t pad;
  uint64_t data_size;   /* the DB's size (descriptor state, always set);
                           payload presence = rdzv_txid / trailing bytes */
  uint64_t rdzv_txid;   /* write-completion pairing id (0 = no PUT) */
  uint64_t rdzv_cookie; /* requester's landing handle, echoed verbatim */
  /* followed by:
   *   arts_msg_rank_version_pair_s pairs[map_entry_count];
   *   uint8_t data[data_size];   (self-dispatch only — wire payload is PUT)
   */
};

/* PUBLISH — the dirty payload does NOT ride the wire; it travels one-sided
 * into a fresh home landing.  Two-phase from one packet layout:
 *   data_size > 0, rdzv_txid == 0 : announce ("I hold data_size dirty bytes")
 *       — home allocates a fresh landing and replies PUBLISH_CTS; nothing
 *       is installed yet.
 *   data_size > 0, rdzv_txid != 0 : commit — the releaser PUT the bytes into
 *       the CTS landing (rdzv_cookie echoes home's handle); home pairs
 *       {packet, txid}, installs, and ACKs.
 *   data_size == 0                : data-less publish (sentinel DB /
 *       ordering-only round) — install nothing, ACK immediately.
 * cv: opaque address of the releaser's stack-local rendezvous (sem + landing
 * slot), valid only at the releaser rank; home echoes it verbatim in
 * PUBLISH_CTS and PUBLISH_ACK so the releaser matches by pointer identity
 * (no seq tracking). */
struct ARTS_PACKED arts_msg_publish_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t version;
  uint64_t cv;
  uint64_t data_size;
  uint64_t rdzv_txid;
  uint64_t rdzv_cookie;
};

/* PUBLISH_CTS — home → releaser: the fresh home landing for an announced
 * dirty publish.  cv is echoed verbatim; the releaser-side handler writes
 * the landing into the blocked releaser's stack rendezvous and posts it. */
struct ARTS_PACKED arts_msg_publish_cts_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  struct arts_msg_rdzv_landing_s landing;
  uint64_t cv;
};

struct ARTS_PACKED arts_msg_publish_ack_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t cv; /* releaser's sem_t address, forwarded verbatim from PUBLISH
                * (legacy/control legs; 0 under the WT flight path, whose
                * ACK is cache-keyed by db_guid instead) */
  uint64_t version; /* highest version this ACK covers (WT flight path) */
  /* Next publish credit: the DB's stable home buffer + a receiver-minted
   * txid.  Every WT ACK refills it, so the legacy CTS leg's final ACK is
   * what delivers a rank's FIRST durable credit.  Zero under WB/control. */
  uint64_t credit_addr;
  uint64_t credit_rkey;
  uint64_t credit_txid;
};

/* GRANT_INVALIDATE: body = db_guid(8) + new_owner_rank(4) + pad(4) = 16.
 * Total = 44 + 16 = 60 (not 8-aligned; add pad4[] → 64).
 * Both write policies set new_owner_rank so the current holder knows where to ship the
 * owner→owner GRANT_RESPONSE without a round-trip to home. */
struct ARTS_PACKED arts_msg_grant_invalidate_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t new_owner_rank;
  uint8_t pad[4];
  /* The NEW owner's landing (from its queued GRANT_REQUEST), forwarded so
   * the current holder can PUT the transfer payload without a home
   * round-trip.  txid==0 = sentinel DB round (data-less transfer). */
  struct arts_msg_rdzv_landing_s new_owner_rdzv;
};

/* GRANT_CONFIRM_ACK: home → new owner C. Body = db_guid(8). No version: C
 * already holds its installed version. WB-only by use (WT never sends it);
 * the struct is unconditional. */
struct ARTS_PACKED arts_msg_grant_confirm_ack_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  /* WB: the CONFIRM_ACK that advances the round piggybacks the next
   * transfer target so the new owner's handler applies the INVALIDATE effect
   * (publish incoming_new_owner + withdraw the sentinel) in the same message,
   * eliminating the separate INVALIDATE and its CONFIRM_ACK↔INVALIDATE reorder
   * window.  ARTS_NO_PENDING_OWNER ⇒ plain ack, no piggybacked invalidate.
   * Mirrors GRANT_INVALIDATE's new_owner_rank field. */
  uint32_t new_owner_rank;
  uint8_t pad[4];
  /* Piggybacked next-owner landing (mirrors GRANT_INVALIDATE's field). */
  struct arts_msg_rdzv_landing_s new_owner_rdzv;
};

struct ARTS_PACKED arts_msg_snapshot_request_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  arts_guid_t edt_guid; /* parked EDT to resume on the requester rank */
  uint32_t slot;        /* dep slot index in the parked EDT */
  uint8_t pad[4];
  /* Requester's landing for the snapshot payload.  txid==0 = requester does
   * not yet know db_size (first touch): the server answers a size-only CTS
   * response (data_present == 2) and the requester re-issues with a landing. */
  struct arts_msg_rdzv_landing_s rdzv;
};

/* SNAPSHOT_RESPONSE — the snapshot payload does NOT ride the wire; it is PUT into
 * the landing the requester advertised.  Echoes the parked EDT (edt_guid +
 * slot) back so the requester's response handler resumes it directly — no
 * acquire-time list registration (the reorder-buffer design).
 * data_present values:
 *   0 = no payload moved (requester's version is current, or nothing
 *       published yet).  A nonzero rdzv_cookie echoes the unused landing for
 *       recycling.
 *   1 = `db_size` payload bytes were PUT into the requester's landing
 *       (rdzv_cookie); install on {packet, txid} pairing.
 *   2 = size-only CTS: the request carried no landing and data exists —
 *       db_size tells the requester what to allocate; it re-issues the
 *       request with a landing.  No watermark advances on this leg. */
struct ARTS_PACKED arts_msg_snapshot_response_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t version;
  arts_guid_t edt_guid; /* parked EDT to resume (echoed from request) */
  uint32_t slot;        /* dep slot index (echoed from request) */
  uint32_t data_present;
  uint64_t db_size;     /* payload/allocation byte count (see data_present) */
  uint64_t rdzv_txid;   /* write-completion pairing id (0 = no PUT) */
  uint64_t rdzv_cookie; /* requester's landing handle, echoed verbatim */
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

/* ===== WB-write-policy-only wire packets ====================================
 * Sent only between ranks compiled with the WB write policy.
 * A rank compiled with a different protocol that receives these messages fatals
 * immediately (see dispatcher.c). */

/* SNAPSHOT_REDIRECT — home forwards an RO grant request to the current owner.
 * The owner will send data directly to requester_rank using SNAPSHOT_RESPONSE,
 * PUTting the payload into the forwarded requester landing. */
struct ARTS_PACKED arts_msg_snapshot_redirect_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  arts_guid_t edt_guid; /* parked EDT at requester_rank to resume */
  uint32_t requester_rank;
  uint32_t slot; /* dep slot index in the parked EDT */
  /* Requester's landing, forwarded verbatim from its SNAPSHOT_REQUEST. */
  struct arts_msg_rdzv_landing_s rdzv;
};

/* GRANT_RESPONSE's reader-map trailer — owner sends data + version + reader-map
 * to new owner.  Followed by:
 *   arts_msg_rank_version_pair_s pairs[map_entry_count];
 *   uint8_t                         data[db_size];
 */
struct ARTS_PACKED arts_msg_rank_version_pair_s {
  uint32_t rank;
  uint32_t pad;
  uint64_t version;
};

/* GRANT_CONFIRM — new owner C confirms installation of the transferred DB
 * to home A. Carries a fresh version number so home can track the RW round. */
struct ARTS_PACKED arts_msg_grant_confirm_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t version;
};

/* CREATE_RETURN — see the enum comment.  The credit triple mirrors the
 * publish-ACK refill fields. */
struct ARTS_PACKED arts_msg_db_create_return_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t credit_addr;
  uint64_t credit_rkey;
  uint64_t credit_txid;
};

#ifdef ARTS_PROTOCOL_EXCL
struct ARTS_PACKED arts_msg_excl_request_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t mode; /* arts_db_access_mode_t: DB_MODE_RO or DB_MODE_RW */
  uint32_t pad;
  /* requester rank = header.rank; no edt_guid/slot (rank-granular). */
  /* Requester's landing for the grant/deliver payload.  txid==0 = requester
   * does not yet know db_size (first touch): home answers EXCL_CTS and the
   * requester re-issues with a landing (sentinel DBs grant data-less). */
  struct arts_msg_rdzv_landing_s rdzv;
};

/* EXCL_CTS — home → requester: db_size for a first-touch EXCL_REQUEST that
 * carried no landing.  Echoes the mode so the requester re-issues the same
 * request.  The landing-less request was NOT queued/granted. */
struct ARTS_PACKED arts_msg_excl_cts_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t db_size;
  uint32_t mode;
  uint32_t pad;
};

struct ARTS_PACKED arts_msg_excl_grant_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t mode;
  uint32_t pad;
  uint64_t version; /* monotone round counter; buf_install rejects stale */
  /* Grant payload travels by PUT into the requester's landing:
   * rdzv_txid != 0 pairs this packet with the write completion, rdzv_cookie
   * echoes the requester's landing handle.  data_size is the DB's size
   * (descriptor state, ALWAYS set — a hinted first touch skips the size CTS
   * and learns the exact size from whichever reply completes it); payload
   * presence is signaled by rdzv_txid / trailing bytes, never by data_size.
   * txid == 0 = data-less grant (sentinel / nothing published). */
  uint64_t data_size;
  uint64_t rdzv_txid;
  uint64_t rdzv_cookie;
  /* Home's landing for THIS grant's eventual RW release publish (grants and
   * releases pair 1:1): the releaser PUTs its dirty bytes here and echoes
   * {txid, cookie} in EXCL_RELEASE.  txid==0 for RO grants / sentinel DBs. */
  struct arts_msg_rdzv_landing_s pub;
};

struct ARTS_PACKED arts_msg_excl_release_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t mode;
  uint32_t pad;
  uint64_t
      version; /* monotone version bumped by releaser; stale overwrite guard */
  uint64_t cv; /* RW only: releaser's stack-local sem_t address for ACK */
  /* RW dirty payload travels by PUT into the grant's `pub` landing; these echo
   * that landing's {txid, cookie} and count the landed bytes.  txid==0 =
   * data-less release (RO, or sentinel DB). */
  uint64_t data_size;
  uint64_t rdzv_txid;
  uint64_t rdzv_cookie;
};

/* EXCL_RELEASE_ACK: home → RW releaser after installing publish.  Carries
 * cv verbatim from the EXCL_RELEASE so the releaser wakes by pointer identity.
 */
struct ARTS_PACKED arts_msg_excl_release_ack_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t cv; /* releaser's sem_t address, forwarded verbatim from RELEASE */
};

/* ===== EXCL RETAIN-only wire packets ========================================
 * Sent only between ranks compiled with EXCL+RETAIN.  Members unconditional;
 * structs inside the ARTS_PROTOCOL_EXCL guard so they share the EXCL types. */
#ifdef ARTS_RELEASE_RETAIN
/* EXCL_FORWARD: home → current owner.  mode=DB_MODE_RW → migrate ownership to
 * target; mode=DB_MODE_RO → serve one RO reader at target.  Forwards the
 * target's landing (from its EXCL_REQUEST) so the owner can PUT directly. */
struct ARTS_PACKED arts_msg_excl_forward_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t mode;   /* DB_MODE_RW = migrate, DB_MODE_RO = serve reader */
  uint32_t target; /* migrate: new-owner rank; serve: reader rank */
  struct arts_msg_rdzv_landing_s rdzv; /* target's landing, forwarded */
};

/* EXCL_DELIVER: owner → target.  The DB data travels by PUT into the target's
 * landing; this packet pairs with the write completion by rdzv_txid
 * (rdzv_cookie echoes the target's landing handle, data_size counts the
 * landed bytes; txid==0 = data-less deliver).  NO version field — EXCL uses
 * versionless buffer-install (exclusive-lock serialization guarantees no
 * stale write can race). */
struct ARTS_PACKED arts_msg_excl_deliver_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t mode; /* DB_MODE_RW = new owner, DB_MODE_RO = reader copy */
  uint32_t pad;
  uint64_t data_size;
  uint64_t rdzv_txid;
  uint64_t rdzv_cookie;
};

/* EXCL_CONFIRM / EXCL_RORET share one struct: both are data-less
 * db_guid-only messages (new-owner→home confirm, and reader→home RO release).
 */
struct ARTS_PACKED arts_msg_excl_confirm_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
};
#endif /* ARTS_RELEASE_RETAIN */
#endif /* ARTS_PROTOCOL_EXCL */

/* ===== INV wire packets ======================================================
 * Sent only between ranks compiled with ARTS_COHERENCE_PROTOCOL=INV.
 * Members unconditional; structs guarded so they can share host-side types. */
#ifdef ARTS_PROTOCOL_INV
/* A read request names its subject explicitly rather than relying on the
 * header's sender: under the WB write policy a holder that cannot serve
 * re-sends the request on the reader's behalf, so the two differ. */
struct ARTS_PACKED arts_msg_inv_request_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t mode; /* arts_db_access_mode_t — RO; writes use GRANT_REQUEST */
  uint32_t requester;
  /* Requester's landing for the deliver payload.  txid==0 = the requester does
   * not yet know db_size (first touch): the home answers INV_CTS and the
   * requester re-issues with a landing. */
  struct arts_msg_rdzv_landing_s rdzv;
};

struct ARTS_PACKED arts_msg_inv_cts_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t db_size;
  uint32_t mode;
  uint32_t pad;
};

/* The version is the install-lane arbitration stamp ONLY (two asynchronous
 * installs can target one buffer slot); nothing compares it to judge validity. */
struct ARTS_PACKED arts_msg_inv_deliver_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint64_t version;
  uint64_t data_size;
  uint64_t rdzv_txid;
  uint64_t rdzv_cookie;
};

struct ARTS_PACKED arts_msg_inv_invalidate_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
};

struct ARTS_PACKED arts_msg_inv_invalidate_ack_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
};

/* WB write policy only: the home forwards a read it cannot answer. */
struct ARTS_PACKED arts_msg_inv_redirect_packet_s {
  struct arts_msg_header_s header;
  arts_guid_t db_guid;
  uint32_t requester;
  uint32_t pad;
  struct arts_msg_rdzv_landing_s rdzv;
};
#endif /* ARTS_PROTOCOL_INV */


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
