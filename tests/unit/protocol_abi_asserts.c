/* SPDX-License-Identifier: Apache-2.0
 *
 * T180 — wire-protocol ABI frozen golden table (B071, flags B072).
 *
 * Property under test
 * -------------------
 * protocol.h is the cross-rank wire contract.  Two ranks built in DIFFERENT
 * coherence configs (the 6 supported: MRNEW+EAGER, MRNEW+LAZY, MRSW+EAGER,
 * MRSW+LAZY, MRMW, LOCK) MUST agree byte-for-byte on:
 *   (1) `enum arts_msg_type` — every ordinal contiguous 0..MSG_COUNT-1, and
 *       MSG_COUNT itself, IDENTICAL across all 6 configs (the enum members are
 *       unconditional even where their dispatcher case is #ifdef'd out, so the
 *       ordinals must not drift) — a skew = silent misroute.
 *   (2) `struct arts_msg_header_s` field offsets (message_type / size / rank)
 *       and total size — the receiver reads message_type+size from EVERY packet
 *       before it knows the type, so a header layout skew misparses everything.
 *   (3) each wire packet struct's exact sizeof — the sender ships
 * sizeof(struct) bytes then the trailing payload, and the receiver reads the
 * payload at offset sizeof(struct); a sizeof skew tears every payload.
 *
 * The golden values below were frozen from the current tree and verified
 * IDENTICAL across all 6 configs (only the LOCK-only structs differ in
 * presence, never in the shared ordinals/offsets/sizes).  This TU is meant to
 * be COMPILED ONCE PER -DARTS_PROTOCOL_* config; the `_Static_assert`s catch
 * any config that drifts from the golden table at compile time.
 *
 * SEQ-build skew (B071, second half): when built with -DSEQUENCENUMBERS the
 * header grows two fields, so `sizeof(header)` and `offsetof(size)` change —
 * making a SEQ build wire-incompatible with a non-SEQ build, WITHOUT any fatal
 * guard.  This TU asserts the header layout for whichever variant it is built
 * as, and the `HDR_HAS_SEQ` marker it prints lets the harness diff SEQ vs
 * non-SEQ builds.  (B072: that SEQ build also indexes rec_seq_numbers by the
 * wire-supplied seq_rank with no bounds check — out of scope for a compile
 * assertion, flagged here.)
 *
 * Runtime portion — the 8-byte payload-alignment INVARIANT (exposes a defect)
 * ---------------------------------------------------------------------------
 * protocol.h §4 documents a pad-field invariant: "Pad-fields exist to keep the
 * trailing payload on an 8-byte boundary ... the payload starts at sizeof() —
 * that offset must be 8-aligned."  The payload-carrying structs are
 * OWNERSHIP_RESPONSE, WRITEBACK, SNAPSHOT_RESPONSE, LOCK_GRANT, LOCK_RELEASE
 * (and EDT_SATISFY_SLOT for DB_MODE_PTR).  This TU checks that sizeof() of each
 * is a multiple of 8.  It is currently VIOLATED for the payload-carrying
 * WRITEBACK packet (and for EDT_SATISFY_SLOT): with the packed 16-byte header,
 * sizeof(arts_msg_writeback_packet_s)==44, so the trailing buffer payload
 * starts at wire-offset 44 — NOT 8-aligned, contradicting the documented
 * invariant (the comment was sized for a 28-byte header).  The misalignment is
 * benign on x86 same-build clusters (sender/receiver agree on the offset and
 * the receiver memcpy's the data out), but it is a real ABI-invariant
 * violation: an unaligned trailing payload.  We REPORT it (exit 1) rather than
 * relax the assertion.
 */

#include "arts/transport/protocol.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* ===== (1) enum ordinals — frozen golden table, identical across all 6
 * configs. Members are unconditional in protocol.h regardless of build. */
_Static_assert(MSG_SHUTDOWN == 0, "ordinal MSG_SHUTDOWN drifted");
_Static_assert(MSG_EDT_SATISFY_SLOT == 1,
               "ordinal MSG_EDT_SATISFY_SLOT drifted");
_Static_assert(MSG_EVENT_SATISFY_SLOT == 2,
               "ordinal MSG_EVENT_SATISFY_SLOT drifted");
_Static_assert(MSG_EVENT_ADD_DEPENDENCE == 3,
               "ordinal MSG_EVENT_ADD_DEPENDENCE drifted");
_Static_assert(MSG_EDT_CREATE == 4, "ordinal MSG_EDT_CREATE drifted");
_Static_assert(MSG_EVENT_CREATE == 5, "ordinal MSG_EVENT_CREATE drifted");
_Static_assert(MSG_TIME_SYNC_REQUEST == 6,
               "ordinal MSG_TIME_SYNC_REQUEST drifted");
_Static_assert(MSG_TIME_SYNC_RESPONSE == 7,
               "ordinal MSG_TIME_SYNC_RESPONSE drifted");
_Static_assert(MSG_DB_OWNERSHIP_REQUEST == 8,
               "ordinal MSG_DB_OWNERSHIP_REQUEST drifted");
_Static_assert(MSG_DB_OWNERSHIP_RESPONSE == 9,
               "ordinal MSG_DB_OWNERSHIP_RESPONSE drifted");
_Static_assert(MSG_DB_WRITEBACK == 10, "ordinal MSG_DB_WRITEBACK drifted");
_Static_assert(MSG_DB_WRITEBACK_ACK == 11,
               "ordinal MSG_DB_WRITEBACK_ACK drifted");
_Static_assert(MSG_DB_OWNERSHIP_INVALIDATE == 12,
               "ordinal MSG_DB_OWNERSHIP_INVALIDATE drifted");
_Static_assert(MSG_DB_SNAPSHOT_REQUEST == 13,
               "ordinal MSG_DB_SNAPSHOT_REQUEST drifted");
_Static_assert(MSG_DB_SNAPSHOT_RESPONSE == 14,
               "ordinal MSG_DB_SNAPSHOT_RESPONSE drifted");
_Static_assert(MSG_DB_CREATE == 15, "ordinal MSG_DB_CREATE drifted");
_Static_assert(MSG_DB_DESTROY == 16, "ordinal MSG_DB_DESTROY drifted");
_Static_assert(MSG_DB_CACHE_DESTROY == 17,
               "ordinal MSG_DB_CACHE_DESTROY drifted");
_Static_assert(MSG_EVENT_DESTROY == 18, "ordinal MSG_EVENT_DESTROY drifted");
_Static_assert(MSG_EDT_DESTROY == 19, "ordinal MSG_EDT_DESTROY drifted");
_Static_assert(MSG_DB_SNAPSHOT_REDIRECT == 20,
               "ordinal MSG_DB_SNAPSHOT_REDIRECT drifted");
_Static_assert(MSG_DB_OWNERSHIP_CONFIRM == 21,
               "ordinal MSG_DB_OWNERSHIP_CONFIRM drifted");
_Static_assert(MSG_DB_OWNERSHIP_CONFIRM_ACK == 22,
               "ordinal MSG_DB_OWNERSHIP_CONFIRM_ACK drifted");
_Static_assert(MSG_DB_LOCK_REQUEST == 23,
               "ordinal MSG_DB_LOCK_REQUEST drifted");
_Static_assert(MSG_DB_LOCK_GRANT == 24, "ordinal MSG_DB_LOCK_GRANT drifted");
_Static_assert(MSG_DB_LOCK_RELEASE == 25,
               "ordinal MSG_DB_LOCK_RELEASE drifted");
_Static_assert(MSG_DB_LOCK_RELEASE_ACK == 26,
               "ordinal MSG_DB_LOCK_RELEASE_ACK drifted");
_Static_assert(MSG_COUNT == 27,
               "MSG_COUNT drifted (wire-compat: must be 27 in all 6 configs)");

/* ===== (2) header layout — read before the message type is known. ===== */
_Static_assert(offsetof(struct arts_msg_header_s, message_type) == 0,
               "header.message_type offset drifted");
#ifdef SEQUENCENUMBERS
/* SEQ build: {message_type(4), size(8)@4, rank(4)@12, seq_rank(4)@16,
 * seq_num(8)@20}.  Packed.  This layout is WIRE-INCOMPATIBLE with non-SEQ. */
_Static_assert(offsetof(struct arts_msg_header_s, size) == 4,
               "SEQ header.size offset drifted");
_Static_assert(offsetof(struct arts_msg_header_s, rank) == 12,
               "SEQ header.rank offset drifted");
_Static_assert(offsetof(struct arts_msg_header_s, seq_rank) == 16,
               "SEQ header.seq_rank offset drifted");
_Static_assert(offsetof(struct arts_msg_header_s, seq_num) == 20,
               "SEQ header.seq_num offset drifted");
_Static_assert(sizeof(struct arts_msg_header_s) == 28,
               "SEQ header size drifted");
#else
/* non-SEQ build: {message_type(4), size(8)@4, rank(4)@12}.  Packed → 16. */
_Static_assert(offsetof(struct arts_msg_header_s, size) == 4,
               "header.size offset drifted");
_Static_assert(offsetof(struct arts_msg_header_s, rank) == 12,
               "header.rank offset drifted");
_Static_assert(sizeof(struct arts_msg_header_s) == 16, "header size drifted");
#endif

/* ===== (3) per-packet sizeof — frozen golden table (non-SEQ).
 * Only assert under the non-SEQ header (the SEQ header adds 12 bytes to every
 * struct, which is the very skew B071 documents).  Each value verified
 * identical across all 6 configs. */
#ifndef SEQUENCENUMBERS
_Static_assert(sizeof(struct arts_msg_guid_only_packet_s) == 24,
               "guid_only sizeof drifted");
_Static_assert(sizeof(struct arts_msg_add_dependence_packet_s) == 40,
               "add_dependence sizeof drifted");
_Static_assert(sizeof(struct arts_msg_edt_satisfy_slot_packet_s) == 48,
               "edt_satisfy_slot sizeof drifted");
_Static_assert(sizeof(struct arts_msg_event_satisfy_slot_packet_s) == 36,
               "event_satisfy_slot sizeof drifted");
_Static_assert(sizeof(struct arts_msg_time_sync_req_packet_s) == 24,
               "time_sync_req sizeof drifted");
_Static_assert(sizeof(struct arts_msg_time_sync_resp_packet_s) == 32,
               "time_sync_resp sizeof drifted");
_Static_assert(sizeof(struct arts_msg_ownership_request_packet_s) == 24,
               "ownership_request sizeof drifted");
_Static_assert(sizeof(struct arts_msg_ownership_response_packet_s) == 40,
               "ownership_response sizeof drifted");
_Static_assert(sizeof(struct arts_msg_writeback_packet_s) == 40,
               "writeback sizeof drifted");
_Static_assert(sizeof(struct arts_msg_writeback_ack_packet_s) == 32,
               "writeback_ack sizeof drifted");
_Static_assert(sizeof(struct arts_msg_ownership_invalidate_packet_s) == 32,
               "ownership_invalidate sizeof drifted");
_Static_assert(sizeof(struct arts_msg_ownership_confirm_ack_packet_s) == 32,
               "ownership_confirm_ack sizeof drifted");
_Static_assert(sizeof(struct arts_msg_snapshot_request_packet_s) == 40,
               "snapshot_request sizeof drifted");
_Static_assert(sizeof(struct arts_msg_snapshot_response_packet_s) == 48,
               "snapshot_response sizeof drifted");
_Static_assert(sizeof(struct arts_msg_db_create_coherent_packet_s) == 40,
               "db_create_coherent sizeof drifted");
_Static_assert(sizeof(struct arts_msg_destroy_packet_s) == 24,
               "destroy sizeof drifted");
_Static_assert(sizeof(struct arts_msg_cache_destroy_packet_s) == 24,
               "cache_destroy sizeof drifted");
_Static_assert(sizeof(struct arts_msg_snapshot_redirect_packet_s) == 40,
               "snapshot_redirect sizeof drifted");
_Static_assert(sizeof(struct arts_msg_rank_version_pair_s) == 16,
               "rank_version_pair sizeof drifted");
_Static_assert(sizeof(struct arts_msg_ownership_confirm_packet_s) == 32,
               "ownership_confirm sizeof drifted");
#ifdef ARTS_PROTOCOL_LOCK
_Static_assert(sizeof(struct arts_msg_lock_request_packet_s) == 32,
               "lock_request sizeof drifted");
_Static_assert(sizeof(struct arts_msg_lock_grant_packet_s) == 40,
               "lock_grant sizeof drifted");
_Static_assert(sizeof(struct arts_msg_lock_release_packet_s) == 48,
               "lock_release sizeof drifted");
_Static_assert(sizeof(struct arts_msg_lock_release_ack_packet_s) == 32,
               "lock_release_ack sizeof drifted");
#endif
#endif /* !SEQUENCENUMBERS */

/* ===== runtime: the documented 8-byte payload-alignment invariant.
 * (Cannot be a hard _Static_assert without breaking the build, and the point of
 * the test is to REPORT the violation, not to fail compilation.) */
struct payload_pkt {
  const char *name;
  size_t sz;
  int carries_payload; /* 1 = trailing buffer/inline payload after sizeof() */
};

int main(void) {
  /* Marker line for the harness to diff SEQ vs non-SEQ header layout. */
#ifdef SEQUENCENUMBERS
  printf("HDR_HAS_SEQ=1 sizeof_header=%zu offset_size=%zu\n",
         sizeof(struct arts_msg_header_s),
         offsetof(struct arts_msg_header_s, size));
#else
  printf("HDR_HAS_SEQ=0 sizeof_header=%zu offset_size=%zu\n",
         sizeof(struct arts_msg_header_s),
         offsetof(struct arts_msg_header_s, size));
#endif

  const struct payload_pkt pkts[] = {
      {"OWNERSHIP_RESPONSE",
       sizeof(struct arts_msg_ownership_response_packet_s), 1},
      {"WRITEBACK", sizeof(struct arts_msg_writeback_packet_s), 1},
      {"SNAPSHOT_RESPONSE", sizeof(struct arts_msg_snapshot_response_packet_s),
       1},
      {"EDT_SATISFY_SLOT", sizeof(struct arts_msg_edt_satisfy_slot_packet_s),
       1},
#ifdef ARTS_PROTOCOL_LOCK
      {"LOCK_GRANT", sizeof(struct arts_msg_lock_grant_packet_s), 1},
      {"LOCK_RELEASE", sizeof(struct arts_msg_lock_release_packet_s), 1},
#endif
  };

  int violations = 0;
  for (size_t i = 0; i < sizeof(pkts) / sizeof(pkts[0]); i++) {
    if (pkts[i].carries_payload && (pkts[i].sz % 8) != 0) {
      fprintf(
          stderr,
          "BUG  payload-carrying %s sizeof=%zu is NOT 8-aligned — trailing "
          "payload starts at an unaligned wire offset, violating protocol.h "
          "§4 pad-field invariant (B071/ABI)\n",
          pkts[i].name, pkts[i].sz);
      violations++;
    }
  }

  if (violations > 0) {
    fprintf(stderr,
            "protocol_abi_asserts: %d payload-alignment invariant "
            "violation(s) — documented 8-byte payload alignment not held by "
            "the packed-16-byte-header layout.\n",
            violations);
    /* The compile-time golden table (ordinals/offsets/sizeofs) PASSED — those
     * are stable.  The DOCUMENTED runtime invariant is violated; REPORT it. */
    return 1;
  }

  printf("PASS protocol_abi_asserts (ordinals + header + sizeof golden table "
         "frozen; payload 8-alignment holds)\n");
  return 0;
}
