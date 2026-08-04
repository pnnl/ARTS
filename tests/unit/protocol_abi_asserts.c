/* SPDX-License-Identifier: Apache-2.0
 *
 * T180 — wire-protocol ABI frozen golden table (B071, flags B072).
 *
 * Property under test
 * -------------------
 * protocol.h is the cross-rank wire contract.  Two ranks built in DIFFERENT
 * coherence configs (the 7 supported: VAL+HOME, VAL+OWNER, INV+HOME, INV+OWNER,
 * WRF_VAL, EXCL+HOME, EXCL+OWNER) MUST agree byte-for-byte on:
 *   (1) `enum arts_msg_type` — every ordinal contiguous 0..MSG_COUNT-1, and
 *       MSG_COUNT itself, IDENTICAL across all 7 configs (the enum members are
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
 * IDENTICAL across all 7 configs (only the EXCL-only structs, and the
 * EXCL+OWNER-only subset within them, differ in presence, never in the shared
 * ordinals/offsets/sizes).  This TU is meant to be COMPILED ONCE PER
 * -DARTS_PROTOCOL_* config; the `_Static_assert`s catch any config that drifts
 * from the golden table at compile time.
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
 * Runtime portion — the 8-byte payload-alignment INVARIANT
 * ---------------------------------------------------------------------------
 * protocol.h §4 documents a pad-field invariant: "Pad-fields exist to keep the
 * trailing payload on an 8-byte boundary ... the payload starts at sizeof() —
 * that offset must be 8-aligned."  The payload-carrying structs are
 * OWNERSHIP_RESPONSE, PUBLISH, SNAPSHOT_RESPONSE, EDT_SATISFY_SLOT (for
 * DB_MODE_PTR), and — in EXCL builds — LOCK_GRANT, LOCK_RELEASE, and (EXCL+OWNER
 * only) LOCK_DELIVER.  This TU checks that sizeof() of each is a multiple of 8
 * at runtime (the invariant spans a pad field whose width is itself derived
 * from other fields, so it is not expressible as a single `_Static_assert`).
 * All payload-carrying structs currently HOLD the invariant — the check exists
 * to catch a future regression (e.g. a field added to a payload struct without
 * a compensating pad), not to report a live defect.
 */

#include "arts/transport/protocol.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* ===== (1) enum ordinals — frozen golden table, identical across all 7
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
_Static_assert(MSG_DB_GRANT_REQUEST == 8,
               "ordinal MSG_DB_GRANT_REQUEST drifted");
_Static_assert(MSG_DB_GRANT_RESPONSE == 9,
               "ordinal MSG_DB_GRANT_RESPONSE drifted");
_Static_assert(MSG_DB_PUBLISH == 10, "ordinal MSG_DB_PUBLISH drifted");
_Static_assert(MSG_DB_PUBLISH_ACK == 11,
               "ordinal MSG_DB_PUBLISH_ACK drifted");
_Static_assert(MSG_DB_GRANT_INVALIDATE == 12,
               "ordinal MSG_DB_GRANT_INVALIDATE drifted");
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
_Static_assert(MSG_DB_GRANT_CONFIRM == 21,
               "ordinal MSG_DB_GRANT_CONFIRM drifted");
_Static_assert(MSG_DB_GRANT_CONFIRM_ACK == 22,
               "ordinal MSG_DB_GRANT_CONFIRM_ACK drifted");
_Static_assert(MSG_DB_EXCL_REQUEST == 23,
               "ordinal MSG_DB_EXCL_REQUEST drifted");
_Static_assert(MSG_DB_EXCL_GRANT == 24, "ordinal MSG_DB_EXCL_GRANT drifted");
_Static_assert(MSG_DB_EXCL_RELEASE == 25,
               "ordinal MSG_DB_EXCL_RELEASE drifted");
_Static_assert(MSG_DB_EXCL_RELEASE_ACK == 26,
               "ordinal MSG_DB_EXCL_RELEASE_ACK drifted");
_Static_assert(MSG_DB_EXCL_FORWARD == 27,
               "ordinal MSG_DB_EXCL_FORWARD drifted");
_Static_assert(MSG_DB_EXCL_DELIVER == 28,
               "ordinal MSG_DB_EXCL_DELIVER drifted");
_Static_assert(MSG_DB_EXCL_CONFIRM == 29,
               "ordinal MSG_DB_EXCL_CONFIRM drifted");
_Static_assert(MSG_DB_EXCL_RORET == 30, "ordinal MSG_DB_EXCL_RORET drifted");
_Static_assert(MSG_DB_GRANT_CTS == 31,
               "ordinal MSG_DB_GRANT_CTS drifted");
_Static_assert(MSG_DB_PUBLISH_CTS == 32,
               "ordinal MSG_DB_PUBLISH_CTS drifted");
_Static_assert(MSG_DB_EXCL_CTS == 33, "ordinal MSG_DB_EXCL_CTS drifted");
_Static_assert(MSG_RDZV_PUSH_RTS == 34, "ordinal MSG_RDZV_PUSH_RTS drifted");
_Static_assert(MSG_RDZV_PUSH_CTS == 35, "ordinal MSG_RDZV_PUSH_CTS drifted");
_Static_assert(MSG_DB_INV_REQUEST == 36, "ordinal MSG_DB_INV_REQUEST drifted");
_Static_assert(MSG_DB_INV_CTS == 37, "ordinal MSG_DB_INV_CTS drifted");
_Static_assert(MSG_DB_INV_DELIVER == 38, "ordinal MSG_DB_INV_DELIVER drifted");
_Static_assert(MSG_DB_INV_INVALIDATE == 39,
               "ordinal MSG_DB_INV_INVALIDATE drifted");
_Static_assert(MSG_DB_INV_INVALIDATE_ACK == 40,
               "ordinal MSG_DB_INV_INVALIDATE_ACK drifted");
_Static_assert(MSG_DB_INV_REDIRECT == 41,
               "ordinal MSG_DB_INV_REDIRECT drifted");
_Static_assert(MSG_COUNT == 42,
               "MSG_COUNT drifted (wire-compat: must be 42 in all configs)");

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
 * identical across all 7 configs. */
#ifndef SEQUENCENUMBERS
_Static_assert(sizeof(struct arts_msg_guid_only_packet_s) == 24,
               "guid_only sizeof drifted");
_Static_assert(sizeof(struct arts_msg_memory_move_packet_s) == 48,
               "memory_move sizeof drifted");
_Static_assert(sizeof(struct arts_msg_add_dependence_packet_s) == 40,
               "add_dependence sizeof drifted");
_Static_assert(sizeof(struct arts_msg_edt_satisfy_slot_packet_s) == 64,
               "edt_satisfy_slot sizeof drifted");
_Static_assert(sizeof(struct arts_msg_event_satisfy_slot_packet_s) == 36,
               "event_satisfy_slot sizeof drifted");
_Static_assert(sizeof(struct arts_msg_time_sync_req_packet_s) == 24,
               "time_sync_req sizeof drifted");
_Static_assert(sizeof(struct arts_msg_time_sync_resp_packet_s) == 32,
               "time_sync_resp sizeof drifted");
_Static_assert(sizeof(struct arts_msg_grant_request_packet_s) == 56,
               "ownership_request sizeof drifted");
_Static_assert(sizeof(struct arts_msg_grant_response_packet_s) == 64,
               "ownership_response sizeof drifted");
_Static_assert(sizeof(struct arts_msg_publish_packet_s) == 64,
               "publish sizeof drifted");
_Static_assert(sizeof(struct arts_msg_publish_ack_packet_s) == 32,
               "publish_ack sizeof drifted");
_Static_assert(sizeof(struct arts_msg_grant_invalidate_packet_s) == 64,
               "ownership_invalidate sizeof drifted");
_Static_assert(sizeof(struct arts_msg_grant_confirm_ack_packet_s) == 64,
               "ownership_confirm_ack sizeof drifted");
_Static_assert(sizeof(struct arts_msg_snapshot_request_packet_s) == 72,
               "snapshot_request sizeof drifted");
_Static_assert(sizeof(struct arts_msg_snapshot_response_packet_s) == 72,
               "snapshot_response sizeof drifted");
_Static_assert(sizeof(struct arts_msg_db_create_coherent_packet_s) == 40,
               "db_create_coherent sizeof drifted");
_Static_assert(sizeof(struct arts_msg_destroy_packet_s) == 24,
               "destroy sizeof drifted");
_Static_assert(sizeof(struct arts_msg_cache_destroy_packet_s) == 24,
               "cache_destroy sizeof drifted");
_Static_assert(sizeof(struct arts_msg_snapshot_redirect_packet_s) == 72,
               "snapshot_redirect sizeof drifted");
_Static_assert(sizeof(struct arts_msg_rank_version_pair_s) == 16,
               "rank_version_pair sizeof drifted");
_Static_assert(sizeof(struct arts_msg_rdzv_landing_s) == 32,
               "rdzv_landing sizeof drifted");
_Static_assert(sizeof(struct arts_msg_grant_cts_packet_s) == 32,
               "ownership_cts sizeof drifted");
_Static_assert(sizeof(struct arts_msg_publish_cts_packet_s) == 64,
               "publish_cts sizeof drifted");
_Static_assert(sizeof(struct arts_msg_rdzv_push_rts_packet_s) == 32,
               "rdzv_push_rts sizeof drifted");
_Static_assert(sizeof(struct arts_msg_rdzv_push_cts_packet_s) == 56,
               "rdzv_push_cts sizeof drifted");
_Static_assert(sizeof(struct arts_msg_grant_confirm_packet_s) == 32,
               "ownership_confirm sizeof drifted");
#ifdef ARTS_PROTOCOL_EXCL
_Static_assert(sizeof(struct arts_msg_excl_request_packet_s) == 64,
               "lock_request sizeof drifted");
_Static_assert(sizeof(struct arts_msg_excl_grant_packet_s) == 96,
               "lock_grant sizeof drifted");
_Static_assert(sizeof(struct arts_msg_excl_release_packet_s) == 72,
               "lock_release sizeof drifted");
_Static_assert(sizeof(struct arts_msg_excl_release_ack_packet_s) == 32,
               "lock_release_ack sizeof drifted");
_Static_assert(sizeof(struct arts_msg_excl_cts_packet_s) == 40,
               "lock_cts sizeof drifted");
#ifdef ARTS_RELEASE_RETAIN
_Static_assert(sizeof(struct arts_msg_excl_forward_packet_s) == 64,
               "lock_forward sizeof drifted");
_Static_assert(sizeof(struct arts_msg_excl_deliver_packet_s) == 56,
               "lock_deliver sizeof drifted");
_Static_assert(sizeof(struct arts_msg_excl_confirm_packet_s) == 24,
               "lock_confirm sizeof drifted");
#endif /* ARTS_RELEASE_RETAIN */
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
       sizeof(struct arts_msg_grant_response_packet_s), 1},
      {"PUBLISH", sizeof(struct arts_msg_publish_packet_s), 1},
      {"SNAPSHOT_RESPONSE", sizeof(struct arts_msg_snapshot_response_packet_s),
       1},
      {"EDT_SATISFY_SLOT", sizeof(struct arts_msg_edt_satisfy_slot_packet_s),
       1},
#ifdef ARTS_PROTOCOL_EXCL
      {"EXCL_GRANT", sizeof(struct arts_msg_excl_grant_packet_s), 1},
      {"EXCL_RELEASE", sizeof(struct arts_msg_excl_release_packet_s), 1},
#ifdef ARTS_RELEASE_RETAIN
      {"EXCL_DELIVER", sizeof(struct arts_msg_excl_deliver_packet_s), 1},
#endif
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
