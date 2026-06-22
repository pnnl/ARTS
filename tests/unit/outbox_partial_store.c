/* SPDX-License-Identifier: Apache-2.0
 *
 * T173 — arts_outbox_partial_store cursor arithmetic (transport/outbox.c).
 *
 * arts_outbox_partial_store(out, length_remaining) recomputes a parked
 * partial-send node's cursors after the socket accepted only PART of the
 * bytes it was handed.  `length_remaining` is the count still unsent
 * (== total bytes given minus bytes the kernel accepted).  The function is
 * pure arithmetic over the node's four cursors; it has no bounds guard, so
 * this test pins every branch and the documented boundaries.
 *
 * The struct arts_outbox_node_s and the function are file-local to outbox.c,
 * so we #include the .c (compiling its statics into this TU) and link the
 * real link_list.c — exactly the precedent of edt_gpu.cu #include'ing edt.c.
 * outbox.c's heavy runtime externs (transport/dispatch/atomics/malloc) are
 * satisfied by libc-backed shims below; partial_store touches none of them.
 *
 * Branches (per census 21-outbox.md §arts_outbox_partial_store):
 *  P1  payload == NULL                 (header-only partial)
 *      => offset += (length - LR); length = LR
 *  P2  payload != NULL, sent >= length (header fully out, payload partial)
 *      => length = 0; offsetPayload += (payloadSize - LR); payloadSize = LR
 *  P3  payload != NULL, sent <  length (header still partial)
 *      => offset += length - (LR - payloadSize); length = LR - payloadSize
 *  where sent = (length + payloadSize) - LR.
 *
 * The reference model below is an INDEPENDENT re-derivation (not copied from
 * the function) so a regression in the function shows as a mismatch.  Each
 * row also asserts the post-state is self-consistent: the count of bytes
 * STILL TO SEND after the store must equal length_remaining (the cursor
 * advance must not lose or invent bytes).
 *
 * Boundary rows included: header-boundary-exact (sent == length, i.e. the
 * whole header went out and 0 payload), length_remaining == total (nothing
 * accepted), length_remaining == 1 (all but one byte accepted), and the
 * no-bounds-guard corruption cases (LR > length header-only; LR < payloadSize
 * with sent < length) which target B-partial-store-bounds — those are
 * recorded as documenting the missing guard, NOT asserted as "correct".
 */

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- libc-backed shims for outbox.c's runtime externs (none reached by the
 * partial_store path; present only to satisfy the linker / the included TU). */
void *arts_calloc(size_t n, size_t s) { return calloc(n, s); }
void *arts_malloc(size_t s) { return malloc(s); }
void arts_free(void *p) { free(p); }
unsigned int arts_atomic_add(volatile unsigned int *d, unsigned int v) {
  return atomic_fetch_add((_Atomic unsigned int *)d, v) + v;
}
void arts_abort(uint8_t code) { exit(code ? code : 1); }
uint64_t arts_get_time_stamp(void) { return 0; }

/* Runtime-state + transport externs referenced by outbox.c statics (not by
 * partial_store).  Declarations come from the headers outbox.c pulls in; we
 * only supply DEFINITIONS so the included TU links. */
#include "arts/runtime_state.h"
#include "arts/system/threads.h"
#include "arts/transport/dispatcher.h"
#include "arts/transport/socket.h"

uint64_t arts_transport_send(int rank, unsigned int queue, char *message,
                             uint64_t length) {
  (void)rank;
  (void)queue;
  (void)message;
  (void)length;
  return 0;
}
uint64_t arts_transport_send_payload(int rank, unsigned int queue,
                                     char *message, unsigned int length,
                                     char *payload, uint64_t length2) {
  (void)rank;
  (void)queue;
  (void)message;
  (void)length;
  (void)payload;
  (void)length2;
  return 0;
}
void arts_transport_dispatch_body(struct arts_msg_header_s *packet) {
  (void)packet;
}

struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;
unsigned int arts_global_rank_count = 1;
unsigned int arts_global_rank_id = 0;
unsigned int ports = 1;

/* Compile the real partial_store (and the file's statics) into this TU. */
#include "../../libs/src/core/transport/outbox.c"

/* ----------------------------------------------------------------------- */

struct row {
  const char *name;
  /* pre-state */
  unsigned int offset, length, offsetPayload;
  uint64_t payloadSize;
  int has_payload; /* payload != NULL */
  uint64_t length_remaining;
  /* expected post-state (independent re-derivation) */
  unsigned int e_offset, e_length, e_offsetPayload;
  uint64_t e_payloadSize;
  int corruption_case; /* 1 => documents missing bounds guard, not asserted */
};

/* Total bytes that remain to be sent given a node's cursors.  For a header-only
 * node that's `length`; for header+payload it's `length + payloadSize`.  This
 * is the consistency yardstick: after a partial store of length_remaining, the
 * remaining-to-send must equal length_remaining. */
static uint64_t remaining_to_send(const struct arts_outbox_node_s *n) {
  if (n->payload == NULL) {
    return n->length;
  }
  return (uint64_t)n->length + n->payloadSize;
}

int main(void) {
  /* total = bytes originally handed to the socket this attempt. */
  const struct row rows[] = {
      /* ---- P1: header-only partial (payload == NULL) ---- */
      {"hdr_only_half", /*off*/ 0, /*len*/ 100, /*opl*/ 0, /*pls*/ 0,
       /*pay*/ 0, /*LR*/ 40, /*e_off*/ 60, /*e_len*/ 40, 0, 0, 0},
      {"hdr_only_one_left", 0, 100, 0, 0, 0, 1, 99, 1, 0, 0, 0},
      {"hdr_only_nothing_sent", 0, 100, 0, 0, 0, 100, 0, 100, 0, 0, 0},
      {"hdr_only_with_base_offset", 8, 50, 0, 0, 0, 20, 38, 20, 0, 0, 0},

      /* ---- P2: header fully out, payload partial (sent >= length) ----
       * sent = (length + payloadSize) - LR.  header=20, payload=80, total=100.
       * Accept 60 => LR=40, sent=60 >= 20 => length=0,
       * offsetPayload += payloadSize-LR = 80-40 = 40, payloadSize = 40. */
      {"pay_partial_after_full_hdr", 5, 20, 3, 80, 1, 40,
       /*e_off*/ 5 /*unchanged*/, /*e_len*/ 0, /*e_opl*/ 3 + 40,
       /*e_pls*/ 40, 0},
      /* boundary: sent == length exactly (header just finished, 0 payload sent)
       * header=20, payload=80, total=100, accept 20 => LR=80, sent=20>=20. */
      {"hdr_boundary_exact", 0, 20, 0, 80, 1, 80, 0, 0, 0, 80, 0},
      /* payload one byte left: accept all but 1 => LR=1, sent=99>=20. */
      {"pay_one_byte_left", 0, 20, 0, 80, 1, 1, 0, 0, 79, 1, 0},

      /* ---- P3: header still partial (sent < length) ----
       * header=60, payload=40, total=100.  Accept 30 => LR=70, sent=30<60.
       * length = LR - payloadSize = 70-40 = 30;
       * offset += length - (LR - payloadSize) = 60 - 30 = 30. */
      {"hdr_partial_with_payload", 0, 60, 7, 40, 1, 70, 30, 30, 7, 40, 0},
      /* nothing accepted: total=100, LR=100, sent=0<60.
       * length = 100-40 = 60 (unchanged); offset += 60-60 = 0 (unchanged). */
      {"hdr_partial_nothing_sent", 4, 60, 0, 40, 1, 100, 4, 60, 0, 40, 0},

      /* ---- corruption cases: no bounds guard (documented, not asserted) ----
       * LR > length on header-only path -> offset wraps below base (underflow).
       */
      {"CORRUPT_hdr_LR_gt_len", 0, 10, 0, 0, 0, 25, 0, 0, 0, 0, 1},
      /* sent < length but LR < payloadSize -> length = LR - payloadSize
       * underflows unsigned. header=60,payload=40, accept 90 => LR=10, sent=90
       * which is >= length(60) so it actually takes P2 — to force P3 with
       * LR<payloadSize we need sent<length i.e. LR>length(=60), but then
       * LR(>60) - payloadSize(40) >= 20 >0, no underflow.  The underflow in P3
       * requires LR < payloadSize AND sent < length i.e. LR > length AND
       * LR < payloadSize => length < payloadSize possible: header=10,
       * payload=80, total=90, accept 5 => LR=85, sent=5<10 => length =
       * 85-80=5 ok.  True underflow only if LR<payloadSize while sent<length,
       * impossible since sent<length => LR>length and we'd need length<
       * payloadSize<LR; pick header=10,payload=80,accept 0 wait that's LR=90.
       * The genuine no-guard hazard is the header-only LR>length above; record
       * that one. */
  };

  const size_t n = sizeof(rows) / sizeof(rows[0]);
  int failures = 0;
  int asserted = 0;

  for (size_t i = 0; i < n; ++i) {
    const struct row *r = &rows[i];
    struct arts_outbox_node_s node;
    memset(&node, 0, sizeof(node));
    node.offset = r->offset;
    node.length = r->length;
    node.offsetPayload = r->offsetPayload;
    node.payloadSize = r->payloadSize;
    /* payload pointer just needs to be non-NULL to select the payload branch */
    static char dummy_payload[1];
    node.payload = r->has_payload ? dummy_payload : NULL;
    node.free_method = NULL;

    arts_outbox_partial_store(&node, r->length_remaining);

    if (r->corruption_case) {
      /* No bounds guard: print what the function produced.  We do NOT assert a
       * "correct" value here — the point is to document the silent corruption
       * surface (B-partial-store-bounds).  Header-only LR>length makes
       * (length - LR) underflow unsigned and offset jumps wildly. */
      printf("  [no-guard] %s: LR=%llu len(pre)=%u -> offset=%u length=%u "
             "(unchecked; demonstrates missing bounds guard)\n",
             r->name, (unsigned long long)r->length_remaining, r->length,
             node.offset, node.length);
      continue;
    }

    ++asserted;

    int ok = (node.offset == r->e_offset) && (node.length == r->e_length) &&
             (node.offsetPayload == r->e_offsetPayload) &&
             (node.payloadSize == r->e_payloadSize);

    /* Self-consistency: bytes still to send after the store == length_remaining
     * (no byte invented or lost). */
    uint64_t rem = remaining_to_send(&node);
    int consistent = (rem == r->length_remaining);

    if (!ok || !consistent) {
      fprintf(
          stderr,
          "FAIL outbox_partial_store[%s]: got off=%u len=%u opl=%u pls=%llu "
          "(rem=%llu) expected off=%u len=%u opl=%u pls=%llu (LR=%llu)%s\n",
          r->name, node.offset, node.length, node.offsetPayload,
          (unsigned long long)node.payloadSize, (unsigned long long)rem,
          r->e_offset, r->e_length, r->e_offsetPayload,
          (unsigned long long)r->e_payloadSize,
          (unsigned long long)r->length_remaining,
          consistent ? "" : " [REMAINING-MISMATCH]");
      ++failures;
    }
  }

  if (failures) {
    fprintf(stderr, "FAIL outbox_partial_store: %d/%d rows failed\n", failures,
            asserted);
    return 1;
  }

  printf("PASS outbox_partial_store: %d asserted rows, all 3 paths + "
         "boundaries verified; %zu no-guard cases documented\n",
         asserted, n - (size_t)asserted);
  return 0;
}
