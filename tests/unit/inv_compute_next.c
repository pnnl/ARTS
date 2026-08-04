/* SPDX-License-Identifier: Apache-2.0
 *
 * INV inv_cache_compute_next / inv_dir_compute_next — the PURE transition
 * functions over the packed reader word [ro|inflight|head_ro] and the packed
 * directory word [round_open|acks].
 *
 * Both arbiters are identical under either placement (placement decides where
 * bytes come FROM, never what a word does), so this one truth table covers the
 * whole protocol.  Expected values are hand-derived, not produced by an oracle
 * that re-implements the functions.  Covered:
 *
 *   cache (reader plane):
 *   - a covering copy makes a read SELF_SERVE with the word UNCHANGED — the
 *     property that makes a read on a valid copy wait-free;
 *   - opening a fetch and joining one are the SAME decision CAS that links
 *     self_idx as the chain head, so a parked node cannot miss what it joined;
 *   - a landed reply grabs the whole chain in that same atom (head -> 0);
 *   - the kill mark (REQ -> REQ_KILL) still serves its cohort once, then owes
 *     the round an ack;
 *   - a reply with no fetch open is a superseded orphan and DROPs;
 *
 *   dir (invalidation round):
 *   - round claim mutual exclusion;
 *   - ack arming, decrement, and the unique 0-edge close;
 *   - close drops the claim and nothing else.
 *
 * There is deliberately NO writer coverage: write ownership is the migrating
 * sentinel grant, which lives in plain counters (coherence/grant.c), not in
 * either word.
 *
 * Built standalone by #including coherence/inv/arbiters.c.  INV-only;
 * self-skips elsewhere.
 */

#include <stdio.h>

#if !defined(ARTS_PROTOCOL_INV)
int main(void) {
  printf("PASS inv_compute_next: skipped (INV only; the packed cache/dir "
         "words exist only in that build)\n");
  return 0;
}
#else

#include "arts/coherence/inv/types.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── word-packing arithmetic (compile-time pins) ───────────────────────── */
_Static_assert(MSI_CACHE_RO_SHIFT + MSI_CACHE_ST_BITS == 64,
               "cache word must pack to exactly 64 bits");
_Static_assert(MSI_CACHE_HEAD_RO_SHIFT == 0 && MSI_CACHE_INFLIGHT_SHIFT == 61 &&
                   MSI_CACHE_RO_SHIFT == 62,
               "cache word field shifts drifted");
_Static_assert(MSI_DIR_ROUND_SHIFT == 63 && MSI_DIR_ACKS_SHIFT == 49,
               "dir word field shifts drifted");

#include "core/coherence/inv/arbiters.c"

static int g_fail;

static void expect_cache(const char *what, uint64_t cur, int op, uint32_t idx,
                         uint64_t want_next, uint32_t want_act) {
  uint32_t act = 0xFFFFFFFFu;
  uint64_t next = inv_cache_compute_next(cur, op, idx, &act);
  if (next != want_next || act != want_act) {
    (void)fprintf(stderr,
                  "FAIL inv_compute_next: %s — got word %016llx act %u, "
                  "want word %016llx act %u\n",
                  what, (unsigned long long)next, act,
                  (unsigned long long)want_next, want_act);
    g_fail = 1;
  }
}

static void expect_dir(const char *what, uint64_t cur, int op, unsigned int arg,
                       uint64_t want_next, uint32_t want_act) {
  uint32_t act = 0xFFFFFFFFu;
  uint64_t next = inv_dir_compute_next(cur, op, arg, &act);
  if (next != want_next || act != want_act) {
    (void)fprintf(stderr,
                  "FAIL inv_compute_next: %s — got word %016llx act %u, "
                  "want word %016llx act %u\n",
                  what, (unsigned long long)next, act,
                  (unsigned long long)want_next, want_act);
    g_fail = 1;
  }
}

int main(void) {
  const uint32_t ME = 7u;   /* this acquire's pool node */
  const uint32_t OTH = 12u; /* a node already on the chain */

  /* ---- reader plane ---------------------------------------------------- */

  /* A covering copy is served with NO word change: this is the property that
   * makes a read on a valid copy wait-free. */
  {
    uint64_t w = MSI_CACHE_MAKE(MSI_RO_VALID, 0u, 0u);
    expect_cache("valid copy read is a no-op self-serve", w,
                 MSI_CACHE_OP_ACQ_RO, ME, w, MSI_CACHE_ACT_SELF_SERVE);
  }

  /* No copy, no fetch: open one AND park in the same transition. */
  expect_cache("first read opens the fetch and parks",
               MSI_CACHE_MAKE(MSI_RO_IDLE, 0u, 0u), MSI_CACHE_OP_ACQ_RO, ME,
               MSI_CACHE_MAKE(MSI_RO_REQ, 1u, ME), MSI_CACHE_ACT_SEND_RO);

  /* A fetch is open: join it — same CAS, head replaced, no second message. */
  expect_cache("second read joins the open fetch",
               MSI_CACHE_MAKE(MSI_RO_REQ, 1u, OTH), MSI_CACHE_OP_ACQ_RO, ME,
               MSI_CACHE_MAKE(MSI_RO_REQ, 1u, ME), MSI_CACHE_ACT_PARK);

  /* A kill-marked fetch is still joinable: its reply serves the cohort once
   * before the reserved purge, so nobody is stranded. */
  expect_cache("read joins a kill-marked fetch",
               MSI_CACHE_MAKE(MSI_RO_REQ_KILL, 1u, OTH), MSI_CACHE_OP_ACQ_RO,
               ME, MSI_CACHE_MAKE(MSI_RO_REQ_KILL, 1u, ME),
               MSI_CACHE_ACT_PARK);

  /* The reply publishes and grabs the whole chain in one atom. */
  expect_cache("reply publishes and grabs the chain",
               MSI_CACHE_MAKE(MSI_RO_REQ, 1u, ME), MSI_CACHE_OP_DELIVER, 0u,
               MSI_CACHE_MAKE(MSI_RO_VALID, 0u, 0u), MSI_CACHE_ACT_PUBLISH);

  /* A doomed reply serves its cohort, then owes the purge + ack. */
  expect_cache("kill-marked reply serves once, then purges",
               MSI_CACHE_MAKE(MSI_RO_REQ_KILL, 1u, ME), MSI_CACHE_OP_DELIVER,
               0u, MSI_CACHE_MAKE(MSI_RO_VALID, 0u, 0u),
               MSI_CACHE_ACT_PUBLISH_KILL);

  /* A reply with no fetch open is an orphan. */
  expect_cache("orphan reply drops", MSI_CACHE_MAKE(MSI_RO_IDLE, 0u, 0u),
               MSI_CACHE_OP_DELIVER, 0u, MSI_CACHE_MAKE(MSI_RO_IDLE, 0u, 0u),
               MSI_CACHE_ACT_DROP);

  /* Invalidate: retire a copy, mark a fetch, or be idempotent. */
  expect_cache("invalidate retires a valid copy",
               MSI_CACHE_MAKE(MSI_RO_VALID, 0u, 0u), MSI_CACHE_OP_INVALIDATE,
               0u, MSI_CACHE_MAKE(MSI_RO_IDLE, 0u, 0u),
               MSI_CACHE_ACT_PURGE_ACK);
  expect_cache("invalidate marks an open fetch",
               MSI_CACHE_MAKE(MSI_RO_REQ, 1u, ME), MSI_CACHE_OP_INVALIDATE, 0u,
               MSI_CACHE_MAKE(MSI_RO_REQ_KILL, 1u, ME),
               MSI_CACHE_ACT_KILL_MARKED);
  expect_cache("invalidate on nothing is idempotent",
               MSI_CACHE_MAKE(MSI_RO_IDLE, 0u, 0u), MSI_CACHE_OP_INVALIDATE,
               0u, MSI_CACHE_MAKE(MSI_RO_IDLE, 0u, 0u),
               MSI_CACHE_ACT_NOOP_ACK);
  expect_cache("the reserved purge retires the served copy",
               MSI_CACHE_MAKE(MSI_RO_VALID, 0u, 0u), MSI_CACHE_OP_KILL_PURGE,
               0u, MSI_CACHE_MAKE(MSI_RO_IDLE, 0u, 0u), MSI_CACHE_ACT_NONE);

  /* ---- invalidation round ---------------------------------------------- */

  expect_dir("an idle directory admits a round", MSI_DIR_MAKE(0u, 0u),
             MSI_DIR_OP_ROUND_CLAIM, 0u, MSI_DIR_MAKE(1u, 0u),
             MSI_DIR_ACT_CLAIMED);
  expect_dir("a busy directory refuses a second round", MSI_DIR_MAKE(1u, 3u),
             MSI_DIR_OP_ROUND_CLAIM, 0u, MSI_DIR_MAKE(1u, 3u),
             MSI_DIR_ACT_NONE);
  expect_dir("arming sets the snapshot size", MSI_DIR_MAKE(1u, 0u),
             MSI_DIR_OP_ACKS_ARM, 5u, MSI_DIR_MAKE(1u, 5u), MSI_DIR_ACT_NONE);
  expect_dir("a non-final ack just decrements", MSI_DIR_MAKE(1u, 5u),
             MSI_DIR_OP_ACK_DEC, 0u, MSI_DIR_MAKE(1u, 4u), MSI_DIR_ACT_NONE);
  expect_dir("the last ack closes, and only it", MSI_DIR_MAKE(1u, 1u),
             MSI_DIR_OP_ACK_DEC, 0u, MSI_DIR_MAKE(1u, 0u), MSI_DIR_ACT_CLOSE);
  expect_dir("close drops the claim and nothing else", MSI_DIR_MAKE(1u, 0u),
             MSI_DIR_OP_ROUND_CLOSE, 0u, MSI_DIR_MAKE(0u, 0u),
             MSI_DIR_ACT_NONE);

  if (g_fail) {
    return 1;
  }
  printf("PASS inv_compute_next: reader plane + invalidation round truth "
         "table\n");
  return 0;
}

#endif /* ARTS_PROTOCOL_INV */
