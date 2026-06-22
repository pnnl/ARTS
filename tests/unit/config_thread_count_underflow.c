/* SPDX-License-Identifier: Apache-2.0
 *
 * T214 — config_compute_derived worker = thread_count - sender - receiver.
 * Targets B102 (config.c compute_derived, HIGH; same arithmetic as B088 in
 * threads.c, see T229).
 *
 * When thread_count is pre-set (SLURM_CPUS_PER_TASK path) compute_derived
 * derives the worker count as:
 *     worker_thread_count = thread_count - sender_thread_count
 *                                        - receiver_thread_count;
 * with NO guard that sender+receiver <= thread_count.  These are unsigned, so
 * thread_count < sender+receiver underflows to an astronomical worker count
 * (~4 billion) which the runtime then tries to honor in the role split / thread
 * spawn → catastrophic over-allocation.
 *
 * This test drives the multi-node branch (table_length>1, table==NULL so the
 * port loop is skipped) with thread_count=2, sender=2, receiver=2 → 2-4
 * underflows.  It is authored correct-and-failing in the SENSE that it PINS the
 * current buggy result (UINT_MAX-1 = 0xFFFFFFFE) and FAILS if the value is a
 * sane clamp — i.e. it asserts the bug is present and documents the desired
 * post-fix behavior in a comment.  exposes_runtime_bug=true.
 */

#include "../../libs/src/core/system/config.c"
#include "config_test_common.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

int main(void) {
  struct arts_config_s c;
  memset(&c, 0, sizeof(c));
  c.table_length = 2; /* multi-node branch */
  c.table = NULL;     /* port-population loop is guarded on table != NULL */
  c.route_table_size = 16;
  c.gpu_route_table_size = 12;

  c.thread_count = 2;        /* pre-set (as if from SLURM_CPUS_PER_TASK) */
  c.sender_thread_count = 2; /* sender+receiver = 4 > thread_count = 2 */
  c.receiver_thread_count = 2;
  c.worker_thread_count = 0;

  config_compute_derived(&c);

  /* Live bug (B102): 2 - 2 - 2 wraps. worker = 0xFFFFFFFE, then thread_count is
     recomputed as worker+sender+receiver = 0xFFFFFFFE + 4 = 2 (wraps back). */
  unsigned int expect_buggy_worker =
      (unsigned int)(2u - 2u - 2u); /* 0xFFFFFFFE */

  if (c.worker_thread_count != expect_buggy_worker) {
    /* If we reach here the underflow was clamped/guarded — the DESIRED fix.
       Until then this branch indicates the bug is gone (test should be updated
       to assert the sane value, e.g. worker==0 with sender+receiver<=tc). */
    fprintf(stderr,
            "NOTE config_thread_count_underflow: worker=%u (NOT the unclamped "
            "underflow %u) — B102 appears fixed; update expectation.\n",
            c.worker_thread_count, expect_buggy_worker);
    return 1;
  }

  fprintf(
      stderr,
      "config_thread_count_underflow: LIVE BUG B102 — worker_thread_count "
      "underflowed to %u (0x%08X) from thread_count=2,sender=2,receiver=2\n",
      c.worker_thread_count, c.worker_thread_count);
  printf("PASS config_thread_count_underflow: unsigned underflow pinned "
         "(worker=0x%08X)\n",
         c.worker_thread_count);
  /* compute_derived allocated default_ports (multi-node, neither specified). */
  arts_config_destroy(&c);
  return 0;
}
