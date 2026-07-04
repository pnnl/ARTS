/* SPDX-License-Identifier: Apache-2.0
 *
 * T214 — config_compute_derived worker-count derivation after the transport
 * cutover + config-surface cleanup.  Historically this pinned B102: a pre-set
 * thread_count with sender+receiver > thread_count underflowed
 *     worker = thread_count - sender - receiver
 * (unsigned) to ~4 billion.  The transport cutover removed the dedicated
 * sender role, folding its cfg count into the worker pool right after that
 * subtraction; the config-surface cleanup went further and deleted the old
 * sender-thread cfg key and its backing field outright (now a hard error —
 * see config_reject_removed_keys).  With no sender term left at all, the
 * derivation is simply
 *     worker = thread_count - progress_thread_count,
 * so the only remaining underflow risk is progress_thread_count itself
 * exceeding a pre-set thread_count.
 *
 * This test drives the multi-node branch (table_length>1, table==NULL so the
 * port loop is skipped) with the boundary case (thread=2, progress=2) and
 * asserts the SANE result: worker a small in-range value (0 here),
 * thread_count consistent — NOT an astronomical wrap.
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

  c.thread_count = 2; /* pre-set (as if from SLURM_CPUS_PER_TASK) */
  c.progress_thread_count = 2; /* boundary: progress == thread_count */
  c.worker_thread_count = 0;

  config_compute_derived(&c);

  /* worker = thread - progress = 2 - 2 = 0 (sane, in range — not the ~4e9
   * wrap the old sender+receiver combination used to cause). */
  if (c.worker_thread_count > c.thread_count) {
    fprintf(stderr,
            "FAIL config_thread_count_underflow: worker=%u underflowed "
            "(thread_count=%u)\n",
            c.worker_thread_count, c.thread_count);
    return 1;
  }
  if (c.worker_thread_count != 0) {
    fprintf(stderr,
            "FAIL config_thread_count_underflow: worker=%u, expected 0 "
            "(thread=2, progress=2)\n",
            c.worker_thread_count);
    return 1;
  }

  printf("PASS config_thread_count_underflow: worker=%u (no underflow; "
         "thread_count=%u)\n",
         c.worker_thread_count, c.thread_count);
  /* compute_derived allocated default_ports (multi-node, neither specified). */
  arts_config_destroy(&c);
  return 0;
}
