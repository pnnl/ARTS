/* SPDX-License-Identifier: Apache-2.0
 *
 * T213 — config_compute_derived route_table_entries = 1U << route_table_size.
 * Targets B103 (config.c compute_derived, MED/UB).
 *
 * compute_derived sets:
 *     config->route_table_entries     = 1U << config->route_table_size;
 *     config->gpu_route_table_entries = 1U << config->gpu_route_table_size;
 * with NO clamp on the shift count.  Properties / boundaries:
 *   size=20  -> 1<<20 = 1048576           (normal)
 *   size=31  -> 1<<31 = 0x80000000        (high bit set, still defined)
 *   size=32  -> shift of a 32-bit unsigned by >= width is UNDEFINED BEHAVIOR
 *
 * The size<=31 cases run in-process and PIN the unclamped values (documenting
 * the absence of a clamp: a config typo route_table_size=31 silently asks for
 * a 2-billion-entry table → OOM).  The size=32 UB case is exercised in a forked
 * child under UBSan so the parent still reports PASS; we record that the shift
 * is UB (no clamp) rather than asserting a specific value.
 *
 * compute_derived also derives thread counts; we keep table_length<=1 so it
 * takes the simple single-node reclaim path and does not touch ports.
 */

#include "../../libs/src/core/system/config.c"
#include "config_test_common.h"

#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static void minimal(struct arts_config_s *c) {
  memset(c, 0, sizeof(*c));
  c->table_length = 1; /* single-node path */
  c->worker_thread_count = 4;
  c->gpu_route_table_size = 12; /* keep gpu shift well-defined */
}

int main(void) {
  /* size = 20 -> 1<<20 */
  {
    struct arts_config_s c;
    minimal(&c);
    c.route_table_size = 20;
    config_compute_derived(&c);
    if (c.route_table_entries != (1u << 20)) {
      fprintf(stderr, "FAIL shift: size 20 -> %u (want %u)\n",
              c.route_table_entries, 1u << 20);
      return 1;
    }
    if (c.gpu_route_table_entries != (1u << 12)) {
      fprintf(stderr, "FAIL shift: gpu size 12 -> %u\n",
              c.gpu_route_table_entries);
      return 1;
    }
  }

  /* size = 31 -> high bit set, still defined; PIN the unclamped huge value. */
  {
    struct arts_config_s c;
    minimal(&c);
    c.route_table_size = 31;
    config_compute_derived(&c);
    if (c.route_table_entries != 0x80000000u) {
      fprintf(stderr, "FAIL shift: size 31 -> 0x%x (want 0x80000000)\n",
              c.route_table_entries);
      return 1;
    }
  }

  /* size = 32 -> UB (1U << 32 on a 32-bit unsigned). Run in a forked child
     under whatever sanitizer is active; we only require the parent survives and
     records that NO clamp exists (the runtime would silently emit garbage /
     trap, never a clean error). */
  {
    pid_t pid = fork();
    if (pid == 0) {
      struct arts_config_s c;
      minimal(&c);
      c.route_table_size = 32;
      config_compute_derived(&c);
      /* If UBSan is on, it already reported; print the (garbage) result. */
      fprintf(stderr, "child: size 32 -> %u (UB, unclamped)\n",
              c.route_table_entries);
      _exit(0);
    }
    int status = 0;
    (void)waitpid(pid, &status, 0);
    /* Do not assert child exit code: UBSan may abort it. The point is the
       PARENT proves the in-range cases and documents the UB at size>=32. */
  }

  printf("PASS config_route_table_size_shift: 20/31 unclamped pinned, "
         "32 documented UB\n");
  return 0;
}
