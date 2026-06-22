/* SPDX-License-Identifier: Apache-2.0
 *
 * T204 — extract_nodelist_lsf node_list[-1] OOB write + *cnt uninit.
 * Targets B098 (config.c:87, HIGH).
 *
 * extract_nodelist_lsf allocates node_list, then appends "host," for each
 * DISTINCT non-first token.  With a SINGLE host (LSB_HOSTS="host0") — the
 * common 1-node LSF case — strtok yields the first token, the second strtok
 * returns NULL, so the while body never appends: list_str_length stays 0.  The
 * final statement
 *     node_list[list_str_length - 1] = '\0';   // node_list[(unsigned)-1]
 * is then an out-of-bounds write at node_list[0xFFFFFFFF...].
 *
 * Also: *cnt must be set (it is, to 0 here) — but when the env var is UNSET the
 * function returns NULL WITHOUT touching *cnt, so a caller that doesn't
 * pre-zero reads garbage; we pin both.
 *
 * Run under ASan: a heap-buffer-overflow (or global) report = live bug.  This
 * test is authored correct-and-failing (exposes_runtime_bug=true): it returns 0
 * only if no OOB occurred (post-fix: empty list handled, e.g. node_list[0]=0).
 *
 * NOTE: extract_nodelist_lsf calls strtok on the getenv() buffer in place, so
 * we must setenv with a writable copy (the C library owns it after setenv).
 */

#include "../../libs/src/core/system/config.c"
#include "config_test_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  /* Case 1: env var UNSET -> returns NULL, *cnt left untouched. Pre-seed cnt
     with a sentinel and confirm the function does not return a list. */
  unsetenv("LSB_TEST_HOSTS");
  unsigned int cnt = 0xABCDu;
  char *r = extract_nodelist_lsf("LSB_TEST_HOSTS", 1, &cnt);
  if (r != NULL) {
    fprintf(stderr, "FAIL lsf: unset env should return NULL\n");
    return 1;
  }
  /* (cnt is documented-uninitialized on this path; do not assert its value.) */

  /* Case 2: single host -> the OOB write path (B098). */
  setenv("LSB_TEST_HOSTS", "host0", 1);
  cnt = 0xABCDu;
  char *list = extract_nodelist_lsf("LSB_TEST_HOSTS", 1, &cnt);

  /* If we reach here without ASan firing, the empty-list edge was handled. */
  if (cnt != 0) {
    fprintf(stderr, "FAIL lsf: single host should yield count 0, got %u\n",
            cnt);
    arts_free(list);
    return 1;
  }
  arts_free(list);

  printf("PASS config_lsf_single_host_oob: no node_list[-1] OOB, cnt=0\n");
  return 0;
}
