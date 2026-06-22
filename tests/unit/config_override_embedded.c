/* SPDX-License-Identifier: Apache-2.0
 *
 * T218 — config_open_file priority chain + slurm hostname digit truncation.
 *
 * Two config-parser behaviours, exercised against the linked libarts (no
 * runtime started):
 *
 * (A) config_open_file priority chain (config.c).  The documented priority is:
 *       1. compiler-embedded override DATA  (fmemopen)
 *       2. compiler-injected override PATH  (fopen)
 *       3. ARTS_CONFIG env  ->  ./arts.cfg fallback
 *     The override DATA/PATH live in FILE-STATIC pointers
 *     (arts_config_override_path / arts_config_override_data) that a linked
 * test CANNOT set (they are only patched at link time for a self-contained
 *     binary).  So this test verifies the externally observable rung —
 *     priority 3, ARTS_CONFIG — by loading a crafted cfg through the env var
 * and confirming its values land in the resolved struct.  The embedded
 *     DATA/PATH rungs are a documented coverage gap requiring a test built INTO
 *     config.c (the tests/unit/ #include-config.c pattern); flagged here, not
 *     silently skipped.
 *
 * (B) arts_config_get_slurm_hostname() digit truncation (config.c, exported).
 *     The pad width equals strlen(digit_sample); a value with MORE digits than
 *     the sample width has its HIGH-ORDER digits silently dropped (the
 * value/=10 loop runs only digit_length times).  This pins that documented
 * behaviour so a future fix that widens / errors is detected.
 *
 * Orthogonal to the coherence protocol axis.
 */

#include "arts.h"
#include "arts/system/config.h"
#include "arts/utils/malloc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Exported helpers with external linkage but no public-header prototype.
 * arts_config_get_slurm_hostname returns memory allocated by arts_malloc, which
 * carries a hidden allocation header, so it MUST be released with arts_free
 * (NOT libc free). */
char *arts_config_get_slurm_hostname(char *name, char *digit_sample,
                                     unsigned int value);
void arts_free(void *ptr);

static int fails = 0;

/* (A) ARTS_CONFIG (priority 3) is honored by config_open_file. */
static void test_arts_config_priority(void) {
  char cfg_path[256];
  snprintf(cfg_path, sizeof(cfg_path), "config_override_embedded_%ld.cfg",
           (long)getpid());
  FILE *f = fopen(cfg_path, "w");
  if (!f) {
    printf("FAIL config_override_embedded: cannot write temp cfg\n");
    fails++;
    return;
  }
  /* A distinctive, single-node, non-default value we can read back. */
  fputs("[ARTS]\n"
        "launcher=local\n"
        "node_count=1\n"
        "worker_threads=7\n"
        "route_table_size=12\n",
        f);
  (void)fclose(f);

  setenv("ARTS_CONFIG", cfg_path, 1);
  unsetenv("default_ports");
  unsetenv("port_count");

  struct arts_config_s config;
  arts_config_load(&config);

  /* worker_threads=7 single node -> reclaim leaves worker_thread_count==7. */
  if (config.worker_thread_count != 7) {
    printf("FAIL config_override_embedded: ARTS_CONFIG not honored "
           "(worker=%u expected 7)\n",
           config.worker_thread_count);
    fails++;
  }
  /* route_table_size=12 -> route_table_entries == 1<<12 == 4096. */
  if (config.route_table_entries != (1U << 12)) {
    printf(
        "FAIL config_override_embedded: route_table_entries=%u expected %u\n",
        config.route_table_entries, 1U << 12);
    fails++;
  }

  arts_config_destroy(&config);
  (void)remove(cfg_path);
}

/* (B) slurm hostname zero-pad + high-order digit truncation. */
static void expect_hostname(char *name, char *sample, unsigned int value,
                            const char *want) {
  char *got = arts_config_get_slurm_hostname(name, sample, value);
  if (!got || strcmp(got, want) != 0) {
    printf("FAIL config_override_embedded: slurm_hostname(%s,%s,%u)=%s "
           "expected %s\n",
           name, sample, value, got ? got : "(null)", want);
    fails++;
  }
  /* arts_config_get_slurm_hostname allocates via the ARTS allocator
   * (arts_malloc), which prepends a custom header; it must be released with the
   * matching arts_free, not libc free (a bad-free under sanitizers). */
  arts_free(got);
}

static void test_slurm_hostname_truncation(void) {
  /* Fits in the pad width: zero-padded normally. */
  expect_hostname("node", "01", 5, "node05");
  expect_hostname("node", "001", 42, "node042");
  /* Exactly the pad width. */
  expect_hostname("node", "01", 42, "node42");
  /* OVERFLOW: value has more digits than the sample width -> the high-order
     digit(s) are silently dropped.  123 with width 2 keeps only "23". */
  expect_hostname("node", "01", 123, "node23");
  /* 4567 with width 3 keeps only "567". */
  expect_hostname("host", "000", 4567, "host567");
}

int main(void) {
  test_arts_config_priority();
  test_slurm_hostname_truncation();

  /* Document the unreachable rungs of the priority chain as a known coverage
     gap (informational, not a failure). */
  printf("INFO config_override_embedded: override DATA/PATH rungs "
         "(priority 1/2) are file-static and untestable from a linked binary; "
         "covered only by the #include-config.c unit pattern\n");

  if (fails == 0) {
    printf("PASS config_override_embedded: ARTS_CONFIG priority honored + "
           "slurm hostname truncation pinned\n");
    return 0;
  }
  return 1;
}
