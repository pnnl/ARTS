/* SPDX-License-Identifier: Apache-2.0
 *
 * T206 — parse_port_spec grammar matrix (config.c, static helper).
 *
 * Property: parse_port_spec(spec, &count) translates a port specification into
 * a malloc'd uint array and *count, per its documented grammar:
 *     NULL / ""            -> (NULL, 0)
 *     "50000"              -> ([50000], 1)
 *     "[50000-50002]"      -> ([50000,50001,50002], 3)
 *     "50000,50020,50040"  -> 3 entries
 *     trailing comma       -> count is the real token count (no over-read)
 * Documented foot-gun (B104): an invalid / reversed range silently falls back
 * to a SINGLE port hard-coded to 75563 with count 1, masking bad config.  This
 * test PINS that behavior (it is not a crash, just a silent magic default) so a
 * future fix that surfaces an error is detected as a deliberate change.
 *
 * Pure unit: #include config.c for the static, libc-shim the runtime deps.
 */

#include "../../libs/src/core/system/config.c"
#include "config_test_common.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;

static void expect_null(const char *spec) {
  unsigned int c = 0xDEAD;
  unsigned int *p = parse_port_spec(spec, &c);
  if (p != NULL || c != 0) {
    fprintf(stderr, "FAIL parse_port_spec(%s): expected (NULL,0) got (%p,%u)\n",
            spec ? spec : "NULL", (void *)p, c);
    fails++;
  }
  arts_free(p);
}

static void expect_array(const char *spec, const unsigned int *want,
                         unsigned int want_n) {
  unsigned int c = 0xDEAD;
  unsigned int *p = parse_port_spec(spec, &c);
  if (c != want_n || p == NULL) {
    fprintf(stderr, "FAIL parse_port_spec(%s): count=%u want=%u p=%p\n", spec,
            c, want_n, (void *)p);
    fails++;
    arts_free(p);
    return;
  }
  for (unsigned int i = 0; i < want_n; i++) {
    if (p[i] != want[i]) {
      fprintf(stderr, "FAIL parse_port_spec(%s): [%u]=%u want=%u\n", spec, i,
              p[i], want[i]);
      fails++;
    }
  }
  arts_free(p);
}

int main(void) {
  /* NULL / empty -> (NULL, 0). */
  expect_null(NULL);
  expect_null("");

  /* Single port. */
  {
    unsigned int w[] = {50000};
    expect_array("50000", w, 1);
  }

  /* Range. */
  {
    unsigned int w[] = {50000, 50001, 50002};
    expect_array("[50000-50002]", w, 3);
  }

  /* Reversed range -> invalid -> silent 75563 fallback (B104). */
  {
    unsigned int w[] = {75563};
    expect_array("[50005-50000]", w, 1);
  }

  /* Garbage range -> invalid -> 75563 fallback (B104). */
  {
    unsigned int w[] = {75563};
    expect_array("[abc-def]", w, 1);
  }

  /* Comma list. */
  {
    unsigned int w[] = {50000, 50020, 50040};
    expect_array("50000,50020,50040", w, 3);
  }

  /* Trailing comma: n = comma_count+1 = 2 allocated, but only one real token
     is produced before strtok returns NULL, so *count must be 1 (no over-read
     of the second uninitialized slot). */
  {
    unsigned int w[] = {50000};
    expect_array("50000,", w, 1);
  }

  /* Single garbage -> strtoul yields 0 (silent), count 1. */
  {
    unsigned int w[] = {0};
    expect_array("notaport", w, 1);
  }

  if (fails) {
    fprintf(stderr, "FAIL config_parse_port_spec: %d checks failed\n", fails);
    return 1;
  }
  printf("PASS config_parse_port_spec: grammar + 75563 fallback pinned\n");
  return 0;
}
