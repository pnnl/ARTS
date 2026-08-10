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
 * Anything else — a reversed or unterminated range, a non-numeric token, a
 * number outside the TCP port space, a list that yields fewer entries than it
 * promises — is malformed and reports (NULL, 0).  Substituting a port for a
 * spec the operator got wrong would bind something nobody named, and a list
 * shorter than its promised length is read past its end by the caller, whose
 * entry count doubles as the per-node connection count.
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

  /* Reversed range -> malformed -> (NULL, 0), so the caller can reject the
     config instead of binding a port nobody named. */
  expect_null("[50005-50000]");

  /* Garbage range -> malformed. */
  expect_null("[abc-def]");

  /* Unterminated range -> malformed. */
  expect_null("[50000-50001");

  /* Out of the TCP port space -> malformed. */
  expect_null("[65535-65536]");
  expect_null("70000");
  expect_null("0");

  /* Comma list. */
  {
    unsigned int w[] = {50000, 50020, 50040};
    expect_array("50000,50020,50040", w, 3);
  }

  /* Trailing comma: the list promises comma_count+1 entries but yields one, so
     the spec is malformed rather than a short list the caller would read past
     the end of. */
  expect_null("50000,");

  /* A token that is not a number at all. */
  expect_null("notaport");
  expect_null("50000,notaport");

  if (fails) {
    fprintf(stderr, "FAIL config_parse_port_spec: %d checks failed\n", fails);
    return 1;
  }
  printf("PASS config_parse_port_spec: grammar pinned; malformed specs "
         "report no ports\n");
  return 0;
}
