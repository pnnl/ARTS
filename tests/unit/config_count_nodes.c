/* SPDX-License-Identifier: Apache-2.0
 *
 * T207 — arts_config_count_nodes new-syntax matrix (config.c).
 *
 * Property: arts_config_count_nodes() returns the number of physical nodes a
 * routing-table string expands to:
 *   "a"                    -> 1
 *   "a,b,c"                -> 3
 *   "node[01-10]"          -> 10  (bracket range)
 *   "node[1-5]:[50000-...]"-> 5   (colon/port spec ignored for count)
 *   "a,b[1-3],c"           -> 5   (mixed)
 *   "node[10-01]" reversed -> 10  (abs diff)
 * This count sizes the routing-table calloc in create_routing_table; a
 * disagreement with the actual parser is the heap-overflow precondition (B097,
 * see T203).  Two malformed cases are PINNED as documented undercounts (B108):
 *   "node[1-"  (unbalanced '[' no ']')  -> undercount by 1
 *   bracket content >= 64 chars         -> range silently ignored
 *
 * NOTE: count_nodes tokenizes via strchr only (no strtok), but it strncpy's
 * bracket content into a 64-byte stack buffer; we pass mutable buffers.
 */

#include "../../libs/src/core/system/config.c"
#include "config_test_common.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;

static void expect(const char *literal, unsigned int want) {
  char buf[256];
  strncpy(buf, literal, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';
  unsigned int got = arts_config_count_nodes(buf);
  if (got != want) {
    fprintf(stderr, "FAIL count_nodes(\"%s\"): got %u want %u\n", literal, got,
            want);
    fails++;
  }
}

int main(void) {
  /* Well-formed grammar. */
  expect("a", 1);
  expect("a,b,c", 3);
  expect("node[01-10]", 10);
  expect("node[1-5]", 5);
  /* B108-adjacent overcount: count_nodes does NOT actually ignore a port-range
     bracket after the colon — it expands the SECOND '[50000-50004]' as another
     node range too, yielding 5 + 5 - 1 = 9, NOT the 5 the SSH routing-table
     parser will emit (it strips at the first ':').  That count-vs-parser
     disagreement is the heap-overflow precondition behind B097 (T203).  We PIN
     the real (buggy) value 9 so a fix that makes count_nodes ignore the port
     bracket is detected. */
  expect("node[1-5]:[50000-50004]", 9);
  expect("a,b[1-3],c", 5);   /* 1 + 3 + 1 */
  expect("node[10-01]", 10); /* reversed handled by abs diff */
  expect("node[5-5]", 1);    /* single-element range */

  /* Documented undercounts (B108) — PIN current behavior. */
  /* Unbalanced '[' with no ']': the entry-count had -1 applied for the bracket
     but the range size is never added (no closing ']'), so the lone entry is
     undercounted to 0. */
  expect("node[1-5", 0);

  /* Bracket content >= 64 chars: range_len >= sizeof(range_spec) guard skips
     the whole range block, leaving only the -1 already applied → 0 for a
     single oversized entry. */
  {
    char big[160];
    strcpy(big, "node[");
    /* 80 'x' characters then close: content length 80 > 64. */
    for (int i = 0; i < 80; i++) {
      strcat(big, "1");
    }
    strcat(big, "]");
    unsigned int got = arts_config_count_nodes(big);
    if (got != 0) {
      fprintf(stderr, "FAIL count_nodes(oversized bracket): got %u want 0\n",
              got);
      fails++;
    }
  }

  if (fails) {
    fprintf(stderr, "FAIL config_count_nodes: %d checks failed\n", fails);
    return 1;
  }
  printf("PASS config_count_nodes: ranges + mixed + undercount pins\n");
  return 0;
}
