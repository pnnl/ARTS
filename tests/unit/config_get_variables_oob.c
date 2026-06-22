/* SPDX-License-Identifier: Apache-2.0
 *
 * T208 — arts_config_get_variables empty-value + non-terminated key.
 * Probes B101 (config.c:163,171, suspected HIGH) and PINS the verified result.
 *
 * arts_config_get_variables parses a cfg FILE* line-by-line as key=value.
 * Two suspected hazards, exercised via fmemopen (precise EOF / no-newline):
 *   (1) val[size-1] (line 163), size=strlen(val): a line "key=" with NO
 *       trailing newline at EOF makes strtok return val=="" (size 0), so
 *       val[size-1] == val[-1].
 *   (2) strncpy(variable, var, 255) (line 171): a key >= 255 chars is NOT
 *       NUL-terminated, so remove_white_spaces() scans past the 255-byte field.
 *
 * VERIFIED FINDING (this test runs ASan-CLEAN, so exposes_runtime_bug=false):
 * neither suspected hazard is an actual out-of-bounds heap access, because of
 * the getline/struct buffer layout:
 *   (1) `val` always points at least one byte INTO the getline line buffer
 *       (just past the '=' strtok replaced with '\0'); val[-1] therefore lands
 *       on that former-'=' byte, which is in-bounds of the line allocation and
 *       holds '\0' (never '\n'), so the guarded write never executes.  The
 *       suspected val[-1] OOB cannot manifest.
 *   (2) the node is malloc'd sizeof(struct)+size with value[] copied size+1
 *       bytes (NUL included), so remove_white_spaces' scan over the
 *       non-terminated `variable[255]` field runs straight into value[] and
 *       stops at value's own NUL — still inside the allocation.
 * The test therefore PINS that get_variables handles empty trailing values and
 * over-long keys WITHOUT a detectable OOB; a future struct/layout change that
 * makes either land out of bounds would flip ASan to red here.
 */

#include "../../libs/src/core/system/config.c"
#include "config_test_common.h"

#include <stdio.h>
#include <string.h>

int main(void) {
  /* Build a buffer whose FINAL line is "key=" with no trailing newline, so
     getline returns "key=" and strtok(val) == "".  Precede it with a normal
     line and a no-'=' line (must be skipped). */
  char buf[512];
  int n = snprintf(buf, sizeof(buf),
                   "good=val\n" /* normal: parsed */
                   "noequals\n" /* skipped (val NULL) */
                   "key=");     /* trailing empty value, NO newline at EOF */
  (void)n;

  FILE *f = fmemopen(buf, strlen(buf), "r");
  if (!f) {
    fprintf(stderr, "FAIL get_variables: fmemopen failed\n");
    return 1;
  }
  struct arts_config_variable_s *vars = arts_config_get_variables(f);
  (void)fclose(f);

  /* Walk + free. Only "good=val" yields a node: "noequals" has no '=' (val
     NULL -> skipped), and the trailing "key=" with an EMPTY field makes
     strtok(NULL,"=") return NULL too (strtok never yields a "" token for a
     lone trailing delimiter) -> val is NULL and the val[-1] code at line 163 is
     NEVER reached.  This is the structural reason the suspected B101 part-1 OOB
     is unreachable: there is no input for which `val` is the empty string.
     => exactly 1 node. */
  unsigned int count = 0;
  for (struct arts_config_variable_s *p = vars; p; p = p->next) {
    count++;
  }
  if (count != 1) {
    fprintf(stderr, "FAIL get_variables: expected 1 node, got %u\n", count);
    config_free_variables(vars);
    return 1;
  }
  config_free_variables(vars);

  /* Now the >=255-char key (B101 part 2): non-terminated strncpy. */
  char big[512];
  memset(big, 'k', 300);
  big[300] = '=';
  big[301] = 'v';
  big[302] = '\n';
  big[303] = '\0';
  FILE *f2 = fmemopen(big, strlen(big), "r");
  if (!f2) {
    fprintf(stderr, "FAIL get_variables: fmemopen(big) failed\n");
    return 1;
  }
  struct arts_config_variable_s *vars2 = arts_config_get_variables(f2);
  (void)fclose(f2);
  /* remove_white_spaces over the non-terminated 255-byte field runs inside
     get_variables already; if ASan is clean, just free. */
  config_free_variables(vars2);

  printf("PASS config_get_variables_oob: empty-value + long-key handled\n");
  return 0;
}
