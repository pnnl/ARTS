/* SPDX-License-Identifier: Apache-2.0
 *
 * T205 — arts_config_find_variable env-override splice hazards.
 * Targets B099 (uninit new_var->next splice, HIGH) and B100 (flexible-array
 * +size vs memcpy size+1 1-byte OOB; variable[255] unbounded copy, MED).
 *
 * arts_config_find_variable walks a linked list for `string`; if an env var of
 * that name exists it builds a replacement node and splices it in.  The splice
 * has three cases:
 *   (a) not found, empty list (last==NULL): prepend — new_var->next=*head. OK.
 *   (b) found (next!=NULL): new_var->next = next->next; free(next). OK.
 *   (c) not found, NON-empty list (last!=NULL, next==NULL):  <-- B099
 *       last->next = new_var; but the `if (next)` block that sets
 *       new_var->next is NOT taken → new_var->next is UNINITIALIZED → the list
 *       tail points at a wild pointer.  Traversing/freeing the list then walks
 *       garbage.
 *
 * Allocation hazard B100: the node is malloc'd sizeof(struct)+size where
 * size=strlen(value), but value[] is a flexible array and memcpy copies size+1
 * bytes → 1-byte heap-buffer-overflow of the trailing NUL.  ASan catches it on
 * every override.
 *
 * Authored correct-and-failing (exposes_runtime_bug=true): a clean ASan run
 * means both the uninitialized-next splice and the +size/+size+1 off-by-one
 * were fixed; today ASan reports a heap overflow (B100) and/or the
 * walk-the-list step crashes (B099).
 */

#include "../../libs/src/core/system/config.c"
#include "config_test_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Build a single list node with the given name/value via the same allocation
   discipline config_get_variables uses, so the list is well-formed input. */
static struct arts_config_variable_s *make_node(const char *name,
                                                const char *value) {
  unsigned int size = (unsigned int)strlen(value);
  struct arts_config_variable_s *v =
      (struct arts_config_variable_s *)arts_malloc(
          sizeof(struct arts_config_variable_s) + size + 1);
  v->size = size;
  v->next = NULL;
  strncpy(v->variable, name, 254);
  v->variable[254] = '\0';
  memcpy(v->value, value, size + 1);
  return v;
}

int main(void) {
  /* List has ONE element named "alpha" that does NOT match the lookup key
     "beta".  We set env beta so the override path runs with
     last != NULL (alpha), next == NULL (beta not found) — case (c), B099. */
  struct arts_config_variable_s *head = make_node("alpha", "1");

  setenv("beta", "value_of_beta", 1);

  struct arts_config_variable_s *res = arts_config_find_variable(&head, "beta");
  if (res == NULL) {
    fprintf(stderr, "FAIL find_variable: override should return new node\n");
    return 1;
  }
  if (strcmp(res->value, "value_of_beta") != 0) {
    fprintf(stderr, "FAIL find_variable: wrong value '%s'\n", res->value);
    return 1;
  }

  /* The spliced node is now the tail after alpha.  A well-formed list must be
     fully traversable and NUL-terminated.  If B099 is live, res->next is
     uninitialized garbage and this walk dereferences it. */
  unsigned int count = 0;
  for (struct arts_config_variable_s *p = head; p != NULL; p = p->next) {
    /* Touch each node's name to force a real read of the (possibly wild)
       pointer chain. */
    volatile char c = p->variable[0];
    (void)c;
    if (++count > 16) {
      fprintf(stderr, "FAIL find_variable: list not terminated (cycle/wild)\n");
      return 1;
    }
  }
  if (count != 2) {
    fprintf(stderr, "FAIL find_variable: expected 2 nodes, walked %u\n", count);
    return 1;
  }

  /* Free the list — exercises the same wild-pointer chain under ASan. */
  config_free_variables(head);

  printf("PASS config_find_variable: override splice well-formed (%u nodes)\n",
         count);
  return 0;
}
