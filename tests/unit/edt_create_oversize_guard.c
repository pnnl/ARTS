/* SPDX-License-Identifier: Apache-2.0
 *
 * A remote EDT whose serialized form exceeds the control-message bound must
 * be REFUSED at the create site, naming the counts that set the size.
 *
 * The control ring bounds one two-sided message, and a task's parameter and
 * dependence counts are the one control-message size a program chooses.
 * Nothing quietly rescues an over-bound task: it cannot be split, and the
 * transport ceiling below it would report only bytes.  The run is therefore
 * expected to die on that guard, and the verdict is its message rather than
 * the exit status (PASS_REGULAR_EXPRESSION) — reaching the prints below
 * would mean the guard went missing.
 */

#include "arts.h"

#include <stdint.h>
#include <stdlib.h>

/* Comfortably past a 2 MiB control bound at 8 bytes per parameter, while
 * staying one allocation. */
#define OVERSIZE_PARAMC (400u * 1024u)

void victim_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  /* Never reached: the create that would place this task is refused. */
  arts_printf("FAIL: oversize EDT was shipped and ran\n");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  if (arts_get_total_ranks() < 2) {
    arts_printf("FAIL: this test needs at least two ranks\n");
    arts_shutdown();
    return;
  }

  uint64_t *params = (uint64_t *)calloc(OVERSIZE_PARAMC, sizeof(uint64_t));
  if (params == NULL) {
    arts_printf("FAIL: could not allocate the parameter vector\n");
    arts_shutdown();
    return;
  }

  /* Aim at a rank that is not this one, so the task has to travel. */
  arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
  hint.rank = (arts_get_current_rank() + 1) % arts_get_total_ranks();

  (void)arts_edt_create(victim_edt, OVERSIZE_PARAMC, params, 0, &hint);

  free(params);
  arts_printf("FAIL: oversize remote EDT create returned\n");
  arts_shutdown();
}

int main(int argc, char **argv) { return arts_rt(argc, argv); }
