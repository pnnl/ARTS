/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/

/// @file val_publish_before_decrement.c
/// @brief Force the publish-before-decrement dedup between release_rw's 0-edge
///        and a concurrent INVALIDATE handler shipping the same transfer.
///
/// When an owner finishes its RW EDT (arts_db_release_rw decrements
/// writer_count) at the SAME moment home INVALIDATEs it for the next requester
/// (the INVALIDATE handler also publishes incoming_new_owner and decrements),
/// EXACTLY ONE of {the releaser, the INVALIDATE handler} must observe the
/// positive->0 edge with incoming_new_owner already set and ship exactly one
/// owner->owner transfer.  The discipline is: incoming_new_owner is published
/// BEFORE the decrement, and arts_db_send_grant_response re-arms the field
/// to the sentinel BEFORE sending (so a self-transfer, owner==home, can
/// republish without losing the fresh value).  A reordering double-ships
/// (corruption) or drops the transfer (a queued RW acquirer hangs).
///
/// SCENARIO.  A single RW DataBlock; for each step we enqueue TWO unordered RW
/// EDTs on DIFFERENT remote ranks plus one on home — three contending RW
/// acquirers per step, all gated only on the DB, so home always has a queued
/// next-requester when the current owner releases.  That makes every step a
/// release-0-edge-vs-next-INVALIDATE race, repeated densely.  The owner==home
/// step exercises the self-transfer re-arm path.  Each EDT atomically
/// increments the shared buffer; the final RO read must equal the exact total —
/// a double-ship would corrupt the buffer version chain and a dropped transfer
/// would hang (caught by ctest TIMEOUT).
///
/// VAL-only, both placements (the publish-before-decrement is identical in the
/// INVALIDATE handler of home.c and owner.c).  Self-skips elsewhere; 1n is a
/// degenerate (no-transfer) pass.

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_VAL)

int main(void) {
  printf("SKIP val_publish_before_decrement: VAL-only\n");
  return 0;
}

#else

#define STEPS 80u

static void inc_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                    arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  _Atomic unsigned int *d = (_Atomic unsigned int *)depv[0].ptr;
  if (d != NULL) {
    atomic_fetch_add_explicit(d, 1u, memory_order_relaxed);
  }
}

static void check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int expect = (unsigned int)paramv[0];
  if (d == NULL || d[0] != expect) {
    (void)fprintf(stderr,
                  "FAIL val_publish_before_decrement: expected %u got %u\n",
                  expect, d ? d[0] : 0u);
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== val_publish_before_decrement ===\n");

  unsigned int nranks = arts_get_total_ranks();

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  unsigned int total = 0u;
  for (unsigned int s = 0; s < STEPS; s++) {
    arts_guid_t e = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    /* Three contending RW acquirers per step: home + two distinct remotes
     * (degrades gracefully to home when nranks==1).  A queued next-requester is
     * always present when the current owner releases. */
    unsigned int r1 = (nranks > 1) ? 1u : 0u;
    unsigned int r2 = (nranks > 2) ? 2u : ((nranks > 1) ? 1u : 0u);
    unsigned int ranks[3] = {0u, r1, r2};
    for (unsigned int k = 0; k < 3; k++) {
      arts_guid_t w = arts_edt_create(
          inc_edt, 0, NULL, 1,
          &(arts_edt_hint_t){.rank = ranks[k], .finish_event = e});
      arts_add_dependence(db, w, 0, DB_MODE_RW);
      total++;
    }
    arts_event_wait(e);
  }

  uint64_t expect = (uint64_t)total;
  arts_guid_t ec = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t c =
      arts_edt_create(check_edt, 1, &expect, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = ec});
  arts_add_dependence(db, c, 0, DB_MODE_RO);
  arts_event_wait(ec);

  arts_printf("PASS val_publish_before_decrement sum=%u\n", total);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif
