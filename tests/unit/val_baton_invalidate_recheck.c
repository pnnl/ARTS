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

/// @file val_baton_invalidate_recheck.c
/// @brief Home invalidate_in_flight baton: at-most-one INVALIDATE per round,
///        and the classic release-then-recheck lost-wakeup window.
///
/// arts_handler_db_grant_request pushes the requester onto the home
/// pending_rw Vyukov FIFO, then CAS invalidate_in_flight 0->1; the WINNER
/// starts a round (one INVALIDATE), LOSERS return (piggyback, stay queued).
/// When a round closes, ownership_confirm releases the baton and RECHECKS the
/// queue: if a racer was enqueued between the baton release and the
/// empty-check, the CAS-reacquire must pick it up and start the next round.  A
/// lost wakeup here — a requester enqueued in exactly that window with nobody
/// re-driving the round — strands that acquirer forever (distributed hang).
/// Exactly one INVALIDATE per round is the at-most-one mutual-exclusion
/// guarantee.
///
/// SCENARIO.  A single home RW DataBlock.  Per batch, a large burst of mutually
/// UNORDERED RW EDTs round-robin across ALL ranks, all gated only on that DB,
/// so many OWNERSHIP_REQUESTs hit home concurrently — a deep FIFO with constant
/// baton contention and a tight stream of round closes, each a fresh
/// release-recheck window.  Each EDT atomically increments; the final RO read
/// must equal the exact total.  A dropped racer (lost-wakeup) hangs (ctest
/// TIMEOUT); a double-INVALIDATE / mis-served round would corrupt the sum.
///
/// VAL-only, both placements (the baton + request handler are shared
/// grant.c; the confirm recheck loop exists in both home.c and owner.c
/// confirm bodies). Clean skip otherwise.  Needs 2+ ranks for cross-rank baton
/// contention; 1n is a degenerate (local) pass.

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_VAL)

int main(void) {
  printf("SKIP val_baton_invalidate_recheck: VAL-only\n");
  return 0;
}

#else

#define BATCHES 30u
#define PER_BATCH 32u

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
                  "FAIL val_baton_invalidate_recheck: expected %u got %u\n",
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

  arts_printf("=== val_baton_invalidate_recheck ===\n");

  unsigned int nranks = arts_get_total_ranks();

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  unsigned int total = 0u;
  for (unsigned int b = 0; b < BATCHES; b++) {
    arts_guid_t e = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (unsigned int i = 0; i < PER_BATCH; i++) {
      unsigned int rank = (nranks > 1) ? ((b * 7u + i) % nranks) : 0u;
      arts_guid_t w =
          arts_edt_create(inc_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = rank, .finish_event = e});
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

  arts_printf("PASS val_baton_invalidate_recheck sum=%u\n", total);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}

#endif
