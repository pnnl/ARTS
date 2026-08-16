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

/// @file val_wb_confirm_ack_round.c
/// @brief OWNER CONFIRM_ACK piggyback path that ADVANCES the ownership round.
///
/// Under OWNER, arts_handler_db_grant_confirm (home A) pops the NEXT queued
/// requester and piggybacks it into CONFIRM_ACK (new_owner_rank field).  The
/// new owner C, on arts_handler_db_grant_confirm_ack, opens its gate (clears
/// ARTS_GRANT_UNCONFIRMED from writer_count), drains its deferred RW waiters, and applies the
/// piggybacked INVALIDATE effect (publish incoming_new_owner + sentinel
/// withdrawal under the +1 drain guard) so the round ADVANCES to the queued
/// requester via the merged ack — NOT a standalone INVALIDATE.  This merge is
/// the single highest-value OWNER correctness property: it removes the former
/// CONFIRM_ACK<->INVALIDATE reorder window.
///
/// SCENARIO.  To make the second requester be QUEUED DURING the first transfer,
/// each step enqueues TWO unordered RW EDTs on DISTINCT remote ranks gated only
/// on the same DB.  Home grants the first; while that transfer is in flight the
/// second is already on the home pending_rw FIFO, so home advances it through
/// the CONFIRM_ACK piggyback rather than waiting.  Each EDT atomically
/// increments; the final RO read must equal the exact total.  A lost piggyback
/// advance strands the second requester (hang -> ctest TIMEOUT); a mis-applied
/// piggyback INVALIDATE corrupts the sum (arts_abort).
///
/// OWNER-only (the piggyback lives in owner.c's confirm/confirm_ack; home.c has
/// no CONFIRM_ACK).  Clean skip otherwise.  Needs 2+ ranks for a real transfer;
/// 1n is a degenerate pass.

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_VAL) || !defined(ARTS_WRITE_POLICY_WB)

int main(void) {
  printf("SKIP val_wb_confirm_ack_round: VAL+OWNER-only\n");
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
                  "FAIL val_wb_confirm_ack_round: expected %u got %u\n",
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

  arts_printf("=== val_wb_confirm_ack_round ===\n");

  unsigned int nranks = arts_get_total_ranks();

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  unsigned int total = 0u;
  for (unsigned int s = 0; s < STEPS; s++) {
    /* Two back-to-back RW requesters on distinct remote ranks: the second is
     * queued on the home FIFO while the first transfer is in flight, so home
     * advances it through the CONFIRM_ACK piggyback. */
    unsigned int a = (nranks > 1) ? 1u : 0u;
    unsigned int b = (nranks > 2) ? 2u : ((nranks > 1) ? 1u : 0u);
    arts_guid_t e = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t w1 = arts_edt_create(
        inc_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = a, .finish_event = e});
    arts_add_dependence(db, w1, 0, DB_MODE_RW);
    arts_guid_t w2 = arts_edt_create(
        inc_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = b, .finish_event = e});
    arts_add_dependence(db, w2, 0, DB_MODE_RW);
    total += 2u;
    arts_event_wait(e);
  }

  uint64_t expect = (uint64_t)total;
  arts_guid_t ec = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t c =
      arts_edt_create(check_edt, 1, &expect, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = ec});
  arts_add_dependence(db, c, 0, DB_MODE_RO);
  arts_event_wait(ec);

  arts_printf("PASS val_wb_confirm_ack_round sum=%u\n", total);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}

#endif
