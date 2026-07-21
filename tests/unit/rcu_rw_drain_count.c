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

/// @file rcu_rw_drain_count.c
/// @brief GRANT drain accounting: every queued RW waiter is woken exactly once
///        with the correct buffer pointer, #writer_count bumps == #drained.
///
/// rw_drain_cb (per waiter): writer_count += 1, then mark_edt_secured, then
/// mark_edt_ready (secure BEFORE ready — secure is idempotent and never
/// schedules; ready may schedule/free).  arts_db_drain_pending_rw_after_grant
/// drains the WHOLE cache->pending_rw stack via that cb.  The accounting
/// invariant: the number of writer_count increments here exactly equals the
/// number of waiters drained, and each parked EDT runs exactly once with a
/// non-NULL ptr.  A double-drain double-counts (count never returns to 0 ->
/// later transfer mis-fires) or a missed waiter strands an acquirer.
///
/// SCENARIO.  A home RW DataBlock sized as an N-slot array.  N RW EDTs, each
/// owning a UNIQUE slot index (paramv[0]), are all spawned at once gated only
/// on the same DB on a SINGLE remote rank, so they pile onto that rank's
/// pending_rw Treiber stack and are drained together at the GRANT.  Each EDT
/// writes its index into its own slot (no inter-EDT data race; one slot each).
/// A final RO reader asserts every slot holds its own index — proving every
/// waiter was woken exactly once with the correct (non-NULL) pointer.  A
/// missing or duplicated drain leaves a slot wrong or hangs (ctest TIMEOUT).
///
/// RCU-only, both timings (the drain cb + accounting is shared ownership.c;
/// EAGER drains at RESPONSE, LAZY at CONFIRM_ACK, but the per-waiter accounting
/// is identical).  Clean skip otherwise.  1n: waiters drain locally; still
/// exact.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_RCU)

int main(void) {
  printf("SKIP rcu_rw_drain_count: RCU-only\n");
  return 0;
}

#else

#define N 64u
#define ROUNDS 8u

static void slot_writer_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int idx = (unsigned int)paramv[0];
  unsigned int *arr = (unsigned int *)depv[0].ptr;
  if (arr != NULL) {
    arr[idx] = idx + 1u; /* idx+1 so a missed (zero) slot is detectable */
  }
}

static void check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *arr = (unsigned int *)depv[0].ptr;
  if (arr == NULL) {
    (void)fprintf(stderr, "FAIL rcu_rw_drain_count: NULL buffer at check\n");
    arts_abort(1);
  }
  for (unsigned int i = 0; i < N; i++) {
    if (arr[i] != i + 1u) {
      (void)fprintf(stderr,
                    "FAIL rcu_rw_drain_count: slot %u expected %u got %u\n",
                    i, i + 1u, arr[i]);
      arts_abort(1);
    }
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== rcu_rw_drain_count ===\n");

  unsigned int nranks = arts_get_total_ranks();
  unsigned int W = (nranks > 1) ? 1u : 0u; /* single remote rank: deep pile */

  for (unsigned int r = 0; r < ROUNDS; r++) {
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, N * sizeof(unsigned int), ARTS_DB,
                       ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 0});
    for (unsigned int i = 0; i < N; i++) {
      ((unsigned int *)ptr)[i] = 0u;
    }
    arts_db_release(db, DB_MODE_RW);

    arts_guid_t e = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (unsigned int i = 0; i < N; i++) {
      uint64_t idx = (uint64_t)i;
      arts_guid_t w =
          arts_edt_create(slot_writer_edt, 1, &idx, 1,
                          &(arts_edt_hint_t){.rank = W, .finish_event = e});
      arts_add_dependence(db, w, 0, DB_MODE_RW);
    }
    arts_event_wait(e);

    arts_guid_t ec = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t c =
        arts_edt_create(check_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = ec});
    arts_add_dependence(db, c, 0, DB_MODE_RO);
    arts_event_wait(ec);
  }

  arts_printf("PASS rcu_rw_drain_count N=%u rounds=%u\n", N, ROUNDS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif
