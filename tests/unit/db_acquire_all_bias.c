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

/// @file db_acquire_all_bias.c
/// @brief The +1 bias in arts_db_acquire_all must prevent a remote GRANT
///        arriving DURING the Pass-1/Pass-2 fire (on a receiver thread) from
///        driving acquire_remaining to 0 before the final bias decrement.
///
/// arts_db_acquire_all sets acquire_remaining = (1 + n), fires every
/// non-serialized dep (Pass 1, order-independent local-hits / remote parks),
/// then fires the serialized RW deps (Pass 2), and only THEN removes the +1
/// bias.  The bias guarantees that even if all n deps resolve (including remote
/// GRANTs landing while the fire loop is still running on the worker thread),
/// the EDT cannot be scheduled until the worker completes the fire and drops
/// the bias.  If the bias math is wrong, the EDT either schedules early (some
/// dep's data not yet delivered -> stale/NULL pointer) or never (a decrement is
/// lost
/// -> hang).
///
/// Stress shape: an EDT with a MIX of dep kinds whose resolution timing varies:
///   - several RO deps on DBs homed locally (rank 0) -> immediate local hits
///     (resolve synchronously inside Pass 1).
///   - several RW deps on DBs homed REMOTELY -> remote ownership rounds that
///     PARK and resolve later via a GRANT on a receiver thread, racing the
///     worker's fire loop.
/// The EDT verifies every dep's pointer is non-NULL and the writes land; a
/// premature schedule corrupts a dep -> arts_abort, a lost decrement hangs ->
/// ctest TIMEOUT.  Repeated for many iterations to widen the race window.
///
/// ownership protocols (VAL); needs >= 2 ranks for the remote-park arm.
/// SKIPs cleanly single-node and under WRF_VAL (no ownership park).  EXCL also
/// serializes RW and parks remotely, so it runs there too.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define ITERS 100u
#define NRO 3u /* RO local-hit deps */
#define NRW 3u /* RW remote-park deps */
#define NDEP (NRO + NRW)
#define RW_BASE 0x9A000u

/// mixed_edt: NRO RO deps (slots 0..NRO-1) + NRW RW deps (slots NRO..NDEP-1).
/// Every slot must be non-NULL; writes a per-RW-slot sentinel.
void mixed_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  for (uint32_t i = 0; i < depc; i++) {
    if (depv[i].ptr == NULL) {
      (void)fprintf(stderr,
                    "FAIL: db_acquire_all_bias slot %u NULL (early schedule)\n",
                    i);
      arts_abort(1);
      return;
    }
  }
  for (uint32_t i = NRO; i < depc; i++) {
    ((unsigned int *)depv[i].ptr)[0] = RW_BASE + i;
  }
}

/// rw_checker: RO verify one of the RW DBs got its sentinel.
void rw_checker(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int slot = (unsigned int)paramv[0];
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d == NULL || d[0] != RW_BASE + slot) {
    (void)fprintf(stderr, "FAIL: db_acquire_all_bias RW slot %u got 0x%x\n",
                  slot, d ? d[0] : 0u);
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_acquire_all_bias ===\n");

#if defined(ARTS_PROTOCOL_WRF_VAL)
  arts_printf("SKIP db_acquire_all_bias: WRF_VAL has no ownership park\n");
  arts_shutdown();
  return;
#else
  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 2) {
    arts_printf("SKIP db_acquire_all_bias: needs >= 2 ranks (have %u)\n",
                nranks);
    arts_shutdown();
    return;
  }

  for (unsigned int it = 0; it < ITERS; it++) {
    arts_guid_t ro[NRO];
    arts_guid_t rw[NRW];

    /* RO deps homed locally (rank 0) -> synchronous local hits. */
    for (unsigned int i = 0; i < NRO; i++) {
      void *p = NULL;
      ro[i] = arts_db_create(&p, sizeof(unsigned int), ARTS_DB,
                             ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 0});
      if (p != NULL) {
        ((unsigned int *)p)[0] = 0u;
      }
      arts_db_release(ro[i], DB_MODE_RW);
    }
    /* RW deps homed remotely (rank 1+) -> ownership parks resolved by GRANT. */
    for (unsigned int i = 0; i < NRW; i++) {
      unsigned int home = 1u + (i % (nranks - 1u));
      void *p = NULL;
      rw[i] =
          arts_db_create(&p, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                         &(arts_db_hint_t){.rank = home});
      arts_db_release(rw[i], DB_MODE_RW);
    }

    arts_guid_t e_m = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t m =
        arts_edt_create(mixed_edt, 0, NULL, NDEP,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = e_m});
    for (unsigned int i = 0; i < NRO; i++) {
      arts_add_dependence(ro[i], m, i, DB_MODE_RO);
    }
    for (unsigned int i = 0; i < NRW; i++) {
      arts_add_dependence(rw[i], m, NRO + i, DB_MODE_RW);
    }
    arts_event_wait(e_m);

    arts_guid_t e_c = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (unsigned int i = 0; i < NRW; i++) {
      uint64_t slot = NRO + i;
      arts_guid_t c =
          arts_edt_create(rw_checker, 1, &slot, 1,
                          &(arts_edt_hint_t){.rank = 0, .finish_event = e_c});
      arts_add_dependence(rw[i], c, 0, DB_MODE_RO);
    }
    arts_event_wait(e_c);
  }

  arts_printf("PASS: db_acquire_all_bias %u iterations\n", ITERS);
  arts_shutdown();
#endif
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
