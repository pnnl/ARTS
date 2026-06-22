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

/// @file mrsw_lazy_transfer_map.c
/// @brief MRSW LAZY OWNERSHIP_RESPONSE map deserialize ordering
/// (config_specific).
///
/// In MRSW LAZY, arts_handler_db_ownership_response (TRANSFER install):
///   1. reconstructs the owner-side dedup map from the serialized payload,
///      DESTROYING the old map first (no leak / no UAF on the previous map),
///   2. sets ownership_unconfirmed = 1 BEFORE the writer_count 0->2 bump
///      (ordered so a fresh RW gate-checks unconfirmed before it can observe
///      the owned count),
///   3. DEFERS the RW drain to CONFIRM_ACK (home has not flipped rw_holder).
///
/// To force a NON-EMPTY serialized map across the transfer, each ownership
/// round is preceded by RO readers on several ranks (REDIRECT_RO registers them
/// in last_sent_version).  When ownership ships to the next owner, that map is
/// serialized, deserialized, and the OLD map destroyed on the receiver.  The
/// chain then writes under the freshly-installed (and confirmed) ownership.
///
/// Correctness pin: a strict RW-write chain across ranks, each writer +1, with
/// interleaved RO fans; the final reader asserts value == number of writes.  A
/// botched map deserialize / reordered unconfirmed-vs-bump / dropped deferred
/// drain surfaces as a wrong count, a stale RO read (caught by the RO bound
/// check), a SIGSEGV (double-free of the old map), or a hang (ctest TIMEOUT).
///
/// LAZY + MRSW only.  Requires >= 2 ranks to transfer; SKIPs on 1n.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_MRSW) || !defined(ARTS_TIMING_LAZY)

int main(void) {
  printf("SKIP mrsw_lazy_transfer_map: MRSW+LAZY-only\n");
  return 0;
}

#else

#define ROUNDS 50u
#define N_RO_PER_ROUND 3u

static void rw_inc_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  if (d != NULL) {
    d[0] = d[0] + 1u;
  }
}

static void ro_check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  const uint64_t *d = (const uint64_t *)depv[0].ptr;
  uint64_t hi = paramv[0];
  uint64_t v = (d != NULL) ? d[0] : 0u;
  if (v > hi) {
    (void)fprintf(stderr, "FAIL: RO read %llu > total %llu (bad map xfer)\n",
                  (unsigned long long)v, (unsigned long long)hi);
    arts_abort(1);
  }
}

static unsigned int g_total = 0u;

static void final_check_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const uint64_t *d = (const uint64_t *)depv[0].ptr;
  uint64_t v = (d != NULL) ? d[0] : 0u;
  if (v != (uint64_t)g_total) {
    (void)fprintf(stderr, "FAIL: final value %llu != %u\n",
                  (unsigned long long)v, g_total);
    arts_abort(1);
  }
  arts_printf("PASS: mrsw_lazy_transfer_map final=%u\n", g_total);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== mrsw_lazy_transfer_map ===\n");

  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 2u) {
    arts_printf("SKIP: mrsw_lazy_transfer_map requires >= 2 ranks\n");
    arts_shutdown();
    return;
  }

  g_total = ROUNDS;

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((uint64_t *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  /* Phase 1: per round, RO readers on several ranks populate the owner-side
   * dedup map (and legitimately race mid-flight values), then a single RW on a
   * rotating rank ships ownership (serializing that non-empty map across the
   * transfer).  The DB RW chain orders the writers.  Waiting on the finish
   * event blocks main_edt until every writer has run AND written back. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (unsigned int r = 0; r < ROUNDS; r++) {
    for (unsigned int j = 0; j < N_RO_PER_ROUND; j++) {
      uint64_t hi = (uint64_t)g_total;
      arts_guid_t ro = arts_edt_create(
          ro_check_edt, 1, &hi, 1,
          &(arts_edt_hint_t){.rank = j % nranks, .finish_event = fe});
      arts_add_dependence(db, ro, 0, DB_MODE_RO);
    }
    unsigned int wrank = (r % (nranks - 1u)) + 1u; /* rotate over non-home */
    arts_guid_t w =
        arts_edt_create(rw_inc_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = wrank, .finish_event = fe});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
  }
  arts_event_wait(fe); /* blocks until ALL writers ran + wrote back */

  /* Phase 2: the final checker, created only now that phase 1 has fully
   * quiesced, so its RO dependency is registered after the committed value is
   * in place and its snapshot observes every increment. */
  arts_guid_t fe2 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t fin =
      arts_edt_create(final_check_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe2});
  arts_add_dependence(db, fin, 0, DB_MODE_RO);
  arts_event_wait(fe2);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_MRSW && ARTS_TIMING_LAZY */
