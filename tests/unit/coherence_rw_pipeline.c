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

/// @file coherence_rw_pipeline.c
/// @brief Regression for the DB-acquire pipelining path.
///   Phase A: ONE EDT on rank 0 with K cross-rank RW deps (one DB homed per
///            rank). Exercises the per-EDT RW pipelining plus parallel RO
///            reads.
///   Phase B: TWO EDTs on rank 1 both RW-acquiring the SAME DB homed on rank 0.
///   Each phase uses a finish event so the value check is race-free.  A verify
///   EDT arts_abort()s on mismatch, so a wrong result is a non-zero exit code;
///   a stranded waiter is caught by the ctest TIMEOUT (no in-test watchdog).
///   OCR model (eager/lazy) only — the relaxed model has no ownership round.

#include "arts.h"

#include <stdio.h>

/* Phase A: write a per-DB marker (1) into each RW dep. */
void multi_rw_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  for (uint32_t i = 0; i < depc; i++) {
    int *d = (int *)depv[i].ptr;
    if (d == NULL) {
      (void)fprintf(stderr, "FAIL: multi_rw_edt slot %u NULL\n", i);
      arts_abort(1);
    }
    d[0] = 1;
  }
}

/* Phase A check: every DB observed its marker (RO deps read in parallel). */
void multi_ro_check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  int sum = 0;
  for (uint32_t i = 0; i < depc; i++) {
    int *d = (int *)depv[i].ptr;
    sum += (d ? d[0] : 0);
  }
  if (sum != (int)depc) {
    (void)fprintf(stderr, "FAIL A: expected %u got %d\n", depc, sum);
    arts_abort(1);
  }
  arts_printf("PASS A: %u cross-rank RW deps all written\n", depc);
}

/* Phase B: each same-rank RW EDT writes its OWN distinct slot of the shared DB.
 * ARTS RW is per-NODE exclusive (multiple local EDTs share the node's RW
 * ownership concurrently — the app orders intra-node), so two same-rank EDTs
 * incrementing the SAME word would race a read-modify-write; distinct slots
 * make the check race-free. The point is to exercise PROCEED advancing BOTH
 * local waiters parked on one remote DB. */
void inc_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int *d = (int *)depv[0].ptr;
  if (d == NULL) {
    (void)fprintf(stderr, "FAIL: inc_edt NULL\n");
    arts_abort(1);
  }
  d[(int)paramv[0]] = 1; /* this EDT's own slot */
}

/* Phase B check: both same-rank RW EDTs landed their distinct slots -> sum 2.
 */
void inc_check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *d = (int *)depv[0].ptr;
  int sum = d ? (d[0] + d[1]) : -1;
  if (sum != 2) {
    (void)fprintf(stderr, "FAIL B: expected 2 got %d\n", sum);
    arts_abort(1);
  }
  arts_printf("PASS B: two same-rank RW EDTs both landed (d[0]+d[1]=2)\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("=== coherence_rw_pipeline ===\n");

#ifdef ARTS_PROTOCOL_WRF_RCU
  arts_printf("SKIP: RELAXED has no exclusive-RW ownership round\n");
  arts_shutdown();
  return;
#endif

  unsigned int nranks = arts_get_total_ranks();
  if (nranks < 2) {
    arts_printf("SKIP: requires 2+ ranks (got %u)\n", nranks);
    arts_shutdown();
    return;
  }

  /* ---- Phase A: K DBs, one homed per rank; one EDT (rank 0) RW-acquires all.
   */
  unsigned int K = nranks;
  arts_guid_t dbs[64];
  if (K > 64) {
    K = 64;
  }
  for (unsigned int i = 0; i < K; i++) {
    void *p = NULL;
    dbs[i] = arts_db_create(&p, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE,
                            &(arts_db_hint_t){.rank = i});
    ((int *)p)[0] = 0;
    arts_db_release(dbs[i], DB_MODE_RW);
  }
  arts_guid_t ea = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t w =
      arts_edt_create(multi_rw_edt, 0, NULL, K,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = ea});
  for (unsigned int i = 0; i < K; i++) {
    arts_add_dependence(dbs[i], w, i, DB_MODE_RW);
  }
  arts_event_wait(ea);

  arts_guid_t ea2 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t chk =
      arts_edt_create(multi_ro_check_edt, 0, NULL, K,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = ea2});
  for (unsigned int i = 0; i < K; i++) {
    arts_add_dependence(dbs[i], chk, i, DB_MODE_RO);
  }
  arts_event_wait(ea2);

  /* ---- Phase B: one DB (2 ints) homed on rank 0; two RW EDTs both on rank 1,
   * each writing its OWN slot. Both park on the same remote DB's RW -> one
   * OWNERSHIP_REQUEST -> home PROCEEDs rank 1 -> BOTH local waiters advance. */
  void *pb = NULL;
  arts_guid_t dbb =
      arts_db_create(&pb, sizeof(int) * 2, ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((int *)pb)[0] = 0;
  ((int *)pb)[1] = 0;
  arts_db_release(dbb, DB_MODE_RW);

  arts_guid_t eb = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (int k = 0; k < 2; k++) {
    uint64_t slot_idx = (uint64_t)k;
    arts_guid_t iw =
        arts_edt_create(inc_edt, 1, &slot_idx, 1,
                        &(arts_edt_hint_t){.rank = 1, .finish_event = eb});
    arts_add_dependence(dbb, iw, 0, DB_MODE_RW);
  }
  arts_event_wait(eb);

  arts_guid_t eb2 = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t cb =
      arts_edt_create(inc_check_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = eb2});
  arts_add_dependence(dbb, cb, 0, DB_MODE_RO);
  arts_event_wait(eb2);

  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
