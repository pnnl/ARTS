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

/// @file mrsw_acquire_fastpath.c
/// @brief MRSW arts_handler_db_acquire RW fast-path TOCTOU (config_specific).
///
/// The MRSW acquire handler reads is_owner (writer_count > 0) ONCE as a
/// fast-reject gate, then routes:
///   - is_owner==true  -> arts_db_acquire_rw_local_fast, which internally
///     re-reads writer_count in a loop and self-validates; it ALWAYS returns
///     true and the run path delivers dep->ptr, so the handler RETURNS WITHOUT
///     resolving (resolving here would double-account the dependency).
///   - is_owner==false -> arts_db_acquire_remote_rw, which push+kicks.
/// A stale is_owner read at the gate is harmless: a stale-true routes to the
/// self-rechecking fast path; a stale-false routes to remote_rw (correct push).
/// The hazard the test guards against is a double-account (a writer counted
/// twice, or once via both fast-path resolve and run-path deliver) or a lost
/// account (a writer that never runs).
///
/// Construction: a flat fan of RW writers split between the home rank (drives
/// the LOCAL is_owner==true fast path with rapid 1<->2 token reclaim) and a
/// remote rank (drives is_owner==false -> remote_rw, and flips ownership so the
/// home-side is_owner reads go stale).  Each writer increments by exactly 1;
/// final MUST equal the writer count.  A double/lost account -> wrong final or
/// a hang (ctest TIMEOUT).
///
/// MRSW only (the fast-path is MRSW ownership.c machinery).  config-agnostic on
/// rank count; on 1n all writers take the local fast path.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_MRSW)

int main(void) {
  printf("SKIP mrsw_acquire_fastpath: MRSW-only\n");
  return 0;
}

#else

#define N_WRITERS 192u

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

static void final_check_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  const uint64_t *d = (const uint64_t *)depv[0].ptr;
  uint64_t v = (d != NULL) ? d[0] : 0u;
  if (v != (uint64_t)N_WRITERS) {
    (void)fprintf(stderr,
                  "FAIL: final value %llu != %u (double/lost account)\n",
                  (unsigned long long)v, N_WRITERS);
    arts_abort(1);
  }
  arts_printf("PASS: mrsw_acquire_fastpath final=%u\n", N_WRITERS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== mrsw_acquire_fastpath ===\n");

  unsigned int nranks = arts_get_total_ranks();

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((uint64_t *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  /* Phase 1: alternate writers between home (local fast path, stale
   * is_owner==true window) and a remote rank (remote_rw, flips ownership so
   * home reads go stale).  No inter-writer ordering -> the token funnels them
   * and the is_owner gate is exercised under churn.  Waiting on the finish
   * event blocks main_edt until every writer has run AND written back. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (unsigned int i = 0; i < N_WRITERS; i++) {
    unsigned int rank = (nranks > 1u && (i & 1u)) ? 1u : 0u;
    arts_guid_t w =
        arts_edt_create(rw_inc_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = rank, .finish_event = fe});
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

#endif /* ARTS_PROTOCOL_MRSW */
