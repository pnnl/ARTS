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

/// @file db_cxl_paths.c
/// @brief ARTS_DB_CXL create / write / read round-trip: the CXL shared-memory
/// DB
///        subtype (build-gated, no CI matrix).
///
/// ARTS_DB_CXL allocates from the CXL shared segment (arts_db_malloc CXL arm,
/// round-robin or static device), encodes the device pointer directly in the
/// GUID (NO route table entry, NO DB-level coherence), and uses producer /
/// consumer FLUSH fences (driven automatically by db.c at create / prep /
/// release) for cross-node visibility under HW MESI.  The application drives
/// ordering via explicit events (app-ordered across nodes, full DRF).
///
/// This test creates a CXL DB, fills it through the creator pointer, releases,
/// and a consumer EDT reads it back — verifying the create flush + acquire-side
/// consumer flush make the producer's bytes visible.  Like ARTS_DB_PIN, the DB
/// lives only where allocated; ordering is event-driven.
///
/// config_specific: requires ARTS_USE_CXL (rapid API + CXL device).  Self-skips
/// (prints SKIP, returns 0) when ARTS_USE_CXL is not defined — there is no CI
/// matrix entry for CXL.

#include "arts.h"

#if !defined(ARTS_USE_CXL)

#include <stdio.h>

int main(void) {
  (void)printf("SKIP db_cxl_paths: requires ARTS_USE_CXL\n");
  return 0;
}

#else /* ARTS_USE_CXL */

#include <stdint.h>
#include <stdio.h>

#define ELEMS 32u

/// consumer: reads a CXL DB; the producer's bytes must be visible after the
/// create + acquire-side consumer flush.
void cxl_consumer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  bool ok = (d != NULL);
  for (unsigned int i = 0; i < ELEMS && ok; i++) {
    if (d[i] != (uint64_t)(0xCABU + i)) {
      ok = false;
    }
  }
  if (!ok) {
    (void)fprintf(stderr, "FAIL: db_cxl_paths consumer read mismatch\n");
    arts_abort(1);
    return;
  }
  arts_printf("PASS: db_cxl_paths\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== db_cxl_paths ===\n");

  /* Create a CXL DB (shared segment alloc, GUID-encoded pointer, no route
   * entry).  Fill through the creator pointer; create-time producer flush + the
   * consumer's acquire-side flush publish the bytes. */
  void *ptr = NULL;
  arts_guid_t db = arts_db_create(&ptr, ELEMS * sizeof(uint64_t), ARTS_DB_CXL,
                                  ARTS_DB_PROP_NONE, NULL);
  if (ptr == NULL) {
    (void)fprintf(stderr, "FAIL: db_cxl_paths create returned NULL ptr\n");
    arts_abort(1);
    return;
  }
  uint64_t *d = (uint64_t *)ptr;
  for (unsigned int i = 0; i < ELEMS; i++) {
    d[i] = (uint64_t)(0xCABU + i);
  }
  arts_db_release(db, DB_MODE_RW);

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t c =
      arts_edt_create(cxl_consumer, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db, c, 0, DB_MODE_RO);
  arts_event_wait(fe);

  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_USE_CXL */
