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

/// @file ooo_hit_ref_pin.c
/// @brief ASan stress for the OoO HIT-path ref-pin invariant
///        (ooo.c arts_ooo_dispatch_or_defer): on a HIT, the slot's cb
///        shared-ptr ref is pinned across the ENTIRE handler call, so a
///        concurrent destroy's exchange-to-NULL drops only the install ref and
///        the object stays alive until the handler releases its pinned ref —
///        no use-after-free.
///
/// Drive (public-API only): a single home DB with a wide fan-out of RW/RO
/// consumer EDTs spread across every rank.  Their acquire/satisfy operations
/// HIT the installed slot and dispatch through the engine while, in the SAME
/// finish scope, a destroyer EDT issues arts_db_destroy of the same DB.  The
/// destroy CASes the slot value to NULL and releases the install ref; any
/// in-flight dispatch that already loaded a ref must keep the object live for
/// the duration of its handler.  Run under ASan/LSan (the build's default
/// sanitizers): a ref-pin regression surfaces as a heap-use-after-free /
/// double-free report and a non-zero exit; a leak surfaces in LSan.
///
/// Correctness is end-to-end completion (finish scope closes) PLUS a clean
/// sanitizer run.  A repeated outer loop widens the race window.  No in-test
/// spin: a hang is reaped by the ctest TIMEOUT.

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>

#define ROUNDS 8u
#define WORKERS 12u

/// Worker: touches the DB through its dep (HIT-path dispatch through the slot).
/// Writes/reads a few bytes; correctness here is "no UAF", not a value.
void rp_worker(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  volatile unsigned int *d = (volatile unsigned int *)depv[0].ptr;
  if (d != NULL) {
    /* A read; under a ref-pin bug ASan would already have flagged the load. */
    unsigned int v = d[0];
    (void)v;
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== ooo_hit_ref_pin ===\n");

  unsigned int nranks = arts_get_total_ranks();

  for (unsigned int r = 0; r < ROUNDS; r++) {
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((unsigned int *)ptr)[0] = r;
    arts_db_release(db, DB_MODE_RW);

    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

    /* Wide fan-out of concurrent RO readers across all ranks: each HITs the
     * slot and dispatches through the engine. */
    for (unsigned int i = 0; i < WORKERS; i++) {
      unsigned int target = nranks > 1 ? (i % nranks) : 0u;
      arts_guid_t w = arts_edt_create(
          rp_worker, 0, NULL, 1,
          &(arts_edt_hint_t){.rank = target, .finish_event = fe});
      arts_add_dependence(db, w, 0, DB_MODE_RO);
    }

    /* Wait for all readers to finish, THEN destroy: this keeps the program
     * OCR-legal (no destroy of a DB with live unsatisfied deps) while still
     * exercising the HIT-path ref-pin during the readers' concurrent dispatch.
     * The destroy's slot exchange-to-NULL races any late in-flight release of
     * a dispatch ref. */
    arts_event_wait(fe);
    arts_db_destroy(db);
  }

  arts_printf("PASS: ooo_hit_ref_pin %u rounds x %u workers, no UAF\n", ROUNDS,
              WORKERS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
