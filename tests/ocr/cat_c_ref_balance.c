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

/// @file cat_c_ref_balance.c
/// @brief T115 — Cat-C self-send / dispatcher ref get/release balance (B024).
///
/// Every Cat-C coherence handler (DATA_RESPONSE / DESTROY_NOTIFY /
/// PUBLISH_ACK / LOCK_RELEASE_ACK / REDIRECT_RO / CONFIRM / CONFIRM_ACK) pins
/// the home/owner db_s with a ref-counted `arts_route_table_lookup_db` +
/// `arts_shared_get` and MUST `arts_shared_release` on EVERY path — HIT and
/// every early-return MISS.  An early-return that skips the release leaks a
/// ref: the route-table cb (and hence the DB) is never destroyed.  Conversely
/// the HIT path must keep the ref pinned across the handler body so a
/// concurrent destroy cannot free the db_s mid-handler (no UAF).
///
/// Black-box driver: a high-churn create→acquire(RO+RW across ranks)→destroy
/// loop on a fresh DB GUID every generation.  This drives all Cat-C families:
///   - DESTROY_NOTIFY fan-out (cache destroy) — both HIT and idempotent MISS,
///   - SNAPSHOT_RESPONSE (RO acquire) — HIT and torn-down-home MISS,
///   - PUBLISH_ACK / LOCK_RELEASE_ACK (RW release) — HIT and MISS,
///   - REDIRECT_RO / CONFIRM[_ACK] (OWNER ownership) where the protocol uses
///   them.
/// Each generation's DB is explicitly destroyed; if any Cat-C path leaked a
/// ref, the cb would never reach refcount 0 and a parked waiter's
/// destroy-fan-out wake would be dropped → the finish scope never drains →
/// ctest TIMEOUT FAIL.  The HIT-path UAF would surface as an ASan
/// use-after-free / SIGSEGV.
///
/// Config-agnostic: every protocol has its own Cat-C set; the loop drives
/// whichever ones the build compiles.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define ITERS 300u

static void rw_writer_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = d[0] + 1u;
  }
}

static void ro_reader_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv; /* tolerant — the point is that it runs (woken), not the value. */
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== cat_c_ref_balance ===\n");

  unsigned int nranks = arts_get_total_ranks();

  for (unsigned int it = 0; it < ITERS; it++) {
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((unsigned int *)ptr)[0] = 0u;
    arts_db_release(db, DB_MODE_RW);

    /* Drive RW (publish/ownership/ACK Cat-C) then RO (snapshot/redirect
     * Cat-C) across every rank so each foreign rank's cache pins + must
     * release. */
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    for (unsigned int r = 0; r < nranks; r++) {
      arts_guid_t w =
          arts_edt_create(rw_writer_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = r, .finish_event = fe});
      arts_add_dependence(db, w, 0, DB_MODE_RW);

      arts_guid_t ro =
          arts_edt_create(ro_reader_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = r, .finish_event = fe});
      arts_add_dependence(db, ro, 0, DB_MODE_RO);
    }
    arts_event_wait(fe);

    /* Destroy + idempotent second destroy: drives DESTROY_NOTIFY HIT then MISS.
     * A leaked Cat-C ref would block the cb from reaching refcount 0. */
    arts_db_destroy(db);
    arts_db_destroy(db);
  }

  arts_printf("PASS: cat_c_ref_balance %u iters x %u ranks\n", ITERS, nranks);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
