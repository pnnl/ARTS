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

/// @file val_no_acquire.c
/// @brief NO_ACQUIRE create: home is the idle initial owner; the first
///        OWNERSHIP_REQUEST drives the wc 0->GRANT transition.
///
/// arts_db_create with ARTS_DB_PROP_NO_ACQUIRE: the creator EDT does NOT
/// auto-acquire RW; the home rank is the initial IDLE owner.  Under VAL the
/// creator-home cache initializes writer_count to the install sentinel (2) and
/// then withdraws the creator-EDT slot (-1) since there is no auto-acquire —
/// the home holds the data but is not "running" RW.  The FIRST RW acquirer's
/// OWNERSHIP_REQUEST makes home (the rw_holder) drive the round: a
/// self-targeted INVALIDATE-to-self brings the count through the 0 edge and
/// ships the GRANT to the requester.  A miscounted NO_ACQUIRE init (phantom
/// holder / count never reaching the 0 edge) strands the first acquirer
/// (distributed hang).
///
/// SCENARIO.  Create a NO_ACQUIRE home RW DataBlock (returns NULL payload, no
/// creator-token to release).  An initializer RW EDT on home stamps a seed,
/// then a chain of RW EDTs on alternating ranks each increment, then an RO
/// reader must observe the exact accumulated value.  The first RW acquirer
/// exercises the idle-home -> GRANT transition; the chain proves ownership
/// keeps transferring.  A stranded first acquirer hangs (ctest TIMEOUT); a lost
/// increment fails the exact check (arts_abort).
///
/// VAL-only, both placements (the NO_ACQUIRE creator-home init + first-request
/// transition is in arts_handler_db_create / start_grant_round, shared).
/// Clean skip otherwise.  1n: home idle owner, local acquires; still exact.

#include "arts.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_VAL)

int main(void) {
  printf("SKIP val_no_acquire: VAL-only\n");
  return 0;
}

#else

#define STEPS 48u
#define SEED 1000u

static void init_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (unsigned int)paramv[0];
  }
}

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
    (void)fprintf(stderr, "FAIL val_no_acquire: expected %u got %u\n", expect,
                  d ? d[0] : 0u);
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== val_no_acquire ===\n");

  unsigned int nranks = arts_get_total_ranks();

  /* NO_ACQUIRE: home is the idle initial owner; no creator-token, NULL ptr. */
  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB,
                     ARTS_DB_PROP_NO_ACQUIRE, &(arts_db_hint_t){.rank = 0});

  /* First RW acquirer: drives the idle-home -> GRANT transition + seeds. */
  uint64_t seed = (uint64_t)SEED;
  arts_guid_t ei = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t in = arts_edt_create(
      init_edt, 1, &seed, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = ei});
  arts_add_dependence(db, in, 0, DB_MODE_RW);
  arts_event_wait(ei);

  unsigned int added = 0u;
  for (unsigned int s = 0; s < STEPS; s++) {
    unsigned int rank = (nranks > 1) ? (s % nranks) : 0u;
    arts_guid_t e = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t w =
        arts_edt_create(inc_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = rank, .finish_event = e});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
    added++;
    arts_event_wait(e);
  }

  uint64_t expect = (uint64_t)(SEED + added);
  arts_guid_t ec = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t c =
      arts_edt_create(check_edt, 1, &expect, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = ec});
  arts_add_dependence(db, c, 0, DB_MODE_RO);
  arts_event_wait(ec);

  arts_printf("PASS val_no_acquire value=%u\n", (unsigned int)expect);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif
