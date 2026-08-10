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

/// @file val_wt_publish_ack.c
/// @brief Exercise the HOME synchronous PUBLISH ACK rendezvous as a unit.
///
/// Under HOME, arts_db_release_rw on a NON-home owner performs a SYNCHRONOUS
/// PUBLISH to home: it builds a stack-local `sem_t cv`, ships its address
/// over the wire (arts_send_db_publish), then blocks in await_publish_ack
/// until arts_handler_db_publish_ack sem_posts it.  The dispatcher posts the
/// ACK on BOTH the home HIT and the home-torn-down MISS so the blocked releaser
/// is never stranded.  Fragile surfaces: (a) sem_t pointer-identity (the ACK
/// must post the exact stack address that is still live); (b) await returns
/// early on the shutdown flag; (c) a late ACK arriving after the frame is
/// reclaimed.
///
/// SCENARIO.  A home-resident RW DataBlock; many remote RW writer EDTs each
/// stamp a per-step value and RELEASE — each release is a full PUBLISH->ACK
/// round.  After each remote write, a home RO reader must observe that exact
/// value (home is canonical under HOME only because the synchronous PUBLISH
/// landed before the reader runs).  Densely repeated to hammer the rendezvous;
/// any stranded releaser hangs (caught by ctest TIMEOUT), any lost/late ACK
/// corrupts the observed value (arts_abort).
///
/// HOME-only (the synchronous PUBLISH + stack-sem path exists ONLY in
/// home.c; owner.c drops the buffer ref and never blocks on an ACK).  Compiles
/// to a clean skip under any other configuration.  Needs 2+ ranks for a remote
/// owner; at 1n the owner is home (no PUBLISH) and the chain still passes.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_VAL) || !defined(ARTS_WRITE_POLICY_WT)

int main(void) {
  printf("SKIP val_wt_publish_ack: VAL+HOME-only\n");
  return 0;
}

#else

#define STEPS 200u
#define VAL_BASE 0xE0000000u

static void wr_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = (unsigned int)paramv[0];
  }
}

static void rd_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  unsigned int expect = (unsigned int)paramv[0];
  if (d == NULL || d[0] != expect) {
    (void)fprintf(stderr,
                  "FAIL val_wt_publish_ack: expected 0x%x got 0x%x\n",
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

  arts_printf("=== val_wt_publish_ack ===\n");

  unsigned int nranks = arts_get_total_ranks();
  unsigned int W = (nranks > 1) ? 1u : 0u; /* remote owner forces PUBLISH */

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  ((unsigned int *)ptr)[0] = 0u;
  arts_db_release(db, DB_MODE_RW);

  for (unsigned int s = 0; s < STEPS; s++) {
    uint64_t v = (uint64_t)(VAL_BASE + s);

    arts_guid_t ew = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t w = arts_edt_create(
        wr_edt, 1, &v, 1, &(arts_edt_hint_t){.rank = W, .finish_event = ew});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
    arts_event_wait(ew);

    arts_guid_t er = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t r = arts_edt_create(
        rd_edt, 1, &v, 1, &(arts_edt_hint_t){.rank = 0, .finish_event = er});
    arts_add_dependence(db, r, 0, DB_MODE_RO);
    arts_event_wait(er);
  }

  arts_printf("PASS val_wt_publish_ack %u steps\n", STEPS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}

#endif
