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

/// @file excl_release_ack_post_on_miss.c
/// @brief T110 — LOCK_RELEASE_ACK cv-guarded post, even on a cache MISS
///        (B018).
///
/// Under the EXCL protocol an RW release publishes to the home and blocks on
/// a stack-local sem_t whose address rides the wire `cv` field; the home wakes
/// the releaser via LOCK_RELEASE_ACK by pointer identity.  Unlike the HOME
/// PUBLISH_ACK (which posts unconditionally), the LOCK_RELEASE_ACK path
/// guards the post with `cv != 0` — both the self-send shortcut and the RX
/// dispatcher. The invariant under test: the post still happens on a cache MISS
/// (home cache torn down concurrently) so long as cv != 0; otherwise the
/// blocked RW releaser is stranded → distributed hang.
///
/// Black-box driver: a long cross-rank RW ping-pong forces a EXCL release +
/// LOCK_RELEASE_ACK on every ownership handoff; the same DBs are destroyed and
/// recreated each generation so an ACK can race a torn-down home (the MISS).  A
/// stranded releaser is caught by the ctest TIMEOUT (no in-test spin).
///
/// Config gate: LOCK_RELEASE_ACK exists only under the EXCL protocol; every
/// other protocol uses PUBLISH_ACK (HOME/WRF_VAL) or async transfer (OWNER).
/// Compile-time self-skip on non-EXCL.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_EXCL)
int main(void) {
  printf("SKIP excl_release_ack_post_on_miss: EXCL-only\n");
  return 0;
}
#else

#define ITERS 300u

/// RW holder: bump the per-DB counter (a real write so the EXCL release does a
/// non-empty publish before its LOCK_RELEASE_ACK).
static void rw_bump_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                        arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *d = (unsigned int *)depv[0].ptr;
  if (d != NULL) {
    d[0] = d[0] + 1u;
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== excl_release_ack_post_on_miss ===\n");

  unsigned int nranks = arts_get_total_ranks();

  for (unsigned int it = 0; it < ITERS; it++) {
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((unsigned int *)ptr)[0] = 0u;
    arts_db_release(db, DB_MODE_RW);

    /* RW ping-pong: each handoff releases the lock (publish + ACK).  The
     * releaser blocks on its stack sem until the home posts LOCK_RELEASE_ACK.
     */
    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    unsigned int hops = (nranks > 1) ? (2u * nranks) : 2u;
    for (unsigned int h = 0; h < hops; h++) {
      unsigned int target = (nranks > 1) ? (h % nranks) : 0u;
      arts_guid_t e = arts_edt_create(
          rw_bump_edt, 0, NULL, 1,
          &(arts_edt_hint_t){.rank = target, .finish_event = fe});
      arts_add_dependence(db, e, 0, DB_MODE_RW);
    }
    arts_event_wait(fe);

    arts_db_destroy(db);
  }

  arts_printf("PASS: excl_release_ack_post_on_miss %u iters x %u ranks\n",
              ITERS, nranks);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}

#endif /* ARTS_PROTOCOL_EXCL */
