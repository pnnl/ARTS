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

/// @file writeback_ack_post_on_miss.c
/// @brief T109 — WRITEBACK_ACK posts the releaser's sem on HIT and MISS
///        (B017/B018).
///
/// The eager (and WRF_RCU) RW release performs a synchronous WRITEBACK round: the
/// releaser blocks on a stack-local sem_t whose address is carried in the wire
/// `cv` field; the home posts that sem via WRITEBACK_ACK by pointer identity.
/// The Cat-C SPECIAL invariant is that the ACK posts the sem on BOTH HIT and
/// MISS — if the home cache is torn down (destroyed) concurrently with the ACK
/// and the handler dropped the post on a MISS, the blocked releaser would be
/// stranded forever → distributed hang.  The abuf-free vs dispatch_or_defer
/// copy contract (the WRITEBACK self-send frees its heap args immediately) is
/// the UAF guard exercised on the home-self path.
///
/// Black-box driver: a long cross-rank RW ping-pong forces a writeback+ACK
/// round on every ownership handoff (home redirects owner→owner; the shedding
/// owner's release blocks on its stack sem until the ACK posts).  Concurrently
/// the same DBs are destroyed and recreated each generation so an ACK can race
/// a torn-down home cache (the MISS path).  If any ACK is dropped the releaser
/// never wakes and the finish scope never drains → ctest TIMEOUT FAIL.
///
/// Config gate: WRITEBACK / WRITEBACK_ACK exist only under EAGER (RCU+EAGER,
/// and WRF_RCU — LAZY has no synchronous writeback (its dispatcher
/// fatals on the message) and RWLOCK uses LOCK_RELEASE_ACK instead.  Compile-time
/// self-skip on LAZY/RWLOCK.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if defined(ARTS_PROTOCOL_RWLOCK) || defined(ARTS_TIMING_LAZY)
int main(void) {
  printf(
      "SKIP writeback_ack_post_on_miss: EAGER (RCU+E) + WRF_RCU only\n");
  return 0;
}
#else

#define ITERS 300u

/// RW holder: bump the per-DB counter so each generation does a real write
/// (forcing a non-empty synchronous writeback payload on release).
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

  arts_printf("=== writeback_ack_post_on_miss ===\n");

  unsigned int nranks = arts_get_total_ranks();
  /* Cross-rank ownership handoff is what forces a real wire WRITEBACK + ACK.
   * On a single rank the self-send WRITEBACK_ACK path still posts the sem (the
   * HIT/MISS post is cache-independent), exercising the unconditional-post
   * contract without a wire round. */

  for (unsigned int it = 0; it < ITERS; it++) {
    /* Re-create the DB each generation: the destroy at the end of the previous
     * iteration can still be draining when this generation's writeback ACK is
     * in flight, opening the torn-down-home MISS window. */
    void *ptr = NULL;
    arts_guid_t db =
        arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB, ARTS_DB_PROP_NONE,
                       &(arts_db_hint_t){.rank = 0});
    ((unsigned int *)ptr)[0] = 0u;
    arts_db_release(db, DB_MODE_RW);

    /* RW ping-pong across every rank: each handoff sheds ownership with a
     * synchronous writeback; the releaser blocks on its stack sem until the
     * home's WRITEBACK_ACK posts it. */
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

    /* Destroy the home: drives the next generation's possible ACK-on-MISS. */
    arts_db_destroy(db);
  }

  arts_printf("PASS: writeback_ack_post_on_miss %u iters x %u ranks\n", ITERS,
              nranks);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* skip */
