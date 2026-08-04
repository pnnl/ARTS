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

/// @file coherence_satisfy_before_release.c
/// @brief Add-dependence-then-release hand-off: a consumer whose dependence
/// was satisfied BEFORE the producer's release must observe the state the
/// producer published AT the release.
///
/// The legal producer idiom under test (write happens-before the satisfy):
///   create DB (home on ANOTHER rank)  ->  write payload  ->
///   add_dependence(db, consumer) [slot satisfies IMMEDIATELY]  ->
///   ... work ...  ->  release (the publication point).
///
/// Every consumer therefore becomes READY — and issues its acquire — long
/// before the DB's home rank has ever seen a byte of data (a cross-rank
/// create installs home METADATA only; the creator's first publish delivers
/// the first buffer).  The runtime must hold a home-side acquire (home-local
/// reader AND cross-rank GET_DATA reader alike) until the creator's release
/// publishes, then serve the released bytes.  Because the write
/// happened-before every satisfy, NO legal snapshot can miss it: a NULL ptr
/// or zero bytes at any consumer is a serve-before-publication bug (abort).
///
/// Placement forces every interesting leg deterministically:
///   - DB home = rank 1..N-1 round-robin, ALWAYS != rank 0 (the creator), so
///     home never has creator-seeded data;
///   - one consumer pinned ON the DB home (the home-local acquire leg);
///   - one consumer pinned on (home+1)%N (at >= 3 ranks this skips the
///     creator: the cross-rank GET_DATA leg against an empty home).
/// A deliberate producer-side delay between add_dependence and the release
/// widens the window so a serve-early bug fails EVERY run, not occasionally.
/// Failure modes: consumer NULL/zero read (FAIL print / abort), or a missing
/// PASS (hang = a parked acquire never woken).  Auto-skips on < 2 ranks.

#include "arts.h"

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define N_DBS 4
#define N_INTS 1024u /* 4 KiB — the bug is ordering, not payload size */

static inline int pattern(uint32_t d, uint32_t j) {
  return (int)((d + 1u) * 2654435761u + j * 0x9e3779b9u);
}

static void consumer_edt(uint32_t paramc, const uint64_t *paramv,
                         uint32_t depc, arts_edt_dep_t depv[]) {
  (void)depc;
  if (paramc < 1) {
    arts_printf("FAIL: consumer missing paramv\n");
    arts_abort(1);
  }
  uint32_t d = (uint32_t)paramv[0];
  arts_printf("consumer db=%u on rank %u\n", d, arts_get_current_rank());
  const int *data = (const int *)depv[0].ptr;
  if (data == NULL) {
    /* Served before the producer's release published a buffer. */
    arts_printf("FAIL: consumer %u got NULL ptr (served pre-release)\n", d);
    arts_abort(1);
  }
  for (uint32_t j = 0; j < N_INTS; j++) {
    if (data[j] != pattern(d, j)) {
      arts_printf("FAIL: consumer %u element %u: got %d expected %d "
                  "(pre-release state served)\n",
                  d, j, data[j], pattern(d, j));
      arts_abort(1);
    }
  }
}

static void shutdown_edt(uint32_t paramc, const uint64_t *paramv,
                         uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("PASS: %d DBs observed at their release-published value by "
              "every pre-satisfied consumer\n", N_DBS);
  arts_shutdown();
}

/* Bounded wall-clock delay: widens the satisfy->release window so a
 * serve-early bug loses the race deterministically instead of occasionally. */
static void spin_delay_ms(unsigned ms) {
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (;;) {
    clock_gettime(CLOCK_MONOTONIC, &t1);
    uint64_t el = (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ull +
                  (uint64_t)(t1.tv_nsec - t0.tv_nsec);
    if (el >= (uint64_t)ms * 1000000ull) {
      return;
    }
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int nnodes = arts_get_total_ranks();
  if (nnodes < 2) {
    arts_printf("SKIP: requires 2+ ranks (got %u)\n", nnodes);
    arts_shutdown();
    return;
  }

  arts_printf("=== coherence_satisfy_before_release (%d DBs, %u ranks) ===\n",
              N_DBS, nnodes);

  arts_guid_t shut =
      arts_edt_create(shutdown_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0});
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, shut, 0, DB_MODE_NULL);

  /* Step 1: create every DB with a NON-creator home and WRITE the pattern —
   * the write happens-before every satisfy below, so no legal snapshot may
   * miss it. */
  arts_guid_t dbs[N_DBS];
  for (uint32_t d = 0; d < N_DBS; d++) {
    unsigned int home = 1u + (d % (nnodes - 1u)); /* never rank 0 */
    void *raw = NULL;
    dbs[d] = arts_db_create(&raw, N_INTS * sizeof(int), ARTS_DB,
                            ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = home});
    int *data = (int *)raw;
    for (uint32_t j = 0; j < N_INTS; j++) {
      data[j] = pattern(d, j);
    }
  }

  /* Step 2: wire every consumer — each slot satisfies IMMEDIATELY, so every
   * consumer becomes ready and acquires while the DBs are still HELD (no
   * release yet; their homes have never seen a byte). */
  for (uint32_t d = 0; d < N_DBS; d++) {
    unsigned int home = 1u + (d % (nnodes - 1u));
    uint64_t dp = d;
    /* Consumer A: ON the DB home (home-local acquire against an empty home). */
    arts_guid_t ca =
        arts_edt_create(consumer_edt, 1, &dp, 1,
                        &(arts_edt_hint_t){.rank = home, .finish_event = fe});
    arts_add_dependence(dbs[d], ca, 0, DB_MODE_RO);
    /* Consumer B: a second reader on the next rank; at >= 3 ranks skip the
     * creator (its own cache already holds the written bytes — that serve is
     * legal at any time and is not the leg under test). */
    unsigned int other = (home + 1u) % nnodes;
    if (other == 0u && nnodes > 2u) {
      other = (home + 2u) % nnodes;
    }
    arts_guid_t cb =
        arts_edt_create(consumer_edt, 1, &dp, 1,
                        &(arts_edt_hint_t){.rank = other, .finish_event = fe});
    arts_add_dependence(dbs[d], cb, 0, DB_MODE_RO);
  }

  /* Step 3: widen the window — every consumer is satisfied, ready, and
   * (absent the hold-at-home rule) racing to acquire an empty home. */
  spin_delay_ms(200);

  /* Step 4: release — the publication point.  Home-parked acquires must be
   * served exactly the bytes written in step 1. */
  for (uint32_t d = 0; d < N_DBS; d++) {
    arts_db_release(dbs[d], DB_MODE_RW);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
