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

/// @file random_thread_safe.c
/// @brief Runtime exercise of arts_thread_safe_random() (libs/src/core/utils/
///        random.c).  Two properties are asserted:
///
///   (1) PER-THREAD INDEPENDENT STREAMS.  A thread's counter and key are
///       ARTS_THREAD_LOCAL, so every worker walks its own stream.  We fan out
///       many sampling EDTs (which the scheduler spreads across workers), have
///       each draw a short sequence, and require both intra-stream variation (a
///       worker's successive draws differ) and overall variation (the global
///       multiset is not a single repeated value).  This catches a shared or
///       uninitialised state regression.
///
///   (2) THE WHOLE WIDTH IS RANDOM.  A generator that only fills the lower half
///       -- or that widens a signed 32-bit draw, leaving bits 32..63 a copy of
///       the sign bit -- gives a high word that is only ever 0x00000000 or
///       0xffffffff, and a caller reducing modulo a bucket count then draws
///       from two disjoint residue classes instead of one uniform range.  We
///       require at least one sample whose high word is NEITHER of those two
///       values: impossible in either degenerate case, near-certain otherwise.
///
/// Single-node test; protocol-agnostic.  Completion is driven by a finish event
/// + arts_event_wait, never by spinning.

#include <stdint.h>

#include "arts.h"
#include "arts/utils/random.h" /* arts_thread_safe_random (internal util) */

#define NUM_SAMPLER_EDTS 32u
#define DRAWS_PER_EDT 8u

/// What one sampler recorded.  Each sampler owns its own slot and touches no
/// other: ARTS's RW is OCR RW — the write right belongs to the NODE, not to one
/// EDT — so two samplers can hold the block at the same time and a shared
/// read-modify-write would silently lose increments.  Folding is the checker's
/// job, once every sampler has finished.
typedef struct {
  uint64_t draws;       /* samples this sampler contributed                 */
  uint64_t first;       /* its first draw (for cross-sampler variation)     */
  uint64_t high_entropy; /* saw a high word that is neither 0 nor ~0        */
  uint64_t intra_varies; /* its own successive draws were not all equal     */
} sample_rec_t;

typedef struct {
  sample_rec_t rec[NUM_SAMPLER_EDTS];
} agg_t;

/// One sampling EDT: draws DRAWS_PER_EDT values on whatever worker runs it and
/// records them in its own slot (paramv[0]).
void rng_sampler(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  agg_t *a = (agg_t *)depv[0].ptr;
  if (!a) {
    arts_printf("  FAIL: sampler got NULL aggregation DB\n");
    return;
  }

  uint64_t local[DRAWS_PER_EDT];
  uint64_t local_high_entropy = 0;
  for (uint32_t i = 0; i < DRAWS_PER_EDT; i++) {
    local[i] = arts_thread_safe_random();
    uint64_t high = local[i] >> 32;
    if (high != 0u && high != 0xffffffffu) {
      local_high_entropy = 1;
    }
  }

  /* Did this worker's own stream produce more than one distinct value? */
  bool intra_varies = false;
  for (uint32_t i = 1; i < DRAWS_PER_EDT; i++) {
    if (local[i] != local[0]) {
      intra_varies = true;
      break;
    }
  }

  sample_rec_t *me = &a->rec[(uint32_t)paramv[0]];
  me->first = local[0];
  me->high_entropy = local_high_entropy;
  me->intra_varies = intra_varies ? 1u : 0u;
  me->draws = DRAWS_PER_EDT;
}

/// Runs after every sampler has folded in (joined the same finish scope via the
/// main EDT's wait).  Evaluates the two properties and prints the verdict.
void rng_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  agg_t *a = (agg_t *)depv[0].ptr;
  bool pass = true;

  /* Fold the per-sampler slots.  Every sampler has finished and released by the
     time this runs, so the records are complete and nobody else is writing. */
  uint64_t total_draws = 0;
  uint64_t intra_ok = 0;
  uint64_t high_entropy = 0;
  bool any_global_diff = false;
  if (a) {
    for (uint32_t i = 0; i < NUM_SAMPLER_EDTS; i++) {
      total_draws += a->rec[i].draws;
      intra_ok += a->rec[i].intra_varies;
      high_entropy |= a->rec[i].high_entropy;
      if (a->rec[i].first != a->rec[0].first) {
        any_global_diff = true;
      }
    }
  }

  if (!a || total_draws != (uint64_t)NUM_SAMPLER_EDTS * DRAWS_PER_EDT) {
    arts_printf("  FAIL: expected %lu draws, got %lu\n",
                (uint64_t)NUM_SAMPLER_EDTS * DRAWS_PER_EDT, total_draws);
    pass = false;
  }

  /* (1) Per-thread independent streams: variation across samplers plus at
   *     least one sampler whose own draws varied (catches a constant RNG). */
  if (a && !any_global_diff) {
    arts_printf("  FAIL: every sampler's first draw was identical (no RNG "
                "variation)\n");
    pass = false;
  } else {
    arts_printf("  PASS: thread-safe RNG produced varying values\n");
  }
  if (a && intra_ok == 0) {
    arts_printf(
        "  FAIL: no worker stream produced distinct successive draws\n");
    pass = false;
  }

  /* (2) The upper half must carry entropy of its own, not the sign bit. */
  if (a && !high_entropy) {
    arts_printf("  FAIL: every sample's high word was 0 or 0xffffffff — the "
                "upper half is a sign-extended copy, not random\n");
    pass = false;
  } else {
    arts_printf("  PASS: upper half carries its own entropy\n");
  }

  if (pass) {
    arts_printf("PASS random_thread_safe\n");
  } else {
    arts_printf("FAIL random_thread_safe\n");
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== random_thread_safe ===\n");

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(agg_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  agg_t *a = (agg_t *)ptr;
  for (uint32_t i = 0; i < NUM_SAMPLER_EDTS; i++) {
    a->rec[i].draws = 0;
    a->rec[i].first = 0;
    a->rec[i].high_entropy = 0;
    a->rec[i].intra_varies = 0;
  }
  arts_db_release(db, DB_MODE_RW);

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  for (uint32_t i = 0; i < NUM_SAMPLER_EDTS; i++) {
    uint64_t slot = (uint64_t)i;
    arts_guid_t s =
        arts_edt_create(rng_sampler, 1, &slot, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(db, s, 0, DB_MODE_RW);
  }

  /* Wait for all samplers to finish folding in, then run the checker which
   * reads the fully-aggregated record RO and shuts the runtime down. */
  arts_event_wait(fe);

  arts_guid_t chk = arts_edt_create(rng_check, 0, NULL, 1, NULL);
  arts_add_dependence(db, chk, 0, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
