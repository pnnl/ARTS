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
///   (1) PER-THREAD INDEPENDENT STREAMS.  arts_thread_info.drand_buf[3] is
///       ARTS_THREAD_LOCAL, so every worker thread owns its own jrand48 xsubi
///       state.  We fan out many sampling EDTs (which the scheduler spreads
///       across workers), have each draw a short sequence, and require both
///       intra-stream variation (a worker's successive draws differ) and
///       overall variation (the global multiset is not a single repeated
///       value).  This catches a shared/uninitialised state regression.
///
///   (2) SIGN-EXTENSION REGRESSION (targets B129).  jrand48() returns a SIGNED
///       long in [-2^31, 2^31).  random.c does `(uint64_t)temp`, so a negative
///       draw sign-extends: bits 32..63 all become 1.  A caller expecting a
///       uniform unsigned value (e.g. `value % N` bucket selection) is then
///       badly skewed.  The correct behaviour is the unsigned 32-bit value
///       `(uint64_t)(uint32_t)temp`, whose high 32 bits are always zero.  We
///       therefore assert that NO sample has any of bits 32..63 set.  Under
///       the current sign-extending implementation roughly half the samples
///       (the negative ones) light up the high word, so this assertion FAILS
///       on purpose until B129 is fixed.  This is an exposes-bug test: it has
///       no PASS_REGULAR_EXPRESSION and must visibly print FAIL.
///
/// Single-node test; protocol-agnostic (plain DB RW aggregation).  Completion
/// is driven by a finish event + arts_event_wait, never by spinning.

#include <stdint.h>

#include "arts.h"
#include "arts/utils/random.h" /* arts_thread_safe_random (internal util) */

#define NUM_SAMPLER_EDTS 32u
#define DRAWS_PER_EDT 8u

/// Shared aggregation record (single DB, RW-serialised across samplers).
typedef struct {
  uint64_t total_draws;     /* how many samples contributed                 */
  uint64_t high_bits_or;    /* OR of bits 32..63 across every sample        */
  uint64_t distinct_first;  /* first sample value seen (for global variation) */
  uint64_t any_global_diff; /* set if any sample differs from distinct_first */
  uint64_t
      intra_stream_ok; /* count of EDTs whose own draws were not all equal */
  uint64_t intra_stream_bad;
} agg_t;

/// One sampling EDT: draws DRAWS_PER_EDT values on whatever worker runs it and
/// folds them into the shared aggregation DB (acquired RW = per-node exclusive,
/// so the read-modify-write below is serialised).
void rng_sampler(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  agg_t *a = (agg_t *)depv[0].ptr;
  if (!a) {
    arts_printf("  FAIL: sampler got NULL aggregation DB\n");
    return;
  }

  uint64_t local[DRAWS_PER_EDT];
  uint64_t local_high_or = 0;
  for (uint32_t i = 0; i < DRAWS_PER_EDT; i++) {
    local[i] = arts_thread_safe_random();
    local_high_or |= (local[i] >> 32);
  }

  /* Did this worker's own stream produce more than one distinct value? */
  bool intra_varies = false;
  for (uint32_t i = 1; i < DRAWS_PER_EDT; i++) {
    if (local[i] != local[0]) {
      intra_varies = true;
      break;
    }
  }

  if (a->total_draws == 0) {
    a->distinct_first = local[0];
  } else {
    for (uint32_t i = 0; i < DRAWS_PER_EDT; i++) {
      if (local[i] != a->distinct_first) {
        a->any_global_diff = 1;
        break;
      }
    }
  }

  a->total_draws += DRAWS_PER_EDT;
  a->high_bits_or |= local_high_or;
  if (intra_varies) {
    a->intra_stream_ok += 1;
  } else {
    a->intra_stream_bad += 1;
  }
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

  if (!a || a->total_draws != (uint64_t)NUM_SAMPLER_EDTS * DRAWS_PER_EDT) {
    arts_printf("  FAIL: expected %lu draws, got %lu\n",
                (uint64_t)NUM_SAMPLER_EDTS * DRAWS_PER_EDT,
                a ? a->total_draws : 0);
    pass = false;
  }

  /* (1) Per-thread independent streams: overall variation + at least one
   *     worker stream that itself varied (catches a degenerate constant RNG).
   */
  if (a && a->any_global_diff == 0) {
    arts_printf("  FAIL: all %lu samples were identical (no RNG variation)\n",
                a->total_draws);
    pass = false;
  } else {
    arts_printf("  PASS: thread-safe RNG produced varying values\n");
  }
  if (a && a->intra_stream_ok == 0) {
    arts_printf(
        "  FAIL: no worker stream produced distinct successive draws\n");
    pass = false;
  }

  /* (2) Sign-extension regression (B129).  Correct impl keeps bits 32..63 at
   *     zero for every draw.  Sign-extension lights them up for negatives. */
  if (a && a->high_bits_or != 0) {
    arts_printf("  FAIL: B129 sign-extension: high 32 bits set (or=0x%lx) — "
                "jrand48 negative draws sign-extend to huge uint64\n",
                a->high_bits_or);
    pass = false;
  } else {
    arts_printf("  PASS: no sign-extension; high 32 bits clear\n");
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
  a->total_draws = 0;
  a->high_bits_or = 0;
  a->distinct_first = 0;
  a->any_global_diff = 0;
  a->intra_stream_ok = 0;
  a->intra_stream_bad = 0;
  arts_db_release(db, DB_MODE_RW);

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  for (uint32_t i = 0; i < NUM_SAMPLER_EDTS; i++) {
    arts_guid_t s =
        arts_edt_create(rng_sampler, 0, NULL, 1,
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
