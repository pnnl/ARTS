/* SPDX-License-Identifier: Apache-2.0
 *
 * Write-right churn: every rank, including the block's home, keeps taking and
 * giving up turns on one block.
 *
 * The interesting interleavings all live where a home-LOCAL write turn
 * overlaps a remote one.  The home is the rank that hands turns out, so while
 * it is writing it cannot hand anything out, and a turn it takes for itself
 * never travels at all — the two are served by different paths that meet on
 * the same word and the same queue.  Overlapping them is what puts a remote
 * request in the queue behind a local writer, a hand-back in flight against
 * the home's own idle edge, and a request arriving at a home that has just
 * gone idle.
 *
 * The oracle is a count: every turn adds exactly one, so the total is exact
 * and any lost, doubled or reordered turn shows up as a wrong sum.  Rounds are
 * ordered against each other so the run cannot finish early with turns still
 * outstanding, while the turns WITHIN a round are unordered and race freely.
 *
 * Policy-independent by construction: it asserts what every release policy
 * must compute, not how the right travelled — which is why it runs everywhere
 * and is the vehicle for interleaving injection.
 */

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#include "../test_failure_status.h"

#define ROUNDS 8
#define TURNS_PER_ROUND 6

static arts_guid_t g_db;

static void turn_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *p = (uint64_t *)depv[0].ptr;
  if (p == NULL) {
    (void)fprintf(stderr, "FAIL: grant_purge_churn turn got no storage\n");
    arts_test_fail();
    arts_shutdown();
    return;
  }
  *p += 1u;
}

/* Opens the next round, or verifies once the last one has drained.  Chaining
 * from an EDT rather than wiring every round up front keeps each round's
 * turns concurrent with each other and with nothing else. */
static void round_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  unsigned int round = (unsigned int)paramv[0];
  unsigned int nranks = (unsigned int)paramv[1];

  if (round == ROUNDS) {
    arts_guid_t v = arts_edt_create(round_edt, 2,
                                    (uint64_t[]){ROUNDS + 1u, nranks}, 1,
                                    &(arts_edt_hint_t){.rank = 0u});
    arts_add_dependence(g_db, v, 0, DB_MODE_RO);
    return;
  }
  if (round > ROUNDS) {
    const uint64_t *p = (const uint64_t *)depv[0].ptr;
    uint64_t want = (uint64_t)ROUNDS * (uint64_t)TURNS_PER_ROUND;
    uint64_t got = (p != NULL) ? *p : 0u;
    if (got != want) {
      (void)fprintf(stderr,
                    "FAIL: grant_purge_churn counter=%llu want %llu\n",
                    (unsigned long long)got, (unsigned long long)want);
      arts_test_fail();
    } else {
      arts_printf("PASS: grant_purge_churn %d rounds x %d turns = %llu\n",
                  ROUNDS, TURNS_PER_ROUND, (unsigned long long)got);
    }
    arts_shutdown();
    return;
  }

  /* One round: turns spread over every rank, so rank 0 — the home — takes
   * turns of its own while the others are asking it for theirs. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (unsigned int i = 0; i < TURNS_PER_ROUND; i++) {
    unsigned int r = (i + round) % nranks;
    arts_guid_t t =
        arts_edt_create(turn_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = r, .finish_event = fe});
    arts_add_dependence(g_db, t, 0, DB_MODE_RW);
  }
  uint64_t np[2] = {(uint64_t)(round + 1u), (uint64_t)nranks};
  arts_guid_t next =
      arts_edt_create(round_edt, 2, np, 1, &(arts_edt_hint_t){.rank = 0u});
  arts_add_dependence(fe, next, 0, DB_MODE_NULL);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== grant_purge_churn ===\n");

  unsigned int nranks = arts_get_total_ranks();
  void *p = NULL;
  g_db = arts_db_create(&p, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                        &(arts_db_hint_t){.rank = 0u});
  *(uint64_t *)p = 0u;
  arts_db_release(g_db, DB_MODE_RW);

  uint64_t rp[2] = {0u, (uint64_t)nranks};
  arts_guid_t first =
      arts_edt_create(round_edt, 2, rp, 0, &(arts_edt_hint_t){.rank = 0u});
  (void)first;
}

int main(int argc, char **argv) {
  int rc = arts_rt(argc, argv);
  return rc ? 1 : arts_test_status();
}
