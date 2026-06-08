/* SPDX-License-Identifier: Apache-2.0
 *
 * Manual diagnostic — NOT a CTest.  ONE EDT per rank, each does N
 * increments in a tight loop on the same DB.  No DAG ordering between
 * the two EDTs.  DB is homed on rank 0; rank-0 EDT acquires the home
 * buffer locally, rank-1 EDT acquires a remote working copy.
 *
 * Observed behavior (both RC and LC builds):
 *   counter == N     (most runs, "lost-update")
 *   counter == 0     (rare, verifier ordering edge case)
 *   counter == 2N    (not observed in this configuration)
 *
 * RC's per-node-exclusive RW protocol (LOCK_REQ → INVALIDATE → GRANT)
 * fully serializes ownership transfer between two NON-HOME ranks, but
 * the home rank reading/writing its own buffer does not park on the
 * same chain — when one writer is home and the other is non-home, the
 * two execute concurrently and the non-home WRITEBACK overwrites the
 * home's local increments (or vice versa).  LC has no LOCK_REQ chain
 * by design and shows the same lost-update pattern.
 *
 * For a clean RC/LC differentiation see CTest: coherence_stress_dist
 * and coherence_lock_req_before_create both PASS in RC and FAIL in LC
 * because they exercise patterns that LC does not implement.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts.h"

#define INCREMENTS_PER_RANK 1000

static void incrementer_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *counter = (uint64_t *)depv[0].ptr;
  if (counter) {
    for (uint32_t i = 0; i < INCREMENTS_PER_RANK; i++) {
      *counter += 1;
    }
  }
}

static void verifier_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *counter = (uint64_t *)depv[0].ptr;
  uint64_t v = counter ? *counter : 0;
  uint64_t expected = (uint64_t)2 * INCREMENTS_PER_RANK;
  if (v == expected) {
    arts_printf("LC_COUNTER_RACE: counter=%lu (expected=%lu) — DETERMINISTIC "
                "(RC-like)\n",
                v, expected);
  } else if (v == INCREMENTS_PER_RANK) {
    arts_printf("LC_COUNTER_RACE: counter=%lu (expected=%lu) — RACED (LC-like, "
                "lost-update)\n",
                v, expected);
  } else {
    arts_printf("LC_COUNTER_RACE: counter=%lu (expected=%lu) — UNEXPECTED\n", v,
                expected);
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  if (arts_get_total_ranks() < 2) {
    arts_printf("LC_COUNTER_RACE: SKIP requires 2+ ranks\n");
    arts_shutdown();
    return;
  }

  void *addr = NULL;
  arts_guid_t db =
      arts_db_create(&addr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  *(uint64_t *)addr = 0;
  arts_db_release(db, DB_MODE_RW);

  arts_guid_t ver = arts_edt_create(verifier_edt, 0, NULL, 1, NULL);
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, ver, 1, DB_MODE_NULL);

  /* Exactly ONE EDT per rank — cross-rank race only. */
  for (unsigned int rank = 0; rank < 2; rank++) {
    arts_guid_t edt =
        arts_edt_create(incrementer_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = rank, .finish_event = fe});
    arts_add_dependence(db, edt, 0, DB_MODE_RW);
    (void)edt;
  }
  arts_add_dependence(db, ver, 0, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
