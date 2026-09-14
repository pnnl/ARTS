/* SPDX-License-Identifier: Apache-2.0
 *
 * Hand-back followed by a re-grant to the SAME rank while the previous turn's
 * releaser is still between its publish acknowledgement and its settle.
 *
 * On every rank but the home, several writer CHAINS advance one block: each
 * chain is a strictly sequential run of RW turns (the next turn depends on the
 * previous turn's output event).  Independent chains may hold RW concurrently
 * on one rank, so each increment is atomic.  A turn that finds itself the rank's
 * only holder hands the write right back with its own publish; the rank's
 * other chain then re-requests it, and the home re-grants it to the same rank
 * as soon as the hand-back has landed — while the first turn's releaser may
 * not have run its post-publish bookkeeping yet.  Every non-home rank does
 * this at once, so the right also moves between ranks under the same
 * conditions.  The home rank holds no chain: its idle edge serves rather than
 * hands back, and that path is exercised elsewhere.
 *
 * The oracle is an exact sum: every turn adds exactly one, so a hand-back
 * that lets the next holder start from a stale copy shows as a short count,
 * and a commit that arrives after the right has moved on is caught by the
 * runtime itself.  Policy-independent by construction; not under the DB-WRF
 * memory model, whose contract leaves unordered write-write turns undefined.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>

#include "arts.h"
#include "../test_failure_status.h"

#define CHAINS_PER_RANK 2u
#define STEPS_PER_CHAIN 150u

static void turn_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  _Atomic uint64_t *p = depv[0].ptr;
  if (p == NULL) {
    (void)fprintf(stderr, "FAIL: grant_purge_handback_regrant turn got no storage\n");
    arts_test_fail();
    arts_shutdown();
    return;
  }
  atomic_fetch_add_explicit(p, 1u, memory_order_relaxed);
}

static void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t want = paramv[0];
  const _Atomic uint64_t *p = depv[0].ptr;
  uint64_t got = (p != NULL) ? atomic_load_explicit(p, memory_order_relaxed) : 0u;
  if (got != want) {
    (void)fprintf(stderr,
                  "FAIL: grant_purge_handback_regrant counter=%" PRIu64
                  " want %" PRIu64 "\n",
                  got, want);
    arts_test_fail();
  } else {
    arts_printf("PASS: grant_purge_handback_regrant %" PRIu64 " turns\n", got);
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== grant_purge_handback_regrant ===\n");

  unsigned int nranks = arts_get_total_ranks();
  void *p = NULL;
  arts_guid_t db = arts_db_create(&p, sizeof(_Atomic uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                        &(arts_db_hint_t){.rank = 0u});
  atomic_init((_Atomic uint64_t *)p, 0u);
  arts_db_release(db, DB_MODE_RW);

  /* A single rank exercises the concurrent local writer case. */
  unsigned int first = (nranks > 1u) ? 1u : 0u;
  unsigned int nchains = (nranks - first) * CHAINS_PER_RANK;
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (unsigned int r = first; r < nranks; r++) {
    for (unsigned int c = 0; c < CHAINS_PER_RANK; c++) {
      arts_guid_t prev = NULL_GUID;
      for (unsigned int s = 0; s < STEPS_PER_CHAIN; s++) {
        arts_guid_t oe = arts_event_create(&ARTS_EVENT_HINT_LATCH(1));
        arts_guid_t t = arts_edt_create(
            turn_edt, 0, NULL, (prev == NULL_GUID) ? 1u : 2u,
            &(arts_edt_hint_t){.rank = r, .output_event = oe,
                               .finish_event = fe});
        arts_add_dependence(db, t, 0, DB_MODE_RW);
        if (prev != NULL_GUID) {
          arts_add_dependence(prev, t, 1, DB_MODE_NULL);
        }
        prev = oe;
      }
    }
  }
  uint64_t vp[1] = {(uint64_t)nchains * (uint64_t)STEPS_PER_CHAIN};
  arts_guid_t v =
      arts_edt_create(verify_edt, 1, vp, 2, &(arts_edt_hint_t){.rank = 0u});
  arts_add_dependence(db, v, 0, DB_MODE_RO);
  arts_add_dependence(fe, v, 1, DB_MODE_NULL);
}

int main(int argc, char **argv) {
  int rc = arts_rt(argc, argv);
  return rc ? 1 : arts_test_status();
}
