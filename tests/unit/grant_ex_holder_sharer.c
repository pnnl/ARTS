/* SPDX-License-Identifier: Apache-2.0
 *
 * After a grant moves, the rank that gave it up is a SHARER.
 *
 * A migrating grant leaves the ex-holder holding the bytes it wrote — it does
 * not drop them, because dropping them would forfeit the free re-read that
 * makes the sharer plane worth having. So from the flip onward that rank is an
 * ordinary reader copy, and the next owner's first release must retire it like
 * any other. Miss that registration and the ex-holder reads its own stale
 * bytes forever: nothing else in the protocol ever targets it.
 *
 * Shape: rank 0 writes, then rank 1 takes the grant and writes twice more, then
 * rank 0 READS. Rank 0's copy predates both of rank 1's rounds, so a correct
 * implementation must have invalidated it and rank 0 must refetch. Everything
 * is event-ordered, so the read has exactly one legal answer.
 *
 * SCOPE, measured, not assumed. This pins the invariant but is NOT a
 * regression test for the timing bug that motivated writing it (a round that
 * self-excluded `rw_holder` — stale until the CONFIRM — instead of the ranks
 * that actually wrote). Re-introducing that bug leaves this test passing 10/10:
 * its handoffs are event-ordered and quiescent, so the directory has always
 * caught up by the time the next round opens. The window only opens when a new
 * owner closes a round BEFORE the home processes its CONFIRM, which needs
 * concurrent home-side dispatch — `coherence_writer_ro_overlap_2n_io` is the
 * only test that produces it (6/10 with the bug restored). Keep both: this one
 * states the invariant in a form a reader can check, that one catches it.
 */

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define ROUNDS 2

static arts_guid_t g_db;

static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    (void)fprintf(stderr, "FAIL grant_ex_holder_sharer: writer got NULL ptr\n");
    arts_abort(1);
  }
  data[0] = data[0] + 1;
}

static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int want = (int)paramv[0];
  const int *data = (const int *)depv[0].ptr;
  if (data == NULL) {
    (void)fprintf(stderr, "FAIL grant_ex_holder_sharer: reader got NULL ptr\n");
    arts_abort(1);
  }
  if (data[0] != want) {
    (void)fprintf(stderr,
                  "FAIL grant_ex_holder_sharer: ex-holder read %d, expected "
                  "%d — its copy survived the new owner's release rounds\n",
                  data[0], want);
    arts_abort(1);
  }
  printf("PASS grant_ex_holder_sharer: ex-holder was invalidated and refetched "
         "(%d)\n", data[0]);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  if (arts_get_total_ranks() < 2) {
    printf("SKIP grant_ex_holder_sharer: needs 2+ ranks (the grant has to "
           "leave the rank that later reads)\n");
    arts_shutdown();
    return;
  }

  void *ptr = NULL;
  g_db = arts_db_create(&ptr, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE,
                        &(arts_db_hint_t){.rank = 0});
  ((int *)ptr)[0] = 0;
  arts_db_release(g_db, DB_MODE_RW);

  /* Rank 0 writes first, so it holds the grant and a copy of round 1. */
  arts_guid_t e = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t w0 = arts_edt_create(writer_edt, 0, NULL, 1,
                                   &(arts_edt_hint_t){.rank = 0,
                                                      .finish_event = e});
  arts_add_dependence(g_db, w0, 0, DB_MODE_RW);
  arts_guid_t prev = e;

  /* Rank 1 then takes the grant and releases repeatedly.  Each of its rounds
   * must retire rank 0's copy; only the registration timing decides whether
   * the FIRST one does, which is what this test pins. */
  for (unsigned int i = 0; i < ROUNDS; i++) {
    arts_guid_t ei = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t w = arts_edt_create(writer_edt, 0, NULL, 2,
                                    &(arts_edt_hint_t){.rank = 1,
                                                       .finish_event = ei});
    arts_add_dependence(g_db, w, 0, DB_MODE_RW);
    arts_add_dependence(prev, w, 1, DB_MODE_NULL);
    prev = ei;
  }

  /* Back on rank 0 — the ex-holder — read.  Its cached copy is two rounds
   * behind; the only legal answer is the full count. */
  uint64_t want = (uint64_t)(1 + ROUNDS);
  arts_guid_t r = arts_edt_create(reader_edt, 1, &want, 2,
                                  &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(g_db, r, 0, DB_MODE_RO);
  arts_add_dependence(prev, r, 1, DB_MODE_NULL);
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
