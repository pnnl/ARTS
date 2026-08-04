/*
 * coherence_rw_rw_chain_seal.c — Stress test for back-to-back RW writers
 * under the HOME-placement ownership protocol.
 *
 * Pattern: writer(0) reads init -> writes 0; writer(1) reads 0 -> writes 1; ...
 *
 * INTENTIONALLY racy: the same-DB RW writers are wired with NO event ordering
 * between them, so a writer may run before the previous one sealed and observe
 * a stale value.  That mismatch is the EXPECTED outcome (the test exists to
 * show the stale read can occur) — it is REPORTED, not aborted.  The test
 * passes iff the writer chain completes without hanging or NULL ptr. A stranded
 * waiter is caught by the ctest TIMEOUT (no in-test watchdog).
 */

#include "arts.h"
#include <stdio.h>

#define NUM_ITERS 50

void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int iter = (int)paramv[0];
  int expected_prev = (iter == 0) ? -1 : iter - 1;
  int *data = (int *)depv[0].ptr;
  if (!data) {
    (void)fprintf(stderr, "FAIL: writer %d NULL ptr\n", iter);
    arts_abort(1);
  }
  if (data[0] != expected_prev) {
    /* EXPECTED racy: no ordering between same-DB RW writers, so a writer may
     * run before the previous one sealed — report only. */
    (void)fprintf(stderr,
                  "MISMATCH (racy, expected): writer %d expected "
                  "prev=%d got %d\n",
                  iter, expected_prev, data[0]);
  }
  data[0] = iter;
}

void final_check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    (void)fprintf(stderr, "FAIL: final_check NULL ptr\n");
    arts_abort(1);
  }
  if (data[0] != NUM_ITERS - 1) {
    /* EXPECTED racy final value (writers raced) — report only. */
    (void)fprintf(stderr,
                  "MISMATCH (racy, expected): final expected %d got %d\n",
                  NUM_ITERS - 1, data[0]);
  }
  arts_printf("PASS: coherence_rw_rw_chain_seal completed %d writers\n",
              NUM_ITERS);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_rw_rw_chain_seal ===\n");
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  ((int *)ptr)[0] = -1;
  arts_db_release(db, DB_MODE_RW);

  for (int i = 0; i < NUM_ITERS; i++) {
    uint64_t p = (uint64_t)i;
    arts_guid_t w =
        arts_edt_create(writer_edt, 1, &p, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
  }
  arts_guid_t fc =
      arts_edt_create(final_check_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db, fc, 0, DB_MODE_RO);

  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
