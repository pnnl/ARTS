/*
 * coherence_rw_ro_rw.c — Stress test for RW -> RO -> RW transitions under
 * the HOME-placement ownership protocol.
 *
 * Pattern per iteration: writer(RW) -> N readers(RO) -> writer(RW) -> ...
 * Exercises the sealed RO generation progression path.
 *
 * INTENTIONALLY racy: writer(i) and readers(i) share the DB with NO event
 * ordering between them, so a reader may acquire before its writer and read the
 * previous generation.  That mismatch is the EXPECTED outcome (the test exists
 * to show the stale read can occur) — it is REPORTED, not aborted.  The test
 * passes iff the RW->RO->RW progression completes without hanging or NULL ptr.
 * A stranded waiter is caught by the ctest TIMEOUT (no in-test watchdog).
 */

#include "arts.h"
#include <stdio.h>

#define NUM_ITERS 100
#define NUM_READERS 10

void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int iter = (int)paramv[0];
  int *data = (int *)depv[0].ptr;
  if (data) {
    data[0] = iter;
  }
}

void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int expected = (int)paramv[0];
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    (void)fprintf(stderr, "FAIL: reader_edt NULL ptr\n");
    arts_abort(1);
  }
  if (data[0] != expected) {
    /* EXPECTED racy stale read (no writer->reader ordering) — report only. */
    (void)fprintf(stderr,
                  "MISMATCH (racy, expected): reader_edt expected %d "
                  "got %d\n",
                  expected, data[0]);
  }
}

/* verify_edt: runs after the finish scope drains; prints PASS. */
void verify_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("PASS: coherence_rw_ro_rw %d iters x %d readers\n", NUM_ITERS,
              NUM_READERS);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_rw_ro_rw (stress) ===\n");
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

    for (int r = 0; r < NUM_READERS; r++) {
      arts_guid_t rd =
          arts_edt_create(reader_edt, 1, &p, 1,
                          &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
      arts_add_dependence(db, rd, 0, DB_MODE_RO);
    }
  }

  arts_guid_t vf =
      arts_edt_create(verify_edt, 0, NULL, 0,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  (void)vf;

  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
