/*
 * coherence_ro_accumulation.c — Pure RO accumulation stress test.
 *
 * Pattern: one RW writer sets value V, then NUM_READERS RO readers verify
 * V.  No writers between readers.  Catches bugs where RO readers fail to
 * join an open RO generation or where RO -> RO transitions drop readers.
 *
 * Any mismatch calls arts_abort(1) for a non-zero exit code.
 * A stranded waiter is caught by the ctest TIMEOUT (no in-test watchdog).
 */

#include "arts.h"
#include <stdio.h>

#define NUM_READERS 50
#define MAGIC_VALUE 0xDEADBEEF

void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  if (data) {
    data[0] = MAGIC_VALUE;
  }
}

void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  if (data == NULL || data[0] != MAGIC_VALUE) {
    (void)fprintf(stderr, "FAIL: reader_edt expected 0x%x got 0x%x\n",
                  MAGIC_VALUE, data ? data[0] : 0u);
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_ro_accumulation ===\n");

  void *ptr = NULL;
  arts_guid_t db = arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB,
                                  ARTS_DB_PROP_NONE, NULL);
  ((unsigned int *)ptr)[0] = 0;
  arts_db_release(db, DB_MODE_RW);

  /* Phase 1: the RW writer sets V.  Its own finish event establishes the
   * happens-before edge writer -> readers — the readers are created only AFTER
   * the writer's write has committed, so every RO snapshot observes V.  Without
   * this edge the RW writer and the RO readers are unordered and a reader may
   * observe the pre-write value (a happens-before omission in the test, not a
   * runtime defect: the model does not order unrelated RW/RO accesses). */
  arts_guid_t we = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t w =
      arts_edt_create(writer_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = we});
  arts_add_dependence(db, w, 0, DB_MODE_RW);
  arts_event_wait(we);

  /* Phase 2: NUM_READERS concurrent RO readers, all causally after the writer;
   * each verifies it observes V (concurrent-readers-see-committed-value). */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  for (int r = 0; r < NUM_READERS; r++) {
    arts_guid_t rd =
        arts_edt_create(reader_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(db, rd, 0, DB_MODE_RO);
  }

  arts_event_wait(fe);
  arts_printf("PASS: coherence_ro_accumulation %d readers all verified\n",
              NUM_READERS);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
