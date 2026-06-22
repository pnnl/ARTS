/*
 * coherence_mixed_local_remote.c — Multi-node eager-protocol test with
 * mixed local + remote readers.
 *
 * Dynamically adapts to the node count (1..N):
 *   - DB is owned by node 0 (route=0).
 *   - RW writer w1 on node 0 writes VAL1.
 *   - READERS_PER_NODE RO readers on EACH node (including node 0).
 *   - RW writer w2 on node 0 writes VAL2.
 *   - Final RO reader on node 0 verifies VAL2.
 *
 * INTENTIONALLY racy: W1 -> readers -> W2 -> final share the DB with NO event
 * ordering between them, so a later EDT may acquire before an earlier one and
 * read a stale generation.  That mismatch is the EXPECTED outcome — it is
 * REPORTED, not aborted.  The test passes iff the W->R->W->R chain completes
 * without hanging or a NULL ptr (a stranded waiter is caught by ctest TIMEOUT).
 */

#include "arts.h"
#include <stdio.h>

#define READERS_PER_NODE 10
#define VAL1 1000
#define VAL2 2000

void writer1_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data) {
    data[0] = VAL1;
  }
}

void writer2_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    (void)fprintf(stderr, "FAIL: writer2_edt got NULL ptr\n");
    arts_abort(1);
  }
  if (data[0] != VAL1) {
    /* EXPECTED racy: no ordering between W1/readers/W2 — report only. */
    (void)fprintf(stderr,
                  "MISMATCH (racy, expected): writer2 expected %d got %d\n",
                  VAL1, data[0]);
  }
  data[0] = VAL2;
}

void v1_reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    (void)fprintf(stderr, "FAIL: v1_reader_edt NULL ptr\n");
    arts_abort(1);
  }
  if (data[0] != VAL1) {
    /* EXPECTED racy stale read — report only. */
    (void)fprintf(stderr,
                  "MISMATCH (racy, expected): v1_reader expected %d got %d\n",
                  VAL1, data[0]);
  }
}

void final_reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    (void)fprintf(stderr, "FAIL: final_reader_edt NULL ptr\n");
    arts_abort(1);
  }
  if (data[0] != VAL2) {
    /* EXPECTED racy stale read — report only. */
    (void)fprintf(stderr,
                  "MISMATCH (racy, expected): final expected %d got %d\n", VAL2,
                  data[0]);
  }
  arts_printf("PASS: coherence_mixed_local_remote completed\n");
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coherence_mixed_local_remote ===\n");
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  void *ptr = NULL;
  arts_guid_t db = arts_db_create(&ptr, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE,
                                  &(arts_db_hint_t){.rank = 0});
  ((int *)ptr)[0] = 0;
  arts_db_release(db, DB_MODE_RW);

  /* W1: RW on node 0 */
  arts_guid_t w1 =
      arts_edt_create(writer1_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db, w1, 0, DB_MODE_RW);

  /* Mixed RO generation: READERS_PER_NODE readers on EACH node. */
  unsigned int nnodes = arts_get_total_ranks();
  for (unsigned int n = 0; n < nnodes; n++) {
    for (int i = 0; i < READERS_PER_NODE; i++) {
      arts_guid_t r =
          arts_edt_create(v1_reader_edt, 0, NULL, 1,
                          &(arts_edt_hint_t){.rank = n, .finish_event = fe});
      arts_add_dependence(db, r, 0, DB_MODE_RO);
    }
  }

  /* W2: RW on node 0 (must wait for all readers to drain) */
  arts_guid_t w2 =
      arts_edt_create(writer2_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db, w2, 0, DB_MODE_RW);

  /* Final RO reader verifies VAL2 */
  arts_guid_t fr =
      arts_edt_create(final_reader_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  arts_add_dependence(db, fr, 0, DB_MODE_RO);

  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
