/* A datablock created without the creator's auto-acquire must still hand its
 * FIRST consumer real storage.
 *
 * ARTS_DB_PROP_NO_ACQUIRE says the creator does not take the block and the
 * home rank is its initial idle owner, so the first consumer performs an
 * ordinary acquire to obtain ownership.  That consumer is entitled to a
 * writable pointer of the declared size: the block exists and has a size, and
 * "nobody has written it yet" is a statement about its CONTENTS, not about
 * whether it has any.
 *
 * Every consumer is placed off the home rank so the acquire travels the
 * ownership-transfer path, which is where an owner with no bytes of its own
 * decides what to hand over.  Both access modes are checked: a reader is as
 * entitled to storage as a writer.
 *
 * The block's GUID travels in paramv: an EDT body runs on whichever rank the
 * placement hint named, and a file-scope variable assigned by the creating EDT
 * exists only in that rank's process image.
 */

#include "arts.h"
#include "../test_failure_status.h"

#include <stdint.h>
#include <string.h>

#define DB_ELEMS 256u
#define DB_BYTES (DB_ELEMS * sizeof(uint64_t))

/* paramv = { db guid, home rank }. */
void consumer_rw(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  if (depv[0].ptr == NULL) {
    arts_printf("FAIL: RW acquire of a NO_ACQUIRE datablock returned no "
                "storage (rank %u)\n",
                arts_get_current_rank());
    arts_test_fail();
    arts_shutdown();
    return;
  }
  /* Writable for its whole declared extent. */
  uint64_t *p = (uint64_t *)depv[0].ptr;
  for (unsigned i = 0; i < DB_ELEMS; i++) {
    p[i] = i + 1u;
  }
  for (unsigned i = 0; i < DB_ELEMS; i++) {
    if (p[i] != i + 1u) {
      arts_printf("FAIL: datablock did not retain a written value\n");
      arts_test_fail();
      arts_shutdown();
      return;
    }
  }
  arts_printf("PASS: RW first touch of a NO_ACQUIRE datablock has storage\n");
  arts_shutdown();
}

/* paramv = { db guid, home rank }. */
void consumer_ro(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t db = (arts_guid_t)paramv[0];
  unsigned int home = (unsigned int)paramv[1];

  if (depv[0].ptr == NULL) {
    arts_printf("FAIL: RO acquire of a NO_ACQUIRE datablock returned no "
                "storage (rank %u)\n",
                arts_get_current_rank());
    arts_test_fail();
    arts_shutdown();
    return;
  }
  /* Readable for its whole declared extent (contents are undefined). */
  const volatile uint64_t *p = (const volatile uint64_t *)depv[0].ptr;
  uint64_t sink = 0;
  for (unsigned i = 0; i < DB_ELEMS; i++) {
    sink += p[i];
  }
  (void)sink;
  arts_printf("PASS: RO first touch of a NO_ACQUIRE datablock has storage\n");

  /* Now let a writer take it, exercising the transfer out of an owner that
   * still holds no written bytes.  Keep it off the home rank, and off this
   * rank too where the run is wide enough, so the write turn has to be
   * migrated rather than found already in place. */
  unsigned int n = arts_get_total_ranks();
  arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
  hint.rank = (home + ((n > 2u) ? 2u : 1u)) % n;
  uint64_t pv[2] = {(uint64_t)db, (uint64_t)home};
  arts_guid_t w = arts_edt_create(consumer_rw, 2, pv, 1, &hint);
  arts_add_dependence(db, w, 0, DB_MODE_RW);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int n = arts_get_total_ranks();
  if (n < 2) {
    arts_printf("FAIL: this test needs at least two ranks\n");
    arts_test_fail();
    arts_shutdown();
    return;
  }

  /* Home this rank, so a consumer elsewhere must acquire across ranks. */
  unsigned int home = arts_get_current_rank();
  arts_db_hint_t dh = ARTS_DB_HINT_DEFAULTS;
  dh.rank = home;
  void *addr = NULL;
  arts_guid_t db = arts_db_create(&addr, DB_BYTES, ARTS_DB_DEFAULT,
                                  ARTS_DB_PROP_NO_ACQUIRE, &dh);
  if (db == NULL_GUID) {
    arts_printf("FAIL: could not create the datablock\n");
    arts_test_fail();
    arts_shutdown();
    return;
  }
  /* The creator gets no pointer by contract — that part is expected. */

  arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
  hint.rank = (home + 1u) % n;
  uint64_t pv[2] = {(uint64_t)db, (uint64_t)home};
  arts_guid_t r = arts_edt_create(consumer_ro, 2, pv, 1, &hint);
  arts_add_dependence(db, r, 0, DB_MODE_RO);
}

int main(int argc, char **argv) {
  int rc = arts_rt(argc, argv);
  return rc ? 1 : arts_test_status();
}
