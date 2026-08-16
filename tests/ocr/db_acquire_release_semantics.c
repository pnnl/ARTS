/* What an acquire and a release promise, case by case.
 *
 * The model gives an EDT a datablock through a dependence in some access mode,
 * and gives the program one explicit way to end that hold early
 * (arts_db_release); otherwise the hold ends when the EDT does.  The promises
 * that follow from that, and which this exercises:
 *
 *   - an acquire in any mode yields storage of the declared size, and RW
 *     storage is writable through its whole extent;
 *   - a release makes what the holder wrote visible to whoever acquires next,
 *     wherever that acquirer runs;
 *   - releasing early is not the same as never having held it: an EDT that
 *     releases and then finishes must not double-release, and the block must
 *     survive for its next acquirer;
 *   - several readers may hold one block at once and all see the same bytes;
 *   - a writer that follows readers sees what the last writer left, not what
 *     a reader's stale copy held;
 *   - a value dependence delivers the value itself rather than storage.
 *
 * Each case runs across ranks where crossing ranks is what makes it
 * meaningful, and the whole file runs unchanged under every coherence
 * configuration — the arms may move bytes differently but owe the same
 * answers.  GUIDs travel in paramv: an EDT body runs on whichever rank its
 * placement hint named, so a file-scope variable set by one EDT means nothing
 * to another.
 */

#include "arts.h"
#include "../test_failure_status.h"

#include <stdint.h>

#define ELEMS 64u
#define BYTES (ELEMS * sizeof(uint64_t))

#define MAGIC_WRITER 0x1111000000000000ull
#define MAGIC_REWRITE 0x2222000000000000ull
#define VALUE_PAYLOAD 0xFEEDFACEull

/* Case identifiers, also the order they run in. */
enum {
  C_RW_RELEASE_THEN_READ = 0, /* release publishes to a later reader */
  C_MULTI_READER,             /* concurrent readers agree */
  C_WRITER_AFTER_READERS,     /* a writer follows readers and sees the data */
  C_VALUE_MODE,               /* a value dependence carries a value */
  C_COUNT
};

static void start_case(unsigned int idx, unsigned int home);

static void next_case(unsigned int idx, unsigned int home) {
  if (idx + 1u < (unsigned int)C_COUNT) {
    start_case(idx + 1u, home);
  } else {
    arts_shutdown();
  }
}

static unsigned int away(unsigned int rank, unsigned int step) {
  unsigned int n = arts_get_total_ranks();
  return (rank + step) % n;
}

/* ---- case 1: write, release early, then read from another rank ---------- */

/* paramv = { db, home, case } */
void c1_reader(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int home = (unsigned int)paramv[1];
  const uint64_t *p = (const uint64_t *)depv[0].ptr;
  if (p == NULL) {
    arts_printf("FAIL: rw_release_then_read — reader got no storage\n");
    arts_test_fail();
    arts_shutdown();
    return;
  }
  bool ok = true;
  for (unsigned i = 0; i < ELEMS; i++) {
    ok = ok && (p[i] == (MAGIC_WRITER | i));
  }
  arts_printf("%s: rw_release_then_read — reader sees the released writes\n",
              ok ? "PASS" : "FAIL");
  if (!ok) {
    arts_test_fail();
  }
  next_case(C_RW_RELEASE_THEN_READ, home);
}

/* paramv = { db, home, case } */
void c1_writer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t db = (arts_guid_t)paramv[0];
  unsigned int home = (unsigned int)paramv[1];
  uint64_t *p = (uint64_t *)depv[0].ptr;
  if (p == NULL) {
    arts_printf("FAIL: rw_release_then_read — writer got no storage\n");
    arts_test_fail();
    arts_shutdown();
    return;
  }
  for (unsigned i = 0; i < ELEMS; i++) {
    p[i] = MAGIC_WRITER | i;
  }
  /* End the hold before the EDT ends.  The block must survive this and the
   * EDT's own end must not release it a second time. */
  arts_db_release(db, DB_MODE_RW);

  arts_edt_hint_t h = ARTS_EDT_HINT_DEFAULTS;
  h.rank = away(home, 1);
  uint64_t pv[3] = {(uint64_t)db, home, C_RW_RELEASE_THEN_READ};
  arts_guid_t r = arts_edt_create(c1_reader, 3, pv, 1, &h);
  arts_add_dependence(db, r, 0, DB_MODE_RO);
}

/* ---- case 2: concurrent readers ---------------------------------------- */

/* paramv = { db, home, case, reader index, collector } */
void c2_reader(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int idx = (unsigned int)paramv[3];
  arts_guid_t coll = (arts_guid_t)paramv[4];
  const uint64_t *p = (const uint64_t *)depv[0].ptr;
  bool ok = (p != NULL);
  if (ok) {
    for (unsigned i = 0; i < ELEMS; i++) {
      ok = ok && (p[i] == (MAGIC_WRITER | i));
    }
  }
  arts_printf("%s: multi_reader — reader %u agrees\n", ok ? "PASS" : "FAIL",
              idx);
  if (!ok) {
    arts_test_fail();
  }
  /* One decrement per reader; the latch was created with the reader count and
   * fires when the last of them lands.  The slot selects the latch OPERATION,
   * not a per-waiter index. */
  arts_event_satisfy_slot(coll, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* paramv = { db, home, case } */
void c2_join(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
             arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  next_case(C_MULTI_READER, (unsigned int)paramv[1]);
}

/* ---- case 3: a writer after readers ------------------------------------ */

/* paramv = { db, home, case } */
void c3_check(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int home = (unsigned int)paramv[1];
  const uint64_t *p = (const uint64_t *)depv[0].ptr;
  bool ok = (p != NULL);
  if (ok) {
    for (unsigned i = 0; i < ELEMS; i++) {
      ok = ok && (p[i] == (MAGIC_REWRITE | i));
    }
  }
  arts_printf("%s: writer_after_readers — the rewrite is what survives\n",
              ok ? "PASS" : "FAIL");
  if (!ok) {
    arts_test_fail();
  }
  next_case(C_WRITER_AFTER_READERS, home);
}

/* paramv = { db, home, case } */
void c3_writer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t db = (arts_guid_t)paramv[0];
  unsigned int home = (unsigned int)paramv[1];
  uint64_t *p = (uint64_t *)depv[0].ptr;
  if (p == NULL) {
    arts_printf("FAIL: writer_after_readers — writer got no storage\n");
    arts_test_fail();
    arts_shutdown();
    return;
  }
  /* The writer must be looking at the previous writer's bytes, not at some
   * reader's stale copy. */
  bool saw_prev = true;
  for (unsigned i = 0; i < ELEMS; i++) {
    saw_prev = saw_prev && (p[i] == (MAGIC_WRITER | i));
  }
  if (!saw_prev) {
    arts_printf("FAIL: writer_after_readers — writer did not see the "
                "previously released bytes\n");
    arts_test_fail();
  }
  for (unsigned i = 0; i < ELEMS; i++) {
    p[i] = MAGIC_REWRITE | i;
  }

  arts_edt_hint_t h = ARTS_EDT_HINT_DEFAULTS;
  h.rank = away(home, 1);
  uint64_t pv[3] = {(uint64_t)db, home, C_WRITER_AFTER_READERS};
  arts_guid_t c = arts_edt_create(c3_check, 3, pv, 1, &h);
  arts_add_dependence(db, c, 0, DB_MODE_RO);
}

/* ---- case 4: value dependence ------------------------------------------ */

/* paramv = { db, home, case } */
void c5_value_dep(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int home = (unsigned int)paramv[1];
  bool ok = (depv[0].guid == (arts_guid_t)VALUE_PAYLOAD);
  arts_printf("%s: value_mode — the value arrives as itself\n",
              ok ? "PASS" : "FAIL");
  if (!ok) {
    arts_test_fail();
  }
  next_case(C_VALUE_MODE, home);
}

/* ---- case driver ------------------------------------------------------- */

static void start_case(unsigned int idx, unsigned int home) {
  arts_db_hint_t dh = ARTS_DB_HINT_DEFAULTS;
  dh.rank = home;
  arts_edt_hint_t eh = ARTS_EDT_HINT_DEFAULTS;

  if (idx == C_RW_RELEASE_THEN_READ) {
    void *addr = NULL;
    arts_guid_t db = arts_db_create(&addr, BYTES, ARTS_DB_DEFAULT,
                                    ARTS_DB_PROP_NO_ACQUIRE, &dh);
    eh.rank = away(home, 1);
    uint64_t pv[3] = {(uint64_t)db, home, idx};
    arts_guid_t w = arts_edt_create(c1_writer, 3, pv, 1, &eh);
    arts_add_dependence(db, w, 0, DB_MODE_RW);
    return;
  }

  if (idx == C_MULTI_READER || idx == C_WRITER_AFTER_READERS) {
    /* Both cases start from a block that already holds MAGIC_WRITER, written
     * by a producer whose hold has ended. */
    void *addr = NULL;
    arts_guid_t db = arts_db_create(&addr, BYTES, ARTS_DB_DEFAULT,
                                    ARTS_DB_PROP_NONE, &dh);
    if (addr == NULL) {
      arts_printf("FAIL: producer acquire yielded no storage\n");
      arts_test_fail();
      arts_shutdown();
      return;
    }
    uint64_t *p = (uint64_t *)addr;
    for (unsigned i = 0; i < ELEMS; i++) {
      p[i] = MAGIC_WRITER | i;
    }
    arts_db_release(db, DB_MODE_RW);

    if (idx == C_WRITER_AFTER_READERS) {
      eh.rank = away(home, 1);
      uint64_t pv[3] = {(uint64_t)db, home, idx};
      arts_guid_t w = arts_edt_create(c3_writer, 3, pv, 1, &eh);
      arts_add_dependence(db, w, 0, DB_MODE_RW);
      return;
    }

    /* Fan three readers out across ranks, joined by a channel-free latch. */
    const unsigned int NR = 3u;
    arts_event_hint_t evh = ARTS_EVENT_HINT_LATCH(NR);
    arts_guid_t coll = arts_event_create(&evh);
    uint64_t jv[3] = {(uint64_t)db, home, idx};
    eh.rank = home;
    arts_guid_t j = arts_edt_create(c2_join, 3, jv, 1, &eh);
    arts_add_dependence(coll, j, 0, DB_MODE_NULL);
    for (unsigned int i = 0; i < NR; i++) {
      arts_edt_hint_t rh = ARTS_EDT_HINT_DEFAULTS;
      rh.rank = away(home, i + 1u);
      uint64_t pv[5] = {(uint64_t)db, home, idx, i, (uint64_t)coll};
      arts_guid_t r = arts_edt_create(c2_reader, 5, pv, 1, &rh);
      arts_add_dependence(db, r, 0, DB_MODE_RO);
    }
    return;
  }

  /* C_VALUE_MODE: no block at all — the dependence carries the value. */
  eh.rank = away(home, 1);
  uint64_t pv[3] = {0, home, idx};
  arts_guid_t e = arts_edt_create(c5_value_dep, 3, pv, 1, &eh);
  arts_add_dependence((arts_guid_t)VALUE_PAYLOAD, e, 0, DB_MODE_VAL);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  if (arts_get_total_ranks() < 2) {
    arts_printf("FAIL: this test needs at least two ranks\n");
    arts_test_fail();
    arts_shutdown();
    return;
  }
  start_case(0, arts_get_current_rank());
}

int main(int argc, char **argv) {
  int rc = arts_rt(argc, argv);
  return rc ? 1 : arts_test_status();
}
