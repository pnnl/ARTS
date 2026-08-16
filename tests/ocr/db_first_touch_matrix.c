/* Every way a datablock can first be reached must yield usable storage.
 *
 * `arts_db_create` declares a size, so an EDT whose dependence on that block
 * has been satisfied is entitled to a pointer it can address for that size —
 * in RO as much as in RW.  Whether anyone has WRITTEN the block, and whether
 * the creator took it at create time (ARTS_DB_PROP_NO_ACQUIRE says it did
 * not, and that the first consumer acquires normally), are statements about
 * its contents and its ownership history, not about whether storage exists.
 * The one legitimate exception is a genuinely zero-sized block.
 *
 * The cases below walk that surface along three axes:
 *
 *   - creator-acquired versus not, and read versus write;
 *   - the block homed on the creating rank versus away from it (a remote home
 *     is the shape in which the home rank holds directory state only, so an
 *     arm whose home keeps no payload starts out with nothing to hand over);
 *   - the consumer on the creating rank, one rank away, or two (which, with a
 *     remote home, puts it on neither the home nor the creator).
 *
 * They run one after another off a chain so a single run covers the matrix,
 * and each case re-homes relative to whichever rank the previous consumer ran
 * on, so the walk rotates through the ranks.  The same source runs under every
 * coherence configuration — which is the point, since the arms answer a first
 * touch differently.
 */

#include "arts.h"
#include "../test_failure_status.h"

#include <stdint.h>

#define DB_ELEMS 128u
#define DB_BYTES (DB_ELEMS * sizeof(uint64_t))

struct case_s {
  const char *name;
  unsigned int db_flags;      /* ARTS_DB_PROP_* */
  arts_db_access_mode_t mode; /* how the consumer takes it */
  unsigned int home_off;      /* home rank = creator + home_off */
  unsigned int cons_off;      /* consumer rank = creator + cons_off */
};

/* The table is const and identical on every rank, so a chained case may read
 * it wherever it runs. */
static const struct case_s CASES[] = {
    {"no_acquire/RO/home=here/cons=here", ARTS_DB_PROP_NO_ACQUIRE, DB_MODE_RO,
     0u, 0u},
    {"no_acquire/RW/home=here/cons=here", ARTS_DB_PROP_NO_ACQUIRE, DB_MODE_RW,
     0u, 0u},
    {"no_acquire/RO/home=here/cons=+1", ARTS_DB_PROP_NO_ACQUIRE, DB_MODE_RO,
     0u, 1u},
    {"no_acquire/RW/home=here/cons=+1", ARTS_DB_PROP_NO_ACQUIRE, DB_MODE_RW,
     0u, 1u},
    {"no_acquire/RO/home=here/cons=+2", ARTS_DB_PROP_NO_ACQUIRE, DB_MODE_RO,
     0u, 2u},
    {"no_acquire/RW/home=here/cons=+2", ARTS_DB_PROP_NO_ACQUIRE, DB_MODE_RW,
     0u, 2u},
    {"no_acquire/RO/home=+1/cons=here", ARTS_DB_PROP_NO_ACQUIRE, DB_MODE_RO,
     1u, 0u},
    {"no_acquire/RW/home=+1/cons=here", ARTS_DB_PROP_NO_ACQUIRE, DB_MODE_RW,
     1u, 0u},
    {"no_acquire/RO/home=+1/cons=+1", ARTS_DB_PROP_NO_ACQUIRE, DB_MODE_RO, 1u,
     1u},
    {"no_acquire/RW/home=+1/cons=+1", ARTS_DB_PROP_NO_ACQUIRE, DB_MODE_RW, 1u,
     1u},
    {"no_acquire/RO/home=+1/cons=+2", ARTS_DB_PROP_NO_ACQUIRE, DB_MODE_RO, 1u,
     2u},
    {"no_acquire/RW/home=+1/cons=+2", ARTS_DB_PROP_NO_ACQUIRE, DB_MODE_RW, 1u,
     2u},
    {"acquired/RO/home=here/cons=here", ARTS_DB_PROP_NONE, DB_MODE_RO, 0u, 0u},
    {"acquired/RW/home=here/cons=here", ARTS_DB_PROP_NONE, DB_MODE_RW, 0u, 0u},
    {"acquired/RO/home=here/cons=+1", ARTS_DB_PROP_NONE, DB_MODE_RO, 0u, 1u},
    {"acquired/RW/home=here/cons=+1", ARTS_DB_PROP_NONE, DB_MODE_RW, 0u, 1u},
    {"acquired/RO/home=here/cons=+2", ARTS_DB_PROP_NONE, DB_MODE_RO, 0u, 2u},
    {"acquired/RW/home=here/cons=+2", ARTS_DB_PROP_NONE, DB_MODE_RW, 0u, 2u},
    {"acquired/RO/home=+1/cons=here", ARTS_DB_PROP_NONE, DB_MODE_RO, 1u, 0u},
    {"acquired/RW/home=+1/cons=here", ARTS_DB_PROP_NONE, DB_MODE_RW, 1u, 0u},
    {"acquired/RO/home=+1/cons=+1", ARTS_DB_PROP_NONE, DB_MODE_RO, 1u, 1u},
    {"acquired/RW/home=+1/cons=+1", ARTS_DB_PROP_NONE, DB_MODE_RW, 1u, 1u},
    {"acquired/RO/home=+1/cons=+2", ARTS_DB_PROP_NONE, DB_MODE_RO, 1u, 2u},
    {"acquired/RW/home=+1/cons=+2", ARTS_DB_PROP_NONE, DB_MODE_RW, 1u, 2u},
};
#define NCASES (sizeof(CASES) / sizeof(CASES[0]))

static void run_case(unsigned int idx);

/* paramv[0] = case index. */
void case_consumer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int idx = (unsigned int)paramv[0];
  const struct case_s *c = &CASES[idx];

  if (depv[0].ptr == NULL) {
    arts_printf("FAIL: %s — acquire yielded no storage\n", c->name);
    arts_test_fail();
  } else if (c->mode == DB_MODE_RW) {
    uint64_t *p = (uint64_t *)depv[0].ptr;
    for (unsigned i = 0; i < DB_ELEMS; i++) {
      p[i] = (idx * 1000u) + i;
    }
    bool ok = true;
    for (unsigned i = 0; i < DB_ELEMS; i++) {
      ok = ok && (p[i] == (idx * 1000u) + i);
    }
    arts_printf("%s: %s — writable storage\n", ok ? "PASS" : "FAIL", c->name);
    if (!ok) {
      arts_test_fail();
    }
  } else {
    /* Contents are undefined for a block nobody has published; addressability
     * is not. */
    const volatile uint64_t *p = (const volatile uint64_t *)depv[0].ptr;
    uint64_t sink = 0;
    for (unsigned i = 0; i < DB_ELEMS; i++) {
      sink += p[i];
    }
    (void)sink;
    arts_printf("PASS: %s — readable storage\n", c->name);
  }

  if (idx + 1u < NCASES) {
    run_case(idx + 1u);
  } else {
    arts_shutdown();
  }
}

static void run_case(unsigned int idx) {
  const struct case_s *c = &CASES[idx];
  unsigned int self = arts_get_current_rank();
  unsigned int nranks = arts_get_total_ranks();

  arts_db_hint_t dh = ARTS_DB_HINT_DEFAULTS;
  dh.rank = (self + c->home_off) % nranks;
  void *addr = NULL;
  arts_guid_t db =
      arts_db_create(&addr, DB_BYTES, ARTS_DB_DEFAULT, c->db_flags, &dh);
  if (db == NULL_GUID) {
    arts_printf("FAIL: %s — datablock create failed\n", c->name);
    arts_test_fail();
    arts_shutdown();
    return;
  }
  if (c->db_flags == ARTS_DB_PROP_NONE) {
    /* The creator holds it; hand it over so the consumer's acquire is a
     * first touch by another holder rather than a wait on us. */
    if (addr == NULL) {
      arts_printf("FAIL: %s — creator acquire yielded no storage\n", c->name);
      arts_test_fail();
      arts_shutdown();
      return;
    }
    arts_db_release(db, DB_MODE_RW);
  }

  arts_edt_hint_t eh = ARTS_EDT_HINT_DEFAULTS;
  eh.rank = (self + c->cons_off) % nranks;
  uint64_t pv = idx;
  arts_guid_t e = arts_edt_create(case_consumer, 1, &pv, 1, &eh);
  arts_add_dependence(db, e, 0, c->mode);
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
  run_case(0);
}

int main(int argc, char **argv) {
  int rc = arts_rt(argc, argv);
  return rc ? 1 : arts_test_status();
}
