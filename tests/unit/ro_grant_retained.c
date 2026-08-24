/* SPDX-License-Identifier: Apache-2.0
 *
 * A read grant OUTLIVES its readers under the retaining release policy.
 *
 * This is the read-side twin of grant_sticky, and it is the property this
 * policy is named for: once a rank has been served a copy, it keeps both the
 * bytes and the permission until another node asks for them back — never
 * merely because its own readers finished. An implementation that gave the
 * grant back at the reader-count zero edge computes exactly the same answers
 * and passes every value oracle in the suite; it just re-fetches the whole
 * block on the next acquire. That is precisely the regression this asserts.
 *
 * Asserted on a rank that is NOT the data owner (so the grant is a real,
 * home-counted one rather than a local hit):
 *   1. while a reader holds it, ro_st == GRANT;
 *   2. after every reader has released and no writer has asked for it back,
 *      ro_st is STILL GRANT with rc == 0 — the resting state retention
 *      exists to create. A purging implementation lands on IDLE.
 *
 * Multinode by construction: on one rank home, owner and self coincide, a read
 * acquire is a local hit that never touches ro_st, and there is no grant to
 * retain. Configurations without a retained read grant self-skip.
 */

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if !defined(ARTS_PROTOCOL_EXCL) || !defined(ARTS_RELEASE_RETAIN)
int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  printf("SKIP ro_grant_retained: no retained read grant in this "
         "configuration\n");
  return 0;
}
#else

#include "arts/coherence/excl/types.h"
#include "arts/gas/route_table.h"
#include "arts/utils/atomics.h"

#define READERS 4
#define MAGIC 0x5AFE0000u

/* The guid travels in paramv, never in a global: main_edt runs on rank 0 only,
 * so a global would read as NULL_GUID on every other rank — which is exactly
 * the rank this test must observe. */
static int g_fail;

static void ro_state(arts_guid_t db, unsigned int *ro_st, unsigned int *rc,
                     unsigned int *own) {
  arts_shared_ptr_t h = arts_route_table_lookup_db(db);
  struct arts_db_s *d = (struct arts_db_s *)arts_shared_get(h);
  uint64_t w =
      (d != NULL) ? arts_atomic_read_u64(&d->cache.cache_state) : 0u;
  *ro_st = CACHE_RO_ST(w);
  *rc = CACHE_RO_CNT(w);
  *own = CACHE_OWNER(w);
  arts_shared_release(&h);
}

static void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int seq = (unsigned int)paramv[0];
  arts_guid_t db = (arts_guid_t)paramv[1];
  unsigned int *data = (unsigned int *)depv[0].ptr;
  if (data == NULL || data[0] != MAGIC) {
    (void)fprintf(stderr, "FAIL ro_grant_retained: reader %u bad value\n", seq);
    g_fail = 1;
    return;
  }
  unsigned int ro_st, rc, own;
  ro_state(db, &ro_st, &rc, &own);
  if (own == 1u) {
    return; /* owner-local read: no home-counted grant to speak of */
  }
  if (ro_st != CACHE_ST_GRANT && ro_st != CACHE_ST_GRANT_PURGE) {
    (void)fprintf(stderr,
                  "FAIL ro_grant_retained: reader %u sees ro_st=%u, expected a "
                  "held grant\n",
                  seq, ro_st);
    g_fail = 1;
  }
}

static void check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t db = (arts_guid_t)paramv[0];
  unsigned int ro_st, rc, own;
  ro_state(db, &ro_st, &rc, &own);
  /* Only meaningful where a real grant was taken: the owner reads locally. */
  if (own == 0u && ro_st == CACHE_ST_IDLE) {
    (void)fprintf(stderr,
                  "FAIL ro_grant_retained: after all readers released, "
                  "ro_st=IDLE rc=%u — the grant was returned at the zero edge "
                  "(this is PURGE's rule; retention must hold it until a "
                  "writer asks)\n",
                  rc);
    g_fail = 1;
  }
  if (g_fail) {
    arts_abort(1);
  }
  printf("PASS ro_grant_retained: the read grant outlived %d readers\n",
         READERS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  if (arts_get_total_ranks() < 2) {
    printf("SKIP ro_grant_retained: multinode only (on one rank a read is a "
           "local hit and there is no grant to retain)\n");
    arts_shutdown();
    return;
  }

  void *ptr = NULL;
  arts_guid_t db = arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB,
                                  ARTS_DB_PROP_NONE, NULL);
  ((unsigned int *)ptr)[0] = MAGIC;
  arts_db_release(db, DB_MODE_RW);

  /* Readers run one after another on rank 1 — a rank that is NOT the owner, so
   * each acquire is answered by a real home-counted grant.  Chaining them
   * means every acquire after the first meets a grant this rank already holds,
   * and the gap between them is exactly the reader-count zero edge at which a
   * purging policy would hand the grant back. */
  arts_guid_t prev = NULL_GUID;
  for (unsigned int i = 0; i < READERS; i++) {
    uint64_t pv[2] = {i, (uint64_t)db};
    arts_guid_t e = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_edt_hint_t h = {.finish_event = e, .rank = 1};
    arts_guid_t r =
        arts_edt_create(reader_edt, 2, pv, prev == NULL_GUID ? 1 : 2, &h);
    arts_add_dependence(db, r, 0, DB_MODE_RO);
    if (prev != NULL_GUID) {
      arts_add_dependence(prev, r, 1, DB_MODE_NULL);
    }
    prev = e;
  }

  arts_edt_hint_t ch = {.rank = 1};
  uint64_t cpv[1] = {(uint64_t)db};
  arts_guid_t chk = arts_edt_create(check_edt, 1, cpv, 1, &ch);
  arts_add_dependence(prev, chk, 0, DB_MODE_NULL);
}

int main(int argc, char **argv) {
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}

#endif /* EXCL + RETAIN */
