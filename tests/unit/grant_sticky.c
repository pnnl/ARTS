/* SPDX-License-Identifier: Apache-2.0
 *
 * The migrating grant is STICKY: a grant outlives the writer that opened it.
 *
 * This is the property that distinguishes a grant from a per-acquire turn, and
 * it is the whole reason consecutive local writers cost no wire traffic. No
 * value oracle can see it: an implementation that returned ownership at every
 * last release computes exactly the same answers, just with a home round trip
 * per writer. So the property needs an assertion of its own, or it regresses
 * silently.
 *
 * What is asserted, on the rank that holds the grant:
 *   1. while a writer holds it, writer_count >= 2 — its own hold plus the
 *      sentinel that IS the grant;
 *   2. after every writer has released and no foreign rank has asked for it,
 *      writer_count == 1 — the sentinel alone. An implementation that gave
 *      ownership back at the zero edge lands on 0 here, and the next local
 *      writer would have to re-acquire remotely.
 *
 * Single-rank by construction: with no other rank there is nothing that could
 * revoke the grant, so the sentinel's survival is attributable to stickiness
 * and to nothing else.
 *
 * EXCL states the same property in its own vocabulary. There is no sentinel
 * there; the write PERMISSION lives in cache_state's rw_st, and RETAIN means it
 * is relinquished only when another node asks — never merely because the local
 * writers finished. So the EXCL arm asserts rw_st == GRANT with wc == 0 after
 * every writer has released. An implementation that relinquished at the zero
 * edge (or that demoted unconditionally when driving the engine) lands on IDLE
 * here and every later local write pays a home round trip — the policy
 * silently degrades to PURGE while computing identical answers, which is
 * exactly the class of regression a value oracle cannot see. PURGE lands on
 * IDLE legitimately and is excluded; WRF_VAL has no grant at all and skips.
 */

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#if defined(ARTS_PROTOCOL_WRF_VAL) ||                                          \
    (defined(ARTS_PROTOCOL_EXCL) && !defined(ARTS_RELEASE_RETAIN))
int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  printf("SKIP grant_sticky: no retained grant in this configuration\n");
  return 0;
}
#else

#include "arts/coherence/types.h"
#include "arts/gas/route_table.h"
#include "arts/utils/atomics.h"

#define WRITERS 4

static arts_guid_t g_db;
static int g_fail;

#ifdef ARTS_PROTOCOL_EXCL
#include "arts/coherence/excl/types.h"

/* EXCL: the permission is rw_st, and wc counts only the live local writers. */
static void excl_perm(arts_guid_t db, unsigned int *rw_st, unsigned int *wc) {
  arts_shared_ptr_t h = arts_route_table_lookup_db(db);
  struct arts_db_s *d = (struct arts_db_s *)arts_shared_get(h);
  uint64_t w =
      (d != NULL) ? arts_atomic_read_u64(&d->cache.cache_state) : 0u;
  *rw_st = CACHE_RW_ST(w);
  *wc = CACHE_RW_CNT(w);
  arts_shared_release(&h);
}
#else
/* Read the grant counter on this rank without disturbing it. */
static unsigned int lease_count(arts_guid_t db) {
  arts_shared_ptr_t h = arts_route_table_lookup_db(db);
  struct arts_db_s *d = (struct arts_db_s *)arts_shared_get(h);
  unsigned int wc = (d != NULL) ? arts_atomic_read(&d->cache.writer_count) : 0u;
  arts_shared_release(&h);
  return wc;
}
#endif

static void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  unsigned int seq = (unsigned int)paramv[0];
  int *data = (int *)depv[0].ptr;
  if (data == NULL) {
    (void)fprintf(stderr, "FAIL grant_sticky: writer %u got NULL ptr\n", seq);
    g_fail = 1;
    return;
  }
  data[0] = data[0] + 1;
#ifdef ARTS_PROTOCOL_EXCL
  /* A writer runs only under a held permission; wc counts its own hold. */
  unsigned int rw_st, wc;
  excl_perm(g_db, &rw_st, &wc);
  if (rw_st != CACHE_ST_GRANT || wc < 1u) {
    (void)fprintf(stderr,
                  "FAIL grant_sticky: writer %u sees rw_st=%u wc=%u, "
                  "expected GRANT with its own hold counted\n",
                  seq, rw_st, wc);
    g_fail = 1;
  }
#else
  /* Own hold + sentinel.  A writer that reached here without the sentinel
   * would mean the grant was granted per acquire rather than held. */
  unsigned int wc = lease_count(g_db);
  if (wc < 2u) {
    (void)fprintf(stderr,
                  "FAIL grant_sticky: writer %u sees writer_count=%u, "
                  "expected >= 2 (own hold + sentinel)\n",
                  seq, wc);
    g_fail = 1;
  }
#endif
}

static void check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
#ifdef ARTS_PROTOCOL_EXCL
  /* Every writer has released and nobody else exists to ask for it back, so
   * the permission must still be held with no live writer under it. */
  unsigned int rw_st, wc;
  excl_perm(g_db, &rw_st, &wc);
  if (rw_st != CACHE_ST_GRANT || wc != 0u) {
    (void)fprintf(stderr,
                  "FAIL grant_sticky: after all releases rw_st=%u wc=%u, "
                  "expected GRANT with wc==0 (the permission survives its "
                  "writers; IDLE here means it degraded to PURGE)\n",
                  rw_st, wc);
    g_fail = 1;
  }
#else
  /* Every writer has released and nobody else exists to revoke: the sentinel,
   * and only the sentinel, must remain. */
  unsigned int wc = lease_count(g_db);
  if (wc != 1u) {
    (void)fprintf(stderr,
                  "FAIL grant_sticky: after all releases writer_count=%u, "
                  "expected 1 (the grant survives its writers)\n",
                  wc);
    g_fail = 1;
  }
#endif
  if (g_fail) {
    arts_abort(1);
  }
  printf("PASS grant_sticky: the grant outlived %d writers\n", WRITERS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  if (arts_get_total_ranks() != 1) {
    printf("SKIP grant_sticky: single-rank only (a second rank could revoke "
           "the grant, which is exactly what this test must exclude)\n");
    arts_shutdown();
    return;
  }

  void *ptr = NULL;
  g_db = arts_db_create(&ptr, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  ((int *)ptr)[0] = 0;
  arts_db_release(g_db, DB_MODE_RW);

  /* Writers run one after another (each gated on the previous), so every
   * acquire after the first meets a grant this rank already holds. */
  arts_guid_t prev = NULL_GUID;
  for (unsigned int i = 0; i < WRITERS; i++) {
    uint64_t seq = i;
    arts_guid_t e = arts_event_create(&ARTS_EVENT_HINT_FINISH);
    arts_guid_t w = arts_edt_create(writer_edt, 1, &seq, prev == NULL_GUID ? 1 : 2,
                                    &(arts_edt_hint_t){.finish_event = e});
    arts_add_dependence(g_db, w, 0, DB_MODE_RW);
    if (prev != NULL_GUID) {
      arts_add_dependence(prev, w, 1, DB_MODE_NULL);
    }
    prev = e;
  }

  arts_guid_t chk = arts_edt_create(check_edt, 0, NULL, 1, NULL);
  arts_add_dependence(prev, chk, 0, DB_MODE_NULL);
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}

#endif /* grant-bearing protocols */
