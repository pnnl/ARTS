/* SPDX-License-Identifier: Apache-2.0
 *
 * T048 — DB cache / db_s static layout invariants (B029: stub_size /
 * home_initialized layout, the historical "P6 teardown SIGSEGV" class).
 *
 * A non-home / lazy / creator-remote DB is allocated as a *cache-only stub* of
 * arts_db_cache_stub_size() bytes instead of the full sizeof(struct arts_db_s).
 * The cache destructor reads db->home_initialized on EVERY free to decide
 * whether to tear the home directory down; on a stub it must read a (zeroed)
 * false and skip teardown.  Therefore the following MUST hold, by construction,
 * for every protocol:
 *
 *   1. stub_size > offsetof(arts_db_s, db_type)         — db_type in bounds
 *      (every coherence lookup/free reads db_type through a stub).
 *   2. stub_size > offsetof(arts_db_s, home_initialized) — home_initialized in
 *      bounds (the destructor reads it on a stub; this is the exact field whose
 *      OOB read caused the P6 SIGSEGV).
 *   3. stub_size == offsetof(arts_db_s, <first home-arm field>) — the stub ends
 *      exactly at the first home-directory field (rw_holder for the ownership
 *      protocols, last_sent_version for MRMW, lock_state for LOCK), so it
 *      INCLUDES home_initialized but omits the bulky home directory.
 *   4. home_initialized is laid out AFTER db_type which is AFTER the cache —
 *      i.e. offsetof(cache)==0 < offsetof(db_type) <
 * offsetof(home_initialized).
 *   5. Buffer FAM data[] lands at offset 64 (cache-line / CXL boundary).
 *   6. snapshot_waiter.link is FIRST (offset 0) — required by arts_lf_stack_t.
 *   7. arts_db_total_size(db) == sizeof(arts_db_s) + db->cache.db_size.
 *
 * This is a pure static-layout test: it starts no runtime and links nothing.
 * Most checks are _Static_assert (compile-time); a couple that need a live
 * db_size (total_size) run at main().  Built per protocol; the first home-arm
 * field is protocol-dependent and selected by the same #if ladder the runtime
 * header (coherence/types.h) uses.
 */

#include "arts/coherence/types.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* The protocol-dependent first home-arm field, mirroring
 * arts_db_cache_stub_size() in coherence/types.h. */
#if defined(ARTS_PROTOCOL_MRMW)
#define FIRST_HOME_ARM_OFF offsetof(struct arts_db_s, last_sent_version)
#elif defined(ARTS_PROTOCOL_LOCK)
#define FIRST_HOME_ARM_OFF offsetof(struct arts_db_s, lock_state)
#else
#define FIRST_HOME_ARM_OFF offsetof(struct arts_db_s, rw_holder)
#endif

/* --- Compile-time invariants (the load-bearing ones) ------------------- */

/* (5) buffer data[] at offset 64. */
_Static_assert(offsetof(struct arts_db_buffer_s, data) == 64,
               "buffer FAM data[] must land at offset 64 (cache-line / CXL)");

/* (6) snapshot waiter link first (Treiber/arts_lf_stack_t contract). */
_Static_assert(offsetof(struct arts_db_snapshot_waiter_s, link) == 0,
               "snapshot_waiter.link must be the FIRST member");

/* (4) cache first, then db_type, then home_initialized. */
_Static_assert(offsetof(struct arts_db_s, cache) == 0,
               "cache must be the FIRST member of arts_db_s");
_Static_assert(offsetof(struct arts_db_s, db_type) >
                   offsetof(struct arts_db_s, cache),
               "db_type must follow the cache");
_Static_assert(offsetof(struct arts_db_s, home_initialized) >
                   offsetof(struct arts_db_s, db_type),
               "home_initialized must follow db_type");

int main(void) {
  int rc = 0;
  uint64_t stub = arts_db_cache_stub_size();
  size_t off_db_type = offsetof(struct arts_db_s, db_type);
  size_t off_home_init = offsetof(struct arts_db_s, home_initialized);
  size_t off_first_home = FIRST_HOME_ARM_OFF;

  /* (1) db_type in bounds of the stub. */
  if (!(stub > off_db_type)) {
    (void)fprintf(stderr,
                  "FAIL db_cache_layout: stub_size(%llu) must be > "
                  "offsetof(db_type)=%zu\n",
                  (unsigned long long)stub, off_db_type);
    rc = 1;
  }
  /* (2) home_initialized in bounds — the P6 SIGSEGV guard. */
  if (!(stub > off_home_init)) {
    (void)fprintf(stderr,
                  "FAIL db_cache_layout: stub_size(%llu) must be > "
                  "offsetof(home_initialized)=%zu (P6 teardown OOB)\n",
                  (unsigned long long)stub, off_home_init);
    rc = 1;
  }
  /* The stub must still leave room to read the whole bool home_initialized. */
  if (!(stub >=
        off_home_init + sizeof(((struct arts_db_s *)0)->home_initialized))) {
    (void)fprintf(stderr,
                  "FAIL db_cache_layout: stub_size(%llu) truncates "
                  "home_initialized (end=%zu)\n",
                  (unsigned long long)stub,
                  off_home_init +
                      sizeof(((struct arts_db_s *)0)->home_initialized));
    rc = 1;
  }
  /* (3) stub ends exactly at the first home-arm field. */
  if (stub != off_first_home) {
    (void)fprintf(stderr,
                  "FAIL db_cache_layout: stub_size(%llu) != "
                  "offsetof(first home-arm field)=%zu\n",
                  (unsigned long long)stub, off_first_home);
    rc = 1;
  }
  /* first home-arm field must be at or after home_initialized. */
  if (!(off_first_home >= off_home_init)) {
    (void)fprintf(stderr,
                  "FAIL db_cache_layout: first home-arm off(%zu) precedes "
                  "home_initialized(%zu)\n",
                  off_first_home, off_home_init);
    rc = 1;
  }

  /* (7) total_size == sizeof(arts_db_s) + db_size, for a couple of sizes. */
  struct arts_db_s db;
  memset(&db, 0, sizeof(db));
  uint64_t sizes[] = {0, 1, 64, 4096};
  for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
    db.cache.db_size = sizes[i];
    uint64_t want = (uint64_t)sizeof(struct arts_db_s) + sizes[i];
    uint64_t got = arts_db_total_size(&db);
    if (got != want) {
      (void)fprintf(stderr,
                    "FAIL db_cache_layout: total_size(db_size=%llu)=%llu, "
                    "expected %llu\n",
                    (unsigned long long)sizes[i], (unsigned long long)got,
                    (unsigned long long)want);
      rc = 1;
    }
  }

  /* arts_db_of_cache round-trip (cache is first ⇒ same address; NULL-safe). */
  if (arts_db_of_cache(&db.cache) != &db) {
    (void)fprintf(stderr, "FAIL db_cache_layout: of_cache(&db.cache) != &db\n");
    rc = 1;
  }
  if (arts_db_of_cache(NULL) != NULL) {
    (void)fprintf(stderr, "FAIL db_cache_layout: of_cache(NULL) != NULL\n");
    rc = 1;
  }

  if (rc != 0) {
    return 1;
  }
  printf("PASS db_cache_layout: stub=%llu db_type=%zu home_init=%zu "
         "first_home_arm=%zu data=64 link=0\n",
         (unsigned long long)stub, off_db_type, off_home_init, off_first_home);
  return 0;
}
