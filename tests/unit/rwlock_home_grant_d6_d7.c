/* SPDX-License-Identifier: Apache-2.0
 *
 * rwlock_home_grant_d6_d7 — RWLOCK-config-specific runtime exercise of the home
 * arbiter's D6 (rw→rw chain) and D7 (new RO grants while RW waiting) decision
 * paths (census 11-lock §1 decision table + rwlock_compute_next / lock_home_grant
 * in lock/home.c).  The pure truth-table is covered by the unit test
 * rwlock_compute_next; this drives the same transitions through the live runtime.
 *
 * D6 — rw→rw chain: a sequence of RW writers on one DB.  When a writer
 *   releases with another writer still counted (w-1>0), the home grants the
 *   next RW (LOCK_GRANT_ONE_RW pop).  Every writer's increment must land — a
 *   broken chain strands a writer → its finish event never fires → TIMEOUT.
 *
 * D7 — RO extends while RW waiting: an active RO phase (bit=RO) with an RW
 *   parked (w>0); a NEW RO request must still be granted immediately
 *   (LOCK_GRANT_ALL_RO, RO phase extends) rather than queued behind the writer.
 *   The waiting RW is eventually served when the RO phase drains.  All readers
 *   must observe the producer's value and the writer must run — a starved
 *   reader or writer stalls the finish scope → TIMEOUT.
 *
 * To put many participants in the same home phase simultaneously they are
 * launched depc=0 (ready immediately) and contend on the lock; the writers'
 * happens-before is enforced only where the data value is asserted (the seed
 * writer runs first via a finish-event gate so readers have a defined value).
 *
 * Self-skips on any non-RWLOCK build at compile time.
 */

#if !defined(ARTS_PROTOCOL_RWLOCK)
#include <stdio.h>
int main(void) {
  printf("SKIP rwlock_home_grant_d6_d7: RWLOCK-only\n");
  return 0;
}
#else

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arts.h"

#define N_CHAIN_WRITERS 32  /* D6: rw→rw chain length */
#define N_EXTEND_READERS 32 /* D7: RO extenders */
#define SEED 0xABCD1234u

/* DB layout (uint64_t): [0] = writer accumulator (D6) / seeded value (D7). */

/* D6 writer: RW-acquire, increment the accumulator, drop the D6 latch. */
static void chain_writer_edt(uint32_t paramc, const uint64_t *paramv,
                             uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t latch = (arts_guid_t)paramv[0];
  _Atomic uint64_t *acc = (_Atomic uint64_t *)depv[0].ptr;
  if (acc != NULL) {
    atomic_fetch_add_explicit(acc, 1u, memory_order_acq_rel);
  }
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* D6 verify: the chain accumulator must equal the writer count. */
static void chain_verify_edt(uint32_t paramc, const uint64_t *paramv,
                             uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  _Atomic uint64_t *acc = (_Atomic uint64_t *)depv[1].ptr;
  uint64_t got = atomic_load_explicit(acc, memory_order_acquire);
  if (got != (uint64_t)N_CHAIN_WRITERS) {
    (void)fprintf(stderr, "FAIL D6: chain acc=%llu (want %d)\n",
                  (unsigned long long)got, N_CHAIN_WRITERS);
    arts_abort(1);
  }
  printf("rwlock_home_grant_d6_d7 D6: rw->rw chain %d — OK\n", N_CHAIN_WRITERS);
}

/* D7 seed writer: stamp SEED so the extending readers have a defined value. */
static void seed_writer_edt(uint32_t paramc, const uint64_t *paramv,
                            uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *cell = (uint64_t *)depv[0].ptr;
  if (cell != NULL) {
    cell[0] = SEED;
  }
}

/* D7 extending reader: RO-acquire, must observe SEED, drop the D7 latch. */
static void extend_reader_edt(uint32_t paramc, const uint64_t *paramv,
                              uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t latch = (arts_guid_t)paramv[0];
  const uint64_t *cell = (const uint64_t *)depv[0].ptr;
  if (cell == NULL || cell[0] != SEED) {
    (void)fprintf(stderr, "FAIL D7: RO read 0x%llx (want 0x%x)\n",
                  cell ? (unsigned long long)cell[0] : 0ull, SEED);
    arts_abort(1);
  }
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* D7 trailing writer: RW concurrent with the RO burst; must still be served
 * (not starved behind the extending RO phase).  It does NOT mutate the value:
 * the RO extenders contend on the same lock and their observed value must
 * remain SEED regardless of whether the home grants this writer's phase before
 * or after any given reader.  Proving the writer runs (drops the latch) is the
 * anti-starvation check; mutating the value would make the readers' assertion
 * phase-order-dependent and thus racy. */
static void trail_writer_edt(uint32_t paramc, const uint64_t *paramv,
                             uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  arts_guid_t latch = (arts_guid_t)paramv[0];
  volatile uint64_t *cell = (volatile uint64_t *)depv[0].ptr;
  if (cell != NULL) {
    (void)cell[0]; /* touch under RW exclusivity; do not change the value */
  }
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* Final shutdown: bound to the D7 latch (which counts all extenders + the
 * trailing writer). */
static void shutdown_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  printf("rwlock_home_grant_d6_d7 D7: RO-extend %d + trail writer — PASS\n",
         N_EXTEND_READERS);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== rwlock_home_grant_d6_d7 ===\n");
  unsigned int nranks = arts_get_total_ranks();
  unsigned int home = 0u;

  /* ---- D6: rw->rw chain on its own DB. ---- */
  void *d6p = NULL;
  arts_guid_t d6 =
      arts_db_create(&d6p, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = home});
  *(uint64_t *)d6p = 0u;
  arts_db_release(d6, DB_MODE_RW);

  arts_event_hint_t d6_latch_hint = ARTS_EVENT_HINT_LATCH(N_CHAIN_WRITERS);
  d6_latch_hint.rank = 0;
  arts_guid_t d6_latch = arts_event_create(&d6_latch_hint);
  uint64_t d6_pv[1] = {(uint64_t)d6_latch};
  for (int i = 0; i < N_CHAIN_WRITERS; i++) {
    unsigned int r = (nranks > 1) ? (unsigned int)(i % nranks) : 0u;
    arts_guid_t w = arts_edt_create(chain_writer_edt, 1, d6_pv, 1,
                                    &(arts_edt_hint_t){.rank = r});
    arts_add_dependence(d6, w, 0, DB_MODE_RW);
  }
  arts_guid_t d6v = arts_edt_create(chain_verify_edt, 0, NULL, 2,
                                    &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(d6_latch, d6v, 0, DB_MODE_NULL);
  arts_add_dependence(d6, d6v, 1, DB_MODE_RO);

  /* ---- D7: RO-extend-while-RW-waiting on its own DB. ----
   * Seed writer runs first (finish-event gated) so readers have SEED. */
  void *d7p = NULL;
  arts_guid_t d7 =
      arts_db_create(&d7p, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = home});
  *(uint64_t *)d7p = 0u;
  arts_db_release(d7, DB_MODE_RW);

  arts_guid_t e_seed = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t seed =
      arts_edt_create(seed_writer_edt, 0, NULL, 1,
                      &(arts_edt_hint_t){.rank = home, .finish_event = e_seed});
  arts_add_dependence(d7, seed, 0, DB_MODE_RW);
  arts_event_wait(e_seed); /* ownership/value established before the storm */

  /* D7 latch counts the RO extenders + the trailing writer. */
  arts_event_hint_t d7_latch_hint = ARTS_EVENT_HINT_LATCH(N_EXTEND_READERS + 1);
  d7_latch_hint.rank = 0;
  arts_guid_t d7_latch = arts_event_create(&d7_latch_hint);
  uint64_t d7_pv[1] = {(uint64_t)d7_latch};

  /* A trailing RW writer parked at home (w>0) plus a burst of RO requests: the
   * RO phase must extend (D7) and serve all readers, then the writer runs. */
  arts_guid_t tw = arts_edt_create(trail_writer_edt, 1, d7_pv, 1,
                                   &(arts_edt_hint_t){.rank = home});
  arts_add_dependence(d7, tw, 0, DB_MODE_RW);
  for (int i = 0; i < N_EXTEND_READERS; i++) {
    unsigned int r = (nranks > 1) ? (unsigned int)(i % nranks) : 0u;
    arts_guid_t rd = arts_edt_create(extend_reader_edt, 1, d7_pv, 1,
                                     &(arts_edt_hint_t){.rank = r});
    arts_add_dependence(d7, rd, 0, DB_MODE_RO);
  }

  arts_guid_t shut =
      arts_edt_create(shutdown_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0});
  arts_add_dependence(d7_latch, shut, 0, DB_MODE_NULL);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}

#endif /* ARTS_PROTOCOL_RWLOCK */
