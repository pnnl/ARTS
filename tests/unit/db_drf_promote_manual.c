/* SPDX-License-Identifier: Apache-2.0
 *
 * T107 — DB-DRF manual-diagnostic promotion: a CI-safe, deterministically
 *        asserting guard for the two manual lost-update diagnostics
 *        (db_drf_multi_writer_concurrent, db_drf_counter_race).
 *
 * Those two diagnostics document the DB-DRF "last-writer-wins" / lost-update
 * behavior but were left out of CTest because their numeric outcome is
 * non-deterministic and cannot be a PASS_REGULAR_EXPRESSION.  The behavior is
 * still worth a regression guard: the outcome must always land in the set of
 * outcomes the DB-DRF contract PERMITS, and never outside it.  This test
 * asserts exactly that legal-outcome set, so it is green-under-CI while still
 * failing if the protocol ever produced an out-of-contract result (e.g. a torn
 * buffer or an impossible counter value).
 *
 * Two sub-scenarios, run in sequence within one process:
 *   (A) multi-writer (from db_drf_multi_writer_concurrent): two ranks each fill
 *       the whole DB with their rank-id; the final buffer must be internally
 *       consistent and equal to exactly ONE writer's id {0,1}.  A torn mix is a
 *       contract violation (FAIL).
 *   (B) counter race (from db_drf_counter_race): two ranks each do N untracked
 *       increments on a shared counter with no DAG ordering; the final value
 *       must be in {N, 2N} (lost-update or fully-serialized) — never anything
 *       else.  This pins the legal lost-update window without requiring a
 *       specific deterministic answer.
 *
 * config_specific: MRMW only — the no-ownership concurrent cross-rank RW is the
 * DB-DRF contract; ownership protocols serialize and would yield 2N for (B) and
 * are exercised by their own coherence suites.
 */

#include "arts.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define DB_SIZE 4096
#define INCREMENTS_PER_RANK 1000u

#if !defined(ARTS_PROTOCOL_MRMW)
void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("SKIP db_drf_promote_manual: MRMW-only\n");
  arts_shutdown();
}
#else

/* ----- Sub-scenario A: multi-writer last-writer-wins ----- */

static void mw_writer_edt(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint8_t my_rank = (uint8_t)paramv[0];
  uint8_t *p = (uint8_t *)depv[0].ptr;
  if (p != NULL) {
    memset(p, my_rank, DB_SIZE);
  }
}

/* ----- Sub-scenario B: counter race ----- */

static void cr_inc_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                       arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *counter = (uint64_t *)depv[0].ptr;
  if (counter != NULL) {
    for (uint32_t i = 0; i < INCREMENTS_PER_RANK; i++) {
      *counter += 1u;
    }
  }
}

/* Final verifier: checks BOTH sub-scenarios' results against the DB-DRF
 * legal-outcome sets, then ends the run. */
static void verifier_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint8_t *mw = (uint8_t *)depv[0].ptr;   /* multi-writer DB (RO) */
  uint64_t *cr = (uint64_t *)depv[1].ptr; /* counter DB (RO) */

  /* (A) internally consistent + value in {0,1}. */
  if (mw == NULL) {
    arts_printf("db_drf_promote_manual: FAIL (A) NULL buffer\n");
    arts_abort(1);
  }
  uint8_t v = mw[0];
  for (size_t i = 1; i < DB_SIZE; i++) {
    if (mw[i] != v) {
      arts_printf("db_drf_promote_manual: FAIL (A) torn buffer @%zu\n", i);
      arts_abort(1);
    }
  }
  if (v != 0 && v != 1) {
    arts_printf("db_drf_promote_manual: FAIL (A) value=%u outside {0,1}\n",
                (unsigned)v);
    arts_abort(1);
  }

  /* (B) value in {N, 2N}. */
  uint64_t got = cr ? *cr : 0u;
  uint64_t n = (uint64_t)INCREMENTS_PER_RANK;
  if (got != n && got != 2u * n) {
    arts_printf("db_drf_promote_manual: FAIL (B) counter=%llu outside {%llu, "
                "%llu}\n",
                (unsigned long long)got, (unsigned long long)n,
                (unsigned long long)(2u * n));
    arts_abort(1);
  }

  arts_printf("db_drf_promote_manual: PASS (A) final=%u (B) counter=%llu "
              "(both in DB-DRF legal set)\n",
              (unsigned)v, (unsigned long long)got);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  if (arts_get_total_ranks() < 2) {
    arts_printf("SKIP db_drf_promote_manual: requires 2+ ranks\n");
    arts_shutdown();
    return;
  }

  /* (A) multi-writer DB. */
  void *mw_addr = NULL;
  arts_guid_t mw_db =
      arts_db_create(&mw_addr, DB_SIZE, ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  memset(mw_addr, 0xff, DB_SIZE);
  arts_db_release(mw_db, DB_MODE_RW);

  /* (B) counter DB. */
  void *cr_addr = NULL;
  arts_guid_t cr_db =
      arts_db_create(&cr_addr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE,
                     &(arts_db_hint_t){.rank = 0});
  *(uint64_t *)cr_addr = 0u;
  arts_db_release(cr_db, DB_MODE_RW);

  /* verifier reads both DBs RO; it consumes the writers' finish scope (fe) at
   * slot 2 so it runs only after all four writers released, and lives in its
   * own finish scope (e_ver) for orderly completion. */
  arts_guid_t e_ver = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_guid_t ver =
      arts_edt_create(verifier_edt, 0, NULL, 3,
                      &(arts_edt_hint_t){.rank = 0, .finish_event = e_ver});
  arts_add_dependence(mw_db, ver, 0, DB_MODE_RO);
  arts_add_dependence(cr_db, ver, 1, DB_MODE_RO);

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  arts_add_dependence(fe, ver, 2, DB_MODE_NULL);

  /* Two multi-writers + two counter incrementers, one EDT per rank each, all
   * inside the fe finish scope (no DAG ordering among them). */
  for (unsigned int r = 0; r < 2; r++) {
    uint64_t rp = (uint64_t)r;
    arts_guid_t mw_edt =
        arts_edt_create(mw_writer_edt, 1, &rp, 1,
                        &(arts_edt_hint_t){.rank = r, .finish_event = fe});
    arts_add_dependence(mw_db, mw_edt, 0, DB_MODE_RW);
    (void)mw_edt;

    arts_guid_t cr_edt =
        arts_edt_create(cr_inc_edt, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = r, .finish_event = fe});
    arts_add_dependence(cr_db, cr_edt, 0, DB_MODE_RW);
    (void)cr_edt;
  }
}
#endif

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
