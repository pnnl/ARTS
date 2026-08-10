/* SPDX-License-Identifier: Apache-2.0
 *
 * T105 — WRF_VAL sentinel-DB snapshot serve: a zero-size DB acquired cross-rank
 *        must wake the parked reader (it does not hang), with ptr == NULL.
 *
 * Targets the master==NULL reply path in arts_handler_db_snapshot_request
 * (census 10, hazard #4).  Under WRF_VAL a non-home acquire parks on a
 * SNAPSHOT_REQUEST.  When the home buffer is NULL the home replies version=0
 * with no data.  master==NULL covers two sub-cases, BOTH exercised here:
 *   (a) sentinel DB — db_size == 0, so home never installs a buffer; and
 *   (b) HOME_RECV pre-PUBLISH — a remote acquire arriving before the home
 *       has any installed buffer for the GUID.
 * In both cases the parked reader must wake with ptr==NULL and complete (no
 * hang).  A stranded reader is caught by ctest's TIMEOUT (no in-test spin).
 *
 * Structure (2+ ranks; self-skips otherwise):
 *   home = rank 0; readers run on rank 1.  Per round a fresh zero-size DB is
 *   created (sentinel), released, then acquired RO and RW from rank 1 inside a
 *   finish scope.  The reader/writer bodies require ptr==NULL (a non-NULL ptr
 *   for a zero-size DB would be a contract violation).
 *
 * config_specific: WRF_VAL only — the unified-acquire/SNAPSHOT model is WRF_VAL's;
 * the ownership protocols serve zero-size DBs through a different path.
 */

#include "arts.h"

#include <stdint.h>
#include <stdio.h>

#define ROUNDS 16u

#if !defined(ARTS_PROTOCOL_WRF_VAL)
void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_printf("SKIP wrf_val_sentinel_snapshot: WRF_VAL-only\n");
  arts_shutdown();
}
#else

/* Acquires a zero-size sentinel DB; the runtime must deliver ptr==NULL and the
 * EDT must wake (not hang) regardless of mode. */
static void sentinel_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                         arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  uint64_t mode = paramv[0];
  if (depv[0].ptr != NULL) {
    arts_printf("wrf_val_sentinel_snapshot: FAIL mode=%llu non-NULL ptr for "
                "zero-size DB\n",
                (unsigned long long)mode);
    arts_abort(1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  if (arts_get_total_ranks() < 2) {
    arts_printf("SKIP wrf_val_sentinel_snapshot: requires 2+ ranks\n");
    arts_shutdown();
    return;
  }

  for (unsigned int rnd = 0; rnd < ROUNDS; rnd++) {
    /* Zero-size DB: db_size == 0 → home installs no buffer → snapshot serve
     * takes the master==NULL / version=0 reply path. */
    void *addr = NULL;
    arts_guid_t db = arts_db_create(&addr, 0, ARTS_DB, ARTS_DB_PROP_NONE,
                                    &(arts_db_hint_t){.rank = 0});
    arts_db_release(db, DB_MODE_RW);

    arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

    /* RO acquire on a non-home rank — parks on SNAPSHOT_REQUEST, woken by the
     * version=0 NULL reply. */
    uint64_t ro = (uint64_t)DB_MODE_RO;
    arts_guid_t rd =
        arts_edt_create(sentinel_edt, 1, &ro, 1,
                        &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
    arts_add_dependence(db, rd, 0, DB_MODE_RO);

    /* RW acquire on the same non-home rank — also parks under WRF_VAL (unified
     * acquire); the version=0 NULL reply must wake it too. */
    uint64_t rw = (uint64_t)DB_MODE_RW;
    arts_guid_t wr =
        arts_edt_create(sentinel_edt, 1, &rw, 1,
                        &(arts_edt_hint_t){.rank = 1, .finish_event = fe});
    arts_add_dependence(db, wr, 0, DB_MODE_RW);

    arts_event_wait(fe);
    arts_db_destroy(db);
  }

  arts_printf("PASS wrf_val_sentinel_snapshot: %u rounds RO+RW, parked readers "
              "woke (NULL ptr)\n",
              ROUNDS);
  arts_shutdown();
}
#endif

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
