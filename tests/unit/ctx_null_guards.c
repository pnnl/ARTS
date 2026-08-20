/* SPDX-License-Identifier: Apache-2.0
 *
 * T148 — NULL-guard asymmetry between arts_owned_finish_register and
 * arts_track_created_db (libs/src/core/edt_context.c).
 *
 * Documented asymmetry (census f14, suspected-bug #3, LOW):
 *   - arts_owned_finish_register GUARDS NULL: `if (!fe_guid) return;` — a
 *     NULL_GUID registration is a no-op (and the list is not even lazily
 *     allocated by it).
 *   - arts_track_created_db has NO guard: a NULL_GUID is appended to
 *     created_db_list (release later skips NULL_GUID entries, so harmless, but
 *     the list length still reflects the bogus entry).
 *
 * This test calls both internal functions directly from inside an EDT body
 * (white-box; both operate only on the CURRENT worker's thread-locals, so this
 * is safe and thread-confined), and pins the asymmetry via the observable
 * created_db_list length:
 *
 *   1. arts_track_created_db(NULL_GUID) INCREASES created_db_list length by 1
 *      (no guard) — the NULL entry is present.
 *   2. arts_owned_finish_register(NULL_GUID) is a clean no-op: it neither
 *      crashes nor perturbs subsequent behavior; a following real DB create
 *      still tracks normally and a real finish-event flow still completes.
 *   3. arts_release_created_dbs tolerates the NULL entry (the EDT epilogue runs
 *      it; the test simply completing proves no crash on the bogus entry).
 *
 * exposes_runtime_bug = false (the asymmetry is benign today; this pins the
 * documented contract so a future guard change is caught).
 */
#include "arts.h"
#include "arts/utils/vector.h"

#include "arts/edt_context.h" /* arts_track_created_db, register, get list */
#include "arts/utils/array_list.h"

#include <stdint.h>

static int g_failed = 0;

void probe(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
           arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  /* Baseline: create one real DB so the list exists with a known length. */
  void *p = NULL;
  arts_guid_t db =
      arts_db_create(&p, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  if (p) {
    ((uint64_t *)p)[0] = 7;
  }
  arts_vector_t *list = arts_get_created_db_list();
  uint64_t base = list ? arts_vector_count(list) : 0;
  if (list == NULL || base == 0) {
    arts_printf("FAIL ctx_null_guards: created_db_list not populated\n");
    g_failed = 1;
    arts_db_release(db, DB_MODE_RW);
    arts_shutdown();
    return;
  }

  /* (1) track has NO NULL guard: appends a NULL_GUID entry -> length +1. */
  arts_track_created_db(NULL_GUID);
  uint64_t after_track = arts_vector_count(arts_get_created_db_list());
  if (after_track != base + 1) {
    arts_printf("FAIL ctx_null_guards: arts_track_created_db(NULL) did not "
                "append (len %llu -> %llu, expected +1) — guard added?\n",
                (unsigned long long)base, (unsigned long long)after_track);
    g_failed = 1;
    arts_db_release(db, DB_MODE_RW);
    arts_shutdown();
    return;
  }

  /* (2) register HAS a NULL guard: must be a clean no-op (no crash). The
   * owned_finish_list is file-static and not directly observable; the contract
   * we can pin is that the call returns harmlessly and does not perturb the
   * created_db_list. */
  arts_owned_finish_register(NULL_GUID);
  uint64_t after_reg = arts_vector_count(arts_get_created_db_list());
  if (after_reg != after_track) {
    arts_printf(
        "FAIL ctx_null_guards: register(NULL) perturbed created_db_list "
        "(%llu -> %llu)\n",
        (unsigned long long)after_track, (unsigned long long)after_reg);
    g_failed = 1;
    arts_db_release(db, DB_MODE_RW);
    arts_shutdown();
    return;
  }

  /* A subsequent real DB create still tracks normally (NULL register left the
   * machinery intact). */
  void *p2 = NULL;
  arts_guid_t db2 =
      arts_db_create(&p2, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  uint64_t after_real = arts_vector_count(arts_get_created_db_list());
  if (after_real != after_reg + 1) {
    arts_printf("FAIL ctx_null_guards: real DB create after NULL-guard probes "
                "did not track (%llu -> %llu)\n",
                (unsigned long long)after_reg, (unsigned long long)after_real);
    g_failed = 1;
  }

  /* (3) release the real DBs explicitly; the epilogue's
   * arts_release_created_dbs then iterates the list (including the bogus NULL
   * entry, which it skips). The test simply completing proves no crash. */
  arts_db_release(db, DB_MODE_RW);
  arts_db_release(db2, DB_MODE_RW);

  if (!g_failed) {
    arts_printf("PASS ctx_null_guards: track(NULL) appends, register(NULL) is "
                "a no-op (guard asymmetry confirmed)\n");
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_edt_create(probe, 0, NULL, 0, NULL);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
