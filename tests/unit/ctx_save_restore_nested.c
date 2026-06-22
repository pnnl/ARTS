/* SPDX-License-Identifier: Apache-2.0
 *
 * T145 — EDT thread-local context save/restore across nested arts_event_wait.
 *
 * Target: arts_edt_ctx_save / arts_edt_ctx_restore
 * (libs/src/core/edt_context.c) exercised indirectly by arts_event_wait
 * (event.c), which stashes the outer EDT context, spins the scheduler running
 * OTHER EDTs (some of which create DBs and finish events), then restores the
 * outer context.
 *
 * Invariants pinned (white-box, all reads on the same worker thread that runs
 * the orchestrator EDT, so the thread-local context is the orchestrator's):
 *
 *   1. After each arts_event_wait returns, the outer EDT's identity
 *      (current_edt + current_edt_guid mirror) is intact — exactly the
 *      orchestrator EDT, never NULL or a nested EDT.
 *   2. The orchestrator's own created_db_list pointer is unchanged across the
 *      wait (the nested run's list is deleted by restore; the saved outer list
 *      pointer comes back identically) and its length is unchanged — the
 *      orchestrator's created-DB set is not corrupted by nested DB creation.
 *   3. No per-wait growth: repeating arts_event_wait MANY times over nested
 *      DB-creating EDTs must not grow the orchestrator's created_db_list length
 *      (restore's single-delete-assumption means at most one nested list exists
 *      at restore; the outer list is byte-for-byte the same object each time).
 *
 * A stranded waiter (broken finish-scope drain) surfaces as a ctest TIMEOUT.
 * On any invariant violation the test prints FAIL and arts_shutdown()s; the
 * pass token "PASS ctx_save_restore_nested" is only printed on full success.
 *
 * exposes_runtime_bug = false (pins the documented correct save/restore
 * contract; B-ctx-restore single-delete leak is a latent fragility, this test
 * guards against its regression by asserting no growth).
 */
#include "arts.h"

#include "arts/edt_context.h"   /* current_edt, arts_get_created_db_list */
#include "arts/runtime_state.h" /* arts_thread_info.current_edt_guid */
#include "arts/utils/array_list.h"

#include <stdint.h>

#define WAVES 16 /* number of nested arts_event_wait rounds */
#define LEAVES 4 /* leaves per wave, each creating a DB */

static int g_failed = 0;

/* Nested leaf: runs on the same worker while the orchestrator is parked in
 * arts_event_wait.  It creates a DB (tracked into the NESTED context's
 * created_db_list, not the orchestrator's) so we can prove the outer list is
 * untouched.  The DB is released RW so the auto-release epilogue reclaims it.
 */
void nested_leaf(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  void *p = NULL;
  arts_guid_t db =
      arts_db_create(&p, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  if (p) {
    ((uint64_t *)p)[0] = 0xC0FFEE;
  }
  arts_db_release(db, DB_MODE_RW);
}

void orchestrator(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  /* Snapshot the orchestrator's own context BEFORE any wait. */
  struct arts_edt_s *outer_edt = current_edt;
  arts_guid_t outer_guid = arts_thread_info.current_edt_guid;
  if (outer_edt == NULL || outer_guid == NULL_GUID) {
    arts_printf("FAIL ctx_save_restore_nested: no running EDT context\n");
    g_failed = 1;
    arts_shutdown();
    return;
  }

  /* Create an orchestrator-owned DB so its created_db_list is non-NULL and
   * has a known length to compare against after each wait. */
  void *op = NULL;
  arts_guid_t odb =
      arts_db_create(&op, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  if (op) {
    ((uint64_t *)op)[0] = 1;
  }
  arts_array_list_t *outer_list0 = arts_get_created_db_list();
  uint64_t outer_len0 = outer_list0 ? arts_length_array_list(outer_list0) : 0;
  if (outer_list0 == NULL || outer_len0 == 0) {
    arts_printf("FAIL ctx_save_restore_nested: created_db_list not populated "
                "by orchestrator DB create\n");
    g_failed = 1;
    arts_db_release(odb, DB_MODE_RW);
    arts_shutdown();
    return;
  }
  arts_db_release(odb, DB_MODE_RW);

  for (int w = 0; w < WAVES; w++) {
    arts_event_hint_t fh = ARTS_EVENT_HINT_FINISH;
    arts_guid_t fe = arts_event_create(&fh);
    arts_edt_hint_t eh = ARTS_EDT_HINT_DEFAULTS;
    eh.finish_event = fe;
    for (int i = 0; i < LEAVES; i++) {
      arts_edt_create(nested_leaf, 0, NULL, 0, &eh);
    }

    /* Park the orchestrator: save outer ctx, spin scheduler (runs leaves that
     * each create+release a DB under the nested context), restore outer ctx. */
    arts_event_wait(fe);

    /* (1) outer identity intact after restore. */
    if (current_edt != outer_edt ||
        arts_thread_info.current_edt_guid != outer_guid) {
      arts_printf("FAIL ctx_save_restore_nested: outer EDT identity corrupted "
                  "after wait %d (current_edt=%p want %p)\n",
                  w, (void *)current_edt, (void *)outer_edt);
      g_failed = 1;
      arts_shutdown();
      return;
    }

    /* (2)+(3) outer created_db_list is the same object, same length — nested
     * DB creation did not corrupt or grow it. */
    arts_array_list_t *outer_list = arts_get_created_db_list();
    uint64_t outer_len = outer_list ? arts_length_array_list(outer_list) : 0;
    if (outer_list != outer_list0) {
      arts_printf("FAIL ctx_save_restore_nested: created_db_list pointer "
                  "changed after wait %d (%p -> %p)\n",
                  w, (void *)outer_list0, (void *)outer_list);
      g_failed = 1;
      arts_shutdown();
      return;
    }
    if (outer_len != outer_len0) {
      arts_printf("FAIL ctx_save_restore_nested: created_db_list grew across "
                  "wait %d (%llu -> %llu)\n",
                  w, (unsigned long long)outer_len0,
                  (unsigned long long)outer_len);
      g_failed = 1;
      arts_shutdown();
      return;
    }
  }

  arts_printf("PASS ctx_save_restore_nested: outer ctx intact + no list growth "
              "across %d nested waits\n",
              WAVES);
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_edt_create(orchestrator, 0, NULL, 0, NULL);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
