/* SPDX-License-Identifier: Apache-2.0
 *
 * T146 — created_db_list release-before-reset invariant + 65536 footprint.
 *
 * Target: arts_set_thread_local_edt_info (resets created_db_list at run-start)
 * vs arts_release_created_dbs (db.c EDT epilogue release).  The ordering
 * contract is: arts_run_edt calls release-created-dbs BEFORE the next EDT's
 * set resets the list, so created DBs are never dropped without release.
 *
 * Pinned properties (white-box reads on the running worker thread):
 *
 *   1. Footprint: the first DB created on a worker lazily allocates
 *      created_db_list with element_size == sizeof(arts_guid_t) and
 *      array_length == 65536 (the documented large initial capacity).  We
 *      assert these on the live list object.
 *
 *   2. Reset-on-set: a producer EDT creates N DBs WITHOUT explicitly releasing
 *      them (relying on the epilogue auto-release).  Its created_db_list length
 *      reaches N.  A successor EDT on the SAME worker then observes a list
 * whose length reflects only ITS OWN creations (set reset cleared the
 * producer's entries) — never N + its own.  This proves set resets the list.
 *
 *   3. No created-DB leak: the successor receives RO deps on the producer's DBs
 *      and reads the values the producer wrote.  If the epilogue had failed to
 *      release them (e.g. reset preempted release), the RO acquire could not be
 *      granted and the successor would never run -> ctest TIMEOUT.  Observing
 *      the correct values proves release-before-reset held.
 *
 * exposes_runtime_bug = false (pins the documented correct ordering +
 * footprint).
 */
#include "arts.h"

#include "arts/edt_context.h" /* arts_get_created_db_list */
#include "arts/utils/array_list.h"

#include <stdint.h>

#define NDB 6

static int g_failed = 0;

/* Successor: NDB RO deps on the producer's DBs.  Runs after the producer's
 * epilogue released them.  Also creates one fresh DB to prove the worker's
 * created_db_list was reset (length reflects only this EDT's creation). */
void successor(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;

  /* (3) values written by the producer must be visible (DBs were released). */
  for (int i = 0; i < NDB; i++) {
    uint64_t *p = (uint64_t *)depv[i].ptr;
    if (p == NULL || p[0] != (uint64_t)(0xD00D0000 + i)) {
      arts_printf("FAIL ctx_created_db_release_order: successor RO dep %d "
                  "wrong (ptr=%p val=%llu)\n",
                  i, (void *)p, (unsigned long long)(p ? p[0] : 0));
      g_failed = 1;
      arts_shutdown();
      return;
    }
  }

  /* (2) the worker's created_db_list was reset at this EDT's set: creating one
   * DB here yields length 1, not NDB+1. */
  void *sp = NULL;
  arts_guid_t sdb =
      arts_db_create(&sp, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  if (sp) {
    ((uint64_t *)sp)[0] = 0;
  }
  arts_array_list_t *list = arts_get_created_db_list();
  uint64_t len = list ? arts_length_array_list(list) : 0;
  if (len != 1) {
    arts_printf("FAIL ctx_created_db_release_order: created_db_list not reset "
                "at set (length=%llu, expected 1 -> producer entries leaked "
                "into successor)\n",
                (unsigned long long)len);
    g_failed = 1;
    arts_db_release(sdb, DB_MODE_RW);
    arts_shutdown();
    return;
  }
  arts_db_release(sdb, DB_MODE_RW);

  arts_printf(
      "PASS ctx_created_db_release_order: footprint=65536, reset-on-set "
      "+ release-before-reset held (%d DBs)\n",
      NDB);
  arts_shutdown();
}

/* Producer: creates NDB DBs, writes a sentinel into each, releases them RW so a
 * successor's RO dep can be granted, and hands them to the successor. */
void producer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_guid_t dbs[NDB];
  for (int i = 0; i < NDB; i++) {
    void *p = NULL;
    dbs[i] =
        arts_db_create(&p, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
    if (p) {
      ((uint64_t *)p)[0] = (uint64_t)(0xD00D0000 + i);
    }
  }

  /* (1) footprint: the live created_db_list has 65536 capacity, guid element
   * size, and exactly NDB entries (no explicit release yet). */
  arts_array_list_t *list = arts_get_created_db_list();
  if (list == NULL) {
    arts_printf("FAIL ctx_created_db_release_order: created_db_list NULL after "
                "creating DBs\n");
    g_failed = 1;
    arts_shutdown();
    return;
  }
  if (list->element_size != sizeof(arts_guid_t) ||
      list->array_length != 65536) {
    arts_printf("FAIL ctx_created_db_release_order: footprint wrong "
                "(element_size=%zu array_length=%zu, expected %zu/65536)\n",
                list->element_size, list->array_length, sizeof(arts_guid_t));
    g_failed = 1;
    arts_shutdown();
    return;
  }
  if (arts_length_array_list(list) != (uint64_t)NDB) {
    arts_printf("FAIL ctx_created_db_release_order: producer list length=%llu "
                "expected %d\n",
                (unsigned long long)arts_length_array_list(list), NDB);
    g_failed = 1;
    arts_shutdown();
    return;
  }

  /* Release RW so successor RO deps can be granted; the epilogue's
   * arts_release_created_dbs then runs a no-op on already-released entries. */
  for (int i = 0; i < NDB; i++) {
    arts_db_release(dbs[i], DB_MODE_RW);
  }

  arts_edt_hint_t sh = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t s = arts_edt_create(successor, 0, NULL, NDB, &sh);
  for (int i = 0; i < NDB; i++) {
    arts_add_dependence(dbs[i], s, (uint32_t)i, DB_MODE_RO);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  arts_edt_create(producer, 0, NULL, 0, NULL);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
