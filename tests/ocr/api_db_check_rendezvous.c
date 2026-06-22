/* SPDX-License-Identifier: Apache-2.0
 *
 * T289 — DB labeled-create CHECK / rendezvous (arts_db_hint_t.check).
 *
 * Target: the DB-create CHECK path in libs/src/core/db.c
 *   if (check) arts_route_table_install_if_absent(...)   // first-wins
 *   else       arts_route_table_install(...)             // replace
 * The DB case of arts_db_hint_t.check is suspected unexercised (event.check is
 * covered by labeled-event tests; the grep hits on "check" in coherence tests
 * are local variable names, not the hint field).
 *
 * Correct behavior pinned (OCR GUID_PROP_CHECK semantics):
 *   - check = false : a labeled-GUID reuse OVERWRITES the prior generation
 *                     (install replaces; the displaced cb is released).
 *   - check = true  : a create at an already-occupied home-local GUID FAILS
 *                     to install (install_if_absent CAS loses), so the FIRST
 *                     generation's data persists and a later reader observes
 *                     the first creator's payload, not the second's.
 *
 * Both behaviors are observed by a reader EDT that depends RO on the labeled
 * GUID and reads back the surviving sentinel:
 *   GUID A: first writes 0xAAAA, then a CHECK-create tries 0xBBBB  -> reader
 *           must see 0xAAAA (CHECK collision kept the first generation).
 *   GUID B: first writes 0xCCCC, then a replace-create writes 0xDDDD -> reader
 *           must see 0xDDDD (default replace overwrote the first generation).
 *
 * Config-agnostic: pure single-node public-API DB semantics.
 * exposes_runtime_bug = false (pins correct CHECK-vs-replace install).
 */
#include "arts.h"
#include <stdint.h>

#define SENT_A_FIRST 0xAAAAu
#define SENT_A_SECOND 0xBBBBu
#define SENT_B_FIRST 0xCCCCu
#define SENT_B_SECOND 0xDDDDu

static int g_failed = 0;

/* Reader gated on both labeled DBs being RO-readable.
 *   depv[0] = labeled GUID A (RO)  -> must read SENT_A_FIRST (check kept gen 0)
 *   depv[1] = labeled GUID B (RO)  -> must read SENT_B_SECOND (replace won) */
void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint32_t a = depv[0].ptr ? *(uint32_t *)depv[0].ptr : 0u;
  uint32_t b = depv[1].ptr ? *(uint32_t *)depv[1].ptr : 0u;

  if (a != SENT_A_FIRST) {
    arts_printf("FAIL api_db_check_rendezvous: CHECK collision did not keep "
                "first generation (got 0x%X want 0x%X)\n",
                a, SENT_A_FIRST);
    g_failed = 1;
  }
  if (b != SENT_B_SECOND) {
    arts_printf(
        "FAIL api_db_check_rendezvous: default replace did not overwrite "
        "(got 0x%X want 0x%X)\n",
        b, SENT_B_SECOND);
    g_failed = 1;
  }
  if (!g_failed) {
    arts_printf("PASS api_db_check_rendezvous: CHECK keeps first gen, replace "
                "overwrites\n");
  }
  arts_shutdown();
}

/* Create a labeled DB at `guid`, write `sentinel`, release.  `check` selects
 * install_if_absent (first-wins) vs install (replace). */
static void labeled_write(arts_guid_t guid, uint32_t sentinel, bool check) {
  arts_db_hint_t h = ARTS_DB_HINT_DEFAULTS;
  h.guid = guid;
  h.check = check;
  void *p = NULL;
  arts_db_create(&p, sizeof(uint32_t), ARTS_DB, ARTS_DB_PROP_NONE, &h);
  /* On a CHECK collision the install fails: p points at the abandoned second
   * buffer (or NULL).  Writing into it is harmless — it is never published —
   * but guard against NULL.  The visible generation is the one in the route
   * table, which the reader observes. */
  if (p) {
    *(uint32_t *)p = sentinel;
  }
  arts_db_release(guid, DB_MODE_RW);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== api_db_check_rendezvous ===\n");

  /* Pre-reserve two labeled DB GUIDs local to this rank. */
  arts_guid_t ga = arts_guid_reserve(ARTS_GUID_DB, arts_get_current_rank());
  arts_guid_t gb = arts_guid_reserve(ARTS_GUID_DB, arts_get_current_rank());

  /* GUID A: first generation 0xAAAA (replace path), then a CHECK-create that
   * must lose the install and keep 0xAAAA. */
  labeled_write(ga, SENT_A_FIRST, /*check=*/false);
  labeled_write(ga, SENT_A_SECOND, /*check=*/true);

  /* GUID B: first generation 0xCCCC, then a default (replace) create that must
   * overwrite to 0xDDDD. */
  labeled_write(gb, SENT_B_FIRST, /*check=*/false);
  labeled_write(gb, SENT_B_SECOND, /*check=*/false);

  arts_guid_t r = arts_edt_create(reader_edt, 0, NULL, 2, NULL);
  arts_add_dependence(ga, r, 0, DB_MODE_RO);
  arts_add_dependence(gb, r, 1, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
