/* SPDX-License-Identifier: Apache-2.0
 *
 * T291 — arts_current_finish_event NULL_GUID outside a finish scope.
 *
 * Target: the public ambient-finish-scope accessor arts_current_finish_event(),
 * which "returns the ambient finish-scope GUID inherited or joined by the
 * running EDT, or NULL_GUID if it belongs to no finish scope."  The only prior
 * exercise was inside termination_detection; the NULL_GUID (no-scope) return is
 * the thin/under-tested branch.
 *
 * Correct behavior pinned:
 *   - main_edt is created with ARTS_EDT_HINT_DEFAULTS (finish_event =
 *     NULL_GUID, inheriting the caller's ambient scope), and the top-level
 *     runtime has NO ambient finish scope -> arts_current_finish_event()
 *     MUST return NULL_GUID inside main_edt.
 *   - An EDT created with hint.finish_event = fe joins that scope ->
 *     arts_current_finish_event() inside it MUST return exactly fe.
 *   - A child EDT created (without an explicit finish_event) inside the scoped
 *     EDT inherits the ambient scope -> also returns fe.
 *
 * Config-agnostic single-node public-API check.
 * exposes_runtime_bug = false (pins the NULL-scope and inherit return values).
 */
#include "arts.h"
#include <stdint.h>

static int g_failed = 0;

/* Runs inside finish scope fe (passed both as paramv[0] and as the EDT's
 * finish_event).  Verifies the accessor returns fe, then spawns a child that
 * must INHERIT the same ambient scope. */
void scoped_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]);
void child_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]);

void child_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  arts_guid_t fe = (paramc >= 1) ? (arts_guid_t)paramv[0] : NULL_GUID;
  arts_guid_t cur = arts_current_finish_event();
  if (cur != fe) {
    arts_printf("FAIL api_current_finish_event_null: child did not inherit "
                "ambient scope (got %ld want %ld)\n",
                (long)cur, (long)fe);
    g_failed = 1;
  } else {
    arts_printf("  ok: child inherited ambient finish scope %ld\n", (long)fe);
  }
}

void scoped_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)depc;
  (void)depv;
  arts_guid_t fe = (paramc >= 1) ? (arts_guid_t)paramv[0] : NULL_GUID;
  arts_guid_t cur = arts_current_finish_event();
  if (cur != fe) {
    arts_printf("FAIL api_current_finish_event_null: scoped EDT current scope "
                "%ld != joined fe %ld\n",
                (long)cur, (long)fe);
    g_failed = 1;
  } else {
    arts_printf("  ok: scoped EDT reports joined finish scope %ld\n", (long)fe);
  }
  /* Child inherits the ambient scope (no explicit finish_event in its hint). */
  uint64_t pv[1] = {(uint64_t)fe};
  arts_edt_create(child_edt, 1, pv, 0, NULL);
}

/* Finalizer: gated on the finish event draining; reports the verdict. */
void done_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  if (!g_failed) {
    arts_printf("PASS api_current_finish_event_null: NULL outside scope, fe "
                "inside, inherited by child\n");
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== api_current_finish_event_null ===\n");

  /* (1) Top-level: main_edt belongs to NO finish scope. */
  arts_guid_t top = arts_current_finish_event();
  if (top != NULL_GUID) {
    arts_printf("FAIL api_current_finish_event_null: top-level scope is %ld, "
                "expected NULL_GUID\n",
                (long)top);
    g_failed = 1;
  } else {
    arts_printf("  ok: top-level main_edt has no finish scope (NULL_GUID)\n");
  }

  /* (2) Create a finish scope and an EDT that joins it. */
  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);
  uint64_t pv[1] = {(uint64_t)fe};
  arts_edt_hint_t sh = ARTS_EDT_HINT_DEFAULTS;
  sh.finish_event = fe;
  arts_edt_create(scoped_edt, 1, pv, 0, &sh);

  /* (3) done_edt fires when the finish scope drains (scoped + child done). */
  arts_edt_hint_t dh = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t done = arts_edt_create(done_edt, 0, NULL, 1, &dh);
  arts_add_dependence(fe, done, 0, DB_MODE_NULL);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
