/* SPDX-License-Identifier: Apache-2.0
 *
 * T139 (C13) — arts_edt_register_cb_deleter constructor-order vs route_table.
 *
 * arts_edt_register_cb_deleter is a __attribute__((constructor)) that registers
 * the EDT cb deleter into the route table's per-kind deleter table BEFORE main.
 * The risk it guards against is constructor-ordering vs the route_table's own
 * initialization: if the EDT deleter registration ran before the route table's
 * per-kind table existed, every EDT destroy would free with the wrong (or NULL)
 * deleter.  There is no direct getter for "the registered deleter for a kind",
 * so this test verifies the registration ORDER held end-to-end by:
 *
 *   1. Asserting arts_edt_get_deleter() (the single source of truth the
 *      constructor registers) is non-NULL by the time the runtime is up.
 *   2. Driving a real create -> install -> destroy of an EDT GUID so the
 *      route_table actually invokes the registered deleter (a wrong/absent
 *      registration would leak or crash here under ASan).
 *
 * pure_unit/runtime, all configs.  Single node.
 */
#include "arts.h"
#include "arts/edt.h"

#include <stdint.h>
#include <stdio.h>

static int g_failed = 0;

/* A trivial EDT we create only to exercise install + destroy of an EDT GUID
 * through the route table (which selects the deleter registered by the
 * constructor under test). */
void noop_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_register_cb_deleter ===\n");

  /* (1) The constructor must have populated the single-source-of-truth
   * deleter pointer before main / runtime startup. */
  void (*deleter)(void *) = arts_edt_get_deleter();
  if (deleter == NULL) {
    arts_printf("FAIL edt_register_cb_deleter: arts_edt_get_deleter() is NULL "
                "(constructor did not run / ran out of order)\n");
    g_failed = 1;
    arts_shutdown();
    return;
  }

  /* (2) Create an EDT that never fires (one un-satisfied dep), then destroy it.
   * The route_table's set_destroyed -> last-ref-drop path must invoke the
   * registered EDT deleter; an unregistered/wrong deleter would leak (ASan) or
   * crash. */
  arts_guid_t reserved =
      arts_guid_reserve(ARTS_GUID_EDT, arts_get_current_rank());
  arts_edt_hint_t h = ARTS_EDT_HINT_DEFAULTS;
  h.guid = reserved;
  arts_guid_t e = arts_edt_create(noop_edt, 0, NULL, 1, &h);
  if (e == NULL_GUID) {
    arts_printf("FAIL edt_register_cb_deleter: create returned NULL_GUID\n");
    g_failed = 1;
    arts_shutdown();
    return;
  }
  arts_edt_destroy(e);

  arts_printf("PASS edt_register_cb_deleter: EDT deleter registered before "
              "main and route_table destroy path is clean\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
