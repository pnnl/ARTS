/* SPDX-License-Identifier: Apache-2.0
 *
 * T142 (C13) — arts_edt_satisfy_slot / arts_event_add_dependence direct call
 *              with a DB_MODE_PTR inline payload.
 *
 * The entity-specific satisfy/add-dependence APIs are normally reached only via
 * arts_add_dependence or the wire dispatcher.  This test calls them DIRECTLY:
 *
 *   - arts_edt_satisfy_slot(target, slot, NULL_GUID, DB_MODE_PTR, ptr, size):
 *     the mode-discriminated apply core mallocs+copies `size` bytes from `ptr`
 *     into depv[slot].ptr (a copied buffer slice, not a GUID).  The consumer
 *     EDT must observe the exact bytes.
 *   - arts_event_add_dependence(evt, target, slot, DB_MODE_RO): wiring an event
 *     source directly; satisfying the event must deliver to the target slot.
 *
 * DB_MODE_PTR is a runtime-internal mode (arts/db.h), so this test pulls in the
 * internal header to name it — it directly exercises the internal apply path.
 *
 * runtime_single, all configs.
 */
#include "arts.h"
#include "arts/db.h" /* DB_MODE_PTR (internal mode) */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_failed = 0;

#define PTR_ELEMS 8u

/* Consumer:
 *   depv[0] = DB_MODE_PTR copied payload (PTR_ELEMS uint64s)
 *   depv[1] = event-delivered RO DB
 * paramv[0] = counter DB guid (RW via depv[2]) to record pass/fail
 */
void consumer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;

  bool ok = true;

  /* (1) DB_MODE_PTR inline payload was copied into depv[0].ptr. */
  uint64_t *p = (uint64_t *)depv[0].ptr;
  if (p == NULL) {
    arts_printf("FAIL edt_satisfy_slot_direct: DB_MODE_PTR slot ptr NULL\n");
    ok = false;
  } else {
    for (unsigned int i = 0; i < PTR_ELEMS; i++) {
      if (p[i] != (uint64_t)(0x100 + i)) {
        arts_printf("FAIL edt_satisfy_slot_direct: PTR payload[%u]=%llu "
                    "expected %llu\n",
                    i, (unsigned long long)p[i],
                    (unsigned long long)(0x100 + i));
        ok = false;
      }
    }
  }

  /* (2) event_add_dependence delivered the RO DB to depv[1]. */
  uint64_t *d = (uint64_t *)depv[1].ptr;
  if (d == NULL || d[0] != 0xBEEF) {
    arts_printf("FAIL edt_satisfy_slot_direct: event-delivered RO slot wrong "
                "(ptr=%p)\n",
                (void *)d);
    ok = false;
  }

  if (ok) {
    arts_printf("PASS edt_satisfy_slot_direct: DB_MODE_PTR payload + direct "
                "event_add_dependence delivery verified\n");
  } else {
    g_failed = 1;
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_satisfy_slot_direct ===\n");

  /* RO DB delivered through an event. */
  uint64_t *dptr = NULL;
  arts_guid_t ro_db = arts_db_create((void **)&dptr, sizeof(uint64_t) * 4,
                                     ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  dptr[0] = 0xBEEF;
  arts_db_release(ro_db, DB_MODE_RW);

  arts_guid_t evt = arts_event_create(NULL);

  /* Consumer with 2 deps. */
  arts_edt_hint_t h = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t c = arts_edt_create(consumer, 0, NULL, 2, &h);

  /* (B) Direct arts_event_add_dependence: event source -> consumer slot 1 RO.
   */
  arts_event_add_dependence(evt, c, 1, DB_MODE_RO);

  /* (A) Direct arts_edt_satisfy_slot with a DB_MODE_PTR inline payload -> slot
   * 0.  The runtime mallocs + copies `size` bytes; our local buffer can go out
   * of scope afterwards. */
  uint64_t payload[PTR_ELEMS];
  for (unsigned int i = 0; i < PTR_ELEMS; i++) {
    payload[i] = (uint64_t)(0x100 + i);
  }
  arts_edt_satisfy_slot(c, 0, NULL_GUID, DB_MODE_PTR, payload,
                        (unsigned int)sizeof(payload));

  /* Fire the event to satisfy the RO dep (delivers ro_db to slot 1). */
  arts_event_satisfy_slot(evt, ro_db, 0);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
