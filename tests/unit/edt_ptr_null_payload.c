/* SPDX-License-Identifier: Apache-2.0
 *
 * T132 — DB_MODE_PTR satisfy with ptr==NULL && size>0 (exposes B076).
 *
 * Bug under test (B076, edt_defer_satisfy / edt_apply_satisfy)
 * -----------------------------------------------------------
 * For DB_MODE_PTR the satisfy path treats `size > 0` as "an inline payload of
 * `size` bytes follows".  Both edt_defer_satisfy and edt_apply_satisfy keep
 * `size`/`payload` nonzero even when the caller passed `ptr == NULL`, while
 * skipping the actual memcpy of the source bytes:
 *
 *   - edt_defer_satisfy: allocates args + `size` trailing bytes but copies
 *     nothing (the `payload > 0 && ptr != NULL` guard fails), so the trailing
 *     region is uninitialized; a->size stays == size.
 *   - the handler then sets `ptr = (void *)(a + 1)` (non-NULL, uninitialized)
 *     because `a->size > 0`, and edt_apply_satisfy malloc+memcpy copies `size`
 *     UNINITIALIZED bytes into the dep's freshly malloc'd buffer.
 *
 * The EDT therefore receives a non-NULL depv[slot].ptr pointing at garbage of
 * the requested size.  Under ASan/MSan this surfaces as an uninitialized read /
 * use-of-uninitialized-value; functionally the delivered bytes are undefined.
 *
 * Scenario
 * --------
 * Satisfy slot 0 of a 1-dep EDT with DB_MODE_PTR, ptr == NULL, size == 64.
 * The EDT body checks the contract that a sound runtime should honor for a
 * NULL ptr with size>0: either the slot ptr is NULL (no phantom payload) OR the
 * delivered bytes are well-defined.  The current runtime delivers a non-NULL
 * pointer to uninitialized memory, which the body flags as FAIL (and the
 * sanitizer build additionally trips on the read).  This is the bug surface.
 *
 * Config-agnostic single-rank EDT lifecycle.
 */

#include "arts.h"
#include "arts/db.h" /* DB_MODE_PTR (runtime-internal access mode) */

#include <stdint.h>
#include <stdio.h>

#define PAYLOAD_SIZE 64u

/* depv[0] = the DB_MODE_PTR slot satisfied with ptr==NULL, size>0. */
void consumer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  void *p = depv[0].ptr;
  if (p == NULL) {
    /* Sound behavior: a NULL source surfaces as a NULL slot pointer. */
    arts_printf("PASS edt_ptr_null_payload: NULL ptr surfaced as NULL slot\n");
    arts_shutdown();
    return;
  }
  /* B076: a non-NULL pointer to `size` uninitialized bytes was delivered.
   * Reading them is the uninitialized read the sanitizer catches; we touch the
   * bytes so the read is materialized, then report the contract violation. */
  volatile unsigned char acc = 0;
  const unsigned char *b = (const unsigned char *)p;
  for (unsigned i = 0; i < PAYLOAD_SIZE; i++) {
    acc = (unsigned char)(acc ^ b[i]);
  }
  (void)acc;
  arts_printf("FAIL edt_ptr_null_payload: DB_MODE_PTR ptr==NULL size>0 "
              "delivered a non-NULL pointer to uninitialized memory (B076)\n");
  arts_abort(1);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_ptr_null_payload ===\n");

  arts_guid_t e =
      arts_edt_create(consumer, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0});
  /* The misuse: DB_MODE_PTR with ptr == NULL but size == PAYLOAD_SIZE > 0. */
  arts_edt_satisfy_slot(e, 0, NULL_GUID, DB_MODE_PTR, NULL, PAYLOAD_SIZE);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
