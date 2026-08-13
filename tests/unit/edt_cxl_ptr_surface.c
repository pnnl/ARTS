/* SPDX-License-Identifier: Apache-2.0
 *
 * T144 (C13) — CXL ptr-surfacing in arts_edt_satisfy_slot (route 1).
 *
 * For a CXL DB the GUID encodes the shared-segment pointer directly (no route
 * table entry).  arts_edt_satisfy_slot route 1 detects a CXL GUID being
 * delivered with ptr==NULL and a non-PTR/non-VAL mode, and surfaces the
 * (arts_db_s*)+1 payload pointer onto the dep slot so the consumer's
 * prep/release flush helpers see the right pointer.  This test creates a CXL
 * DB, fills it, then satisfies a consumer EDT slot with the CXL GUID (ptr=NULL)
 * and verifies the consumer receives a non-NULL, correctly-surfaced pointer.
 *
 * config_specific: requires ARTS_USE_CXL.  Self-skips (prints SKIP, exits 0)
 * when ARTS_USE_CXL is not defined.
 */
#include "arts.h"

#if !defined(ARTS_USE_CXL)

#include <stdio.h>

int main(void) {
  (void)printf("SKIP edt_cxl_ptr_surface: requires ARTS_USE_CXL\n");
  return 0;
}

#else /* ARTS_USE_CXL */

#include <stdint.h>
#include <stdio.h>

static int g_failed = 0;

#define ELEMS 16u

/* depv[0] = CXL DB surfaced via route 1 (RO). */
void cxl_consumer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  uint64_t *d = (uint64_t *)depv[0].ptr;
  bool ok = (d != NULL);
  for (unsigned int i = 0; i < ELEMS && ok; i++) {
    if (d[i] != (uint64_t)(0xC0FFEEUL + i)) {
      ok = false;
    }
  }
  if (ok) {
    arts_printf("PASS edt_cxl_ptr_surface: CXL GUID surfaced to dep slot ptr "
                "and bytes visible\n");
  } else {
    arts_printf("FAIL edt_cxl_ptr_surface: surfaced ptr NULL or byte "
                "mismatch (ptr=%p)\n",
                (void *)d);
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

  arts_printf("=== edt_cxl_ptr_surface ===\n");

  void *ptr = NULL;
  arts_guid_t db = arts_db_create(&ptr, ELEMS * sizeof(uint64_t), ARTS_DB_CXL,
                                  ARTS_DB_PROP_NONE, NULL);
  if (ptr == NULL) {
    arts_printf("FAIL edt_cxl_ptr_surface: CXL create returned NULL ptr\n");
    g_failed = 1;
    arts_shutdown();
    return;
  }
  uint64_t *d = (uint64_t *)ptr;
  for (unsigned int i = 0; i < ELEMS; i++) {
    d[i] = (uint64_t)(0xC0FFEEUL + i);
  }

  arts_edt_hint_t ch = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t c = arts_edt_create(cxl_consumer, 0, NULL, 1, &ch);

  /* Satisfy with the CXL GUID and ptr==NULL: route 1 must surface the
   * shared-segment payload pointer onto the consumer's dep slot. */
  arts_edt_satisfy_slot(c, 0, db, DB_MODE_RO);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}

#endif /* ARTS_USE_CXL */
