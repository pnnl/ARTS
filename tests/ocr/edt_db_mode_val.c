/* SPDX-License-Identifier: Apache-2.0
 *
 * T143 (C13) — a dependency with DB_MODE_VAL (raw uint64 value, not a GUID)
 *              delivered to depv[slot].
 *
 * DB_MODE_VAL carries a raw 64-bit value rather than a DataBlock GUID: the
 * value rides in the dependency's `source`/data_guid field and the apply core
 * writes it verbatim into depv[slot].guid (no pointer, no DB lookup).  This
 * test wires several DB_MODE_VAL deps with distinct raw values and verifies the
 * consumer reads each back exactly from depv[slot].guid.
 *
 * runtime_single, all configs.
 */
#include "arts.h"

#include <stdint.h>
#include <stdio.h>

static int g_failed = 0;

#define N_VALS 4u

/* Raw values delivered through DB_MODE_VAL deps. */
static const uint64_t kVals[N_VALS] = {
    0x0ULL,                /* zero value must survive */
    0x1ULL,                /* one */
    0xDEADBEEFCAFEBABEULL, /* full-width pattern */
    0xFFFFFFFFFFFFFFFFULL, /* all-ones */
};

/* depv[0..N_VALS-1] = DB_MODE_VAL raw values in .guid. */
void val_consumer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                  arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;

  bool ok = true;
  for (unsigned int i = 0; i < N_VALS; i++) {
    uint64_t got = (uint64_t)depv[i].guid;
    if (got != kVals[i]) {
      arts_printf("FAIL edt_db_mode_val: slot %u got 0x%llx expected 0x%llx\n",
                  i, (unsigned long long)got, (unsigned long long)kVals[i]);
      ok = false;
    }
    /* DB_MODE_VAL carries no payload pointer. */
    if (depv[i].ptr != NULL) {
      arts_printf("FAIL edt_db_mode_val: slot %u ptr non-NULL for VAL dep\n",
                  i);
      ok = false;
    }
  }

  if (ok) {
    arts_printf("PASS edt_db_mode_val: %u raw uint64 values delivered via "
                "DB_MODE_VAL\n",
                N_VALS);
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

  arts_printf("=== edt_db_mode_val ===\n");

  arts_edt_hint_t h = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t c = arts_edt_create(val_consumer, 0, NULL, N_VALS, &h);

  /* Each DB_MODE_VAL dep: the raw value is the `source` (carried in .guid). */
  for (unsigned int i = 0; i < N_VALS; i++) {
    arts_add_dependence((arts_guid_t)kVals[i], c, i, DB_MODE_VAL);
  }
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}
