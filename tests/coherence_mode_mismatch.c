/* SPDX-License-Identifier: Apache-2.0
 *
 * Mode-mismatch detection: launch the two ranks from binaries built with
 * different memory-model/protocol configurations (e.g. one eager, one
 * lazy).  The first cross-mode message should produce a fatal error within
 * seconds, because the wire protocol tag differs between the two builds.
 *
 * Driven by tests/run_mode_mismatch.sh standalone. */

#include <arts.h>
#include <stdint.h>
#include <stdio.h>

void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t *depv) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  volatile uint64_t *p = (volatile uint64_t *)depv[0].ptr;
  if (p) {
    printf("READER: %lu\n", (unsigned long)*p);
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t *depv) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  void *addr;
  arts_guid_t db =
      arts_db_create(&addr, sizeof(uint64_t), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  *(uint64_t *)addr = 42;
  arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
  hint.rank = 1;
  arts_guid_t edt = arts_edt_create(reader_edt, 0, NULL, 1, &hint);
  arts_add_dependence(db, edt, 0, DB_MODE_RO);
  arts_db_release(db, DB_MODE_RW);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
