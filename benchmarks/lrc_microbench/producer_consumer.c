/* SPDX-License-Identifier: Apache-2.0
 *
 * LRC microbench: rank 0 writes a DB, rank 1 reads.  Time end-to-end.
 * Vary db_size via command line. */

#include <arts.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t g_db_size = 4096;
static int g_iters = 100;

void consumer_edt(uint32_t paramc, const uint64_t *paramv,
                  uint32_t depc, arts_edt_dep_t *depv) {
  (void)paramc; (void)depc;
  volatile uint8_t *p = (volatile uint8_t *)depv[0].ptr;
  uint64_t sum = 0;
  for (uint64_t i = 0; i < g_db_size; i++) { sum += p[i]; }
  if (paramv[0] == (uint64_t)(g_iters - 1)) {
    printf("CONSUMER: final sum=%lu\n", (unsigned long)sum);
    arts_shutdown();
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv,
              uint32_t depc, arts_edt_dep_t *depv) {
  (void)paramc; (void)paramv; (void)depc; (void)depv;
  if (arts_get_total_ranks() < 2) {
    printf("PRODUCER_CONSUMER: need at least 2 ranks\n");
    arts_shutdown();
    return;
  }
  for (int i = 0; i < g_iters; i++) {
    void *addr;
    arts_guid_t db = arts_db_create(&addr, g_db_size, ARTS_DB,
                                    ARTS_DB_PROP_NONE, NULL);
    memset(addr, (char)i, g_db_size);

    uint64_t param = (uint64_t)i;
    arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
    hint.rank = 1;
    arts_guid_t edt = arts_edt_create(consumer_edt, 1, &param, 1, &hint);
    arts_add_dependence(db, edt, 0, DB_MODE_RO);
    arts_db_release(db);
  }
}

int main(int argc, char **argv) {
  if (argc > 1) { g_db_size = (uint64_t)strtoll(argv[1], NULL, 10); }
  if (argc > 2) { g_iters = (int)strtol(argv[2], NULL, 10); }
  arts_rt(argc, argv);
  return 0;
}
