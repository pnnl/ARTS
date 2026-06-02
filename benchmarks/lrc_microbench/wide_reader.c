/* SPDX-License-Identifier: Apache-2.0
 *
 * 1 writer, K readers in parallel; varies dedup pressure. */

#include <arts.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t g_db_size = 4096;
static int g_iters = 100;
static int g_k = 4;

void reader_edt(uint32_t paramc, const uint64_t *paramv,
                uint32_t depc, arts_edt_dep_t *depv) {
  (void)paramc; (void)depc;
  volatile uint8_t *p = (volatile uint8_t *)depv[0].ptr;
  uint64_t sum = 0;
  for (uint64_t i = 0; i < g_db_size; i++) { sum += p[i]; }
  if (paramv[0] == (uint64_t)(g_iters - 1) &&
      paramv[1] == (uint64_t)(g_k - 1)) {
    printf("WIDE_READER: final sum=%lu (k=%d)\n",
           (unsigned long)sum, g_k);
    arts_shutdown();
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv,
              uint32_t depc, arts_edt_dep_t *depv) {
  (void)paramc; (void)paramv; (void)depc; (void)depv;
  unsigned int n = arts_get_total_ranks();
  if ((unsigned int)g_k >= n) {
    printf("WIDE_READER: k=%d must be < %u (rank count)\n", g_k, n);
    arts_shutdown();
    return;
  }
  for (int i = 0; i < g_iters; i++) {
    void *addr;
    arts_guid_t db = arts_db_create(&addr, g_db_size, ARTS_DB,
                                    ARTS_DB_PROP_NONE, NULL);
    memset(addr, (char)i, g_db_size);
    for (int kk = 0; kk < g_k; kk++) {
      uint64_t param[2] = {(uint64_t)i, (uint64_t)kk};
      arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
      hint.rank = (unsigned int)(kk + 1);
      arts_guid_t edt = arts_edt_create(reader_edt, 2, param, 1, &hint);
      arts_add_dependence(db, edt, 0, DB_MODE_RO);
    }
    arts_db_release(db, DB_MODE_RW);
  }
}

int main(int argc, char **argv) {
  if (argc > 1) { g_k = (int)strtol(argv[1], NULL, 10); }
  if (argc > 2) { g_db_size = (uint64_t)strtoll(argv[2], NULL, 10); }
  if (argc > 3) { g_iters = (int)strtol(argv[3], NULL, 10); }
  arts_rt(argc, argv);
  return 0;
}
