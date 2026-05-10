/* SPDX-License-Identifier: Apache-2.0
 *
 * N-rank ring; each iteration rotates DB through 0->1->2->...->0. */

#include <arts.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_iters = 100;

void hop_edt(uint32_t paramc, const uint64_t *paramv,
             uint32_t depc, arts_edt_dep_t *depv) {
  (void)paramc; (void)depc;
  uint64_t iter = paramv[0];
  uint64_t hop  = paramv[1];
  uint8_t *p = (uint8_t *)depv[0].ptr;
  if (p) { p[0] = (uint8_t)(p[0] + 1); }
  unsigned int n = arts_get_total_ranks();
  if (hop + 1 < n) {
    arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
    hint.rank = (unsigned int)((hop + 1) % n);
    uint64_t next_param[2] = {iter, hop + 1};
    arts_guid_t next = arts_edt_create(hop_edt, 2, next_param, 1, &hint);
    arts_add_dependence(depv[0].guid, next, 0, DB_MODE_RW);
  } else {
    if (iter + 1 == (uint64_t)g_iters) {
      printf("RING_CHAIN: iter=%lu byte0=%u\n",
             (unsigned long)iter, p ? p[0] : 0u);
      arts_shutdown();
    }
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv,
              uint32_t depc, arts_edt_dep_t *depv) {
  (void)paramc; (void)paramv; (void)depc; (void)depv;
  for (int i = 0; i < g_iters; i++) {
    void *addr;
    arts_guid_t db = arts_db_create(&addr, 4096, ARTS_DB,
                                    ARTS_DB_PROP_NONE, NULL);
    memset(addr, 0, 4096);
    uint64_t param[2] = {(uint64_t)i, 0};
    arts_edt_hint_t hint = ARTS_EDT_HINT_DEFAULTS;
    hint.rank = 0;
    arts_guid_t edt = arts_edt_create(hop_edt, 2, param, 1, &hint);
    arts_add_dependence(db, edt, 0, DB_MODE_RW);
    arts_db_release(db);
  }
}

int main(int argc, char **argv) {
  if (argc > 1) { g_iters = (int)strtol(argv[1], NULL, 10); }
  arts_rt(argc, argv);
  return 0;
}
