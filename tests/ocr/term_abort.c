/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/* Termination-robustness test: trigger EDT on rank argv[1] prints
 * "TERM_ABORT rc=<code>" then calls arts_abort(<code>) where <code> is
 * argv[2]. Master-origin expected to exit with that code (regex on the
 * printed sentinel). Non-master origin expected to produce clean
 * TERM_TEST_EXIT from master after EOF detection. */
#include <stdio.h>
#include <stdlib.h>

#include "arts.h"

void trigger_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_printf("TERM_ABORT rc=%lu\n", paramv[0]);
  arts_abort((uint8_t)paramv[0]);
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  int argc = (int)paramv[0];
  char **argv = (char **)paramv[1];
  unsigned int origin =
      (argc > 1) ? (unsigned int)strtoul(argv[1], NULL, 10) : 0;
  if (origin >= arts_get_total_ranks()) {
    arts_printf("origin_rank %u >= total_nodes %u — cfg mismatch\n", origin,
                arts_get_total_ranks());
    arts_shutdown();
    return;
  }
  uint64_t rc = (argc > 2) ? strtoull(argv[2], NULL, 10) : 0;
  uint64_t args[1] = {rc};
  arts_edt_create(trigger_edt, 1, args, 0, &(arts_edt_hint_t){.rank = origin});
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  printf("[rank %u] TERM_TEST_EXIT\n", arts_get_current_rank());
  return 0;
}
