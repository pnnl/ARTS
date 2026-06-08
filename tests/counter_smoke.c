/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/

#include "arts.h"
#include <stdio.h>
#include <sys/stat.h>

#define NUM_TASKS 10

void worker_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
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

  arts_guid_t fe =
      arts_event_create(&ARTS_EVENT_HINT_FINISH);

  for (unsigned int i = 0; i < NUM_TASKS; i++) {
    arts_edt_create(worker_edt, 0, NULL, 0,
                    &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
  }

  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);

  struct stat st;
  /* Check for per-node counter file (default config level: NODE).
     Per-thread files (n0_t0.json) require ARTS_COUNTER_LEVEL_THREAD,
     which is set via counters.cfg. For now, verify per-node counters. */
  if (stat("./counters/n0.json", &st) == 0 && st.st_size > 0) {
    printf("COUNTER_SMOKE_PASS\n");
  } else {
    printf("COUNTER_SMOKE_FAIL: ./counters/n0.json not found or empty\n");
  }
  return 0;
}
