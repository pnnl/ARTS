/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/// @file event_channel_lifetime.c
/// @brief Verifies that CHANNEL events do NOT auto-destroy regardless of
///        the number of fire generations.  Only explicit arts_event_destroy
///        tears the event down.

#include "arts.h"
#include "arts/gas/route_table.h"

#include <stdio.h>
#include <unistd.h>

#define N_GENS 100

static void noop_dep(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
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

  arts_printf("=== event_channel_lifetime ===\n");

  arts_guid_t fe = arts_event_create(&ARTS_EVENT_HINT_FINISH);

  arts_event_hint_t hint = ARTS_EVENT_HINT_CHANNEL;
  arts_guid_t ch = arts_event_create(&hint);

  /* Drive N fire generations: pair add_dep + satisfy each iteration. */
  for (int i = 0; i < N_GENS; i++) {
    arts_guid_t edt =
        arts_edt_create(noop_dep, 0, NULL, 1,
                        &(arts_edt_hint_t){.rank = 0, .finish_event = fe});
    arts_add_dependence(ch, edt, 0, DB_MODE_RW);
    arts_event_satisfy_slot(ch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  }

  usleep(50000);

  /* Event must still be alive after N fires. */
  arts_shared_ptr_t h = arts_route_table_lookup_event(ch);
  struct arts_event_s *e = (struct arts_event_s *)arts_shared_get(h);
  if (!e) {
    arts_printf(
        "FAIL: CHANNEL destroyed after %d fires (expected persistent)\n",
        N_GENS);
    arts_shared_release(&h);
    arts_shutdown();
    return;
  }
  arts_shared_release(&h);

  /* Explicit destroy releases it. */
  arts_event_destroy(ch);
  usleep(50000);

  h = arts_route_table_lookup_event(ch);
  e = (struct arts_event_s *)arts_shared_get(h);
  if (e) {
    arts_printf("FAIL: CHANNEL still alive after explicit destroy\n");
    arts_shared_release(&h);
    arts_shutdown();
    return;
  }
  arts_shared_release(&h);

  arts_printf(
      "  PASS: CHANNEL persistent across %d fires; destroyed explicitly\n",
      N_GENS);
  arts_event_wait(fe);
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
