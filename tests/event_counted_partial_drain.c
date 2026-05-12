/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/// @file event_counted_partial_drain.c
/// @brief Verifies COUNTED(N) destroy timing: event survives a fire with
///        fewer than N waiters registered, and destroys only after the
///        Nth waiter delivers via the late-binder fast path.

#include "arts.h"
#include "arts/gas/route_table.h"

#include <stdio.h>
#include <unistd.h>

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

  arts_printf("=== event_counted_partial_drain ===\n");

  arts_guid_t epoch = arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
  arts_epoch_start(epoch);

  /* COUNTED(5): latch=1, life_count=5, error_on_neg_latch=false. */
  arts_event_hint_t hint = ARTS_EVENT_HINT_COUNTED(5);
  arts_guid_t evt = arts_event_create(&hint);

  /* Register 3 of 5 waiters BEFORE satisfy. */
  for (int i = 0; i < 3; i++) {
    arts_guid_t e = arts_edt_create(
        noop_dep, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
    arts_add_dependence(evt, e, 0, DB_MODE_RW);
  }

  /* Single satisfy fires the event. */
  arts_event_satisfy_slot(evt, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  usleep(20000);

  /* Event must still be alive: only 3 of 5 deps consumed → life_count > 0. */
  struct arts_event_s *e = arts_route_table_lookup_event_safe(evt);
  if (!e) {
    arts_printf("FAIL: COUNTED destroyed too early (3/5 delivered)\n");
    arts_shutdown();
    return;
  }
  arts_route_table_release(evt);

  /* Register the remaining 2 as late binders. */
  for (int i = 0; i < 2; i++) {
    arts_guid_t edt = arts_edt_create(
        noop_dep, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
    arts_add_dependence(evt, edt, 0, DB_MODE_RW);
  }
  usleep(20000);

  /* Now life_count == 0 and latch <= 0 → destroyed. */
  e = arts_route_table_lookup_event_safe(evt);
  if (e) {
    arts_printf("FAIL: COUNTED still alive after all 5 deliveries\n");
    arts_route_table_release(evt);
    arts_shutdown();
    return;
  }

  arts_printf("  PASS: COUNTED destroyed exactly at 5th delivery\n");
  arts_epoch_wait(epoch);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
