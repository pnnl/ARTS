/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/// @file event_idem_silent_oversatisfy.c
/// @brief Verifies that IDEM events tolerate over-satisfy: subsequent
///        satisfies past the unique fire are silent no-ops (no error,
///        event remains alive).

#include "arts.h"
#include "arts/gas/route_table.h"

#include <stdio.h>
#include <unistd.h>

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== event_idem_silent_oversatisfy ===\n");

  /* IDEMPOTENT: latch=1, life_count=INT32_MAX, error_on_neg_latch=false. */
  arts_event_hint_t hint = ARTS_EVENT_HINT_IDEMPOTENT;
  arts_guid_t ev = arts_event_create(&hint);

  /* Fire it. */
  arts_event_satisfy_slot(ev, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  /* Four more over-satisfies must NOT raise ARTS_ERROR. */
  for (int i = 0; i < 4; i++) {
    arts_event_satisfy_slot(ev, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
  }
  usleep(20000);

  /* Event still alive (life_count = INT32_MAX - 0 since no add_dep ran). */
  struct arts_event_s *e = arts_route_table_lookup_event_safe(ev);
  if (!e) {
    arts_printf("FAIL: IDEM destroyed after over-satisfy\n");
    arts_shutdown();
    return;
  }
  arts_route_table_release(ev);
  arts_event_destroy(ev);

  arts_printf("  PASS: IDEM tolerated 5 satisfies, no error\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
