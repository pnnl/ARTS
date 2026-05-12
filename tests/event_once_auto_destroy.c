/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/// @file event_once_auto_destroy.c
/// @brief Verifies that a ONCE event is destroyed by its terminal satisfy,
///        regardless of whether any waiters were registered at fire time.
///
/// New design invariant (latch+life_count): ONCE has life_count=0 by default,
/// so on fire (latch reaches <= 0) the destroy invariant
/// (latch<=0 && life_count<=0) is met and the event is mark_deleted.

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

  arts_printf("=== event_once_auto_destroy ===\n");

  /* Default hint = ONCE (latch=1, life_count=0, error_on_neg_latch=false). */
  arts_guid_t ev = arts_event_create(NULL);
  if (ev == NULL_GUID) {
    arts_printf("FAIL: arts_event_create returned NULL_GUID\n");
    arts_shutdown();
    return;
  }

  /* Satisfy with no waiters registered.  Drain delivers to empty stack,
   * then maybe_destroy(latch=0, life_count=0) -> mark_delete. */
  arts_event_satisfy_slot(ev, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);

  /* Brief settle window for the destroy to propagate through the
   * route_table refcount path. */
  usleep(50000);

  struct arts_event_s *e = arts_route_table_lookup_event_safe(ev);
  if (e != NULL) {
    arts_printf("FAIL: ONCE event still alive after satisfy (leak)\n");
    arts_route_table_release(ev);
    arts_shutdown();
    return;
  }

  arts_printf("  PASS: ONCE event destroyed on fire\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
