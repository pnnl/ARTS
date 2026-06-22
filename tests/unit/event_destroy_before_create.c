/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_destroy_before_create — C15 / T161 (single-node + multinode).
 *
 * Target: arts_event_destroy -> OOO_EVENT_DESTROY defer (event.c).
 *
 * arts_event_destroy on a home-local GUID routes through
 * arts_ooo_dispatch_or_defer_guid(OOO_EVENT_DESTROY).  If the destroy races
 * AHEAD of the event's create (the slot is still absent), it must DEFER on the
 * slot and replay when the event installs: event_install (replace path) drains
 * the OoO list, running the deferred destroy body
 * (arts_route_table_set_destroyed, idempotent).  Net effect: a destroy that
 * arrived before the create still tears the event down once it appears.
 *
 * Deterministic single-run scenario (home-local on rank 0):
 *   1. Reserve an event GUID on the current rank.
 *   2. arts_event_destroy(g) BEFORE any create -> the destroy DEFERS (slot
 *      absent).
 *   3. arts_event_create at g (check=false replace) -> install drains the OoO
 *      list and replays the deferred destroy.
 *   4. Assert the event is NOT present (the deferred destroy applied on
 * install).
 *
 * The integrator registers this both single-node and multinode; the OoO defer
 * is a home-rank mechanism, so the body runs on rank 0 in both.
 *
 * PASS criterion: after destroy-then-create the GUID is detached.
 */

#include <stdio.h>

#include "arts.h"
#include "arts/gas/route_table.h"

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  unsigned int rank = arts_get_current_rank();
  arts_guid_t g = arts_guid_reserve(ARTS_GUID_EVENT, rank);
  if (g == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: guid_reserve returned NULL_GUID\n");
    arts_abort(1);
  }

  /* Destroy BEFORE create: the slot is absent -> the destroy defers on it. */
  arts_event_destroy(g);

  /* Now create the event at that GUID.  Install replays the deferred destroy.
   */
  arts_event_hint_t h = ARTS_EVENT_HINT_LATCH(1);
  h.guid = g;
  h.check = false; /* unconditional replace path drains the OoO list */
  arts_guid_t r = arts_event_create(&h);
  if (r != g) {
    (void)fprintf(stderr,
                  "FAIL: create at reserved GUID returned %lu want %lu\n",
                  (uint64_t)r, (uint64_t)g);
    arts_abort(1);
  }

  /* The deferred destroy must have replayed on install -> event detached. */
  arts_shared_ptr_t lh = arts_route_table_lookup_event(g);
  struct arts_event_s *e = (struct arts_event_s *)arts_shared_get(lh);
  bool present = (e != NULL);
  arts_shared_release(&lh);

  if (present) {
    (void)fprintf(stderr,
                  "FAIL: event present after destroy-before-create (deferred "
                  "destroy did not replay on install)\n");
    arts_event_destroy(g);
    arts_abort(1);
  }

  printf("event_destroy_before_create: deferred destroy replayed on install, "
         "event detached — PASS\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
