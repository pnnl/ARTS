/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_check_collision — C15 / T159.
 *
 * Target: event_install's install_if_absent path (event.c) — CHECK / rendezvous
 * create collision at the event level.
 *
 * With hint.check=true and a pre-reserved GUID, a create at an
 * already-occupied home-local slot must FAIL (return NULL_GUID) instead of
 * overwriting — OCR GUID_PROP_CHECK semantics: the first creator wins, a later
 * one observes the collision.  (db has labeled-GUID check tests; event-level
 * check-collision was uncovered.)
 *
 * Single-node, home-local, deterministic (both creates run on main_edt):
 *   1. Reserve one event GUID on the current rank.
 *   2. First create with check=true at that GUID -> must SUCCEED (returns the
 *      reserved GUID).
 *   3. Second create with check=true at the SAME GUID -> must FAIL
 *      (returns NULL_GUID); the loser is freed via the cb deleter.
 *   4. Sanity: a create with check=FALSE at the same GUID replaces (returns the
 *      GUID), confirming the failure in step 3 was specifically the check gate,
 *      not an unconditional reject.
 *
 * PASS criterion: step 2 returns the GUID, step 3 returns NULL_GUID, step 4
 * returns the GUID.
 */

#include <stdio.h>

#include "arts.h"

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

  /* First check-create at the reserved GUID: must win. */
  arts_event_hint_t h1 = ARTS_EVENT_HINT_LATCH(1);
  h1.guid = g;
  h1.check = true;
  arts_guid_t r1 = arts_event_create(&h1);
  if (r1 != g) {
    (void)fprintf(stderr,
                  "FAIL: first check-create returned %lu, want reserved %lu\n",
                  (uint64_t)r1, (uint64_t)g);
    arts_abort(1);
  }

  /* Second check-create at the SAME GUID: must lose -> NULL_GUID. */
  arts_event_hint_t h2 = ARTS_EVENT_HINT_LATCH(1);
  h2.guid = g;
  h2.check = true;
  arts_guid_t r2 = arts_event_create(&h2);
  if (r2 != NULL_GUID) {
    (void)fprintf(stderr,
                  "FAIL: colliding check-create returned %lu, want NULL_GUID\n",
                  (uint64_t)r2);
    arts_abort(1);
  }

  /* Sanity: a non-check create at the same GUID replaces (returns the GUID),
   * proving step 2's NULL was the check gate, not an unconditional reject. */
  arts_event_hint_t h3 = ARTS_EVENT_HINT_LATCH(1);
  h3.guid = g;
  h3.check = false;
  arts_guid_t r3 = arts_event_create(&h3);
  if (r3 != g) {
    (void)fprintf(stderr, "FAIL: non-check replace returned %lu, want %lu\n",
                  (uint64_t)r3, (uint64_t)g);
    arts_abort(1);
  }

  arts_event_destroy(g);
  printf(
      "event_check_collision: check-create wins once, colliding check-create "
      "-> NULL_GUID, non-check replaces — PASS\n");
  arts_shutdown();
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}
