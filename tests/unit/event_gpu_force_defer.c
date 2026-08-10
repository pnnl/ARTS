/******************************************************************************
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License").           **
******************************************************************************/

/* event_gpu_force_defer — C15 / T163 (config_specific: GPU).
 *
 * Target: arts_event_satisfy_slot route 1 (event.c) — the invalidate_count
 * force-defer branch:
 *
 *     if (current_edt && current_edt->invalidate_count > 0)
 *         arts_ooo_push_guid(current_edt->guid, OOO_EVENT_SATISFY_SLOT, ...);
 *
 * This branch exists for GPU LC invalidation drain ordering: a GPU EDT wrapper
 * sets invalidate_count=1 for the duration of its body and clears it at
 * wrap-up, where arts_ooo_drain_guid replays the deferred satisfies.  The
 * branch is UNREACHABLE on a pure CPU EDT (invalidate_count is always 0 there),
 * so it can only be exercised in a GPU build.
 *
 * A `lib` GPU EDT runs its body on the CPU (host function with GPU stream
 * access) WHILE invalidate_count==1.  We make that lib body call
 * arts_event_satisfy_slot on a downstream latch: the satisfy MUST take route 1
 * (force-defer onto the wrapper EDT's GUID) and then replay at wrap-up.  A
 * successor EDT gated on the latch fires only if the deferred satisfy is
 * correctly replayed.
 *
 * config_specific: compiles to a SKIP stub unless ARTS_USE_GPU is defined (the
 * GPU symbols arts_edt_create_gpu / arts_dim3_t live only in a GPU libarts).
 *
 * PASS criterion (GPU build): the successor fires after the lib-EDT's
 * force-deferred satisfy replays.  Non-GPU: prints SKIP and exits 0.
 */

#include <stdio.h>

#include "arts.h"

#ifndef ARTS_USE_GPU

int main(void) {
  printf("SKIP event_gpu_force_defer: GPU-only (ARTS_USE_GPU not defined)\n");
  return 0;
}

#else /* ARTS_USE_GPU */

#include "arts/gpu.h"

/* lib GPU EDT body — runs on the CPU with invalidate_count==1.  Calling
 * arts_event_satisfy_slot here forces route 1 (force-defer); the deferred
 * satisfy replays when the wrapper clears invalidate_count at wrap-up.
 * paramv[0] = downstream latch guid. */
static void lib_satisfier(uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  (void)depv;
  arts_guid_t latch = (arts_guid_t)paramv[0];
  /* invalidate_count > 0 inside a GPU wrapper -> this takes the force-defer
   * route and is replayed at wrap-up. */
  arts_event_satisfy_slot(latch, NULL_GUID, ARTS_EVENT_LATCH_DECR_SLOT);
}

/* successor — fires only after the force-deferred satisfy replayed. */
static void successor(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  printf("event_gpu_force_defer: force-deferred satisfy replayed at GPU "
         "wrap-up, successor fired — PASS\n");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  /* Downstream latch fired by the lib EDT's deferred satisfy. */
  arts_event_hint_t lh = ARTS_EVENT_HINT_LATCH(1);
  lh.rank = arts_get_current_rank();
  arts_guid_t latch = arts_event_create(&lh);
  if (latch == NULL_GUID) {
    (void)fprintf(stderr, "FAIL: latch create returned NULL_GUID\n");
    arts_abort(1);
  }

  /* Successor gated on the latch. */
  arts_guid_t s = arts_edt_create(successor, 0, NULL, 1, NULL);
  arts_add_dependence(latch, s, 0, DB_MODE_NULL);

  /* lib GPU EDT (depc=0, runs immediately): its body satisfies the latch while
   * invalidate_count==1, exercising the force-defer route. */
  uint64_t pv[1] = {(uint64_t)latch};
  arts_dim3_t one = {1, 1, 1};
  arts_edt_create_gpu(lib_satisfier, 1, pv, 0, one, one,
                      &(arts_gpu_hint_t){.lib = true});
}

int main(int argc, char **argv) {
  /* Non-zero when a rank this process spawned ended badly: their exit status
     reaches nobody else, and a run with a dead rank did not succeed. */
  return arts_rt(argc, argv) != 0 ? 1 : 0;
}

#endif /* ARTS_USE_GPU */
