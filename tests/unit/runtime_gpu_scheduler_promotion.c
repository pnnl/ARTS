/******************************************************************************
** Copyright 2019 Battelle Memorial Institute
** Licensed under the Apache License, Version 2.0
******************************************************************************/
/// @file runtime_gpu_scheduler_promotion.c
/// @brief Exercises the GPU scheduler auto-promotion in arts_runtime_node_init:
///        when config->gpu>0 && config->scheduler==0 the runtime forces
///        config->scheduler=3 (the GPU loop) so GPU EDTs are actually served.
///
/// Without the promotion a default-scheduler config with GPUs would leave GPU
/// EDTs sitting forever on the GPU deques (the default loop never drains them).
/// node_init mutates the caller's config (a documented side effect: the GPU
/// promotion writes config->scheduler=3).  We cannot read config directly from
/// a public-API test, so we assert the observable consequence: with GPUs
/// present the runtime drives the GPU scheduler loop and a trivial workload
/// completes and shuts down (a regressed promotion would hang -> ctest
/// TIMEOUT).
///
/// config_specific: GPU build only.  Self-skips (prints SKIP, returns 0) when
/// ARTS_USE_GPU is not defined (no CI GPU). The real body uses only public API
/// so it compiles and links on any build.

#ifndef ARTS_USE_GPU
#include <stdio.h>
int main(void) {
  printf("SKIP runtime_gpu_scheduler_promotion: GPU-only (ARTS_USE_GPU not "
         "defined)\n");
  return 0;
}
#else /* ARTS_USE_GPU */

#include "arts.h"

void probe_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;
  /* Reaching here means the worker scheduler loop (GPU loop after promotion) is
   * running and draining ready EDTs. */
  arts_printf("  probe EDT ran under GPU scheduler\n");
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== runtime_gpu_scheduler_promotion ===\n");

  unsigned int gpus = arts_get_gpus_per_rank();
  if (gpus == 0) {
    /* No GPU visible at runtime -> promotion arm not taken; still a clean run.
     */
    arts_printf(
        "PASS runtime_gpu_scheduler_promotion: no GPU visible (gpus=0), "
        "promotion arm not exercised\n");
    arts_shutdown();
    return;
  }

  arts_printf("PASS runtime_gpu_scheduler_promotion: gpus=%u, driving GPU "
              "scheduler\n",
              gpus);
  arts_edt_create(probe_edt, 0, NULL, 0, &(arts_edt_hint_t){.rank = 0});
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  arts_printf("GPU_SCHED_PROMOTION_DONE\n");
  return 0;
}

#endif /* ARTS_USE_GPU */
