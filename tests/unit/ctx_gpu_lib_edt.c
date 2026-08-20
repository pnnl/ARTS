/* SPDX-License-Identifier: Apache-2.0
 *
 * T151 — lib-EDT nested context set/unset (GPU build only).
 *
 * Target: the GPU library-EDT run path (libs/src/core/gpu/gpu_stream.cu) which
 * brackets the lib function body with arts_set_thread_local_edt_info /
 * arts_unset_thread_local_edt_info and then calls arts_release_created_dbs on
 * the WORKER thread (not the CUDA callback thread).  This is the second caller
 * of the edt_context.c run-start/run-end hooks (the scheduler is the first).
 *
 * Config-specific: the GPU library API (arts/gpu.h, arts_edt_create_gpu with
 * hint.lib = true) only exists in a CUDA-enabled build, so the real body is
 * compiled only when ARTS_TEST_GPU is defined (the GPU build defines it).  In
 * any non-GPU build the file is a clean SKIP that links against the ordinary
 * (CPU) libarts.  Pattern mirrors the protocol-macro self-skip idiom.
 *
 * What the real body pins inside a lib EDT body (where current_edt is the lib
 * EDT and the worker's created_db_list starts empty — the previous EDT's
 * epilogue drained it):
 *   1. A DB created inside the lib body is tracked into THIS lib EDT's context
 *      (created_db_list length == 1 after one create — the epilogue left it
 *      empty, track appended exactly one).
 *   2. arts_current_finish_event()/current_edt are non-NULL inside the lib body
 *      (the lib EDT context is established by set before the body runs).
 *   3. The lib body writes a sentinel into a counter DB (RW dep); a successor
 *      gated on the lib EDT's completion (finish event) reads it back, proving
 *      the lib EDT ran to completion through the set/unset bracket.
 *
 * The nested DB the lib body creates exercises the worker-side
 * arts_release_created_dbs (gpu_stream.cu releases on the worker, not the CUDA
 * callback thread).  The lib body leaves it to the epilogue auto-release; the
 * test completing without a stranded acquisition is the observable signal.
 *
 * exposes_runtime_bug = false (pins correct lib-EDT nested context handling).
 */
#include "arts.h"

#if !defined(ARTS_TEST_GPU)

#include <stdio.h>
int main(void) {
  printf("SKIP ctx_gpu_lib_edt: GPU-only (build with CUDA + ARTS_TEST_GPU)\n");
  return 0;
}

#else /* ARTS_TEST_GPU — real body, requires a CUDA-enabled libarts */

#include "arts/edt_context.h" /* current_edt, arts_get_created_db_list */
#include "arts/gpu.h"
#include "arts/utils/vector.h"

#include <stdint.h>

static int g_failed = 0;

/* Library EDT body: runs on the worker thread with the GPU stream available,
 * bracketed by set/unset.  depv[0] = counter DB (RW). */
void lib_body(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *c = (int *)depv[0].ptr;

  /* (2) the lib EDT context is established by set before the body runs. */
  if (current_edt == NULL) {
    arts_printf("FAIL ctx_gpu_lib_edt: lib body has no current_edt context\n");
    g_failed = 1;
  }

  /* (1) a DB created here is tracked into the lib EDT's context: the previous
   * EDT's epilogue drained the worker's created_db_list, so exactly this one
   * entry. */
  void *np = NULL;
  arts_guid_t ndb =
      arts_db_create(&np, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  if (np) {
    ((int *)np)[0] = 0xBEEF;
  }
  arts_vector_t *list = arts_get_created_db_list();
  if (arts_vector_count(list) != 1) {
    arts_printf("FAIL ctx_gpu_lib_edt: lib-body created_db_list length != 1 "
                "(epilogue did not drain / track did not append)\n");
    g_failed = 1;
  }
  /* Leave ndb to the worker-side epilogue arts_release_created_dbs (do NOT
   * explicitly release): exercises the worker-thread release path. */

  /* (3) write sentinel into the counter DB. */
  if (c) {
    c[0] = 1;
  }
}

/* successor: gated on the lib EDT finish event; depv[0] = counter DB (RO). */
void successor(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *c = (int *)depv[0].ptr;
  if (c == NULL || c[0] != 1) {
    arts_printf("FAIL ctx_gpu_lib_edt: counter not written by lib body\n");
    g_failed = 1;
  }
  if (!g_failed) {
    arts_printf("PASS ctx_gpu_lib_edt: lib-EDT nested set/unset + tracking + "
                "worker-side release OK\n");
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  if (arts_get_num_gpus() == 0) {
    arts_printf("SKIP ctx_gpu_lib_edt: no GPU present\n");
    arts_shutdown();
    return;
  }

  int *c = NULL;
  arts_guid_t cdb = arts_db_create((void **)&c, sizeof(int), ARTS_DB,
                                   ARTS_DB_PROP_NONE, NULL);
  c[0] = 0;
  arts_db_release(cdb, DB_MODE_RW);

  /* successor: counter RO (slot 0) + a completion-signal slot (slot 1) the lib
   * EDT fills on completion via gh.end_guid/slot, gating it on the lib EDT
   * finishing the set/unset bracket. */
  arts_edt_hint_t sh = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t s = arts_edt_create(successor, 0, NULL, 2, &sh);
  arts_add_dependence(cdb, s, 0, DB_MODE_RO);

  arts_gpu_hint_t gh = {0};
  gh.lib = true;
  gh.end_guid = s; /* signal the successor on lib-EDT completion ... */
  gh.slot = 1;     /* ... filling its slot 1 */
  arts_guid_t lib =
      arts_edt_create_gpu(lib_body, 0, NULL, 1, (arts_dim3_t){1, 1, 1},
                          (arts_dim3_t){1, 1, 1}, &gh);
  arts_add_dependence(cdb, lib, 0, DB_MODE_RW);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}

#endif /* ARTS_TEST_GPU */
