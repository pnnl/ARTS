/* SPDX-License-Identifier: Apache-2.0
 *
 * T293 — GPU context accessors inside a library EDT.
 *
 * Target: the inside-a-lib-EDT GPU context accessors in arts/gpu.h:
 *   arts_dim3_t *arts_get_gpu_grid(void);
 *   arts_dim3_t *arts_get_gpu_block(void);
 *   void        *arts_get_gpu_stream(void);   // cudaStream_t* as void*
 *   int          arts_get_gpu_id(void);       // == arts_get_current_gpu()
 *   int          arts_get_current_gpu(void);
 * grid/block/stream have NO test today; arts_get_gpu_id is only hit in gpu_lib.
 * A library EDT (arts_gpu_hint_t.lib = true) runs on the worker thread with the
 * GPU stream + dims established, so it can read them back.
 *
 * Correct behavior pinned:
 *   - inside the lib body, arts_get_gpu_grid()/_block() return non-NULL and the
 *     dims match exactly what arts_edt_create_gpu was given;
 *   - arts_get_gpu_stream() returns non-NULL (a live CUDA stream handle);
 *   - arts_get_gpu_id() == arts_get_current_gpu() and is a valid device index
 *     (>= 0) when a GPU is present.
 *   A successor gated on lib-EDT completion reports the verdict written into a
 *   counter DB by the lib body.
 *
 * Config-specific (GPU): the GPU library API only exists in a CUDA-enabled
 * build, so the real body is compiled only when ARTS_TEST_GPU is defined (the
 * GPU build defines it).  In any non-GPU build the file is a clean SKIP that
 * links against the ordinary (CPU) libarts.  Mirrors ctx_gpu_lib_edt.c.
 *
 * exposes_runtime_bug = false (pins correct lib-EDT GPU-context reads).
 */
#include "arts.h"

#if !defined(ARTS_TEST_GPU)

#include <stdio.h>
int main(void) {
  printf("SKIP api_gpu_context_accessors: GPU-only (build with CUDA + "
         "ARTS_TEST_GPU)\n");
  return 0;
}

#else /* ARTS_TEST_GPU — real body, requires a CUDA-enabled libarts */

#include "arts/gpu.h"
#include <stdint.h>

static int g_failed = 0;

/* Grid/block dims the lib EDT is created with; the lib body reads them back. */
#define GX 7u
#define GY 3u
#define GZ 2u
#define BX 32u
#define BY 4u
#define BZ 1u

/* Library EDT body: runs on the worker thread with GPU context established.
 * depv[0] = verdict counter DB (RW): 1 = all accessors correct, 0 = failure. */
void lib_body(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *verdict = (int *)depv[0].ptr;
  int ok = 1;

  arts_dim3_t *grid = arts_get_gpu_grid();
  arts_dim3_t *block = arts_get_gpu_block();
  void *stream = arts_get_gpu_stream();
  int gid = arts_get_gpu_id();
  int cur = arts_get_current_gpu();

  if (grid == NULL || grid->x != GX || grid->y != GY || grid->z != GZ) {
    arts_printf("FAIL api_gpu_context_accessors: grid mismatch\n");
    ok = 0;
  }
  if (block == NULL || block->x != BX || block->y != BY || block->z != BZ) {
    arts_printf("FAIL api_gpu_context_accessors: block mismatch\n");
    ok = 0;
  }
  if (stream == NULL) {
    arts_printf("FAIL api_gpu_context_accessors: NULL GPU stream\n");
    ok = 0;
  }
  if (gid != cur || gid < 0) {
    arts_printf("FAIL api_gpu_context_accessors: gpu id %d != current %d (or "
                "invalid)\n",
                gid, cur);
    ok = 0;
  }

  if (verdict) {
    *verdict = ok;
  }
}

/* Successor gated on lib-EDT completion.  depv[0] = verdict counter DB (RO),
 * slot 1 = completion signal filled by the lib EDT. */
void successor(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
               arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *verdict = (int *)depv[0].ptr;
  if (verdict == NULL || *verdict != 1) {
    arts_printf("FAIL api_gpu_context_accessors: lib body reported failure\n");
    g_failed = 1;
  } else {
    arts_printf("PASS api_gpu_context_accessors: grid/block/stream/id correct "
                "inside lib EDT\n");
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
    arts_printf("SKIP api_gpu_context_accessors: no GPU present\n");
    arts_shutdown();
    return;
  }

  int *v = NULL;
  arts_guid_t vdb = arts_db_create((void **)&v, sizeof(int), ARTS_DB,
                                   ARTS_DB_PROP_NONE, NULL);
  *v = 0;
  arts_db_release(vdb, DB_MODE_RW);

  /* successor: verdict RO (slot 0) + completion-signal slot (slot 1). */
  arts_edt_hint_t sh = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t s = arts_edt_create(successor, 0, NULL, 2, &sh);
  arts_add_dependence(vdb, s, 0, DB_MODE_RO);

  arts_gpu_hint_t gh = {0};
  gh.lib = true;   /* host-side library EDT: GPU context available */
  gh.end_guid = s; /* signal the successor on completion ... */
  gh.slot = 1;     /* ... filling its slot 1 */
  arts_guid_t lib =
      arts_edt_create_gpu(lib_body, 0, NULL, 1, (arts_dim3_t){GX, GY, GZ},
                          (arts_dim3_t){BX, BY, BZ}, &gh);
  arts_add_dependence(vdb, lib, 0, DB_MODE_RW);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}

#endif /* ARTS_TEST_GPU */
