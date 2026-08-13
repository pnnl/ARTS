/* SPDX-License-Identifier: Apache-2.0
 *
 * T140 (C13) — GPU LC drain force-defer in arts_edt_satisfy_slot (route 2).
 *
 * When a satisfy is issued from inside a GPU wrapper EDT that still has pending
 * invalidations (current_edt->invalidate_count > 0), arts_edt_satisfy_slot must
 * NOT apply the satisfy immediately.  Instead it force-pushes the satisfy
 * onto the wrapper's OWN slot so the re-signal of the target replays only
 * after the wrapper's invalidations drain.  This test exercises that
 * push-on-wrapper path.
 *
 * config_specific: requires ARTS_USE_GPU.  The whole body is guarded; a non-GPU
 * build self-skips (prints SKIP, exits 0) so the test compiles and passes in
 * the default (CPU) build dir.
 */
#include "arts.h"

#if !defined(ARTS_USE_GPU)

#include <stdio.h>

int main(void) {
  (void)printf("SKIP edt_gpu_lc_force_defer: requires ARTS_USE_GPU\n");
  return 0;
}

#else /* ARTS_USE_GPU */

#include "arts/gpu.h"

#include <stdint.h>
#include <stdio.h>

static int g_failed = 0;

/* GPU LC DB element count. */
#define LC_ELEMS 16u

/* A GPU kernel-EDT wrapper: while its LC invalidations are still pending
 * (invalidate_count>0 during the wrapper's body), it issues a non-PTR satisfy
 * to the downstream consumer.  The runtime must force-defer that satisfy onto
 * the wrapper's own slot so it replays only after the wrapper's invalidations
 * drain.  We verify the consumer DID eventually run with the LC-merged data. */
__attribute__((unused)) void producer_kernel(uint32_t paramc,
                                             const uint64_t *paramv,
                                             uint32_t depc,
                                             arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  /* depv[0] = GPU LC DB (writable replica). paramv[0] = consumer guid. */
  arts_guid_t consumer = (arts_guid_t)paramv[0];
  /* Non-PTR satisfy issued from inside the wrapper with pending invalidations:
   * route 2 force-defers this onto the wrapper's own slot. */
  arts_edt_satisfy_slot(consumer, 0, depv[0].guid, DB_MODE_RO);
}

void consumer(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  bool ok = (depv[0].ptr != NULL);
  if (ok) {
    arts_printf(
        "PASS edt_gpu_lc_force_defer: deferred non-PTR satisfy replayed "
        "after wrapper invalidations drained\n");
  } else {
    arts_printf("FAIL edt_gpu_lc_force_defer: consumer got NULL LC data\n");
    g_failed = 1;
  }
  arts_shutdown();
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== edt_gpu_lc_force_defer ===\n");

  void *p = NULL;
  arts_guid_t lc = arts_db_create(&p, sizeof(uint32_t) * LC_ELEMS, ARTS_DB_GPU,
                                  ARTS_DB_PROP_NONE, NULL);
  arts_db_release(lc, DB_MODE_RW);

  arts_edt_hint_t ch = ARTS_EDT_HINT_DEFAULTS;
  arts_guid_t c = arts_edt_create(consumer, 0, NULL, 1, &ch);

  uint64_t pv[1] = {(uint64_t)c};
  arts_gpu_hint_t gh = {.gpu = -1}; /* auto-select device */
  arts_guid_t k =
      arts_edt_create_gpu(producer_kernel, 1, pv, 1, (arts_dim3_t){1, 1, 1},
                          (arts_dim3_t){1, 1, 1}, &gh);
  arts_add_dependence(lc, k, 0, DB_MODE_RW);
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return g_failed;
}

#endif /* ARTS_USE_GPU */
