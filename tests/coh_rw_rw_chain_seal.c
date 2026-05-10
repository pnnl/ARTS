/*
 * coh_rw_rw_chain_seal.c — Stress test for back-to-back RW writers.
 *
 * Pattern: writer(0) reads init → writes 0; writer(1) reads 0 → writes 1; ...
 * Each writer must observe EXACTLY the previous writer's value. This catches
 * any sealing bug where an RW is allowed to run before the previous RW has
 * released.
 */

#include "arts.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <unistd.h>

#define NUM_ITERS 50

static atomic_int g_writer_count = 0;
static atomic_int g_writer_fail = 0;
static atomic_int g_final_check = 0;

static void *wd_thread(void *a) {
  (void)a;
  int last = -1, stuck = 0;
  for (;;) {
    sleep(2);
    int w = atomic_load(&g_writer_count);
    int fc = atomic_load(&g_final_check);
    if (fc) {
      return NULL;
    }
    if (w == last) {
      if (++stuck >= 3) {
        fprintf(stderr, "HANG: writers=%d (expected %d)\n", w, NUM_ITERS);
        fflush(stderr);
        _exit(1);
      }
    } else {
      stuck = 0;
    }
    last = w;
    if (w == NUM_ITERS) {
      return NULL;
    }
  }
}

void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int iter = (int)paramv[0];
  int expected_prev = (iter == 0) ? -1 : iter - 1;
  int *data = (int *)depv[0].ptr;
  if (!data || data[0] != expected_prev) {
    atomic_fetch_add(&g_writer_fail, 1);
    fprintf(stderr, "writer %d: expected prev=%d got %d\n", iter,
            expected_prev, data ? data[0] : -999);
    fflush(stderr);
  }
  if (data) {
    data[0] = iter;
  }
  atomic_fetch_add(&g_writer_count, 1);
}

void final_check_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                     arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data && data[0] == NUM_ITERS - 1) {
    atomic_store(&g_final_check, 1);
  } else {
    atomic_store(&g_final_check, -1);
    fprintf(stderr, "final_check: expected %d got %d\n", NUM_ITERS - 1,
            data ? data[0] : -999);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coh_rw_rw_chain_seal ===\n");
  arts_guid_t epoch = arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
  arts_epoch_start(epoch);

  void *ptr = NULL;
  arts_guid_t db = arts_db_create(&ptr, sizeof(int), ARTS_DB, ARTS_DB_PROP_NONE, NULL);
  ((int *)ptr)[0] = -1;
  arts_db_release(db);

  for (int i = 0; i < NUM_ITERS; i++) {
    uint64_t p = (uint64_t)i;
    arts_guid_t w = arts_edt_create(writer_edt, 1, &p, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
    arts_add_dependence(db, w, 0, DB_MODE_RW);
  }
  arts_guid_t fc = arts_edt_create(final_check_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence(db, fc, 0, DB_MODE_RO);

  {
    pthread_t wdt;
    pthread_create(&wdt, NULL, wd_thread, NULL);
    pthread_detach(wdt);
  }
  arts_epoch_wait(epoch);

  int w = atomic_load(&g_writer_count);
  int f = atomic_load(&g_writer_fail);
  int fc_res = atomic_load(&g_final_check);
  fprintf(stderr, "writers=%d fail=%d final_check=%d (expect %d/0/1)\n", w, f,
          fc_res, NUM_ITERS);
  if (w == NUM_ITERS && f == 0 && fc_res == 1) {
    fprintf(stderr, "TEST PASS\n");
  } else {
    fprintf(stderr, "TEST FAIL\n");
  }
  fflush(stderr);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
