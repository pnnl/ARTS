/*
 * coh_rw_ro_rw.c — Stress test for RW → RO → RW coherence transitions.
 *
 * Pattern per iteration: writer(RW) → N readers(RO) → writer(RW) → ...
 * This exercises the "sealed RO generation" progression path.
 */

#include "arts.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <unistd.h>

#define NUM_ITERS 100
#define NUM_READERS 10

static atomic_int g_writer_count = 0;
static atomic_int g_reader_count = 0;
static atomic_int g_reader_fail = 0;

static void *wd_thread(void *a) {
  (void)a;
  int last_w = -1, last_r = -1, stuck = 0;
  for (;;) {
    sleep(2);
    int w = atomic_load(&g_writer_count);
    int r = atomic_load(&g_reader_count);
    if (w == last_w && r == last_r) {
      if (++stuck >= 2) {
        fprintf(stderr, "HANG: w=%d r=%d (expect %d/%d)\n", w, r, NUM_ITERS,
                NUM_ITERS * NUM_READERS);
        fflush(stderr);
        _exit(1);
      }
    } else {
      stuck = 0;
    }
    last_w = w;
    last_r = r;
    if (w == NUM_ITERS && r == NUM_ITERS * NUM_READERS) {
      return NULL;
    }
  }
}

void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int iter = (int)paramv[0];
  int *data = (int *)depv[0].ptr;
  if (data) {
    data[0] = iter;
  }
  atomic_fetch_add(&g_writer_count, 1);
}

void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)depc;
  int expected = (int)paramv[0];
  int *data = (int *)depv[0].ptr;
  if (data && data[0] == expected) {
    atomic_fetch_add(&g_reader_count, 1);
  } else {
    atomic_fetch_add(&g_reader_fail, 1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  arts_printf("=== coh_rw_ro_rw (stress) ===\n");
  arts_guid_t epoch = arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
  arts_epoch_start(epoch);

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(int), ARTS_DB_RC, ARTS_DB_PROP_NONE, NULL);
  ((int *)ptr)[0] = -1;
  arts_db_release(db);

  for (int i = 0; i < NUM_ITERS; i++) {
    uint64_t p = (uint64_t)i;
    arts_guid_t w = arts_edt_create(writer_edt, 1, &p, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
    arts_add_dependence(db, w, 0, DB_MODE_RW);

    for (int r = 0; r < NUM_READERS; r++) {
      arts_guid_t rd = arts_edt_create(reader_edt, 1, &p, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
      arts_add_dependence(db, rd, 0, DB_MODE_RO);
    }
  }

  fprintf(stderr, "ALL_DEPS_ISSUED, waiting\n");
  fflush(stderr);
  {
    pthread_t wdt;
    pthread_create(&wdt, NULL, wd_thread, NULL);
    pthread_detach(wdt);
  }
  arts_epoch_wait(epoch);
  int w = atomic_load(&g_writer_count);
  int r = atomic_load(&g_reader_count);
  int f = atomic_load(&g_reader_fail);
  fprintf(stderr, "writers=%d readers=%d fail=%d (expect %d/%d)\n", w, r, f,
          NUM_ITERS, NUM_ITERS * NUM_READERS);
  if (w != NUM_ITERS || r != NUM_ITERS * NUM_READERS || f != 0) {
    fprintf(stderr, "TEST FAIL\n");
  } else {
    fprintf(stderr, "TEST PASS\n");
  }
  fflush(stderr);
  arts_shutdown();
}

int main(int argc, char **argv) {
  arts_rt(argc, argv);
  return 0;
}
