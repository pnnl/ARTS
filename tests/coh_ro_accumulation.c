/*
 * coh_ro_accumulation.c — Pure RO accumulation stress test.
 *
 * Pattern: one RW sets value V, then NUM_READERS RO readers verify V.
 * No writers between readers. Catches bugs where RO readers fail to join
 * an OPEN RO generation or where RO→RO transitions drop readers.
 */

#include "arts.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <unistd.h>

#define NUM_READERS 50
#define MAGIC_VALUE 0xDEADBEEF

static atomic_int g_writer_done = 0;
static atomic_int g_reader_ok = 0;
static atomic_int g_reader_fail = 0;

static void *wd_thread(void *a) {
  (void)a;
  int last_ok = -1, stuck = 0;
  for (;;) {
    sleep(2);
    int ok = atomic_load(&g_reader_ok);
    int fail = atomic_load(&g_reader_fail);
    if (ok + fail == NUM_READERS) {
      return NULL;
    }
    if (ok == last_ok) {
      if (++stuck >= 3) {
        fprintf(stderr, "HANG: ok=%d fail=%d (expect %d total)\n", ok, fail,
                NUM_READERS);
        fflush(stderr);
        _exit(1);
      }
    } else {
      stuck = 0;
    }
    last_ok = ok;
  }
}

void writer_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  if (data) {
    data[0] = MAGIC_VALUE;
  }
  atomic_store(&g_writer_done, 1);
}

void reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  unsigned int *data = (unsigned int *)depv[0].ptr;
  if (data && data[0] == MAGIC_VALUE) {
    atomic_fetch_add(&g_reader_ok, 1);
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

  arts_printf("=== coh_ro_accumulation ===\n");
  arts_guid_t epoch = arts_initialize_and_start_epoch(NULL_GUID, 0);

  void *ptr = NULL;
  arts_guid_t db =
      arts_db_create(&ptr, sizeof(unsigned int), ARTS_DB_DIST, ARTS_DB_PROP_NONE, NULL);
  ((unsigned int *)ptr)[0] = 0;
  arts_db_release(db);

  arts_guid_t w = arts_edt_create_with_epoch(writer_edt, 0, NULL, 1, epoch,
                                             &(arts_hint_t){.route = 0});
  arts_add_dependence(db, w, 0, DB_MODE_RW);

  for (int r = 0; r < NUM_READERS; r++) {
    arts_guid_t rd = arts_edt_create_with_epoch(reader_edt, 0, NULL, 1, epoch,
                                                &(arts_hint_t){.route = 0});
    arts_add_dependence(db, rd, 0, DB_MODE_RO);
  }

  {
    pthread_t wdt;
    pthread_create(&wdt, NULL, wd_thread, NULL);
    pthread_detach(wdt);
  }
  arts_wait_on_handle(epoch);

  int wdone = atomic_load(&g_writer_done);
  int ok = atomic_load(&g_reader_ok);
  int fail = atomic_load(&g_reader_fail);
  fprintf(stderr, "writer_done=%d readers_ok=%d readers_fail=%d (expect 1/%d/0)\n",
          wdone, ok, fail, NUM_READERS);
  if (wdone == 1 && ok == NUM_READERS && fail == 0) {
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
