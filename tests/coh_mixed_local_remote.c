/*
 * coh_mixed_local_remote.c — Multi-node coherence with mixed local + remote
 * readers.
 *
 * Dynamically adapts to the node count (1..N):
 *   - DB is owned by node 0 (route=0).
 *   - RW writer w1 on node 0 writes VAL1.
 *   - READERS_PER_NODE RO readers on EACH node (including node 0).
 *   - RW writer w2 on node 0 writes VAL2.
 *   - Final RO reader on node 0 verifies VAL2.
 *
 * Correctness invariant: writer2 must observe VAL1 (checked on node 0).
 * If any reader on any node observed wrong data, or if writer2 ran
 * before all readers drained, the invariant fails.
 */

#include "arts.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <unistd.h>

#define READERS_PER_NODE 10
#define VAL1 1000
#define VAL2 2000

static atomic_int g_w1_done = 0;
static atomic_int g_w2_done = 0;
static atomic_int g_w2_order_bug = 0;
static atomic_int g_local_readers_ok = 0;
static atomic_int g_local_readers_bug = 0;
static atomic_int g_final_ok = 0;
static atomic_int g_epoch_done = 0;

static void *wd_thread(void *a) {
  (void)a;
  int last = 0, stuck = 0;
  for (;;) {
    sleep(2);
    if (atomic_load(&g_epoch_done)) {
      return NULL;
    }
    int sum = atomic_load(&g_w1_done) + atomic_load(&g_w2_done) +
              atomic_load(&g_local_readers_ok) + atomic_load(&g_final_ok);
    if (sum == last) {
      if (++stuck >= 3) {
        fprintf(stderr, "HANG: w1=%d w2=%d local_ok=%d final=%d\n",
                atomic_load(&g_w1_done), atomic_load(&g_w2_done),
                atomic_load(&g_local_readers_ok), atomic_load(&g_final_ok));
        fflush(stderr);
        _exit(1);
      }
    } else {
      stuck = 0;
    }
    last = sum;
  }
}

void writer1_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data) {
    data[0] = VAL1;
  }
  atomic_store(&g_w1_done, 1);
}

void writer2_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                 arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data) {
    if (data[0] != VAL1) {
      atomic_store(&g_w2_order_bug, data[0]);
    }
    data[0] = VAL2;
  }
  atomic_store(&g_w2_done, 1);
}

void v1_reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                   arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data && data[0] == VAL1) {
    atomic_fetch_add(&g_local_readers_ok, 1);
  } else {
    atomic_fetch_add(&g_local_readers_bug, 1);
  }
}

void final_reader_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
                      arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  int *data = (int *)depv[0].ptr;
  if (data && data[0] == VAL2) {
    atomic_store(&g_final_ok, 1);
  } else {
    atomic_store(&g_final_ok, -1);
  }
}

void main_edt(uint32_t paramc, const uint64_t *paramv, uint32_t depc,
              arts_edt_dep_t depv[]) {
  (void)paramc;
  (void)paramv;
  (void)depc;
  (void)depv;

  fprintf(stderr, "=== coh_mixed_local_remote ===\n");
  fflush(stderr);
  arts_guid_t epoch = arts_epoch_create(arts_get_current_rank(), NULL_GUID, 0);
  arts_epoch_start(epoch);

  void *ptr = NULL;
  arts_guid_t db = arts_db_create(&ptr, sizeof(int), ARTS_DB_RC,
                                  ARTS_DB_PROP_NONE, &(arts_db_hint_t){.rank = 0});
  ((int *)ptr)[0] = 0;
  arts_db_release(db);

  /* W1: RW on node 0 */
  arts_guid_t w1 = arts_edt_create(writer1_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence(db, w1, 0, DB_MODE_RW);

  /* Mixed RO generation: READERS_PER_NODE readers on EACH node. */
  unsigned int nnodes = arts_get_total_ranks();
  for (unsigned int n = 0; n < nnodes; n++) {
    for (int i = 0; i < READERS_PER_NODE; i++) {
      arts_guid_t r = arts_edt_create(v1_reader_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = n, .epoch = epoch});
      arts_add_dependence(db, r, 0, DB_MODE_RO);
    }
  }

  /* W2: RW on node 0 (must wait for all 2*READERS_PER_NODE readers to drain) */
  arts_guid_t w2 = arts_edt_create(writer2_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence(db, w2, 0, DB_MODE_RW);

  /* Final RO reader verifies VAL2 */
  arts_guid_t fr = arts_edt_create(final_reader_edt, 0, NULL, 1, &(arts_edt_hint_t){.rank = 0, .epoch = epoch});
  arts_add_dependence(db, fr, 0, DB_MODE_RO);

  {
    pthread_t wdt;
    pthread_create(&wdt, NULL, wd_thread, NULL);
    pthread_detach(wdt);
  }
  arts_epoch_wait(epoch);
  atomic_store(&g_epoch_done, 1);

  int w1d = atomic_load(&g_w1_done);
  int w2d = atomic_load(&g_w2_done);
  int w2bug = atomic_load(&g_w2_order_bug);
  int ok = atomic_load(&g_local_readers_ok);
  int bug = atomic_load(&g_local_readers_bug);
  int fr_res = atomic_load(&g_final_ok);
  fprintf(stderr,
          "w1=%d w2=%d w2_order_bug_val=%d local_ok=%d local_bug=%d final=%d "
          "(expect 1/1/0/%d/0/1)\n",
          w1d, w2d, w2bug, ok, bug, fr_res, READERS_PER_NODE);
  if (w1d == 1 && w2d == 1 && w2bug == 0 && ok == READERS_PER_NODE &&
      bug == 0 && fr_res == 1) {
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
