/* SPDX-License-Identifier: Apache-2.0
 *
 * route_table_install_replace — arts_route_table_install REPLACE semantics
 * over a LIVE object, with deferred-free correctness (census 18-gas GAP 1).
 *
 * arts_route_table_install is the UNCONDITIONAL install: it atomic_exchanges a
 * fresh cb into the slot whatever the slot held, and releases the displaced
 * cb.  The displaced object's deleter must run exactly once — but only after
 * the last outstanding reader handle on the OLD generation is released, never
 * a use-after-free.  This test pins that contract:
 *
 *   1. Install obj A.  A lookup handle is held (a reader ref on A's cb).
 *   2. Install obj B over the LIVE A.  A is displaced; because a reader handle
 *      is still outstanding, A's deleter has NOT run yet (deferred free).
 *   3. Release the stale reader handle → A's deleter runs exactly once.
 *   4. A fresh lookup now returns B; install does NOT bump gen (only destroy
 *      does), so was_destroyed stays false across the REPLACE.
 *   5. A concurrent storm of installers + lookups + releases on one slot:
 *      every lookup handle is either NULL or a live, readable object; no UAF
 *      / double-free (ASan); deleter invocation count == #displaced cbs.
 *
 * No runtime: route_table.c + guid.c + shared.c are #include'd with libc-backed
 * shims; arts_ooo_drain/free_all are faithful no-ops (no OoO payloads pushed).
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

unsigned int arts_global_rank_id = 0;
unsigned int arts_global_rank_count = 1;
void arts_abort(uint8_t code) { _exit(code ? code : 70); }
void *arts_malloc(size_t s) { return malloc(s); }
void *arts_calloc(size_t n, size_t s) { return calloc(n, s); }
void *arts_calloc_aligned(size_t n, size_t s, size_t a) {
  void *p = NULL;
  if (posix_memalign(&p, a < sizeof(void *) ? sizeof(void *) : a, n * s)) {
    return NULL;
  }
  memset(p, 0, n * s);
  return p;
}
void arts_free(void *p) { free(p); }

struct arts_route_item_s;
void arts_ooo_drain(struct arts_route_item_s *s) { (void)s; }
void arts_ooo_free_all(struct arts_route_item_s *s) { (void)s; }

/* Wait-free counter primitives for the DB seq allocator (libc-free unit
 * pattern: mirror the atomics.c definitions verbatim). */
uint64_t arts_atomic_fetch_add_u64(volatile uint64_t *d, uint64_t v) {
  return __sync_fetch_and_add(d, v);
}
uint64_t arts_atomic_cswap_u64(volatile uint64_t *d, uint64_t o, uint64_t n) {
  return __sync_val_compare_and_swap(d, o, n);
}
uint64_t arts_atomic_read_u64(const volatile uint64_t *d) {
  return __atomic_load_n(d, __ATOMIC_ACQUIRE);
}

#include "../../libs/src/core/gas/guid.c"
#include "../../libs/src/core/gas/route_table.c"
#include "../../libs/src/core/utils/shared.c"

struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

/* deleter bookkeeping */
static _Atomic int g_deletes;
static void test_deleter(void *obj) {
  atomic_fetch_add_explicit(&g_deletes, 1, memory_order_relaxed);
  free(obj);
}

#define FAIL(...)                                                              \
  do {                                                                         \
    (void)fprintf(stderr, "FAIL route_table_install_replace: " __VA_ARGS__);   \
    return 1;                                                                  \
  } while (0)

static void rt_init(void) {
  arts_thread_info.thread_id = 0;
  arts_node_info.total_thread_count = 1;
  arts_node_info.gpu = 0;
  arts_node_info.keys = (uint64_t **)calloc(1, sizeof(uint64_t *));
  arts_node_info.keys[0] = (uint64_t *)calloc(
      (size_t)ARTS_GUID_LAST * arts_global_rank_count, sizeof(uint64_t));
  for (unsigned i = 0; i < ARTS_GUID_LAST * arts_global_rank_count; i++) {
    arts_node_info.keys[0][i] = 1;
  }
  arts_node_info.global_guid_thread_id =
      (uint64_t *)calloc(1, sizeof(uint64_t));
  num_tables = 1;
  min_global_guid_thread = 0;
  max_global_guid_thread = 1;
  keys_per_thread = 1u << 20;
  global_guid_on = 0;

  /* DB seq allocator (mirrors set_guid_generator_after_parallel_start). */
  arts_db_seq_budget =
      ((ARTS_GUID_DB_SEQ_MASK + 1) - ARTS_GUID_DB_STARTUP_RESERVE) /
      arts_global_rank_count;
  db_seq_creator_base = arts_db_seq_budget * arts_global_rank_id;
  if (db_seq_next) {
    free((void *)db_seq_next);
  }
  db_seq_next = (volatile uint64_t *)malloc(sizeof(uint64_t) *
                                            arts_global_rank_count);
  for (unsigned r = 0; r < arts_global_rank_count; r++) {
    db_seq_next[r] = db_seq_creator_base + 1;
  }
  free(t_db_cursor);
  t_db_cursor = NULL;
  arts_node_info.route_table =
      (arts_route_table_t **)calloc(1, sizeof(arts_route_table_t *));
  arts_node_info.route_table[0] = arts_new_route_table(1024, 10);
  for (int s = 0; s < ARTS_REMOTE_ROUTE_SHARDS; s++) {
    arts_node_info.remote_route_table[s] = arts_new_route_table(64, 10);
  }
}

/* ── concurrent storm ──────────────────────────────────────────────────── */
#define STORM_THREADS 8
#define STORM_ITERS 50000

typedef struct {
  arts_guid_t guid;
  atomic_int *gate;
  int role;  /* 0 = installer, 1 = reader */
  long objs; /* objects this installer allocated (== potential displacements) */
} storm_ctx_t;

static void *storm_worker(void *vp) {
  storm_ctx_t *c = (storm_ctx_t *)vp;
  while (atomic_load_explicit(c->gate, memory_order_acquire) == 0) {
  }
  if (c->role == 0) {
    for (int i = 0; i < STORM_ITERS; i++) {
      int *obj = (int *)malloc(sizeof(int));
      *obj = i;
      /* unconditional install: displaces whatever is there. */
      arts_route_table_install(obj, c->guid, 0, false);
      c->objs++;
    }
  } else {
    for (int i = 0; i < STORM_ITERS; i++) {
      arts_shared_ptr_t h = arts_route_table_lookup(c->guid);
      if (h) {
        int *p = (int *)arts_shared_get(h);
        /* read it — must be live memory (ASan catches UAF). */
        volatile int sink = *p;
        (void)sink;
        arts_shared_release(&h);
      }
    }
  }
  return NULL;
}

int main(void) {
  rt_init();
  /* install needs a per-kind deleter for DB to actually free objects. */
  arts_route_table_register_deleter(ARTS_GUID_DB, test_deleter);

  arts_guid_t g = arts_guid_reserve(ARTS_GUID_DB, 0);

  /* ---- 1-4: serial REPLACE-over-live with deferred free. ---- */
  int *A = (int *)malloc(sizeof(int));
  *A = 0xA;
  arts_route_table_install(A, g, 0, false);

  /* hold a reader handle on A */
  arts_shared_ptr_t hA = arts_route_table_lookup(g);
  if (!hA || arts_shared_get(hA) != A) {
    FAIL("lookup after install(A) did not return A\n");
  }

  int *B = (int *)malloc(sizeof(int));
  *B = 0xB;
  arts_route_table_install(B, g, 0, false); /* displaces A's cb */

  /* A's deleter must NOT have run yet — hA still pins it. */
  if (atomic_load(&g_deletes) != 0) {
    FAIL("A freed while a reader handle still held (UAF risk): deletes=%d\n",
         atomic_load(&g_deletes));
  }
  /* A is still readable through the stale handle. */
  if (*(int *)arts_shared_get(hA) != 0xA) {
    FAIL("stale handle A corrupted\n");
  }

  /* now drop the stale handle → A's deleter runs exactly once. */
  arts_shared_release(&hA);
  if (atomic_load(&g_deletes) != 1) {
    FAIL("A deleter ran %d times, want 1\n", atomic_load(&g_deletes));
  }

  /* fresh lookup returns B; REPLACE did not bump gen. */
  arts_shared_ptr_t hB = arts_route_table_lookup(g);
  if (!hB || arts_shared_get(hB) != B) {
    FAIL("lookup after install(B) did not return B\n");
  }
  arts_shared_release(&hB);
  if (arts_route_table_was_destroyed(g)) {
    FAIL("REPLACE wrongly registered as a destroyed generation\n");
  }

  /* ---- 5: concurrent storm. ---- */
  /* seed slot with an initial object so readers always find something live;
   * this initial cb is displaced by the first installer.  Install it BEFORE
   * resetting the delete counter so the prior B's displacement isn't
   * miscounted into the storm tally. */
  int *seed = (int *)malloc(sizeof(int));
  *seed = -1;
  arts_route_table_install(seed, g, 0, false);
  atomic_store(&g_deletes, 0); /* count displacements only in the storm */

  pthread_t tids[STORM_THREADS];
  storm_ctx_t ctx[STORM_THREADS];
  atomic_int gate;
  atomic_init(&gate, 0);
  for (int i = 0; i < STORM_THREADS; i++) {
    ctx[i].guid = g;
    ctx[i].gate = &gate;
    ctx[i].role = (i % 2); /* half installers, half readers */
    ctx[i].objs = 0;
    if (pthread_create(&tids[i], NULL, storm_worker, &ctx[i]) != 0) {
      FAIL("pthread_create %d\n", i);
    }
  }
  atomic_store_explicit(&gate, 1, memory_order_release);
  long installed = 0;
  for (int i = 0; i < STORM_THREADS; i++) {
    pthread_join(tids[i], NULL);
    installed += ctx[i].objs;
  }

  /* After the storm, exactly ONE object remains in the slot (the last
   * installed cb); every other installed object + the seed was displaced and
   * freed.  #displaced = (#installed objects) + 1 seed − 1 survivor.
   * The survivor is freed by the final teardown below.  So pre-teardown
   * deletes == installed + 1(seed) − 1(survivor) == installed. */
  long pre_teardown = atomic_load(&g_deletes);
  if (pre_teardown != installed) {
    FAIL("storm displaced-free count %ld != installed %ld (lost/double "
         "free)\n",
         pre_teardown, installed);
  }

  /* teardown frees the survivor's cb (one more delete). */
  arts_clean_up_route_table(arts_node_info.route_table[0]);
  if (atomic_load(&g_deletes) != installed + 1) {
    FAIL("teardown did not free the survivor exactly once: %d vs %ld\n",
         atomic_load(&g_deletes), installed + 1);
  }

  printf("PASS route_table_install_replace: deferred-free over live object, "
         "no UAF, %ld storm installs all accounted\n",
         installed);
  return 0;
}
