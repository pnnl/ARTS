/* SPDX-License-Identifier: Apache-2.0
 *
 * route_item_mirror_bridge — arts_route_item_install_data / _acquire, the
 * C-linkage bridge for non-global mirror tables (GPU per-device route tables)
 * (census 18-gas GAP 6).
 *
 * Unlike the global installs, these operate on a CALLER-LOCATED slot:
 *   - arts_route_item_install_data(item, obj, deleter): install-or-fail CAS.
 *     If the slot is already occupied it releases the read handle and returns
 *     false (the caller keeps owning obj — install-race loser semantics).  On
 *     an empty slot it makes a cb and CAS-installs it; on a lost CAS it
 *     ABANDONS the spare cb (no deleter run) and returns false.  A NULL
 *     deleter means the cb never frees obj.  It does NOT drain the OoO list.
 *   - arts_route_item_acquire(item): returns a caller-owned handle (ref held)
 *     or NULL; the caller must arts_shared_release it.  NULL-safe on item.
 *
 * Scenarios:
 *   1. Serial: install into an empty slot succeeds; acquire returns the obj;
 *      a second install_data into the OCCUPIED slot fails and does NOT free
 *      either object; NULL-deleter cb never frees obj; acquire(NULL) == NULL.
 *   2. Concurrent install_data storm on one slot: exactly one winner; losers'
 *      objects are NOT freed (abandon, not delete); every acquire handle is
 *      live/readable; no UAF / double-free (ASan).
 *
 * No runtime: route_table.c + guid.c + shared.c #include'd; OoO drain no-op.
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

#include "../../libs/src/core/gas/guid.c"
#include "../../libs/src/core/gas/route_table.c"
#include "../../libs/src/core/utils/shared.c"

struct arts_runtime_shared_s arts_node_info;
ARTS_THREAD_LOCAL struct arts_runtime_private_s arts_thread_info;

static _Atomic int g_deletes;
static void test_deleter(void *obj) {
  atomic_fetch_add_explicit(&g_deletes, 1, memory_order_relaxed);
  free(obj);
}

#define FAIL(...)                                                              \
  do {                                                                         \
    (void)fprintf(stderr, "FAIL route_item_mirror_bridge: " __VA_ARGS__);      \
    return 1;                                                                  \
  } while (0)

/* A standalone mirror table (caller-located slots; no global key→table map). */
static arts_route_table_t *g_mirror;

/* ── Scenario 2: concurrent install_data storm ─────────────────────────── */
#define S_THREADS 12

typedef struct {
  arts_route_item_t *item;
  atomic_int *gate;
  int *obj;
  int won;
} s_ctx_t;

static void *s_worker(void *vp) {
  s_ctx_t *c = (s_ctx_t *)vp;
  while (atomic_load_explicit(c->gate, memory_order_acquire) == 0) {
  }
  /* NULL deleter so the cb never frees — losers must keep their objects and
   * the winner's object is owned by the test (freed manually at the end). */
  c->won = arts_route_item_install_data(c->item, c->obj, NULL) ? 1 : 0;
  /* every thread also acquires + releases — must always be live or NULL. */
  arts_shared_ptr_t h = arts_route_item_acquire(c->item);
  if (h) {
    volatile int sink = *(int *)arts_shared_get(h);
    (void)sink;
    arts_shared_release(&h);
  }
  return NULL;
}

int main(void) {
  g_mirror = arts_new_route_table(16, 4);

  /* ---- 1. Serial install-or-fail. ---- */
  {
    arts_guid_t g = ARTS_GUID_MAKE(ARTS_GUID_DB, 0, 12345);
    arts_route_item_t *item =
        arts_route_table_search_for_empty(g_mirror, g, false);
    if (!item) {
      FAIL("1: search_for_empty NULL\n");
    }
    atomic_store(&g_deletes, 0);

    int *A = (int *)malloc(sizeof(int));
    *A = 0xA;
    if (!arts_route_item_install_data(item, A, test_deleter)) {
      FAIL("1: install into empty slot returned false\n");
    }
    /* acquire returns A. */
    arts_shared_ptr_t h = arts_route_item_acquire(item);
    if (!h || arts_shared_get(h) != A) {
      FAIL("1: acquire did not return installed obj\n");
    }
    arts_shared_release(&h);

    /* second install into OCCUPIED slot fails, frees neither. */
    int *B = (int *)malloc(sizeof(int));
    *B = 0xB;
    if (arts_route_item_install_data(item, B, test_deleter)) {
      FAIL("1: install into occupied slot returned true\n");
    }
    if (atomic_load(&g_deletes) != 0) {
      FAIL("1: occupied-install freed something (deletes=%d)\n",
           atomic_load(&g_deletes));
    }
    /* slot still holds A. */
    h = arts_route_item_acquire(item);
    if (!h || arts_shared_get(h) != A) {
      FAIL("1: slot changed after a failed install\n");
    }
    arts_shared_release(&h);
    free(B); /* caller still owns B (loser keeps obj) */

    /* tear down the slot: detach + release the cb → A freed once. */
    arts_shared_ptr_t old = arts_atomic_shared_exchange(&item->value, NULL);
    arts_shared_release(&old);
    if (atomic_load(&g_deletes) != 1) {
      FAIL("1: A not freed exactly once on detach (%d)\n",
           atomic_load(&g_deletes));
    }

    /* acquire(NULL) is NULL-safe. */
    if (arts_route_item_acquire(NULL) != NULL) {
      FAIL("1: acquire(NULL) != NULL\n");
    }
  }

  /* ---- 1b. NULL-deleter cb never frees. ---- */
  {
    arts_guid_t g = ARTS_GUID_MAKE(ARTS_GUID_DB, 0, 22222);
    arts_route_item_t *item =
        arts_route_table_search_for_empty(g_mirror, g, false);
    int *C = (int *)malloc(sizeof(int));
    *C = 0xC;
    atomic_store(&g_deletes, 0);
    if (!arts_route_item_install_data(item, C, NULL)) {
      FAIL("1b: NULL-deleter install returned false\n");
    }
    arts_shared_ptr_t old = arts_atomic_shared_exchange(&item->value, NULL);
    arts_shared_release(&old); /* cb gone, but NULL deleter → C not freed */
    if (atomic_load(&g_deletes) != 0) {
      FAIL("1b: NULL-deleter cb wrongly freed obj\n");
    }
    free(C); /* the test owns C */
  }

  /* ---- 2. Concurrent install_data storm (one winner). ---- */
  {
    arts_guid_t g = ARTS_GUID_MAKE(ARTS_GUID_DB, 0, 33333);
    arts_route_item_t *item =
        arts_route_table_search_for_empty(g_mirror, g, false);
    pthread_t tids[S_THREADS];
    s_ctx_t ctx[S_THREADS];
    atomic_int gate;
    atomic_init(&gate, 0);
    for (int i = 0; i < S_THREADS; i++) {
      ctx[i].item = item;
      ctx[i].gate = &gate;
      ctx[i].obj = (int *)malloc(sizeof(int));
      *ctx[i].obj = i;
      ctx[i].won = 0;
      if (pthread_create(&tids[i], NULL, s_worker, &ctx[i]) != 0) {
        FAIL("2: pthread_create %d\n", i);
      }
    }
    atomic_store_explicit(&gate, 1, memory_order_release);
    int winners = 0, win = -1;
    for (int i = 0; i < S_THREADS; i++) {
      pthread_join(tids[i], NULL);
      if (ctx[i].won) {
        winners++;
        win = i;
      }
    }
    if (winners != 1) {
      FAIL("2: %d winners, want 1\n", winners);
    }
    arts_shared_ptr_t h = arts_route_item_acquire(item);
    if (!h || arts_shared_get(h) != ctx[win].obj) {
      FAIL("2: slot does not hold the winner's obj\n");
    }
    arts_shared_release(&h);
    /* NULL deleter → cb never frees; the test owns every object. */
    for (int i = 0; i < S_THREADS; i++) {
      free(ctx[i].obj);
    }
  }

  /* clean_up detaches+releases every surviving slot cb (the storm winner's),
   * then delete frees the table memory. */
  arts_clean_up_route_table(g_mirror);
  arts_delete_route_table(g_mirror);
  printf("PASS route_item_mirror_bridge: install-or-fail CAS, NULL-deleter, "
         "acquire handle lifecycle, single concurrent winner\n");
  return 0;
}
