/* Slot reclamation under concurrent duplicate destroy, churn, and the two
 * ingredients an identity confusion needs:
 *
 *  - duplicate destroys of one GUID (designed-in legal traffic), racing
 *    installs whose slots recycle fast (32 slots total);
 *  - RE-RESERVATIONS of already-destroyed GUIDs (a late message for a dead
 *    GUID legally re-enters the table as a value-less reservation), which is
 *    what lets a freed slot return to a previously observed key while a
 *    reader is between its value load and its identity check;
 *  - readers looking up a GUID that is being destroyed and whose slot is
 *    being recycled underneath them.
 *
 * Oracles (all derived from the public contract, none from comments):
 *  O1  a cb deleter must not run for an object whose owner never asked for a
 *      destroy                        -> "duplicate destroy freed a stranger"
 *  O2  right after install_if_absent() returns true, lookup(g) must find that
 *      object                          -> "stale key=0 orphaned a live install"
 *  O3  lookup(g) must never return another GUID's object.
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
void arts_ooo_redrive_all(struct arts_route_item_s *s) { (void)s; }
void arts_ooo_free_all(struct arts_route_item_s *s) { (void)s; }

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

/* ---- oracles ---------------------------------------------------------- */
typedef struct {
  arts_guid_t guid;
  _Atomic int destroy_requested; /* owner armed a destroy for this object */
} obj_t;

static _Atomic long v_premature_free;  /* O1 */
static _Atomic long v_orphaned;        /* O2 */
static _Atomic long v_crossguid;       /* O3 */
static _Atomic long n_installs;

static void test_deleter(void *p) {
  obj_t *o = (obj_t *)p;
  if (atomic_load_explicit(&o->destroy_requested, memory_order_acquire) == 0) {
    atomic_fetch_add_explicit(&v_premature_free, 1, memory_order_relaxed);
  }
  free(o);
}

static void rt_init(unsigned size, unsigned shift) {
  arts_thread_info.thread_id = 0;
  arts_node_info.total_thread_count = 1;
  arts_node_info.gpu = 0;
  arts_node_info.keys = (uint64_t **)calloc(1, sizeof(uint64_t *));
  arts_node_info.keys[0] = (uint64_t *)calloc(
      (size_t)ARTS_GUID_LAST * arts_global_rank_count, sizeof(uint64_t));
  for (unsigned i = 0; i < ARTS_GUID_LAST * arts_global_rank_count; i++) {
    arts_node_info.keys[0][i] = 1;
  }
  arts_node_info.global_guid_thread_id = (uint64_t *)calloc(1, sizeof(uint64_t));
  num_tables = 1;
  min_global_guid_thread = 0;
  max_global_guid_thread = 1;
  keys_per_thread = 1u << 20;
  global_guid_on = 0;
  arts_db_seq_budget =
      ((ARTS_GUID_DB_SEQ_MASK + 1) - ARTS_GUID_DB_STARTUP_RESERVE) /
      arts_global_rank_count;
  db_seq_creator_base = arts_db_seq_budget * arts_global_rank_id;
  if (db_seq_next) { free((void *)db_seq_next); }
  db_seq_next = (volatile uint64_t *)malloc(sizeof(uint64_t) * arts_global_rank_count);
  for (unsigned r = 0; r < arts_global_rank_count; r++) {
    db_seq_next[r] = db_seq_creator_base + 1;
  }
  free(t_db_cursor); t_db_cursor = NULL;
  arts_node_info.route_table = (arts_route_table_t **)calloc(1, sizeof(arts_route_table_t *));
  arts_node_info.route_table[0] = arts_new_route_table(size, shift);
  for (int s = 0; s < ARTS_REMOTE_ROUTE_SHARDS; s++) {
    arts_node_info.remote_route_table[s] = arts_new_route_table(64, 10);
  }
}

/* ---- the race ---------------------------------------------------------- */
#define ROUNDS   20000
#define CHURN_THREADS 6
#define DEAD_RING 64

static pthread_barrier_t bar;
static _Atomic arts_guid_t g_victim;   /* the GUID both destroyers tear down */
static _Atomic int g_done;

/* Ring of recently destroyed GUIDs, fed by main (victims) and churn (a
 * sample); the late-messenger re-reserves them so dead keys keep re-entering
 * freed slots — the state a slot-word identity check cannot tell apart. */
static _Atomic arts_guid_t g_dead[DEAD_RING];
static _Atomic unsigned g_dead_w;

static void dead_push(arts_guid_t g) {
  unsigned w = atomic_fetch_add_explicit(&g_dead_w, 1, memory_order_relaxed);
  atomic_store_explicit(&g_dead[w % DEAD_RING], g, memory_order_release);
}

/* A late message for a destroyed GUID: reserve_or_lookup with no install.
 * Sweeps the ring continuously — reserving a GUID that is still live, or one
 * already re-reserved, just finds the existing slot, so the sweep keeps
 * re-reserving each ring GUID the instant its slot is returned while the
 * table's growth stays bounded by the distinct GUIDs ever pushed. */
static void *late_messenger(void *vp) {
  (void)vp;
  while (!atomic_load_explicit(&g_done, memory_order_acquire)) {
    for (int i = 0; i < DEAD_RING; i++) {
      arts_guid_t g =
          atomic_load_explicit(&g_dead[i], memory_order_acquire);
      if (g == 0) {
        continue;
      }
      arts_route_item_t *it = NULL;
      arts_route_table_reserve_or_lookup(g, &it);
    }
  }
  return NULL;
}

/* Reader racing the destroy+recycle: O3 across the full window of a lookup,
 * not only the instant after an install. */
static void *reader(void *vp) {
  (void)vp;
  while (!atomic_load_explicit(&g_done, memory_order_acquire)) {
    arts_guid_t g = atomic_load_explicit(&g_victim, memory_order_acquire);
    if (g == 0) {
      continue;
    }
    arts_shared_ptr_t h = arts_route_table_lookup(g);
    if (h) {
      obj_t *o = (obj_t *)arts_shared_get(h);
      if (o != NULL && o->guid != g) {
        atomic_fetch_add_explicit(&v_crossguid, 1, memory_order_relaxed);
      }
      arts_shared_release(&h);
    }
  }
  return NULL;
}

/* Two destroyer threads issue the SAME destroy concurrently.  Every duplicate
 * DESTROY_NOTIFY / completion-vs-wire-destroy pair in the runtime looks
 * exactly like this. */
static void *destroyer(void *vp) {
  (void)vp;
  for (int r = 0; r < ROUNDS; r++) {
    pthread_barrier_wait(&bar);
    arts_guid_t g = atomic_load_explicit(&g_victim, memory_order_acquire);
    (void)arts_route_table_set_destroyed(g);
    pthread_barrier_wait(&bar);
  }
  return NULL;
}

/* Churn: install a fresh GUID, immediately verify it is findable and is our
 * own object, then destroy it.  These are the "unrelated objects" whose slots
 * the duplicate destroy can land on. */
static void *churn(void *vp) {
  (void)vp;
  for (int r = 0; r < ROUNDS; r++) {
    pthread_barrier_wait(&bar);
    for (int k = 0; k < 4; k++) {
      arts_guid_t g = arts_guid_reserve(ARTS_GUID_DB, 0);
      obj_t *o = (obj_t *)calloc(1, sizeof(obj_t));
      o->guid = g;
      atomic_store_explicit(&o->destroy_requested, 0, memory_order_release);
      if (!arts_route_table_install_if_absent(o, g, 0, false)) {
        free(o);
        continue;
      }
      atomic_fetch_add_explicit(&n_installs, 1, memory_order_relaxed);
      arts_shared_ptr_t h = arts_route_table_lookup(g);
      void *got = h ? arts_shared_get(h) : NULL;
      if (got == NULL) {
        atomic_fetch_add_explicit(&v_orphaned, 1, memory_order_relaxed);
      } else if (((obj_t *)got)->guid != g) {
        atomic_fetch_add_explicit(&v_crossguid, 1, memory_order_relaxed);
      }
      if (h) { arts_shared_release(&h); }
      atomic_store_explicit(&o->destroy_requested, 1, memory_order_release);
      (void)arts_route_table_set_destroyed(g);
      if ((r & 15) == 0 && k == 0) {
        dead_push(g); /* sampled: keeps the reservation growth bounded */
      }
    }
    pthread_barrier_wait(&bar);
  }
  return NULL;
}

int main(void) {
  rt_init(4, 2); /* 4 buckets x 8 probes = 32 slots -> slots recycle fast */
  arts_route_table_register_deleter(ARTS_GUID_DB, test_deleter);

  pthread_t d[2], c[CHURN_THREADS], lm, rd[2];
  pthread_barrier_init(&bar, NULL, 2 + CHURN_THREADS + 1);
  for (int i = 0; i < 2; i++) { pthread_create(&d[i], NULL, destroyer, NULL); }
  for (int i = 0; i < CHURN_THREADS; i++) { pthread_create(&c[i], NULL, churn, NULL); }
  pthread_create(&lm, NULL, late_messenger, NULL);
  for (int i = 0; i < 2; i++) { pthread_create(&rd[i], NULL, reader, NULL); }

  for (int r = 0; r < ROUNDS; r++) {
    arts_guid_t g = arts_guid_reserve(ARTS_GUID_DB, 0);
    obj_t *o = (obj_t *)calloc(1, sizeof(obj_t));
    o->guid = g;
    atomic_store_explicit(&o->destroy_requested, 1, memory_order_release);
    (void)arts_route_table_install_if_absent(o, g, 0, false);
    atomic_store_explicit(&g_victim, g, memory_order_release);
    /* Into the ring BEFORE it dies: the sweep then re-reserves this GUID the
     * moment the destroyers return its slot, while the readers are still
     * looking it up — the exact overlap an identity check must survive. */
    dead_push(g);
    pthread_barrier_wait(&bar);   /* destroyers + churn go */
    pthread_barrier_wait(&bar);   /* round end */
  }
  for (int i = 0; i < 2; i++) { pthread_join(d[i], NULL); }
  for (int i = 0; i < CHURN_THREADS; i++) { pthread_join(c[i], NULL); }
  atomic_store_explicit(&g_done, 1, memory_order_release);
  pthread_join(lm, NULL);
  for (int i = 0; i < 2; i++) { pthread_join(rd[i], NULL); }

  long p = atomic_load(&v_premature_free);
  long o2 = atomic_load(&v_orphaned);
  long x = atomic_load(&v_crossguid);
  printf("installs=%ld  O1 premature_free=%ld  O2 orphaned_install=%ld  "
         "O3 crossguid=%ld\n", atomic_load(&n_installs), p, o2, x);
  if (p || o2 || x) {
    printf("FAIL rt_reclaim_dupdestroy\n");
    return 1;
  }
  printf("PASS rt_reclaim_dupdestroy\n");
  return 0;
}
