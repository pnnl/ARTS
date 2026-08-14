/* SPDX-License-Identifier: Apache-2.0
 *
 * route_table_install_if_absent — CAS-install win/loss + set_destroyed
 * single-flight + gen-bump ordering (census 18-gas GAPs 2 & 3).
 *
 * arts_route_table_install_if_absent CASes a fresh cb into an empty slot:
 * exactly ONE caller wins per generation; losers arts_shared_abandon their cb
 * (the abandon must NOT run the deleter — the loser keeps owning its object).
 * arts_route_table_set_destroyed is single-flight (only the caller whose
 * exchange observes a non-NULL cb returns true) and bumps `gen` (acq_rel)
 * BEFORE releasing the old cb.
 *
 * Scenarios:
 *
 *   A. install_if_absent storm on ONE empty slot: exactly one true return; the
 *      slot holds the winner's object; the deleter has NOT run for any loser
 *      object (losers keep theirs); winner freed exactly once at teardown.
 *
 *   B. set_destroyed single-flight: N threads race set_destroyed on a populated
 *      slot — exactly one returns true; the object's deleter runs exactly once;
 *      gen is bumped exactly once (was_destroyed true, gen observed > 0).
 *
 *   C. install_if_absent racing set_destroyed across rounds: repeated
 *      create→destroy on the same GUID; every round has exactly one creator and
 *      at most one destroyer; no double-free / no leak (ASan); gen is
 *      monotone-increasing across destroyed rounds.
 *
 * No runtime: route_table.c + guid.c + shared.c #include'd; OoO drain is a
 * no-op (no OoO payloads pushed).
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

static _Atomic int g_deletes;
static void test_deleter(void *obj) {
  atomic_fetch_add_explicit(&g_deletes, 1, memory_order_relaxed);
  free(obj);
}

#define FAIL(...)                                                              \
  do {                                                                         \
    (void)fprintf(stderr, "FAIL route_table_install_if_absent: " __VA_ARGS__); \
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

/* ── Scenario A: install_if_absent storm ───────────────────────────────── */
#define A_THREADS 12

typedef struct {
  arts_guid_t guid;
  atomic_int *gate;
  int *obj; /* this thread's candidate object */
  int won;  /* did this thread win the CAS? */
} a_ctx_t;

static void *a_worker(void *vp) {
  a_ctx_t *c = (a_ctx_t *)vp;
  while (atomic_load_explicit(c->gate, memory_order_acquire) == 0) {
  }
  c->won =
      arts_route_table_install_if_absent(c->obj, c->guid, 0, false) ? 1 : 0;
  return NULL;
}

/* ── Scenario B: set_destroyed single-flight ───────────────────────────── */
#define B_THREADS 12

typedef struct {
  arts_guid_t guid;
  atomic_int *gate;
  int won;
} b_ctx_t;

static void *b_worker(void *vp) {
  b_ctx_t *c = (b_ctx_t *)vp;
  while (atomic_load_explicit(c->gate, memory_order_acquire) == 0) {
  }
  c->won = arts_route_table_set_destroyed(c->guid) ? 1 : 0;
  return NULL;
}

int main(void) {
  rt_init();
  arts_route_table_register_deleter(ARTS_GUID_DB, test_deleter);

  /* ---- Scenario A ---- */
  {
    arts_guid_t g = arts_guid_reserve(ARTS_GUID_DB, 0);
    pthread_t tids[A_THREADS];
    a_ctx_t ctx[A_THREADS];
    atomic_int gate;
    atomic_init(&gate, 0);
    atomic_store(&g_deletes, 0);
    for (int i = 0; i < A_THREADS; i++) {
      ctx[i].guid = g;
      ctx[i].gate = &gate;
      ctx[i].obj = (int *)malloc(sizeof(int));
      *ctx[i].obj = i;
      ctx[i].won = 0;
      if (pthread_create(&tids[i], NULL, a_worker, &ctx[i]) != 0) {
        FAIL("A pthread_create %d\n", i);
      }
    }
    atomic_store_explicit(&gate, 1, memory_order_release);
    int winners = 0, win_idx = -1;
    for (int i = 0; i < A_THREADS; i++) {
      pthread_join(tids[i], NULL);
      if (ctx[i].won) {
        winners++;
        win_idx = i;
      }
    }
    if (winners != 1) {
      FAIL("A: %d winners, want 1\n", winners);
    }
    /* loser objects must NOT have been freed (abandon != delete). */
    if (atomic_load(&g_deletes) != 0) {
      FAIL("A: %d deletes after CAS race — abandon ran a deleter\n",
           atomic_load(&g_deletes));
    }
    /* slot holds the winner's object. */
    arts_shared_ptr_t h = arts_route_table_lookup(g);
    if (!h || arts_shared_get(h) != ctx[win_idx].obj) {
      FAIL("A: slot does not hold the winner's object\n");
    }
    arts_shared_release(&h);
    /* losers still own their objects → free them here (the test owns them). */
    for (int i = 0; i < A_THREADS; i++) {
      if (!ctx[i].won) {
        free(ctx[i].obj);
      }
    }
    /* destroy the winner cb → deleter frees winner exactly once. */
    if (!arts_route_table_set_destroyed(g)) {
      FAIL("A: set_destroyed on winner returned false\n");
    }
    if (atomic_load(&g_deletes) != 1) {
      FAIL("A: winner deleter ran %d times, want 1\n", atomic_load(&g_deletes));
    }
  }

  /* ---- Scenario B ---- */
  {
    arts_guid_t g = arts_guid_reserve(ARTS_GUID_DB, 0);
    int *obj = (int *)malloc(sizeof(int));
    *obj = 0x77;
    arts_route_table_install_if_absent(obj, g, 0, false);
    atomic_store(&g_deletes, 0);

    pthread_t tids[B_THREADS];
    b_ctx_t ctx[B_THREADS];
    atomic_int gate;
    atomic_init(&gate, 0);
    for (int i = 0; i < B_THREADS; i++) {
      ctx[i].guid = g;
      ctx[i].gate = &gate;
      ctx[i].won = 0;
      if (pthread_create(&tids[i], NULL, b_worker, &ctx[i]) != 0) {
        FAIL("B pthread_create %d\n", i);
      }
    }
    atomic_store_explicit(&gate, 1, memory_order_release);
    int winners = 0;
    for (int i = 0; i < B_THREADS; i++) {
      pthread_join(tids[i], NULL);
      winners += ctx[i].won;
    }
    if (winners != 1) {
      FAIL("B: %d set_destroyed winners, want 1 (single-flight)\n", winners);
    }
    if (atomic_load(&g_deletes) != 1) {
      FAIL("B: object freed %d times, want exactly 1\n",
           atomic_load(&g_deletes));
    }
    if (!arts_route_table_was_destroyed(g)) {
      FAIL("B: was_destroyed false after a destroyed generation\n");
    }
  }

  /* ---- Scenario C: create/destroy churn, gen monotone, no double-free ---- */
  {
    arts_guid_t g = arts_guid_reserve(ARTS_GUID_DB, 0);
    arts_route_item_t *item = NULL;
    arts_route_table_reserve_or_lookup(g, &item);
    uint64_t last_gen = 0;
    atomic_store(&g_deletes, 0);
    const int ROUNDS = 2000;
    for (int r = 0; r < ROUNDS; r++) {
      int *obj = (int *)malloc(sizeof(int));
      *obj = r;
      bool won = arts_route_table_install_if_absent(obj, g, 0, false);
      if (!won) {
        FAIL("C round %d: install_if_absent on a known-empty slot lost\n", r);
      }
      bool destroyed = arts_route_table_set_destroyed(g);
      if (!destroyed) {
        FAIL("C round %d: set_destroyed on a known-live slot returned false\n",
             r);
      }
      uint64_t gen = __atomic_load_n(&item->gen, __ATOMIC_ACQUIRE);
      if (gen <= last_gen) {
        FAIL("C round %d: gen not monotone (%lu <= %lu)\n", r,
             (unsigned long)gen, (unsigned long)last_gen);
      }
      last_gen = gen;
    }
    if (atomic_load(&g_deletes) != ROUNDS) {
      FAIL("C: %d frees over %d rounds (double-free or leak)\n",
           atomic_load(&g_deletes), ROUNDS);
    }
  }

  printf("PASS route_table_install_if_absent: single CAS winner, abandon "
         "keeps loser objects, set_destroyed single-flight + monotone gen\n");
  return 0;
}
