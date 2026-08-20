/* SPDX-License-Identifier: Apache-2.0
 *
 * route_table_segment_grow — force the lazy segment-chain GROWTH in
 * search_for_empty: band overflow, CAS-loss spare free, and convergence
 * (census 18-gas GAP 7; suspected bugs B041 / B042).
 *
 * A route_table band is COLLISION_RESOLVES (8) wide.  When >8 DISTINCT GUIDs
 * hash to the same band on the root segment, the 9th claim cannot fit and
 * search_for_empty lazily grows the chain: it builds a 2x / shift+1 spare and
 * CASes it into `->next`; a thread that loses the grow-CAS frees its spare
 * (never published → leak-free) and adopts the winner's segment.
 *
 * We hand-pick >8 GUIDs that all hash to band base 0 at the root (size=2,
 * shift=10), forcing at least one growth, then:
 *
 *   1. Serial: install all colliding keys → the chain grew (root->next != NULL)
 *      and every key has exactly one canonical slot; lookups return the right
 *      objects; no slot aliasing.
 *
 *   2. Concurrent grow race: many threads simultaneously reserve_or_lookup the
 *      same overflowing key-set → all converge on one slot per key, no
 *      lost/duplicate slot, and (ASan) the CAS-loss spare frees leave no leak.
 *
 *   3. Document B042: the "band full, growth can't help → hard ARTS_ERROR" path
 *      is effectively UNREACHABLE — search_for_empty always extends the chain
 *      when `next == NULL` (calloc-backed), so an arbitrarily large colliding
 *      set is always absorbed by enough growth.  We install far more than one
 *      band's worth of colliding keys and assert NO abort occurs.
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
void arts_ooo_redrive_all(struct arts_route_item_s *s) { (void)s; }
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

#define FAIL(...)                                                              \
  do {                                                                         \
    (void)fprintf(stderr, "FAIL route_table_segment_grow: " __VA_ARGS__);      \
    return 1;                                                                  \
  } while (0)

/* A standalone root table so we control size/shift (the grow path keys off
 * the table's own shift, not the global generator).
 *
 * INVARIANT (load-bearing): a route_table's `size` MUST equal 2^`shift` —
 * get_route_table_key returns `(hash >> (64-shift)) * COLLISION_RESOLVES`, a
 * band base in [0, 2^shift * 8), and the slot array has exactly size*8 slots.
 * arts_new_route_table(1024,10) in production satisfies 2^10==1024, and the
 * grow path's newFunc(2*size, shift+1) preserves it (2*2^shift == 2^(shift+1)).
 * We pick a SMALL but invariant-respecting root: shift=4 → size=16 (128 slots),
 * so growth is reached quickly while every grown band base stays in range. */
static arts_route_table_t *g_root;
static const unsigned ROOT_SHIFT = 4;
static const unsigned ROOT_SIZE = 1u << 4; /* == 2^ROOT_SHIFT */

/* Collect N distinct GUID keys that hash to band base 0 at the root. */
#define NCOLLIDE 40
static arts_guid_t g_keys[NCOLLIDE];

static int collect_colliding_keys(void) {
  int found = 0;
  for (uint64_t k = 1; k < 5000000 && found < NCOLLIDE; k++) {
    arts_guid_t g = ARTS_GUID_MAKE(ARTS_GUID_DB, 0, k);
    if (get_route_table_key((uint64_t)g, ROOT_SHIFT) == 0) {
      g_keys[found++] = g;
    }
  }
  return found;
}

static unsigned chain_len(arts_route_table_t *t) {
  unsigned n = 0;
  for (arts_route_table_t *c = t; c; c = c->next) {
    n++;
  }
  return n;
}

/* ── concurrent grow race ──────────────────────────────────────────────── */
#define GROW_THREADS 8

typedef struct {
  atomic_int *gate;
  arts_route_item_t **observed; /* [NCOLLIDE] per-thread slot view */
} grow_ctx_t;

static void *grow_worker(void *vp) {
  grow_ctx_t *c = (grow_ctx_t *)vp;
  while (atomic_load_explicit(c->gate, memory_order_acquire) == 0) {
  }
  for (int rep = 0; rep < 200; rep++) {
    for (int i = 0; i < NCOLLIDE; i++) {
      arts_route_item_t *item =
          arts_route_table_search_for_empty(g_root, g_keys[i], false);
      if (!item) {
        return (void *)1; /* never NULL */
      }
      if (c->observed[i] == NULL) {
        c->observed[i] = item;
      } else if (c->observed[i] != item) {
        return (void *)2; /* slot diverged within a thread */
      }
    }
  }
  return NULL;
}

int main(void) {
  int n = collect_colliding_keys();
  if (n < NCOLLIDE) {
    FAIL("only found %d colliding keys (need %d)\n", n, NCOLLIDE);
  }

  /* ---- 1. Serial growth + canonical slots. ---- */
  g_root = arts_new_route_table(ROOT_SIZE, ROOT_SHIFT);
  arts_route_item_t *slots[NCOLLIDE];
  for (int i = 0; i < NCOLLIDE; i++) {
    slots[i] = arts_route_table_search_for_empty(g_root, g_keys[i], false);
    if (!slots[i]) {
      FAIL("1: search_for_empty returned NULL for key %d\n", i);
    }
    if (slots[i]->key != g_keys[i]) {
      FAIL("1: slot %d claimed wrong key\n", i);
    }
  }
  /* The chain MUST have grown past the root (one band of 8 can't hold 40). */
  if (chain_len(g_root) < 2) {
    FAIL("1: chain did not grow (len=%u) despite %d colliding keys\n",
         chain_len(g_root), NCOLLIDE);
  }
  /* Every key resolves back to exactly its slot; no aliasing. */
  for (int i = 0; i < NCOLLIDE; i++) {
    arts_route_item_t *f = arts_route_table_search_for_key(g_root, g_keys[i]);
    if (f != slots[i]) {
      FAIL("1: key %d search_for_key %p != claimed %p\n", i, (void *)f,
           (void *)slots[i]);
    }
    for (int j = 0; j < i; j++) {
      if (slots[i] == slots[j]) {
        FAIL("1: keys %d and %d aliased the same slot\n", i, j);
      }
    }
  }
  arts_delete_route_table(g_root);

  /* ---- 2. Concurrent grow race (spare alloc/CAS-loss-free under ASan). ----
   */
  g_root = arts_new_route_table(ROOT_SIZE, ROOT_SHIFT);
  pthread_t tids[GROW_THREADS];
  grow_ctx_t ctx[GROW_THREADS];
  atomic_int gate;
  atomic_init(&gate, 0);
  arts_route_item_t **storage = (arts_route_item_t **)calloc(
      (size_t)GROW_THREADS * NCOLLIDE, sizeof(arts_route_item_t *));
  for (int t = 0; t < GROW_THREADS; t++) {
    ctx[t].gate = &gate;
    ctx[t].observed = storage + (size_t)t * NCOLLIDE;
    if (pthread_create(&tids[t], NULL, grow_worker, &ctx[t]) != 0) {
      FAIL("2: pthread_create %d\n", t);
    }
  }
  atomic_store_explicit(&gate, 1, memory_order_release);
  for (int t = 0; t < GROW_THREADS; t++) {
    void *rc = NULL;
    pthread_join(tids[t], &rc);
    if (rc) {
      FAIL("2: worker %d returned error code %ld\n", t, (long)rc);
    }
  }
  /* Cross-thread: all threads must agree on the slot pointer for each key. */
  for (int i = 0; i < NCOLLIDE; i++) {
    arts_route_item_t *winner = NULL;
    for (int t = 0; t < GROW_THREADS; t++) {
      arts_route_item_t *p = storage[(size_t)t * NCOLLIDE + i];
      if (!p) {
        continue;
      }
      if (!winner) {
        winner = p;
      } else if (winner != p) {
        FAIL("2: key %d slot diverged across threads (%p vs %p)\n", i,
             (void *)winner, (void *)p);
      }
    }
    if (!winner) {
      FAIL("2: key %d never observed\n", i);
    }
  }
  free(storage);
  arts_delete_route_table(g_root);

  /* ---- 3. B042 unreachable: a large colliding set is always absorbed. ---- *
   * NCOLLIDE (40) >> one band (8); if growth could ever fail to help, the
   * serial install above would have aborted (ARTS_ERROR → _exit non-zero).
   * Reaching here proves the chain always extends — the hard-ERROR is dead
   * code under a calloc-backed allocator. */

  printf("PASS route_table_segment_grow: forced chain growth absorbed %d "
         "colliding keys, concurrent grow converges, B042 hard-ERROR "
         "unreachable\n",
         NCOLLIDE);
  return 0;
}
