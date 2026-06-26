/* SPDX-License-Identifier: Apache-2.0
 *
 * T021 — last-element tie-break (libs/src/core/utils/deque.c).
 * When exactly ONE element is in the deque, the owner's pop_front (b==t case,
 * which CASes top: t->t+1) races every stealer's pop_back (which CASes the
 * same top: t->t+1).  The Chase-Lev invariant: EXACTLY ONE of {owner, one
 * stealer} wins the element; everyone else gets NULL.  Never zero winners
 * (lost element), never two winners (duplicate).
 *
 * Harness: ROUNDS rounds.  Each round:
 *   - the owner pushes exactly one unique item, then races its own pop_front
 *     against N stealers all doing pop_back on the same single element;
 *   - a per-round gate (g_round_gate := round#) releases all stealers at once
 *     to maximize the b==t collision;
 *   - each stealer does exactly one pop_back then increments g_done;
 *   - the owner waits until all N stealers reported done (deterministic
 *     barrier) so the winner read can never race a slow stealer's increment;
 *   - winners for that round's element must be == 1.
 *
 * This is the narrowest, highest-value concurrency property of the deque:
 * the self-stealing CAS arbitration.  A bug (e.g. owner returning the item
 * without CASing, or pop_back returning a slot whose CAS failed) shows up as
 * winners==0 (loss) or winners>=2 (dup).
 */

#include "arts/utils/deque.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NSTEALERS 6
#define ROUNDS (100 * 1000)

static struct arts_deque_s *g_deque;
static atomic_int g_round_gate; /* bumped to round# to release racers */
static atomic_int g_quit;
static _Atomic int g_winners; /* winners for the current round's item */
static _Atomic int g_done;    /* stealers that finished THIS round's pop_back */

static inline void *enc(uint64_t v) { return (void *)(uintptr_t)(v + 1); }

static void *stealer(void *arg) {
  int last_seen = 0;
  (void)arg;
  for (;;) {
    int g;
    /* wait for the next round (or quit). */
    while ((g = atomic_load_explicit(&g_round_gate, memory_order_acquire)) ==
           last_seen) {
      if (atomic_load_explicit(&g_quit, memory_order_acquire))
        return NULL;
    }
    last_seen = g;
    void *o = arts_deque_pop_back(g_deque);
    if (o)
      atomic_fetch_add_explicit(&g_winners, 1, memory_order_acq_rel);
    /* Signal this round complete AFTER the winner increment is visible so the
     * owner's barrier establishes happens-before with every stealer's result,
     * eliminating any read-too-early window. */
    atomic_fetch_add_explicit(&g_done, 1, memory_order_acq_rel);
  }
}

int main(void) {
  g_deque = arts_deque_new(4);
  atomic_init(&g_round_gate, 0);
  atomic_init(&g_quit, 0);
  atomic_init(&g_winners, 0);
  atomic_init(&g_done, 0);

  pthread_t stealers[NSTEALERS];
  for (int i = 0; i < NSTEALERS; i++) {
    pthread_create(&stealers[i], NULL, stealer, NULL);
  }

  uint64_t bad_zero = 0, bad_multi = 0;
  for (int r = 1; r <= ROUNDS; r++) {
    atomic_store_explicit(&g_winners, 0, memory_order_release);
    atomic_store_explicit(&g_done, 0, memory_order_release);
    /* publish exactly one element. */
    arts_deque_push_front(g_deque, enc((uint64_t)r), 0);
    /* release stealers for THIS round. */
    atomic_store_explicit(&g_round_gate, r, memory_order_release);
    /* owner races for the same single element. */
    void *o = arts_deque_pop_front(g_deque);
    if (o)
      atomic_fetch_add_explicit(&g_winners, 1, memory_order_acq_rel);

    /* Wait for all stealers to complete this round (deterministic barrier). */
    while (atomic_load_explicit(&g_done, memory_order_acquire) < NSTEALERS) {
    }

    /* If a stealer happened to push top past bottom is impossible here; the
     * deque is single-element so at most one pop succeeds.  But the OWNER may
     * have lost: ensure the element is fully consumed (drain front; now safe,
     * all stealers done). */
    void *x;
    while ((x = arts_deque_pop_front(g_deque)) != NULL) {
      atomic_fetch_add_explicit(&g_winners, 1, memory_order_acq_rel);
    }

    int w = atomic_load_explicit(&g_winners, memory_order_acquire);
    if (w == 0)
      bad_zero++;
    else if (w > 1)
      bad_multi++;
  }

  atomic_store_explicit(&g_quit, 1, memory_order_release);
  /* nudge gate so any waiting stealer re-checks quit. */
  atomic_fetch_add_explicit(&g_round_gate, 1, memory_order_release);
  for (int i = 0; i < NSTEALERS; i++) {
    pthread_join(stealers[i], NULL);
  }
  arts_deque_delete(g_deque);

  if (bad_zero || bad_multi) {
    (void)fprintf(
        stderr,
        "FAIL deque_last_element_tiebreak: rounds_with_zero_winner=%" PRIu64
        " rounds_with_multi_winner=%" PRIu64 " of %d\n",
        bad_zero, bad_multi, ROUNDS);
    return 1;
  }

  printf("PASS deque_last_element_tiebreak: %d rounds, exactly one winner "
         "each (owner vs %d stealers)\n",
         ROUNDS, NSTEALERS);
  return 0;
}

/* ── libc-backed shims so the test links deque.c without the ARTS runtime ── */
void *arts_calloc_aligned(size_t nmemb, size_t size, size_t align) {
  void *p = NULL;
  size_t total = nmemb * size;
  if (align < sizeof(void *))
    align = sizeof(void *);
  size_t rem = total % align;
  if (rem)
    total += align - rem;
  if (posix_memalign(&p, align, total) != 0)
    return NULL;
  memset(p, 0, total);
  return p;
}
void arts_free(void *ptr) { free(ptr); }
uint64_t arts_atomic_cswap_u64(volatile uint64_t *destination, uint64_t old_val,
                               uint64_t swap_in) {
  __atomic_compare_exchange_n(destination, &old_val, swap_in, false,
                              __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  return old_val;
}
