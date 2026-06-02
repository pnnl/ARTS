/* SPDX-License-Identifier: Apache-2.0
 *
 * Drain-gate mutual exclusion for arts_mpsc_t.  Many contender threads race
 * arts_mpsc_try_drain_begin concurrently while producers push.  The gate
 * must admit exactly one drainer at a time: inside the critical section a
 * non-atomic sentinel is flipped and checked (ThreadSanitizer flags any
 * overlap), and an atomic active-drainer counter must read exactly 1.  All
 * pushed messages must still be drained exactly once.
 *
 * Standalone — header-only queue, no library needed.
 */

#include "arts/utils/mpsc.h"

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PRODUCERS 3
#define CONTENDERS 6
#define PER_PRODUCER 150000
#define TOTAL ((size_t)PRODUCERS * PER_PRODUCER)

typedef struct {
  arts_lf_link_t link;
  uint32_t producer;
  uint32_t seq;
} msg_t;

static arts_mpsc_t g_q;
static _Atomic(int) g_prod_done;
static _Atomic(size_t) g_got;
static _Atomic(int) g_active; /* drainers currently inside the gate */
static int g_sentinel;        /* non-atomic; touched only inside the gate */

static void *producer(void *arg) {
  uint32_t id = (uint32_t)(uintptr_t)arg;
  for (uint32_t s = 0; s < PER_PRODUCER; ++s) {
    msg_t *m = (msg_t *)malloc(sizeof(msg_t));
    m->producer = id;
    m->seq = s;
    arts_mpsc_push(&g_q, &m->link);
  }
  atomic_fetch_add_explicit(&g_prod_done, 1, memory_order_release);
  return NULL;
}

static void *contender(void *arg) {
  (void)arg;
  for (;;) {
    if (atomic_load_explicit(&g_got, memory_order_acquire) >= TOTAL &&
        atomic_load_explicit(&g_prod_done, memory_order_acquire) == PRODUCERS)
      break;
    if (!arts_mpsc_try_drain_begin(&g_q))
      continue; /* lost the gate — do NOT spin in the critical section */

    /* Critical section: exactly one drainer here. */
    int prev = atomic_fetch_add_explicit(&g_active, 1, memory_order_acq_rel);
    assert(prev == 0); /* no other drainer admitted */
    g_sentinel++;      /* non-atomic — TSan trips if two drainers overlap */
    int local_sentinel = g_sentinel;

    for (;;) {
      arts_lf_link_t *n = arts_mpsc_pop(&g_q);
      if (!n)
        break;
      msg_t *m = (msg_t *)n;
      free(m);
      atomic_fetch_add_explicit(&g_got, 1, memory_order_relaxed);
    }

    assert(g_sentinel == local_sentinel); /* nobody else mutated it */
    int after = atomic_fetch_sub_explicit(&g_active, 1, memory_order_acq_rel);
    assert(after == 1);
    arts_mpsc_drain_end(&g_q);
  }
  return NULL;
}

int main(void) {
  arts_mpsc_init(&g_q);

  pthread_t prod[PRODUCERS], cont[CONTENDERS];
  for (int i = 0; i < CONTENDERS; ++i)
    pthread_create(&cont[i], NULL, contender, NULL);
  for (int i = 0; i < PRODUCERS; ++i)
    pthread_create(&prod[i], NULL, producer, (void *)(uintptr_t)i);

  for (int i = 0; i < PRODUCERS; ++i)
    pthread_join(prod[i], NULL);
  for (int i = 0; i < CONTENDERS; ++i)
    pthread_join(cont[i], NULL);

  /* Final sweep in case the last messages landed after every contender's
   * exit check (producers-done observed before the final links were seen). */
  assert(arts_mpsc_try_drain_begin(&g_q));
  for (;;) {
    arts_lf_link_t *n = arts_mpsc_pop(&g_q);
    if (!n)
      break;
    free((msg_t *)n);
    atomic_fetch_add_explicit(&g_got, 1, memory_order_relaxed);
  }
  arts_mpsc_drain_end(&g_q);

  size_t got = atomic_load(&g_got);
  printf("mpsc_drain_race: drained %zu / %zu\n", got, TOTAL);
  assert(got == TOTAL);
  printf("mpsc_drain_race: OK\n");
  return 0;
}
