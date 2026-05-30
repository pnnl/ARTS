/* SPDX-License-Identifier: Apache-2.0
 *
 * Concurrent correctness for the Vyukov arts_mpsc_t: N producers each push
 * K tagged nodes; one drainer pops them all.  Asserts (1) no message is
 * lost or duplicated (exactly N*K popped) and (2) per-producer FIFO order
 * is preserved (each producer's sequence numbers come out monotonically).
 *
 * Standalone — the queue is header-only inline, so this needs no library.
 */

#include "arts/sync/mpsc.h"

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PRODUCERS 4
#define PER_PRODUCER 200000
#define TOTAL ((size_t)PRODUCERS * PER_PRODUCER)

typedef struct {
  arts_lf_link_t link; /* first member — recover via cast */
  uint32_t producer;
  uint32_t seq;
} msg_t;

static arts_mpsc_t g_q;
static _Atomic(int) g_prod_done; /* producers that have finished pushing */

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

int main(void) {
  arts_mpsc_init(&g_q);

  pthread_t prod[PRODUCERS];
  for (int i = 0; i < PRODUCERS; ++i)
    pthread_create(&prod[i], NULL, producer, (void *)(uintptr_t)i);

  /* Single drainer: hold the gate, pop until we have every message.  pop()
   * may transiently return NULL while a producer is mid-link, so loop until
   * the full count is collected. */
  uint32_t next_seq[PRODUCERS];
  for (int i = 0; i < PRODUCERS; ++i)
    next_seq[i] = 0;
  size_t got = 0;
  int idle_after_done = 0;
  while (got < TOTAL) {
    bool drainer = arts_mpsc_try_drain_begin(&g_q);
    assert(drainer); /* single drainer — always wins */
    size_t before = got;
    for (;;) {
      arts_lf_link_t *n = arts_mpsc_pop(&g_q);
      if (!n)
        break;
      msg_t *m = (msg_t *)n; /* link is first member */
      assert(m->producer < PRODUCERS);
      assert(m->seq == next_seq[m->producer]); /* per-producer FIFO */
      next_seq[m->producer] = m->seq + 1;
      free(m);
      ++got;
    }
    arts_mpsc_drain_end(&g_q);
    /* Termination guard: once every producer has finished, a drain round
     * that makes no progress means a broken (lossy) queue — bail so the
     * final tally assertion fails instead of spinning forever. */
    if (atomic_load_explicit(&g_prod_done, memory_order_acquire) == PRODUCERS) {
      if (got == before) {
        if (++idle_after_done > 1000)
          break;
      } else {
        idle_after_done = 0;
      }
    }
  }

  for (int i = 0; i < PRODUCERS; ++i)
    pthread_join(prod[i], NULL);

  /* Nothing left behind. */
  assert(arts_mpsc_try_drain_begin(&g_q));
  assert(arts_mpsc_pop(&g_q) == NULL);
  arts_mpsc_drain_end(&g_q);

  assert(got == TOTAL); /* no lost or duplicated messages */
  for (int i = 0; i < PRODUCERS; ++i)
    assert(next_seq[i] == PER_PRODUCER);

  printf("mpsc_concurrent: drained %zu messages, per-producer FIFO OK\n", got);
  printf("mpsc_concurrent: OK\n");
  return 0;
}
