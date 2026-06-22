/* SPDX-License-Identifier: Apache-2.0
 *
 * T010 — arts_mpsc_pop INV-5: stub re-thread on the single-remaining-node
 * path (mpsc.h).  Census 29.md §5 GAP / INV-5.
 *
 * When the consumer pops down to exactly one node (tail == head), arts_mpsc_pop
 * re-pushes the embedded stub (arts_mpsc_push(&q->stub)) so the last node
 * becomes poppable, then re-reads `next`.  This re-thread is the subtlest path
 * in the file and is only incidentally exercised whenever the queue drains to
 * exactly one node under contention.  A bug here (stub + the new node not both
 * correctly linked when the re-thread races a producer xchg) shows up as a
 * lost or duplicated node, or the stub leaking into the popped stream.
 *
 * To MAXIMIZE the hit rate on this path we keep the queue NARROW: a single
 * consumer pops as fast as it can while producers push, so the queue spends a
 * large fraction of its time at depth 0/1 and the tail==head branch fires
 * constantly.  Correctness is checked the same way as the general MPSC test
 * (no loss / no dup / per-producer FIFO) PLUS an explicit guard that the stub
 * pointer is never returned by pop.
 */

#include "arts/utils/mpsc.h"

#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PRODUCERS 4
#define PER_PRODUCER 300000
#define TOTAL ((size_t)PRODUCERS * PER_PRODUCER)

typedef struct {
  arts_lf_link_t link;
  uint32_t producer;
  uint32_t seq;
} msg_t;

static arts_mpsc_t g_q;
static atomic_int g_start;
static _Atomic int g_prod_done;

static void *producer(void *arg) {
  uint32_t id = (uint32_t)(uintptr_t)arg;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  for (uint32_t s = 0; s < PER_PRODUCER; ++s) {
    msg_t *m = (msg_t *)malloc(sizeof(msg_t));
    m->producer = id;
    m->seq = s;
    arts_mpsc_push(&g_q, &m->link);
    /* Yield occasionally so the consumer can drain to depth<=1, forcing the
     * tail==head stub re-thread path. */
    if ((s & 0x3F) == 0) {
      sched_yield();
    }
  }
  atomic_fetch_add_explicit(&g_prod_done, 1, memory_order_release);
  return NULL;
}

int main(void) {
  arts_mpsc_init(&g_q);
  atomic_init(&g_start, 0);
  atomic_init(&g_prod_done, 0);

  pthread_t prod[PRODUCERS];
  for (int i = 0; i < PRODUCERS; ++i) {
    pthread_create(&prod[i], NULL, producer, (void *)(uintptr_t)i);
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);

  /* Single consumer (holds the gate the whole time) — pops one at a time so
   * the queue is frequently at depth 0/1. */
  uint32_t next_seq[PRODUCERS];
  for (int i = 0; i < PRODUCERS; ++i) {
    next_seq[i] = 0;
  }
  size_t got = 0;
  int idle = 0;
  bool gate = arts_mpsc_try_drain_begin(&g_q);
  if (!gate) {
    (void)fprintf(stderr, "FAIL mpsc_rethread_stub: lost own gate\n");
    return 1;
  }
  while (got < TOTAL) {
    arts_lf_link_t *l = arts_mpsc_pop(&g_q);
    if (!l) {
      if (atomic_load_explicit(&g_prod_done, memory_order_acquire) ==
              PRODUCERS &&
          ++idle > 2000000) {
        break; /* broken/lossy queue — fail the tally below */
      }
      continue;
    }
    idle = 0;
    if (l == &g_q.stub) {
      (void)fprintf(stderr,
                    "FAIL mpsc_rethread_stub: stub returned from pop\n");
      return 1;
    }
    msg_t *m = (msg_t *)l;
    if (m->producer >= PRODUCERS || m->seq != next_seq[m->producer]) {
      (void)fprintf(
          stderr, "FAIL mpsc_rethread_stub: order break p=%u seq=%u want %u\n",
          m->producer, m->seq,
          m->producer < PRODUCERS ? next_seq[m->producer] : 0);
      return 1;
    }
    next_seq[m->producer] = m->seq + 1;
    free(m);
    ++got;
  }
  arts_mpsc_drain_end(&g_q);

  for (int i = 0; i < PRODUCERS; ++i) {
    pthread_join(prod[i], NULL);
  }

  if (got != TOTAL) {
    (void)fprintf(stderr, "FAIL mpsc_rethread_stub: got %zu of %zu\n", got,
                  TOTAL);
    return 1;
  }
  for (int i = 0; i < PRODUCERS; ++i) {
    if (next_seq[i] != PER_PRODUCER) {
      (void)fprintf(stderr, "FAIL mpsc_rethread_stub: producer %d seq %u\n", i,
                    next_seq[i]);
      return 1;
    }
  }

  printf("PASS mpsc_rethread_stub: %zu nodes via depth<=1 churn, stub never "
         "popped, per-producer FIFO held\n",
         TOTAL);
  return 0;
}
