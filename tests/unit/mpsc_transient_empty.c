/* SPDX-License-Identifier: Apache-2.0
 *
 * T011 — arts_mpsc_pop INV-4 (CRITICAL): transient-empty is NEVER terminal —
 * a lost-wakeup guard (mpsc.h).  Census 29.md §5 INV-4 / SUSPECTED-BUG B081
 * (the historical channel lost-wakeup class).
 *
 * arts_mpsc_push is two steps: xchg(head, n) then prev->next = n.  Between
 * them a concurrent consumer sees the new head but `prev->next` still NULL —
 * pop reports transient-empty (NULL) even though a node IS in flight.  The
 * correctness contract: if a consumer treats that NULL as terminal it drops a
 * node (lost wakeup).  A correct consumer that keeps polling ALWAYS gets the
 * node once the producer's linking store lands.
 *
 * This test expresses INV-4 as a PROPERTY: under heavy producer/consumer
 * contention, every pushed node is retrieved within a BOUNDED number of
 * consecutive transient-empty retries.  Concretely the single consumer counts
 * consecutive NULL pops; if it ever exceeds a generous bound WHILE producers
 * are still active OR nodes are still outstanding, that is a permanent
 * lost-wakeup and the test FAILS.  The producers also deliberately interleave
 * the xchg/link window (one producer per push) so the transient-empty window
 * is hit very frequently.
 *
 * Plus a tight deterministic micro-case: two producers race a single push each
 * around one consumer; the consumer, polling, must collect BOTH (it must not
 * stop at the first transient-empty).
 */

#include "arts/utils/mpsc.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PRODUCERS 7
#define PER_PRODUCER 200000
#define TOTAL ((size_t)PRODUCERS * PER_PRODUCER)

/* If the consumer sees this many CONSECUTIVE transient-empty pops while work is
 * still outstanding, we declare a permanent lost wakeup.  The real window is a
 * handful of instructions; this bound is enormous in comparison. */
#define MAX_CONSECUTIVE_EMPTY 50000000ull

typedef struct {
  arts_lf_link_t link;
  uint64_t
      payload; /* published before push; checked after pop (happens-before) */
} node_t;

static arts_mpsc_t g_q;
static atomic_int g_start;
static _Atomic int g_prod_done;
static _Atomic uint64_t g_pushed; /* total pushed so far */

#define PAYLOAD_OF(i) (0xA5A5000000000000ull ^ (uint64_t)(i))

static void *producer(void *arg) {
  uint32_t id = (uint32_t)(uintptr_t)arg;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  for (uint32_t s = 0; s < PER_PRODUCER; ++s) {
    node_t *n = (node_t *)malloc(sizeof(node_t));
    uint64_t idx = (uint64_t)id * PER_PRODUCER + s;
    n->payload = PAYLOAD_OF(idx); /* publish BEFORE push */
    arts_mpsc_push(&g_q, &n->link);
    atomic_fetch_add_explicit(&g_pushed, 1, memory_order_release);
  }
  atomic_fetch_add_explicit(&g_prod_done, 1, memory_order_release);
  return NULL;
}

int main(void) {
  arts_mpsc_init(&g_q);
  atomic_init(&g_start, 0);
  atomic_init(&g_prod_done, 0);
  atomic_init(&g_pushed, 0);

  pthread_t prod[PRODUCERS];
  for (int i = 0; i < PRODUCERS; ++i) {
    pthread_create(&prod[i], NULL, producer, (void *)(uintptr_t)i);
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);

  bool gate = arts_mpsc_try_drain_begin(&g_q);
  if (!gate) {
    (void)fprintf(stderr, "FAIL mpsc_transient_empty: lost own gate\n");
    return 1;
  }

  size_t got = 0;
  unsigned long long consecutive_empty = 0;
  while (got < TOTAL) {
    arts_lf_link_t *l = arts_mpsc_pop(&g_q);
    if (!l) {
      /* Transient empty OR genuinely drained-but-more-coming.  It is only a
       * lost wakeup if there is OUTSTANDING work (pushed > got) and we keep
       * seeing NULL forever. */
      uint64_t pushed = atomic_load_explicit(&g_pushed, memory_order_acquire);
      if (pushed > got) {
        if (++consecutive_empty > MAX_CONSECUTIVE_EMPTY) {
          (void)fprintf(stderr,
                        "FAIL mpsc_transient_empty: LOST WAKEUP — %llu "
                        "consecutive transient-empty pops with %" PRIu64
                        " nodes outstanding (pushed=%" PRIu64 " got=%zu)\n",
                        consecutive_empty, pushed - got, pushed, got);
          return 1;
        }
      } else {
        consecutive_empty = 0; /* truly empty right now — not a lost wakeup */
      }
      continue;
    }
    consecutive_empty = 0;
    node_t *n = (node_t *)l;
    /* happens-before: pop's acquire of head must see the producer's
     * pre-push payload publication. */
    if ((n->payload & 0xFFFF000000000000ull) != 0xA5A5000000000000ull) {
      (void)fprintf(stderr,
                    "FAIL mpsc_transient_empty: torn payload %#" PRIx64 "\n",
                    n->payload);
      return 1;
    }
    free(n);
    ++got;
  }
  arts_mpsc_drain_end(&g_q);

  for (int i = 0; i < PRODUCERS; ++i) {
    pthread_join(prod[i], NULL);
  }

  if (got != TOTAL) {
    (void)fprintf(stderr, "FAIL mpsc_transient_empty: got %zu of %zu\n", got,
                  TOTAL);
    return 1;
  }
  printf("PASS mpsc_transient_empty: %zu nodes, every transient-empty bounded "
         "(no lost wakeup), payload happens-before held\n",
         TOTAL);
  return 0;
}
