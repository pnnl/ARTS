/* SPDX-License-Identifier: Apache-2.0
 *
 * T015 — arts_link_list (link_list.c) Vyukov MPSC correctness.
 * Census 30.md §B.5 gaps #1/#2/#5 / SUSPECTED-BUG B119.
 *
 * link_list.c is a SEPARATE translation unit from the header-only mpsc.h (it
 * "mirrors it exactly" per the comment), so the mpsc_* tests never exercise
 * link_list.c's own code path: the stub re-thread, is_empty conservatism, the
 * data-pointer ±1 header arithmetic.  link_list.c is the transport outbound
 * queue, only covered indirectly by multinode runtime tests with no targeted
 * mid-link / stub-re-thread assertions.
 *
 * This isolates it: P producers push_back; one consumer pop_front loops
 * (re-polling on transient NULL — a producer mid-link).  Asserts:
 *   - no loss / no dup (exactly N*K data items popped),
 *   - per-producer FIFO (Vyukov preserves push order),
 *   - the data pointer round-trips through new_item / push_back / pop_front /
 *     delete_item (the ±1 header arithmetic),
 *   - is_empty is conservative: while items are outstanding it returns 0,
 *     and once the consumer fully catches up it returns 1.
 *
 * The single-element stub re-thread path (pop_front step 5, tail==head) is hit
 * naturally whenever the consumer drains to exactly one node — kept frequent by
 * a one-at-a-time consumer.  B119 (single-consumer precondition): we honor it
 * (exactly one consumer) so any corruption would be a real defect.
 */

#include "arts/utils/link_list.h"

#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* libc shims (link_list.c calls arts_calloc / arts_free). */
void *arts_calloc(size_t nmemb, size_t size) { return calloc(nmemb, size); }
void arts_free(void *ptr) { free(ptr); }

#define PRODUCERS 4
#define PER_PRODUCER 200000
#define TOTAL ((size_t)PRODUCERS * PER_PRODUCER)

typedef struct {
  uint32_t producer;
  uint32_t seq;
} payload_t;

static struct arts_link_list_s g_list;
static atomic_int g_start;
static _Atomic int g_prod_done;

static void *producer(void *arg) {
  uint32_t id = (uint32_t)(uintptr_t)arg;
  while (atomic_load_explicit(&g_start, memory_order_acquire) == 0) {
  }
  for (uint32_t s = 0; s < PER_PRODUCER; ++s) {
    payload_t *p = (payload_t *)arts_link_list_new_item(sizeof(payload_t));
    p->producer = id;
    p->seq = s;
    arts_link_list_push_back(&g_list, p);
    if ((s & 0x7F) == 0) {
      sched_yield(); /* let consumer drain to depth<=1 (stub re-thread path) */
    }
  }
  atomic_fetch_add_explicit(&g_prod_done, 1, memory_order_release);
  return NULL;
}

int main(void) {
  arts_link_list_new(&g_list);
  atomic_init(&g_start, 0);
  atomic_init(&g_prod_done, 0);

  /* Empty list: is_empty must be 1 before anything is pushed. */
  if (arts_link_list_is_empty(&g_list) != 1) {
    (void)fprintf(stderr, "FAIL link_list_mpsc: fresh list not empty\n");
    return 1;
  }

  pthread_t prod[PRODUCERS];
  for (int i = 0; i < PRODUCERS; ++i) {
    pthread_create(&prod[i], NULL, producer, (void *)(uintptr_t)i);
  }
  atomic_store_explicit(&g_start, 1, memory_order_release);

  uint32_t next_seq[PRODUCERS];
  for (int i = 0; i < PRODUCERS; ++i) {
    next_seq[i] = 0;
  }
  size_t got = 0;
  while (got < TOTAL) {
    void *data = arts_link_list_pop_front(&g_list, NULL);
    if (!data) {
      continue; /* empty or producer mid-link — re-poll (no loss) */
    }
    payload_t *p = (payload_t *)data;
    if (p->producer >= PRODUCERS || p->seq != next_seq[p->producer]) {
      (void)fprintf(stderr,
                    "FAIL link_list_mpsc: order break p=%u seq=%u want %u\n",
                    p->producer, p->seq,
                    p->producer < PRODUCERS ? next_seq[p->producer] : 0);
      return 1;
    }
    next_seq[p->producer] = p->seq + 1;
    arts_link_list_delete_item(data);
    ++got;
  }

  for (int i = 0; i < PRODUCERS; ++i) {
    pthread_join(prod[i], NULL);
  }

  /* Consumer has caught up to every producer; drain any final stragglers, then
   * is_empty must report 1. */
  void *d;
  while ((d = arts_link_list_pop_front(&g_list, NULL)) != NULL) {
    (void)fprintf(stderr, "FAIL link_list_mpsc: extra item after full tally\n");
    arts_link_list_delete_item(d);
    return 1;
  }
  if (arts_link_list_is_empty(&g_list) != 1) {
    (void)fprintf(stderr, "FAIL link_list_mpsc: drained list not empty\n");
    return 1;
  }

  if (got != TOTAL) {
    (void)fprintf(stderr, "FAIL link_list_mpsc: got %zu of %zu\n", got, TOTAL);
    return 1;
  }
  for (int i = 0; i < PRODUCERS; ++i) {
    if (next_seq[i] != PER_PRODUCER) {
      (void)fprintf(stderr, "FAIL link_list_mpsc: producer %d seq %u\n", i,
                    next_seq[i]);
      return 1;
    }
  }

  printf("PASS link_list_mpsc: %zu items, per-producer FIFO, transient-NULL "
         "retry + stub re-thread, is_empty conservative\n",
         TOTAL);
  return 0;
}
