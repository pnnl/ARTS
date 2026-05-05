/******************************************************************************
** This material was prepared as an account of work sponsored by an agency   **
** of the United States Government.  Neither the United States Government    **
** nor the United States Department of Energy, nor Battelle, nor any of      **
** their employees, nor any jurisdiction or organization that has cooperated **
** in the development of these materials, makes any warranty, express or     **
** implied, or assumes any legal liability or responsibility for the accuracy,*
** completeness, or usefulness or any information, apparatus, product,       **
** software, or process disclosed, or represents that its use would not      **
** infringe privately owned rights.                                          **
**                                                                           **
** Reference herein to any specific commercial product, process, or service  **
** by trade name, trademark, manufacturer, or otherwise does not necessarily **
** constitute or imply its endorsement, recommendation, or favoring by the   **
** United States Government or any agency thereof, or Battelle Memorial      **
** Institute. The views and opinions of authors expressed herein do not      **
** necessarily state or reflect those of the United States Government or     **
** any agency thereof.                                                       **
**                                                                           **
**                      PACIFIC NORTHWEST NATIONAL LABORATORY                **
**                                  operated by                              **
**                                    BATTELLE                               **
**                                     for the                               **
**                      UNITED STATES DEPARTMENT OF ENERGY                   **
**                         under Contract DE-AC05-76RL01830                  **
**                                                                           **
** Copyright 2019 Battelle Memorial Institute                                **
** Licensed under the Apache License, Version 2.0 (the "License");           **
** you may not use this file except in compliance with the License.          **
** You may obtain a copy of the License at                                   **
**                                                                           **
**    https://www.apache.org/licenses/LICENSE-2.0                            **
**                                                                           **
** Unless required by applicable law or agreed to in writing, software       **
** distributed under the License is distributed on an "AS IS" BASIS, WITHOUT **
** WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the  **
** License for the specific language governing permissions and limitations   **
******************************************************************************/
/* Vyukov MPSC queue.  Replaces the prior Treiber-stack-with-reverse OO
 * list.  Pure FIFO, lock-free producer side, single consumer.
 *
 * Algorithm (Vyukov, intrusive single-linked list with permanent stub):
 *
 *   init():
 *     stub.next = NULL
 *     head = tail = &stub
 *
 *   push(data):                            // multi-producer
 *     node = malloc; node->data = data; node->next = NULL
 *     prev = atomic_xchg_acq_rel(tail, node)
 *     atomic_store_release(prev->next, node)
 *
 *   pop()/drain():                         // single consumer
 *     head = list->head
 *     next = atomic_load_acquire(head->next)
 *     if next == NULL:
 *       if atomic_load_acquire(tail) == head: empty -> done
 *       else: producer mid-push -> spin briefly + retry
 *     // process next->data
 *     list->head = next                    // plain store: single consumer
 *     if head != &list->stub: free(head)   // first iteration's old head IS
 *                                          // the embedded stub — never free
 *
 * The stub is "consumed" on the first push: after the xchg, tail points
 * at the new node, and the next pop sees stub.next pointing at it.  The
 * old head (the stub) is then advanced past, and from then on every
 * "old head" we walk past is a malloc'd node — which we DO free.
 *
 * Memory ordering proof sketch:
 *   - producer: xchg(tail) is acq_rel — the prev pointer it returns is
 *     an exclusive handle no one else has.  store_release on prev->next
 *     publishes the link.
 *   - consumer: load_acquire on head->next pairs with the producer's
 *     store_release on prev->next.  After observing a non-NULL next,
 *     the data field of *next was written before the producer's
 *     atomic_xchg (program order on the producer), and the xchg
 *     synchronizes-with the chain of acquire loads, so the consumer
 *     sees data correctly.
 *   - tail-equality empty check: load_acquire pairs with the producer's
 *     atomic_xchg on tail.  If we observe tail == head AFTER a NULL
 *     head->next, no producer has begun a push since head was last
 *     advanced — truly empty.
 */
#include "arts/gas/out_of_order_list.h"

#include <sched.h>
#include <stdatomic.h>
#include <stddef.h>

#include "arts/utils/malloc.h"

void arts_oo_list_init(struct arts_oo_list_s *list) {
  atomic_store_explicit(&list->stub.next, (struct arts_oo_node_s *)NULL,
                        memory_order_relaxed);
  list->stub.data = NULL;
  atomic_store_explicit(&list->head, &list->stub, memory_order_relaxed);
  atomic_store_explicit(&list->tail, &list->stub, memory_order_relaxed);
}

oo_push_result_t arts_oo_list_push(struct arts_oo_list_s *list, void *data) {
  struct arts_oo_node_s *node =
      (struct arts_oo_node_s *)arts_malloc(sizeof(*node));
  node->data = data;
  atomic_store_explicit(&node->next, (struct arts_oo_node_s *)NULL,
                        memory_order_relaxed);
  /* Multi-producer: claim a slot in the chain via atomic_xchg on tail.
   * The previous tail value is our exclusive predecessor — no other
   * producer can observe it again (since tail now points to us). */
  struct arts_oo_node_s *prev =
      atomic_exchange_explicit(&list->tail, node, memory_order_acq_rel);
  /* Publish the link.  The store_release pairs with the consumer's
   * atomic_load_acquire on head->next when it walks past `prev`. */
  atomic_store_explicit(&prev->next, node, memory_order_release);
  return OO_PUSH_OK;
}

/* Helper: advance the consumer's head by one step; returns the data
 * payload via *out_data, or false if the queue is observed empty (or
 * a producer is mid-push and we've spun out our retry budget).
 *
 * The "mid-push" case is handled by a brief sched_yield-driven retry
 * loop — drain is called from a single consumer thread and producer
 * windows close in nanoseconds, so a tight retry suffices.
 */
static bool oo_pop_step(struct arts_oo_list_s *list, void **out_data,
                        struct arts_oo_node_s **out_old_head) {
  for (;;) {
    struct arts_oo_node_s *head =
        atomic_load_explicit(&list->head, memory_order_relaxed);
    struct arts_oo_node_s *next =
        atomic_load_explicit(&head->next, memory_order_acquire);
    if (next == NULL) {
      struct arts_oo_node_s *tail =
          atomic_load_explicit(&list->tail, memory_order_acquire);
      if (tail == head) {
        return false; /* truly empty */
      }
      /* Producer mid-push: yield and retry.  Window is the gap between
       * xchg(tail) and store_release(prev->next) on the producer. */
      sched_yield();
      continue;
    }
    *out_data = next->data;
    /* Single consumer: plain store on head is correct.  We use a
     * relaxed atomic store so other consumers (none in steady state, but
     * potential debug paths) at least see a consistent value.  Memory
     * ordering on data was already established by load_acquire on next. */
    atomic_store_explicit(&list->head, next, memory_order_relaxed);
    *out_old_head = head;
    return true;
  }
}

void arts_oo_list_drain(struct arts_oo_list_s *list,
                        void (*callback)(void *data, void *ctx), void *ctx) {
  void *data;
  struct arts_oo_node_s *old_head;
  while (oo_pop_step(list, &data, &old_head)) {
    callback(data, ctx);
    /* The very first pop's old head is the embedded stub — never free
     * that.  Every subsequent old head is a malloc'd node from a
     * previous push and must be freed here. */
    if (old_head != &list->stub) {
      arts_free(old_head);
    }
  }
}

void arts_oo_list_drop_all(struct arts_oo_list_s *list) {
  void *data;
  struct arts_oo_node_s *old_head;
  while (oo_pop_step(list, &data, &old_head)) {
    arts_free(data); /* destroy path: payload free */
    if (old_head != &list->stub) {
      arts_free(old_head);
    }
  }
}
