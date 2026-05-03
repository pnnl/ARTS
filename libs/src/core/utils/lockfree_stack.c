/* SPDX-License-Identifier: Apache-2.0
 *
 * Tagged-pointer Treiber stack implementation.  See header for design.
 */

#include "arts/utils/lockfree_stack.h"

#include <stddef.h>

#include "arts/utils/atomics.h"

/* 16-bit counter occupies the high bits, 48-bit pointer the low bits. */
#define STACK_PTR_BITS 48
#define STACK_PTR_MASK (((uint64_t)1 << STACK_PTR_BITS) - 1)

static inline uint64_t stack_pack(uint16_t counter,
                                  arts_lockfree_stack_node_t *ptr) {
  return ((uint64_t)counter << STACK_PTR_BITS) |
         ((uintptr_t)ptr & STACK_PTR_MASK);
}

static inline arts_lockfree_stack_node_t *stack_ptr(uint64_t word) {
  return (arts_lockfree_stack_node_t *)(uintptr_t)(word & STACK_PTR_MASK);
}

static inline uint16_t stack_counter(uint64_t word) {
  return (uint16_t)(word >> STACK_PTR_BITS);
}

void arts_lockfree_stack_init(arts_lockfree_stack_t *s) { s->top = 0; }

void arts_lockfree_stack_push(arts_lockfree_stack_t *s,
                              arts_lockfree_stack_node_t *node) {
  for (;;) {
    uint64_t old_top = s->top;
    /* Wire the incoming node to the current head BEFORE the CAS.  If our
     * CAS loses we'll re-read top and re-thread `node->next` — concurrent
     * pushes that observe `node` mid-update never read `node->next`
     * because the CAS publishing `node` to top is the linearization
     * point and happens after this assignment. */
    node->next = stack_ptr(old_top);
    uint64_t new_top = stack_pack((uint16_t)(stack_counter(old_top) + 1), node);
    if (arts_atomic_cswap_u64(&s->top, old_top, new_top) == old_top) {
      return;
    }
  }
}

arts_lockfree_stack_node_t *arts_lockfree_stack_pop(arts_lockfree_stack_t *s) {
  for (;;) {
    uint64_t old_top = s->top;
    arts_lockfree_stack_node_t *node = stack_ptr(old_top);
    if (node == NULL) {
      return NULL;
    }
    /* Read `node->next` BEFORE the CAS.  This read can race with another
     * popper that owns `node` after a successful CAS — but if their CAS
     * lands first our `old_top` becomes stale and our CAS will fail. */
    arts_lockfree_stack_node_t *next = node->next;
    uint64_t new_top = stack_pack((uint16_t)(stack_counter(old_top) + 1), next);
    if (arts_atomic_cswap_u64(&s->top, old_top, new_top) == old_top) {
      return node;
    }
  }
}

bool arts_lockfree_stack_empty(const arts_lockfree_stack_t *s) {
  return stack_ptr(s->top) == NULL;
}
