/* SPDX-License-Identifier: Apache-2.0
 *
 * Marked-next linked list implementation.  See header for the contract.
 */

#include "arts/utils/marked_list.h"

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

#include "arts/utils/atomics.h"

/* Tagged-pointer encoding of the .next word. */
#define MARKED_LINK_MARK_BIT ((uint64_t)1 << 63)
#define MARKED_LINK_GEN_SHIFT 48
#define MARKED_LINK_GEN_BITS 15
#define MARKED_LINK_GEN_MAX ((uint64_t)0x7FFF)
#define MARKED_LINK_GEN_MASK (MARKED_LINK_GEN_MAX << MARKED_LINK_GEN_SHIFT)
#define MARKED_LINK_PTR_MASK (((uint64_t)1 << 48) - 1)

static inline uint64_t link_pack(bool mark, uint16_t gen,
                                 const arts_marked_list_node_t *ptr) {
  return (mark ? MARKED_LINK_MARK_BIT : 0) |
         (((uint64_t)gen & MARKED_LINK_GEN_MAX) << MARKED_LINK_GEN_SHIFT) |
         ((uintptr_t)ptr & MARKED_LINK_PTR_MASK);
}

static inline arts_marked_list_node_t *link_ptr(uint64_t word) {
  return (arts_marked_list_node_t *)(uintptr_t)(word & MARKED_LINK_PTR_MASK);
}

static inline bool link_is_marked(uint64_t word) {
  return (word & MARKED_LINK_MARK_BIT) != 0;
}

static inline uint16_t link_gen(uint64_t word) {
  return (uint16_t)((word >> MARKED_LINK_GEN_SHIFT) & MARKED_LINK_GEN_MAX);
}

static inline uint16_t link_gen_inc(uint64_t word) {
  /* Wrap is OK — counter is 15-bit but the witness is the full word
   * (mark+gen+ptr); a node recycled to the same address with the same
   * (gen mod 2^15) AND the same mark bit AND the same ptr would be
   * required to fool the CAS, which is astronomically improbable
   * within any realistic operation window. */
  return (uint16_t)((link_gen(word) + 1) & MARKED_LINK_GEN_MAX);
}

/* The recycle pool stores marked-list nodes via the same 8-byte .next
 * slot, but the encoding differs: chain-time uses [mark:1|gen:15|ptr:48]
 * and pool-time uses [gen:15|<unused:1>|ptr:48] where the low 48 bits
 * are the next-pool-node pointer and the high 16 bits *preserve* the
 * chain-time generation across the pool cycle.  This preservation is
 * load-bearing — without it, a recycled node's .next would have gen=0
 * after pool transit, exposing an ABA window where a stale chain
 * reader's CAS witness matches the freshly-pushed node (same gen=k
 * after push's link_gen_inc(0)=1 stamp coincidentally equals the
 * stale gen for some k=1).  Reusing the existing high bits as a gen
 * carrier costs nothing and closes that window.
 *
 * We therefore implement a marked-list-private push/pop pair that
 * shares Treiber-stack mechanics with arts_lockfree_stack but encodes
 * the next-pool-node pointer in the low 48 bits only, leaving high 16
 * untouched as the chain-time witness. */
#define POOL_PTR_MASK MARKED_LINK_PTR_MASK
#define POOL_HIGH_MASK (~POOL_PTR_MASK)

static void pool_push(arts_lockfree_stack_t *pool,
                      arts_marked_list_node_t *node) {
  for (;;) {
    uint64_t old_top = pool->top;
    /* low 48 of top = current pool head ptr. */
    uintptr_t pool_head = (uintptr_t)(old_top & POOL_PTR_MASK);
    /* Update node->next: keep its high 16 bits (preserved chain gen),
     * replace low 48 with pool_head. */
    uint64_t old_node_next = node->next;
    node->next = (old_node_next & POOL_HIGH_MASK) | (pool_head & POOL_PTR_MASK);
    /* Stack top tagging — same scheme as arts_lockfree_stack: bump the
     * 16-bit counter on every CAS, store node ptr in low 48. */
    uint16_t counter = (uint16_t)(old_top >> 48);
    uint64_t new_top =
        (((uint64_t)(counter + 1)) << 48) | ((uintptr_t)node & POOL_PTR_MASK);
    if (arts_atomic_cswap_u64(&pool->top, old_top, new_top) == old_top) {
      return;
    }
  }
}

static arts_marked_list_node_t *pool_pop(arts_lockfree_stack_t *pool) {
  for (;;) {
    uint64_t old_top = pool->top;
    arts_marked_list_node_t *node =
        (arts_marked_list_node_t *)(uintptr_t)(old_top & POOL_PTR_MASK);
    if (node == NULL) {
      return NULL;
    }
    /* Read node->next; low 48 is the next-pool-node pointer. */
    uint64_t node_next_word = node->next;
    uintptr_t next_pool = (uintptr_t)(node_next_word & POOL_PTR_MASK);
    uint16_t counter = (uint16_t)(old_top >> 48);
    uint64_t new_top =
        (((uint64_t)(counter + 1)) << 48) | (next_pool & POOL_PTR_MASK);
    if (arts_atomic_cswap_u64(&pool->top, old_top, new_top) == old_top) {
      return node;
    }
  }
}

void arts_marked_list_init(arts_marked_list_t *list, size_t element_size) {
  /* head.next initially points to tail; tail.next always 0. */
  list->head.next = link_pack(false, 0, &list->tail);
  list->tail.next = 0;
  arts_lockfree_stack_init(&list->recycle);
  list->element_size = element_size;
  /* Caller's struct must put arts_marked_list_node_t at offset 0. */
  assert(element_size >= sizeof(arts_marked_list_node_t));
}

void arts_marked_list_destroy(arts_marked_list_t *list) {
  /* Walk the chain head→tail and free every node we encounter.  This
   * includes nodes that are marked-but-not-physically-unlinked — they
   * are still attached to some prev's .next slot, so we visit them too.
   * Sentinels are inside the list struct so we skip them. */
  arts_marked_list_node_t *cur = link_ptr(list->head.next);
  while (cur != NULL && cur != &list->tail) {
    arts_marked_list_node_t *next = link_ptr(cur->next);
    free(cur);
    cur = next;
  }
  /* Drain the private pool. */
  arts_marked_list_node_t *p;
  while ((p = pool_pop(&list->recycle)) != NULL) {
    free(p);
  }
  /* Reset head/tail (defensive — caller may zero-init the struct after
   * destroy and reuse the storage). */
  list->head.next = 0;
  list->tail.next = 0;
}

arts_marked_list_node_t *arts_marked_list_alloc(arts_marked_list_t *list) {
  arts_marked_list_node_t *node = pool_pop(&list->recycle);
  if (node != NULL) {
    return node;
  }
  /* Cold-start path: malloc one element_size buffer.  Caller's payload
   * fields beyond the .next word are uninitialized; caller fills them
   * before push. */
  return (arts_marked_list_node_t *)calloc(1, list->element_size);
}

void arts_marked_list_push(arts_marked_list_t *list,
                           arts_marked_list_node_t *node) {
  /* Insert at head: between head and head.next. */
  for (;;) {
    uint64_t old_head_next = list->head.next;
    /* head.next must never be marked (head sentinel is permanent). */
    assert(!link_is_marked(old_head_next));
    /* Wire incoming node to point at whatever currently follows head.
     *
     * IMPORTANT: when a node has been recycled from the private pool,
     * its `.next` slot already carries history from prior chain
     * cycles.  Resetting the slot's generation to 0 opens an ABA
     * window — a stale reader holding (unmarked, gen=0, oldY) could
     * match the recycled (unmarked, gen=0, newY) if the head's
     * successor happens to coincide (oldY == newY), letting a
     * spurious mark CAS sneak through.  Instead, advance the slot's
     * generation monotonically: bump the existing gen and combine
     * with the new mark+ptr.  Cold-start nodes get gen=1 on first
     * push, which is fine because no prior reader could have
     * captured gen=0 from this slot. */
    uint64_t old_node_next = node->next;
    node->next =
        link_pack(false, link_gen_inc(old_node_next), link_ptr(old_head_next));
    uint64_t new_head_next =
        link_pack(false, link_gen_inc(old_head_next), node);
    if (arts_atomic_cswap_u64(&list->head.next, old_head_next, new_head_next) ==
        old_head_next) {
      return;
    }
  }
}

bool arts_marked_list_mark(arts_marked_list_node_t *node) {
  for (;;) {
    uint64_t old_next = node->next;
    if (link_is_marked(old_next)) {
      return false; /* already claimed by another thread */
    }
    uint64_t new_next =
        link_pack(true, link_gen_inc(old_next), link_ptr(old_next));
    if (arts_atomic_cswap_u64(&node->next, old_next, new_next) == old_next) {
      return true;
    }
    /* CAS failed because either (a) another thread marked first, or
     * (b) someone updated gen/ptr concurrently (e.g., a helper unlinked
     * the node that follows ours).  Retry — re-read node->next. */
  }
}

void arts_marked_list_traverse(arts_marked_list_t *list,
                               arts_marked_list_visit_fn_t visit, void *ctx) {
  /* Restart-on-CAS-failure traversal.  The walk pattern:
   *
   *   prev = &head
   *   prev_word = head.next
   *   loop:
   *     cur = ptr(prev_word)
   *     if cur in {NULL, &tail}: done
   *     cur_word = cur.next
   *     if cur is marked (via cur.next mark bit):
   *         CAS(prev.next, prev_word, unmark(cur_word) + new gen)
   *         on success: push cur to recycle, advance prev_word to new
   *         on failure: restart from head
   *     else:
   *         visit(cur)
   *         re-read cur.next (visit may have marked)
   *         if marked now: try unlink (same CAS pattern)
   *         else: advance prev = cur, prev_word = cur_word
   */
  /* Two-pass strategy.
   *
   * Pass 1 (visit): walk head→tail and call `visit` on every unmarked
   *   node.  We DO NOT modify the chain here — no helping unlink — so
   *   each visit's caller (typically marked_list_mark via the user
   *   callback) operates on the node as-is.  This makes pass 1 safe
   *   under concurrent unlink: even if another helper unlinks our cur
   *   between read and visit, the worst case is a redundant visit on
   *   a node that's already been pulled out (visit's mark CAS will
   *   simply observe marked → false → no count++).
   *
   * Pass 2 (cleanup): walk again from head and physically unlink any
   *   marked node we encounter.  Each unlink is a single CAS at
   *   prev->next; failure → re-read prev->next and retry on the SAME
   *   prev (restart-from-head would livelock under a hot pusher).
   *
   * Splitting visit and unlink eliminates the trickiest race in the
   * original interleaved design: the "prev becomes a former cur, then
   * gets unlinked + repushed by another helper" scenario, which left
   * our prev pointing into the pool's chain (and made cur derefence a
   * non-chain node).  In the two-pass version, pass 1 only reads
   * cur->next via cur (a stable pointer once we've reached it) and
   * never advances prev=cur, so no advance-into-the-pool window
   * exists.  Pass 2's prev tracking is local to the unlink loop and
   * uses the same prev=&head reset on logical-delete detection. */

  /* ---- Pass 1: visit every unmarked node observed during one walk. */
  arts_marked_list_node_t *cur = link_ptr((uint64_t)list->head.next);
  while (cur != NULL && cur != &list->tail) {
    uint64_t cur_word = cur->next;
    if (!link_is_marked(cur_word)) {
      visit(cur, ctx);
    }
    /* Advance using the snapshot taken before visit.  If cur was just
     * marked + unlinked by another helper, link_ptr(cur_word) still
     * points to what was cur's chain successor (an in-pool node never
     * has its high-bit chain meta reset until re-pushed, but link_ptr
     * extracts only the low-48 successor pointer regardless of pool /
     * chain state — a concurrent recycle replaces the low 48 with a
     * pool node, but we already snapshotted cur_word above).  Worst
     * case we follow a stale pointer to either (a) a still-in-chain
     * node we re-visit harmlessly, (b) a recycled node whose mark CAS
     * fails (pool node has chain mark=true preserved), or (c) the
     * tail sentinel and exit. */
    cur = link_ptr(cur_word);
  }

  /* ---- Pass 2: cleanup — physically unlink marked nodes. */
  arts_marked_list_node_t *prev = &list->head;
  uint64_t prev_word = prev->next;
  while (1) {
    if (link_is_marked(prev_word)) {
      /* prev itself was unlinked by another cleanup helper. */
      prev = &list->head;
      prev_word = prev->next;
      continue;
    }
    arts_marked_list_node_t *c = link_ptr(prev_word);
    if (c == NULL || c == &list->tail) {
      return;
    }
    uint64_t c_word = c->next;
    if (link_is_marked(c_word)) {
      uint64_t new_prev_word =
          link_pack(false, link_gen_inc(prev_word), link_ptr(c_word));
      if (arts_atomic_cswap_u64(&prev->next, prev_word, new_prev_word) ==
          prev_word) {
        pool_push(&list->recycle, c);
        prev_word = new_prev_word;
        continue;
      }
      /* CAS lost — re-read prev->next, retry on same prev. */
      prev_word = prev->next;
      continue;
    }
    /* Unmarked node — leave for the next round to visit/mark/unlink. */
    prev = c;
    prev_word = prev->next;
  }
}
