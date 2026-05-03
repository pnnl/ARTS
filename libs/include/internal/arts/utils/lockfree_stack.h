/* SPDX-License-Identifier: Apache-2.0
 *
 * Lock-free intrusive Treiber stack with tagged-pointer ABA defense.
 *
 * The `top` field packs [counter:16 | ptr:48] into a single 64-bit word so
 * push/pop linearize on one CAS each.  Counter increments on every CAS
 * that updates `top.ptr`, so a node recycled back to the same address
 * fails the witness because the counter advanced.
 *
 * Intrusive: caller owns node memory.  `arts_lockfree_stack_node_t` MUST
 * be the FIRST field of any struct that participates in this stack —
 * pop returns the node pointer and the caller container_of-casts back to
 * its own struct.  This avoids per-node malloc and lets the same buffer
 * type alternate between "live" and "pooled" states using a single
 * union-like overlay.
 *
 * Address space: assumes the platform uses 48-bit virtual addresses (true
 * for x86-64 and aarch64 user-space; sign-extended bit 47 is OK because
 * we always re-cast to a pointer through `(void *)`).  Heap addresses
 * fit cleanly in the low 48 bits.
 */

#ifndef ARTS_UTILS_LOCKFREE_STACK_H
#define ARTS_UTILS_LOCKFREE_STACK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

typedef struct arts_lockfree_stack_node_s {
  struct arts_lockfree_stack_node_s *next;
} arts_lockfree_stack_node_t;

typedef struct arts_lockfree_stack_s {
  /* Packed [counter:16 | ptr:48].  volatile so the compiler always reloads
   * inside the CAS loop. */
  volatile uint64_t top;
} arts_lockfree_stack_t;

/* Initialize an empty stack.  Must be called before any push/pop. */
void arts_lockfree_stack_init(arts_lockfree_stack_t *s);

/* Push `node` onto the stack.  `node` must not currently be on this or
 * any other lockfree stack — caller owns its lifetime. */
void arts_lockfree_stack_push(arts_lockfree_stack_t *s,
                              arts_lockfree_stack_node_t *node);

/* Pop the top node, or return NULL if empty. */
arts_lockfree_stack_node_t *arts_lockfree_stack_pop(arts_lockfree_stack_t *s);

/* Snapshot emptiness check (informational only — concurrent pushers may
 * make the stack non-empty immediately after this returns). */
bool arts_lockfree_stack_empty(const arts_lockfree_stack_t *s);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_UTILS_LOCKFREE_STACK_H */
