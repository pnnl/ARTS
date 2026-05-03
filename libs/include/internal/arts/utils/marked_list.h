/* SPDX-License-Identifier: Apache-2.0
 *
 * Lock-free marked-next linked list (Harris 2001 + Michael 2002 helping
 * protocol) with tagged-pointer ABA defense and a module-private
 * recycle pool.
 *
 * Use case: per-DB pending-RW / pending-RO waiter chains in the
 * coherence protocol.  Each push enqueues a waiter; a successful mark
 * is the exactly-once trigger gate; helping during traversal physically
 * unlinks marked nodes and pushes them to the private pool.
 *
 * Node layout convention: the caller struct MUST embed
 * `arts_marked_list_node_t` as its FIRST field.  The module operates on
 * `arts_marked_list_node_t *` and casts back through container_of
 * semantics (offsetof == 0).  Example:
 *
 *     struct arts_db_rw_waiter_s {
 *         arts_marked_list_node_t link;   // FIRST field — required
 *         arts_guid_t             edt_guid;
 *     };
 *
 * Tagged-pointer encoding of the .next word:
 *   bit 63       — mark flag (1 = logically deleted)
 *   bits 62..48  — 15-bit generation counter
 *   bits 47..0   — 48-bit pointer to next node (NULL terminates)
 *
 * The generation counter increments on every CAS that updates a slot,
 * defeating ABA: a node recycled to the same address via the private
 * pool will still appear with a newer generation, so any captured-then-
 * compared `next` witness is invalidated.
 *
 * Concurrency contract:
 *
 *   marked_list_alloc → caller-side payload init → marked_list_push:
 *       always paired (alloc gives an uninitialized node; caller fills
 *       payload; caller pushes).  Push is atomic at the head sentinel.
 *
 *   marked_list_mark:
 *       returns true on the SOLE successful mark (caller is the unique
 *       trigger actor); after a successful mark the caller MUST NOT
 *       touch the node — the module may physically unlink and recycle
 *       it during the next traversal.  Read needed payload fields
 *       (e.g. edt_guid) into local variables BEFORE calling mark.
 *
 *   marked_list_traverse visits every currently-unmarked node in
 *       traversal order and helps unlink any node it observes as
 *       marked.  A traversal may miss a concurrent head-push (Harris
 *       traversal walks head→tail, never backtracks), but the missed
 *       node will be picked up by a later traversal or by the inserter's
 *       own self-check / next-round scheduling.
 *
 *   marked_list_destroy walks the entire chain (including marked-but-
 *       not-yet-unlinked nodes) plus the private recycle pool, freeing
 *       every node, then frees the head/tail sentinels.  Caller is
 *       responsible for ensuring no concurrent operations are in flight
 *       when destroy is called (typical lifecycle: DB teardown after
 *       all acquires have either completed or aborted).
 */

#ifndef ARTS_UTILS_MARKED_LIST_H
#define ARTS_UTILS_MARKED_LIST_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "arts/utils/lockfree_stack.h"

/* Embedded link.  The .next word holds the tagged pointer described in
 * the file header.  Caller's struct must place this at offset 0. */
typedef struct arts_marked_list_node_s {
  volatile uint64_t next;
} arts_marked_list_node_t;

/* Visit callback used by marked_list_traverse.  May call
 * marked_list_mark on the visited node (typical pattern: claim-and-
 * trigger).  Must not free or push the node — the module owns recycle. */
typedef void (*arts_marked_list_visit_fn)(arts_marked_list_node_t *node,
                                          void *ctx);

typedef struct arts_marked_list_s {
  /* Sentinels live inside the list struct so destroy can free them in
   * one shot.  head.next is the first real node (or &tail when empty);
   * tail.next is always 0 (NULL). */
  arts_marked_list_node_t head;
  arts_marked_list_node_t tail;
  /* Private intrusive recycle pool.  Nodes are pushed here AFTER the
   * helping protocol has physically unlinked them from the chain.  The
   * caller never touches this pool. */
  arts_lockfree_stack_t recycle;
  /* Element size for cold-start malloc when the recycle pool is empty. */
  size_t element_size;
} arts_marked_list_t;

/* Initialize an empty list.  element_size is the sizeof(caller-struct);
 * `arts_marked_list_node_t` (8 bytes) must be the first field. */
void arts_marked_list_init(arts_marked_list_t *list, size_t element_size);

/* Walk the entire chain (incl. marked-but-not-physically-unlinked nodes)
 * and the private recycle pool, freeing every node.  Sentinels live
 * inside the list struct and need no separate free. */
void arts_marked_list_destroy(arts_marked_list_t *list);

/* Allocate a node from the private pool, or malloc on cold-start miss.
 * The returned node has uninitialized payload; its .next word will be
 * set up by the next push. */
arts_marked_list_node_t *arts_marked_list_alloc(arts_marked_list_t *list);

/* Push at head (after head sentinel).  Caller fills payload before
 * calling.  Single CAS; bumps generation on the head's slot. */
void arts_marked_list_push(arts_marked_list_t *list,
                           arts_marked_list_node_t *node);

/* Logical delete: CAS the node's .next word from unmarked to marked
 * (preserving gen+ptr but bumping gen).  Returns true if THIS caller
 * set the mark (sole trigger actor); false if it was already marked. */
bool arts_marked_list_mark(arts_marked_list_node_t *node);

/* Visit every currently-unmarked node in head→tail order.  Performs
 * helping unlink for any marked node encountered, pushing physically
 * unlinked nodes to the private recycle pool. */
void arts_marked_list_traverse(arts_marked_list_t *list,
                               arts_marked_list_visit_fn visit, void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_UTILS_MARKED_LIST_H */
