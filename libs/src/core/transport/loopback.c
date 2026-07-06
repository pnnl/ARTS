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

/* Control-plane send facade + self-loopback.
 *
 * This translation unit is compiled in EVERY build (unlike net.c, whose fabric
 * body is #ifdef ARTS_TRANSPORT_OFI): it holds the public send names the rest of
 * the runtime calls, the outbound-size census, and the same-rank self-loopback
 * queue — none of which depend on the fabric being present.  A remote target is
 * forwarded to the fabric core (arts_net_send_core) when the OFI transport is
 * compiled in; single-node and OFI-off builds never present a remote target, so
 * those calls degrade to warn-and-drop and only the self-loopback carries
 * traffic. */

#include "arts/transport/net.h" /* public API + (OFI) arts_net_send_core */

#include <stdatomic.h>
#include <string.h>

#include "arts/counter/counter.h" /* NET_MSG_* census */
#include "arts/system/identity.h" /* arts_global_rank_id / arts_global_rank_count */
#include "arts/system/print.h"
#include "arts/transport/dispatcher.h" /* arts_transport_dispatch_body */
#include "arts/utils/lockfree_lifo.h"  /* arts_lf_stack_t */
#include "arts/utils/malloc.h"

/* ===== Self-loopback ======================================================
 * A message addressed to the sending rank itself must NOT be delivered by
 * calling the handler inline: a protocol whose acquire round hops home -> owner
 * -> home all on one rank would re-enter its own handler on the caller's stack
 * and recurse without bound.  A self-send is instead copied onto this lock-free
 * stack and delivered later, on a scheduler / progress tick, through the same
 * dispatch path a wire arrival uses.  This makes a self-send "fire and return,"
 * processed asynchronously on a fresh stack — identical to how a peer rank would
 * have received it.
 *
 * Delivery is SERIALIZED to one drainer at a time (a CAS token; a thread that
 * loses it does other work, never spins — lock-free, not a lock).  This
 * reproduces the single per-rank inbound processor that orders ALL coherence on
 * a multi-node run: an ownership round depends on that order (a PROCEED reaches
 * the new front before its CONFIRM; an INVALIDATE publishes the transfer target
 * before the matching release ships), and draining with many threads reorders
 * those steps and strands a transfer.  The serialized section is only the brief
 * message PROCESSING — the data reads it unblocks run on the worker pool, fully
 * concurrent.  Within a turn the chain is taken whole (reverse_drain, FIFO) and
 * each node freed after dispatch, never re-pushed — upholding the stack's
 * single-membership invariant. */
static arts_lf_stack_t g_loopback;
static _Atomic(int) g_loopback_draining;

struct loopback_node_s {
  arts_lf_link_t link;
  unsigned int size;
  /* packet bytes follow */
};

void arts_transport_loopback_post(const void *packet, unsigned int size) {
  struct loopback_node_s *node = (struct loopback_node_s *)arts_malloc(
      sizeof(struct loopback_node_s) + size);
  node->size = size;
  memcpy(node + 1, packet, size);
  arts_lf_stack_push(&g_loopback, &node->link);
}

bool arts_transport_loopback_drain(void) {
  /* Scalable empty gate: this is polled from every worker's scheduler
   * iteration, so the nothing-to-do case must stay READ-ONLY — plain loads
   * keep the head and token lines in shared cache state across all pollers,
   * whereas an unconditional CAS (even a failing one) takes the line
   * exclusive on every poll and ping-pongs it between cores/sockets. */
  if (arts_lf_stack_empty(&g_loopback)) {
    return false;
  }
  if (atomic_load_explicit(&g_loopback_draining, memory_order_relaxed) != 0) {
    return false; /* someone is already draining — find other work */
  }
  /* Single drainer at a time (see the file-scope note): CAS the token; a thread
   * that loses it returns to find other work rather than spinning. */
  int expected = 0;
  if (!atomic_compare_exchange_strong_explicit(&g_loopback_draining, &expected,
                                                1, memory_order_acq_rel,
                                                memory_order_relaxed)) {
    return false;
  }
  arts_lf_link_t *head = arts_lf_stack_reverse_drain(&g_loopback);
  bool did_work = (head != NULL);
  while (head != NULL) {
    /* Save next before dispatch: the handler may post fresh self-sends, but
     * those land on the now-empty stack head (a separate chain) — this node's
     * link is consumed by the free below. */
    arts_lf_link_t *next =
        atomic_load_explicit(&head->next, memory_order_relaxed);
    struct loopback_node_s *node = (struct loopback_node_s *)head;
    /* dispatch_body, not dispatch_packet: a self-send carries no per-sender
     * wire sequence number, so it must skip the SEQUENCENUMBERS wire-ordering
     * check (whose rec_seq_numbers[seq_rank] index would read an unstamped
     * field). */
    arts_transport_dispatch_body((struct arts_msg_header_s *)(node + 1));
    arts_free(node);
    head = next;
  }
  atomic_store_explicit(&g_loopback_draining, 0, memory_order_release);
  return did_work;
}

void arts_loopback_cleanup(void) {
  /* Quiescent teardown: free any self-sends never drained (no dispatch —
   * handlers must not run against torn-down state at shutdown). */
  arts_lf_link_t *lb = arts_lf_stack_reverse_drain(&g_loopback);
  while (lb != NULL) {
    arts_lf_link_t *next = atomic_load_explicit(&lb->next, memory_order_relaxed);
    arts_free(lb);
    lb = next;
  }
}

/* ===== Outbound census + guards =========================================== */

/* Outbound message size census: one switch-free arithmetic bucket on the total
 * wire size (header + payload), taken at the async send entry points — the
 * population distribution a future eager/rendezvous threshold decision needs. */
static inline void arts_net_msg_census(uint64_t total_size) {
  if (total_size <= 64) {
    INCREMENT_NET_MSG_LE64_BY(1);
  } else if (total_size <= 512) {
    INCREMENT_NET_MSG_LE512_BY(1);
  } else if (total_size <= 4096) {
    INCREMENT_NET_MSG_LE4K_BY(1);
  } else if (total_size <= 65536) {
    INCREMENT_NET_MSG_LE64K_BY(1);
  } else {
    INCREMENT_NET_MSG_GT64K_BY(1);
  }
  INCREMENT_NET_MSG_TOTAL_BY(1);
}

static inline void size_send_check(uint64_t size) {
  if (size == 0) {
    ARTS_ERROR("Cannot send zero-size message");
  }
}

/* Reject a self-addressed or out-of-range target.  A caller that wants same-rank
 * delivery uses arts_transport_loopback_post directly; reaching a public send
 * wrapper with a self target is a bug, warned-and-dropped exactly as the former
 * outbox did (this also covers single-node, where every rank is self). */
static inline bool net_target_ok(int rank) {
  if ((unsigned int)rank == arts_global_rank_id ||
      (unsigned int)rank >= arts_global_rank_count) {
    ARTS_WARN("Cannot send to rank %u (self=%u, total=%u)", (unsigned int)rank,
              arts_global_rank_id, arts_global_rank_count);
    return false;
  }
  return true;
}

/* ===== Public send API ==================================================== */

void arts_transport_send_async(int rank, char *message, unsigned int length) {
  if (!net_target_ok(rank)) {
    return;
  }
  arts_net_msg_census(length);
#ifdef ARTS_TRANSPORT_OFI
  arts_net_send_core(rank, message, length, NULL, 0, 0, NULL);
#endif
}

void arts_transport_send_payload_async(int rank, char *message,
                                       unsigned int length, char *payload,
                                       uint64_t size) {
  if (!net_target_ok(rank)) {
    return;
  }
  size_send_check(length);
  size_send_check(size);
  arts_net_msg_census((uint64_t)length + size);
#ifdef ARTS_TRANSPORT_OFI
  arts_net_send_core(rank, message, length, payload, 0, size, NULL);
#else
  (void)payload;
#endif
}

void arts_transport_send_payload_async_free(int rank, char *message,
                                            unsigned int length, char *payload,
                                            unsigned int offset, uint64_t size,
                                            void (*free_method)(void *)) {
  if (!net_target_ok(rank)) {
    return;
  }
  size_send_check(length);
  size_send_check(size);
  arts_net_msg_census((uint64_t)length + size);
#ifdef ARTS_TRANSPORT_OFI
  arts_net_send_core(rank, message, length, payload, offset, size, free_method);
#else
  (void)payload;
  (void)offset;
  (void)free_method;
#endif
}
