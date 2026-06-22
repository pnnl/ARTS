/* SPDX-License-Identifier: Apache-2.0
 *
 * LOCK protocol home-side translation unit.
 *
 * Defines: arts_handler_db_lock_request, arts_handler_db_lock_release,
 *          arts_send_db_lock_grant, arts_handler_db_destroy,
 *          arts_db_home_init, arts_db_home_teardown,
 *          arts_home_lockreq_queue_{init,push,pop,peek,empty,destroy},
 *          arts_rank_bitset_{init,set,for_each,destroy}.
 *
 * The lockreq queue and rank_bitset bodies are verbatim from mrnew/home.c
 * (same Vyukov MPSC + bit-packed bitset shapes; LOCK reuses the same struct
 * definitions from lock/types.h).
 *
 * Compiled only for ARTS_COHERENCE_PROTOCOL=LOCK.
 */

/* lock/types.h must precede home.h: it defines arts_home_lockreq_node_s and
 * arts_home_lockreq_queue_s for the LOCK build (home.h declares functions
 * that take these types by pointer, and coherence.h embeds them in
 * arts_db_s). */
#include "arts/coherence/lock/types.h"

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/coherence/home.h"
#include "arts/gas/route_table.h"
#include "arts/ooo.h"
#include "arts/system/identity.h"
#include "arts/transport/outbox.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

/* ===== Home OWNERSHIP-REQUEST FIFO (Vyukov MPSC) =======================
 *
 * Standard Vyukov MPSC FIFO storing an unsigned int requester rank.  The
 * embedded stub sentinel in arts_home_lockreq_queue_s is the permanent queue
 * sentinel; it is never malloc'd or free'd separately.
 */

void arts_home_lockreq_queue_init(struct arts_home_lockreq_queue_s *q) {
  atomic_store_explicit(&q->stub.next, (struct arts_home_lockreq_node_s *)NULL,
                        memory_order_relaxed);
  q->stub.rank = 0;
  atomic_store_explicit(&q->tail, &q->stub, memory_order_relaxed);
  atomic_store_explicit(&q->head, &q->stub, memory_order_relaxed);
}

void arts_home_lockreq_queue_push(struct arts_home_lockreq_queue_s *q,
                                  unsigned int rank) {
  struct arts_home_lockreq_node_s *n =
      (struct arts_home_lockreq_node_s *)malloc(sizeof(*n));
  n->rank = rank;
  atomic_store_explicit(&n->next, (struct arts_home_lockreq_node_s *)NULL,
                        memory_order_relaxed);
  struct arts_home_lockreq_node_s *prev =
      atomic_exchange_explicit(&q->tail, n, memory_order_acq_rel);
  atomic_store_explicit(&prev->next, n, memory_order_release);
}

bool arts_home_lockreq_queue_pop(struct arts_home_lockreq_queue_s *q,
                                 unsigned int *out_rank) {
  for (;;) {
    struct arts_home_lockreq_node_s *head =
        atomic_load_explicit(&q->head, memory_order_acquire);
    struct arts_home_lockreq_node_s *next =
        atomic_load_explicit(&head->next, memory_order_acquire);
    if (next == NULL) {
      struct arts_home_lockreq_node_s *tail =
          atomic_load_explicit(&q->tail, memory_order_acquire);
      if (head == tail) {
        return false;
      }
      continue;
    }
    *out_rank = next->rank;
    atomic_store_explicit(&q->head, next, memory_order_release);
    if (head != &q->stub) {
      free(head);
    }
    return true;
  }
}

bool arts_home_lockreq_queue_peek(const struct arts_home_lockreq_queue_s *q,
                                  unsigned int *out_rank) {
  struct arts_home_lockreq_node_s *head = atomic_load_explicit(
      (_Atomic(struct arts_home_lockreq_node_s *) *)&q->head,
      memory_order_acquire);
  struct arts_home_lockreq_node_s *next =
      atomic_load_explicit(&head->next, memory_order_acquire);
  if (next == NULL) {
    return false;
  }
  *out_rank = next->rank;
  return true;
}

bool arts_home_lockreq_queue_empty(const struct arts_home_lockreq_queue_s *q) {
  struct arts_home_lockreq_node_s *head = atomic_load_explicit(
      (_Atomic(struct arts_home_lockreq_node_s *) *)&q->head,
      memory_order_acquire);
  struct arts_home_lockreq_node_s *next =
      atomic_load_explicit(&head->next, memory_order_acquire);
  if (next != NULL) {
    return false;
  }
  struct arts_home_lockreq_node_s *tail = atomic_load_explicit(
      (_Atomic(struct arts_home_lockreq_node_s *) *)&q->tail,
      memory_order_acquire);
  return head == tail;
}

void arts_home_lockreq_queue_destroy(struct arts_home_lockreq_queue_s *q) {
  struct arts_home_lockreq_node_s *cur =
      atomic_load_explicit(&q->head, memory_order_relaxed);
  while (cur != NULL) {
    struct arts_home_lockreq_node_s *nxt =
        atomic_load_explicit(&cur->next, memory_order_relaxed);
    if (cur != &q->stub) {
      free(cur);
    }
    cur = nxt;
  }
  atomic_store_explicit(&q->tail, (struct arts_home_lockreq_node_s *)NULL,
                        memory_order_relaxed);
  atomic_store_explicit(&q->head, (struct arts_home_lockreq_node_s *)NULL,
                        memory_order_relaxed);
}

/* ===== rank bit-set ===================================================== */

void arts_rank_bitset_init(struct arts_rank_bitset_s *r, unsigned int nranks) {
  r->nranks = nranks;
  r->nwords = (nranks + 63) / 64;
  r->words = (_Atomic(uint64_t) *)calloc(r->nwords, sizeof(_Atomic(uint64_t)));
}

void arts_rank_bitset_destroy(struct arts_rank_bitset_s *r) {
  free(r->words);
  r->words = NULL;
  r->nwords = 0;
}

bool arts_rank_bitset_set(struct arts_rank_bitset_s *r, unsigned int rank) {
  if (rank >= r->nranks) {
    return false;
  }
  unsigned int word_idx = rank / 64;
  uint64_t bit = (uint64_t)1 << (rank % 64);
  uint64_t prev =
      atomic_fetch_or_explicit(&r->words[word_idx], bit, memory_order_acq_rel);
  return (prev & bit) == 0;
}

void arts_rank_bitset_for_each(const struct arts_rank_bitset_s *r,
                               void (*cb)(unsigned int rank, void *ctx),
                               void *ctx) {
  for (unsigned int w = 0; w < r->nwords; w++) {
    uint64_t snap = atomic_load_explicit(&r->words[w], memory_order_acquire);
    while (snap) {
      unsigned int b = (unsigned int)__builtin_ctzll(snap);
      cb(w * 64 + b, ctx);
      snap &= snap - 1;
    }
  }
}

/* ===== RO waiter node (rank-granular RO queue entry) ===================
 * Placed before teardown and lock_home_grant which both use ARTS_CONTAINER_OF
 * on this type. */
struct arts_lock_ro_node_s {
  arts_lf_link_t link; /* FIRST */
  unsigned int rank;
};

/* ===== Home-directory lifecycle ========================================= */

void arts_db_home_init(struct arts_db_s *db, unsigned int rw_holder,
                       unsigned int nranks) {
  (void)rw_holder; /* LOCK tracks mode via lock_state, not rw_holder field */
  /* Seed the creator's RW hold: arts_db_create defaults to an RW acquire (OCR
   * contract), so the home arbiter starts with one writer (the creator rank)
   * holding the lock — its matching release drives w back to 0.  NO_ACQUIRE
   * resets this to a free lock in arts_db_create (mirroring the cache seed). */
  atomic_store_explicit(&db->lock_state, LOCK_MAKE_STATE(0u, 1u, 0u),
                        memory_order_relaxed);
  arts_home_lockreq_queue_init(&db->rw_waiters);
  arts_lf_stack_init(&db->ro_waiters);
  arts_rank_bitset_init(&db->cached_ranks, nranks);
}

void arts_db_home_teardown(struct arts_db_s *db) {
  if (db == NULL) {
    return;
  }
  arts_home_lockreq_queue_destroy(&db->rw_waiters);
  /* Drain + free any rank-nodes still on ro_waiters (single-threaded at
   * teardown — the route slot is already absent, no concurrent push). */
  arts_lf_link_t *node = arts_lf_stack_drain(&db->ro_waiters);
  while (node != NULL) {
    arts_lf_link_t *nx =
        atomic_load_explicit(&node->next, memory_order_relaxed);
    arts_free(ARTS_CONTAINER_OF(node, struct arts_lock_ro_node_s, link));
    node = nx;
  }
  arts_rank_bitset_destroy(&db->cached_ranks);
}

/* ===== Lock state transition ============================================
 *
 * op codes: 0 = RW_acquire (w+1), 1 = RO_acquire (r+1),
 *           2 = RW_release (w-1), 3 = RO_release (r-1).
 * Returns the next lock_state; *out_grant set to:
 *   0 = no grant, 1 = grant one RW (pop), 2 = grant all RO (drain).
 *
 * op codes + grant codes + prototype now live in lock/types.h (shared so the
 * acquire/release handlers can run this arbiter locally on a home==self hit).
 */
uint64_t lock_compute_next(uint64_t cur, int op, uint32_t *out_grant) {
  uint32_t w = LOCK_STATE_W(cur);
  uint32_t r = LOCK_STATE_R(cur);
  uint32_t bit = LOCK_STATE_BIT(cur);
  uint32_t grant = LOCK_GRANT_NONE;
  switch (op) {
  case LOCK_OP_RW_ACQ:
    /* w+1.  none->rw (w==0 && r==0) grants one RW.  A NEW RW participant
     * (w:0->1) arriving while readers hold the lock (w==0 && r>0) means the RO
     * phase is held and this RW parks → state_bit=RO.  If this rank is ALREADY
     * an RW participant (w>0, the RW phase is in progress, rw->rw) the held
     * writer serves it: state_bit MUST be left unchanged — flipping it to RO
     * here would corrupt the live RW phase into an RO phase.  Hence the guard
     * is `w == 0 && r > 0`, NOT `r > 0`. */
    if (w == 0 && r == 0) {
      grant = LOCK_GRANT_ONE_RW; /* none -> rw */
    } else if (w == 0 && r > 0) {
      bit = LOCK_PHASE_BIT_RO; /* RO held (w was 0), RW waits */
    }
    /* else w>0 (rw->rw): bit unchanged, no grant — the held writer serves it.
     */
    w += 1;
    break;
  case LOCK_OP_RO_ACQ:
    /* r+1.  none->ro / ro->ro drains all RO (D7: new RO grants immediately even
     * with RW waiting); if w>0 (RW held) the RO parks (state_bit=RW). */
    if (w == 0) {
      grant = LOCK_GRANT_ALL_RO; /* none->ro or ro->ro */
      bit = LOCK_PHASE_BIT_RO;
    } else if (bit == LOCK_PHASE_BIT_RO && r > 0) {
      grant = LOCK_GRANT_ALL_RO; /* RO phase extends (D7) */
    } else {
      bit = LOCK_PHASE_BIT_RW; /* RW held, RO waits */
    }
    r += 1;
    break;
  case LOCK_OP_RW_REL:
    /* w-1.  w-1>0 grants next writer (D6); w-1==0 && r>0 flips to RO + drains;
     * w-1==0 && r==0 -> none. */
    w -= 1;
    if (w > 0) {
      grant = LOCK_GRANT_ONE_RW; /* rw->rw */
    } else if (r > 0) {
      grant = LOCK_GRANT_ALL_RO; /* rw->ro */
      bit = LOCK_PHASE_BIT_RO;
    }
    break;
  case LOCK_OP_RO_REL:
    /* r-1.  r-1==0 && w>0 flips to RW + grants one; else nothing. */
    r -= 1;
    if (r == 0 && w > 0) {
      grant = LOCK_GRANT_ONE_RW; /* ro->rw */
      bit = LOCK_PHASE_BIT_RW;
    }
    break;
  default:
    break;
  }
  /* Normalize state_bit when one counter hit 0 (state_bit only meaningful with
   * both > 0). */
  if (w == 0 || r == 0) {
    bit = 0;
  }
  *out_grant = grant;
  return LOCK_MAKE_STATE(bit, w, r);
}

/* ===== Grant dispatcher (shared by request + release handlers) ========= */

/* Acquire the home buffer's data for a grant payload and fan-out grants.
 * After pop (rw) / drain (ro), for each target rank: self → direct handler
 * call (local hit, wire 0) / remote → MSG_DB_LOCK_GRANT. */
static void lock_home_grant(struct arts_db_s *db, struct arts_db_cache_s *cache,
                            uint32_t grant) {
  if (grant == LOCK_GRANT_NONE) {
    return;
  }
  arts_shared_ptr_t buf_h = arts_db_buf_acquire(cache);
  struct arts_db_buffer_s *buf =
      (struct arts_db_buffer_s *)arts_shared_get(buf_h);
  const void *data = buf ? buf->data : NULL;
  uint64_t data_size = buf ? cache->db_size : 0;
  /* The grant carries the home buffer's own version.  No separate counter is
   * needed: home commits (RW writeback installs) are sequentialized by the
   * protocol (RW single-owner inter-node + ACK-gated release), so buf->version
   * is monotone across rounds.  The requester's buf_install guard
   * (old.version >= new_version → reject) uses this version to accept a fresh
   * grant and discard a stale duplicate. */
  uint64_t version = buf ? buf->version : 0;
  if (grant == LOCK_GRANT_ONE_RW) {
    unsigned int rank;
    if (arts_home_lockreq_queue_pop(&db->rw_waiters, &rank)) {
      arts_send_db_lock_grant(rank, cache->db_guid, DB_MODE_RW, version, data,
                              data_size);
    }
  } else { /* LOCK_GRANT_ALL_RO: drain ro_waiters, fan-out to each rank. */
    arts_lf_link_t *node = arts_lf_stack_drain(&db->ro_waiters);
    while (node != NULL) {
      arts_lf_link_t *nx =
          atomic_load_explicit(&node->next, memory_order_relaxed);
      struct arts_lock_ro_node_s *rn =
          ARTS_CONTAINER_OF(node, struct arts_lock_ro_node_s, link);
      arts_send_db_lock_grant(rn->rank, cache->db_guid, DB_MODE_RO, version,
                              data, data_size);
      arts_free(rn);
      node = nx;
    }
  }
  arts_db_buf_release(&buf_h);
}

/* ===== arts_send_db_lock_grant ========================================= */

void arts_send_db_lock_grant(unsigned int requester_rank, arts_guid_t db_guid,
                             arts_db_access_mode_t mode, uint64_t version,
                             const void *data, uint64_t data_size) {
  struct arts_msg_lock_grant_packet_s p;
  uint64_t total = sizeof(p) + data_size;
  arts_fill_packet_header(&p.header, total, MSG_DB_LOCK_GRANT);
  p.header.rank = arts_global_rank_id;
  p.db_guid = db_guid;
  p.mode = (uint32_t)mode;
  p.pad = 0;
  p.version = version;
  if (requester_rank == arts_global_rank_id) {
    /* Self-send (home == requester): build the contiguous buffer the handler
     * expects (header + data) and call the body directly (local hit, wire 0).
     */
    char *buf = (char *)arts_malloc(total);
    memcpy(buf, &p, sizeof(p));
    if (data_size > 0 && data != NULL) {
      memcpy(buf + sizeof(p), data, data_size);
    }
    arts_handler_db_lock_grant(buf, (size_t)total);
    arts_free(buf);
    return;
  }
  if (data == NULL || data_size == 0) {
    arts_transport_send_async((int)requester_rank, (char *)&p, sizeof(p));
    return;
  }
  /* Assemble header + payload into a single contiguous allocation and send
   * in one call.  arts_transport_send_payload_async stores only the payload
   * pointer, which points into the home buffer (buf->data).  That buffer can
   * be freed or overwritten by a concurrent writeback before the sender thread
   * reads it, producing a dangling-pointer read and stale data on the wire.
   * Copying into a fresh allocation here makes the transport packet
   * self-contained and owner-independent of the home buffer lifetime. */
  char *pkt = (char *)arts_malloc((size_t)total);
  memcpy(pkt, &p, sizeof(p));
  memcpy(pkt + sizeof(p), data, (size_t)data_size);
  arts_transport_send_async((int)requester_rank, pkt, (unsigned int)total);
  arts_free(pkt);
}

/* ===== arts_handler_db_lock_request ==================================== */

void arts_handler_db_lock_request(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_db_cache_s *cache = &db->cache;
  struct arts_ooo_args_db_lock_request_s *a =
      (struct arts_ooo_args_db_lock_request_s *)args_v;
  unsigned int requester = a->requester;
  arts_db_access_mode_t mode = (arts_db_access_mode_t)a->mode;

  /* (1) push-before-CAS: enqueue this requester rank in its mode's queue
   * BEFORE reading lock_state, so the CAS transition already sees this
   * participant counted. */
  if (mode == DB_MODE_RW) {
    arts_home_lockreq_queue_push(&db->rw_waiters, requester);
  } else {
    struct arts_lock_ro_node_s *n =
        (struct arts_lock_ro_node_s *)arts_malloc(sizeof(*n));
    n->rank = requester;
    arts_lf_stack_push(&db->ro_waiters, &n->link);
  }
  arts_rank_bitset_set(&db->cached_ranks,
                       requester); /* destroy fan-out roster */

  /* (2) read -> compute next -> CAS retry on contention. */
  int op = (mode == DB_MODE_RW) ? LOCK_OP_RW_ACQ : LOCK_OP_RO_ACQ;
  uint32_t grant;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->lock_state, memory_order_acquire);
    next = lock_compute_next(cur, op, &grant);
  } while (!atomic_compare_exchange_weak_explicit(
      &db->lock_state, &cur, next, memory_order_acq_rel, memory_order_acquire));

  /* (3) grant per the transition case. */
  lock_home_grant(db, cache, grant);
}

/* ===== arts_handler_db_lock_release ==================================== */

void arts_handler_db_lock_release(void *item_v, void *args_v) {
  struct arts_db_s *db = (struct arts_db_s *)item_v;
  struct arts_db_cache_s *cache = &db->cache;
  struct arts_ooo_args_db_lock_release_s *a =
      (struct arts_ooo_args_db_lock_release_s *)args_v;
  arts_db_access_mode_t mode = (arts_db_access_mode_t)a->mode;

  /* (1) RW: write the writeback into home's stable buffer in place, then ACK.
   * Under exclusive-lock serialization the releaser held the sole RW grant and
   * home grants the next holder only after this writeback completes, so no
   * reader is touching the buffer here — the in-place overwrite is safe and the
   * buffer address stays fixed (preserving DBs with internal self-pointers).
   * No versioning: home commits are already sequentialized by the protocol (RW
   * single-owner + ACK-gated release). */
  if (mode == DB_MODE_RW && a->data_size > 0) {
    const void *data = (const char *)a + sizeof(*a);
    arts_db_buf_write_inplace(cache, data, a->data_size);
  }
  if (mode == DB_MODE_RW && a->cv != 0) {
    arts_send_db_lock_release_ack(a->releaser, a->db_guid, a->cv);
  }

  /* (2) transition CAS. */
  int op = (mode == DB_MODE_RW) ? LOCK_OP_RW_REL : LOCK_OP_RO_REL;
  uint32_t grant;
  uint64_t cur, next;
  do {
    cur = atomic_load_explicit(&db->lock_state, memory_order_acquire);
    next = lock_compute_next(cur, op, &grant);
  } while (!atomic_compare_exchange_weak_explicit(
      &db->lock_state, &cur, next, memory_order_release, memory_order_acquire));

  /* (3) grant the next holder(s). */
  lock_home_grant(db, cache, grant);
}

/* ===== arts_handler_db_destroy =========================================
 * OOO_DB_DESTROY Cat-B body for the LOCK protocol.  Fan-out DESTROY_NOTIFY
 * to every rank in cached_ranks (the destroy roster), wake parked waiters,
 * then detach the route-table slot. */
void arts_handler_db_destroy(void *item_v, void *args_v) {
  struct arts_db_cache_s *cache = &((struct arts_db_s *)item_v)->cache;
  struct arts_ooo_args_db_destroy_s *a =
      (struct arts_ooo_args_db_destroy_s *)args_v;
  struct arts_db_s *db = arts_db_of_cache(cache);
  if (db == NULL) {
    return;
  }
  unsigned int self = arts_global_rank_id;
  /* Fan-out DESTROY_NOTIFY to every rank that ever acquired this DB.
   * Inline the bitset word-walk (arts_rank_bitset_for_each requires a
   * callback; C has no local functions, so we open-code the loop). */
  {
    const struct arts_rank_bitset_s *bs = &db->cached_ranks;
    for (unsigned int w = 0; w < bs->nwords; w++) {
      uint64_t snap = atomic_load_explicit(&bs->words[w], memory_order_acquire);
      while (snap) {
        unsigned int b = (unsigned int)__builtin_ctzll(snap);
        unsigned int rank = w * 64 + b;
        if (rank != self) {
          arts_send_db_cache_destroy(rank, a->db_guid);
        }
        snap &= snap - 1;
      }
    }
  }
  /* Also drain queued RW requesters that haven't been granted yet. */
  {
    unsigned int q_rank;
    while (arts_home_lockreq_queue_pop(&db->rw_waiters, &q_rank)) {
      if (q_rank != self) {
        arts_send_db_cache_destroy(q_rank, a->db_guid);
      }
    }
  }
  (void)arts_route_table_set_destroyed(a->db_guid);
}
