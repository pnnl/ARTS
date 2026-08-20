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
#include "arts/transport/net.h"

#include <ifaddrs.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>

#include "arts/counter/Preamble.h" /* INCREMENT_{BYTES,NUM}_REMOTE_* */
#include "arts/counter/counter.h"
#include "arts/defs.h"
#include "arts/memory/regpool.h"
#include "arts/system/identity.h" /* arts_global_rank_id / arts_global_rank_count */
#include "arts/system/print.h"
#include "arts/system/threads.h"        /* arts_thread_info, ARTS_THREAD_LOCAL */
#include "arts/transport/dispatcher.h"  /* arts_transport_dispatch_packet */
#include "arts/transport/protocol.h"    /* arts_msg_header_s, MSG_* */

/* ------------------------------------------------------------------------- */
/* Tunables                                                                    */
/* ------------------------------------------------------------------------- */

/* ARTS_NET_RECV_BUF_SIZE / ARTS_NET_MIN_MULTI_RECV / ARTS_NET_MSG_MAX live in
 * net.h: senders on any plane consult the control ceiling to decide inline vs
 * push-rendezvous delivery. */
#define ARTS_NET_RECV_BUF_COUNT 2u

/* Per-thread EAGAIN retry ring.  A handful of slots are reserved so
 * progress-side ACK replies (publish/lock-release ACK, shutdown) can always
 * be buffered even when data sends have saturated the ring — otherwise a full
 * ring could wedge the very progress that would drain it. */
#define ARTS_NET_RING_CAP 1024u
#define ARTS_NET_RING_RESERVED 64u
#define ARTS_NET_MAX_RINGS 1024u

#define ARTS_NET_CQ_BURST 16

/* ------------------------------------------------------------------------- */
/* One pending / outstanding transmit                                          */
/* ------------------------------------------------------------------------- */

/* A posted transmit that owns (or borrows) its buffer until local completion.
 * Two operations share the bookkeeping:
 *   NET_TXN_SEND  — two-sided fi_send of a control message.  `bounce`, when
 *                   non-NULL, is a registered staging copy freed on completion;
 *                   `free_method`, when non-NULL, runs on `free_arg` on
 *                   completion (completion-gated free of a caller payload).
 *   NET_TXN_WRITE — one-sided fi_writedata PUT of a payload straight out of
 *                   the caller's registered buffer (zero-copy: bounce is always
 *                   NULL).  `raddr`/`rkey` name the peer's advertised landing,
 *                   `imm` is the rendezvous txid delivered to the peer's CQ.
 *                   `free_method` is the on-local-done hook that releases the
 *                   source buffer once the provider no longer reads it. */
enum net_txn_op { NET_TXN_SEND, NET_TXN_WRITE };

struct net_txn_s {
  fi_addr_t addr;
  void *buf;
  size_t len;
  void *desc;
  void *bounce;
  void (*free_method)(void *);
  void *free_arg;
  enum net_txn_op op;
  uint64_t raddr; /* NET_TXN_WRITE: peer landing address (per peer mr_mode) */
  uint64_t rkey;  /* NET_TXN_WRITE: peer landing protection key             */
  uint64_t imm;   /* NET_TXN_WRITE: immediate data (rendezvous txid)        */
};

/* ------------------------------------------------------------------------- */
/* Per-thread EAGAIN retry ring (SPSC: the owning thread is the sole producer; a *
 * single drainer at a time — owner or progress — consumes under a CAS token).   */
/* ------------------------------------------------------------------------- */

struct net_ring_s {
  _Atomic uint32_t head; /* consumer cursor (monotone)                        */
  _Atomic uint32_t tail; /* producer cursor (monotone)                        */
  _Atomic int draining;  /* single-drainer token                              */
  struct net_txn_s *slots[ARTS_NET_RING_CAP];
};

/* ------------------------------------------------------------------------- */
/* Receive-buffer context                                                      */
/* ------------------------------------------------------------------------- */

struct net_recv_ctx_s {
  unsigned idx; /* which of the ARTS_NET_RECV_BUF_COUNT landing buffers        */
};

/* ------------------------------------------------------------------------- */
/* Module state                                                                */
/* ------------------------------------------------------------------------- */

static struct {
  struct fi_info *info;
  struct fid_fabric *fabric;
  struct fid_domain *domain;
  struct fid_av *av;
  struct fid_cq *cq;
  struct fid_ep *ep;

  int mr_local;         /* FI_MR_LOCAL negotiated -> pass an mr desc           */
  uint32_t mr_mode;     /* negotiated mr_mode bits                             */
  size_t inject_size;   /* fi_inject ceiling                                   */
  size_t max_msg;       /* provider max message size                          */
  uint64_t tx_order;    /* negotiated tx_attr->msg_order (FI_ORDER_* bits)     */

  fi_addr_t *peers;     /* rank-indexed AV handles (fi_addr_t rank == rank)    */
  unsigned peer_count;

  /* Multi-recv landing buffers are drawn from the registered pool (so their MR
   * is the enclosing slab's — no module-private fi_mr_reg); `rxdesc` is that
   * slab's local descriptor on FI_MR_LOCAL providers, else NULL. */
  void *rxbuf[ARTS_NET_RECV_BUF_COUNT];
  void *rxdesc[ARTS_NET_RECV_BUF_COUNT];
  struct net_recv_ctx_s rxctx[ARTS_NET_RECV_BUF_COUNT];
  /* Bitmask of landing buffers whose FI_MULTI_RECV repost returned -FI_EAGAIN
   * (provider queues momentarily full — a legal transient, NOT an error).  The
   * repost is retried at the top of every CQ reap pass: reaping is exactly
   * what frees provider queue slots, so retry-after-reap converges.  Only
   * touched under the CQ consumer token (or pre-thread init), so a plain
   * unsigned suffices. */
  unsigned rx_repost_pending;

  _Atomic uint64_t tx_outstanding; /* posted-not-yet-completed fi_sends        */
  _Atomic uint64_t eagain_count;   /* observability for the retry path         */
  _Atomic bool quiescing;          /* teardown began: refuse new submissions   */
} g_net;

/* Global registry of per-thread rings so progress can drain rings owned by
 * threads that are not currently sending.  Grow-only, published like the
 * regpool slab table (release on append, acquire on scan). */
static struct net_ring_s *g_rings[ARTS_NET_MAX_RINGS];
static _Atomic uint32_t g_ring_count;
static pthread_mutex_t g_ring_reg_lock = PTHREAD_MUTEX_INITIALIZER;
static ARTS_THREAD_LOCAL struct net_ring_s *t_ring;

/* ------------------------------------------------------------------------- */
/* Small helpers                                                               */
/* ------------------------------------------------------------------------- */

static inline void *net_desc(const void *p) {
  if (!g_net.mr_local) {
    return NULL;
  }
  const arts_regpool_mr_t *m = arts_regpool_lookup(p);
  return (m != NULL && m->mr != NULL) ? fi_mr_desc((struct fid_mr *)m->mr)
                                      : NULL;
}

/* A registered staging buffer for a bounce copy.  regpool fails loudly rather
 * than return an unregistered pointer, so a NULL here is a genuine exhaustion. */
static inline void *net_bounce(size_t size) {
  void *b = arts_regpool_alloc_aligned(size, 64);
  if (b == NULL) {
    ARTS_ERROR("arts_net: registered bounce alloc failed (%zu bytes)", size);
  }
  return b;
}

/* Shutdown / ACK-class messages must never be starved by data-send
 * backpressure — a home rank's ACK is often exactly what unblocks the peer
 * that would otherwise keep the ring full. */
static inline bool net_is_ack_class(unsigned int msg_type) {
  switch (msg_type) {
  case MSG_SHUTDOWN:
  case MSG_DB_PUBLISH_ACK:
  case MSG_DB_PUBLISH_CTS:
    return true;
  default:
    return false;
  }
}

static struct net_ring_s *net_get_ring(void) {
  if (t_ring != NULL) {
    return t_ring;
  }
  struct net_ring_s *r =
      (struct net_ring_s *)calloc(1, sizeof(struct net_ring_s));
  if (r == NULL) {
    ARTS_ERROR("arts_net: retry-ring allocation failed");
  }
  pthread_mutex_lock(&g_ring_reg_lock);
  uint32_t idx = atomic_load_explicit(&g_ring_count, memory_order_relaxed);
  if (idx >= ARTS_NET_MAX_RINGS) {
    pthread_mutex_unlock(&g_ring_reg_lock);
    ARTS_ERROR("arts_net: exceeded %u retry rings", ARTS_NET_MAX_RINGS);
  }
  g_rings[idx] = r;
  atomic_store_explicit(&g_ring_count, idx + 1, memory_order_release);
  pthread_mutex_unlock(&g_ring_reg_lock);
  t_ring = r;
  return r;
}

/* Run a transmit's completion-gated releases and free its bookkeeping. */
static inline void net_txn_complete(struct net_txn_s *txn) {
  if (txn->free_method != NULL) {
    txn->free_method(txn->free_arg);
  }
  if (txn->bounce != NULL) {
    arts_regpool_free(txn->bounce);
  }
  free(txn);
}

static struct net_txn_s *net_txn_new(fi_addr_t addr, void *buf, size_t len,
                                     void *desc, void *bounce,
                                     void (*free_method)(void *),
                                     void *free_arg) {
  struct net_txn_s *txn = (struct net_txn_s *)malloc(sizeof(struct net_txn_s));
  if (txn == NULL) {
    ARTS_ERROR("arts_net: txn allocation failed");
  }
  txn->addr = addr;
  txn->buf = buf;
  txn->len = len;
  txn->desc = desc;
  txn->bounce = bounce;
  txn->free_method = free_method;
  txn->free_arg = free_arg;
  txn->op = NET_TXN_SEND;
  txn->raddr = 0;
  txn->rkey = 0;
  txn->imm = 0;
  return txn;
}

/* Post a txn once (fi_send or fi_writedata by op).  0 = accepted (now
 * outstanding), -FI_EAGAIN = provider busy, other = fatal. */
static inline ssize_t net_try_post(struct net_txn_s *txn) {
  ssize_t rc;
  if (txn->op == NET_TXN_WRITE) {
    rc = fi_writedata(g_net.ep, txn->buf, txn->len, txn->desc, txn->imm,
                      txn->addr, txn->raddr, txn->rkey, txn);
  } else {
    rc = fi_send(g_net.ep, txn->buf, txn->len, txn->desc, txn->addr, txn);
  }
  if (rc == 0) {
    atomic_fetch_add_explicit(&g_net.tx_outstanding, 1, memory_order_relaxed);
  } else if (rc != -FI_EAGAIN) {
    ARTS_ERROR("arts_net: %s failed: %s",
               txn->op == NET_TXN_WRITE ? "fi_writedata" : "fi_send",
               fi_strerror((int)-rc));
  }
  return rc;
}

/* Retry queued txns FIFO until one EAGAINs again or the ring empties.  Single
 * drainer at a time (CAS token); the owner enqueues concurrently (SPSC). */
static bool net_ring_drain(struct net_ring_s *r) {
  int expected = 0;
  if (!atomic_compare_exchange_strong_explicit(
          &r->draining, &expected, 1, memory_order_acq_rel,
          memory_order_relaxed)) {
    return false;
  }
  bool did = false;
  for (;;) {
    uint32_t h = atomic_load_explicit(&r->head, memory_order_relaxed);
    uint32_t t = atomic_load_explicit(&r->tail, memory_order_acquire);
    if (h == t) {
      break;
    }
    struct net_txn_s *txn = r->slots[h % ARTS_NET_RING_CAP];
    ssize_t rc = net_try_post(txn);
    if (rc == -FI_EAGAIN) {
      break; /* still blocked — leave it (and the rest) for the next pass */
    }
    atomic_store_explicit(&r->head, h + 1, memory_order_release);
    did = true;
  }
  atomic_store_explicit(&r->draining, 0, memory_order_release);
  return did;
}

/* Drain every registered retry ring once (this thread's and any other thread's
 * whose owner is not currently sending).  Pure TX progress: it posts queued
 * fi_sends and touches neither the CQ nor RX dispatch, so a producer thread
 * blocked in send backpressure makes forward progress WITHOUT reaping the CQ or
 * running an inbound handler on its own stack (which would nest coherence
 * dispatch on a thread that may hold a DB lock).  Returns true if it posted
 * anything. */
static bool net_drain_all_rings(void) {
  bool did = false;
  uint32_t rc = atomic_load_explicit(&g_ring_count, memory_order_acquire);
  for (uint32_t k = 0; k < rc; k++) {
    if (g_rings[k] != NULL && net_ring_drain(g_rings[k])) {
      did = true;
    }
  }
  return did;
}

/* Enqueue a materialized txn for later retry; false if this class has no room
 * (non-ACK caps out ARTS_NET_RING_RESERVED slots below capacity). */
static bool net_ring_try_enqueue(struct net_ring_s *r, struct net_txn_s *txn,
                                 bool is_ack) {
  uint32_t h = atomic_load_explicit(&r->head, memory_order_acquire);
  uint32_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
  uint32_t used = t - h;
  uint32_t cap =
      is_ack ? ARTS_NET_RING_CAP : (ARTS_NET_RING_CAP - ARTS_NET_RING_RESERVED);
  if (used >= cap) {
    return false;
  }
  r->slots[t % ARTS_NET_RING_CAP] = txn;
  atomic_store_explicit(&r->tail, t + 1, memory_order_release);
  return true;
}

/* Reap a bounded CQ burst WITHOUT dispatching (defined below): completes TX,
 * copies inbound messages onto the pending-dispatch list, reposts released
 * landing buffers — but runs no handler.  Used by the backpressure spin so a
 * saturating producer keeps the CQ moving (its own TX completions free ring
 * slots; draining RX un-sticks a mutually-saturating peer) without re-entering
 * dispatch on a stack that may already hold a coherence lock. */
static bool net_reap_no_dispatch(void);

/* Hand an already-materialized txn to the fabric.  Drains any earlier queued
 * txns first (per-thread FIFO), tries a direct post when the ring is empty, and
 * on EAGAIN buffers the txn — or, when the ring is full for this class, applies
 * deliberate producer backpressure by spinning progress until the provider
 * accepts. */
static void net_submit(struct net_txn_s *txn, bool is_ack) {
  struct net_ring_s *r = net_get_ring();
  net_ring_drain(r);

  uint32_t h = atomic_load_explicit(&r->head, memory_order_acquire);
  uint32_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
  if (h == t) {
    ssize_t rc = net_try_post(txn);
    if (rc == 0) {
      return;
    }
    atomic_fetch_add_explicit(&g_net.eagain_count, 1, memory_order_relaxed);
  }
  if (net_ring_try_enqueue(r, txn, is_ack)) {
    return;
  }
  /* Ring saturated for this class: block the producer, assisting progress until
   * a slot opens, then enqueue BEHIND the existing entries — never direct-post
   * ahead of them, which would break this thread's FIFO ordering.
   *
   * Assisting progress here must be liveness-complete but dispatch-free.  A
   * saturating send may itself be issued from inside an inbound handler on the
   * progress thread, so we must NOT re-enter RX dispatch on this stack (it could
   * nest coherence on a thread already holding a DB lock).  But draining rings
   * ALONE is not enough: if the peer is also saturated and stalled in its own
   * backpressure spin, neither side reaps its CQ, kernel socket buffers fill,
   * the provider's autonomous TX progress wedges, and the two ranks deadlock.
   * So we also take the CQ token (safe: this thread never holds it — dispatch
   * always runs after the token is released) and reap WITHOUT dispatching:
   * completing our TX frees the bounces/slots the ring is waiting on, and
   * draining the peer's inbound messages (onto the pending list, handled later
   * by the progress loop) advances the peer's socket so its TX un-wedges.  This
   * breaks the mutual-saturation cycle while keeping dispatch off this stack. */
  for (;;) {
    net_drain_all_rings();
    net_reap_no_dispatch();
    if (net_ring_try_enqueue(r, txn, is_ack)) {
      return;
    }
  }
}

/* Header-only inject (no completion, buffer reusable on return).  A direct
 * fi_inject is only taken when this thread's retry ring is EMPTY: if earlier
 * sends from this thread are still queued, a fresh inject would jump ahead of
 * them and break same-thread->same-peer FIFO (the ordering FI_ORDER_SAS is meant
 * to give).  When the ring is non-empty, or the inject EAGAINs, the header is
 * copied into a registered bounce and submitted BEHIND the queue as a
 * completion-gated fi_send. */
static void net_inject_header(fi_addr_t addr, const void *buf, size_t len,
                              bool is_ack) {
  struct net_ring_s *r = net_get_ring();
  uint32_t h = atomic_load_explicit(&r->head, memory_order_acquire);
  uint32_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
  if (h == t) {
    ssize_t rc = fi_inject(g_net.ep, buf, len, addr);
    if (rc == 0) {
      return;
    }
    if (rc != -FI_EAGAIN) {
      ARTS_ERROR("arts_net: fi_inject failed: %s", fi_strerror((int)-rc));
    }
    atomic_fetch_add_explicit(&g_net.eagain_count, 1, memory_order_relaxed);
  }
  void *b = net_bounce(len);
  memcpy(b, buf, len);
  net_submit(net_txn_new(addr, b, len, net_desc(b), b, NULL, NULL), is_ack);
}

/* ------------------------------------------------------------------------- */
/* Send core                                                                   */
/* ------------------------------------------------------------------------- */

void arts_net_send_core(int rank, char *message, unsigned int length,
                        char *payload, unsigned int offset, uint64_t size,
                        void (*free_method)(void *)) {
  /* Teardown has begun (threads have joined): drop the send, but still run its
   * completion-gated free so the caller payload is not leaked. */
  if (atomic_load_explicit(&g_net.quiescing, memory_order_acquire)) {
    if (free_method != NULL) {
      free_method(payload);
    }
    return;
  }
  /* A single message must fit one RX landing buffer.  Control-plane ceiling:
   * bulk payloads travel one-sided (arts_net_put_payload) and never come
   * through here, so an oversized total is a control message that outgrew the
   * landing capacity — a protocol bug, rejected loudly rather than split into
   * frames the connectionless RX cannot reassemble. */
  uint64_t wire_total = (uint64_t)length + size;
  if (wire_total > ARTS_NET_MSG_MAX) {
    ARTS_ERROR("arts_net: control message to rank %d is %llu bytes, exceeds RX "
               "landing capacity %zu — bulk payloads must use the one-sided "
               "rendezvous path",
               rank, (unsigned long long)wire_total, (size_t)ARTS_NET_MSG_MAX);
  }
  fi_addr_t addr = g_net.peers[rank];
  bool is_ack =
      net_is_ack_class(((struct arts_msg_header_s *)message)->message_type);
  /* One logical wire message accepted (inject and bounce paths both deliver). */
  INCREMENT_BYTES_REMOTE_SENT_BY(wire_total);
  INCREMENT_NUM_REMOTE_SEND_BY(1);

  if (payload == NULL) {
    if (length <= g_net.inject_size) {
      net_inject_header(addr, message, length, is_ack);
    } else {
      void *b = net_bounce(length);
      memcpy(b, message, length);
      net_submit(net_txn_new(addr, b, length, net_desc(b), b, NULL, NULL),
                 is_ack);
    }
    return;
  }

  /* Header + payload ship as ONE fabric message.  The connectionless RX lands a
   * whole fi_send contiguously in one landing buffer and does no reassembly, so
   * a split header/payload pair would arrive as two independent frames the
   * receiver cannot rejoin (the header frame's declared size would run past the
   * bytes that actually landed, and the raw payload frame carries no header).
   * `wire_total` was already bounded to ARTS_NET_MSG_MAX above, so the whole
   * message is guaranteed to fit; gather it into one registered bounce and
   * submit a single completion-gated send.  The caller payload is released only
   * once that send completes (free_method rides the txn). */
  uint64_t total = (uint64_t)length + size;
  void *b = net_bounce((size_t)total);
  memcpy(b, message, length);
  memcpy((char *)b + length, payload + offset, size);
  net_submit(
      net_txn_new(addr, b, (size_t)total, net_desc(b), b, free_method, payload),
      is_ack);
}

/* ------------------------------------------------------------------------- */
/* Receive posting + progress                                                  */
/* ------------------------------------------------------------------------- */

static void net_post_recv(unsigned idx) {
  struct iovec iov = {.iov_base = g_net.rxbuf[idx],
                      .iov_len = ARTS_NET_RECV_BUF_SIZE};
  struct fi_msg msg = {
      .msg_iov = &iov,
      .desc = g_net.mr_local ? &g_net.rxdesc[idx] : NULL,
      .iov_count = 1,
      .addr = FI_ADDR_UNSPEC,
      .context = &g_net.rxctx[idx],
      .data = 0,
  };
  ssize_t rc = fi_recvmsg(g_net.ep, &msg, FI_MULTI_RECV);
  if (rc == -FI_EAGAIN) {
    /* Transient: provider queues are full right now.  Defer — the CQ reap
     * pass retries pending reposts after draining completions (which is what
     * frees the queue slots).  Fatal only for real errors below. */
    g_net.rx_repost_pending |= (1u << idx);
    return;
  }
  if (rc != 0) {
    ARTS_ERROR("arts_net: fi_recvmsg(buf %u) failed: %s", idx,
               fi_strerror((int)-rc));
  }
  g_net.rx_repost_pending &= ~(1u << idx);
}

/* The CQ is a single-consumer resource: at most one thread reaps completions
 * and reposts landing buffers at a time.  A send thread that spins progress
 * under backpressure must not race the dedicated progress thread on the CQ — a
 * concurrent FI_MULTI_RECV repost could hand a landing buffer back to the
 * provider while another thread is still copying a just-received message out of
 * it.  Ring draining stays lock-free (each ring is token-serialized). */
static _Atomic int g_net_cq_busy;

static inline bool net_cq_trylock(void) {
  int expected = 0;
  return atomic_compare_exchange_strong_explicit(
      &g_net_cq_busy, &expected, 1, memory_order_acq_rel, memory_order_relaxed);
}
static inline void net_cq_unlock(void) {
  atomic_store_explicit(&g_net_cq_busy, 0, memory_order_release);
}

/* Pending inbound-dispatch list.  Inbound completions are recorded under the
 * CQ token and appended here; the progress loop (net_reap_and_dispatch)
 * splices the whole list out — also under the token — and dispatches it AFTER
 * releasing the token, so a handler that sends (and may block in backpressure)
 * never runs under the CQ consumer lock.  Two node kinds share the list:
 *   - a MESSAGE node (rdzv_txid == 0): the fi_send bytes were COPIED off the
 *     multi-recv landing buffer (len bytes follow the node inline) and are
 *     dispatched through arts_transport_dispatch_packet;
 *   - a RENDEZVOUS-DATA node (rdzv_txid != 0, len == 0): a remote-write
 *     completion (FI_REMOTE_CQ_DATA) whose immediate is the rendezvous txid;
 *     dispatch runs the txid pairing (net_rdzv_data_arrived) — the pairing
 *     callback may install a buffer and send, so it too must run only after
 *     the token is released.
 * A producer stalled in net_submit's backpressure spin also reaps the CQ
 * (net_reap_no_dispatch) and appends here, but never dispatches (its stack may
 * hold a coherence lock) — the completion it drained off the wire to un-stick
 * a saturated peer is still delivered, in arrival order, by the progress loop.
 * Every mutation happens under the CQ token, so plain pointers suffice.
 * Per-peer FIFO holds per splice: the token-serialized CQ consumer reaps in
 * arrival order and each spliced batch dispatches head-first.  With MULTIPLE
 * progress threads (io configs run two) two batches can dispatch concurrently,
 * so cross-batch per-peer FIFO is NOT total order — no coherence path relies
 * on it (all wire pairs are reorder-tolerant by version/txid pairing). */
struct net_pending_s {
  struct net_pending_s *next;
  unsigned len; /* message byte count; the packet bytes follow the node inline */
  uint64_t rdzv_txid; /* != 0: rendezvous-data node (no inline bytes) */
};
static struct net_pending_s *g_pending_head; /* FIFO head (oldest undispatched) */
static struct net_pending_s *g_pending_tail; /* FIFO tail (newest)              */

/* ------------------------------------------------------------------------- */
/* Rendezvous txid pairing                                                     */
/* ------------------------------------------------------------------------- */

/* A rendezvous transfer is TWO independent arrivals with no inter-message
 * ordering assumed: the metadata packet (fi_send, dispatched to a handler that
 * registers an expectation) and the payload write completion (imm == txid).
 * Whichever lands first parks in this table; the second one fires the
 * expectation callback.  Keyed by txid.  A txid is allocated by the rank
 * whose table pairs it — the receiver advertises the landing buffer and the
 * txid together — so uniqueness is a local property, and a 32-bit value
 * suffices: it must only be unique among this rank's in-flight transfers,
 * and it fits the narrowest immediate a real fabric grants (InfiniBand
 * write-with-imm carries 4 bytes).
 *
 * An entry is either an EXPECTATION (cb != NULL; metadata arrived first) or an
 * ARRIVAL MARKER (cb == NULL; data arrived first).  Entries are short-lived:
 * exactly one insert and one remove per transfer.
 *
 * Locking: a plain mutex.  Both sides run on the dispatch path of a progress
 * thread (never under the CQ token), but io configs run TWO progress threads,
 * so the table must tolerate concurrent expect/arrival — the mutex covers the
 * find-and-remove; the callback always runs OUTSIDE the lock. */
struct net_rdzv_ent_s {
  struct net_rdzv_ent_s *next;
  uint64_t txid;
  void (*cb)(void *);
  void *arg;
};

#define ARTS_NET_RDZV_BUCKETS 256u
static struct net_rdzv_ent_s *g_rdzv_tab[ARTS_NET_RDZV_BUCKETS];
static pthread_mutex_t g_rdzv_lock = PTHREAD_MUTEX_INITIALIZER;
/* txid counter — starts at 1 so a full txid is never 0 (0 is the wire
 * sentinel for "no landing advertised / no payload moved"). */
static _Atomic uint64_t g_rdzv_txid_ctr = 1;
/* Observability + whitebox-test hook: remote-write completions processed. */
static _Atomic uint64_t g_rdzv_arrived_count;

static inline unsigned net_rdzv_bucket(uint64_t txid) {
  /* The low bits are a local counter, already uniform across buckets. */
  return (unsigned)(txid & (ARTS_NET_RDZV_BUCKETS - 1));
}

/* Unlink and return the entry for txid, or NULL.  Caller holds g_rdzv_lock. */
static struct net_rdzv_ent_s *net_rdzv_take_locked(uint64_t txid) {
  struct net_rdzv_ent_s **pp = &g_rdzv_tab[net_rdzv_bucket(txid)];
  while (*pp != NULL) {
    if ((*pp)->txid == txid) {
      struct net_rdzv_ent_s *e = *pp;
      *pp = e->next;
      return e;
    }
    pp = &(*pp)->next;
  }
  return NULL;
}

static void net_rdzv_insert_locked(uint64_t txid, void (*cb)(void *),
                                   void *arg) {
  struct net_rdzv_ent_s *e =
      (struct net_rdzv_ent_s *)malloc(sizeof(struct net_rdzv_ent_s));
  if (e == NULL) {
    ARTS_ERROR("arts_net: rdzv table node alloc failed");
  }
  e->txid = txid;
  e->cb = cb;
  e->arg = arg;
  unsigned b = net_rdzv_bucket(txid);
  e->next = g_rdzv_tab[b];
  g_rdzv_tab[b] = e;
}

uint64_t arts_net_rdzv_txid_next(void) {
  uint64_t ctr =
      atomic_fetch_add_explicit(&g_rdzv_txid_ctr, 1, memory_order_relaxed);
  /* [1, 2^32-1]: never the wire sentinel 0, and a wrap collision would need
   * one transfer to stay in flight across 2^32-2 subsequent allocations by
   * this same rank — its landing buffer alone forbids that. */
  return 1u + (uint32_t)(ctr % 0xFFFFFFFFULL);
}

void arts_net_rdzv_expect(uint64_t txid, void (*on_data)(void *), void *arg) {
  pthread_mutex_lock(&g_rdzv_lock);
  struct net_rdzv_ent_s *e = net_rdzv_take_locked(txid);
  if (e == NULL) {
    /* Metadata first: park the expectation for the write completion. */
    net_rdzv_insert_locked(txid, on_data, arg);
    pthread_mutex_unlock(&g_rdzv_lock);
    return;
  }
  pthread_mutex_unlock(&g_rdzv_lock);
  if (e->cb != NULL) {
    /* Two expectations on one txid = a protocol bug (each transfer registers
     * exactly one continuation) — never merge them silently. */
    ARTS_ERROR("arts_net: duplicate rdzv expectation for txid %llx",
               (unsigned long long)txid);
  }
  free(e); /* arrival marker consumed */
  on_data(arg); /* data already landed: fire inline (dispatch-path caller) */
}

/* Dispatch-side processing of a remote-write completion (imm == txid).  Runs
 * with the CQ token RELEASED (pending-list dispatch), so the paired callback —
 * which may install a coherence buffer and send — is safe here. */
static void net_rdzv_data_arrived(uint64_t txid) {
  atomic_fetch_add_explicit(&g_rdzv_arrived_count, 1, memory_order_acq_rel);
  pthread_mutex_lock(&g_rdzv_lock);
  struct net_rdzv_ent_s *e = net_rdzv_take_locked(txid);
  if (e == NULL) {
    /* Data first: park an arrival marker for the metadata handler. */
    net_rdzv_insert_locked(txid, NULL, NULL);
    pthread_mutex_unlock(&g_rdzv_lock);
    return;
  }
  pthread_mutex_unlock(&g_rdzv_lock);
  if (e->cb == NULL) {
    ARTS_WARN("arts_net: duplicate rdzv data arrival for txid %llx (dropped)",
              (unsigned long long)txid);
    free(e);
    return;
  }
  void (*cb)(void *) = e->cb;
  void *arg = e->arg;
  free(e);
  cb(arg);
}

bool arts_net_rdzv_local(const void *p, uint64_t len, uint64_t *raddr,
                         uint64_t *rkey) {
  const arts_regpool_mr_t *m = arts_regpool_lookup(p);
  if (m == NULL || m->mr == NULL) {
    return false; /* unregistered (regpool off / no domain): no RDMA landing */
  }
  if ((const char *)p + len > (const char *)m->base + m->len) {
    ARTS_ERROR("arts_net: rdzv landing [%p +%llu) exceeds its registered slab",
               p, (unsigned long long)len);
  }
  /* Target-address semantics follow OUR negotiated mr_mode (homogeneous
   * build/provider across ranks): with FI_MR_VIRT_ADDR the peer targets our
   * virtual address; without it, the offset from the registered base. */
  *raddr = (g_net.mr_mode & FI_MR_VIRT_ADDR)
               ? (uint64_t)(uintptr_t)p
               : (uint64_t)((const char *)p - (const char *)m->base);
  *rkey = m->rkey;
  return true;
}

void arts_net_put_payload(int rank, uint64_t raddr, uint64_t rkey,
                          uint64_t txid, const void *src, uint64_t len,
                          void (*on_local_done)(void *), void *arg) {
  if (len == 0 || txid == 0) {
    ARTS_ERROR("arts_net: put_payload requires a payload and a txid "
               "(len=%llu txid=%llx)",
               (unsigned long long)len, (unsigned long long)txid);
  }
  /* Every datablock payload leaves through this one call, so it is where the
   * payload half of the traffic can be told apart from the control half that
   * BYTES_REMOTE_SENT lumps together with it. */
  INCREMENT_BYTES_DB_PAYLOAD_SENT_BY(len);
  /* Teardown has begun: drop the PUT but still run the local-done hook so the
   * source buffer's completion-gated release is not leaked (mirrors
   * arts_net_send_core's quiescing contract). */
  if (atomic_load_explicit(&g_net.quiescing, memory_order_acquire)) {
    if (on_local_done != NULL) {
      on_local_done(arg);
    }
    return;
  }
  /* One-sided payload PUT is its own wire transfer. */
  INCREMENT_BYTES_REMOTE_SENT_BY(len);
  INCREMENT_NUM_REMOTE_SEND_BY(1);
  struct net_txn_s *txn = net_txn_new(g_net.peers[rank], (void *)src,
                                      (size_t)len, net_desc(src),
                                      /*bounce=*/NULL, on_local_done, arg);
  txn->op = NET_TXN_WRITE;
  txn->raddr = raddr;
  txn->rkey = rkey;
  txn->imm = txid;
  net_submit(txn, /*is_ack=*/false);
}

/* Reclaim a failed transmit's txn from a CQ error entry (both live and teardown
 * paths).  A failed send still owns its bookkeeping; reclaim it so it does not
 * leak and the outstanding count stays honest for the shutdown drain. */
static void net_reap_cq_err(void) {
  struct fi_cq_err_entry err;
  memset(&err, 0, sizeof(err));
  if (fi_cq_readerr(g_net.cq, &err, 0) == 1) {
    ARTS_WARN("arts_net: cq error: %s",
              fi_cq_strerror(g_net.cq, err.prov_errno, err.err_data, NULL, 0));
    if ((err.flags & (FI_SEND | FI_WRITE)) && err.op_context != NULL) {
      net_txn_complete((struct net_txn_s *)err.op_context);
      atomic_fetch_sub_explicit(&g_net.tx_outstanding, 1, memory_order_relaxed);
    }
  }
}

/* Teardown-only reap: run single-threaded after every worker/progress thread
 * has joined (no CQ token needed — nothing else can touch the CQ).  Completes
 * outstanding TX so completion-gated frees run; inbound completions are ignored
 * (no dispatch, no re-arm — the recvs are about to be cancelled).  Returns true
 * if any completion or error entry was seen. */
static bool net_reap_tx_only(void) {
  struct fi_cq_data_entry ents[ARTS_NET_CQ_BURST];
  ssize_t n = fi_cq_read(g_net.cq, ents, ARTS_NET_CQ_BURST);
  if (n > 0) {
    for (ssize_t i = 0; i < n; i++) {
      /* FI_SEND and (local) FI_WRITE completions both carry our txn as
       * op_context; a target-side FI_REMOTE_WRITE completion does not set
       * FI_WRITE, so this cannot mistake an inbound RMA event for a txn. */
      if (ents[i].flags & (FI_SEND | FI_WRITE)) {
        net_txn_complete((struct net_txn_s *)ents[i].op_context);
        atomic_fetch_sub_explicit(&g_net.tx_outstanding, 1,
                                  memory_order_relaxed);
      }
    }
    return true;
  }
  if (n == -FI_EAVAIL) {
    net_reap_cq_err();
    return true;
  }
  if (n != -FI_EAGAIN) {
    ARTS_ERROR("arts_net: fi_cq_read failed: %s", fi_strerror((int)-n));
  }
  return false;
}

/* Reap ONE bounded CQ burst under the CQ token (caller MUST hold it): validate
 * and copy each inbound message onto the pending-dispatch list, repost released
 * landing buffers, and complete TX.  Does NOT dispatch — the caller decides
 * when (net_reap_and_dispatch dispatches after releasing the token; net_reap_no_
 * dispatch never dispatches).  The RX bytes are copied out while the token is
 * held so the multi-recv buffer can be reposted immediately with no borrow
 * outliving the reap.  Returns true if any completion or error entry was seen. */
static bool net_reap_locked(void) {
  /* Retry any landing-buffer repost the provider EAGAIN-deferred: this runs
   * under the CQ token on every reap pass, and each pass drains completions —
   * exactly the action that frees the provider queue slots the repost needs.
   * net_post_recv re-marks the bit if the provider is still saturated. */
  if (g_net.rx_repost_pending != 0) {
    unsigned pending = g_net.rx_repost_pending;
    g_net.rx_repost_pending = 0;
    for (unsigned i = 0; i < ARTS_NET_RECV_BUF_COUNT; i++) {
      if (pending & (1u << i)) {
        net_post_recv(i);
      }
    }
  }
  struct fi_cq_data_entry ents[ARTS_NET_CQ_BURST];
  ssize_t n = fi_cq_read(g_net.cq, ents, ARTS_NET_CQ_BURST);
  if (n > 0) {
    for (ssize_t i = 0; i < n; i++) {
      struct fi_cq_data_entry *e = &ents[i];
      /* Copy the message out BEFORE the buffer can be reposted (an entry can
       * carry both FI_RECV and FI_MULTI_RECV, and the multi-recv completion for
       * a buffer always follows the FI_RECV completions that landed in it, so
       * copy-then-repost order holds within and across entries in this burst). */
      if (e->flags & FI_RECV) {
        /* Framing guard (PERMANENT).  A whole fi_send lands contiguously in one
         * landing buffer, so the peer-declared header.size MUST equal the byte
         * count the fabric actually landed.  A mismatch means a framing bug —
         * e.g. a payload split across frames the connectionless RX cannot rejoin
         * — so fail loud rather than dispatch garbage (a handler reading
         * header.size would run off the end of this copy). */
        if (e->len < sizeof(struct arts_msg_header_s)) {
          ARTS_ERROR("arts_net: RX runt frame — landed %zu bytes < header %zu",
                     e->len, sizeof(struct arts_msg_header_s));
        }
        const struct arts_msg_header_s *hdr =
            (const struct arts_msg_header_s *)e->buf;
        if (hdr->size != e->len) {
          ARTS_ERROR("arts_net: RX framing mismatch — header.size=%llu but "
                     "fabric landed %zu bytes (type=%u from rank=%u)",
                     (unsigned long long)hdr->size, e->len, hdr->message_type,
                     hdr->rank);
        }
        struct net_pending_s *pn = (struct net_pending_s *)malloc(
            sizeof(struct net_pending_s) + e->len);
        if (pn == NULL) {
          ARTS_ERROR("arts_net: RX copy-out alloc failed (%zu bytes)", e->len);
        }
        pn->next = NULL;
        pn->len = (unsigned)e->len;
        pn->rdzv_txid = 0;
        memcpy(pn + 1, e->buf, e->len);
        if (g_pending_tail != NULL) {
          g_pending_tail->next = pn;
        } else {
          g_pending_head = pn;
        }
        g_pending_tail = pn;
      }
      if (e->flags & FI_MULTI_RECV) {
        net_post_recv(((struct net_recv_ctx_s *)e->op_context)->idx);
      }
      /* Target-side completion of a peer's fi_writedata: the payload bytes are
       * already fully placed in the advertised landing buffer ("imm seen =>
       * landing valid" — the ONLY transport ordering this plane relies on).
       * Queue a rendezvous-data node carrying the immediate (txid); the txid
       * pairing — which may run an install callback that sends — happens on
       * the dispatch side, after the token is released.  A remote write
       * without immediate data is never issued by this runtime; ignore it
       * defensively rather than fabricate a zero txid. */
      if ((e->flags & FI_REMOTE_WRITE) && (e->flags & FI_REMOTE_CQ_DATA)) {
        /* One-sided payload landed: count it as its own remote receive (the
         * pairing control message is counted separately at dispatch). */
        INCREMENT_BYTES_REMOTE_RECEIVED_BY(e->len);
        INCREMENT_NUM_REMOTE_RECEIVE_BY(1);
        struct net_pending_s *pn =
            (struct net_pending_s *)malloc(sizeof(struct net_pending_s));
        if (pn == NULL) {
          ARTS_ERROR("arts_net: rdzv completion node alloc failed");
        }
        pn->next = NULL;
        pn->len = 0;
        pn->rdzv_txid = e->data;
        if (g_pending_tail != NULL) {
          g_pending_tail->next = pn;
        } else {
          g_pending_head = pn;
        }
        g_pending_tail = pn;
      }
      if (e->flags & (FI_SEND | FI_WRITE)) {
        net_txn_complete((struct net_txn_s *)e->op_context);
        atomic_fetch_sub_explicit(&g_net.tx_outstanding, 1,
                                  memory_order_relaxed);
      }
    }
    return true;
  }
  if (n == -FI_EAVAIL) {
    net_reap_cq_err();
    return true;
  }
  if (n != -FI_EAGAIN) {
    ARTS_ERROR("arts_net: fi_cq_read failed: %s", fi_strerror((int)-n));
  }
  return false;
}

/* Live reap-and-dispatch — the progress loop's inbound driver.
 *
 * MANDATORY INVARIANT — no send runs under the CQ consumer lock.  An inbound
 * coherence handler may itself send (e.g. a snapshot response), and a send can
 * block in net_submit's backpressure spin; running that under the CQ token would
 * self-deadlock the very reap that frees the ring / re-arms the landing buffers.
 * So the token guards ONLY the CQ read plus the copy-out, repost, and pending-
 * list bookkeeping; dispatch happens AFTER the token is released.
 *
 * The whole pending list is spliced out under the token (this burst's messages
 * at the tail, plus anything a backpressured producer's net_reap_no_dispatch
 * left at the head) and dispatched FIFO with the token released, preserving
 * per-peer arrival order (oldest first).  Returns true if it reaped or
 * dispatched anything. */
static bool net_reap_and_dispatch(void) {
  if (!net_cq_trylock()) {
    return false; /* another progress thread is consuming the CQ */
  }
  bool progressed = net_reap_locked();
  struct net_pending_s *head = g_pending_head;
  g_pending_head = NULL;
  g_pending_tail = NULL;
  net_cq_unlock();

  while (head != NULL) {
    struct net_pending_s *next = head->next;
    if (head->rdzv_txid != 0) {
      net_rdzv_data_arrived(head->rdzv_txid);
    } else {
      INCREMENT_BYTES_REMOTE_RECEIVED_BY(head->len);
      INCREMENT_NUM_REMOTE_RECEIVE_BY(1);
      arts_transport_dispatch_packet((struct arts_msg_header_s *)(head + 1));
    }
    free(head);
    head = next;
    progressed = true;
  }
  return progressed;
}

/* Backpressure-assist reap: reap a bounded CQ burst under the token but do NOT
 * dispatch — leave the inbound messages on the pending list for the progress
 * loop.  Safe from any thread that does not already hold the CQ token (dispatch
 * always runs with the token released, so a producer stalled in net_submit never
 * holds it, and a worker never touches the CQ except here).  Completing TX frees
 * the ring slots the caller is spinning on; draining RX advances a mutually
 * saturated peer's socket so its wedged TX progresses.  Returns true if it
 * reaped anything. */
static bool net_reap_no_dispatch(void) {
  if (!net_cq_trylock()) {
    return false; /* the progress thread (or another producer) has the CQ */
  }
  bool progressed = net_reap_locked();
  net_cq_unlock();
  return progressed;
}

bool arts_net_progress(void) {
  /* Reap the CQ (single-consumer; copy-out RX under the token, dispatch after
   * release), then drain every registered retry ring (this thread's and any
   * other thread's whose owner is not currently sending).  Non-blocking by
   * contract: idle pacing is the caller's policy (the dedicated progress
   * thread busy-polls with the architectural pause hint — a sleeping poll here
   * would put a fixed per-hop latency floor under every message an idle rank
   * receives, which multiplies across latency-bound message chains). */
  bool did = net_reap_and_dispatch();
  did |= net_drain_all_rings();
  return did;
}

/* ------------------------------------------------------------------------- */
/* Bootstrap glue                                                              */
/* ------------------------------------------------------------------------- */

unsigned arts_net_own_address(void *buf, unsigned buflen) {
  size_t len = buflen;
  int rc = fi_getname(&g_net.ep->fid, buf, &len);
  if (rc != 0) {
    ARTS_ERROR("arts_net: fi_getname failed: %s", fi_strerror(-rc));
  }
  return (unsigned)len;
}

void arts_net_av_insert_table(const void *addrs, unsigned addrlen,
                              unsigned count) {
  fi_addr_t *fiaddrs = (fi_addr_t *)malloc(sizeof(fi_addr_t) * count);
  if (fiaddrs == NULL) {
    ARTS_ERROR("arts_net: av-insert table alloc failed");
  }
  int inserted = fi_av_insert(g_net.av, addrs, count, fiaddrs, 0, NULL);
  if (inserted != (int)count) {
    ARTS_ERROR("arts_net: fi_av_insert inserted %d of %u addresses", inserted,
               count);
  }
  g_net.peers = (fi_addr_t *)malloc(sizeof(fi_addr_t) * count);
  if (g_net.peers == NULL) {
    ARTS_ERROR("arts_net: peer table alloc failed");
  }
  for (unsigned i = 0; i < count; i++) {
    /* FI_AV_TABLE assigns sequential handles from 0, so inserting in rank order
     * yields fi_addr_t == rank; assert rather than silently mis-route. */
    if (fiaddrs[i] != (fi_addr_t)i) {
      ARTS_ERROR("arts_net: AV handle %u != rank %u (FI_AV_TABLE assumption "
                 "violated)",
                 (unsigned)fiaddrs[i], i);
    }
    g_net.peers[i] = fiaddrs[i];
  }
  g_net.peer_count = count;
  free(fiaddrs);
  /* fi_av_insert strides `addrs` by the provider's own address length; the
   * caller must have laid the table out at exactly that stride. */
  (void)addrlen;
}

struct fid_domain *arts_net_domain(void) { return g_net.domain; }

/* ------------------------------------------------------------------------- */
/* Init / teardown                                                             */
/* ------------------------------------------------------------------------- */

/* First AF_INET address of the interface named `ifname` (exact match wins;
 * a prefix match is accepted so one config can say "ib" across hosts whose
 * suffixes differ).  Returns false if no interface matches or the matching
 * ones carry no IPv4 address. */
static bool net_lookup_iface_addr(const char *ifname,
                                  struct sockaddr_in *out) {
  struct ifaddrs *ifap = NULL;
  if (getifaddrs(&ifap) != 0) {
    return false;
  }
  bool found = false;
  bool exact = false;
  size_t want_len = strlen(ifname);
  for (struct ifaddrs *ifa = ifap; ifa != NULL; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == NULL || ifa->ifa_addr->sa_family != AF_INET) {
      continue;
    }
    bool is_exact = strcmp(ifa->ifa_name, ifname) == 0;
    bool is_prefix = strncmp(ifa->ifa_name, ifname, want_len) == 0;
    if (!is_exact && !is_prefix) {
      continue;
    }
    if (is_exact || !found) {
      memcpy(out, ifa->ifa_addr, sizeof(*out));
      found = true;
      exact = is_exact;
    }
    if (exact) {
      break;
    }
  }
  freeifaddrs(ifap);
  return found;
}

void arts_net_init(const char *provider, const char *fabric_domain,
                   const char *net_interface) {
  bool provider_pinned = provider && provider[0] != '\0';
  if (provider_pinned) {
    /* Config is the deliberate artifact, the env var is ambient: overwrite
     * rather than merely default, so an explicit choice here always wins
     * even if FI_PROVIDER was already set to something else. */
    setenv("FI_PROVIDER", provider, 1);
  }
  /* Provider precedence: cfg > FI_PROVIDER env > tcp-first-auto.  With no
   * explicit choice anywhere, fi_getinfo's unconstrained pick is whatever
   * provider happens to sort first on the host (e.g. a datagram emulation
   * stack) — an unpredictable default for a transport with delivery-robustness
   * differences between providers.  Prefer the connection-oriented core "tcp"
   * provider first; fall back to unconstrained auto-selection only when tcp
   * yields no usable RDM info on this host. */
  const char *env_provider = getenv("FI_PROVIDER");
  bool env_pinned =
      !provider_pinned && env_provider != NULL && env_provider[0] != '\0';
  bool tcp_first = !provider_pinned && !env_pinned;

  struct fi_info *hints = fi_allocinfo();
  if (hints == NULL) {
    ARTS_ERROR("arts_net: fi_allocinfo failed");
  }
  /* FI_RMA_EVENT: the rendezvous data plane's target side needs a CQ
   * completion (with the immediate txid) when a peer's fi_writedata lands —
   * "imm seen => landing buffer valid" is the plane's one ordering axiom. */
  hints->caps = FI_MSG | FI_RMA | FI_MULTI_RECV | FI_RMA_EVENT;
  hints->mode = 0;
  hints->ep_attr->type = FI_EP_RDM;
  hints->domain_attr->threading = FI_THREAD_SAFE;
  /* Request send-after-send ordering (FI_ORDER_SAS): two messages this rank
   * sends to a given peer are delivered to that peer's endpoint in send order
   * (per-EP-pair FIFO).  After the transport cutover this replaces the outbox's
   * per-(thread->dest) FIFO for the coherence rounds that ship a state update
   * and then its acknowledgement to the same peer from the same thread.  The
   * value the provider actually grants is stored and logged below; a provider
   * that does not honor it would surface here rather than silently. */
  hints->tx_attr->msg_order = FI_ORDER_SAS;
  hints->rx_attr->msg_order = FI_ORDER_SAS;
  /* Offer the common mr_mode bits and accept whatever subset the provider
   * requires (the tcp provider requires none). */
  hints->domain_attr->mr_mode =
      FI_MR_LOCAL | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR;
  if (provider_pinned) {
    /* fi_freeinfo() below frees fabric_attr->prov_name itself, so it must be
     * a heap string fi_getinfo/fi_freeinfo owns from here on, not the cfg's
     * storage. */
    hints->fabric_attr->prov_name = strdup(provider);
  } else if (tcp_first) {
    hints->fabric_attr->prov_name = strdup("tcp");
  }
  if (fabric_domain && fabric_domain[0] != '\0') {
    /* Same ownership rule as prov_name: fi_freeinfo frees domain_attr->name. */
    hints->domain_attr->name = strdup(fabric_domain);
  }

  /* Source-interface bind, IP providers only.  The effective provider is an
   * IP one when it was explicitly named tcp/sockets, or when the tcp-first
   * default is in play.  For anything else (verbs, shm, ...) an interface
   * name is meaningless — addressing there is by fabric domain. */
  const char *effective_provider =
      provider_pinned ? provider : (env_pinned ? env_provider : NULL);
  bool ip_provider =
      tcp_first ||
      (effective_provider != NULL && (strcmp(effective_provider, "tcp") == 0 ||
                                      strcmp(effective_provider, "sockets") == 0));
  if (net_interface && net_interface[0] != '\0' && ip_provider) {
    /* fi_freeinfo frees hints->src_addr, so it must be heap-owned. */
    struct sockaddr_in *src =
        (struct sockaddr_in *)calloc(1, sizeof(struct sockaddr_in));
    if (src == NULL || !net_lookup_iface_addr(net_interface, src)) {
      ARTS_ERROR("arts_net: net_interface=%s has no usable IPv4 address — "
                 "refusing to fall back to the default route",
                 net_interface);
    }
    src->sin_port = 0; /* any port; only the address constrains the bind */
    hints->addr_format = FI_SOCKADDR_IN;
    hints->src_addr = src;
    hints->src_addrlen = sizeof(struct sockaddr_in);
  }

  int rc = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL, NULL,
                      0, hints, &g_net.info);
  if ((rc != 0 || g_net.info == NULL) && provider_pinned) {
    /* A layering provider can deliver remote-write completions without
     * advertising FI_RMA_EVENT: rxm forwards the core provider's
     * FI_REMOTE_WRITE completion — immediate included — yet never names the
     * bit in its caps, so a capability filter refuses a stack that in fact
     * provides the event.  Ask again without the bit, on the pinned path
     * only: a provider that does advertise it (tcp) keeps being asked, since
     * there the request is what arms remote events. */
    hints->caps &= ~(uint64_t)FI_RMA_EVENT;
    rc = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL, NULL,
                    0, hints, &g_net.info);
    if (rc == 0 && g_net.info != NULL) {
      ARTS_INFO("arts_net: provider %s matched without FI_RMA_EVENT; relying "
                "on its layered remote-write completion delivery",
                provider);
    }
  }
  if ((rc != 0 || g_net.info == NULL) && tcp_first) {
    /* tcp-first probe found nothing usable on this host — retry without a
     * provider constraint and take fi_getinfo's own pick. */
    free(hints->fabric_attr->prov_name);
    hints->fabric_attr->prov_name = NULL;
    rc = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL, NULL,
                    0, hints, &g_net.info);
  }
  fi_freeinfo(hints);
  if (rc != 0 || g_net.info == NULL) {
    ARTS_ERROR("arts_net: fi_getinfo found no FI_EP_RDM provider%s%s: %s",
               provider_pinned ? " matching requested provider=" : "",
               provider_pinned ? provider : "", fi_strerror(-rc));
  }

  g_net.mr_mode = (uint32_t)g_net.info->domain_attr->mr_mode;
  g_net.mr_local = (g_net.mr_mode & FI_MR_LOCAL) != 0;
  g_net.inject_size = g_net.info->tx_attr->inject_size;
  g_net.max_msg = g_net.info->ep_attr->max_msg_size;
  g_net.tx_order = g_net.info->tx_attr->msg_order;

  /* The rendezvous txid is receiver-allocated and 32-bit by design, sized to
   * the narrowest immediate a real fabric grants (InfiniBand write-with-imm
   * carries 4 bytes; tcp grants 8).  A provider below even that would
   * silently truncate the immediate and pair the wrong transfers. */
  if (g_net.info->domain_attr->cq_data_size < sizeof(uint32_t)) {
    ARTS_ERROR("arts_net: provider cq_data_size %zu < 4 — rendezvous txids "
               "need at least a 32-bit immediate",
               g_net.info->domain_attr->cq_data_size);
  }

  rc = fi_fabric(g_net.info->fabric_attr, &g_net.fabric, NULL);
  if (rc != 0) {
    ARTS_ERROR("arts_net: fi_fabric failed: %s", fi_strerror(-rc));
  }
  rc = fi_domain(g_net.fabric, g_net.info, &g_net.domain, NULL);
  if (rc != 0) {
    ARTS_ERROR("arts_net: fi_domain failed: %s", fi_strerror(-rc));
  }

  struct fi_av_attr av_attr;
  memset(&av_attr, 0, sizeof(av_attr));
  av_attr.type = FI_AV_TABLE; /* fi_addr_t == insertion index == rank */
  rc = fi_av_open(g_net.domain, &av_attr, &g_net.av, NULL);
  if (rc != 0) {
    ARTS_ERROR("arts_net: fi_av_open failed: %s", fi_strerror(-rc));
  }

  struct fi_cq_attr cq_attr;
  memset(&cq_attr, 0, sizeof(cq_attr));
  cq_attr.format = FI_CQ_FORMAT_DATA; /* need buf/len/flags for RX + data slot */
  cq_attr.wait_obj = FI_WAIT_NONE;
  rc = fi_cq_open(g_net.domain, &cq_attr, &g_net.cq, NULL);
  if (rc != 0) {
    ARTS_ERROR("arts_net: fi_cq_open failed: %s", fi_strerror(-rc));
  }

  rc = fi_endpoint(g_net.domain, g_net.info, &g_net.ep, NULL);
  if (rc != 0) {
    ARTS_ERROR("arts_net: fi_endpoint failed: %s", fi_strerror(-rc));
  }
  rc = fi_ep_bind(g_net.ep, &g_net.av->fid, 0);
  if (rc != 0) {
    ARTS_ERROR("arts_net: fi_ep_bind(av) failed: %s", fi_strerror(-rc));
  }
  rc = fi_ep_bind(g_net.ep, &g_net.cq->fid, FI_TRANSMIT | FI_RECV);
  if (rc != 0) {
    ARTS_ERROR("arts_net: fi_ep_bind(cq) failed: %s", fi_strerror(-rc));
  }
  rc = fi_enable(g_net.ep);
  if (rc != 0) {
    ARTS_ERROR("arts_net: fi_enable failed: %s", fi_strerror(-rc));
  }

  size_t min_mr = ARTS_NET_MIN_MULTI_RECV;
  rc = fi_setopt(&g_net.ep->fid, FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
                 &min_mr, sizeof(min_mr));
  if (rc != 0) {
    ARTS_ERROR("arts_net: fi_setopt(MIN_MULTI_RECV) failed: %s",
               fi_strerror(-rc));
  }

  ARTS_INFO("arts_net: fabric up provider=%s inject_size=%zu mr_mode=0x%x "
            "mr_local=%d multi_recv=%uMiB max_msg=%zu msg_order=0x%llx sas=%d",
            g_net.info->fabric_attr->prov_name, g_net.inject_size, g_net.mr_mode,
            g_net.mr_local,
            (unsigned)(ARTS_NET_RECV_BUF_SIZE / (1024 * 1024)), g_net.max_msg,
            (unsigned long long)g_net.tx_order,
            (g_net.tx_order & FI_ORDER_SAS) == FI_ORDER_SAS);
}

void arts_net_rx_arm(void) {
  /* Landing buffers come from the registered pool (initialized just before this
   * call), so recv memory is pinned/registered exactly like send bounces — one
   * MR-key namespace, and no duplicate requested_key on a provider that needs
   * FI_MR_LOCAL without FI_MR_PROV_KEY (a class of bug the pool already avoids
   * with a monotone key counter).  When the provider requires a local
   * descriptor, it is the enclosing slab's MR handle; otherwise it is NULL. */
  for (unsigned i = 0; i < ARTS_NET_RECV_BUF_COUNT; i++) {
    g_net.rxbuf[i] = arts_regpool_alloc_aligned(ARTS_NET_RECV_BUF_SIZE, 4096);
    if (g_net.rxbuf[i] == NULL) {
      ARTS_ERROR("arts_net: recv buffer %u allocation failed", i);
    }
    g_net.rxctx[i].idx = i;
    if (g_net.mr_local) {
      g_net.rxdesc[i] = net_desc(g_net.rxbuf[i]);
      if (g_net.rxdesc[i] == NULL) {
        ARTS_ERROR("arts_net: recv buffer %u not in a registered slab", i);
      }
    } else {
      g_net.rxdesc[i] = NULL;
    }
  }
  for (unsigned i = 0; i < ARTS_NET_RECV_BUF_COUNT; i++) {
    net_post_recv(i);
  }
}

void arts_net_quiesce(void) {
  /* Phase 1 of teardown, run BEFORE the registered pool is torn down and after
   * every worker/sender/receiver thread has joined.  Discarding sends that were
   * never accepted by the provider at final teardown is deliberate — it mirrors
   * a bounded drain-then-exit and must never block shutdown on a peer that has
   * already gone.  The ordering matters: everything that references a slab MR
   * (send bounces still queued, and the recv buffers themselves) is released
   * here, while the pool is still live, so the pool cleanup that follows can
   * close those MRs without any outstanding reference — and no desc ever names a
   * closed MR. */
  atomic_store_explicit(&g_net.quiescing, true, memory_order_release);

  /* Discard queued-but-unposted retry txns.  These never reached the provider,
   * so they are NOT part of tx_outstanding; run their completion-gated frees
   * (caller payload + bounce back to the still-live pool) and drop them. */
  uint32_t nr = atomic_load_explicit(&g_ring_count, memory_order_acquire);
  for (uint32_t k = 0; k < nr; k++) {
    struct net_ring_s *r = g_rings[k];
    uint32_t h = atomic_load_explicit(&r->head, memory_order_relaxed);
    uint32_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
    for (; h != t; h++) {
      net_txn_complete(r->slots[h % ARTS_NET_RING_CAP]);
    }
    free(r);
    g_rings[k] = NULL;
  }
  atomic_store_explicit(&g_ring_count, 0, memory_order_release);
  t_ring = NULL;

  /* Free any inbound messages reaped but not yet dispatched (a producer's
   * net_reap_no_dispatch can leave entries on the pending list).  Quiescent
   * teardown: drop them without dispatch — handlers must not run against
   * torn-down state at shutdown (mirrors arts_loopback_cleanup). */
  while (g_pending_head != NULL) {
    struct net_pending_s *next = g_pending_head->next;
    free(g_pending_head);
    g_pending_head = next;
  }
  g_pending_tail = NULL;

  /* Drop unpaired rendezvous entries (an expectation whose data never landed,
   * or data whose metadata never arrived — both only possible when a peer died
   * or shutdown cut a transfer mid-flight).  Quiescent: no callback runs. */
  pthread_mutex_lock(&g_rdzv_lock);
  for (unsigned b = 0; b < ARTS_NET_RDZV_BUCKETS; b++) {
    while (g_rdzv_tab[b] != NULL) {
      struct net_rdzv_ent_s *e = g_rdzv_tab[b];
      g_rdzv_tab[b] = e->next;
      free(e);
    }
  }
  pthread_mutex_unlock(&g_rdzv_lock);

  /* Reap already-posted sends so their completion-gated frees run (bounces back
   * to the still-live pool).  TX-only: inbound completions are ignored and no
   * landing buffer is re-armed — they are about to be cancelled.  Bounded — the
   * bound guards against a completion that never arrives on a wedged provider.
   * Called single-threaded, after every worker/progress thread has joined, so
   * net_reap_tx_only needs no CQ token — no other thread can contend the CQ. */
  for (int i = 0;
       i < 100000 &&
       atomic_load_explicit(&g_net.tx_outstanding, memory_order_relaxed) != 0;
       i++) {
    net_reap_tx_only();
  }

  /* Close the endpoint (cancels the posted recvs), then the CQ and AV. */
  if (g_net.ep != NULL) {
    fi_close(&g_net.ep->fid);
    g_net.ep = NULL;
  }
  if (g_net.cq != NULL) {
    fi_close(&g_net.cq->fid);
    g_net.cq = NULL;
  }
  if (g_net.av != NULL) {
    fi_close(&g_net.av->fid);
    g_net.av = NULL;
  }

  /* Return the recv landing buffers to the pool while it is still live; their
   * slab MR is closed by the pool cleanup that follows. */
  for (unsigned i = 0; i < ARTS_NET_RECV_BUF_COUNT; i++) {
    if (g_net.rxbuf[i] != NULL) {
      arts_regpool_free(g_net.rxbuf[i]);
      g_net.rxbuf[i] = NULL;
    }
    g_net.rxdesc[i] = NULL;
  }
}

void arts_net_teardown(void) {
  /* Phase 2 of teardown, run AFTER arts_regpool_cleanup: the pool has closed
   * every slab MR (which backed both sends and the recv buffers), so nothing
   * registered outlives its domain — close the domain, then the fabric. */
  if (g_net.domain != NULL) {
    fi_close(&g_net.domain->fid);
    g_net.domain = NULL;
  }
  if (g_net.fabric != NULL) {
    fi_close(&g_net.fabric->fid);
    g_net.fabric = NULL;
  }
  if (g_net.info != NULL) {
    fi_freeinfo(g_net.info);
    g_net.info = NULL;
  }
  if (g_net.peers != NULL) {
    free(g_net.peers);
    g_net.peers = NULL;
  }
  g_net.peer_count = 0;
}

void arts_net_drain_outstanding(unsigned int deadline_ms) {
  /* Bounded wait for accepted fabric sends to complete, so the shutdown
   * broadcast's frames leave this node before teardown.  The dedicated progress
   * thread is still running and reaps completions; we assist by reaping too and
   * poll the outstanding count.  Single-node never brought the fabric up. */
  if (arts_global_rank_count <= 1) {
    return;
  }
  uint64_t deadline_ns =
      (uint64_t)deadline_ms * 1000000ULL; /* ms -> ns budget */
  struct timespec start;
  (void)clock_gettime(CLOCK_MONOTONIC, &start);
  for (;;) {
    if (atomic_load_explicit(&g_net.tx_outstanding, memory_order_acquire) == 0) {
      return;
    }
    arts_net_progress();
    struct timespec now;
    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    uint64_t elapsed = (uint64_t)(now.tv_sec - start.tv_sec) * 1000000000ULL +
                       (uint64_t)(now.tv_nsec - start.tv_nsec);
    if (elapsed >= deadline_ns) {
      ARTS_INFO("arts_net: shutdown TX drain timeout (%llu sends still "
                "outstanding)",
                (unsigned long long)atomic_load_explicit(
                    &g_net.tx_outstanding, memory_order_relaxed));
      return;
    }
    struct timespec ts = {.tv_sec = 0, .tv_nsec = 200000L /* 0.2 ms */};
    nanosleep(&ts, NULL);
  }
}
