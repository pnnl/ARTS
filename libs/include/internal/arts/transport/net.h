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
#ifndef ARTS_TRANSPORT_NET_H
#define ARTS_TRANSPORT_NET_H

/* libfabric (OFI) message transport core.
 *
 * The module is brought up only when the run is multinode
 * (arts_global_rank_count > 1); a single-node run stays entirely fabric-free
 * (no fi_getinfo, no domain, no registration).  Bring-up runs alongside — and
 * before — the registered slab pool so the pool's slabs can register against
 * the domain this module creates.
 *
 * Lifecycle (all on the main thread, before any worker/sender/receiver thread
 * is spawned):
 *   arts_net_init()              — fi_getinfo, fabric/domain/av/cq/ep (no RX yet)
 *   arts_net_exchange_addresses()— fi_getname blob swap over the TCP mesh + av_insert
 *   arts_regpool_init(domain)    — (caller) registers slabs against arts_net_domain()
 *   arts_net_rx_arm()            — landing buffers from the pool, post multi-recv
 *   ... runtime runs ...
 *   arts_net_quiesce()           — refuse new sends, discard/reap in-flight TX,
 *                                  close ep/cq/av, return recv buffers to the pool
 *   arts_regpool_cleanup()       — (caller) closes slab MRs
 *   arts_net_teardown()          — closes domain/fabric
 *
 * The RX landing buffers are drawn from the registered pool (hence arming them
 * AFTER regpool_init) rather than a module-private registration: one MR-key
 * namespace, and no duplicate requested_key on a provider that requires
 * FI_MR_LOCAL without FI_MR_PROV_KEY.  Teardown is two-phase around the pool
 * cleanup so no send desc or recv buffer references a slab MR after it closes.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

/* Wire ABI (packet structs, arts_msg_type, arts_fill_packet_header).  Included
 * unconditionally so the ~200 control-plane call sites that formerly reached it
 * transitively through outbox.h still compile after the outbox is deleted and
 * their include is repointed here. */
#include "arts/transport/protocol.h"

/* ==========================================================================
 * Public control-plane send API.
 *
 * These are the sole names the rest of the runtime calls; the outbox that used
 * to back them is gone.  A remote target rides the fabric
 * (arts_net_send_core); a self-addressed or out-of-range target is
 * warned-and-dropped, preserving the former outbox contract exactly (a caller
 * that wants same-rank delivery uses arts_transport_loopback_post, never these).
 * In a single-node run every target is self/out-of-range, so these degrade to
 * the same warn-and-drop and the fabric is never touched; only the
 * self-loopback path carries traffic.
 * ========================================================================== */
void arts_transport_send_async(int rank, char *message, unsigned int length);
void arts_transport_send_payload_async(int rank, char *message,
                                       unsigned int length, char *payload,
                                       uint64_t size);
void arts_transport_send_payload_async_free(int rank, char *message,
                                            unsigned int length, char *payload,
                                            unsigned int offset, uint64_t size,
                                            void (*free_method)(void *));

/* Self-loopback: post a self-addressed packet for asynchronous delivery on a
 * scheduler / progress tick (arts_transport_loopback_drain), avoiding the
 * unbounded inline-handler recursion a same-rank acquire round would hit. */
void arts_transport_loopback_post(const void *packet, unsigned int size);

/* Deliver every pending self-send through arts_transport_dispatch_body.  In a
 * multinode run the sole progress thread is the only drainer (all coherence
 * stays serialized on it); single-node, each worker drains its own from the
 * scheduler loop.  Serialized to one drainer at a time by a CAS token; returns
 * true if it dispatched at least one. */
bool arts_transport_loopback_drain(void);

/* Free any self-sends still queued at teardown (quiescent: no dispatch). */
void arts_loopback_cleanup(void);

/* Control-plane sizing.  Two multi-recv landing buffers of RECV_BUF_SIZE each
 * catch all two-sided traffic; the provider keeps landing messages into a
 * buffer while its free tail is >= MIN_MULTI_RECV, so the largest fi_send
 * (MSG_MAX) must not exceed that headroom — the three-way invariant
 *     ARTS_NET_MSG_MAX <= ARTS_NET_MIN_MULTI_RECV <= ARTS_NET_RECV_BUF_SIZE
 * guarantees every accepted message lands whole.  This ceiling bounds ONLY
 * control traffic: bulk payloads travel one-sided (arts_net_put_payload /
 * the push rendezvous), which it does not bound — senders holding a payload
 * whose wire total would exceed MSG_MAX must take the rendezvous path. */
#define ARTS_NET_RECV_BUF_SIZE ((size_t)4 * 1024 * 1024)
#define ARTS_NET_MIN_MULTI_RECV (ARTS_NET_RECV_BUF_SIZE / 2)
#define ARTS_NET_MSG_MAX ARTS_NET_MIN_MULTI_RECV

/* Bounded wait for accepted fabric sends to complete, so the shutdown-broadcast
 * frames leave this node before teardown.  No-op single-node. */
void arts_net_drain_outstanding(unsigned int deadline_ms);

/* ==========================================================================
 * Rendezvous one-sided data plane.
 *
 * Bulk payloads move by fi_writedata PUT into a pre-registered landing buffer
 * the RECEIVER advertised: the landing side resolves a regpool pointer to a
 * wire advertisement {raddr, rkey} (arts_net_rdzv_local) and allocates a txid
 * (arts_net_rdzv_txid_next); the sending side PUTs with imm == txid
 * (arts_net_put_payload).  A transfer completes at the receiver when BOTH the
 * metadata packet (a normal control fi_send) and the write completion (the
 * txid immediate) have arrived — in either order (no inter-message ordering is
 * assumed): whichever lands first parks in the txid pairing table and the
 * second fires the continuation registered by arts_net_rdzv_expect.  The one
 * transport guarantee relied on is write-with-immediate atomicity: the
 * target-side immediate is delivered only after the write's bytes are fully
 * placed, so "imm seen => landing buffer valid".
 *
 * txid 0 is reserved as the wire sentinel for "no landing advertised / no
 * payload moved" (txid_next never returns it).
 * ========================================================================== */

/* Allocate a process-unique rendezvous txid: (rank << 48) | counter, never 0. */
uint64_t arts_net_rdzv_txid_next(void);

/* Resolve a registered-pool pointer to its wire landing advertisement.  False
 * when `p` lies in no fabric-registered slab (single-node / regpool without a
 * domain) — the caller must then not attempt a rendezvous.  The
 * address semantics follow the negotiated mr_mode (virtual address under
 * FI_MR_VIRT_ADDR, else offset from the registered base). */
bool arts_net_rdzv_local(const void *p, uint64_t len, uint64_t *raddr,
                         uint64_t *rkey);

/* Register the continuation for txid's payload arrival.  If the write
 * completion already landed, `on_data` fires inline (caller is a dispatch-path
 * progress thread); otherwise it fires when the immediate arrives.  Exactly
 * one expectation per txid. */
void arts_net_rdzv_expect(uint64_t txid, void (*on_data)(void *), void *arg);

/* Generic push rendezvous — sender side.  Used when a bulk payload must reach
 * a peer that did not ask for it (EDT/event moves) and
 * the wire total would exceed the control ceiling: sends RDZV_PUSH_RTS(size),
 * and on the peer's CTS PUTs the payload into the advertised landing, patches
 * {rdzv_txid, rdzv_cookie(, rdzv_size)} into the retained control packet by
 * message type, and sends it.  `free_method` (may be NULL) releases `payload`
 * once the PUT's local completion fires.  Defined in dispatcher.c with the
 * matching RX sides. */
void arts_transport_send_pushed_payload(int rank,
                                        const struct arts_msg_header_s *packet,
                                        unsigned int packet_len, char *payload,
                                        uint64_t size,
                                        void (*free_method)(void *));

/* One-sided PUT of `len` bytes from the registered buffer `src` into the
 * peer's advertised landing {raddr, rkey}, delivering `txid` as the immediate.
 * `on_local_done(arg)` runs once the provider no longer reads `src` (local
 * completion) — the source buffer must stay valid until then.  Thread-safe;
 * same per-thread FIFO/backpressure discipline as the control sends. */
void arts_net_put_payload(int rank, uint64_t raddr, uint64_t rkey,
                          uint64_t txid, const void *src, uint64_t len,
                          void (*on_local_done)(void *), void *arg);

/* Upper bound on a serialized fabric (fi_getname) address blob. */
#define ARTS_NET_ADDR_MAX 256u

/* Accept a bootstrap address frame iff the embedded rank is in range AND the
 * blob length exactly equals this run's fixed provider address length.  Strict
 * equality (not a <= bound) is deliberate: a homogeneous provider run has ONE
 * address length, so any deviation is a corrupt or misframed peer and must be
 * rejected outright, never truncated.  Kept a pure predicate so the bootstrap
 * reader's length bound is unit-testable in isolation from its socket I/O. */
static inline bool arts_net_addr_frame_ok(uint32_t rank, uint32_t len,
                                          unsigned nranks, unsigned own_len) {
  return rank < nranks && len == own_len;
}

struct fid_domain;

/* Bring up the fabric: fi_getinfo(FI_EP_RDM, FI_MSG|FI_RMA, FI_THREAD_SAFE),
 * one fabric/domain/av (FI_AV_TABLE)/cq (FI_CQ_FORMAT_DATA)/ep per rank.  Stores
 * the negotiated provider name, mr_mode, and inject size.  RX is armed
 * separately (arts_net_rx_arm) once the registered pool exists.  Every failure
 * is fatal in place with the fi_strerror text, so there is no error return.
 *
 * `provider`: NULL/empty selects auto (fi_getinfo picks amongst every
 * provider the ambient FI_PROVIDER env var, if any, still admits).  A
 * non-empty value pins fi_getinfo's prov_name hint AND overwrites FI_PROVIDER
 * for this process, so an explicit config choice always wins over whatever
 * was ambient in the environment.
 *
 * `fabric_domain`: NULL/empty selects the provider's first domain.  A
 * non-empty value pins fi_getinfo's domain_attr->name hint — the way to pick
 * one HCA (e.g. "mlx5_0") on a multi-rail host, since RDMA domains are named
 * after the device, not after an IP interface.
 *
 * `net_interface`: NULL/empty leaves the source address unconstrained.  For
 * IP-based providers (tcp, sockets) a non-empty value binds the endpoint's
 * source address to the first AF_INET address of the interface with that
 * exact name (or, failing that, that name prefix) — the way to steer an IP
 * provider onto a specific network (e.g. IPoIB) when the host's default
 * route points elsewhere.  Ignored for non-IP providers, whose addressing is
 * not interface-based.  Naming an interface that has no usable address is
 * fatal: silently falling back to the default route would move all data
 * traffic to the wrong network. */
void arts_net_init(const char *provider, const char *fabric_domain,
                   const char *net_interface);

/* The domain created by arts_net_init, for arts_regpool_init to register slabs
 * against.  NULL before init / after teardown. */
struct fid_domain *arts_net_domain(void);

/* Allocate the multi-recv landing buffers from the registered pool and post
 * them.  Must run AFTER arts_regpool_init (the buffers, and their local
 * descriptor on FI_MR_LOCAL providers, come from the pool) and before any
 * fabric traffic can arrive — i.e. before worker/receiver threads spawn. */
void arts_net_rx_arm(void);

/* Swap every rank's fi_getname address over the already-established TCP data
 * mesh and fi_av_insert the full table in rank order (fi_addr_t == rank).
 * Must run after arts_net_init and after the socket mesh is up; the data
 * sockets stay open afterward. */
void arts_net_exchange_addresses(void);

/* Reap the completion queue (rx dispatch + tx completion frees), repost
 * released multi-recv buffers, and drain the per-thread EAGAIN retry rings.
 * Returns true if it did any work.  Non-blocking: idle pacing is caller
 * policy (the dedicated progress thread busy-polls; a sleeping poll would put
 * a fixed latency floor under every idle-rank message arrival).  Production
 * caller is the progress thread; safe from any thread (e.g. a test's main). */
bool arts_net_progress(void);

/* Teardown phase 1, run BEFORE arts_regpool_cleanup: refuse new sends, discard
 * queued-but-unposted retry txns (their completion-gated frees still run, so no
 * caller payload leaks and bounces return to the still-live pool), reap
 * already-posted sends within a bound, then close ep/cq/av (cancelling posted
 * recvs) and return the recv landing buffers to the pool.  After this, no send
 * desc or recv buffer references a slab MR — so the pool cleanup that follows
 * can close those MRs safely. */
void arts_net_quiesce(void);

/* Teardown phase 2, run AFTER arts_regpool_cleanup: close the domain and
 * fabric.  The registered slab pool's MRs (which back both sends and the recv
 * buffers) are already closed by then, so nothing registered outlives its
 * domain. */
void arts_net_teardown(void);

/* Fabric send core — thread-safe, callable from any thread.  The caller has
 * already screened out self / out-of-range targets (the public wrappers in
 * loopback.c do that, matching the former outbox contract), so `rank` is always
 * a real remote peer here.  A header-only send passes payload==NULL; otherwise
 * `payload+offset` (`size` bytes) is appended, released by `free_method` (may be
 * NULL) once the send completes.  A total wire size exceeding what the RX
 * landing buffer can accept fails loudly (ARTS_ERROR) rather than corrupting. */
void arts_net_send_core(int rank, char *message, unsigned int length,
                        char *payload, unsigned int offset, uint64_t size,
                        void (*free_method)(void *));

/* Bootstrap glue used by bootstrap.c (kept here so the exchange lives in its
 * own translation unit while the fabric state stays private to net.c). */

/* Copy this rank's fi_getname blob into `buf` (capacity `buflen`); returns its
 * byte length, or 0 on failure. */
unsigned arts_net_own_address(void *buf, unsigned buflen);

/* fi_av_insert `count` fixed-size (`addrlen`) addresses laid contiguously in
 * `addrs`, in rank order, storing the resulting fi_addr_t table (fi_addr_t i
 * must equal rank i under FI_AV_TABLE — asserted). */
void arts_net_av_insert_table(const void *addrs, unsigned addrlen,
                              unsigned count);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_TRANSPORT_NET_H */
