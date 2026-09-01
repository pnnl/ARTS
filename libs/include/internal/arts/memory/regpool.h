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
#ifndef ARTS_MEMORY_REGPOOL_H
#define ARTS_MEMORY_REGPOOL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h> /* FILE — the node-availability parser is stream-pure */

/* Registered slab pool.
 *
 * Payload buffers destined for one-sided RDMA must live in memory that is
 * pinned and pre-registered with the fabric, so the NIC can target it with no
 * per-allocation registration.  The pool carves large NUMA-bound slabs, pins
 * each behind a memory-registration handle, and hands the slab to an allocator
 * arena; every pointer it returns therefore falls inside exactly one
 * registered slab and can be resolved back to its registration handle and
 * remote key.
 *
 * Confinement is a hard invariant: a pointer that cannot be resolved by
 * arts_regpool_lookup would be an unregistered address the NIC cannot reach,
 * so the pool fails loudly rather than return one. */

/* Forward declarations of the libfabric object handles keep fabric headers out
 * of every includer of this header.  When the runtime is built without the OFI
 * transport, or the pool is initialized with a NULL domain, these stay opaque
 * and unused: `mr` is NULL and `rkey` is 0. */
struct fid_domain;
struct fid_ep;
struct fid_mr;

/* One registered slab.  A slab backs either an allocator arena (many
 * allocations) or a single oversize direct allocation.  `arts_regpool_lookup`
 * returns a pointer to the record covering a payload pointer; the record
 * exposes the registration handle and remote key that describe the enclosing
 * pinned range. */
typedef struct arts_regpool_mr_s {
  void *base;        /* first byte of the registered range                    */
  size_t len;        /* length of the registered range                        */
  struct fid_mr *mr; /* registration handle; NULL when unregistered           */
  uint64_t rkey;     /* remote key for one-sided RDMA; 0 when unregistered     */
  int numa_node;     /* NUMA node the range is bound to                        */
} arts_regpool_mr_t;

/* Initialize the pool: detect NUMA topology (numa_nodes==0 auto-detects),
 * carve one slab per node (best-effort: a node without room is refused with
 * a warning and left without an arena — its threads are then served from a
 * fallback node's arena and the node joins the pool when a later demand-time
 * grow succeeds; only ZERO carved nodes fails the init), bind each slab,
 * register it against `domain_or_null`, and hand it to a per-node allocator
 * arena.  A NULL domain skips registration (single-node runs / unit tests)
 * while carving arenas identically, so allocation behavior is unchanged.  A non-NULL `ep_or_null` selects the
 * endpoint-bound registration discipline some providers require
 * (FI_MR_ENDPOINT): each slab MR is bound to that endpoint and enabled after
 * registration, and its remote key is read only after the enable — the
 * endpoint must already be enabled, and it must outlive every registered
 * slab (see arts_regpool_unregister).  `slab_bytes` is rounded up to the
 * allocator's minimum arena granularity.  Returns false if already
 * initialized or a slab could not be mapped/registered. */
bool arts_regpool_init(struct fid_domain *domain_or_null,
                       struct fid_ep *ep_or_null, size_t slab_bytes,
                       unsigned int numa_nodes);

/* Release pool bookkeeping and unregister every slab.  Must be called only at
 * teardown with no thread still allocating from the pool. */
void arts_regpool_cleanup(void);

/* Close every live slab registration and detach the pool from the fabric
 * (domain and endpoint references cleared): later allocations still succeed
 * but are no longer fabric-registered, and later cleanup skips the closed
 * handles.  Exists for endpoint-bound registrations, whose MRs hold
 * references the endpoint cannot close under — the transport calls this
 * before closing its endpoint.  Single-threaded teardown only, like
 * arts_regpool_cleanup. */
void arts_regpool_unregister(void);

/* Allocate `size` bytes aligned to at least `align` (floored to the payload
 * alignment invariant) from the calling thread's NUMA-local arena, growing the
 * pool on demand.  Oversize requests take a dedicated direct-slab path.  The
 * returned pointer is guaranteed to resolve through arts_regpool_lookup; a
 * pointer that escaped the registered slabs is a fatal error. */
void *arts_regpool_alloc_aligned(size_t size, size_t align);

/* As arts_regpool_alloc_aligned, but the returned bytes are zero.  Prefer
 * this over alloc+memset for zero-initialized payloads: fresh slab memory is
 * already kernel-zeroed, so only blocks recycled from dirty pages are
 * actually cleared — the full-payload touch (and its page faults) leaves the
 * caller's critical path. */
void *arts_regpool_zalloc_aligned(size_t size, size_t align);

/* Return a pointer previously obtained from arts_regpool_alloc_aligned. */
void arts_regpool_free(void *p);

/* Resolve `p` to the registered slab that contains it, or NULL if `p` lies in
 * no registered slab (an escaped pointer).  Lock-free; safe to call
 * concurrently with allocation. */
const arts_regpool_mr_t *arts_regpool_lookup(const void *p);

/* Map, bind, register, and publish one additional slab for `numa_node`.
 * Called automatically when an arena is exhausted; exposed so callers can
 * pre-grow.  Returns false if the slab could not be created. */
bool arts_regpool_grow(int numa_node);

/* Availability estimate over one NUMA node's meminfo stream: free pages
 * plus reclaimable file cache (the file LRU lists minus writeback-bound
 * pages, discounted by half — the kernel's own MemAvailable haircut), so a
 * cache-heavy node is not misjudged as full while an anon-full node still
 * reports ~MemFree.  SIZE_MAX when MemFree cannot be read (unknown must
 * not veto growth); MemFree alone when the LRU fields are absent.  Pure
 * over the stream — exposed for hermetic testing. */
size_t arts_regpool_parse_node_avail(FILE *f);

#ifdef __cplusplus
}
#endif
#endif /* ARTS_MEMORY_REGPOOL_H */
