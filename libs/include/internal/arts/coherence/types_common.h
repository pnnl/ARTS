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
******************************************************************************/
#ifndef ARTS_MEMORY_COHERENCE_TYPES_COMMON_H
#define ARTS_MEMORY_COHERENCE_TYPES_COMMON_H
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file types_common.h
 * @brief Protocol-agnostic DB-coherence layout types shared by every
 *        ARTS_COHERENCE_PROTOCOL build (RCU / WRF_RCU).
 *
 * The protocol-specific cache/db ownership layout lives in the per-protocol
 * coherence/<proto>/types.h, selected by arts/coherence/types.h.  Anything
 * here is identical across all protocols: the buffer, the RO snapshot
 * reorder-buffer node, the shared defines/typedefs, and container_of.
 *
 * @note Internal header.  User code should include @c arts.h.
 */

#include "arts.h"

#include "arts/defs.h"
#include "arts/utils/lockfree_lifo.h"  /* arts_lf_stack_t / arts_lf_link_t */
#include "arts/utils/lockfree_pool.h"  /* arts_lockfree_pool_t (buf_freelist) */
#include "arts/utils/lockfree_stack.h" /* arts_lockfree_stack_t */
#include "arts/utils/mpsc.h"           /* arts_mpsc_t */
#include "arts/utils/shared.h"         /* arts_shared_ptr_t (buffer cb) */
#include <stdbool.h>
#include <stddef.h> /* offsetof — container_of(cache, arts_db_s, cache) */
#include <stdint.h>
#ifndef __cplusplus
#include <stdatomic.h>
#endif
/* Lazy home metadata embeds a per-rank reader bit-set by value. */
#ifdef ARTS_TIMING_LAZY
#include "arts/rank_bitset.h"
#endif

/* Sentinel for arts_db_cache_s.incoming_new_owner meaning "no ownership
 * transfer pending".  A real rank is always < rank_count, so UINT_MAX is a safe
 * out-of-band value (and rank 0 is a valid owner, so 0 cannot be the sentinel).
 */
#define ARTS_LAZY_NO_PENDING_OWNER ((unsigned int)-1)

/* In-memory rendezvous landing descriptor — the coherence-side mirror of the
 * packed wire struct arts_msg_rdzv_landing_s (protocol.h): where a payload
 * sender may fi_writedata, keyed for pairing.  txid == 0 means "no landing"
 * everywhere (data-less round / size not yet known).  cookie is the
 * advertiser-local landing-buffer pointer, only ever dereferenced back on the
 * advertiser rank. */
struct arts_rdzv_landing_s {
  uint64_t addr;
  uint64_t key;
  uint64_t txid;
  uint64_t cookie;
};

/* Portable atomic unsigned-int for struct fields visible to both C and the
 * C++/nvcc layout-only TUs (which cannot parse C11 _Atomic).  C accesses these
 * via arts_atomic_* on the underlying uint; nvcc only needs the layout. */
#ifdef __cplusplus
typedef unsigned int arts_db_atomic_uint_t;
#else
typedef _Atomic(unsigned int) arts_db_atomic_uint_t;
#endif

/*--- Buffer ---------------------------------------------------------------
 * Holds version + user-visible data bytes (FAM).  Lifetime is managed by an
 * arts_shared_ptr_t control block (cache.buffer is the atomic slot; each
 * acquirer holds a strong ref).  No embedded refcount: the cb's strong count
 * IS the "cache-hold + per-acquirer" count, and the cb deleter frees the
 * buffer once the last holder releases — so a destroy concurrent with an
 * in-flight acquire can never free the bytes out from under a reader.
 *   version  monotonic per-buffer version stamp.
 *   cb       this buffer's own control block (== the slot's cb while
 *            installed).  An EDT recovers it via buf_from_data(dep->ptr)->cb
 *            to drop its acquire ref at release — safe because the EDT's own
 *            ref keeps the buffer (hence buf->cb) alive until that release.
 *   data     FAM holding db_size bytes — user-visible canonical payload,
 *            64-byte aligned (cache-line / CXL atomicity). */
struct arts_db_buffer_s {
  arts_lf_link_t pool_link; /* FIRST — per-DB free-list node when recycled */
  uint64_t version;         /* monotonic per buffer (stale while free-listed) */
  arts_shared_ptr_t cb;     /* this buffer's control block */
  struct arts_db_cache_s
      *owner_cache; /* deleter pushes here; read only at strong==0 */
  char _pad[32];    /* data[] lands at offset 64 (cache-line aligned) */
  char data[];      /* db_size bytes — user-visible */
};

/* Forward decl; per-rank cache (owner of the per-DB buffer free-list). */
struct arts_db_cache_s;

/* Snapshot reorder-buffer node, three parkers share it:
 *   1. requester-side case 3 of arts_handler_db_snapshot_response (a NO_DATA
 *      reply arrived with version > buf->version — transport reordered the
 *      with-data reply behind it);
 *   2. a home-local RO acquire that found no buffer installed yet (cross-rank
 *      create installs home metadata only; the creator's first WRITEBACK is
 *      the publication point that installs the first buffer);
 *   3. a home-side remote GET_DATA serve deferred for the same reason —
 *      `serve` is non-NULL and the drain re-issues the serve with the
 *      original requester/rdzv instead of resuming a local EDT.
 * Drained in full by the next install via a single atomic_exchange on the
 * Treiber stack — monotonic version guarantees every parked node's
 * target_version <= the just-installed version.  RO path; present in every
 * protocol (WRF_RCU parks all modes here). */
struct arts_db_snapshot_waiter_s {
  arts_lf_link_t link; /* FIRST — required by arts_lf_stack_t */
  arts_guid_t edt_guid;
  unsigned int slot;
  uint64_t target_version;
  /* Deferred-serve arm: NULL = local waiter (drain resumes the parked EDT;
   * the resume re-derives dep->ptr from the installed buffer).  Non-NULL =
   * deferred remote serve; the drain invokes serve(cache, w) BEFORE freeing
   * w, and requester/rdzv carry the original request. */
  void (*serve)(struct arts_db_cache_s *cache,
                struct arts_db_snapshot_waiter_s *w);
  unsigned int requester;
  struct arts_rdzv_landing_s rdzv;
};
/* Forward decl; sparse rank-keyed u64 map (owner-side dedup; protocol arms). */
struct arts_rank_to_u64_map_s;

/* Recover the wrapping struct arts_db_s from a coherence cache pointer.  cache
 * is the FIRST member of arts_db_s; container_of degenerates to the cache
 * address but is written as container_of for correctness-by-construction. */
#ifndef ARTS_CONTAINER_OF
#define ARTS_CONTAINER_OF(ptr, type, member)                                   \
  ((type *)((char *)(ptr) - offsetof(type, member)))
#endif

#ifdef __cplusplus
}
#endif

#endif /* ARTS_MEMORY_COHERENCE_TYPES_COMMON_H */
