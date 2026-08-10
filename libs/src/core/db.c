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

#include "arts/db.h"

#include <assert.h>
#include <string.h>

#include "arts.h"
#include "arts/cxl/wrapper.h"
#ifdef ARTS_USE_CXL
#include "arts/cxl/deque.h"
#endif
#include "arts/coherence/buffer.h"
#include "arts/coherence/coherence.h"
#include "arts/coherence/handlers.h"
#include "arts/counter/Preamble.h"
#include "arts/counter/object_counter.h"
#include "arts/edt.h"
#include "arts/edt_context.h" /* current_edt + created-DB tracking */
#include "arts/gas/guid.h"
#include "arts/gas/route_table.h"
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h" /* arts_shared_ptr_t, get/release */

#ifdef ARTS_USE_GPU
#include "arts/gpu/gpu_internal.h"
#endif

/* No-hint DB home policy: CREATOR (default) keeps the home on the creating
 * rank — first-touch, so a block inherits whatever distribution the placement
 * of its creating task achieved, and create/destroy directory traffic stays
 * local.  ROUNDROBIN distributes homes across all ranks regardless of the
 * creation site.  An explicit hint rank or a pre-reserved GUID always wins.
 * CMake sets this for every libarts compile; the fallback covers any TU that
 * pulls in db.c outside the normal build (e.g. direct inclusion). */
#ifndef ARTS_NOHINT_DB_ROUNDROBIN
#define ARTS_NOHINT_DB_ROUNDROBIN 0
#endif

ARTS_TYPE_NAME;
ARTS_DB_TYPE_NAME;
DB_MODE_NAME;

/*
 * arts_db_user_ptr — Return the user-visible data pointer for a DB.
 *
 * For coherent ARTS_DB datablocks this is cache->buffer->data (the
 * canonical payload installed by arts_db_buf_install at create time or
 * by GRANT/DATA_RESPONSE on sharer ranks).  For all other subtypes it
 * is the legacy (db+1) pointer.
 *
 * Returns NULL if `db` itself is NULL or if a coherent ARTS_DB has no
 * buffer installed yet (transient at create-time; callers should treat
 * this as "data not yet available").
 */
void *arts_db_user_ptr(struct arts_db_s *db) {
  if (db == NULL) {
    return NULL;
  }
  if (db->db_type == ARTS_DB) {
    /* Coherent ARTS_DB: canonical payload lives in the installed buffer's
     * data, not at (db+1).  Single-owner context: acquire a ref, read the
     * canonical payload pointer, release.  buf->data is the buffer's FAM and
     * stays valid for the single owner that consumes the returned pointer. */
    arts_shared_ptr_t buf_h = arts_db_buf_acquire(&db->cache);
    struct arts_db_buffer_s *buf =
        (struct arts_db_buffer_s *)arts_shared_get(buf_h);
    void *ptr = buf ? (void *)buf->data : NULL;
    arts_db_buf_release(&buf_h);
    return ptr;
  }
  return (void *)(db + 1);
}

/*
 * arts_db_auto_acquire — Track the creator EDT's hold on a DB.
 *
 * For ARTS_DB: the coherence cache_s was allocated with writer_count=2
 * (sentinel + creator EDT) via ARTS_DB_INIT_CREATOR_HOME or
 * ARTS_DB_INIT_CREATOR_REMOTE, so the creator's hold is already
 * counted in the coherence state machine.  release_rw drops it at EDT
 * epilogue.
 *
 * For pinned subtypes (PIN, GPU_PIN, GPU, CXL): there is no
 * DB-level coherence to track; the creator EDT just owns the pointer
 * until it explicitly destroys or hands it off via events.
 *
 * In both cases the GUID is recorded on created_db_list so the EDT
 * epilogue (arts_release_created_dbs) drives the matching release.
 */
static void arts_db_auto_acquire(struct arts_db_s *db) {
  arts_track_created_db(db->cache.db_guid);
}

/* arts_db_creator_skip_hold — should the coherent (ARTS_DB) creator EDT be kept
 * OFF created_db_list?  Under the RWLOCK protocol the creator takes no implicit
 * lock: the home rank is the sole arbiter and zero-inits the buffer at create
 * time, and a writer only ever holds the lock via a granted LOCK_REQUEST (which
 * bumps the per-rank cache_state rw_count + sets rw_state=GRANT).  A
 * create-time stub has neither, so registering the creator on created_db_list
 * would make the EDT epilogue (arts_release_created_dbs -> release_one_created
 * -> release_rw) run a release on a hold that was never granted.  When a
 * same-rank worker has concurrently JOINed (raising local_count), that bogus
 * release steals the worker's count, drives the 0-edge, and ships a stale
 * publish to home as a spurious RW_REL — corrupting home's lock_state w
 * counter and overwriting the worker's update (the cross-rank lost-update). The
 * creator's stub buffer is still installed (so the user pointer is writable);
 * it just is not tracked for an auto-release.  A creator that must publish
 * initial data does so through a normal RW dependency, like any other writer.
 * Pinned subtypes still need created_db_list (destroy bookkeeping); the
 * single-owner protocols keep the legitimate creator-owns-until-release hold
 * (writer_count pre-stamped to 2).
 */
static inline bool arts_db_creator_skip_hold(arts_db_types_t db_type) {
  /* No protocol skips the creator hold any more: arts_db_create defaults to an
   * RW acquire for every coherent DB, and each protocol seeds that hold at
   * create time (single-owner: writer_count=2; RWLOCK: cache_state RW-GRANT +
   * lock_state w=1).  The matching release (explicit or EDT-epilogue
   * auto-release) drives it back, so the creator is tracked like any holder. */
  (void)db_type;
  return false;
}

void *arts_db_malloc(arts_db_types_t db_type, size_t size) {
  (void)db_type;
  void *ptr = NULL;
#ifdef ARTS_USE_GPU
  if (arts_node_info.gpu) {
    if (db_type == ARTS_DB_GPU)
      ptr = arts_cuda_malloc_host(size * 2);
    else if (db_type == ARTS_DB_GPU_PIN)
      ptr = arts_cuda_malloc_host(size);
  }
#endif
#ifdef ARTS_USE_CXL
  if (db_type == ARTS_DB_CXL) {
    unsigned int dev_idx;
    if (arts_node_info.cxl_db_dev_count > 1) {
      /* Round-robin: atomically advance the index and wrap around. */
      dev_idx = arts_atomic_fetch_add(&arts_node_info.cxl_db_rr_idx, 1U) %
                arts_node_info.cxl_db_dev_count;
    } else {
      /* Static: use the configured device. */
      dev_idx = arts_node_info.cxl_db_static_device;
    }
    ptr = arts_cxl_deque_db_malloc_dev(arts_node_info.cxl_deque,
                                       &arts_node_info.cxl_local_lock, size,
                                       dev_idx);
    assert(ptr && "arts_cxl_deque_db_malloc_dev ptr is valid\n");
  }
#endif
  if (!ptr) {
    /* Full DataBlock backing store: the arts_db_s header sits at offset 0, so
     * it must be cache-line aligned (shared, multi-thread object). */
    ptr = arts_malloc_aligned(size, ARTS_CACHE_LINE_SIZE);
  }
  return ptr;
}

void arts_db_free(void *ptr) {
  struct arts_db_s *db = (struct arts_db_s *)ptr;
  /* Chain into coherence cache teardown if this DB has one.  Only ARTS_DB
   * carries coherence state; other subtypes leave the embedded cache zeroed.
   * The cache is embedded by value as the first member of db_s, so the
   * destructor tears down its sub-resources (the buffer slot's shared_ptr ref +
   * home_s) in place — we do NOT free it separately; the db_s free below
   * reclaims its storage. */
  if (db->db_type == ARTS_DB) {
    arts_db_cache_destructor(&db->cache);
  }
#ifdef ARTS_USE_GPU
  if (arts_node_info.gpu &&
      (db->db_type == ARTS_DB_GPU_PIN || db->db_type == ARTS_DB_GPU)) {
    arts_cuda_free_host(ptr);
    ptr = NULL;
  }
#endif
  if (ptr) {
    arts_free(ptr);
  }
}

/*
 * arts_db_deleter — shared_t deleter (called by route_table.c free_item once
 * the slot's lock count hits 0 with DELETE set).  Entry point that
 * consolidates DB teardown through the route_table generic dispatch.  The
 * cleanup logic itself lives in arts_db_free (hierarchical: cache_destructor
 * → cache_s → GPU-host free → struct).
 */
/* cb deleter (route_table deleter-by-kind for ARTS_GUID_DB).  External
 * linkage so route_table.c references it directly.  `self` is the cb object
 * pointer = &db->cache (cache is the first member), which aliases arts_db_s. */
void arts_db_deleter(void *self) { arts_db_free(self); }

/* Publish the DB cb deleter into the route_table's per-kind table at startup
 * (decoupled registration — see arts_route_table_register_deleter). */
__attribute__((constructor)) static void arts_db_register_cb_deleter(void) {
  arts_route_table_register_deleter(ARTS_GUID_DB, arts_db_deleter);
}

/*
 * db_create_in_place — Initialize a DB header in pre-allocated memory.
 *
 * Sets up the arts_db_s header fields (type, size, version, reader/writer
 * counts, db_list) and records metrics.  The caller is responsible for
 * route-table registration.
 */
static void db_create_in_place(arts_guid_t guid, void *addr, uint64_t len,
                               uint64_t packet_size, arts_db_types_t db_type) {
  (void)len;
  struct arts_db_s *db_res = (struct arts_db_s *)addr;
  /* lifecycle/deleter handled by the route_table cb (deleter-by-kind) on
   * install — no per-object shared field to initialize. */
  db_res->version = 0;
  db_res->reader = 0;
  db_res->writer = 0;
  db_res->db_type = db_type;
  /* ARTS_DB enters the coherence protocol at create time.  Initialize the
   * embedded cache (first member of db_s) with CREATOR_HOME init —
   * arts_db_create only routes here when the local rank is the creator
   * (route == arts_global_rank_id), which for round-robin home is also the
   * home rank.  Non-coherent subtypes leave the embedded cache zeroed.
   *
   * Note: db_create_in_place is called only on the local-create
   * branch of arts_db_create; the remote-create branch builds its own
   * stub directly and does NOT invoke this routine. */
  /* Every DB — coherent or pinned — records its payload length in the cache.
   * Size lives here (cache->db_size), not in a separate per-object header
   * (arts_header_s removed): a DB always carries its own length. */
  uint64_t user_size = packet_size - sizeof(struct arts_db_s);
  struct arts_db_cache_s *cache = &db_res->cache;
  /* addr came from arts_db_malloc (not necessarily zeroed); zero the embedded
   * cache before in-place init. */
  memset(cache, 0, sizeof(*cache));
  cache->db_size = user_size;
  cache->db_guid = guid;
  if (db_type == ARTS_DB) {
    arts_db_cache_init(cache, guid, user_size, ARTS_DB_INIT_CREATOR_HOME,
                       arts_global_rank_id);
    /* Install a fresh buffer so subsequent coherent acquires
     * (acquire_local / mark_edt_ready_by_guid) find a non-NULL
     * cache->buffer.  The user pointer returned by arts_db_create
     * points into this buffer's data[] FAM, so writes by the creator
     * EDT land in buf->data and are published when release_rw bumps
     * the version.  No initial data — zero-init is deterministic and
     * matches the DB_CREATE_COHERENT home-recv path. */
    if (user_size > 0) {
      arts_db_buf_install(cache, /*new_version=*/1,
                          /*data_payload=*/NULL, user_size);
    }
  }
  /* Non-coherent subtypes (ARTS_DB_PIN, ARTS_DB_GPU_PIN, ARTS_DB_GPU,
   * ARTS_DB_CXL) are pinned to the creator rank and have no DB-level
   * coherence.  The embedded cache stays zeroed and db_list stays NULL. */
  if (db_type == ARTS_DB_GPU) {
    void *shadow_copy = (void *)(((char *)addr) + packet_size);
    memcpy(shadow_copy, addr, sizeof(struct arts_db_s));
  }
  INCREMENT_NUM_DB_CREATE_BY(1);
  INCREMENT_BYTES_DB_CREATE_BY(len);
}

/*
 * arts_db_create — Unified DataBlock creation.
 *
 * Handles all DB subtypes (ARTS_DB, ARTS_DB_PIN, ARTS_DB_GPU_PIN, ARTS_DB_GPU,
 * ARTS_DB_CXL).  When hint->rank targets a remote node, only ARTS_DB is
 * supported: a coherent home stub is installed via DB_CREATE_COHERENT.
 * Pinned subtypes return NULL_GUID with a warning.
 */
arts_guid_t arts_db_create(void **addr, uint64_t len, arts_db_types_t db_type,
                           uint16_t flags, const arts_db_hint_t *hint) {
  TIME_DB_CREATE_START();
  /* Route resolution:
   *   hint == NULL                          -> policy-selected home
   *                                             (ARTS_NOHINT_DB_ROUNDROBIN;
   *                                             creator-local by default).
   *   hint->rank == ARTS_HINT_CURRENT_RANK -> caller explicitly requested
   *                                             current node.
   *   hint->rank == specific rank          -> caller-specified rank.
   */
  /* If the caller supplies a pre-reserved GUID its encoded rank is
   * authoritative and overrides hint->rank. */
  arts_guid_t pre_guid = (hint != NULL) ? hint->guid : NULL_GUID;
  /* CHECK / rendezvous: fail (keep existing) instead of overwriting a live
   * labeled GUID.  Default false = unconditional replace on reuse. */
  bool check = (hint != NULL) ? hint->check : false;
  unsigned int rank;
  if (pre_guid != NULL_GUID) {
    rank = arts_guid_get_rank(pre_guid);
  } else if (hint == NULL) {
#if ARTS_NOHINT_DB_ROUNDROBIN
    rank = arts_atomic_fetch_add(&arts_node_info.db_rr_route, 1U) %
           arts_global_rank_count;
#else
    rank = arts_global_rank_id;
#endif
  } else if (hint->rank == ARTS_HINT_CURRENT_RANK) {
    rank = arts_global_rank_id;
  } else {
    rank = hint->rank;
  }
  bool no_acquire = (flags & ARTS_DB_PROP_NO_ACQUIRE) != 0;
  arts_guid_t guid = NULL_GUID;

  if (rank == arts_global_rank_id) {
    uint64_t db_size = len + sizeof(struct arts_db_s);
#ifdef ARTS_USE_CXL
    if (db_type == ARTS_DB_CXL) {
      db_size = ALIGN_UP(db_size, CACHELINE_SIZE);
      void *ptr = arts_db_malloc(ARTS_DB_CXL, db_size);
      if (ptr) {
        guid = arts_cxl_make_guid(ptr);
        db_create_in_place(guid, ptr, len, db_size, ARTS_DB_CXL);
        /* No route table entry — GUID encodes CXL pointer directly */
        // FLUSH_FENCE_PRODUCER(ptr, db_size);
        FLUSH_FENCE_PRODUCER(ptr, sizeof(struct arts_db_s));
        *addr = no_acquire ? NULL : (void *)((struct arts_db_s *)ptr + 1);
        ARTS_DEBUG("arts_db_create: CXL DB[Guid:%lu, Size:%lu] created", guid,
                   len);
      }
    } else
#endif
    {
      void *ptr = arts_db_malloc(db_type, db_size);
      if (ptr) {
        if (pre_guid != NULL_GUID) {
          /* Pre-reserved labeled GUID. */
          guid = pre_guid;
          db_create_in_place(guid, ptr, len, db_size, db_type);
          /* Register the creator's hold BEFORE the DB becomes visible, then
           * install — both install variants fire the OoO list internally on a
           * successful install (no separate fire_oo needed). */
          if (current_edt && !no_acquire &&
              !arts_db_creator_skip_hold(db_type)) {
            arts_db_auto_acquire((struct arts_db_s *)ptr);
          } else if (no_acquire && db_type == ARTS_DB) {
            /* NO_ACQUIRE coherent: the creator never acquires or releases, so
             * home is the sole idle owner.  db_create_in_place pre-stamped the
             * coherent writer_count to 2 (sentinel + creator-hold), but no EDT
             * tracks or releases that hold (auto-acquire is skipped just
             * above), so drop it to the sentinel (1) before the DB becomes
             * visible at install.  Without this the unreleased creator-hold
             * blocks every future writer under a single-writer protocol — the
             * same reason the remote DB_CREATE handler stamps writer_count = 1
             * for NO_ACQUIRE. */
#if defined(ARTS_PROTOCOL_EXCL)
#if defined(ARTS_RELEASE_RETAIN)
            /* OWNER placement: data lives with the owner, not the home — with no creator
             * hold there is no owner unless we make one.  This rank (the GUID
             * home, where a local create runs) becomes the IDLE data owner: it
             * holds the zero-init buffer (installed by db_create_in_place) with
             * owner-bit set but rw_st=IDLE, wc=0.  The first writer's REQUEST
             * migrates that zero buffer from here.  lock_state is the idle
             * directory naming this rank as owner. */
            atomic_store_explicit(&((struct arts_db_s *)ptr)->cache.cache_state,
                                  CACHE_MAKE_FULL(1u, CACHE_ST_IDLE,
                                                  CACHE_ST_IDLE,
                                                  ARTS_LOCK_NO_TARGET, 0u, 0u),
                                  memory_order_relaxed);
            atomic_store_explicit(
                &((struct arts_db_s *)ptr)->lock_state,
                LOCK_MAKE(LOCK_PHASE_IDLE, arts_global_rank_id, 0u, 0u),
                memory_order_relaxed);
#else  /* ARTS_RELEASE_PURGE */
            /* HOME placement: the home holds the canonical buffer; undo the create-time
             * creator RW seed → free lock, so the first acquirer is granted
             * rather than blocked behind a hold no EDT will ever release. */
            atomic_store_explicit(&((struct arts_db_s *)ptr)->cache.cache_state,
                                  0ULL, memory_order_relaxed);
            atomic_store_explicit(&((struct arts_db_s *)ptr)->lock_state, 0ULL,
                                  memory_order_relaxed);
#endif /* ARTS_RELEASE_* */
#else
            /* Grant-bearing arms: this rank keeps the sentinel and becomes the
             * idle owner.  With no creator hold there is nothing to release,
             * so the first foreign request revokes an idle grant rather than
             * queueing behind a hold nobody will ever drop. */
            ((struct arts_db_s *)ptr)->cache.writer_count = 1;
#endif
          }
          if (check) {
            /* CHECK / rendezvous: first-wins install — concurrent installs with
             * the same GUID are safe (a later one keeps the existing). */
            arts_route_table_install_if_absent(ptr, guid, arts_global_rank_id,
                                               true);
          } else {
            /* Default: unconditional replace — a labeled-GUID reuse overwrites
             * the prior generation (the displaced cb is released). */
            arts_route_table_install(ptr, guid, arts_global_rank_id, true);
          }
        } else {
          guid = arts_guid_create_for_rank(arts_global_rank_id, ARTS_GUID_DB);
          db_create_in_place(guid, ptr, len, db_size, db_type);
          if (current_edt && !no_acquire &&
              !arts_db_creator_skip_hold(db_type)) {
            arts_db_auto_acquire((struct arts_db_s *)ptr);
          } else if (no_acquire && db_type == ARTS_DB) {
            /* NO_ACQUIRE coherent: the creator never acquires or releases, so
             * home is the sole idle owner.  db_create_in_place pre-stamped the
             * coherent writer_count to 2 (sentinel + creator-hold), but no EDT
             * tracks or releases that hold (auto-acquire is skipped just
             * above), so drop it to the sentinel (1) before the DB becomes
             * visible at install.  Without this the unreleased creator-hold
             * blocks every future writer under a single-writer protocol — the
             * same reason the remote DB_CREATE handler stamps writer_count = 1
             * for NO_ACQUIRE. */
#if defined(ARTS_PROTOCOL_EXCL)
#if defined(ARTS_RELEASE_RETAIN)
            /* OWNER placement: data lives with the owner, not the home — with no creator
             * hold there is no owner unless we make one.  This rank (the GUID
             * home, where a local create runs) becomes the IDLE data owner: it
             * holds the zero-init buffer (installed by db_create_in_place) with
             * owner-bit set but rw_st=IDLE, wc=0.  The first writer's REQUEST
             * migrates that zero buffer from here.  lock_state is the idle
             * directory naming this rank as owner. */
            atomic_store_explicit(&((struct arts_db_s *)ptr)->cache.cache_state,
                                  CACHE_MAKE_FULL(1u, CACHE_ST_IDLE,
                                                  CACHE_ST_IDLE,
                                                  ARTS_LOCK_NO_TARGET, 0u, 0u),
                                  memory_order_relaxed);
            atomic_store_explicit(
                &((struct arts_db_s *)ptr)->lock_state,
                LOCK_MAKE(LOCK_PHASE_IDLE, arts_global_rank_id, 0u, 0u),
                memory_order_relaxed);
#else  /* ARTS_RELEASE_PURGE */
            /* HOME placement: the home holds the canonical buffer; undo the create-time
             * creator RW seed → free lock, so the first acquirer is granted
             * rather than blocked behind a hold no EDT will ever release. */
            atomic_store_explicit(&((struct arts_db_s *)ptr)->cache.cache_state,
                                  0ULL, memory_order_relaxed);
            atomic_store_explicit(&((struct arts_db_s *)ptr)->lock_state, 0ULL,
                                  memory_order_relaxed);
#endif /* ARTS_RELEASE_* */
#else
            /* Grant-bearing arms: this rank keeps the sentinel and becomes the
             * idle owner.  With no creator hold there is nothing to release,
             * so the first foreign request revokes an idle grant rather than
             * queueing behind a hold nobody will ever drop. */
            ((struct arts_db_s *)ptr)->cache.writer_count = 1;
#endif
          }
          arts_route_table_install(ptr, guid, arts_global_rank_id, true);
        }
        /* For coherent ARTS_DB the canonical user data lives
         * in cache->buffer->data (installed by db_create_in_place),
         * not at (db+1).  arts_db_user_ptr returns the right pointer
         * for both worlds.  NO_ACQUIRE: return NULL so caller cannot
         * accidentally write to the buffer (home is the idle owner). */
        *addr = no_acquire ? NULL : arts_db_user_ptr((struct arts_db_s *)ptr);
        ARTS_DEBUG("arts_db_create: DB[Guid:%lu, Type:%s, Size:%lu] "
                   "created locally",
                   guid, GET_DB_TYPE_NAME(db_type), len);
      }
    }
  } else {
    /* Pre-reserved (labeled) GUID with remote home: use it verbatim so the
     * remote install lands at the application-visible GUID.  Otherwise
     * (NULL hint round-robin / explicit-rank hint) generate a fresh
     * auto-GUID on the home rank's key counter. */
    guid = (pre_guid != NULL_GUID)
               ? pre_guid
               : arts_guid_create_for_rank(rank, ARTS_GUID_DB);
    if (db_type == ARTS_DB) {
      /* For ARTS_DB, ask the home rank to install a coherent cache_s
       * via DB_CREATE_COHERENT.  The home handler
       * (arts_handler_db_create) allocates its own stub +
       * cache_s with ARTS_DB_INIT_HOME_RECV.
       *
       * Also stub-install a creator-side cache_s on this (non-home)
       * rank via arts_db_cache_stub_install.  This is necessary so
       * that home's first INVALIDATE_NOTICE (sent to
       * rw_holder = creator_rank when a foreign OWNERSHIP_REQUEST arrives)
       * finds a cache_s on this rank to drop the sentinel and trigger
       * invalidate_transfer.  Without it, home's invalidation goes to a
       * phantom holder and the first foreign acquirer stalls forever. */
      /* Use ARTS_DB_INIT_CREATOR_REMOTE (writer_count = 2: sentinel +
       * creator EDT) so that the home-side INVALIDATE_NOTICE round-trip
       * works correctly.  When a foreign OWNERSHIP_REQUEST arrives at home,
       * home sends INVALIDATE_NOTICE to rw_holder = creator; creator's
       * fetch_sub takes wc 2 -> 1 (no transfer yet -- creator EDT may
       * still be using the buffer).  Creator EDT release_rw drops wc
       * 1 -> 0, triggering R4 PUBLISH_AND_TRANSFER with the creator's
       * data.  wc = 1 (the implementer's earlier choice) was wrong: it
       * would trigger the transfer immediately on INVALIDATE_NOTICE
       * while the creator EDT was still writing.
       *
       * The cache is built privately (CREATOR_REMOTE init -> wc = 2) and
       * only then published via add_item_race, so the visibility
       * transition (NULL -> cache) already shows wc > 0.
       *
       * Buffer install: CREATOR_REMOTE init does NOT install a buffer
       * (alloc_cache_s contract).  We install one explicitly via
       * arts_db_buf_install so the user's `*addr = ...` write
       * lands in cache->buffer->data, and the buffer is captured by the
       * subsequent PUBLISH_AND_TRANSFER. */
      if (no_acquire) {
        /* NO_ACQUIRE: do NOT stub-install a creator-side cache_s.  The
         * home is the sole idle owner; first consumer EDT triggers a
         * normal OWNERSHIP_REQUEST to acquire ownership.  Wire only carries
         * metadata (no payload bytes). */
        arts_send_db_create_coherent(rank, guid, len, ARTS_DB_PROP_NO_ACQUIRE,
                                     (uint16_t)db_type);
        *addr = NULL;
      } else {
        /* Creator-remote (home != self): cache-only stub — no home directory.
         * arts_db_cache_stub_size() spans cache + db_type but not the home
         * fields, which this rank never touches (home lives on the GUID home).
         * Init the cache in place, then install the buffer. */
        uint64_t stub_sz = arts_db_cache_stub_size();
        struct arts_db_s *creator_stub = (struct arts_db_s *)arts_malloc_aligned(
            stub_sz, ARTS_CACHE_LINE_SIZE);
        memset(creator_stub, 0, stub_sz);
        creator_stub->db_type = ARTS_DB;
        struct arts_db_cache_s *creator_cache = &creator_stub->cache;
        /* On a lost install race we adopt the existing db_s via a pinned handle
         * (released at the end of this block).  On install-success the handle
         * stays NULL and creator_cache points at our own (route-table-owned)
         * stub. */
        arts_shared_ptr_t adopted_h = NULL;
        arts_db_cache_init(creator_cache, guid, len,
                           ARTS_DB_INIT_CREATOR_REMOTE, /*creator_rank=*/0);
        if (len > 0) {
          arts_db_buf_install(creator_cache, /*new_version=*/1,
                              /*data_payload=*/NULL, len);
        }
        if (!arts_route_table_install_if_absent(creator_stub, guid,
                                                arts_global_rank_id,
                                                /*used=*/true)) {
          /* Lost the race -- another thread already installed (e.g. an
           * earlier wire arrival).  Free our stub and adopt the existing
           * cache; bump its writer_count by 2 to account for our sentinel
           * + creator EDT (consistent with CREATOR_REMOTE semantics). */
          arts_db_free(creator_stub);
          adopted_h = arts_route_table_lookup_db(guid);
          struct arts_db_s *adopted_db =
              (struct arts_db_s *)arts_shared_get(adopted_h);
          creator_cache = (adopted_db != NULL && adopted_db->db_type == ARTS_DB)
                              ? &adopted_db->cache
                              : NULL;
          if (creator_cache != NULL) {
#if !defined(ARTS_PROTOCOL_EXCL) && !defined(ARTS_PROTOCOL_INV)
            arts_atomic_add(&creator_cache->writer_count, 2);
#endif
          }
        }
        arts_send_db_create_coherent(rank, guid, len, ARTS_DB_PROP_NONE,
                                     (uint16_t)db_type);
        /* Return the creator-side buffer pointer so the user can write to
         * the local copy.  The data is published to home via PUBLISH
         * when the creator EDT releases (or via per-release publish).
         * Single-owner context: the creator owns the freshly-installed
         * buffer, so acquire a ref, read the payload pointer, release. */
        arts_shared_ptr_t creator_buf_h =
            creator_cache ? arts_db_buf_acquire(creator_cache) : NULL;
        struct arts_db_buffer_s *creator_buf =
            (struct arts_db_buffer_s *)arts_shared_get(creator_buf_h);
        *addr = creator_buf ? (void *)creator_buf->data : NULL;
        arts_db_buf_release(&creator_buf_h);
        if (current_edt && creator_cache &&
            !arts_db_creator_skip_hold(ARTS_DB)) {
          /* Auto-acquire: register the DB on the creator EDT's
           * created_db_list so arts_release_created_dbs at EDT epilogue
           * dispatches coherent release_rw, dropping the creator's
           * writer_count and triggering the eventual WB+TRANSFER. */
          arts_db_auto_acquire(creator_stub);
        }
        /* Drop the adopted-DB pin (NULL on the install-success path). */
        arts_shared_release(&adopted_h);
      }
      ARTS_DEBUG("arts_db_create: DB[Guid:%lu, Type:%s, Size:%lu] "
                 "created remotely on rank %u via DB_CREATE_COHERENT",
                 guid, GET_DB_TYPE_NAME(db_type), len, rank);
    } else {
      /* Non-coherent subtypes (ARTS_DB_PIN, ARTS_DB_GPU_PIN, ARTS_DB_GPU,
       * ARTS_DB_CXL) are pinned to the creator rank.  Creating one on a
       * different rank is a programming error — return NULL_GUID.
       * Cross-rank distribution for these subtypes must use ARTS_DB instead. */
      ARTS_WARN("arts_db_create: only ARTS_DB (coherent) DataBlocks support "
                "remote create; %s on rank %u is pinned to the creator rank. "
                "Returning NULL_GUID.",
                GET_DB_TYPE_NAME(db_type), rank);
      *addr = NULL;
      guid = NULL_GUID;
    }
  }
  TIME_DB_CREATE_STOP();
  return guid;
}

/*
 * arts_db_destroy — Mark a DataBlock for deferred destruction.
 *
 * If the calling EDT currently holds an acquire on this DB (either via
 * the auto-acquired created_db_list or via a dependency slot), the
 * acquire is implicitly released first.  This matches OCR's ocrDbDestroy
 * semantics: "If the EDT has acquired this DB, this call implicitly
 * releases the DB."
 *
 * After the implicit release, the route-table entry is marked for
 * deletion.  New acquire attempts (inc_item) will fail once DELETE_ITEM
 * is set.  The actual memory is freed when the last outstanding
 * route-table reference is returned (deferred deletion).
 */
void arts_db_destroy(arts_guid_t guid) {
  INCREMENT_NUM_DB_DESTROY_BY(1);
  arts_guid_kind_t type = arts_guid_get_kind(guid);
  if (type != ARTS_GUID_DB) {
    ARTS_WARN("arts_db_destroy called with non-DB type %u (GUID %lu)", type,
              guid);
    return;
  }

#ifdef ARTS_USE_CXL
  if (arts_guid_is_cxl(guid)) {
    return;
  }
#endif

  /* Implicit release: if the calling EDT holds an acquire on this DB,
   * release it first (matches OCR ocrDbDestroy semantics).  A created/owned
   * DB releases as RW; a dep release reads the slot mode in Path 2 regardless.
   */
  arts_db_release(guid, DB_MODE_RW);

  arts_shared_ptr_t db_res_h = arts_route_table_lookup_db(guid);
  struct arts_db_s *db_res = (struct arts_db_s *)arts_shared_get(db_res_h);

  /* Coherent ARTS_DB path: hand off to the coherence-layer destroy entry,
   * which sends DESTROY_REQ to home and runs the fan-out / finalize there. */
  if (db_res != NULL && db_res->db_type == ARTS_DB) {
    arts_shared_release(&db_res_h);
    arts_db_destroy_remote(guid);
    return;
  }

  /* Non-coherent pinned subtypes (ARTS_DB_PIN, ARTS_DB_GPU_PIN, ARTS_DB_GPU,
   * ARTS_DB_CXL): the DB lives only on the creator rank.  Route through
   * arts_route_table_set_destroyed — once outstanding refs drop, the cb
   * deleter (arts_db_deleter) runs. */
  if (db_res != NULL) {
    arts_shared_release(&db_res_h);
    arts_route_table_set_destroyed(guid);
  }
}

arts_guid_t arts_db_copy_to_new_type(arts_guid_t old_guid,
                                     arts_db_types_t new_type) {
  arts_guid_t ret = NULL_GUID;
  unsigned int rank = arts_guid_get_rank(old_guid);
  if (rank == arts_global_rank_id) {
    arts_guid_t new_guid = arts_guid_create_for_rank(rank, ARTS_GUID_DB);
    arts_shared_ptr_t h = arts_route_table_lookup_db(old_guid);
    struct arts_db_s *db_res = (struct arts_db_s *)arts_shared_get(h);
    if (db_res != NULL) {
      /* Storage-model migration.  A coherent ARTS_DB keeps its canonical
       * payload in the coherence buffer (cache->buffer->data); every
       * non-coherent subtype keeps it inline at (db+1).  When the model
       * changes, copy the payload to the destination model's location so the
       * retyped DB actually carries the data rather than just flipping the type
       * flag.  This MUST run while db_type is still the OLD type, so
       * arts_db_user_ptr resolves the OLD canonical location; the inline region
       * is always present (the DB was allocated as sizeof(db_s)+db_size). */
      if (db_res->db_type == ARTS_DB && new_type != ARTS_DB) {
        void *old_data = arts_db_user_ptr(db_res); /* buffer->data */
        void *inline_data = (char *)db_res + sizeof(struct arts_db_s);
        if (old_data != NULL && old_data != inline_data) {
          memcpy(inline_data, old_data, db_res->cache.db_size);
        }
        /* The coherent-model resources (versioned buffer + home directory:
         * rank_bitset, snapshot reorder stack, grantreq queues) are now dead —
         * the non-coherent target reads the inline payload and never runs the
         * coherent teardown.  Release them here with the canonical teardown so
         * they are not leaked.  common_destroy_post clears home_initialized, so
         * the eventual real destructor is a no-op for these (no double-free).
         */
        arts_db_cache_common_destroy_pre(&db_res->cache);  /* buffer-NULL */
        arts_db_cache_common_destroy_post(&db_res->cache); /* snapshot+home */
      }
      db_res->cache.db_guid = new_guid;
      db_res->db_type = new_type;
      /* Move the single cb to new_guid (see arts_db_rename_with_guid): one
       * owner across the GUID change, no double-free. */
      if (arts_route_table_move_item(old_guid, new_guid)) {
        ret = new_guid;
      }
      arts_shared_release(&h);
    }
  }
  return ret;
}

/**********************DB MEMORY MODEL*************************************/

/* acquire_one_dep — attempt the single DB dependency depv[i].
 *
 * On a synchronous resolve it writes depv[i].ptr (NULL is valid: a sentinel /
 * version-0 / no-payload DB) and self-accounts via arts_db_acquire_resolved;
 * when the EDT must park (remote ownership/data round, or OoO defer of a
 * not-yet-installed local DB) it leaves the slot for the protocol wake / OoO
 * drain replay and does NOT account.  The 3-way route_table dispatch (local
 * entry / remote-home stub install / home==self-but-not-created → OoO push)
 * lives here, inlined from the old single-DB arts_db_acquire API.  The caller
 * (arts_db_acquire_all / rw_fire_from_cursor) has already filtered NULL_GUID /
 * DB_MODE_VAL / pre-filled slots, so depv[i] is a real, not-yet-acquired DB
 * dependency. */
static void acquire_one_dep(struct arts_edt_s *edt, arts_edt_dep_t *depv,
                            uint32_t i) {
  arts_db_access_mode_t access_mode = depv[i].mode;
  int owner = (int)arts_guid_get_rank(depv[i].guid);
  arts_guid_kind_t guid_type = arts_guid_get_kind(depv[i].guid);

  if (guid_type != ARTS_GUID_DB) {
    return; /* not a DB GUID — nothing to acquire */
  }

  // Update access-mode counters
  if (access_mode == DB_MODE_RO) {
    INCREMENT_NUM_DB_ACQUIRE_READ_BY(1);
  } else if (access_mode == DB_MODE_RW) {
    INCREMENT_NUM_DB_ACQUIRE_WRITE_BY(1);
  }

  ARTS_INFO("Acquiring DB[Guid:%lu, GuidType:%u, AccessMode:%u, Owner:%d, "
            "Rank:%u] in EDT[Id:%lu, Guid:%lu, Slot:%u]",
            depv[i].guid, guid_type, access_mode, owner, arts_global_rank_id,
            edt->arts_id, edt->guid, i);

#ifdef ARTS_USE_CXL
  if (arts_guid_is_cxl(depv[i].guid)) {
    struct arts_db_s *cxl_db =
        (struct arts_db_s *)arts_cxl_get_ptr(depv[i].guid);
    /* Consumer flush deferred to prep_dbs (just before user func) to avoid
     * stale reads after deque wait. */
    if (cxl_db) {
      depv[i].ptr = cxl_db + 1;
      depv[i].subtype = ARTS_DB_CXL;
      arts_db_acquire_resolved(edt, i);
      return;
    }
    /* Not yet allocated in the shared segment — OoO defer (park); the CXL deque
     * ordering makes the producer's allocation visible before the consumer
     * runs.  Data/cursor arrive on the drain replay; no resolved here. */
    {
      struct arts_ooo_args_db_acquire_s a = {
          .edt = edt, .db_guid = depv[i].guid, .slot = i};
      arts_ooo_dispatch_or_defer_guid(depv[i].guid, OOO_DB_ACQUIRE, &a,
                                      sizeof(a));
    }
    return;
  }
#endif

  // Look up DB first — subtype dispatch requires the struct.  lookup_db pairs
  // with the release below (every successful lookup => one release).
  arts_shared_ptr_t db_temp_h = arts_route_table_lookup_db(depv[i].guid);
  struct arts_db_s *db_temp = (struct arts_db_s *)arts_shared_get(db_temp_h);

  /* Coherent ARTS_DB path (either placement, any protocol).  Two entry
   * points:
   *   - Existing local cache_s (db_temp with db_type == ARTS_DB; embedded
   *     cache).
   *   - Remote DB never seen on this rank (db_temp == NULL, owner remote):
   *     stub-install a stub cache_s and dispatch through
   *     arts_handler_db_acquire.
   * Other (pinned) subtypes bypass coherence and fall through below. */
  struct arts_db_cache_s *cache = NULL;
  /* When db_temp misses on a remote-owned DB we stub-install a stub and the
   * call returns a SEPARATE pinned handle (stub_h) to the just-installed db_s;
   * it must be released on every path below, mirroring db_temp_h. */
  arts_shared_ptr_t stub_h = NULL;
  /* B1: which of the two handles keeps `cache`'s descriptor (arts_db_s) alive.
   * If the handler resolves locally (takes the EDT's buffer ref), this handle
   * is MOVED into depv[i].db_pin to pin the descriptor — and the buffer slot +
   * recycle pool embedded in it — for the slot's whole acquire->release span.
   */
  arts_shared_ptr_t *cache_owner_h = NULL;
  if (db_temp != NULL && db_temp->db_type == ARTS_DB) {
    cache = &db_temp->cache;
    cache_owner_h = &db_temp_h;
  } else if (db_temp == NULL && owner != arts_global_rank_id) {
    /* db_size=0 means "size learned on first GRANT/DATA_RESPONSE
     * install_buffer".  Round-robin home is encoded in the GUID, so all
     * ranks agree. */
    stub_h = arts_db_cache_stub_install(depv[i].guid, /*db_size=*/0);
    struct arts_db_s *stub_db = (struct arts_db_s *)arts_shared_get(stub_h);
    if (stub_db != NULL) {
      cache = &stub_db->cache;
      cache_owner_h = &stub_h;
    }
  }
  if (cache != NULL &&
      (access_mode == DB_MODE_RO || access_mode == DB_MODE_RW)) {
    /* The handler self-resolves (writes depv[i].ptr + arts_db_acquire_resolved)
     * on a local hit, or parks on a remote ownership/data round; no return.
     * Record the coherent subtype now (before any park) so release routes by
     * dep->subtype regardless of whether the handler resolves locally or the
     * async data-response fills depv[i].ptr later. */
    depv[i].subtype = ARTS_DB;
    struct arts_ooo_args_db_acquire_s a = {
        .edt = edt, .db_guid = depv[i].guid, .slot = i};
    /* Publish the acquiring task for the span of the decision: the arm that
     * decides whether this rank can answer counts against it there, at the
     * same points it counts the cluster-wide totals.  Restored rather than
     * cleared, because a serve can nest inside another one on this thread. */
    uint64_t prev_task = arts_object_task_enter(edt->arts_id);
    arts_handler_db_acquire(arts_db_of_cache(cache), &a);
    arts_object_task_leave(prev_task);
    /* A handler that answered from here left ptr set, so the payload size is
     * known now.  One that parked is charged its bytes at the resume site,
     * where a datablock this rank had never seen finally has a size. */
    if (depv[i].ptr != NULL) {
      arts_object_record_db_bytes(edt->arts_id, cache->db_size);
      arts_object_trace_db(edt->arts_id, cache->db_size, 0);
    }
    /* B1: a local hit set depv[i].ptr and took the EDT's buffer ref.  Pin the
     * descriptor by MOVING the still-alive cache-owning handle into db_pin
     * (released last in release_one_dep), so a concurrent destroy cannot free
     * the cache out from under the outstanding buffer ref / its
     * recycle-on-drop. A parked handler leaves ptr NULL — the resume site
     * (mark_edt_ready_by_guid) pins instead.  The db_pin==NULL guard keeps the
     * pin balanced across OoO replays. */
    if (depv[i].ptr != NULL &&
        __atomic_load_n(&depv[i].db_pin, __ATOMIC_ACQUIRE) == NULL &&
        cache_owner_h != NULL) {
      __atomic_store_n(&depv[i].db_pin, (void *)*cache_owner_h,
                       __ATOMIC_RELEASE);
      *cache_owner_h = NULL;
    }
    arts_shared_release(&db_temp_h);
    arts_shared_release(&stub_h);
    return;
  }
  /* cache==NULL fall-throughs below never used stub_h (it is only set on the
   * remote-miss arm, which always has a non-NULL cache here unless the DB was
   * destroyed before install — stub_h NULL then); release defensively. */
  arts_shared_release(&stub_h);

  /* Non-coherent pinned subtypes (ARTS_DB_PIN, ARTS_DB_GPU_PIN, ARTS_DB_GPU,
   * ARTS_DB_CXL): the DB lives only on its creator rank — hand back the local
   * pointer if present. */
  if (db_temp != NULL) {
    if (owner != arts_global_rank_id) {
      ARTS_WARN(
          "arts_db_acquire_all: pinned DB[Guid:%lu, Type:%s] referenced from "
          "non-creator rank %u (owner=%u). Only ARTS_DB is internode "
          "relocatable.",
          depv[i].guid, GET_DB_TYPE_NAME(db_temp->db_type), arts_global_rank_id,
          owner);
    }
    depv[i].ptr = db_temp + 1;
    depv[i].subtype = db_temp->db_type;
    arts_shared_release(&db_temp_h);
    arts_db_acquire_resolved(edt, i);
    return;
  }

  /* DB absent locally.  OoO defer keyed on the DB GUID (park): when DB_CREATE
   * installs it (home==self case) the drain re-attempts this dep through
   * arts_handler_db_acquire.  A remote non-coherent (pinned-subtype) reference
   * can never resolve locally and waits here.  No resolved — the data and
   * cursor advance arrive on the drain replay. */
  if (arts_guid_is_local(depv[i].guid)) {
    ARTS_DEBUG("DB[Guid:%lu] out of order request slot %u", depv[i].guid, i);
  } else {
    ARTS_WARN("arts_db_acquire_all: cannot resolve remote DB[Guid:%lu] for "
              "non-coherent (pinned) dep on rank %u — owner=%u. Deferring via "
              "OoO.",
              depv[i].guid, arts_global_rank_id, owner);
  }
  {
    struct arts_ooo_args_db_acquire_s a = {
        .edt = edt, .db_guid = depv[i].guid, .slot = i};
    arts_ooo_dispatch_or_defer_guid(depv[i].guid, OOO_DB_ACQUIRE, &a,
                                    sizeof(a));
  }
}

static bool dep_is_serialized(arts_edt_dep_t *depv, uint32_t i);
static void rw_fire_from_cursor(struct arts_edt_s *edt);
static void resume_enqueue(arts_guid_t edt_guid);
static void flush_resume_list(void);

/* OOO_DB_ACQUIRE replay table entry — see db.h.  Re-attempts the single
 * deferred dep through acquire_one_dep's subtype-aware 3-way: for an ARTS_DB
 * this lands in arts_handler_db_acquire (coherent), for a PIN/GPU/CXL DB in the
 * pinned ptr path, and a still-absent DB re-defers.  Mapping OOO_DB_ACQUIRE
 * straight to arts_handler_db_acquire would mishandle non-coherent subtypes
 * (their embedded cache is zeroed — its pending_rw queue is uninitialised, so
 * the coherent RW path would push onto a NULL-headed queue and crash). */
void arts_db_acquire_replay_dep(void *item, void *args) {
  (void)item; /* the 3-way re-looks-up the installed db_s; the drain pins it */
  struct arts_ooo_args_db_acquire_s *a =
      (struct arts_ooo_args_db_acquire_s *)args;
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(a->edt);
  if (dep_is_serialized(depv, a->slot)) {
    /* Serialized dep deferred at the cursor: re-drive the acquire loop from the
     * cursor.  Route through the flat resume trampoline (enqueue + flush) so
     * the re-drive stays top-level even when this replay runs nested inside
     * another EDT's acquire loop (an inline OoO drain) — never an inline
     * rw_fire_from_cursor that would recurse. */
    resume_enqueue(a->edt->guid);
    flush_resume_list();
  } else {
    /* Non-serialized (Pass-1) dep: re-attempt just this slot; it self-accounts
     * on a local hit and the EDT schedules when the last dep's data lands. */
    acquire_one_dep(a->edt, depv, a->slot);
  }
}

/* GUID-sorted index array — identical on every (re)entry. */
static void sort_dep_indices(arts_edt_dep_t *depv, uint32_t depc,
                             uint32_t *sorted) {
  for (uint32_t k = 0; k < depc; k++) {
    sorted[k] = k;
  }
  for (uint32_t k = 1; k < depc; k++) {
    uint32_t val = sorted[k];
    int j = (int)k - 1;
    while (j >= 0 && depv[sorted[j]].guid > depv[val].guid) {
      sorted[j + 1] = sorted[j];
      j--;
    }
    sorted[j + 1] = val;
  }
}

/* A real DB dep still needing acquisition (not NULL / not a raw value / not
 * pre-filled / actually a DB GUID). */
static bool dep_needs_acquire(arts_edt_dep_t *depv, uint32_t i) {
  return depv[i].guid != NULL_GUID && depv[i].mode != DB_MODE_VAL &&
         depv[i].ptr == NULL &&
         arts_guid_get_kind(depv[i].guid) == ARTS_GUID_DB;
}

/* Ownership-serialized (RW cursor) dep? CXL DBs bypass DB coherence (no
 * ownership round), so they are never serialized — they fire in Pass 1. */
static bool dep_is_serialized(arts_edt_dep_t *depv, uint32_t i) {
#ifdef ARTS_USE_CXL
  if (arts_guid_is_cxl(depv[i].guid)) {
    return false;
  }
#endif
  return arts_db_acquire_is_serialized(depv[i].mode);
}

/* Decrement-and-maybe-schedule. Caller must not touch the EDT afterward. */
void arts_db_acquire_account(struct arts_edt_s *edt) {
  if (arts_atomic_sub(&edt->acquire_remaining, 1) == 0) {
    arts_schedule_ready_edt(edt);
  }
}

/* The EDT whose serialized-acquire loop is in progress in THIS execution
 * context right now (the resume_k loop below records it for its own duration).
 * When a dep is resolved INTRA-RANK and synchronously — reached from inside
 * that running loop (a same-rank grant whose handler runs directly on this
 * rank, no wire) — the running loop already advances to the next dep, so it
 * must NOT re-enter (re-firing would recurse once per dep).  A resume that is
 * NOT for the acquire in progress here (a parked EDT woken by an inter-rank
 * grant, a release grant, or an OoO replay) sees a different (or NULL) value
 * and DOES re-enter to drive that EDT.  Execution-context state, not per-EDT
 * state. */
static ARTS_THREAD_LOCAL struct arts_edt_s *tl_acquiring = NULL;

/* ===== Flat resume trampoline ==========================================
 * A grant/drain that secures a dep for a PARKED EDT (one that is not the EDT
 * whose acquire loop is currently running, i.e. edt != tl_acquiring) must
 * resume that EDT's serialized acquire — but it MUST NOT call
 * rw_fire_from_cursor inline: that EDT's loop would then nest on top of the
 * currently-running loop (continuation recursion, O(depth) stack and O(depc²)
 * repeated work).  Instead the woken EDT's GUID is appended to a thread-local
 * worklist and the resume runs FLAT: when control returns to the top level
 * (no acquire loop in progress, tl_acquiring == NULL) flush_resume_list drains
 * the worklist in a while loop, calling rw_fire_from_cursor once per entry.
 * Re-entrancy-guarded (tl_in_flush) + top-level-guarded (tl_acquiring) so
 * rw_fire_from_cursor can never appear twice on the stack. */
static ARTS_THREAD_LOCAL arts_guid_t *tl_resume_buf = NULL;
static ARTS_THREAD_LOCAL uint32_t tl_resume_len = 0;
static ARTS_THREAD_LOCAL uint32_t tl_resume_cap = 0;
static ARTS_THREAD_LOCAL bool tl_in_flush = false;

/* Continuation-ownership signal for the resume_k loop: set when the dep the
 * loop just fired was resolved IN-FRAME (a synchronous local resolve or a
 * same-thread nested serve — rw_secure / arts_db_acquire_resolved with
 * edt == tl_acquiring).  The loop continues ONLY on this signal; otherwise it
 * breaks and the serving side owns the continuation (its rw_secure enqueued
 * the flat resume on its own thread).  A cursor comparison cannot make this
 * call: a CONCURRENT other-thread serve also advances the cursor, and reading
 * "advanced" as "resolved in-frame" lets two threads drive the same EDT's
 * loop at once — double-firing the next dep (double count, one release, and a
 * double acquire_remaining account).  Being thread-local, this flag is
 * untouchable by other threads' serves, so ownership is race-free. */
static ARTS_THREAD_LOCAL bool tl_inline_advanced = false;

static void resume_enqueue(arts_guid_t edt_guid) {
  if (tl_resume_len == tl_resume_cap) {
    uint32_t ncap = tl_resume_cap ? tl_resume_cap * 2u : 16u;
    tl_resume_buf =
        (arts_guid_t *)arts_realloc(tl_resume_buf, ncap * sizeof(arts_guid_t));
    tl_resume_cap = ncap;
  }
  tl_resume_buf[tl_resume_len++] = edt_guid;
}

/* Drain the resume worklist as a flat loop.  No-op when called from inside an
 * acquire loop (tl_acquiring != NULL) or an in-progress flush — the outermost
 * caller owns the drain, so a wake enqueued deep in the nest is picked up by
 * that single top-level while loop, never by a nested rw_fire_from_cursor. */
static void rw_fire_from_cursor(struct arts_edt_s *edt);
static void flush_resume_list(void) {
  if (tl_acquiring != NULL || tl_in_flush) {
    return;
  }
  tl_in_flush = true;
  while (tl_resume_len > 0) {
    arts_guid_t g = tl_resume_buf[--tl_resume_len];
    arts_shared_ptr_t h = arts_route_table_lookup_edt(g);
    struct arts_edt_s *e = (struct arts_edt_s *)arts_shared_get(h);
    if (e != NULL) {
      rw_fire_from_cursor(e); /* flat: tl_acquiring is NULL here */
    }
    arts_shared_release(&h);
  }
  tl_in_flush = false;
}

/* resume_k loop: walk the serialized deps from the cursor, firing each.  A dep
 * that resolves LOCALLY advances the cursor (arts_db_acquire_resolved / the
 * alias path — advance only, no re-fire) and the loop picks up the next one; a
 * dep that PARKS (its grant is not local) leaves the cursor put and we return —
 * the matching async grant re-enters here (rw_secure) to continue.  Driving the
 * walk as a LOOP (not the handler re-firing recursively) keeps the stack O(1)
 * however many serialized deps resolve in a row — required for RWLOCK, where RW
 * AND RO are both serialized so an EDT can have very many serialized deps. */
static void rw_fire_from_cursor(struct arts_edt_s *edt) {
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
  uint32_t depc = edt->depc;
  const uint32_t *sorted = edt->rw_sorted; /* sorted ONCE in acquire_all */
  struct arts_edt_s *prev_acquiring = tl_acquiring;
  tl_acquiring = edt;
  while (edt->rw_cursor < depc) {
    uint32_t i = sorted[edt->rw_cursor];
    if (!dep_needs_acquire(depv, i) || !dep_is_serialized(depv, i)) {
      edt->rw_cursor++;
      continue;
    }
    /* Re-entrant same-DB exclusive acquire. An EDT acquires each distinct DB
     * exactly once; a later slot naming a DB an earlier serialized slot of the
     * same EDT already secured is an alias, not a fresh writer. The GUID sort
     * makes same-GUID deps adjacent, and the cursor only reaches a slot after
     * every earlier serialized dep secured — so a serialized predecessor with
     * this GUID means the rank already owns the DB and its buffer is installed.
     * Issuing a second ownership round here would, under a single-writer
     * protocol, enqueue the request behind the EDT's own unreleased hold and
     * self-deadlock. Resolve it as a local hit (a fresh per-slot buffer ref,
     * balanced by release_one_dep) — the same payload the predecessor sees. */
    bool reentrant = false;
    for (int p = (int)edt->rw_cursor - 1; p >= 0; p--) {
      uint32_t pj = sorted[p];
      if (depv[pj].guid != depv[i].guid) {
        break; /* GUID-sorted: no earlier dep shares this GUID */
      }
      if (dep_is_serialized(depv, pj)) {
        reentrant = true;
        break;
      }
    }
    if (reentrant) {
      /* Pin the db_s for the cache-deref window (cache is its FIRST member,
       * offset 0).  arts_db_acquire_local takes the buffer's own ref (the EDT's
       * hold); B1 keeps this descriptor pin alive for the slot's whole span. */
      arts_shared_ptr_t db_h = arts_route_table_lookup_db(depv[i].guid);
      struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
      if (db != NULL && db->db_type == ARTS_DB) {
        depv[i].subtype = ARTS_DB;
        /* Alias slot: buffer ref only, no writer_count hold (the earlier
         * serialized slot owns the single acquire).  Recorded so release skips
         * the coherence release for this slot. */
        depv[i].alias = true;
        depv[i].ptr = arts_db_acquire_local(&db->cache);
        arts_db_acquire_resolved(edt, i); /* advances cursor; loop continues */
        /* B1: the alias took a per-slot buffer ref — pin the descriptor for its
         * acquire->release span by MOVING db_h into db_pin (released last in
         * release_one_dep).  ptr==NULL means no buffer was installed (no ref
         * taken); release db_h normally then. */
        if (depv[i].ptr != NULL &&
            __atomic_load_n(&depv[i].db_pin, __ATOMIC_ACQUIRE) == NULL) {
          __atomic_store_n(&depv[i].db_pin, (void *)db_h, __ATOMIC_RELEASE);
          db_h = NULL;
        }
        arts_shared_release(&db_h);
        continue;
      }
      arts_shared_release(&db_h);
      /* Cache unexpectedly absent — fall back to a normal acquire. */
    }
    tl_inline_advanced = false;
    acquire_one_dep(
        edt, depv,
        i); /* local hit advances the cursor; remote/contended parks */
    if (!tl_inline_advanced) {
      /* Parked (or deferred): the serving side owns the continuation — its
       * rw_secure enqueues the flat resume on its own thread.  Do NOT infer
       * "resolved" from cursor movement: a concurrent other-thread serve also
       * advances the cursor, and continuing here would put two threads in the
       * same EDT's loop (double-firing the next dep). */
      break;
    }
    /* resolved in-frame — the loop picks up the next serialized dep */
  }
  tl_acquiring = prev_acquiring;
}

/* Async grant resume: advance past the just-secured `slot` and continue firing
 * from the cursor.  Position-idempotent: only advances when the cursor still
 * points at `slot`, so a redundant secure for an already-passed slot is a no-op
 * (it neither re-fires the in-flight dep nor double-accounts).  This is the
 * path a grant for a PARKED EDT takes (remote grant / release grant) — it
 * re-enters the resume_k loop.  The SYNCHRONOUS local resolve does NOT come
 * through here; it uses arts_db_acquire_resolved (advance only) and the running
 * loop picks up the next dep, so consecutive local resolves never recurse. */
static void rw_secure(struct arts_edt_s *edt, unsigned int slot) {
  uint32_t depc = edt->depc;
  const uint32_t *sorted = edt->rw_sorted; /* sorted ONCE in acquire_all */
  if (edt->rw_cursor < depc && sorted[edt->rw_cursor] == slot) {
    edt->rw_cursor++;
    /* If this resume is for THIS thread's in-flight EDT (a same-rank grant
     * whose handler ran synchronously inside the running loop), signal the
     * loop to continue — it owns the continuation.  Otherwise it is a PARKED
     * EDT woken by a grant: enqueue it for the flat resume drain (NOT an
     * inline rw_fire_from_cursor, which would nest on the running loop =
     * recursion).  Exactly one side continues the loop, never both. */
    if (edt != tl_acquiring) {
      resume_enqueue(edt->guid);
    } else {
      tl_inline_advanced = true;
    }
  }
}

/* A SYNCHRONOUS local resolve (handler set dep->ptr, in the running acquire
 * loop).  For a serialized dep, advance the cursor ONLY — the running
 * rw_fire_from_cursor loop picks up the next dep, so this never re-fires
 * recursively (that is the whole point of the loop: acquire_all is the
 * continuation unit, the loop walks deps, and only an ASYNC resume — rw_secure
 * / the OoO replay — re-enters the loop).  Then account THIS dep; the +1 bias
 * on acquire_remaining keeps the count above zero until the loop completes, so
 * accounting order is free.  The upstream caller holds an EDT ref across the
 * loop, so a schedule from the final account cannot free the EDT mid-loop. */
void arts_db_acquire_resolved(struct arts_edt_s *edt, unsigned int slot) {
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
  if (dep_is_serialized(depv, slot)) {
    uint32_t depc = edt->depc;
    const uint32_t *sorted = edt->rw_sorted; /* sorted ONCE in acquire_all */
    if (edt->rw_cursor < depc && sorted[edt->rw_cursor] == slot) {
      edt->rw_cursor++; /* advance only; the loop fires the next dep */
      if (edt == tl_acquiring) {
        tl_inline_advanced = true; /* in-frame resolve: the loop continues */
      }
    }
  }
  arts_db_acquire_account(
      edt); /* count THIS dep's data; outermost may schedule */
}

/* Driver: fire all non-serialized deps (Pass 1), then fire the serialized ones
 * from the cursor (Pass 2; the handler's resolved path self-continues the RW
 * chain). acquire_remaining is +1-biased so data arrivals during the fire
 * cannot schedule before the fire completes. Called once per EDT from
 * arts_handle_ready_edt. */
void arts_db_acquire_all(struct arts_edt_s *edt) {
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
  uint32_t depc = edt->depc;
  /* Sort the GUID order ONCE for the whole acquire phase.  The order is a pure
   * function of depv (fixed once the EDT is ready), so rw_fire_from_cursor /
   * rw_secure / arts_db_acquire_resolved all consume edt->rw_sorted and none
   * re-sorts — turning the per-dep O(depc²) re-sort into a single O(depc²).
   * Freed at run (arts_run_edt). */
  if (depc > 0 && edt->rw_sorted == NULL) {
    edt->rw_sorted = (uint32_t *)arts_malloc(depc * sizeof(uint32_t));
    sort_dep_indices(depv, depc, edt->rw_sorted);
  }
  const uint32_t *sorted = edt->rw_sorted;

  uint32_t n = 0;
  for (uint32_t k = 0; k < depc; k++) {
    if (dep_needs_acquire(depv, sorted[k])) {
      n++;
    }
  }
  arts_atomic_add(&edt->acquire_remaining, n); /* now (1 + n) with the bias */

  /* Pass 1: every non-serialized real DB dep, order-independent (handler
   * self-accounts via arts_db_acquire_resolved on a local hit; remote parks).
   */
  for (uint32_t k = 0; k < depc; k++) {
    uint32_t i = sorted[k];
    if (!dep_needs_acquire(depv, i) || dep_is_serialized(depv, i)) {
      continue;
    }
    acquire_one_dep(edt, depv, i);
  }

  /* Pass 2: fire the serialized (RW) deps from the cursor (no-op under WRF_RCU).
   */
  rw_fire_from_cursor(edt);

  /* Remove the +1 bias; this decrement may be the one that reaches 0. */
  arts_db_acquire_account(edt);

  /* Drain any cross-EDT resumes enqueued while this EDT's loop ran (flat). */
  flush_resume_list();
}

/* Secured wake (PROCEED / GRANT drain): position-idempotent cursor advance +
 * fire next serialized dep. Holds an EDT ref across rw_secure (which may
 * schedule deep in the recursion). */
void mark_edt_secured_by_guid(arts_guid_t edt_guid, unsigned int slot) {
  if (edt_guid == NULL_GUID) {
    return;
  }
  arts_shared_ptr_t edt_h = arts_route_table_lookup_edt(edt_guid);
  struct arts_edt_s *edt = (struct arts_edt_s *)arts_shared_get(edt_h);
  if (edt == NULL) {
    arts_shared_release(&edt_h);
    return;
  }
  rw_secure(edt, slot);
  arts_shared_release(&edt_h);
  /* Top-level grant-wake: drive the parked EDT's resume flat (no-op when nested
   * inside an acquire loop — the outer loop's flush owns the drain). */
  flush_resume_list();
}

/*
 * prep_dbs — Prepare DB dependencies just before EDT execution.
 *
 * For each WRITE-mode dependency, invalidates remote route table entries
 * (marks other caches stale).  In GPU builds, for every LC (locally-coherent)
 * DB (regardless of access mode), acquires a reader lock and increments the
 * DB version counter; for DB_MODE_LC_SYNC deps specifically, syncs GPU
 * shadow copies.
 *
 * Called from arts_run_edt() after all DB pointers have been resolved.
 */
void prep_dbs(unsigned int depc, arts_edt_dep_t *depv, bool gpu) {
  (void)gpu;
  for (unsigned int i = 0; i < depc; i++) {
    arts_db_access_mode_t access_mode = depv[i].mode;
    if (depv[i].guid == NULL_GUID || depv[i].ptr == NULL) {
      continue;
    }
    /* For coherent ARTS_DB, dep->ptr is buf->data and pointer arithmetic to
     * recover db_s would land in the buffer header, NOT a db_s.  Skip via the
     * subtype recorded at acquire — NOT a borrowed (non-refcounted) route
     * pointer, which would use-after-free if a concurrent destroy freed the
     * arts_db_s while we read db->db_type through it.  Coherent
     * ARTS_DB drives invalidation inside the coherence layer; non-coherent
     * pinned subtypes (ARTS_DB_PIN, ARTS_DB_GPU_PIN, ARTS_DB_GPU, ARTS_DB_CXL)
     * have no DB-level coherence and fall through to their per-subtype prep
     * (their dep->ptr is db+1, so the recovery below is valid for them). */
    if (depv[i].subtype == ARTS_DB) {
      continue;
    }
#ifdef ARTS_USE_CXL
    {
      struct arts_db_s *db_cxl = ((struct arts_db_s *)depv[i].ptr) - 1;
      if (db_cxl->db_type == ARTS_DB_CXL) {
        arts_cxl_consumer_flush(db_cxl->guid);
      }
    }
#endif
#ifdef ARTS_USE_GPU
    if (!gpu && access_mode != DB_MODE_LC_SYNC) {
      struct arts_db_s *db = ((struct arts_db_s *)depv[i].ptr) - 1;
      if (db->db_type == ARTS_DB_GPU) {
        arts_reader_lock(&db->reader, &db->writer);
        arts_atomic_add(&db->version, 1);
      }
    }

    if (!gpu && access_mode == DB_MODE_LC_SYNC) {
      struct arts_db_s *db = ((struct arts_db_s *)depv[i].ptr) - 1;
      ARTS_DEBUG("internalLCSync %lu %p", depv[i].guid, db);
      internal_lc_sync_gpu(depv[i].guid, db);
    }
#endif
  }
}

/*
 * release_one_dep — Single source of truth for "release one dep slot".
 *
 * Used by:
 *   - release_dbs (EDT epilogue, all dep slots)
 *   - arts_db_release Path 2 (mid-EDT release of one depv slot)
 *   - arts_db_release Path 1 + arts_release_created_dbs (via a synthetic
 *     dep built from a created_db_list entry)
 *
 * Per access mode:
 *   - DB_MODE_RO / DB_MODE_RW: only ARTS_DB needs DB-level coherence
 *     work; route through the coherent release entry points
 *     (arts_db_release_ro / arts_db_release_rw).  Non-coherent pinned
 *     subtypes have no DB-level coherence — release is a no-op.
 *   - DB_MODE_PTR: free the malloc'd copy buffer.
 *   - ARTS_DB_GPU subtype (GPU build, non-LC_SYNC mode): release the GPU-LC
 *     reader lock — pure intra-rank multi-device coordination.
 *   - ARTS_DB_CXL subtype: producer-flush and return.
 *
 * Does NOT nullify caller-visible state (guid/ptr/mode).  Callers that
 * need to mark the slot as released (mid-EDT release) do that themselves.
 */
static void release_one_dep(arts_edt_dep_t *dep, bool gpu) {
  arts_db_access_mode_t access_mode = dep->mode;

  /* Coherent release path for ARTS_DB.  Routed by dep->subtype (recorded at
   * acquire), NOT by recovering the subtype from dep->ptr: for a coherent DB
   * dep->ptr is cache->buffer->data (NOT (db+1)), so the pinned-subtype
   * pointer arithmetic below would read a wild address — fatal once a
   * concurrent destroy has removed the route entry (cache lookup then misses).
   * Drop the EDT's per-acquire buffer ref (taken at acquire time:
   * acquire_local / mark_edt_ready_by_guid each do arts_db_buf_acquire)
   * unconditionally via the buffer's own cb: the EDT's ref kept the buffer
   * (hence buf->cb) alive up to here, so the deref is never use-after-free even
   * under a racing destroy.  Dispatch release_rw / release_ro only while the
   * cache is still installed; once destroyed there is no publish / version
   * work left to do (the buffer ref drop above is the only cleanup needed). */
  if (dep->subtype == ARTS_DB &&
      (access_mode == DB_MODE_RO || access_mode == DB_MODE_RW)) {
    if (dep->ptr != NULL) {
      struct arts_db_buffer_s *buf = arts_db_buf_from_data(dep->ptr);
      if (buf != NULL) {
        arts_shared_ptr_t buf_cb = buf->cb;
        arts_db_buf_release(&buf_cb);
      }
    }
    /* B1: take the stashed descriptor pin (set when this slot's buffer ref was
     * secured at acquire).  Dropped LAST — after the buffer-ref drop above and
     * the RW/RO release below — so the cache (buffer slot + recycle pool)
     * outlives the buffer's recycle-on-drop and the coherence release even
     * against a concurrent destroy. */
    /* Consume the descriptor pin published (with release) at the acquire/wake
     * site; acquire-load matches that release across the work-stealing handoff
     * (mirrors the sibling ptr field's atomic discipline — TSan-clean). */
    arts_shared_ptr_t db_pin =
        (arts_shared_ptr_t)__atomic_load_n(&dep->db_pin, __ATOMIC_ACQUIRE);
    __atomic_store_n(&dep->db_pin, NULL, __ATOMIC_RELAXED);
    /* Re-entrant alias slot (a later dep naming a DB an earlier serialized slot
     * of the same EDT already acquired): it took a buffer ref at acquire (so
     * the drop above balances it) but never a writer_count hold — the first
     * slot owns the single coherence acquire/release. Skip release_rw/ro so the
     * owner's writer_count is decremented exactly once per distinct DB.  The
     * alias bit is recorded at acquire (rw_fire_from_cursor), NOT re-derived
     * here, so a mid-EDT release that nulls the owning slot's GUID cannot make
     * an alias masquerade as the owner.  Still drop the alias's own pin. */
    if (dep->alias) {
      arts_shared_release(&db_pin);
      return;
    }
    if (db_pin != NULL) {
      /* Resolve the descriptor via the B1 pin — guaranteed alive (it kept the
       * cache pinned across the whole span), so it is immune to the
       * destroyed-but-lingering-DB re-lookup miss the old path risked. */
      struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_pin);
      if (db != NULL && db->db_type == ARTS_DB) {
        if (access_mode == DB_MODE_RW) {
          arts_db_release_rw(&db->cache);
        } else {
          arts_db_release_ro(&db->cache);
        }
      }
      arts_shared_release(&db_pin);
    } else if (dep->guid != NULL_GUID) {
      /* No pin stashed (a buffer-ref path not covered by B1, or a destroyed
       * DB): fall back to the ref-counted route re-lookup — identical to pre-B1
       * behavior, correct for the held writer_count; it only loses the
       * destroy-race robustness the pin provides. */
      arts_shared_ptr_t db_h = arts_route_table_lookup_db(dep->guid);
      struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
      if (db != NULL) {
        if (db->db_type == ARTS_DB) {
          if (access_mode == DB_MODE_RW) {
            arts_db_release_rw(&db->cache);
          } else {
            arts_db_release_ro(&db->cache);
          }
        }
        arts_shared_release(&db_h);
      }
    }
    return;
  }

  /* Get DB subtype from struct when ptr is available.  Guard with
   * guid != NULL_GUID because arts_db_release may have already nulled
   * the guid while leaving ptr non-NULL (caller responsibility).
   *
   * Reaching this point means the dep is for a non-coherent pinned subtype
   * (ARTS_DB_PIN, ARTS_DB_GPU_PIN, ARTS_DB_GPU, ARTS_DB_CXL) or a special
   * access mode (PTR, VALUE, LC_*, MEMSET) — none of which carry DB-level
   * coherence. */
  arts_db_types_t db_subtype = ARTS_DB;
  if (dep->guid != NULL_GUID && dep->ptr) {
    struct arts_db_s *db_hdr = ((struct arts_db_s *)dep->ptr) - 1;
    db_subtype = db_hdr->db_type;
  }

  ARTS_DEBUG("Releasing DB[Guid:%lu] [AccessMode:%s, DbSubtype:%s]", dep->guid,
             GET_DB_MODE_NAME(access_mode), GET_DB_TYPE_NAME(db_subtype));

#ifdef ARTS_USE_CXL
  if (db_subtype == ARTS_DB_CXL) {
    if (dep->guid != NULL_GUID && dep->ptr &&
        (access_mode == DB_MODE_RW || access_mode == DB_MODE_MEMSET)) {
      arts_cxl_producer_flush(dep->guid);
    }
    return; /* CXL: no route table, HW MESI handles intra-node coherence */
  }
#endif

  if (access_mode == DB_MODE_PTR) {
    if (dep->ptr) {
      arts_free(dep->ptr);
    }
  } else if (!gpu && db_subtype == ARTS_DB_GPU) {
    if (dep->ptr) {
      struct arts_db_s *db = ((struct arts_db_s *)dep->ptr) - 1;
      arts_reader_unlock(&db->reader);
    }
  }
  /* PIN / GPU_PIN / regular RW or RO on non-coherent subtypes: nothing to
   * release at the DB-coherence level.  Hardware coherence and
   * application-level event ordering handle the rest. */
}

/*
 * release_dbs — Release DB dependencies after EDT execution completes.
 * Thin loop over depv calling the single-source-of-truth release_one_dep.
 */
void release_dbs(unsigned int depc, arts_edt_dep_t *depv, bool gpu) {
  for (uint32_t i = 0; i < depc; i++) {
    /* Alias-vs-owner is recorded on the dep at acquire (rw_fire_from_cursor):
     * an alias slot drops only its buffer ref, the owner also releases the
     * single coherence hold.  See arts_edt_dep_t.alias / release_one_dep. */
    release_one_dep(&depv[i], gpu);
  }
}

/*
 * release_one_created — Release a single created DB by GUID.
 *
 * Looks up the DB struct via the route table; for ARTS_DB the
 * creator's hold is the writer_count=2 sentinel set by
 * ARTS_DB_INIT_CREATOR_HOME / ARTS_DB_INIT_CREATOR_REMOTE —
 * release_rw decrements it directly with NO buffer ref to drop
 * (auto_acquire never called acquire_buf).  For non-coherent pinned
 * subtypes (ARTS_DB_PIN, ARTS_DB_GPU_PIN, ARTS_DB_GPU, ARTS_DB_CXL) the
 * creator EDT has no DB-level coherence hold to drop; building a synthetic
 * RW-mode dep and dispatching through release_one_dep handles only the
 * per-mode non-coherence work (GPU-LC reader unlock, CXL producer flush).
 */
static void release_one_created(arts_guid_t guid, arts_db_access_mode_t mode) {
  arts_shared_ptr_t db_h = arts_route_table_lookup_db(guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (!db) {
    return;
  }
  if (db->db_type == ARTS_DB) {
    /* Coherent creator release.  No buffer ref to drop (auto_acquire is a
     * no-op for ARTS_DB — writer_count was pre-stamped to 2 in
     * arts_db_cache_init).  release_rw decrements writer_count, runs
     * R1-R4 transfer logic if rest hits 0, and bumps version.  The coherent
     * creator hold is always RW, so `mode` only steers the pinned-subtype
     * synthetic dep below. */
    arts_db_release_rw(&db->cache);
    arts_shared_release(&db_h);
    return;
  }
  arts_edt_dep_t synthetic = {
      .guid = guid,
      .ptr = (void *)(db + 1),
      .mode = mode,
      .subtype =
          db->db_type, /* pinned subtype (coherent ARTS_DB returned above) */
      .alias = false,
  };
  release_one_dep(&synthetic, false);
  arts_shared_release(&db_h);
}

/*
 * arts_db_release — Release access to a single DB mid-EDT.
 *
 * Two search paths:
 *   1. created_db_list — DBs the current EDT created (auto-acquired EW).
 *   2. depv — DBs received as dependencies (EW or RO mode).
 *
 * Both paths funnel through release_one_dep / release_one_created so the
 * EW / RO / LOCAL / LC / CXL / route-table-ref rules live in exactly one
 * place.  The slot/entry is marked released after the unwind so the
 * epilogue (release_dbs / arts_release_created_dbs) skips it cleanly.
 */
void arts_db_release(arts_guid_t guid, arts_db_access_mode_t mode) {
  /* Path 1: created_db_list (DBs this EDT created) */
  arts_array_list_t *list = arts_get_created_db_list();
  if (list) {
    uint64_t count = arts_length_array_list(list);
    for (uint64_t i = count; i > 0; i--) {
      arts_guid_t *g = (arts_guid_t *)arts_get_from_array_list(list, i - 1);
      if (*g == guid) {
        *g = NULL_GUID;
        release_one_created(guid, mode);
        return;
      }
    }
  }

  /* Path 2: depv (dependency-acquired DBs) */
  if (!current_edt) {
    return;
  }
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(current_edt);
  for (int i = 0; i < current_edt->depc; i++) {
    if (depv[i].guid != guid) {
      continue;
    }
    release_one_dep(&depv[i], false);
    /* Mark the slot released so the epilogue release_dbs skips it.  The alias
     * bit on each slot is independent, so nulling this slot does not affect the
     * remaining slots' alias/owner classification. */
    depv[i].guid = NULL_GUID;
    depv[i].ptr = NULL;
    depv[i].mode = DB_MODE_NULL;
    return;
  }
}

/*
 * arts_release_created_dbs — EDT epilogue helper: release every entry in
 * the thread-local created_db_list that hasn't already been explicitly
 * released by arts_db_release.
 */
void arts_release_created_dbs(void) {
  arts_array_list_t *list = arts_get_created_db_list();
  if (!list) {
    return;
  }
  uint64_t count = arts_length_array_list(list);
  for (uint64_t i = 0; i < count; i++) {
    arts_guid_t *guid = (arts_guid_t *)arts_get_from_array_list(list, i);
    if (*guid == NULL_GUID) {
      continue;
    }
    release_one_created(*guid, DB_MODE_RW);
  }
}

/*
 * arts_wait_release_dbs / arts_wait_reacquire_dbs -- Pre-/post-yield
 * hooks invoked around arts_event_wait.
 *
 * No-op under the OCR model: multi-EDT same-rank concurrent acquire is
 * allowed (writer_count CAS-loop), so the creator's hold persists across the
 * yield and is dropped exactly once at EDT epilogue via
 * arts_release_created_dbs.  Pinned subtypes have no DB-level coherence to
 * drop either.  Kept as stable hooks for future per-EDT release semantics.
 */
void arts_wait_release_dbs(void) {}
void arts_wait_reacquire_dbs(void) {}

/* ── CXL cache-flush helpers ────────────────────────────────────────────────
 */

#ifdef ARTS_USE_CXL
void arts_cxl_producer_flush(arts_guid_t guid) {
  struct arts_db_s *db = (struct arts_db_s *)arts_cxl_get_ptr(guid);
  FLUSH_FENCE_PRODUCER(db, ALIGN_UP(arts_db_total_size(db), CACHELINE_SIZE));
}

void arts_cxl_consumer_flush(arts_guid_t guid) {
  struct arts_db_s *db = (struct arts_db_s *)arts_cxl_get_ptr(guid);
  /* First flush the struct to read the actual size (db_size in the cache). */
  FLUSH_FENCE_CONSUMER(db, ALIGN_UP(sizeof(struct arts_db_s), CACHELINE_SIZE));
  /* Then flush the full DB (struct + payload). */
  if (db->cache.db_size > 0) {
    FLUSH_FENCE_CONSUMER(db, ALIGN_UP(arts_db_total_size(db), CACHELINE_SIZE));
  }
}
#endif /* ARTS_USE_CXL */
