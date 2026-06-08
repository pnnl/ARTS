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
#include "arts/edt.h"
#include "arts/counter/Preamble.h"
#include "arts/gas/guid.h"
#include "arts/gas/route_table.h"
#include "arts/db_coherence.h"
#include "arts/db_coherence_buffer.h"
#include "arts/db_coherence_handlers.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/edt_context.h" /* current_edt + created-DB tracking */
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h" /* arts_shared_ptr_t, get/release */

#ifdef ARTS_USE_GPU
#include "arts/gpu/gpu_internal.h"
#endif

ARTS_TYPE_NAME;
ARTS_DB_TYPE_NAME;
DB_MODE_NAME;

/*
 * arts_db_user_ptr — Return the user-visible data pointer for a DB.
 *
 * For RC (DEFAULT) DBs this is cache->buffer->data (the canonical
 * payload installed by arts_coh_install_buffer at create time or
 * by GRANT/DATA_RESPONSE on sharer ranks).  For all other subtypes it
 * is the legacy (db+1) pointer.
 *
 * Returns NULL if `db` itself is NULL or if a RC DB has no buffer
 * installed yet (transient at create-time; callers should treat this
 * as "data not yet available").
 */
static inline void *arts_db_user_ptr(struct arts_db_s *db) {
  if (db == NULL) {
    return NULL;
  }
  if (db->db_type == ARTS_DB) {
    /* RC: canonical payload lives in the installed buffer's data, not at
     * (db+1).  Unsafe peek — callers use this for the descriptor's payload
     * pointer in single-owner contexts. */
    struct arts_db_buffer_s *buf = arts_coh_buffer_peek(&db->cache);
    return buf ? (void *)buf->data : NULL;
  }
  return (void *)(db + 1);
}

/*
 * arts_db_auto_acquire — Track the creator EDT's hold on a DB.
 *
 * For ARTS_DB: the RC cache_s was allocated with writer_count=2
 * (sentinel + creator EDT) via ARTS_COH_INIT_CREATOR_HOME or
 * ARTS_COH_INIT_CREATOR_REMOTE, so the creator's hold is already
 * counted in the RC state machine.  release_rw drops it at EDT
 * epilogue.
 *
 * For non-RC pinned subtypes (PIN, GPU_PIN, GPU_LC, CXL_LC): there is
 * no DB-level coherence to track; the creator EDT just owns the
 * pointer until it explicitly destroys or hands it off via events.
 *
 * In both cases the GUID is recorded on created_db_list so the EDT
 * epilogue (arts_release_created_dbs) drives the matching release.
 */
static void arts_db_auto_acquire(struct arts_db_s *db) {
  arts_track_created_db(db->cache.db_guid);
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
    ptr = arts_malloc_align(size, 16);
  }
  return ptr;
}

void arts_db_free(void *ptr) {
  struct arts_db_s *db = (struct arts_db_s *)ptr;
  /* Chain into RC cache teardown if this DB has one.  Only ARTS_DB carries
   * coherence state; other subtypes leave the embedded cache zeroed.  The
   * cache is embedded by value as the first member of db_s, so the destructor
   * tears down its sub-resources (buffer pool, home_s) in place — we do NOT
   * free it separately; the db_s free below reclaims its storage. */
  if (db->db_type == ARTS_DB) {
    arts_coh_cache_destructor(&db->cache);
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
                               uint64_t packet_size, arts_db_types_t db_type,
                               uint64_t arts_id) {
  (void)len;
  struct arts_db_s *db_res = (struct arts_db_s *)addr;
  /* lifecycle/deleter handled by the route_table cb (deleter-by-kind) on
   * install — no per-object shared field to initialize. */
  db_res->version = 0;
  db_res->reader = 0;
  db_res->writer = 0;
  db_res->db_type = db_type;
  /* ARTS_DB enters the RC protocol at create time.  Initialize the embedded
   * cache (first member of db_s) with CREATOR_HOME init — arts_db_create only
   * routes here when the local rank is the creator (route ==
   * arts_global_rank_id), which for round-robin home is also the home rank.
   * Non-RC subtypes leave the embedded cache zeroed.
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
    arts_coh_init_cache_s(cache, guid, user_size, ARTS_COH_INIT_CREATOR_HOME,
                          arts_global_rank_id);
    /* Install a fresh buffer so subsequent RC acquires
     * (acquire_local / mark_edt_ready_by_guid) find a non-NULL
     * cache->buffer.  The user pointer returned by arts_db_create
     * points into this buffer's data[] FAM, so writes by the creator
     * EDT land in buf->data and are published when release_rw bumps
     * the version.  No initial data — zero-init is deterministic and
     * matches the DB_CREATE_COHERENT home-recv path. */
    if (user_size > 0) {
      arts_coh_install_buffer(cache, /*new_version=*/1,
                              /*data_payload=*/NULL, user_size);
    }
  }
  /* Non-RC subtypes: PIN, GPU_PIN, GPU_LC, CXL_LC are pinned to the
   * creator rank and have no DB-level coherence.  The embedded cache stays
   * zeroed and db_list stays NULL. */
  if (db_type == ARTS_DB_GPU) {
    void *shadow_copy = (void *)(((char *)addr) + packet_size);
    memcpy(shadow_copy, addr, sizeof(struct arts_db_s));
  }
  // Record per-object DB metrics
  arts_object_record_db(arts_id, packet_size, 0, 0);
  arts_object_trace_db(arts_id, packet_size, 0);
  INCREMENT_NUM_DB_CREATE_BY(1);
  INCREMENT_BYTES_DB_CREATE_BY(len);
}

/*
 * arts_db_create — Unified DataBlock creation.
 *
 * Handles all DB subtypes (RC, PIN, GPU_PIN, GPU_LC, CXL_LC).  When
 * hint->rank targets a remote node, only ARTS_DB is supported:
 * a coherent home stub is installed via DB_CREATE_COHERENT.  Pinned
 * subtypes return NULL_GUID with a warning.
 */
arts_guid_t arts_db_create(void **addr, uint64_t len, arts_db_types_t db_type,
                           uint16_t flags, const arts_db_hint_t *hint) {
  TIME_DB_CREATE_START();
  /* Route resolution:
   *   hint == NULL                          -> round-robin home distribution
   *                                             across all ranks (avoid pinning
   *                                             every DB to the creator).
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
    rank = arts_atomic_fetch_add(&arts_node_info.db_rr_route, 1U) %
           arts_global_rank_count;
  } else if (hint->rank == ARTS_HINT_CURRENT_RANK) {
    rank = arts_global_rank_id;
  } else {
    rank = hint->rank;
  }
  /* DB hint has no profiling id field; arts_db_s.arts_id retained for log
   * compatibility but no longer settable via hint. */
  uint64_t arts_id = 0;
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
        db_create_in_place(guid, ptr, len, db_size, ARTS_DB_CXL, arts_id);
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
          db_create_in_place(guid, ptr, len, db_size, db_type, arts_id);
          /* Register the creator's hold BEFORE the DB becomes visible, then
           * install — both install variants fire the OoO list internally on a
           * successful install (no separate fire_oo needed). */
          if (current_edt && !no_acquire) {
            arts_db_auto_acquire((struct arts_db_s *)ptr);
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
          db_create_in_place(guid, ptr, len, db_size, db_type, arts_id);
          if (current_edt && !no_acquire) {
            arts_db_auto_acquire((struct arts_db_s *)ptr);
          }
          arts_route_table_install(ptr, guid, arts_global_rank_id, true);
        }
        /* For RC (DEFAULT) DBs the canonical user data lives
         * in cache->buffer->data (installed by db_create_in_place),
         * not at (db+1).  arts_db_user_ptr returns the right pointer
         * for both worlds.  NO_ACQUIRE: return NULL so caller cannot
         * accidentally write to the buffer (home is the idle owner). */
        *addr = no_acquire ? NULL : arts_db_user_ptr((struct arts_db_s *)ptr);
        ARTS_DEBUG("arts_db_create: DB[Guid:%lu, Id:%lu, Type:%s, Size:%lu] "
                   "created locally",
                   guid, arts_id, GET_DB_TYPE_NAME(db_type), len);
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
      /* For RC type, ask the home rank to install a coherent cache_s
       * via DB_CREATE_COHERENT.  The home handler
       * (arts_handler_db_create) allocates its own stub +
       * cache_s with ARTS_COH_INIT_HOME_RECV.
       *
       * Also lazy-install a creator-side cache_s on this (non-home)
       * rank via arts_coh_lazy_install_cache_s.  This is necessary so
       * that home's first INVALIDATE_NOTICE (sent to
       * rw_holder = creator_rank when a foreign LOCK_REQ arrives) finds
       * a cache_s on this rank to drop the sentinel and trigger
       * invalidate_transfer.  Without it, home's invalidation goes to a
       * phantom holder and the first foreign acquirer stalls forever. */
      /* Use ARTS_COH_INIT_CREATOR_REMOTE (writer_count = 2: sentinel +
       * creator EDT) so that the home-side INVALIDATE_NOTICE round-trip
       * works correctly.  When a foreign LOCK_REQ arrives at home, home
       * sends INVALIDATE_NOTICE to rw_holder = creator; creator's
       * fetch_sub takes wc 2 -> 1 (no transfer yet -- creator EDT may
       * still be using the buffer).  Creator EDT release_rw drops wc
       * 1 -> 0, triggering R4 WRITEBACK_AND_TRANSFER with the creator's
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
       * arts_coh_install_buffer so the user's `*addr = ...` write
       * lands in cache->buffer->data, and the buffer is captured by the
       * subsequent WRITEBACK_AND_TRANSFER. */
      if (no_acquire) {
        /* NO_ACQUIRE: do NOT lazy-install a creator-side cache_s.  The
         * home is the sole idle owner; first consumer EDT triggers a
         * normal LOCK_REQ to acquire ownership.  Wire only carries
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
        struct arts_db_s *creator_stub =
            (struct arts_db_s *)arts_malloc_align(stub_sz, 16);
        memset(creator_stub, 0, stub_sz);
        creator_stub->db_type = ARTS_DB;
        struct arts_db_cache_s *creator_cache = &creator_stub->cache;
        arts_coh_init_cache_s(creator_cache, guid, len,
                              ARTS_COH_INIT_CREATOR_REMOTE, /*creator_rank=*/0);
        if (len > 0) {
          arts_coh_install_buffer(creator_cache, /*new_version=*/1,
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
          creator_cache = arts_coh_route_table_lookup_cache(guid);
          if (creator_cache != NULL) {
            arts_atomic_add(&creator_cache->writer_count, 2);
          }
        }
        arts_send_db_create_coherent(rank, guid, len, ARTS_DB_PROP_NONE,
                                     (uint16_t)db_type);
        /* Return the creator-side buffer pointer so the user can write to
         * the local copy.  The data is published to home via WRITEBACK
         * when the creator EDT releases (or via per-release writeback).
         * Unsafe peek is fine: the creator owns the freshly-installed buffer.
         */
        struct arts_db_buffer_s *creator_buf =
            creator_cache ? arts_coh_buffer_peek(creator_cache) : NULL;
        *addr = creator_buf ? (void *)creator_buf->data : NULL;
        if (current_edt && creator_cache) {
          /* Auto-acquire: register the DB on the creator EDT's
           * created_db_list so arts_release_created_dbs at EDT epilogue
           * dispatches RC release_rw, dropping the creator's
           * writer_count and triggering the eventual WB+TRANSFER. */
          arts_db_auto_acquire(creator_stub);
        }
      }
      ARTS_DEBUG("arts_db_create: DB[Guid:%lu, Id:%lu, Type:%s, Size:%lu] "
                 "created remotely on rank %u via DB_CREATE_COHERENT",
                 guid, arts_id, GET_DB_TYPE_NAME(db_type), len, rank);
    } else {
      /* Non-RC types (PIN, GPU_PIN, GPU_LC, CXL_LC) are pinned to the
       * creator rank.  Creating one on a different rank is a
       * programming error — return NULL_GUID.  Cross-rank distribution
       * for these subtypes must use ARTS_DB instead. */
      ARTS_WARN("arts_db_create: cannot remote-create non-RC DB type %s on "
                "rank %u (only ARTS_DB is internode relocatable). "
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

  /* RC path: hand off to the coherence-layer destroy entry, which
   * sends DESTROY_REQ to home and runs the fan-out / finalize there. */
  if (db_res != NULL && db_res->db_type == ARTS_DB) {
    arts_shared_release(&db_res_h);
    arts_coh_db_destroy(guid);
    return;
  }

  /* Non-RC pinned subtypes (PIN, GPU_PIN, GPU_LC, CXL_LC): the DB lives
   * only on the creator rank.  Route through
   * arts_route_table_mark_delete — once outstanding refs drop, the cb
   * deleter (arts_db_deleter) runs. */
  if (db_res != NULL) {
    arts_shared_release(&db_res_h);
    arts_route_table_mark_delete(guid);
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
 * Returns ARTS_DB_ACQUIRE_OK with depv[i].ptr filled (NULL is valid: a
 * sentinel / version-0 / no-payload DB) when visibility/ownership is
 * established synchronously, or ARTS_DB_ACQUIRE_PARK when the EDT has been
 * parked on a coherence waiter (or the OoO list) and the protocol's wake will
 * resume the acquire walk.  The 3-way route_table dispatch (local entry /
 * remote-home lazy install / home==self-but-not-created → OoO push) lives
 * here, inlined from the old single-DB arts_db_acquire API.  The caller
 * (arts_db_acquire_all) has already filtered NULL_GUID / DB_MODE_VAL /
 * pre-filled slots, so depv[i] is a real, not-yet-acquired DB dependency. */
static arts_db_acquire_result_t
acquire_one_dep(struct arts_edt_s *edt, arts_edt_dep_t *depv, uint32_t i) {
  arts_db_access_mode_t access_mode = depv[i].mode;
  int owner = (int)arts_guid_get_rank(depv[i].guid);
  arts_guid_kind_t guid_type = arts_guid_get_kind(depv[i].guid);

  if (guid_type != ARTS_GUID_DB) {
    return ARTS_DB_ACQUIRE_OK; /* not a DB GUID — nothing to acquire */
  }

  // Update access-mode counters
  if (access_mode == DB_MODE_RO) {
    INCREMENT_NUM_DB_ACQUIRE_READ_BY(1);
  } else if (access_mode == DB_MODE_RW) {
    INCREMENT_NUM_DB_ACQUIRE_WRITE_BY(1);
    if (owner == arts_global_rank_id) {
      INCREMENT_NUM_OWNER_UPDATE_PERFORMED_BY(1);
    }
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
      return ARTS_DB_ACQUIRE_OK;
    }
    /* Not yet allocated in the shared segment — park; the CXL deque ordering
     * makes the producer's allocation visible before the consumer runs. */
    return ARTS_DB_ACQUIRE_PARK;
  }
#endif

  // Look up DB first — subtype dispatch requires the struct.  lookup_db pairs
  // with the release below (every successful lookup => one release).
  arts_shared_ptr_t db_temp_h = arts_route_table_lookup_db(depv[i].guid);
  struct arts_db_s *db_temp = (struct arts_db_s *)arts_shared_get(db_temp_h);

  /* RC/LRC/LC coherent ARTS_DB path.  Two entry points:
   *   - Existing local cache_s (db_temp with db_type == ARTS_DB; embedded
   *     cache).
   *   - Remote DB never seen on this rank (db_temp == NULL, owner remote):
   *     lazy-install a stub cache_s and dispatch through
   * arts_handler_db_acquire. Other (pinned) subtypes bypass RC and fall through
   * below. */
  struct arts_db_cache_s *coh_cache = NULL;
  if (db_temp != NULL && db_temp->db_type == ARTS_DB) {
    coh_cache = &db_temp->cache;
  } else if (db_temp == NULL && owner != arts_global_rank_id) {
    /* db_size=0 means "size learned on first GRANT/DATA_RESPONSE
     * install_buffer".  Round-robin home is encoded in the GUID, so all
     * ranks agree. */
    coh_cache = arts_coh_lazy_install_cache_s(depv[i].guid, /*db_size=*/0);
  }
  if (coh_cache != NULL &&
      (access_mode == DB_MODE_RO || access_mode == DB_MODE_RW)) {
    arts_db_acquire_result_t r =
        arts_handler_db_acquire(coh_cache, &depv[i], edt->guid, i);
    /* ARTS_DB_ACQUIRE_OK leaves depv[i].ptr written by the handler (it may be
     * NULL for a sentinel db_size==0 / version-0 metadata-only DB; writer_count
     * was bumped on the RW path and release_rw will balance it). */
    if (db_temp != NULL) {
      arts_shared_release(&db_temp_h);
    }
    return r;
  }

  /* Non-RC pinned subtypes (PIN, GPU_PIN, GPU_LC, CXL_LC): the DB lives only
   * on its creator rank — hand back the local pointer if present. */
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
    arts_shared_release(&db_temp_h);
    return ARTS_DB_ACQUIRE_OK;
  }

  /* DB absent locally.  Defer via OoO keyed on the DB GUID and PARK: when
   * DB_CREATE installs it (home==self case) the drain re-attempts this dep
   * via arts_ooo_resolve_db_dep.  A remote non-RC reference can never resolve
   * locally (pinned-subtype contract) and waits here. */
  if (arts_guid_is_local(depv[i].guid)) {
    ARTS_DEBUG("DB[Guid:%lu] out of order request slot %u", depv[i].guid, i);
  } else {
    ARTS_WARN("arts_db_acquire_all: cannot resolve remote DB[Guid:%lu] for "
              "non-RC dep "
              "on rank %u — owner=%u. Deferring via OoO.",
              depv[i].guid, arts_global_rank_id, owner);
  }
  {
    struct arts_ooo_args_db_acquire_s a = {
        .edt = edt, .db_guid = depv[i].guid, .slot = i};
    arts_ooo_dispatch_or_defer_guid(depv[i].guid, OOO_DB_ACQUIRE, &a,
                                    sizeof(a));
  }
  return ARTS_DB_ACQUIRE_PARK;
}

/* arts_db_acquire_all — the arts_db_acquire_all driver.  Acquires the EDT's DB
 * deps in ascending-GUID order, STRICTLY ONE AT A TIME: it advances
 * edt->resume_k over locally-satisfiable deps and, on the first cross-rank dep,
 * PARKs (returns) WITHOUT touching any higher-GUID dep.  The coherence wake
 * (mark_edt_ready_by_guid) or the OoO drain (arts_ooo_resolve_db_dep) advances
 * resume_k and re-enters here; when resume_k == depc all deps are held and the
 * EDT is scheduled.  This strict sequential acquire is what prevents cyclic
 * cross-rank acquire: no two ranks can hold-and-wait on each other's DBs,
 * because each holds at most the prefix below its single outstanding request.
 *
 * Re-entrant + concurrency: at most ONE dep is outstanding (parked) per EDT,
 * so there is at most one wake driving the resume — single-threaded per EDT.
 * After a PARK return the function MUST NOT touch edt (a wake may already be
 * resuming, and may even have scheduled it). */
void arts_db_acquire_all(struct arts_edt_s *edt) {
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
  uint32_t depc = edt->depc;

  /* GUID-sorted index.  Deterministic + stable across re-entries: the only
   * field that mutates mid-acquire is a destroyed dep's guid (→ NULL_GUID),
   * which only moves an already-processed dep further into the processed
   * prefix, so the resume_k boundary is preserved. */
  uint32_t sorted[depc > 0 ? depc : 1];
  for (uint32_t k = 0; k < depc; k++) {
    sorted[k] = k;
  }
  /* Insertion sort by GUID — stable, handles duplicates, fast for small N. */
  for (uint32_t k = 1; k < depc; k++) {
    uint32_t val = sorted[k];
    int j = (int)k - 1;
    while (j >= 0 && depv[sorted[j]].guid > depv[val].guid) {
      sorted[j + 1] = sorted[j];
      j--;
    }
    sorted[j + 1] = val;
  }

  while (edt->resume_k < depc) {
    uint32_t i = sorted[edt->resume_k];

    /* Skip deps that need no DB acquire here:
     *  - guid == NULL_GUID: signaled with no data, or a destroyed dep.
     *  - DB_MODE_VAL: depv[i].guid holds a raw uint64, not a real GUID.
     *  - ptr already set: filled during the satisfy phase (or a prior wake). */
    if (depv[i].guid == NULL_GUID || depv[i].mode == DB_MODE_VAL ||
        depv[i].ptr != NULL) {
      edt->resume_k++;
      continue;
    }

    arts_db_acquire_result_t rc = acquire_one_dep(edt, depv, i);
    if (rc == ARTS_DB_ACQUIRE_PARK) {
      /* Outstanding cross-rank dep at the resume_k frontier — stop the walk.
       * Its wake advances resume_k and re-enters.  Do NOT touch edt below. */
      return;
    }
    ARTS_DEBUG("DB[Guid:%lu, Ptr:%p] acquired", depv[i].guid, depv[i].ptr);
    edt->resume_k++;
  }

  ARTS_INFO("EDT[Id:%lu, Guid:%lu] has finished acquiring DBs", edt->arts_id,
            edt->guid);
  arts_schedule_ready_edt(edt);
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
    /* For RC DBs, dep->ptr is buf->data and pointer arithmetic to
     * recover db_s would land in the buffer header, NOT a db_s.  Detect
     * via the coherence adapter and skip — RC drives invalidation
     * via LOCK_REQ inside the coherence layer.  Non-RC pinned subtypes
     * (PIN, GPU_PIN, GPU_LC, CXL_LC) have no DB-level coherence and
     * therefore no inter-rank invalidation step at prep time. */
    if (arts_coh_route_table_lookup_cache(depv[i].guid) != NULL) {
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
 *     work; route through the RC release entry points
 *     (arts_coh_release_ro / arts_coh_release_rw).  Non-RC pinned
 *     subtypes have no DB-level coherence — release is a no-op.
 *   - DB_MODE_PTR: free the malloc'd copy buffer.
 *   - GPU_LC subtype (GPU build, non-LC_SYNC mode): release the LC reader
 *     lock — pure intra-rank multi-device coordination.
 *   - CXL_LC subtype: producer-flush and return.
 *
 * Does NOT nullify caller-visible state (guid/ptr/mode).  Callers that
 * need to mark the slot as released (mid-EDT release) do that themselves.
 */
static void release_one_dep(arts_edt_dep_t *dep, bool gpu) {
  arts_db_access_mode_t access_mode = dep->mode;

  /* RC release path for ARTS_DB.  For RC DBs, dep->ptr is
   * cache->buffer->data (NOT (db+1)), so we cannot recover the
   * arts_db_s by pointer arithmetic.  Look up by GUID via the
   * coherence adapter; if it returns a cache_s, route through the RC
   * RC release entry points.  Drops the EDT's per-acquire buffer ref
   * taken at acquire time (acquire_local / mark_edt_ready_by_guid each
   * do arts_coh_acquire_buf), then dispatches release_rw /
   * release_ro to handle writeback / ownership transfer / version
   * bump per mode. */
  if (dep->guid != NULL_GUID &&
      (access_mode == DB_MODE_RO || access_mode == DB_MODE_RW)) {
    struct arts_db_cache_s *coh_cache =
        arts_coh_route_table_lookup_cache(dep->guid);
    if (coh_cache != NULL) {
      if (dep->ptr != NULL) {
        struct arts_db_buffer_s *buf = arts_coh_buf_from_data(dep->ptr);
        if (buf != NULL) {
          /* Drop the EDT's acquire ref via the buffer's own cb.  Safe: the
           * EDT's ref kept the buffer (hence buf->cb) alive up to here, so
           * the deref is never use-after-free even under a racing destroy. */
          arts_shared_ptr_t buf_cb = buf->cb;
          arts_coh_release_buf(&buf_cb);
        }
      }
      if (access_mode == DB_MODE_RW) {
        arts_coh_release_rw(coh_cache);
      } else {
        arts_coh_release_ro(coh_cache);
      }
      return;
    }
  }

  /* Get DB subtype from struct when ptr is available.  Guard with
   * guid != NULL_GUID because arts_db_release may have already nulled
   * the guid while leaving ptr non-NULL (caller responsibility).
   *
   * Reaching this point means the dep is for a non-RC pinned subtype
   * (PIN, GPU_PIN, GPU_LC, CXL_LC) or a special access mode (PTR,
   * VALUE, LC_*, MEMSET) — none of which carry DB-level coherence. */
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
  /* PIN / GPU_PIN / regular RW or RO on non-RC subtypes: nothing to
   * release at the DB-coherence level.  Hardware coherence and
   * application-level event ordering handle the rest. */
}

/*
 * release_dbs — Release DB dependencies after EDT execution completes.
 * Thin loop over depv calling the single-source-of-truth release_one_dep.
 */
void release_dbs(unsigned int depc, arts_edt_dep_t *depv, bool gpu) {
  for (int i = 0; i < depc; i++) {
    release_one_dep(&depv[i], gpu);
  }
}

/*
 * release_one_created — Release a single created DB by GUID.
 *
 * Looks up the DB struct via the route table; for ARTS_DB the
 * creator's hold is the writer_count=2 sentinel set by
 * ARTS_COH_INIT_CREATOR_HOME / ARTS_COH_INIT_CREATOR_REMOTE —
 * release_rw decrements it directly with NO buffer ref to drop
 * (auto_acquire never called acquire_buf).  For non-RC pinned
 * subtypes (PIN, GPU_PIN, GPU_LC, CXL_LC) the creator EDT has no
 * DB-level coherence hold to drop; building a synthetic RW-mode dep
 * and dispatching through release_one_dep handles only the per-mode
 * non-coherence work (LC reader unlock, CXL producer flush).
 */
static void release_one_created(arts_guid_t guid, arts_db_access_mode_t mode) {
  arts_shared_ptr_t db_h = arts_route_table_lookup_db(guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(db_h);
  if (!db) {
    return;
  }
  if (db->db_type == ARTS_DB) {
    /* RC creator release.  No buffer ref to drop (auto_acquire is a
     * no-op for RC — writer_count was pre-stamped to 2 in
     * arts_coh_init_cache_s).  release_rw decrements writer_count, runs
     * R1-R4 transfer logic if rest hits 0, and bumps version.  The RC
     * creator hold is always RW, so `mode` only steers the pinned-subtype
     * synthetic dep below. */
    arts_coh_release_rw(&db->cache);
    arts_shared_release(&db_h);
    return;
  }
  arts_edt_dep_t synthetic = {
      .guid = guid,
      .ptr = (void *)(db + 1),
      .mode = mode,
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
    /* Mark the slot released so the epilogue release_dbs skips it. */
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
 * No-op under RC: multi-EDT same-rank concurrent acquire is allowed
 * (writer_count CAS-loop), so the creator's hold persists across the
 * yield and is dropped exactly once at EDT epilogue via
 * arts_release_created_dbs.  Pinned subtypes have no DB-level
 * coherence to drop either.  Kept as stable hooks for future per-EDT
 * release semantics.
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
