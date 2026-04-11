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

#include "arts/memory/db.h"

#include <assert.h>
#include <string.h>

#include "arts.h"
#include "arts/cxl/wrapper.h"
#ifdef ARTS_USE_CXL
#include "arts/cxl/deque.h"
#endif
#include "arts/compute/edt.h"
#include "arts/counter/Preamble.h"
#include "arts/gas/guid.h"
#include "arts/gas/out_of_order.h"
#include "arts/gas/route_table.h"
#include "arts/memory/cdag_lock.h"
#include "arts/remote/handler.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/sync/termination.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/transport/protocol.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

#ifdef ARTS_USE_GPU
#include "arts/gpu/gpu_internal.h"
#endif

ARTS_TYPE_NAME;
ARTS_DB_TYPE_NAME;
DB_MODE_NAME;

extern ARTS_THREAD_LOCAL struct arts_edt_s *current_edt;

// True for DB subtypes that have a CDAG frontier (remote-capable).
static inline bool arts_db_subtype_has_frontier(arts_db_types_t db_type) {
  if (db_type == ARTS_DB_LOCAL) {
    return false;
  }
#ifdef ARTS_USE_CXL
  if (db_type == ARTS_DB_CXL) {
    return false;
  }
#endif
  return true;
}

/*
 * arts_cdag_dispatch_cb — on_advance callback for cdag_lock_release.
 *
 * Invoked once per newly-runnable request when the head generation
 * drains and the next one becomes head. Handles both local EDTs
 * (decrement depc_needed, set depv[slot].ptr, kick scheduler if ready)
 * and remote EDTs (send the DB via remote_db_full_send_now).
 *
 * ctx is the arts_db_s pointer that owns the lock.
 */
void arts_cdag_dispatch_cb(const struct cdag_lock_request_s *req, void *ctx) {
  struct arts_db_s *db = (struct arts_db_s *)ctx;
  if (!db) {
    return;
  }

  if (req->origin_rank == arts_global_rank_id) {
    /* Local dispatch: hook the waiter into its EDT's dep slot. */
    struct arts_edt_s *edt = req->edt;
    arts_guid_t edt_guid = req->edt_guid;
    if (!edt && edt_guid != NULL_GUID) {
      edt = (struct arts_edt_s *)arts_route_table_lookup_item(edt_guid);
    }
    if (!edt) {
      ARTS_INFO("cdag dispatch: missing local EDT[Guid:%lu] for DB[Guid:%lu]",
                edt_guid, db->guid);
      return;
    }
    arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
    /* Acquire a route-table ref matched by return_db in release_dbs. */
    arts_route_table_lookup_db(db->guid, NULL, false);
    depv[req->slot].ptr = db + 1;
    if (arts_atomic_sub(&edt->depc_needed, 1U) == 0) {
      arts_handle_remote_stolen_edt(edt);
    }
  } else {
    /* Remote dispatch: send the full DB (with EDT/slot metadata) to the
     * origin rank. The receiver's arts_remote_handle_db_full_recieved
     * hooks the data into the target EDT's dep slot. */
    arts_remote_db_full_send_now((int)req->origin_rank, db, req->edt_guid,
                                 req->slot, req->mode);
  }
}

/*
 * arts_db_auto_acquire — Automatically acquire WRITE access for the creator
 * EDT.
 *
 * Called when an EDT creates a local DB. Submits a phantom EW request on
 * behalf of the creator so the cdag_lock is held at creation time; any
 * subsequent consumer submit lands in a new generation behind this
 * creator gen. When the creator releases (arts_release_created_dbs or
 * arts_db_release), the gen drains and consumers are dispatched.
 */
static void arts_db_auto_acquire(struct arts_db_s *db) {
  if (db->db_list) {
    struct cdag_lock_s *lock = (struct cdag_lock_s *)db->db_list;
    struct cdag_lock_request_s req = {0};
    req.edt = current_edt;
    req.edt_guid = current_edt ? current_edt->current_edt : NULL_GUID;
    req.origin_rank = arts_global_rank_id;
    req.slot = 0;
    req.mode = DB_MODE_EW;
    (void)cdag_lock_submit(lock, &req);
    /* The creator is already running; it doesn't need a dispatch callback.
     * HEAD_IMMEDIATE means the lock is held from now on until release. */
  }
  arts_track_created_db(db->guid);
}

void *arts_db_malloc(arts_db_types_t db_type, size_t size) {
  (void)db_type;
  void *ptr = NULL;
#ifdef ARTS_USE_GPU
  if (arts_node_info.gpu) {
    if (db_type == ARTS_DB_LC)
      ptr = arts_cuda_malloc_host(size * 2);
    else if (db_type == ARTS_DB_GPU)
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
                                       &arts_node_info.cxl_local_lock,
                                       size, dev_idx);
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
  if (db->db_list && db->db_list != (void *)1) {
    cdag_lock_free((struct cdag_lock_s *)db->db_list);
    db->db_list = NULL;
  }
#ifdef ARTS_USE_GPU
  if (arts_node_info.gpu &&
      (db->db_type == ARTS_DB_GPU || db->db_type == ARTS_DB_LC)) {
    arts_cuda_free_host(ptr);
    ptr = NULL;
  }
#endif
  if (ptr) {
    arts_free(ptr);
  }
}

/*
 * arts_db_create_internal — Initialize a DB header in pre-allocated memory.
 *
 * Sets up the arts_db_s header fields (type, size, version, reader/writer
 * counts, db_list) and records metrics.  The caller is responsible for
 * route-table registration.
 */
void arts_db_create_internal(arts_guid_t guid, void *addr, uint64_t len,
                             uint64_t packet_size, arts_db_types_t db_type,
                             uint64_t arts_id) {
  (void)len;
  struct arts_header_s *header = (struct arts_header_s *)addr;
  header->type = ARTS_DB; // All DB subtypes share one GUID type tag
  header->size = packet_size;

  struct arts_db_s *db_res = (struct arts_db_s *)header;
  db_res->arts_id = arts_id;
  db_res->guid = guid;
  db_res->version = 0;
  db_res->reader = 0;
  db_res->writer = 0;
  db_res->copy_count = 1;
  db_res->db_type = db_type;
  if (arts_db_subtype_has_frontier(db_type)) {
    db_res->db_list = cdag_lock_new();
  } else {
    db_res->db_list = NULL;
  }
  if (db_type == ARTS_DB_LC) {
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
 * Handles all DB subtypes (DEFAULT, LOCAL, GPU, LC).  When hint->route
 * targets a remote node, sends a stub via ARTS_REMOTE_DB_SEND_MSG and
 * sets *addr = NULL.
 */
arts_guid_t arts_db_create(void **addr, uint64_t len, arts_db_types_t db_type,
                           const arts_hint_t *hint) {
  TIME_DB_CREATE_START();
  unsigned int route = (hint && hint->route != ARTS_HINT_CURRENT_NODE)
                           ? hint->route
                           : arts_global_rank_id;
  uint64_t arts_id = hint ? hint->id : 0;
  arts_guid_t guid = NULL_GUID;

  if (route == arts_global_rank_id) {
    uint64_t db_size = len + sizeof(struct arts_db_s);
#ifdef ARTS_USE_CXL
    if (db_type == ARTS_DB_CXL) {
      db_size = ALIGN_UP(db_size, CACHELINE_SIZE);
      void *ptr = arts_db_malloc(ARTS_DB_CXL, db_size);
      if (ptr) {
        guid = arts_cxl_make_guid(ptr);
        arts_db_create_internal(guid, ptr, len, db_size, ARTS_DB_CXL, arts_id);
        /* No route table entry — GUID encodes CXL pointer directly */
        // FLUSH_FENCE_PRODUCER(ptr, db_size);
        FLUSH_FENCE_PRODUCER(ptr, sizeof(struct arts_db_s));
        *addr = (void *)((struct arts_db_s *)ptr + 1);
        ARTS_DEBUG("arts_db_create: CXL DB[Guid:%lu, Size:%lu] created", guid,
                   len);
      }
    } else
#endif
    {
      void *ptr = arts_db_malloc(db_type, db_size);
      if (ptr) {
        guid = arts_guid_create_for_rank(arts_global_rank_id, ARTS_DB);
        arts_db_create_internal(guid, ptr, len, db_size, db_type, arts_id);
        arts_route_table_add_item(ptr, guid, arts_global_rank_id, false);
        if (current_edt) {
          arts_db_auto_acquire((struct arts_db_s *)ptr);
        }
        *addr = (void *)((struct arts_db_s *)ptr + 1);
        ARTS_DEBUG("arts_db_create: DB[Guid:%lu, Id:%lu, Type:%s, Size:%lu] "
                   "created locally",
                   guid, arts_id, GET_DB_TYPE_NAME(db_type), len);
      }
    }
  } else {
    guid = arts_guid_create_for_rank(route, ARTS_DB);
    void *ptr = arts_db_malloc(db_type, sizeof(struct arts_db_s));
    struct arts_db_s *db = (struct arts_db_s *)ptr;
    db->header.type = ARTS_DB;
    db->header.size = len + sizeof(struct arts_db_s);
    db->guid = guid;
    db->db_type = db_type;
    db->db_list = (void *)1;
    // Send stub using arts_remote_db_send_packet_s format (matches receiver).
    // Only the header struct is sent; the receiver allocates the full size.
    struct arts_remote_db_send_packet_s send_pkt;
    uint64_t pkt_size = sizeof(send_pkt) + sizeof(struct arts_db_s);
    arts_fill_packet_header(&send_pkt.header, pkt_size,
                            ARTS_REMOTE_DB_SEND_MSG);
    arts_remote_send_request_payload_async_free(
        (int)route, (char *)&send_pkt, sizeof(send_pkt), (char *)ptr, 0,
        sizeof(struct arts_db_s), arts_db_free);
    arts_route_table_remove_item(guid);
    *addr = NULL;
    ARTS_DEBUG("arts_db_create: DB[Guid:%lu, Id:%lu, Type:%s, Size:%lu] "
               "created remotely on rank %u",
               guid, arts_id, GET_DB_TYPE_NAME(db_type), len, route);
  }
  TIME_DB_CREATE_STOP();
  return guid;
}

// Guid must be for a local DB only
void *arts_db_create_with_guid(arts_guid_t guid, uint64_t len,
                               arts_db_types_t db_type, const void *data,
                               const arts_hint_t *hint) {
  TIME_DB_CREATE_START();
  uint64_t arts_id = hint ? hint->id : 0;

  void *ptr = NULL;
  if (arts_guid_is_local(guid)) {
    uint64_t db_size = len + sizeof(struct arts_db_s);

    ptr = arts_db_malloc(db_type, db_size);
    if (ptr) {
      struct arts_db_s *db_header = (struct arts_db_s *)ptr;
      arts_db_create_internal(guid, db_header, len, db_size, db_type, arts_id);
      if (data) {
        memcpy((void *)(db_header + 1), data, len);
      }
      if (arts_route_table_add_item_race(db_header, guid, arts_global_rank_id,
                                         false)) {
        arts_route_table_fire_oo(guid, arts_out_of_order_handler);
      }
      if (current_edt) {
        arts_db_auto_acquire(db_header);
      }
      ptr = (void *)(db_header + 1);
    }
  }
  ARTS_INFO("Creating DB[Id:%lu, Guid:%lu, Type:%s, Ptr:%p, Route:%d, "
            "Size:%lu]",
            arts_id, guid, GET_DB_TYPE_NAME(db_type), ptr,
            arts_guid_get_rank(guid), len);
  TIME_DB_CREATE_STOP();
  return ptr;
}

void *arts_db_adopt(arts_guid_t guid, struct arts_db_s *db) {
  if (arts_route_table_add_item_race(db, guid, arts_global_rank_id, false)) {
    arts_route_table_fire_oo(guid, arts_out_of_order_handler);
  }
  if (current_edt) {
    arts_db_auto_acquire(db);
  }
  return (void *)(db + 1);
}

void *arts_db_resize_ptr(struct arts_db_s *db_res, unsigned int size,
                         bool copy) {
  if (db_res) {
    unsigned int old_size = db_res->header.size;
    unsigned int new_size = size + sizeof(struct arts_db_s);
    struct arts_db_s *ptr =
        (struct arts_db_s *)arts_calloc_align(1, new_size, 16);
    if (ptr) {
      if (copy) {
        memcpy(ptr, db_res, old_size);
      } else {
        memcpy(ptr, db_res, sizeof(struct arts_db_s));
      }
      arts_free(db_res);
      ptr->header.size = size + sizeof(struct arts_db_s);
      return (void *)(ptr + 1);
    }
  }
  return NULL;
}

// Must be in write mode (or only copy) to update and alloced (no NO_ACQUIRE
// nonsense), otherwise will be racy...
void *arts_db_resize(arts_guid_t guid, unsigned int size, bool copy) {
  struct arts_db_s *db_res =
      (struct arts_db_s *)arts_route_table_lookup_db(guid, NULL, false);
  void *ptr = arts_db_resize_ptr(db_res, size, copy);
  if (ptr) {
    db_res = ((struct arts_db_s *)ptr) - 1;
  }
  if (db_res) {
    arts_route_table_return_db(guid, false);
  }
  return ptr;
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
  arts_type_t type = arts_guid_get_type(guid);
  if (type != ARTS_DB) {
    ARTS_WARN("arts_db_destroy called with non-DB type %u (GUID %lu)", type,
              guid);
    return;
  }

  /* Implicit release: release the caller's acquire if held. */
  arts_db_release(guid);

  struct arts_db_s *db_res =
      (struct arts_db_s *)arts_route_table_lookup_db(guid, NULL, false);
  if (db_res != NULL) {
    if (arts_db_subtype_has_frontier(db_res->db_type)) {
      arts_remote_db_destroy(guid, arts_global_rank_id);
    }
    arts_route_table_return_db(guid, false);
    arts_route_table_mark_delete(guid);
  } else {
    arts_remote_db_destroy(guid, arts_global_rank_id);
  }
}

bool arts_db_rename_with_guid(arts_guid_t new_guid, arts_guid_t old_guid) {
  bool ret = false;
  unsigned int rank = arts_guid_get_rank(old_guid);
  if (rank == arts_global_rank_id) {
    struct arts_db_s *db_res =
        (struct arts_db_s *)arts_route_table_lookup_db(old_guid, NULL, false);
    if (db_res != NULL) {
      db_res->guid = new_guid;
      // This is only being done by the owner...
      arts_route_table_hide_item(old_guid);
      if (arts_route_table_add_item_race(db_res, new_guid, arts_global_rank_id,
                                         false)) {
        arts_route_table_fire_oo(new_guid, arts_out_of_order_handler);
      }
      arts_route_table_return_db(old_guid, false);
      ret = true;
    }
  } else {
    arts_remote_db_rename(new_guid, old_guid);
  }
  return ret;
}

arts_guid_t arts_db_copy_to_new_type(arts_guid_t old_guid,
                                     arts_db_types_t new_type) {
  arts_guid_t ret = NULL_GUID;
  unsigned int rank = arts_guid_get_rank(old_guid);
  if (rank == arts_global_rank_id) {
    arts_guid_t new_guid = arts_guid_create_for_rank(rank, ARTS_DB);
    struct arts_db_s *db_res =
        (struct arts_db_s *)arts_route_table_lookup_db(old_guid, NULL, false);
    if (db_res != NULL) {
      arts_atomic_add(&db_res->copy_count, 1);
      db_res->guid = new_guid;
      db_res->db_type = new_type;
      if (arts_route_table_add_item_race(db_res, new_guid, arts_global_rank_id,
                                         false)) {
        arts_route_table_fire_oo(new_guid, arts_out_of_order_handler);
      }
      arts_route_table_return_db(old_guid, false);
      ret = new_guid;
    }
  }
  return ret;
}

arts_guid_t arts_db_rename(arts_guid_t guid) {
  arts_guid_t new_guid = arts_guid_create_for_rank(arts_guid_get_rank(guid),
                                                   arts_guid_get_type(guid));
  return (arts_db_rename_with_guid(new_guid, guid)) ? new_guid : NULL_GUID;
}

void arts_db_destroy_safe(arts_guid_t guid, bool remote) {
  /* Implicit release: release the caller's acquire if held. */
  arts_db_release(guid);

  struct arts_db_s *db_res =
      (struct arts_db_s *)arts_route_table_lookup_db(guid, NULL, false);
  if (db_res != NULL) {
    if (remote && arts_db_subtype_has_frontier(db_res->db_type)) {
      arts_remote_db_destroy(guid, arts_global_rank_id);
    }
    arts_route_table_return_db(guid, false);
    arts_route_table_mark_delete(guid);
  } else if (remote) {
    // No local copy — forward destroy to remote if this is a DB GUID
    arts_remote_db_destroy(guid, arts_global_rank_id);
  }
}

/**********************DB MEMORY MODEL*************************************/
// Side Effects: edt depc_needed will be incremented, ptr will be updated,
//   and launches out of order handleReadyEdt
// Returns false on out of order and true otherwise
void acquire_dbs(struct arts_edt_s *edt) {
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
  edt->depc_needed = edt->depc + 1;
  ARTS_INFO("Acquiring %u DBs for EDT[Id:%lu, Guid:%lu], depc_needed "
            "initialized to %u",
            edt->depc, edt->arts_id, edt->current_edt, edt->depc_needed);

  /* Build GUID-sorted index array for deadlock-free acquisition order.
   * Acquiring DBs in ascending GUID order prevents circular wait when
   * multiple EDTs need overlapping DB sets in EW mode. */
  uint32_t sorted[edt->depc > 0 ? edt->depc : 1];
  for (uint32_t k = 0; k < edt->depc; k++) {
    sorted[k] = k;
  }
  /* Insertion sort by GUID — stable, handles duplicates, fast for small N. */
  for (uint32_t k = 1; k < edt->depc; k++) {
    uint32_t val = sorted[k];
    int j = (int)k - 1;
    while (j >= 0 && depv[sorted[j]].guid > depv[val].guid) {
      sorted[j + 1] = sorted[j];
      j--;
    }
    sorted[j + 1] = val;
  }

  for (uint32_t si = 0; si < edt->depc; si++) {
    int i = (int)sorted[si]; /* Acquire in GUID order, not slot order. */
    /*
     * A slot with guid == NULL_GUID (0) but mode != DB_MODE_NULL was
     * signaled via an event that carried no data — already satisfied,
     * no DB to acquire.  Count it immediately.
     */
    if (depv[i].guid == NULL_GUID && depv[i].mode != DB_MODE_NULL) {
      arts_atomic_sub(&edt->depc_needed, 1U);
      continue;
    }
    if (depv[i].guid && depv[i].ptr == NULL) {
      arts_db_access_mode_t access_mode = depv[i].mode;

      /*
       * Value signals (DB_MODE_VALUE) store a raw uint64 in
       * depv[slot].guid — it is NOT a real GUID.  Skip DB acquisition
       * entirely; just count this slot as satisfied.
       */
      if (access_mode == DB_MODE_VALUE) {
        arts_atomic_sub(&edt->depc_needed, 1U);
        continue;
      }

      struct arts_db_s *db_found = NULL;
      int owner = (int)arts_guid_get_rank(depv[i].guid);
      arts_type_t guid_type = arts_guid_get_type(depv[i].guid);

      // Update access-mode counters
      if (access_mode == DB_MODE_RO) {
        INCREMENT_NUM_DB_ACQUIRE_READ_BY(1);
      } else if (access_mode == DB_MODE_EW) {
        INCREMENT_NUM_DB_ACQUIRE_WRITE_BY(1);
        if (owner == arts_global_rank_id) {
          INCREMENT_NUM_OWNER_UPDATE_PERFORMED_BY(1);
        }
      }

      ARTS_INFO("Acquiring DB[Guid:%lu, GuidType:%u, AccessMode:%u, Owner:%d, "
                "Rank:%u] in EDT[Id:%lu, Guid:%lu, Slot:%u]",
                depv[i].guid, guid_type, access_mode, owner,
                arts_global_rank_id, edt->arts_id, edt->current_edt, i);

      if (guid_type == ARTS_DB) {
#ifdef ARTS_USE_CXL
        if (arts_guid_is_cxl(depv[i].guid)) {
          struct arts_db_s *cxl_db =
              (struct arts_db_s *)arts_cxl_get_ptr(depv[i].guid);
          arts_cxl_consumer_flush(depv[i].guid);
          if (cxl_db) {
            db_found = cxl_db;
            arts_atomic_sub(&edt->depc_needed, 1U);
          }
        } else
#endif
        {
          // Look up DB first — subtype dispatch requires the struct
          int valid_rank = -1;
          struct arts_db_s *db_temp =
              (struct arts_db_s *)arts_route_table_lookup_db(depv[i].guid,
                                                             &valid_rank, true);
          /* Track whether lookup acquired a route table ref so we can
           * return it if the dep is deferred (frontier/remote/OO). */
          bool lookup_ref_held = (db_temp != NULL);

          if (db_temp && db_temp->db_type == ARTS_DB_LOCAL) {
            // LOCAL: direct access, no frontier
            db_found = db_temp;
            arts_atomic_sub(&edt->depc_needed, 1U);
          } else if (db_temp && access_mode == DB_MODE_LC_SYNC &&
                     owner == arts_global_rank_id) {
            // LC_SYNC on owner — direct lookup, skip frontier
            ARTS_DEBUG("LC_SYNC -> %p", db_temp);
            db_found = db_temp;
            arts_atomic_sub(&edt->depc_needed, 1U);
          } else if (db_temp && owner == arts_global_rank_id) {
            // Owner path: CDAG frontier for DEFAULT/GPU/LC
            bool on_head = false;
            bool duplicate_added = arts_add_db_duplicate(
                db_temp, arts_global_rank_id, edt, edt->current_edt, i,
                access_mode, &on_head);
            if (duplicate_added) {
              ARTS_DEBUG("Adding duplicate DB[Guid:%lu] on_head=%d",
                         depv[i].guid, on_head);
            } else {
              ARTS_DEBUG(
                  "Duplicate not added DB[Guid:%lu] (rank already tracked)",
                  depv[i].guid);
            }

            if (valid_rank == arts_global_rank_id && on_head) {
              db_found = db_temp;
              arts_atomic_sub(&edt->depc_needed, 1U);
            } else if (valid_rank == arts_global_rank_id) {
              ARTS_DEBUG("EDT[Guid:%lu] deferred to frontier for "
                         "DB[Guid:%lu] (non-head local frontier, unique=%d)",
                         edt->current_edt, depv[i].guid, duplicate_added);
            } else {
              if (access_mode == DB_MODE_RO ||
                  db_temp->db_type == ARTS_DB_GPU ||
                  db_temp->db_type == ARTS_DB_LC) {
                arts_remote_db_request(depv[i].guid, valid_rank, edt, i,
                                       access_mode, true);
              } else {
                arts_remote_db_full_request(depv[i].guid, valid_rank,
                                            edt->current_edt, i, access_mode);
              }
            }
          } else if (db_temp) {
            // Non-owner path: cached copy management
            bool local_valid = (valid_rank == arts_global_rank_id);
            if (local_valid) {
              db_found = db_temp;
              arts_atomic_sub(&edt->depc_needed, 1U);
            } else if (access_mode == DB_MODE_EW) {
              arts_remote_db_full_request(depv[i].guid, owner, edt->current_edt,
                                          i, access_mode);
            } else {
              arts_remote_db_request(depv[i].guid, owner, edt, i, access_mode,
                                     true);
            }
          } else {
            // DB not in route table — out-of-order or remote
            if (arts_guid_is_local(depv[i].guid)) {
              ARTS_DEBUG("DB[Guid:%lu] out of order request slot %u",
                         depv[i].guid, i);
              arts_out_of_order_handle_db_request(depv[i].guid, edt, i, true);
            } else {
              // Remote DB not cached locally — request from owner
              if (access_mode == DB_MODE_EW) {
                arts_remote_db_full_request(depv[i].guid, owner,
                                            edt->current_edt, i, access_mode);
              } else {
                arts_remote_db_request(depv[i].guid, owner, edt, i, access_mode,
                                       true);
              }
            }
          }

          /* If the lookup succeeded but the dep was deferred (frontier,
           * remote request, OO), db_found is NULL and the lookup ref was
           * never transferred to depv[i].ptr.  Return it now. */
          if (lookup_ref_held && !db_found) {
            arts_route_table_return_db(depv[i].guid, false);
          }
        } /* end non-CXL ARTS_DB path */
      } else if (guid_type == ARTS_NULL) {
        arts_atomic_sub(&edt->depc_needed, 1U);
      }

      if (db_found) {
        depv[i].ptr = db_found + 1;
      }
      ARTS_DEBUG("DB[Guid:%lu, Ptr:%p] acquired", depv[i].guid, depv[i].ptr);
    } else {
      arts_atomic_sub(&edt->depc_needed, 1U);
    }
  }
  ARTS_INFO("EDT[Id:%lu, Guid:%lu] has finished acquiring DBs", edt->arts_id,
            edt->current_edt);
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
    if (depv[i].guid != NULL_GUID && depv[i].ptr && access_mode == DB_MODE_EW) {
      struct arts_db_s *db = ((struct arts_db_s *)depv[i].ptr) - 1;
      FLUSH_FENCE_CONSUMER(db, sizeof(struct arts_db_s));
      uint64_t data_size = db->header.size - sizeof(struct arts_db_s);
      if (db->db_type != ARTS_DB_LOCAL) {
        arts_remote_update_route_table(depv[i].guid, ARTS_HINT_CURRENT_NODE);
      }
      ARTS_DEBUG("[prep_dbs] DB[Id:%lu, Guid:%lu] ptr=%p, db=%p", db->arts_id,
                 depv[i].guid, depv[i].ptr, db);
    }
#ifdef ARTS_USE_CXL
    if (depv[i].guid != NULL_GUID && depv[i].ptr) {
      struct arts_db_s *db_cxl = ((struct arts_db_s *)depv[i].ptr) - 1;
      if (db_cxl->db_type == ARTS_DB_CXL) {
        arts_cxl_consumer_flush(db_cxl->guid);
      }
    }
#endif
#ifdef ARTS_USE_GPU
    if (!gpu && depv[i].ptr && access_mode != DB_MODE_LC_SYNC) {
      struct arts_db_s *db = ((struct arts_db_s *)depv[i].ptr) - 1;
      if (db->db_type == ARTS_DB_LC) {
        arts_reader_lock(&db->reader, &db->writer);
        internal_inc_db_version(&db->version);
      }
    }

    if (!gpu && access_mode == DB_MODE_LC_SYNC && depv[i].ptr) {
      struct arts_db_s *db = ((struct arts_db_s *)depv[i].ptr) - 1;
      ARTS_DEBUG("internalLCSync %lu %p", depv[i].guid, db);
      internal_lc_sync_gpu(depv[i].guid, db);
    }
#endif
  }
}

/*
 * release_dbs — Release DB dependencies after EDT execution completes.
 *
 * For WRITE-mode deps owned locally: progresses the CDAG frontier.
 * For WRITE-mode deps on remote owners: sends the updated data back to
 * the owner node.  For READ-mode: returns the route table entry.
 * For LC DBs (GPU builds): releases the reader lock.
 */
void release_dbs(unsigned int depc, arts_edt_dep_t *depv, bool gpu) {
  for (int i = 0; i < depc; i++) {
    arts_db_access_mode_t access_mode = depv[i].mode;
    // Get DB subtype from struct when ptr is available.
    // Guard with guid != NULL_GUID: arts_db_release may have nulled the
    // guid while leaving ptr non-NULL (or the DB may have been freed).
    arts_db_types_t db_subtype = ARTS_DB_DEFAULT;
    if (depv[i].guid != NULL_GUID && depv[i].ptr) {
      struct arts_db_s *db_hdr = ((struct arts_db_s *)depv[i].ptr) - 1;
      db_subtype = db_hdr->db_type;
    }

    ARTS_DEBUG("Releasing DB[Guid:%lu] [AccessMode:%s, DbSubtype:%s]",
               depv[i].guid, GET_DB_MODE_NAME(access_mode),
               GET_DB_TYPE_NAME(db_subtype));

#ifdef ARTS_USE_CXL
    if (db_subtype == ARTS_DB_CXL) {
      if (depv[i].guid != NULL_GUID && depv[i].ptr) {
        arts_cxl_producer_flush(depv[i].guid);
      }
      continue; /* CXL: no route table, no frontier */
    }
#endif

    unsigned int owner = arts_guid_get_rank(depv[i].guid);

    if (depv[i].guid != NULL_GUID &&
        (access_mode == DB_MODE_EW || access_mode == DB_MODE_MEMSET)) {
      if (db_subtype == ARTS_DB_LOCAL) {
        ARTS_DEBUG("Pinned DB write release (no cdag_lock update)");
      } else if (owner == arts_global_rank_id) {
        struct arts_db_s *db = ((struct arts_db_s *)depv[i].ptr - 1);
        if (db->db_list) {
          cdag_lock_release((struct cdag_lock_s *)db->db_list,
                            arts_cdag_dispatch_cb, db);
        }
      } else {
        arts_remote_update_db(depv[i].guid, true);
        INCREMENT_NUM_OWNER_UPDATE_PERFORMED_BY(1);
      }
    } else if (depv[i].guid != NULL_GUID && access_mode == DB_MODE_RO) {
      ARTS_DEBUG("DB[Guid:%lu] released in READ mode", depv[i].guid);
      INCREMENT_NUM_OWNER_UPDATE_SAVED_BY(1);
      /* Every local RO reader held a slot in the cdag_lock. Release it.
       * If we're the last reader in the current RO generation, the lock
       * advances to the next generation and dispatches its waiters. */
      if (db_subtype == ARTS_DB_DEFAULT && owner == arts_global_rank_id &&
          depv[i].ptr) {
        struct arts_db_s *db = ((struct arts_db_s *)depv[i].ptr) - 1;
        if (db->db_list) {
          cdag_lock_release((struct cdag_lock_s *)db->db_list,
                            arts_cdag_dispatch_cb, db);
        }
      }
    } else if (access_mode == DB_MODE_PTR) {
      if (depv[i].ptr) {
        arts_free(depv[i].ptr);
      }
    } else if (!gpu && db_subtype == ARTS_DB_LC) {
      if (depv[i].ptr) {
        struct arts_db_s *db = ((struct arts_db_s *)depv[i].ptr) - 1;
        arts_reader_unlock(&db->reader);
      }
    }

    /* Return the route table ref acquired during DB resolution.
     * PTR mode uses a malloc'd copy (no route table ref), VALUE mode stores
     * a raw uint64 (not a real GUID), NULL_GUID has no entry. */
    if (depv[i].guid != NULL_GUID && access_mode != DB_MODE_PTR &&
        access_mode != DB_MODE_VALUE && depv[i].ptr) {
      arts_route_table_return_db(depv[i].guid, false);
    }
  }
}

/*
 * arts_db_release — Release access to a single DB mid-EDT.
 *
 * Two search paths:
 *   1. created_db_list — DBs the current EDT created (auto-acquired WRITE).
 *   2. depv — DBs received as dependencies (EW or RO mode).
 *
 * For WRITE-mode (EW) deps: progresses the CDAG frontier.  For READ-mode
 * (RO) deps: no frontier action needed (readers don't hold locks).
 *
 * Marks the released slot/entry as NULL_GUID to prevent double-release
 * in the epilogue (release_dbs / arts_release_created_dbs).
 */
void arts_db_release(arts_guid_t guid) {
  /* Path 1: check created_db_list (DBs this EDT created) */
  arts_array_list_t *list = arts_get_created_db_list();
  if (list) {
    uint64_t count = arts_length_array_list(list);
    for (uint64_t i = count; i > 0; i--) {
      arts_guid_t *g = (arts_guid_t *)arts_get_from_array_list(list, i - 1);
      if (*g == guid) {
        *g = NULL_GUID;
        struct arts_db_s *db =
            (struct arts_db_s *)arts_route_table_lookup_db(guid, NULL, false);
        if (db) {
          if (db->db_list) {
            cdag_lock_release((struct cdag_lock_s *)db->db_list,
                              arts_cdag_dispatch_cb, db);
          }
          arts_route_table_return_db(guid, false);
        }
        return;
      }
    }
  }

  /* Path 2: check current EDT's depv (dependency-acquired DBs) */
  if (!current_edt) {
    return;
  }
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(current_edt);
  for (int i = 0; i < current_edt->depc; i++) {
    if (depv[i].guid != guid) {
      continue;
    }
    arts_db_access_mode_t mode = depv[i].mode;
    if (mode == DB_MODE_EW || mode == DB_MODE_MEMSET) {
      struct arts_db_s *db = ((struct arts_db_s *)depv[i].ptr) - 1;
      if (db) {
        arts_db_types_t subtype = db->db_type;
        if (subtype == ARTS_DB_LOCAL) {
          /* LOCAL: no cdag_lock, ordering is programmer's responsibility. */
        } else if (arts_guid_get_rank(guid) == arts_global_rank_id) {
          if (db->db_list) {
            cdag_lock_release((struct cdag_lock_s *)db->db_list,
                              arts_cdag_dispatch_cb, db);
          }
        } else {
          arts_remote_update_db(guid, true);
        }
      }
    }
    /* RO mode: no lock-level action needed here; release_dbs will
     * perform the actual cdag_lock_release on the owner path. */
    arts_route_table_return_db(guid, false);
    depv[i].guid = NULL_GUID;
    depv[i].ptr = NULL;
    depv[i].mode = DB_MODE_NULL;
    return;
  }
}

/*
 * arts_release_created_dbs — Release auto-acquired WRITE access for all DBs
 * created by the current EDT.
 *
 * Mirrors release_dbs() but operates on the thread-local created_db_list
 * instead of the EDT's depv[].  For each tracked DB: progresses the frontier
 * (unblocking consumers).
 * Skips entries already released via arts_db_release() (marked NULL_GUID).
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
    struct arts_db_s *db =
        (struct arts_db_s *)arts_route_table_lookup_db(*guid, NULL, false);
    if (db) {
      if (db->db_list) {
        cdag_lock_release((struct cdag_lock_s *)db->db_list,
                          arts_cdag_dispatch_cb, db);
      }
      arts_route_table_return_db(*guid, false);
    }
  }
}

bool arts_add_db_duplicate(struct arts_db_s *db, unsigned int rank,
                           struct arts_edt_s *edt, arts_guid_t edt_guid,
                           unsigned int slot, arts_db_access_mode_t mode,
                           bool *on_head) {
  if (edt && edt_guid == NULL_GUID) {
    edt_guid = edt->current_edt;
  }
  if (!db->db_list) {
    if (on_head) {
      *on_head = false;
    }
    return false;
  }
  struct cdag_lock_request_s req = {0};
  req.edt = edt;
  req.edt_guid = edt_guid;
  req.origin_rank = rank;
  req.slot = slot;
  req.mode = mode;
  enum cdag_submit_result res =
      cdag_lock_submit((struct cdag_lock_s *)db->db_list, &req);
  if (on_head) {
    *on_head = (res == CDAG_SUBMIT_HEAD_IMMEDIATE);
  }
  /* Historically this returned "true if this was the first copy for this
   * rank in the current frontier", to avoid duplicate remote sends.
   * cdag_lock does not deduplicate at the lock layer — every submit
   * succeeds. Return true unconditionally; caller should dedup at a
   * higher level if needed. */
  return true;
}

void internal_get_from_db(arts_guid_t edt_guid, arts_guid_t db_guid,
                          unsigned int slot, unsigned int offset,
                          unsigned int size, unsigned int rank) {
  if (rank == arts_global_rank_id) {
    struct arts_db_s *db =
        (struct arts_db_s *)arts_route_table_lookup_db(db_guid, NULL, false);
    if (db) {
      void *data = (void *)(((char *)(db + 1)) + offset);
      ARTS_INFO("Getting DB[Guid:%lu] From: %p", db_guid, data);
      if (edt_guid != NULL_GUID) {
        arts_signal_edt_ptr(edt_guid, slot, data, size);
      }
      arts_route_table_return_db(db_guid, false);
    } else {
      assert(edt_guid != NULL_GUID && "DB not found and no EDT to signal");
      ARTS_INFO("Getting OO-DB[Guid:%lu] From: %p", db_guid, NULL);
      arts_out_of_order_get_from_db(edt_guid, db_guid, slot, offset, size);
    }
  } else {
    ARTS_DEBUG("Sending DB[Guid:%lu] to Rank %u", db_guid, rank);
    assert(edt_guid != NULL_GUID && "DB not found and no EDT to signal");
    arts_remote_get_from_db(edt_guid, db_guid, slot, offset, size, rank);
  }
}

void arts_get_from_db(arts_guid_t edt_guid, arts_guid_t db_guid,
                      unsigned int slot, unsigned int offset,
                      unsigned int len) {
  TIME_DB_GET_START();
  INCREMENT_NUM_DB_GET_BY(1);
  unsigned int rank = arts_guid_get_rank(db_guid);
  internal_get_from_db(edt_guid, db_guid, slot, offset, len, rank);
  TIME_DB_GET_STOP();
}

void arts_get_from_db_at(arts_guid_t edt_guid, arts_guid_t db_guid,
                         unsigned int slot, unsigned int offset,
                         unsigned int len, unsigned int rank) {
  TIME_DB_GET_START();
  INCREMENT_NUM_DB_GET_BY(1);
  internal_get_from_db(edt_guid, db_guid, slot, offset, len, rank);
  TIME_DB_GET_STOP();
}

void internal_put_in_db(void *ptr, arts_guid_t edt_guid, arts_guid_t db_guid,
                        unsigned int slot, unsigned int offset,
                        unsigned int size, arts_guid_t epoch_guid,
                        unsigned int rank) {
  if (rank == arts_global_rank_id) {
    struct arts_db_s *db =
        (struct arts_db_s *)arts_route_table_lookup_db(db_guid, NULL, false);
    if (db) {
      // Do this so when we increment finished we can check the term status
      increment_queue_epoch(epoch_guid);
      arts_shutdown_epoch_inc_queue();
      void *data = (void *)(((char *)(db + 1)) + offset);
      memcpy(data, ptr, size);
      if (edt_guid != NULL_GUID) {
        arts_signal_edt(edt_guid, slot, db_guid, DB_MODE_EW);
      }
      increment_finished_epoch(epoch_guid);
      arts_shutdown_epoch_inc_finished();
      arts_route_table_return_db(db_guid, false);
    } else {
      void *cpy_ptr = arts_malloc(size);
      memcpy(cpy_ptr, ptr, size);
      arts_out_of_order_put_in_db(cpy_ptr, edt_guid, db_guid, slot, offset,
                                  size, epoch_guid);
    }
  } else {
    void *cpy_ptr = arts_malloc(size);
    memcpy(cpy_ptr, ptr, size);
    arts_remote_put_in_db(cpy_ptr, edt_guid, db_guid, slot, offset, size,
                          epoch_guid, rank);
  }
}

void arts_put_in_db_at(void *ptr, arts_guid_t edt_guid, arts_guid_t db_guid,
                       unsigned int slot, unsigned int offset, unsigned int len,
                       unsigned int rank) {
  TIME_DB_PUT_START();
  INCREMENT_NUM_DB_PUT_BY(1);
  INCREMENT_BYTES_DB_PUT_BY(len);
  arts_guid_t epoch_guid = arts_get_current_epoch_guid();
  ARTS_DEBUG("Epoch [Guid:%lu]", epoch_guid);
  increment_active_epoch(epoch_guid);
  arts_shutdown_epoch_inc_active();
  internal_put_in_db(ptr, edt_guid, db_guid, slot, offset, len, epoch_guid,
                     rank);
  TIME_DB_PUT_STOP();
}

void arts_put_in_db(void *ptr, arts_guid_t edt_guid, arts_guid_t db_guid,
                    unsigned int slot, unsigned int offset, unsigned int len) {
  TIME_DB_PUT_START();
  INCREMENT_NUM_DB_PUT_BY(1);
  INCREMENT_BYTES_DB_PUT_BY(len);
  unsigned int rank = arts_guid_get_rank(db_guid);
  arts_guid_t epoch_guid = arts_get_current_epoch_guid();
  ARTS_DEBUG("Epoch [Guid:%lu]", epoch_guid);
  increment_active_epoch(epoch_guid);
  arts_shutdown_epoch_inc_active();
  internal_put_in_db(ptr, edt_guid, db_guid, slot, offset, len, epoch_guid,
                     rank);
  TIME_DB_PUT_STOP();
}

void arts_put_in_db_epoch(void *ptr, arts_guid_t epoch_guid,
                          arts_guid_t db_guid, unsigned int offset,
                          unsigned int len) {
  TIME_DB_PUT_START();
  INCREMENT_NUM_DB_PUT_BY(1);
  INCREMENT_BYTES_DB_PUT_BY(len);
  unsigned int rank = arts_guid_get_rank(db_guid);
  increment_active_epoch(epoch_guid);
  arts_shutdown_epoch_inc_active();
  internal_put_in_db(ptr, NULL_GUID, db_guid, 0, offset, len, epoch_guid, rank);
  TIME_DB_PUT_STOP();
}

/*
 * arts_wait_release_dbs -- Temporarily release frontier locks for all DBs
 * held by the current EDT, allowing consumer EDTs to proceed while this
 * EDT blocks on arts_wait_on_handle.
 *
 * Handles both:
 *   1. created_db_list (auto-acquired WRITE from arts_db_create)
 *   2. depv (dependency-acquired EW/MEMSET)
 *
 * Only touches frontier locks (arts_progress_frontier). Does NOT return
 * route table entries or null any tracking state -- this is a temporary
 * release, not a final epilogue release.
 */
void arts_wait_release_dbs(void) {
  /* Path 1: created DBs */
  arts_array_list_t *list = arts_get_created_db_list();
  if (list) {
    uint64_t count = arts_length_array_list(list);
    for (uint64_t i = 0; i < count; i++) {
      arts_guid_t *guid = (arts_guid_t *)arts_get_from_array_list(list, i);
      if (*guid == NULL_GUID) {
        continue;
      }
      struct arts_db_s *db =
          (struct arts_db_s *)arts_route_table_lookup_db(*guid, NULL, false);
      if (db && db->db_list) {
        cdag_lock_release((struct cdag_lock_s *)db->db_list,
                          arts_cdag_dispatch_cb, db);
      }
    }
  }

  /* Path 2: depv DBs */
  if (current_edt) {
    arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(current_edt);
    for (int i = 0; i < current_edt->depc; i++) {
      if (depv[i].guid == NULL_GUID) {
        continue;
      }
      if (depv[i].mode != DB_MODE_EW && depv[i].mode != DB_MODE_MEMSET) {
        continue;
      }
      if (!depv[i].ptr) {
        continue;
      }
      struct arts_db_s *db = ((struct arts_db_s *)depv[i].ptr) - 1;
      if (db && db->db_list) {
        cdag_lock_release((struct cdag_lock_s *)db->db_list,
                          arts_cdag_dispatch_cb, db);
      }
    }
  }
}

/*
 * arts_wait_reacquire_dbs -- Re-acquire cdag_locks for all DBs held by
 * the current EDT after arts_wait_on_handle completes. For each held
 * DB, submit a new creator-owner EW request — because this EDT is still
 * the conceptual owner and will be doing more work with the DB after
 * the wait returns.
 */
void arts_wait_reacquire_dbs(void) {
  /* Path 1: created DBs */
  arts_array_list_t *list = arts_get_created_db_list();
  if (list) {
    uint64_t count = arts_length_array_list(list);
    for (uint64_t i = 0; i < count; i++) {
      arts_guid_t *guid = (arts_guid_t *)arts_get_from_array_list(list, i);
      if (*guid == NULL_GUID) {
        continue;
      }
      struct arts_db_s *db =
          (struct arts_db_s *)arts_route_table_lookup_db(*guid, NULL, false);
      if (db && db->db_list) {
        struct cdag_lock_request_s req = {0};
        req.edt = current_edt;
        req.edt_guid = current_edt ? current_edt->current_edt : NULL_GUID;
        req.origin_rank = arts_global_rank_id;
        req.slot = 0;
        req.mode = DB_MODE_EW;
        (void)cdag_lock_submit((struct cdag_lock_s *)db->db_list, &req);
      }
    }
  }

  /* Path 2: depv DBs */
  if (current_edt) {
    arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(current_edt);
    for (int i = 0; i < current_edt->depc; i++) {
      if (depv[i].guid == NULL_GUID) {
        continue;
      }
      if (depv[i].mode != DB_MODE_EW && depv[i].mode != DB_MODE_MEMSET) {
        continue;
      }
      if (!depv[i].ptr) {
        continue;
      }
      struct arts_db_s *db = ((struct arts_db_s *)depv[i].ptr) - 1;
      if (db && db->db_list) {
        struct cdag_lock_request_s req = {0};
        req.edt = current_edt;
        req.edt_guid = current_edt->current_edt;
        req.origin_rank = arts_global_rank_id;
        req.slot = (unsigned int)i;
        req.mode = depv[i].mode;
        (void)cdag_lock_submit((struct cdag_lock_s *)db->db_list, &req);
      }
    }
  }
}

/* ── CXL cache-flush helpers ────────────────────────────────────────────────
 */

#ifdef ARTS_USE_CXL
void arts_cxl_producer_flush(arts_guid_t guid) {
  struct arts_db_s *db = (struct arts_db_s *)arts_cxl_get_ptr(guid);
  FLUSH_FENCE_PRODUCER(db, ALIGN_UP(db->header.size, CACHELINE_SIZE));
}

void arts_cxl_consumer_flush(arts_guid_t guid) {
  struct arts_db_s *db = (struct arts_db_s *)arts_cxl_get_ptr(guid);
  /* First flush the header to read the actual size. */
  FLUSH_FENCE_CONSUMER(db, ALIGN_UP(sizeof(struct arts_db_s), CACHELINE_SIZE));
  /* Then flush the full DB (header + payload). */
  if (db->header.size > sizeof(struct arts_db_s)) {
    FLUSH_FENCE_CONSUMER(db, ALIGN_UP(db->header.size, CACHELINE_SIZE));
  }
}
#endif /* ARTS_USE_CXL */
