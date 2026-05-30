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
#include "arts/compute/edt.h"
#include "arts/memory/db.h"
#include "arts/utils/malloc.h"

#include <string.h>

#include "arts/gas/guid.h"
#include "arts/gas/out_of_order.h"
#include "arts/gas/route_table.h"
#include "arts/remote/handler.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/sync/epoch.h"
#include "arts/sync/shared.h" /* arts_shared_ptr_t, get/release */
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/array_list.h"
#include "arts/utils/atomics.h"

#ifdef ARTS_USE_GPU
#include "arts/gpu/gpu_internal.h"
#endif

#include "arts/cxl/wrapper.h"
#ifdef ARTS_USE_CXL
#include "arts/cxl/deque.h"
#endif

#define MAX_EPOCH_ARRAY_LIST 32

extern unsigned int num_numa_domains;

ARTS_THREAD_LOCAL arts_array_list_t *epoch_list = NULL;
ARTS_THREAD_LOCAL struct arts_edt_s *current_edt = NULL;
ARTS_THREAD_LOCAL arts_array_list_t *created_db_list = NULL;

bool arts_set_current_epoch_guid(arts_guid_t epoch_guid) {
  if (epoch_guid) {
    if (!epoch_list) {
      epoch_list = arts_new_array_list(sizeof(arts_guid_t), 8);
    }
    arts_push_to_array_list(epoch_list, &epoch_guid);
    if (current_edt) {
      current_edt->epoch_guid = epoch_guid;
      return true;
    }
  }
  return false;
}

arts_guid_t arts_epoch_get_current_guid() {
  if (epoch_list) {
    uint64_t length = arts_length_array_list(epoch_list);
    if (length) {
      arts_guid_t *guid =
          (arts_guid_t *)arts_get_from_array_list(epoch_list, length - 1);
      return *guid;
    }
  }
  return NULL_GUID;
}

arts_guid_t *arts_check_epoch_is_root(arts_guid_t to_check) {
  if (epoch_list) {
    uint64_t length = arts_length_array_list(epoch_list);
    for (uint64_t i = 0; i < length; i++) {
      arts_guid_t *guid =
          (arts_guid_t *)arts_get_from_array_list(epoch_list, i);
      if (*guid == to_check) {
        return guid;
      }
    }
  }
  ARTS_INFO("ERROR %lu is not a valid epoch", to_check);
  return NULL;
}

void arts_track_created_db(arts_guid_t guid) {
  if (!created_db_list) {
    created_db_list = arts_new_array_list(sizeof(arts_guid_t), 65536);
  }
  arts_push_to_array_list(created_db_list, &guid);
}

arts_array_list_t *arts_get_created_db_list(void) { return created_db_list; }

void arts_set_thread_local_edt_info(struct arts_edt_s *edt) {
  arts_thread_info.current_edt_guid = edt->guid;
  current_edt = edt;

  if (epoch_list) {
    arts_reset_array_list(epoch_list);
  }

  if (created_db_list) {
    arts_reset_array_list(created_db_list);
  }

  arts_set_current_epoch_guid(current_edt->epoch_guid);
}

void arts_save_thread_local(thread_local_t *tl) {
  TIME_CONTEXT_SWITCH_START();
  tl->current_edt_guid = arts_thread_info.current_edt_guid;
  tl->current_edt = current_edt;
  tl->epoch_list = (void *)epoch_list;
  tl->created_db_list = (void *)created_db_list;

  arts_thread_info.current_edt_guid = NULL_GUID;
  current_edt = NULL;
  epoch_list = NULL;
  created_db_list = NULL;
  TIME_CONTEXT_SWITCH_STOP();
}

void arts_restore_thread_local(thread_local_t *tl) {
  TIME_CONTEXT_SWITCH_START();
  arts_thread_info.current_edt_guid = tl->current_edt_guid;
  current_edt = tl->current_edt;
  if (epoch_list) {
    arts_delete_array_list(epoch_list);
  }
  epoch_list = (arts_array_list_t *)tl->epoch_list;
  if (created_db_list) {
    arts_delete_array_list(created_db_list);
  }
  created_db_list = (arts_array_list_t *)tl->created_db_list;
  TIME_CONTEXT_SWITCH_STOP();
}

void arts_cleanup_edt_tls() {
  if (epoch_list) {
    arts_delete_array_list(epoch_list);
    epoch_list = NULL;
  }
  if (created_db_list) {
    arts_delete_array_list(created_db_list);
    created_db_list = NULL;
  }
}

void arts_increment_finished_epoch_list() {
  if (epoch_list) {

    unsigned int epoch_array_length = arts_length_array_list(epoch_list);
    for (unsigned int i = 0; i < epoch_array_length; i++) {
      arts_guid_t *guid =
          (arts_guid_t *)arts_get_from_array_list(epoch_list, i);
#if ARTS_LOG_LEVEL >= 2
      uint64_t current_id = current_edt ? current_edt->arts_id : 0;
      ARTS_INFO("Current EDT[Id:%lu, Guid:%lu] - Unsetting Epoch [Guid:%lu]",
                current_id, arts_thread_info.current_edt_guid, *guid);
#endif
      if (*guid) {
        increment_finished_epoch(*guid);
      }
    }

    if (epoch_array_length > MAX_EPOCH_ARRAY_LIST) {
      arts_delete_array_list(epoch_list);
      epoch_list = NULL;
    } else {
      arts_reset_array_list(epoch_list);
    }
  }
  arts_shutdown_epoch_inc_finished();
}

void arts_unset_thread_local_edt_info() {
  arts_increment_finished_epoch_list();
  /* finish_event tracking: emit DECR on completion of the current EDT.
   * - Inherited finish_event: balances the INCR emitted at create time.
   * - Own (ARTS_EDT_FLAG_FINISH or cross-node proxy) finish_event:
   *   counter==0 fires the latch, propagating DECR to the parent's
   *   finish_event via the dep registered at allocation.
   * `current_edt` is still valid here (cleared on the next line). */
  if (current_edt && current_edt->finish_event != NULL_GUID) {
    arts_event_satisfy_slot(current_edt->finish_event, NULL_GUID,
                            ARTS_EVENT_LATCH_DECR_SLOT);
  }
  arts_thread_info.current_edt_guid = NULL_GUID;
  current_edt = NULL;
}

/*
 * arts_edt_deleter — shared_t deleter.
 *
 * Invoked by route_table free_item once the slot's lock count hits 0 with
 * DELETE set.  Mirrors the DB pattern (the DB deleter ->
 * arts_db_free): delegates to arts_edt_free, which is the canonical struct-free
 * path.
 *
 * Static-file-scope; remote handler.c reaches the same pointer via
 * arts_edt_get_deleter() so there's a single source of truth.
 */
/* cb deleter (route_table deleter-by-kind for ARTS_GUID_EDT).  External
 * linkage so route_table.c can reference it directly. */
void arts_edt_deleter(void *self) { arts_edt_free((struct arts_edt_s *)self); }

/* Publish the EDT cb deleter into the route_table's per-kind table at startup
 * (decoupled registration — see arts_route_table_register_deleter). */
__attribute__((constructor)) static void arts_edt_register_cb_deleter(void) {
  arts_route_table_register_deleter(ARTS_GUID_EDT, arts_edt_deleter);
}

/* Getter for foreign TUs (e.g. remote handler.c) that allocate arts_edt_s
 * stubs and need to install the same deleter pointer. */
void (*arts_edt_get_deleter(void))(void *) { return arts_edt_deleter; }

/*
 * arts_edt_create_internal — Core EDT allocation and registration.
 *
 * Allocates the EDT struct (header + paramv + depv + modes), assigns its GUID,
 * copies parameters, registers with the epoch system, and places the EDT into
 * the route table so that incoming signals can find it.
 *
 * Two paths exist depending on whether a GUID was pre-reserved:
 *   1. New GUID (created_guid == true):
 *        - arts_route_table_add_item (no race — nobody else knows the GUID
 * yet).
 *        - If depc == 0, the EDT is immediately ready.
 *   2. Pre-reserved GUID (created_guid == false):
 *        - The GUID may already have received out-of-order signals while it was
 *          in RESERVED state. A sentinel (+1 on depc_needed) prevents premature
 *          firing during the race window between route-table insertion and
 *          OOO replay.  See the inline comments for the full protocol.
 *
 * Concurrency notes:
 *   - depc_needed is the primary synchronisation counter.  Every satisfied
 *     dependency atomically decrements it; exactly one thread observes 0
 *     and calls arts_handle_ready_edt.
 *   - The EDT must NOT be visible (in the route table) while its fields
 *     are still being written.
 */
bool arts_edt_create_internal(struct arts_edt_s *edt, arts_guid_kind_t mode,
                              arts_guid_t *guid, unsigned int rank,
                              unsigned int numa_domain, unsigned int edt_space,
                              arts_edt_t func_ptr, uint32_t paramc,
                              const uint64_t *paramv, uint32_t depc,
                              bool use_epoch, arts_guid_t epoch_guid,
                              uint64_t arts_id, uint32_t flags) {
  if (!edt) {
    edt = (struct arts_edt_s *)arts_calloc_align(1, edt_space, 16);
  }
  if (!edt) {
    ARTS_ERROR("EDT allocation failed (size=%u)", edt_space);
  }

  /* lifecycle/deleter handled by the route_table cb (deleter-by-kind) on
   * install — no per-object shared field to initialize.  Kind comes from the
   * GUID (bits 63-62); total size from arts_edt_total_size(paramc/depc) — no
   * per-object header stores them. */
  (void)edt_space;
  edt->arts_id = arts_id;

  bool created_guid = false;
  if (*guid == NULL_GUID) {
    created_guid = true;
    edt->guid = *guid = arts_guid_create_for_rank(rank, mode);
  } else {
    edt->guid = *guid;
  }

  edt->func_ptr = func_ptr;
  edt->depc = depc;
  edt->paramc = paramc;
  edt->epoch_guid = NULL_GUID;
  edt->depc_needed = depc;

  /* Determine finish-scope for this EDT.
   *
   * `current_edt` is the file-static thread-local pointer maintained by
   * arts_set/unset_thread_local_edt_info — direct access, no route_table
   * lookup needed (same TU).
   *
   * Two cases:
   *   ARTS_EDT_FLAG_FINISH set: allocate a fresh LATCH event as this EDT's
   *     own finish-scope and chain it into the caller's finish-scope (if any).
   *     counter_init=1 is the self-alive token; it is released by the
   *     DECR emitted in arts_unset_thread_local_edt_info on completion.
   *     Spawned children each INCR this event
   *     via the plain-inheritance path below, and DECR it on completion.
   *     When the counter reaches 0, the LATCH fires and propagates DECR to
   *     the parent finish-scope.
   *
   *   Otherwise: plain inheritance — adopt the caller's finish_event and
   *     INCR it to register as a descendant. */
  edt->finish_event = NULL_GUID;
  arts_guid_t parent_fe = current_edt ? current_edt->finish_event : NULL_GUID;

  bool need_new_finish_event = (flags & ARTS_EDT_FLAG_FINISH) != 0;

  if (need_new_finish_event) {
    arts_event_hint_t latch_hint = ARTS_EVENT_HINT_LATCH(1);
    arts_guid_t new_fe = arts_event_create(&latch_hint);

    if (parent_fe != NULL_GUID) {
      /* Chain: when new_fe fires it satisfies one DECR slot of parent_fe.
       * INCR parent_fe first to register this finish-scope as a descendant.
       * Both ops are local-sync (parent runs on this node). */
      arts_event_satisfy_slot(parent_fe, NULL_GUID, ARTS_EVENT_LATCH_INCR_SLOT);
      arts_add_dependence(new_fe, parent_fe, ARTS_EVENT_LATCH_DECR_SLOT,
                          DB_MODE_NULL);
    }
    edt->finish_event = new_fe;
  } else if (parent_fe != NULL_GUID) {
    /* Plain inheritance — adopt enclosing finish-scope and INCR it.
     * INCR completes before the new EDT can reach its own DECR (which runs
     * only after the EDT executes — strictly later in this thread). */
    edt->finish_event = parent_fe;
    arts_event_satisfy_slot(parent_fe, NULL_GUID, ARTS_EVENT_LATCH_INCR_SLOT);
  }

  if (use_epoch) {
    arts_guid_t current_epoch_guid = NULL_GUID;
    if (epoch_guid && arts_check_epoch_is_root(epoch_guid)) {
      current_epoch_guid = epoch_guid;
    } else {
      current_epoch_guid = arts_epoch_get_current_guid();
    }

    if (current_epoch_guid) {
      edt->epoch_guid = current_epoch_guid;
      increment_active_epoch(current_epoch_guid);
    }
  }
  arts_shutdown_epoch_inc_active();

  /* Copy inline parameter values into the EDT's trailing storage.
   * Layout: [<edt header> | paramv[paramc] | depv[depc]].
   *
   * The header size depends on the EDT subtype: a GPU EDT embeds extra
   * scheduling metadata (grid/block/...) between the base header and the
   * trailing paramv region.  The paramv base must therefore use the SAME
   * subtype-aware offset that arts_get_depv uses to locate depv, otherwise
   * the copy lands on top of that metadata (corrupting grid/block) and the
   * runtime reads params from the wrong place. */
  if (paramc) {
    unsigned int offset = sizeof(struct arts_edt_s);
#ifdef ARTS_USE_GPU
    if (edt->edt_type == ARTS_EDT_GPU) {
      offset = sizeof(arts_gpu_edt_t);
    }
#endif
    ARTS_DEBUG("EDT paramv copy: edt=%p offset=%u paramc=%u depc=%u "
               "edt_space=%u dep_size=%zu",
               (void *)edt, offset, paramc, depc, edt_space,
               depc * sizeof(arts_edt_dep_t));
    char *tmp = (char *)edt + offset;
    memcpy(tmp, paramv, sizeof(uint64_t) * paramc);
  }

  ARTS_INFO("EDT create [Guid:%lu, Id:%lu, Depc:%u, Route:%u, "
            "PreReserved:%s, Epoch:%lu, FuncPtr:%p]",
            *guid, edt->arts_id, edt->depc, rank, created_guid ? "no" : "yes",
            edt->epoch_guid, (void *)func_ptr);

  if (rank != arts_global_rank_id) {
    /* Remote EDT: serialise and send to the target node. */
    ARTS_INFO("EDT[Guid:%lu] remote move to rank %u", *guid, rank);
    arts_send_memory_move(rank, *guid, (void *)edt,
                          (unsigned int)arts_edt_total_size(edt),
                          MSG_EDT_CREATE, arts_free);
  } else {
    /* Local EDT: register in the route table and check readiness. */
    INC_OUTSTANDING_EDTS(1);
    if (created_guid) {
      /* New GUID path — no race, safe non-atomic insert. */
      arts_route_table_add_item(edt, *guid, arts_global_rank_id, false);
      if (edt->depc_needed == 0) {
        ARTS_INFO("EDT[Guid:%lu] immediately ready (depc=0)", *guid);
        arts_handle_ready_edt(edt);
      } else {
        ARTS_DEBUG("EDT[Guid:%lu] waiting for %u deps", *guid,
                   edt->depc_needed);
      }
    } else {
      /*
       * Pre-reserved GUID path — other threads may already hold this GUID
       * and could have queued out-of-order (OOO) signals.
       *
       * Protocol:
       *   1. Set depc_needed = depc + 1  (sentinel prevents premature 0)
       *   2. Insert into route table (EDT is now globally visible)
       *   3. Replay any queued OOO signals (they decrement depc_needed)
       *   4. Atomically remove sentinel (-1); if result is 0, all deps
       *      were already satisfied and we fire the EDT.
       *
       * Exactly one thread (either the OOO replay callback or us at step 4)
       * will observe depc_needed == 0 and call arts_handle_ready_edt.
       */
      edt->depc_needed = depc + 1;
      ARTS_INFO("EDT[Guid:%lu] pre-reserved path: sentinel depc_needed=%u",
                *guid, edt->depc_needed);
      /* add_item_race installs the cb and fires the OoO list internally
       * (replaying queued signals) — the sentinel set above guarantees those
       * replays cannot drive depc_needed to 0 before we remove it below. */
      arts_route_table_add_item_race(edt, *guid, arts_global_rank_id, false);
      unsigned int remaining = arts_atomic_sub(&edt->depc_needed, 1U);
      ARTS_INFO("EDT[Guid:%lu] sentinel removed: depc_needed=%u", *guid,
                remaining);
      if (remaining == 0) {
        arts_handle_ready_edt(edt);
      }
    }
  }

  INCREMENT_NUM_EDT_CREATE_BY(1);
  return true;
}

arts_guid_t arts_edt_create(arts_edt_t func_ptr, uint32_t paramc,
                            const uint64_t *paramv, uint32_t depc,
                            const arts_edt_hint_t *hint) {
  TIME_EDT_CREATE_START();

  /* Snapshot hint (NULL = ARTS_EDT_HINT_DEFAULTS).  After this all four
   * optional fields are well-defined and follow the documented precedence:
   *   - if .guid != NULL_GUID, the GUID's rank field overrides .rank
   *   - if .epoch == NULL_GUID, the runtime inherits the caller's current
   *     epoch (handled inside arts_edt_create_internal). */
  arts_edt_hint_t snap = hint
                             ? *hint
                             : (arts_edt_hint_t){.rank = ARTS_HINT_CURRENT_RANK,
                                                 .edt_id = 0,
                                                 .guid = NULL_GUID,
                                                 .epoch = NULL_GUID};

  arts_guid_t guid = snap.guid;
  unsigned int rank;
  if (guid != NULL_GUID) {
    rank = arts_guid_get_rank(guid);
  } else if (snap.rank != ARTS_HINT_CURRENT_RANK) {
    rank = snap.rank;
  } else {
    rank = arts_global_rank_id;
  }

  unsigned int edt_space = sizeof(struct arts_edt_s) +
                           (paramc * sizeof(uint64_t)) +
                           (depc * sizeof(arts_edt_dep_t));
  bool ok = arts_edt_create_internal(NULL, ARTS_GUID_EDT, &guid, rank,
                                     arts_thread_info.numa_domain_id, edt_space,
                                     func_ptr, paramc, paramv, depc, true,
                                     snap.epoch, snap.edt_id, snap.flags);
  TIME_EDT_CREATE_STOP();
  return ok ? guid : NULL_GUID;
}

void arts_edt_free(struct arts_edt_s *edt) {
  arts_thread_info.edt_free = 1;
  arts_free(edt);
  arts_thread_info.edt_free = 0;
}

void arts_edt_delete(struct arts_edt_s *edt) {
  if (!edt) {
    ARTS_INFO("EDT delete called with NULL edt on rank %u",
              arts_global_rank_id);
    return;
  }
  ARTS_INFO("EDT delete [Guid:%lu, Id:%lu, Depc:%u, DepcNeeded:%u] on rank %u",
            edt->guid, edt->arts_id, edt->depc, edt->depc_needed,
            arts_global_rank_id);
  /* route through arts_route_table_mark_delete so the deleter
   * (arts_edt_deleter -> arts_edt_free) runs once outstanding refs are
   * returned.  Capturing the GUID up front, then calling mark_delete:
   * if no other thread holds an acquire ref, this drops the install ref
   * and free_item invokes the deleter inline.  If another thread holds
   * a transient lookup_edt_safe ref, free_item is deferred to the last
   * release_item.  Either way `edt` is no longer safe to dereference
   * after this call returns. */
  arts_guid_t guid = edt->guid;
  arts_route_table_mark_delete(guid);
}

void arts_edt_destroy(arts_guid_t guid) {
  /* read needed fields under a paired safe lookup, then route
   * the destruction through arts_route_table_mark_delete.  The deleter
   * (arts_edt_deleter) handles the actual struct free once the install
   * ref + any outstanding acquire refs are returned. */
  arts_shared_ptr_t h = arts_route_table_lookup_edt(guid);
  struct arts_edt_s *edt = (struct arts_edt_s *)arts_shared_get(h);
  if (!edt) {
    ARTS_INFO("EDT destroy missing [Guid:%lu] on rank %u", guid,
              arts_global_rank_id);
    return;
  }
  ARTS_INFO("EDT destroy [Guid:%lu, Id:%lu, Depc:%u, DepcNeeded:%u] on rank %u",
            edt->guid, edt->arts_id, edt->depc, edt->depc_needed,
            arts_global_rank_id);
  /* OCR spec restricts ocrEdtDestroy to pre-runnable EDTs (depc_needed > 0
   * means dependences are not yet all met).  Calling on a runnable/queued/
   * running EDT is undefined behavior; we additionally would corrupt epoch
   * accounting (queued/finished_count would double-count).  Skip free in
   * that case — the EDT is owned by a worker now. */
  if (edt->depc_needed == 0) {
    ARTS_INFO("EDT destroy on runnable/queued EDT [Guid:%lu] — UB; ignoring",
              guid);
    arts_shared_release(&h);
    return;
  }
  /* Mirror the epoch counter balancing that the normal finish path does:
   * the EDT was registered with active_count++ at creation time, but will
   * never run, so we increment finished_count to keep the epoch fire
   * condition (finished_count == active_count) reachable.  Pre-runnable
   * EDTs were never queued, so the queued counter does not need touching. */
  arts_guid_t epoch_guid = edt->epoch_guid;
  arts_shared_release(&h);
  arts_route_table_mark_delete(guid);
  if (epoch_guid != NULL_GUID) {
    increment_finished_epoch(epoch_guid);
  }
}

void *arts_get_depv(void *edt_ptr) {
  struct arts_edt_s *edt = (struct arts_edt_s *)edt_ptr;
  unsigned int paramc = edt->paramc;
  if (edt->edt_type == ARTS_EDT_GPU) {
#ifdef ARTS_USE_GPU
    arts_gpu_edt_t *edtGpu = (arts_gpu_edt_t *)edt_ptr;
    return (void *)((uint64_t *)(edtGpu + 1) + paramc);
#else
    return NULL;
#endif
  }
  return (void *)((uint64_t *)(edt + 1) + paramc);
}

/* arts_get_dep_modes removed — mode now lives in arts_edt_dep_t.mode */

/*
 * arts_edt_satisfy_slot — Satisfy one dependency slot on an EDT.
 *
 * Four dispatch paths:
 *   1. CDAG invalidation (current EDT has pending invalidations) → force-defer
 *      on the wrapper's slot so the replay is ordered after it drains.
 *   2. Local EDT found in route table → write the dep slot and
 *      atomically decrement depc_needed.  If this was the last
 *      dependency (depc_needed hits 0), call arts_handle_ready_edt.
 *   3. Local EDT NOT found (still RESERVED or not yet created) →
 *      dispatch_or_defer on its slot; replayed when the EDT installs.
 *   4. Remote EDT → forward the signal over the network.
 *
 * The OoO replay re-issues arts_edt_satisfy_slot for the target, so the inline
 * hit path above (case 2) IS the single copy of the satisfy logic — the OoO
 * handler does not duplicate it.
 */

/* Defer a non-PTR satisfy on `edt_guid`'s slot (dispatch-or-defer). */
static void edt_defer_satisfy(arts_guid_t edt_guid, arts_guid_t data_guid,
                              uint32_t slot, arts_db_access_mode_t mode) {
  struct arts_ooo_args_edt_satisfy_s a = {
      .edt_guid = edt_guid, .data_guid = data_guid, .slot = slot, .mode = mode};
  arts_ooo_dispatch_or_defer_guid(edt_guid, OOO_EDT_SATISFY_SLOT, &a,
                                  sizeof(a));
}

/* Defer a DB_MODE_PTR satisfy: the inline payload is copied into the OoO
 * payload blob (freed here; dispatch_or_defer makes its own copy). */
static void edt_defer_satisfy_ptr(arts_guid_t edt_guid, arts_guid_t data_guid,
                                  uint32_t slot, void *ptr, unsigned int size) {
  uint32_t asz =
      (uint32_t)sizeof(struct arts_ooo_args_edt_satisfy_ptr_s) + size;
  char *buf = (char *)arts_malloc(asz);
  struct arts_ooo_args_edt_satisfy_ptr_s *a =
      (struct arts_ooo_args_edt_satisfy_ptr_s *)buf;
  a->edt_guid = edt_guid;
  a->data_guid = data_guid;
  a->slot = slot;
  a->size = size;
  if (size > 0 && ptr != NULL) {
    memcpy(buf + sizeof(*a), ptr, size);
  }
  arts_ooo_dispatch_or_defer_guid(edt_guid, OOO_EDT_SATISFY_SLOT_PTR, buf, asz);
  arts_free(buf);
}

/* Pure core — apply a satisfy to an already-acquired, valid EDT.  No lookup /
 * acquire / defer: the caller (arts_handler_edt_satisfy_slot via
 * dispatch_or_defer) guarantees `edt` is live.  Writes depv[slot], decrements
 * depc_needed, and schedules the EDT when the last dependency lands. */
static void edt_apply_satisfy(struct arts_edt_s *edt, uint32_t slot,
                              arts_guid_t data_guid, arts_db_access_mode_t mode,
                              void *ptr, unsigned int size) {
  arts_edt_dep_t *edt_dep = (arts_edt_dep_t *)arts_get_depv(edt);
  if (slot < edt->depc) {
    edt_dep[slot].guid = data_guid;
    if (mode == DB_MODE_PTR && size > 0) {
      void *copy = arts_malloc(size);
      memcpy(copy, ptr, size);
      edt_dep[slot].ptr = copy;
    } else {
      edt_dep[slot].ptr = ptr;
    }
    if (mode != DB_MODE_NULL) {
      edt_dep[slot].mode = mode;
    }
  }
  unsigned int res = arts_atomic_sub(&edt->depc_needed, 1U);
  ARTS_INFO("Signal EDT[Guid:%lu, Slot:%u] DB[Guid:%lu] depc_needed=%u→%u",
            edt->guid, slot, data_guid, res + 1, res);
  if (res == 0) {
    ARTS_INFO("EDT[Guid:%lu] all deps satisfied — firing", edt->guid);
    arts_handle_ready_edt(edt);
  }
}

/* Home-routed handler (OOO_EDT_SATISFY_SLOT): item is the installed EDT. */
void arts_handler_edt_satisfy_slot(void *item, void *vargs) {
  struct arts_ooo_args_edt_satisfy_s *a =
      (struct arts_ooo_args_edt_satisfy_s *)vargs;
  edt_apply_satisfy((struct arts_edt_s *)item, a->slot, a->data_guid, a->mode,
                    NULL, 0);
}

/* Home-routed handler (OOO_EDT_SATISFY_SLOT_PTR): inline payload trails args.
 */
void arts_handler_edt_satisfy_slot_ptr(void *item, void *vargs) {
  struct arts_ooo_args_edt_satisfy_ptr_s *a =
      (struct arts_ooo_args_edt_satisfy_ptr_s *)vargs;
  void *ptr = a->size > 0 ? (void *)(a + 1) : NULL;
  edt_apply_satisfy((struct arts_edt_s *)item, a->slot, a->data_guid,
                    (arts_db_access_mode_t)DB_MODE_PTR, ptr, a->size);
}

/* arts_edt_satisfy_slot — OCR-standard API: supply depv[slot] on an EDT.
 *   home == self → dispatch_or_defer (acquire the EDT → run the handler, or
 *                  defer on the slot until the EDT installs);
 *   home != self → MSG_EDT_SATISFY_SLOT wire (handler runs on the home rank);
 *   CDAG (GPU wrapper has outstanding invalidations) → force-defer on the
 *                  wrapper's slot; the replay re-signals this EDT after drain.
 * The satisfy logic lives once in edt_apply_satisfy (the handler); this entry
 * only routes.  arts_signal_edt is a deprecated alias of the same signature. */
void arts_edt_satisfy_slot(arts_guid_t edt_packet, uint32_t slot,
                           arts_guid_t data_guid, arts_db_access_mode_t mode,
                           void *ptr, unsigned int size) {
  TIME_EDT_SIGNAL_START();
  INCREMENT_NUM_EDT_SIGNAL_BY(1);

#ifdef ARTS_USE_CXL
  /* CXL GUID encodes the pointer directly — surface it on the dep slot so
   * that the prep_dbs/release_dbs flush helpers see the right pointer. */
  if (ptr == NULL && mode != DB_MODE_PTR && mode != DB_MODE_VAL &&
      arts_guid_is_cxl(data_guid)) {
    ptr = (void *)((struct arts_db_s *)arts_cxl_get_ptr(data_guid) + 1);
  }
#endif

  if (current_edt && current_edt->invalidate_count > 0) {
    /* CDAG: hold the satisfy until the GPU wrapper EDT's invalidations drain.
     */
    if (mode == DB_MODE_PTR) {
      edt_defer_satisfy_ptr(edt_packet, data_guid, slot, ptr, size);
    } else {
      struct arts_ooo_args_edt_satisfy_s a = {.edt_guid = edt_packet,
                                              .data_guid = data_guid,
                                              .slot = slot,
                                              .mode = mode};
      arts_ooo_push_guid(current_edt->guid, OOO_EDT_SATISFY_SLOT, &a,
                         sizeof(a));
    }
  } else if (arts_guid_get_rank(edt_packet) == arts_global_rank_id) {
    /* Local home: acquire-or-defer; the handler supplies the dep slot. */
    if (mode == DB_MODE_PTR) {
      edt_defer_satisfy_ptr(edt_packet, data_guid, slot, ptr, size);
    } else {
      edt_defer_satisfy(edt_packet, data_guid, slot, mode);
    }
  } else {
    /* Remote home: one satisfy message carries mode + (DB_MODE_PTR) payload. */
    arts_send_edt_satisfy_slot(edt_packet, data_guid, slot, mode, ptr, size);
  }
  TIME_EDT_SIGNAL_STOP();
}

volatile uint64_t outstanding_edts = 0;
void check_out_edts(uint64_t threshold) {
  static uint64_t count = 0;
  if (arts_atomic_fetch_add_u64(&count, 1) + 1 == threshold) {
    arts_atomic_fetch_sub_u64(&count, threshold);
  }
}

void arts_lc_sync(arts_guid_t edt_guid, uint32_t slot, arts_guid_t data_guid) {
  arts_guid_kind_t type = arts_guid_get_kind(data_guid);
  (void)type;
  arts_edt_satisfy_slot(edt_guid, slot, data_guid,
                        (arts_db_access_mode_t)DB_MODE_LC_SYNC, NULL, 0);
}

void arts_gpu_signal_edt_memset(arts_guid_t edt_guid, uint32_t slot,
                                arts_guid_t data_guid) {
  arts_db_access_mode_t mode = (arts_db_access_mode_t)DB_MODE_MEMSET;
  arts_shared_ptr_t dh = arts_route_table_lookup_db(data_guid);
  struct arts_db_s *db = (struct arts_db_s *)arts_shared_get(dh);
  if (db && db->db_type == ARTS_DB_GPU) {
    mode = (arts_db_access_mode_t)DB_MODE_LC_NO_COPY;
  }
  if (db) {
    arts_shared_release(&dh);
  }
  arts_edt_satisfy_slot(edt_guid, slot, data_guid, mode, NULL, 0);
}

arts_guid_t arts_edt_get_finish_event(arts_guid_t edt_guid) {
  /* Single lookup → read field → release.  Same pattern as the rest of
   * edt.c (e.g. arts_edt_destroy).  Returns NULL_GUID when edt_guid is not
   * registered in this rank's route table — i.e., when the EDT is homed on
   * another rank.  Cross-node queries require a remote handler; that path is
   * deferred to a future extension. */
  if (edt_guid == NULL_GUID) {
    return NULL_GUID;
  }
  arts_shared_ptr_t h = arts_route_table_lookup_edt(edt_guid);
  struct arts_edt_s *edt = (struct arts_edt_s *)arts_shared_get(h);
  if (!edt) {
    return NULL_GUID;
  }
  arts_guid_t fe = edt->finish_event;
  arts_shared_release(&h);
  return fe;
}
