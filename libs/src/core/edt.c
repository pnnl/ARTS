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
#include "arts/edt.h"
#include "arts/db.h"
#include "arts/utils/malloc.h"

#include <string.h>

#include "arts/edt_context.h" /* current_edt + run-start/end ctx hooks */
#include "arts/event.h"       /* arts_event_create (finish-scope proxy) */
#include "arts/gas/guid.h"
#include "arts/gas/route_table.h"
#include "arts/ooo.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/transport/outbox.h"   /* outbound send helpers */
#include "arts/transport/protocol.h" /* wire packet structs */
#include "arts/utils/atomics.h"
#include "arts/utils/shared.h" /* arts_shared_ptr_t, get/release */

/* No-hint EDT placement policy: ROUNDROBIN (default) distributes execution
 * rank across all nodes when the caller expresses no placement preference
 * (NULL hint or ARTS_HINT_ANY_RANK), mirroring arts_db_create's NULL-hint
 * home distribution.  CREATOR pins to the calling rank (legacy behavior).
 * CMake sets this for every libarts compile; the fallback covers any TU
 * that pulls in edt.c outside the normal build (e.g. direct inclusion). */
#ifndef ARTS_NOHINT_EDT_ROUNDROBIN
#define ARTS_NOHINT_EDT_ROUNDROBIN 1
#endif

#ifdef ARTS_USE_GPU
#include "arts/gpu/gpu_internal.h"
#endif

#include "arts/cxl/wrapper.h"
#ifdef ARTS_USE_CXL
#include "arts/cxl/deque.h"
#endif

/* Per-worker EDT-execution context (current_edt, owned-finish-events list,
 * created-DB tracking + save/restore snapshot) lives in sync/edt_context.c.
 * edt.c reads `current_edt` directly and calls the run-start/run-end context
 * hooks below via that header. */

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
/* canonical struct-free path; sole caller is arts_edt_deleter in this TU. */
static void arts_edt_free(struct arts_edt_s *edt);

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

/* Arm the EDT's non-owning self-cb alias immediately after install, before the
 * EDT can become runnable.  Looks up the cb just published into the route slot
 * and stores the bare pointer (the lookup's transient +1 is released so the
 * alias adds no strong count — see the field comment in runtime_types.h).
 * Must be called on the installing thread, with the EDT still pinned against a
 * premature fire (depc > 0 with no queued signals, or the sentinel held). */
static void arts_edt_arm_self_cb(struct arts_edt_s *edt, arts_guid_t guid) {
  arts_shared_ptr_t cb = arts_route_table_lookup_edt(guid);
  edt->self_cb = cb;
  arts_shared_release(&cb);
}

/*
 * arts_edt_create_core — Core EDT allocation and registration.
 *
 * Allocates the EDT struct (header + paramv + depv + modes), assigns its GUID,
 * copies parameters, joins its finish scope (if any), and places the EDT into
 * the route table so that incoming signals can find it.
 *
 * Two paths exist depending on whether a GUID was pre-reserved:
 *   1. New GUID (created_guid == true):
 *        - arts_route_table_install (no race — nobody else knows the GUID
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
bool arts_edt_create_core(struct arts_edt_s *edt, arts_guid_kind_t guid_kind,
                          arts_guid_t *guid, unsigned int rank,
                          unsigned int edt_space, arts_edt_t func_ptr,
                          uint32_t paramc, const uint64_t *paramv,
                          uint32_t depc, arts_guid_t hint_finish_event,
                          arts_guid_t hint_output_event, uint64_t arts_id,
                          uint32_t flags) {
  if (!edt) {
    edt = (struct arts_edt_s *)arts_calloc_aligned(1, edt_space,
                                                 ARTS_CACHE_LINE_SIZE);
  }
  if (!edt) {
    ARTS_ERROR("EDT allocation failed (size=%u)", edt_space);
  }

  /* lifecycle/deleter handled by the route_table cb (deleter-by-kind) on
   * install.  The only per-object shared field is self_cb (a non-owning alias
   * to that cb): zero-initialised here (calloc / left NULL on a caller-provided
   * buffer) and armed by arts_edt_arm_self_cb right after install.  Kind comes
   * from the GUID (bits 63-62); total size from
   * arts_edt_total_size(paramc/depc) — no per-object header stores them. */
  (void)edt_space;
  edt->arts_id = arts_id;

  bool created_guid = false;
  if (*guid == NULL_GUID) {
    created_guid = true;
    edt->guid = *guid = arts_guid_create_for_rank(rank, guid_kind);
  } else {
    edt->guid = *guid;
  }

  edt->func_ptr = func_ptr;
  edt->depc = depc;
  edt->paramc = paramc;
  edt->depc_needed = depc;

  /* Determine finish-scope for this EDT.
   *
   * Finish scopes are created explicitly via arts_event_create(FINISH); an EDT
   * joins one by passing it in hint.finish_event, otherwise it inherits the
   * caller's ambient finish_event (transitive membership).  Either way the EDT
   * INCRs the scope at create and DECRs it on completion (in
   * arts_unset_thread_local_edt_info).  `current_edt` is the file-static
   * thread-local maintained by arts_set/unset_thread_local_edt_info — direct
   * access, no route_table lookup needed (same TU). */
  arts_guid_t parent_fe;
  if (hint_finish_event != NULL_GUID) {
    parent_fe = hint_finish_event;
  } else if (current_edt) {
    parent_fe = current_edt->finish_event;
  } else {
    parent_fe = NULL_GUID;
  }
  edt->finish_event = parent_fe;
  if (parent_fe != NULL_GUID) {
    /* Join/inherit: INCR completes before the new EDT can reach its own DECR
     * (which runs only after the EDT executes — strictly later in this
     * thread). */
    arts_event_satisfy_slot(parent_fe, NULL_GUID, ARTS_EVENT_LATCH_INCR_SLOT);
  }
  /* Output event (per-EDT result channel): never inherited — it belongs to
   * this EDT only.  The run path satisfies it with the EDT's returned GUID
   * after the EDT's data-block releases. */
  edt->output_event = hint_output_event;
  (void)flags;

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
            "PreReserved:%s, FuncPtr:%p]",
            *guid, edt->arts_id, edt->depc, rank, created_guid ? "no" : "yes",
            (void *)func_ptr);

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
      arts_route_table_install(edt, *guid, arts_global_rank_id, false);
      arts_edt_arm_self_cb(edt, *guid);
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
      /* add_item installs the cb unconditionally and fires the OoO list
       * internally (replaying queued signals) — the sentinel set above
       * guarantees those replays cannot drive depc_needed to 0 before we
       * remove it below.  An EDT GUID has a single creator (no rendezvous), so
       * the unconditional install matches the default create contract; on the
       * normal empty slot it is a plain install, and a (UB) re-create replaces
       * the prior generation rather than leaking the new object. */
      arts_route_table_install(edt, *guid, arts_global_rank_id, false);
      arts_edt_arm_self_cb(edt, *guid);
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

  /* Snapshot hint (NULL = ARTS_EDT_HINT_DEFAULTS).  After this all optional
   * fields are well-defined and follow the documented precedence:
   *   - if .guid != NULL_GUID, the GUID's rank field overrides .rank
   *   - if .rank == ARTS_HINT_ANY_RANK (or hint itself is NULL), the rank is
   *     policy-selected below (ARTS_NOHINT_EDT_ROUNDROBIN) rather than taken
   *     from .rank
   *   - if .finish_event == NULL_GUID, the EDT inherits the caller's ambient
   *     finish scope (handled inside arts_edt_create_core). */
  arts_edt_hint_t snap = hint
                             ? *hint
                             : (arts_edt_hint_t){.rank = ARTS_HINT_CURRENT_RANK,
                                                 .edt_id = 0,
                                                 .guid = NULL_GUID};

  arts_guid_t guid = snap.guid;
  unsigned int rank;
  if (guid != NULL_GUID) {
    rank = arts_guid_get_rank(guid);
  } else if (hint == NULL || snap.rank == ARTS_HINT_ANY_RANK) {
    /* No placement preference (NULL hint, or an explicit hint that still
     * needs other fields populated but leaves rank unpinned via
     * ARTS_HINT_ANY_RANK — e.g. the OCR shim's finish/output-event-bearing
     * hint).  ROUNDROBIN distributes execution rank across all nodes
     * (mirrors arts_db_create's NULL-hint home distribution); CREATOR
     * reproduces the legacy pin-to-creator behavior.  Both arms resolve to
     * a concrete rank — the sentinel never reaches guid encoding. */
#if ARTS_NOHINT_EDT_ROUNDROBIN
    rank = arts_atomic_fetch_add(&arts_node_info.edt_rr_route, 1U) %
           arts_global_rank_count;
#else
    rank = arts_global_rank_id;
#endif
  } else if (snap.rank != ARTS_HINT_CURRENT_RANK) {
    rank = snap.rank;
  } else {
    rank = arts_global_rank_id;
  }

  unsigned int edt_space = sizeof(struct arts_edt_s) +
                           (paramc * sizeof(uint64_t)) +
                           (depc * sizeof(arts_edt_dep_t));
  bool ok = arts_edt_create_core(
      NULL, ARTS_GUID_EDT, &guid, rank, edt_space, func_ptr, paramc, paramv,
      depc, snap.finish_event, snap.output_event, snap.edt_id, snap.flags);
  TIME_EDT_CREATE_STOP();
  return ok ? guid : NULL_GUID;
}

/* Register the running EDT's result GUID — delivered as the payload when the
 * run path satisfies the EDT's output_event after its data-block releases.
 * `current_edt` is the worker thread-local; outside a running EDT this is a
 * documented no-op. */
void arts_edt_set_result(arts_guid_t result_guid) {
  if (current_edt) {
    current_edt->output_data = result_guid;
  }
}

static void arts_edt_free(struct arts_edt_s *edt) {
  /* rw_sorted is the GUID-sorted serialized-dep order, allocated once at
   * arts_db_acquire_all entry (NULL if the EDT never reached the acquire
   * phase).  Freeing here — the single canonical struct-free — covers every
   * lifetime end (run completion, cancel, destroy) with no double-free. */
  arts_free(edt->rw_sorted);
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
  /* route through arts_route_table_set_destroyed so the deleter
   * (arts_edt_deleter -> arts_edt_free) runs once outstanding refs are
   * returned.  Capturing the GUID up front, then calling mark_delete:
   * if no other thread holds an acquire ref, this drops the install ref
   * and free_item invokes the deleter inline.  If another thread holds
   * a transient lookup_edt_safe ref, free_item is deferred to the last
   * release_item.  Either way `edt` is no longer safe to dereference
   * after this call returns. */
  arts_guid_t guid = edt->guid;
  arts_route_table_set_destroyed(guid);
}

/* Pure Cat-B body (g_ooo_table[OOO_EDT_DESTROY]).  The EDT is installed and
 * ref-pinned by dispatch_or_defer; this body operates on the live item without
 * any lookup / pin / release of its own.  OCR restricts ocrEdtDestroy to
 * pre-runnable EDTs (depc_needed > 0); destroying a runnable/queued/running EDT
 * is UB and is skipped.  arts_route_table_set_destroyed detaches the slot cb so
 * the deleter (arts_edt_deleter -> arts_edt_free) runs once outstanding refs
 * (including dispatch_or_defer's own pin) drain; it is idempotent, so a
 * duplicate/late replay is a safe no-op. */
void arts_handler_edt_destroy(void *item_v, void *args_v) {
  struct arts_edt_s *edt = (struct arts_edt_s *)item_v;
  struct arts_ooo_args_edt_destroy_s *a =
      (struct arts_ooo_args_edt_destroy_s *)args_v;
  if (edt->depc_needed == 0) {
    ARTS_INFO("EDT destroy on runnable/queued EDT [Guid:%lu] — UB; ignoring",
              a->guid);
    return;
  }
  ARTS_INFO("EDT destroy [Guid:%lu, Id:%lu, Depc:%u, DepcNeeded:%u] on rank %u",
            edt->guid, edt->arts_id, edt->depc, edt->depc_needed,
            arts_global_rank_id);
  arts_route_table_set_destroyed(a->guid);
}

/* Cross-rank send: forward the destroy to the EDT's home rank (symmetric with
 * arts_send_event_destroy / arts_send_db_destroy). */
void arts_send_edt_destroy(unsigned int home_rank, arts_guid_t guid) {
  struct arts_msg_guid_only_packet_s packet;
  packet.guid = guid;
  arts_fill_packet_header(&packet.header, sizeof(packet), MSG_EDT_DESTROY);
  arts_transport_send_async((int)home_rank, (char *)&packet, sizeof(packet));
}

/* arts_edt_destroy — API: destroy an EDT by GUID.
 *   home != self → MSG_EDT_DESTROY wire (handler runs on the home rank);
 *   home == self → dispatch_or_defer (run the destroy body on the live EDT, or
 *                  defer on the slot until the EDT installs — symmetric with
 * the RX path and the satisfy path's local-home branch). */
void arts_edt_destroy(arts_guid_t guid) {
  /* A GUID's home rank is authoritative; an EDT homed elsewhere is destroyed
   * at its home (the route_table entry + finish-scope accounting live there).
   */
  unsigned int home = arts_guid_get_rank(guid);
  if (home != arts_global_rank_id) {
    arts_send_edt_destroy(home, guid);
    return;
  }
  struct arts_ooo_args_edt_destroy_s a = {.guid = guid};
  arts_ooo_dispatch_or_defer_guid(guid, OOO_EDT_DESTROY, &a, sizeof(a));
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
 *   1. GPU LC invalidation drain (current EDT has pending device-replica
 * invalidations) → force-defer on the wrapper's slot so the replay is ordered
 * after it drains.
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

/* Defer a satisfy on `edt_guid`'s slot (dispatch-or-defer), mode-discriminated.
 * For DB_MODE_PTR the inline payload (size bytes) is copied into the args blob
 * right after the args struct; all other modes carry a GUID/value reference
 * only (size == 0, no trailing payload).  The buffer is freed here —
 * dispatch_or_defer makes its own copy.  One helper covers both deliveries (the
 * handler / satisfy core branch on mode), so there is no separate PTR kind. */
static void edt_defer_satisfy(arts_guid_t edt_guid, arts_guid_t data_guid,
                              uint32_t slot, arts_db_access_mode_t mode,
                              void *ptr, unsigned int size) {
  /* An inline payload rides only when a real source pointer accompanies it; a
   * NULL source carries no bytes so the delivered slot pointer is a defined
   * NULL rather than a buffer of undefined contents. */
  uint32_t payload = (mode == DB_MODE_PTR && ptr != NULL) ? size : 0u;
  uint32_t asz = (uint32_t)sizeof(struct arts_ooo_args_edt_satisfy_s) + payload;
  char *buf = (char *)arts_malloc(asz);
  struct arts_ooo_args_edt_satisfy_s *a =
      (struct arts_ooo_args_edt_satisfy_s *)buf;
  a->edt_guid = edt_guid;
  a->data_guid = data_guid;
  a->slot = slot;
  a->mode = mode;
  a->size = payload;
  if (payload > 0 && ptr != NULL) {
    memcpy(buf + sizeof(*a), ptr, payload);
  }
  arts_ooo_dispatch_or_defer_guid(edt_guid, OOO_EDT_SATISFY_SLOT, buf, asz);
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
  /* (uint32_t)-1 is the "no specific slot" sentinel used by control
   * dependences (registered with slot -1): they decrement readiness without
   * writing any dependence-vector entry.  A real, in-range slot writes its
   * entry and decrements.  ANY OTHER slot is genuinely out of range — it is not
   * one of this EDT's dependences, so it must neither write past the vector nor
   * count toward readiness (else the EDT could fire before its real deps land);
   * ignore it. */
  const uint32_t NO_SLOT = (uint32_t)-1;
  bool writes_slot = (slot != NO_SLOT);
  if (writes_slot && slot >= edt->depc) {
    return;
  }
  if (writes_slot) {
    edt_dep[slot].guid = data_guid;
    /* An inline payload is only valid when a real source pointer accompanies
     * it.  A NULL source surfaces as a NULL slot pointer rather than a buffer
     * of undefined bytes, so the consumer never reads uninitialized memory. */
    if (mode == DB_MODE_PTR && size > 0 && ptr != NULL) {
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
  /* Decrement readiness for both a real in-range slot and the no-slot
   * sentinel; only a genuine out-of-range slot (rejected above) is skipped. */
  unsigned int res = arts_atomic_sub(&edt->depc_needed, 1U);
  ARTS_INFO("Signal EDT[Guid:%lu, Slot:%u] DB[Guid:%lu] depc_needed=%u→%u",
            edt->guid, slot, data_guid, res + 1, res);
  if (res == 0) {
    ARTS_INFO("EDT[Guid:%lu] all deps satisfied — firing", edt->guid);
    arts_handle_ready_edt(edt);
  }
}

/* Home-routed handler (OOO_EDT_SATISFY_SLOT): item is the installed EDT.
 * Mode-discriminated — for DB_MODE_PTR the inline payload (a->size bytes)
 * trails the args struct; all other modes carry a GUID/value reference only
 * (size == 0).  The satisfy core copies the payload exactly when mode ==
 * DB_MODE_PTR. */
void arts_handler_edt_satisfy_slot(void *item, void *vargs) {
  struct arts_ooo_args_edt_satisfy_s *a =
      (struct arts_ooo_args_edt_satisfy_s *)vargs;
  void *ptr = (a->mode == DB_MODE_PTR && a->size > 0) ? (void *)(a + 1) : NULL;
  edt_apply_satisfy((struct arts_edt_s *)item, a->slot, a->data_guid, a->mode,
                    ptr, a->size);
}

/* arts_edt_satisfy_slot — OCR-standard API: supply depv[slot] on an EDT.
 *   home == self → dispatch_or_defer (acquire the EDT → run the handler, or
 *                  defer on the slot until the EDT installs);
 *   home != self → MSG_EDT_SATISFY_SLOT wire (handler runs on the home rank);
 *   GPU LC (wrapper has outstanding device-replica invalidations) →
 * force-defer on the wrapper's slot; the replay re-signals this EDT after
 * drain. The satisfy logic lives once in edt_apply_satisfy (the handler);
 * this entry only routes.  arts_signal_edt is a deprecated alias of the same
 * signature. */
void arts_edt_satisfy_slot(arts_guid_t edt_guid, uint32_t slot,
                           arts_guid_t data_guid, arts_db_access_mode_t mode,
                           void *ptr, unsigned int size) {
  TIME_EDT_SIGNAL_START();
  INCREMENT_NUM_EDT_SIGNAL_BY(1);

  /* An inline payload is meaningful only with a real source pointer to copy
   * from. A NULL source carries no bytes, so normalize the size to zero on
   * every routing path; the slot then receives a defined NULL rather than a
   * buffer of undefined contents. */
  if (ptr == NULL) {
    size = 0;
  }

#ifdef ARTS_USE_CXL
  /* CXL GUID encodes the pointer directly — surface it on the dep slot so
   * that the prep_dbs/release_dbs flush helpers see the right pointer. */
  if (ptr == NULL && mode != DB_MODE_PTR && mode != DB_MODE_VAL &&
      arts_guid_is_cxl(data_guid)) {
    ptr = (void *)((struct arts_db_s *)arts_cxl_get_ptr(data_guid) + 1);
  }
#endif

  if (current_edt && current_edt->invalidate_count > 0) {
    /* GPU LC: hold the satisfy until the GPU wrapper EDT's invalidations
     * drain. DB_MODE_PTR dispatch-or-defers on the target (the inline payload
     * rides in the args blob); every other mode force-pushes on the wrapper's
     * slot so the re-signal of this EDT replays only after the wrapper's
     * invalidations drain. */
    if (mode == DB_MODE_PTR) {
      edt_defer_satisfy(edt_guid, data_guid, slot, mode, ptr, size);
    } else {
      struct arts_ooo_args_edt_satisfy_s a = {.edt_guid = edt_guid,
                                              .data_guid = data_guid,
                                              .slot = slot,
                                              .mode = mode,
                                              .size = 0};
      arts_ooo_push_guid(current_edt->guid, OOO_EDT_SATISFY_SLOT, &a,
                         sizeof(a));
    }
  } else if (arts_guid_get_rank(edt_guid) == arts_global_rank_id) {
    /* Local home: acquire-or-defer; the handler supplies the dep slot.  The
     * mode-discriminated helper handles the DB_MODE_PTR inline payload. */
    edt_defer_satisfy(edt_guid, data_guid, slot, mode, ptr, size);
  } else {
    /* Remote home: one satisfy message carries mode + (DB_MODE_PTR) payload.
     */
    arts_send_edt_satisfy_slot(edt_guid, data_guid, slot, mode, ptr, size);
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

void arts_send_memory_move(unsigned int rank, arts_guid_t guid, void *ptr,
                           unsigned int mem_size, unsigned message_type,
                           void (*free_method)(void *)) {
  TIME_REMOTE_MOVE_START();
  struct arts_msg_guid_only_packet_s packet;
  arts_fill_packet_header(&packet.header, sizeof(packet) + mem_size,
                          message_type);
  packet.guid = guid;
  arts_transport_send_payload_async_free((int)rank, (char *)&packet,
                                         sizeof(packet), (char *)ptr, 0,
                                         mem_size, free_method);
  /* route_table slot now persists; Lifecycle redesign is follow-up work. */
  (void)guid;
  TIME_REMOTE_MOVE_STOP();
}

void arts_handler_edt_create(void *ptr) {
  struct arts_msg_guid_only_packet_s *packet =
      (struct arts_msg_guid_only_packet_s *)ptr;
  uint64_t size =
      packet->header.size - sizeof(struct arts_msg_guid_only_packet_s);
  struct arts_edt_s *edt =
      (struct arts_edt_s *)arts_malloc_aligned(size, ARTS_CACHE_LINE_SIZE);

  memcpy(edt, packet + 1, size);
  /* lifecycle/deleter handled by the route_table cb (deleter-by-kind) when
   * this EDT is installed below — no per-object shared field to stamp. */
  /* finish-scope chain: if the EDT arrived with a non-NULL finish_event,
   * the field currently holds the *parent* finish_event GUID (which lives
   * on the source rank).  Allocate a local proxy LATCH and rewrite the
   * field so this EDT's finish_event is local-home.  Register a dep so
   * that proxy fire emits DECR on the remote parent.
   *
   * The matching INCR on the remote parent was already emitted on the
   * source rank inside arts_edt_create_core before the EDT was
   * shipped — race-free under source-rank local sync ordering. */
  if (edt->finish_event != NULL_GUID) {
    arts_guid_t parent_fe = edt->finish_event;
    /* Single-shot proxy: auto_destroy is set at creation (immutable) so the
     * proxy is reclaimed the instant it fires rather than lingering.  A plain
     * LATCH, not a finish hint — the proxy chains explicitly to the remote
     * parent below, not to the local ambient finish scope. */
    arts_event_hint_t proxy_hint = ARTS_EVENT_HINT_LATCH(1);
    proxy_hint.auto_destroy = true;
    arts_guid_t proxy = arts_event_create(&proxy_hint);
    /* add_dependence registers the proxy in its own waiter list (local op).
     * The cross-node satisfy-on-fire is emitted automatically by the
     * LATCH fire path when proxy.counter reaches 0. */
    arts_add_dependence(proxy, parent_fe, ARTS_EVENT_LATCH_DECR_SLOT,
                        DB_MODE_NULL);
    edt->finish_event = proxy;
  }
  /* Sentinel protocol (mirrors the pre-reserved path in
   * arts_edt_create_core): bump depc_needed by 1 before the EDT becomes
   * globally visible.  add_item_race installs the EDT and replays any queued
   * dependency satisfies, and once installed a satisfy may also land
   * concurrently on another receiver thread.  Without the sentinel both that
   * satisfy (observing depc_needed hit 0) and this handler's own readiness
   * check would fire arts_handle_ready_edt for the same EDT — a double
   * dispatch.  The sentinel keeps depc_needed >= 1 across install + replay,
   * so exactly one party observes the 0 transition: the sentinel removal
   * below, or the last satisfy after we return. */
  edt->depc_needed += 1;
  /* add_item_race installs the EDT under the route_table lock.  On
   * rejection (another thread won the install race) free the freshly
   * unmarshaled buffer through the deleter — mirrors event_move's
   * race-loser cleanup pattern. */
  if (!arts_route_table_install_if_absent(edt, packet->guid,
                                          arts_global_rank_id, false)) {
    /* race-loser cleanup: if we allocated a proxy LATCH for the
     * finish-scope chain, drain it.  proxy.counter == 1 (self-alive
     * token, just allocated above).  DECR drives counter to 0 → fire,
     * which emits the cross-node DECR to the remote parent_fe via the
     * dep we just registered.  This cancels the source-rank INCR that
     * was emitted before this EDT was shipped, keeping the parent
     * finish-scope balanced.  proxy itself self-destroys on fire (LATCH
     * auto_destroy semantics). */
    if (edt->finish_event != NULL_GUID) {
      arts_event_satisfy_slot(edt->finish_event, NULL_GUID,
                              ARTS_EVENT_LATCH_DECR_SLOT);
    }
    arts_edt_get_deleter()(edt);
    return;
  }
  ARTS_INFO("EDT[Guid:%lu] Moved to Rank: %d", packet->guid,
            arts_global_rank_id);
  arts_edt_arm_self_cb(edt, packet->guid);
  /* Remove the sentinel.  add_item_race already replayed queued satisfies and
   * any concurrent satisfy decremented too; the unique observer of the 0
   * transition fires the EDT exactly once (here, or the last late satisfy).
   */
  unsigned int remaining = arts_atomic_sub(&edt->depc_needed, 1U);
  if (remaining == 0) {
    arts_handle_ready_edt(edt);
  }
}

void arts_send_edt_satisfy_slot(arts_guid_t edt, arts_guid_t db, uint32_t slot,
                                arts_db_access_mode_t mode, void *ptr,
                                unsigned int size) {
  unsigned int rank = arts_guid_get_rank(edt);
  if (rank == arts_global_rank_id) {
    /* EDT GUID claims a local home but may have migrated — resolve the
     * true owning rank through the route table. */
    rank = arts_route_table_lookup_rank(edt);
  }
  ARTS_INFO(
      "Remote Signal from DB[Guid:%lu] to EDT[Guid:%lu, Slot:%d, Rank:%u]", db,
      edt, slot, rank);

  if (size == 0) {
    /* Reference-only satisfy (GUID / value / NULL): fixed-size header on the
     * stack, no trailing payload. */
    struct arts_msg_edt_satisfy_slot_packet_s packet;
    packet.edt = edt;
    packet.db = db;
    packet.slot = slot;
    packet.mode = mode;
    packet.size = 0;
    arts_fill_packet_header(&packet.header, sizeof(packet),
                            MSG_EDT_SATISFY_SLOT);
    arts_transport_send_async((int)rank, (char *)&packet, sizeof(packet));
    return;
  }

  /* DB_MODE_PTR delivery: header + inline payload copied contiguously so the
   * receiver materializes the data without a follow-up fetch. */
  uint64_t total = sizeof(struct arts_msg_edt_satisfy_slot_packet_s) + size;
  char *buf = (char *)arts_malloc((size_t)total);
  struct arts_msg_edt_satisfy_slot_packet_s *packet =
      (struct arts_msg_edt_satisfy_slot_packet_s *)buf;
  packet->edt = edt;
  packet->db = db;
  packet->slot = slot;
  packet->mode = mode;
  packet->size = size;
  arts_fill_packet_header(&packet->header, total, MSG_EDT_SATISFY_SLOT);
  memcpy(buf + sizeof(*packet), ptr, size);
  arts_transport_send_async((int)rank, buf, (unsigned int)total);
  arts_free(buf);
}
