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
/* OoO engine — unified dispatch_or_defer.
 *
 * One Treiber stack (oooList) per route_table slot accumulates deferred
 * operations that arrived before their target object was installed.  A single
 * arts_ooo_payload_s node type (link first, kind tag, trailing args blob)
 * replaces the former per-kind structs + oo_node wrapper.
 *
 * Two roles, cleanly split:
 *   - Create handler (Cat A) installs the object then calls arts_ooo_drain as
 *     its last step.
 *   - Non-create handler (Cat B/C) receives an already-acquired, valid item
 *     from dispatch_or_defer and operates on it — no lookup/acquire/push in
 *     the handler body.  The g_ooo_table[kind] entries are these handler
 *     bodies, defined in each subsystem TU.
 *
 * Concurrency (lock-free, per-call acquire):
 *   - A producer's dispatch_or_defer reloads slot.value every call.  HIT
 *     (value != NULL) → run the handler with a ref pinned across the call.
 *     MISS (value == NULL) → push the payload; then re-check value and, if an
 *     installer raced in, drain (so the node is not stranded).
 *   - A producer only ever pushes while value == NULL.  Once value is
 *     installed, every producer HITs and dispatches inline (never pushes), so
 *     no push races a create handler's drain.  Pre-install pushes are caught
 *     by the install's drain snapshot; a push that loses that race triggers
 *     its own drain via the post-push re-check.  No drain lock is needed.
 *   - drain takes ONE reverse_drain snapshot and walks it once.  A node that
 *     MISSes mid-walk (a destroy earlier in the same walk NULLed the slot)
 *     re-pushes onto a fresh chain to await the next install (labeled-GUID
 *     reuse) — it is not re-walked in this pass, so no spin.
 */
#include "arts/gas/out_of_order.h"

#include <stdatomic.h>
#include <string.h>

#include "arts.h" /* arts_add_dependence, arts_db_access_mode_t */
#include "arts/compute/edt.h"
#include "arts/gas/route_table.h"
#include "arts/memory/coherence_handlers.h"
#include "arts/memory/db.h"      /* DB_MODE_PTR (internal access mode) */
#include "arts/remote/handler.h" /* arts_db_request_callback */
#include "arts/runtime_state.h"  /* arts_handle_ready_edt */
#include "arts/sync/epoch.h"     /* increment_*_epoch, send_epoch */
#include "arts/sync/event.h"     /* arts_event_satisfy_slot */
#include "arts/sync/shared.h"
#include "arts/transport/protocol.h" /* coherence packet structs */
#include "arts/utils/lockfree_lifo.h"
#include "arts/utils/malloc.h"

/* ===== g_ooo_table — kind → replay handler ================================
 * Each handler replays the operation against the now-installed target by
 * re-issuing the original entry (internal_signal_edt, arts_event_satisfy_slot,
 * arts_handler_db_*, ...).  The entry's own lookup HITs during a drain
 * (drain runs post-install), takes its inline hit path, and does NOT re-enter
 * dispatch_or_defer — so re-issue is one level deep, never recursive.  `item`
 * (the acquired object) is passed through for the one handler (db_acquire)
 * that consumes it directly. */

/* EDT satisfy handlers live in edt.c (arts_handler_edt_satisfy_slot[_ptr]) —
 * pure cores that write the acquired EDT's dep slot. */

/* Event satisfy / add-dependence handlers live in event.c
 * (arts_handler_event_satisfy_slot / arts_handler_event_add_dependence) —
 * pure cores that operate on the acquired event. */

static void ooo_h_handle_ready_edt(void *item, void *vargs) {
  (void)item;
  struct arts_ooo_args_handle_ready_s *a = vargs;
  arts_handle_ready_edt(a->edt);
}

static void ooo_h_db_acquire(void *item, void *vargs) {
  struct arts_ooo_args_db_acquire_s *a = vargs;
  arts_db_request_callback(a->edt, a->slot, (struct arts_db_s *)item);
}

/* Epoch handlers (inc_*, request, send) live in epoch.c as pure cores on the
 * acquired epoch — no re-issue wrappers here. */

static void ooo_h_db_ownership_request(void *item, void *vargs) {
  (void)item;
  struct arts_ooo_args_db_ownership_request_s *a = vargs;
  struct arts_remote_lock_req_packet_s p;
  p.header.rank = a->requester;
  p.db_guid = a->db_guid;
  arts_handler_db_ownership_request(&p);
}

static void ooo_h_db_snapshot_request(void *item, void *vargs) {
  (void)item;
  struct arts_ooo_args_db_snapshot_request_s *a = vargs;
  struct arts_remote_get_data_packet_s p;
  p.header.rank = a->requester;
  p.db_guid = a->db_guid;
  p.waiter_addr = a->waiter_addr;
  arts_handler_db_snapshot_request(&p);
}

static void ooo_h_db_destroy(void *item, void *vargs) {
  (void)item;
  struct arts_ooo_args_db_destroy_s *a = vargs;
  struct arts_remote_destroy_req_packet_s p;
  p.header.rank = a->requester;
  p.db_guid = a->db_guid;
  arts_handler_db_destroy(&p);
}

static void ooo_h_db_writeback(void *item, void *vargs) {
  (void)item;
  struct arts_ooo_args_db_writeback_s *a = vargs;
  struct arts_remote_writeback_packet_s p;
  p.header.rank = a->releaser;
  p.db_guid = a->db_guid;
  p.version = a->version;
  p.cv = a->cv;
  p.flag = a->flag;
  const void *data = a->data_size > 0 ? (const void *)(a + 1) : NULL;
  arts_handler_db_writeback(&p, data, a->data_size);
}

static const arts_ooo_handler_fn g_ooo_table[OOO_KIND_COUNT] = {
    [OOO_EDT_SATISFY_SLOT] = arts_handler_edt_satisfy_slot,
    [OOO_EVENT_SATISFY_SLOT] = arts_handler_event_satisfy_slot,
    [OOO_EVENT_ADD_DEPENDENCE] = arts_handler_event_add_dependence,
    [OOO_HANDLE_READY_EDT] = ooo_h_handle_ready_edt,
    [OOO_DB_ACQUIRE] = ooo_h_db_acquire,
    [OOO_EDT_SATISFY_SLOT_PTR] = arts_handler_edt_satisfy_slot_ptr,
    [OOO_EPOCH_REQUEST] = arts_handler_epoch_request,
    [OOO_EPOCH_SEND] = arts_handler_epoch_send,
    [OOO_EPOCH_INC_ACTIVE] = arts_handler_epoch_inc_active,
    [OOO_EPOCH_INC_FINISHED] = arts_handler_epoch_inc_finished,
    [OOO_EPOCH_INC_QUEUE] = arts_handler_epoch_inc_queue,
    [OOO_DB_OWNERSHIP_REQUEST] = ooo_h_db_ownership_request,
    [OOO_DB_SNAPSHOT_REQUEST] = ooo_h_db_snapshot_request,
    [OOO_DB_DESTROY] = ooo_h_db_destroy,
    [OOO_DB_WRITEBACK] = ooo_h_db_writeback,
};

/* ===== payload alloc ====================================================== */

static struct arts_ooo_payload_s *
arts_ooo_payload_alloc(ooo_kind_t kind, const void *args, uint32_t args_size) {
  struct arts_ooo_payload_s *p = (struct arts_ooo_payload_s *)arts_malloc(
      sizeof(struct arts_ooo_payload_s) + args_size);
  p->kind = kind;
  p->args_size = args_size;
  if (args_size > 0 && args != NULL) {
    memcpy(arts_ooo_payload_args(p), args, args_size);
  }
  return p;
}

/* ===== dispatch_or_defer ================================================== */

void arts_ooo_dispatch_or_defer(struct arts_route_item_s *slot,
                                struct arts_ooo_payload_s *payload,
                                ooo_kind_t kind, const void *args,
                                uint32_t args_size) {
  /* Per-call acquire: (re)load the slot value every entry so a destroy that
   * NULLed it earlier in the same drain walk is observed here. */
  arts_shared_ptr_t h = arts_atomic_shared_load(&slot->value);
  if (h) {
    void *item = arts_shared_get(h);
    /* Ref pinned across the whole handler call — a concurrent destroy's
     * exchange-to-NULL drops only the install ref; `h` keeps the object alive
     * until we release below. */
    g_ooo_table[kind](item, (void *)args);
    arts_shared_release(&h);
    if (payload != NULL) {
      arts_free(payload); /* drain context: the popped node is consumed */
    }
    return;
  }

  /* Miss — defer. */
  if (payload == NULL) {
    payload = arts_ooo_payload_alloc(kind, args, args_size); /* fresh entry */
  }
  /* else: drain re-entry — reuse the same payload (no alloc/free).
   *
   * The oooList is consumed ONLY by whole-chain reverse_drain (a single
   * atomic_exchange of the head); there is deliberately NO single-node pop.
   * That is what makes re-pushing a node back onto the same stack ABA-safe
   * under allocator address reuse — a push only links to "whatever is on top
   * now" and never caches a head->next for a CAS.  Do NOT add a single-node
   * pop on this stack. */
  arts_lf_stack_push(&slot->oooList, &payload->link);

  /* TOCTOU rescue: an installer may have published value between our initial
   * load and the push above.  A full fence before the re-check guarantees we
   * observe that install rather than a stale pre-install NULL: the installer's
   * value-store is sequenced before its reverse_drain (an atomic_exchange of
   * the very head our push just CAS'd), so without the fence a node pushed
   * just after the installer's drain snapshot could be stranded on weak memory
   * models.  If installed, drain so our node is not left waiting. */
  atomic_thread_fence(memory_order_seq_cst);
  h = arts_atomic_shared_load(&slot->value);
  if (h) {
    arts_shared_release(&h);
    arts_ooo_drain(slot);
  }
}

void arts_ooo_dispatch_or_defer_guid(arts_guid_t guid, ooo_kind_t kind,
                                     const void *args, uint32_t args_size) {
  arts_route_item_t *slot;
  arts_route_table_reserve_or_lookup(guid, &slot);
  arts_ooo_dispatch_or_defer(slot, NULL, kind, args, args_size);
}

void arts_ooo_push_guid(arts_guid_t guid, ooo_kind_t kind, const void *args,
                        uint32_t args_size) {
  arts_route_item_t *slot;
  arts_route_table_reserve_or_lookup(guid, &slot);
  struct arts_ooo_payload_s *payload =
      arts_ooo_payload_alloc(kind, args, args_size);
  arts_lf_stack_push(&slot->oooList, &payload->link);
}

/* ===== drain ============================================================== */

void arts_ooo_drain(struct arts_route_item_s *slot) {
  /* One snapshot, walked once.  Re-pushed misses land on a fresh chain and
   * await the next install's drain. */
  arts_lf_link_t *head = arts_lf_stack_reverse_drain(&slot->oooList);
  while (head != NULL) {
    /* Save next first: dispatch_or_defer may re-push this node (re-setting its
     * link->next) on a miss. */
    arts_lf_link_t *next =
        atomic_load_explicit(&head->next, memory_order_relaxed);
    struct arts_ooo_payload_s *payload = (struct arts_ooo_payload_s *)head;
    arts_ooo_dispatch_or_defer(slot, payload, payload->kind,
                               arts_ooo_payload_args(payload),
                               payload->args_size);
    head = next;
  }
}

void arts_ooo_drain_guid(arts_guid_t guid) {
  arts_route_item_t *slot;
  arts_route_table_reserve_or_lookup(guid, &slot);
  arts_ooo_drain(slot);
}

void arts_ooo_free_all(struct arts_route_item_s *slot) {
  arts_lf_link_t *head = arts_lf_stack_reverse_drain(&slot->oooList);
  while (head != NULL) {
    arts_lf_link_t *next =
        atomic_load_explicit(&head->next, memory_order_relaxed);
    arts_free((struct arts_ooo_payload_s *)head);
    head = next;
  }
}
