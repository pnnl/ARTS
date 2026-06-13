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

#include "arts/ooo.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h> /* memcpy (OoO payload alloc) */

#include "arts.h"
#include "arts/coherence/coherence.h" /* arts_handler_db_acquire */
#include "arts/coherence/handlers.h"  /* coherence wire handlers + replay */
#include "arts/db.h"                  /* arts_db_acquire_all */
#include "arts/edt.h"   /* arts_handler_edt_satisfy_slot[_ptr], arts_get_depv */
#include "arts/event.h" /* arts_handler_event_satisfy_slot / add_dependence */
#include "arts/gas/route_table.h"
#include "arts/system/print.h" /* ARTS_WARN / ARTS_INFO */
#include "arts/utils/lockfree_lifo.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"

/* ===========================================================================
 * OoO engine — unified dispatch_or_defer.
 *
 * One Treiber stack (ooo_list) per route_table slot accumulates deferred
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
 * ===========================================================================*/

/* ===== g_ooo_table — kind → replay handler ================================
 * Each handler replays the operation against the now-installed target by
 * re-issuing the original entry (internal_signal_edt, arts_event_satisfy_slot,
 * arts_handler_db_*, ...).  The entry's own lookup HITs during a drain
 * (drain runs post-install), takes its inline hit path, and does NOT re-enter
 * dispatch_or_defer — so re-issue is one level deep, never recursive.  `item`
 * (the acquired object) is passed through for the one handler (db_acquire)
 * that consumes it directly. */

/* EDT satisfy + destroy handlers live in edt.c
 * (arts_handler_edt_satisfy_slot[_ptr], arts_handler_edt_destroy) — pure cores
 * that write the acquired EDT's dep slot / detach the slot on destroy. */

/* Event satisfy / add-dependence / destroy handlers live in event.c
 * (arts_handler_event_satisfy_slot / arts_handler_event_add_dependence /
 * arts_handler_event_destroy) — pure cores that operate on the acquired
 * event. */

/* The OoO replay of a deferred local DB→EDT dependency is the coherence acquire
 * handler itself: arts_handler_db_acquire(item = installed db_s, args). The
 * drain only fires when the slot value is non-NULL (DB installed), so `item` is
 * always a valid db_s; the handler re-attempts the single dep through the
 * proper coherence path (writer_count / buffer-ref handling) and self-accounts
 * / parks. No whole-driver re-run. */

/* The coherence replay bodies are the wire handlers themselves
 * (arts_handler_db_ownership_request / _snapshot_request / _writeback /
 * _destroy) — pure (item, args) Cat-B bodies defined in the coherence TUs.
 * Each model's enum (and this table) carries only that model's OOO_DB_* kinds,
 * so a build references only the bodies it actually defines:
 *   - OWNERSHIP_REQUEST: OCR model only (ownership.c); RELAXED's enum omits it.
 *   - WRITEBACK: EAGER/RELAXED only; LAZY's enum omits it (LAZY fatals on the
 * wire).
 *   - OWNERSHIP_INVALIDATE: EAGER only.  EAGER can see a GRANT/INVALIDATE
 * reorder (or a before-create race) that lands INVALIDATE before the cache
 * installs, so it defers + replays here.  The lazy protocol never defers
 * INVALIDATE (home publishes the rw_holder target only after that rank's
 * cache install, so the dispatcher/self-send call the body directly) and the
 * relaxed model has no ownership transfer, so neither carries this kind. */

/* Event/EDT destroy replay bodies are the wire handlers themselves
 * (arts_handler_event_destroy / arts_handler_edt_destroy) — pure (item, args)
 * Cat-B bodies defined in event.c / edt.c.  The item is installed (drain runs
 * post-install) and ref-pinned by dispatch_or_defer; the body performs the
 * destroy action (route_table_set_destroyed) directly on it. */

/* Mirrors the per-model ooo_kind enum: the model-agnostic slots are always
 * present, and each build's OOO_DB_* arm initializes only that model's kinds
 * (each kind ↔ its handler 1:1). */
static const arts_ooo_handler_fn_t g_ooo_table[OOO_KIND_COUNT] = {
    [OOO_EVENT_SATISFY_SLOT] = arts_handler_event_satisfy_slot,
    [OOO_EDT_SATISFY_SLOT] = arts_handler_edt_satisfy_slot,
    [OOO_EVENT_ADD_DEPENDENCE] = arts_handler_event_add_dependence,
    [OOO_EDT_DESTROY] = arts_handler_edt_destroy,
    [OOO_EVENT_DESTROY] = arts_handler_event_destroy,
    [OOO_DB_DESTROY] = arts_handler_db_destroy,
#if defined(ARTS_COHERENCE_PROTOCOL_EAGER)
    [OOO_DB_ACQUIRE] = arts_db_acquire_replay_dep,
    [OOO_DB_SNAPSHOT_REQUEST] = arts_handler_db_snapshot_request,
    [OOO_DB_OWNERSHIP_REQUEST] = arts_handler_db_ownership_request,
    [OOO_DB_OWNERSHIP_INVALIDATE] = arts_handler_db_ownership_invalidate,
    [OOO_DB_WRITEBACK] = arts_handler_db_writeback,
#elif defined(ARTS_COHERENCE_PROTOCOL_LAZY)
    [OOO_DB_ACQUIRE] = arts_db_acquire_replay_dep,
    [OOO_DB_SNAPSHOT_REQUEST] = arts_handler_db_snapshot_request,
    [OOO_DB_OWNERSHIP_REQUEST] = arts_handler_db_ownership_request,
#elif defined(ARTS_MEMORY_MODEL_RELAXED)
    [OOO_DB_ACQUIRE] = arts_db_acquire_replay_dep,
    [OOO_DB_SNAPSHOT_REQUEST] = arts_handler_db_snapshot_request,
    [OOO_DB_WRITEBACK] = arts_handler_db_writeback,
#endif
};

/* ===== payload alloc ====================================================== */

static struct arts_ooo_payload_s *
arts_ooo_payload_alloc(ooo_kind_t kind, const void *args, uint32_t args_size) {
  struct arts_ooo_payload_s *p = (struct arts_ooo_payload_s *)arts_malloc(
      sizeof(struct arts_ooo_payload_s) + args_size);
  p->kind = kind;
  p->args_size = args_size;
  p->gen_at_defer = 0; /* real epoch snapshot taken at the fresh defer site */
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
    /* Cross-generation guard (drain replay only, kind-gated to INVALIDATE).
     * A payload deferred while one generation was live (e.g. a stale ownership
     * INVALIDATE) must not replay against a different generation installed by a
     * labeled-GUID re-create: doing so would apply a withdrawal that belongs to
     * the prior round to a fresh round's writer_count.  payload != NULL means
     * this is a drain re-entry (a fresh wire/API arrival has payload == NULL
     * and can never be stale, so it always dispatches).  Bump-on-destroy-only
     * makes an unchanged gen mean "same round → replay" and a changed gen mean
     * "a destroy intervened → drop".  Gated to OOO_DB_OWNERSHIP_INVALIDATE —
     * other kinds rely on before-create replay across the install and must not
     * drop.  Eager-only: that kind exists solely in the eager build's enum
     * (lazy/relaxed never defer INVALIDATE). */
#if defined(ARTS_COHERENCE_PROTOCOL_EAGER)
    if (payload != NULL && kind == OOO_DB_OWNERSHIP_INVALIDATE &&
        payload->gen_at_defer !=
            __atomic_load_n(&slot->gen, __ATOMIC_ACQUIRE)) {
      arts_shared_release(&h);
      arts_free(payload);
      return;
    }
#endif
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
    /* Snapshot the install-epoch ONCE, at the fresh defer.  A re-pushed node
     * (drain re-entry, payload != NULL) keeps its original epoch so a node that
     * misses one drain still drops correctly if a destroy later bumps gen. */
    payload->gen_at_defer = __atomic_load_n(&slot->gen, __ATOMIC_ACQUIRE);
  }
  /* else: drain re-entry — reuse the same payload (no alloc/free), preserving
   * its gen_at_defer (do NOT overwrite — it must stay the epoch of first
   * defer).
   *
   * The ooo_list is consumed ONLY by whole-chain reverse_drain (a single
   * atomic_exchange of the head); there is deliberately NO single-node pop.
   * That is what makes re-pushing a node back onto the same stack ABA-safe
   * under allocator address reuse — a push only links to "whatever is on top
   * now" and never caches a head->next for a CAS.  Do NOT add a single-node
   * pop on this stack. */
  arts_lf_stack_push(&slot->ooo_list, &payload->link);

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
  arts_lf_stack_push(&slot->ooo_list, &payload->link);
}

/* ===== drain ============================================================== */

void arts_ooo_drain(struct arts_route_item_s *slot) {
  /* One snapshot, walked once.  Re-pushed misses land on a fresh chain and
   * await the next install's drain. */
  arts_lf_link_t *head = arts_lf_stack_reverse_drain(&slot->ooo_list);
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
  arts_lf_link_t *head = arts_lf_stack_reverse_drain(&slot->ooo_list);
  while (head != NULL) {
    arts_lf_link_t *next =
        atomic_load_explicit(&head->next, memory_order_relaxed);
    arts_free((struct arts_ooo_payload_s *)head);
    head = next;
  }
}
