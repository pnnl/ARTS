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

#include "arts/gas/route_table.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>

#include "arts.h"
#include "arts/compute/edt.h" /* arts_handler_edt_satisfy_slot[_ptr] */
#include "arts/gas/guid.h"
#include "arts/memory/coherence_handlers.h"
#include "arts/memory/db.h"
#include "arts/runtime_state.h" /* arts_handle_ready_edt */
#include "arts/sync/epoch.h"    /* arts_handler_epoch_* */
#include "arts/sync/event.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/transport/protocol.h" /* coherence packet structs */
#include "arts/utils/atomics.h"      /* arts_atomic_sub */
#include "arts/utils/lockfree_lifo.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"
#include <string.h> /* memcpy (OoO payload alloc) */

#define HASH64(x, y) ((uint64_t)(x) * (y))

static inline uint64_t get_route_table_key(uint64_t x, unsigned int shift) {
  uint64_t hash = 14695981039346656037U;
  switch (shift) {
  case 10:
    hash *= 1021;
  case 11:
    hash *= 2039;
  case 12:
    hash *= 4093;
  case 13:
    hash *= 8191;
  case 14:
    hash *= 16381;
  case 15:
    hash *= 32749;
  case 16:
    hash *= 65521;
  case 17:
    hash *= 131071;
  case 18:
    hash *= 262139;
  case 19:
    hash *= 524287;
  case 20:
    hash *= 1048573;
  case 21:
    hash *= 2097143;
  case 22:
    hash *= 4194301;
  case 31:
    hash *= 2147483647;
  case 32:
    hash *= 4294967291;
  default:
    break;
  }

  return (HASH64(x, hash) >> (64 - shift)) * COLLISION_RESOLVES;
}
extern uint64_t num_tables;
extern uint64_t keys_per_thread;
extern uint64_t min_global_guid_thread;
extern uint64_t max_global_guid_thread;

static inline arts_route_table_t *arts_get_route_table(arts_guid_t guid) {
  uint64_t key = ARTS_GUID_GET_KEY(guid);
  if (keys_per_thread) {
    uint64_t global_thread = (key / keys_per_thread);
    if (min_global_guid_thread <= global_thread &&
        global_thread < max_global_guid_thread) {
      return arts_node_info.route_table[global_thread - min_global_guid_thread];
    }
  }
  return arts_node_info
      .remote_route_table[key & (ARTS_REMOTE_ROUTE_SHARDS - 1)];
}

/* ── Per-kind deleter dispatch (registration) ───────────────────────────────
 * Install wraps the object in a shared cb; the cb's deleter is chosen by GUID
 * kind so create call sites keep the (obj, key, ...) signature.  Each object
 * type REGISTERS its deleter here at startup (a constructor in db.c / edt.c /
 * sync/event.c / sync/epoch.c).  route_table does NOT reference the per-type
 * deleter symbols by name — that would create a backward cross-object-library
 * link dependency (arts_gas → arts_memory/arts_compute) that breaks the CUDA
 * lib's separate-object-library structure.  Registration decouples it: each
 * deleter pointer is published into this table from the deleter's own TU. */
typedef void (*arts_deleter_fn)(void *);
static arts_deleter_fn g_deleter_by_kind[ARTS_GUID_LAST];

void arts_route_table_register_deleter(arts_guid_kind_t kind,
                                       void (*deleter)(void *)) {
  if ((unsigned int)kind < (unsigned int)ARTS_GUID_LAST) {
    g_deleter_by_kind[kind] = deleter;
  }
}

static inline void (*deleter_for_kind(arts_guid_kind_t k))(void *) {
  return ((unsigned int)k < (unsigned int)ARTS_GUID_LAST) ? g_deleter_by_kind[k]
                                                          : NULL;
}

arts_route_table_t *arts_new_route_table(unsigned int route_table_size,
                                         unsigned int shift) {
  arts_route_table_t *route_table =
      (arts_route_table_t *)arts_calloc(1, sizeof(arts_route_table_t));
  route_table->data = (arts_route_item_t *)arts_calloc_align(
      (size_t)COLLISION_RESOLVES * route_table_size, sizeof(arts_route_item_t),
      16);
  route_table->size = route_table_size;
  route_table->shift = shift;
  route_table->newFunc = arts_new_route_table;
  /* The per-slot OoO list is a Treiber stack (LIFO, single head pointer); it
   * is zero-initializable (an empty head), so this loop is a clarity no-op on
   * calloc'd storage.  Slot reuse across the table's lifetime is fine because
   * the list is fully drained by destroy/cleanup paths before any new push
   * could land. */
  uint64_t total_slots = (uint64_t)COLLISION_RESOLVES * route_table_size;
  for (uint64_t i = 0; i < total_slots; i++) {
    arts_lf_stack_init(&route_table->data[i].ooo_list);
  }
  return route_table;
}

/* Slot is empty when key == 0 (ARTS GUIDs never have key value 0).  Once
 * claimed, slot is permanent for that key. */
arts_route_item_t *
arts_route_table_search_for_key(arts_route_table_t *route_table,
                                arts_guid_t key) {
  arts_route_table_t *current = route_table;
  arts_route_table_t *next;
  uint64_t key_val;
  while (current) {
    key_val = get_route_table_key((uint64_t)key, current->shift);
    for (int i = 0; i < COLLISION_RESOLVES; i++) {
      arts_guid_t slot_key =
          __atomic_load_n(&current->data[key_val].key, __ATOMIC_ACQUIRE);
      if (slot_key == key) {
        return &current->data[key_val];
      }
      key_val++;
    }
    next = __atomic_load_n(&current->next, __ATOMIC_ACQUIRE);
    current = next;
  }
  return NULL;
}

/* Linearly scan for an empty slot (key == 0) and atomically claim it for
 * `key` via CAS.  Concurrent reservers race here directly (no per-GUID
 * lock) -- the key-CAS is the single point of serialization.  Accepting a
 * slot whose key already equals `key` makes two concurrent reservers agree
 * on the same canonical slot (no orphan-slot leak on the common path). */
arts_route_item_t *
arts_route_table_search_for_empty(arts_route_table_t *route_table,
                                  arts_guid_t key, bool mark_used) {
  (void)mark_used;
  arts_route_table_t *current = route_table;
  arts_route_table_t *next;
  uint64_t key_val;
  while (current != NULL) {
    key_val = get_route_table_key((uint64_t)key, current->shift);
    for (int i = 0; i < COLLISION_RESOLVES; i++) {
      arts_guid_t expected = (arts_guid_t)0;
      if (__atomic_compare_exchange_n(&current->data[key_val].key, &expected,
                                      key, false, __ATOMIC_ACQ_REL,
                                      __ATOMIC_ACQUIRE)) {
        return &current->data[key_val];
      }
      if (expected == key) {
        return &current->data[key_val];
      }
      key_val++;
    }

    next = __atomic_load_n(&current->next, __ATOMIC_ACQUIRE);
    if (!next) {
      /* Lazy-init the next (larger) segment with a single CAS.  The chain
       * only ever grows — a published segment is never unlinked before
       * teardown — so NULL->segment is monotonic and ABA-free.  A thread
       * that loses the CAS frees its spare and adopts the winner's; the
       * spare was never published, so no other thread inserted into it and
       * freeing its data + struct is leak-free. */
      arts_route_table_t *fresh =
          current->newFunc(2 * current->size, current->shift + 1);
      arts_route_table_t *expected = NULL;
      if (__atomic_compare_exchange_n(&current->next, &expected, fresh, false,
                                      __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        next = fresh;
      } else {
        arts_free(fresh->data);
        arts_free(fresh);
        next = expected;
      }
    }
    current = next;
  }
  ARTS_ERROR("Route table search failed: impossible state (table=%p)",
             (void *)route_table);
}

/* Reserve a slot for `key` (or look it up if already present).  Strictly
 * lock-free: no per-GUID spinlock.  On return, *out points to the canonical
 * slot for `key`.  The cb (value) may be NULL. */
void arts_route_table_reserve_or_lookup(arts_guid_t key,
                                        arts_route_item_t **out) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item != NULL) {
    *out = item;
    return;
  }
  arts_route_item_t *claimed =
      arts_route_table_search_for_empty(route_table, key, /*mark_used*/ false);
  item = arts_route_table_search_for_key(route_table, key);
  if (item != NULL) {
    *out = item;
    return;
  }
  (void)claimed;
  arts_route_table_reserve_or_lookup(key, out);
}

/* ── cb-based lifecycle ─────────────────────────────────────────────────── */

void *arts_route_table_lookup_data(arts_guid_t key) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return NULL;
  }
  /* Unsafe peek: take a ref, read the object, drop the ref.  Valid only when
   * the caller has an external liveness guarantee (same contract as before). */
  arts_shared_ptr_t h = arts_atomic_shared_load(&item->value);
  if (!h) {
    return NULL;
  }
  void *obj = arts_shared_get(h);
  arts_shared_release(&h);
  return obj;
}

int arts_route_table_lookup_rank(arts_guid_t key) {
  return (int)arts_guid_get_rank(key);
}

void *arts_route_item_peek_data(arts_route_item_t *item) {
  if (item == NULL) {
    return NULL;
  }
  arts_shared_ptr_t h = arts_atomic_shared_load(&item->value);
  if (!h) {
    return NULL;
  }
  void *obj = arts_shared_get(h);
  arts_shared_release(&h);
  return obj;
}

bool arts_route_item_install_data(arts_route_item_t *item, void *obj,
                                  void (*deleter)(void *)) {
  arts_shared_ptr_t cur = arts_atomic_shared_load(&item->value);
  if (cur) {
    arts_shared_release(&cur);
    return false;
  }
  arts_shared_ptr_t cb = arts_shared_make(obj, deleter);
  arts_shared_ptr_t expected = NULL;
  if (atomic_compare_exchange_strong_explicit(&item->value, &expected, cb,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
    return true;
  }
  /* Lost the install race — abandon our cb (object stays the caller's). */
  arts_shared_abandon(&cb);
  return false;
}

/* Unconditional install: wrap obj in a cb (deleter by kind) and atomic_exchange
 * it into the slot, whether the slot was empty or occupied.  A displaced
 * (stale) cb is released — its deleter runs once the last reader ref also drops
 * (deferred free, never UAF), so a labeled-GUID reuse safely REPLACES the prior
 * generation without an explicit destroy.  Drains the slot's OoO list against
 * the freshly installed item.  Returns `obj`.  For the CHECK / rendezvous case
 * where a second creator must NOT overwrite the first, use the fail-if-exists
 * variant arts_route_table_install_if_absent instead. */
void *arts_route_table_install(void *obj, arts_guid_t key, unsigned int rank,
                               bool used) {
  (void)rank;
  (void)used;
  arts_route_item_t *item;
  arts_route_table_reserve_or_lookup(key, &item);
  arts_shared_ptr_t cb =
      arts_shared_make(obj, deleter_for_kind(arts_guid_get_kind(key)));
  arts_shared_ptr_t old = arts_atomic_shared_exchange(&item->value, cb);
  if (old) {
    arts_shared_release(&old);
  }
  arts_ooo_drain(item);
  return obj;
}

void *arts_route_table_install_with_deleter(void *obj, arts_guid_t key,
                                            void (*deleter)(void *)) {
  arts_route_item_t *item;
  arts_route_table_reserve_or_lookup(key, &item);
  arts_shared_ptr_t cb = arts_shared_make(obj, deleter);
  arts_shared_ptr_t old = arts_atomic_shared_exchange(&item->value, cb);
  if (old) {
    arts_shared_release(&old);
  }
  arts_ooo_drain(item);
  return obj;
}

/* Race install: returns true only if this caller CAS'd its cb into the slot. */
bool arts_route_table_install_if_absent(void *obj, arts_guid_t key,
                                        unsigned int rank, bool used) {
  (void)rank;
  (void)used;
  arts_route_item_t *item;
  arts_route_table_reserve_or_lookup(key, &item);
  arts_shared_ptr_t cb =
      arts_shared_make(obj, deleter_for_kind(arts_guid_get_kind(key)));
  arts_shared_ptr_t expected = NULL;
  if (atomic_compare_exchange_strong_explicit(&item->value, &expected, cb,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
    arts_ooo_drain(item);
    return true;
  }
  /* Lost the install race — abandon our cb; the object stays the caller's
   * (insert-or-fail: the loser still owns its object). */
  arts_shared_abandon(&cb);
  return false;
}

/* Destroy: detach the cb and drop the install ref.  Single-flight — only the
 * caller whose exchange observes a non-NULL cb "won".  The object's deleter
 * runs once the last reader ref also drops, so a destroy concurrent with an
 * in-flight lookup is a deferred free, never a use-after-free.  Idempotent. */
bool arts_route_table_mark_delete(arts_guid_t key) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return false;
  }
  arts_shared_ptr_t old = arts_atomic_shared_exchange(&item->value, NULL);
  if (old) {
    arts_shared_release(&old);
    return true;
  }
  return false;
}

/* ── Typed handle lookups (caller-owned ref) ────────────────────────────── */

static inline arts_shared_ptr_t
arts_route_table_lookup_typed(arts_guid_t guid, arts_guid_kind_t expected) {
  if (arts_guid_get_kind(guid) != expected) {
    return NULL;
  }
  arts_route_table_t *route_table = arts_get_route_table(guid);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, guid);
  if (item == NULL) {
    return NULL;
  }
  return arts_atomic_shared_load(&item->value);
}

arts_shared_ptr_t arts_route_table_lookup(arts_guid_t key) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return NULL;
  }
  return arts_atomic_shared_load(&item->value);
}

arts_shared_ptr_t arts_route_table_lookup_event(arts_guid_t guid) {
  return arts_route_table_lookup_typed(guid, ARTS_GUID_EVENT);
}

arts_shared_ptr_t arts_route_table_lookup_db(arts_guid_t guid) {
  return arts_route_table_lookup_typed(guid, ARTS_GUID_DB);
}

arts_shared_ptr_t arts_route_table_lookup_edt(arts_guid_t guid) {
  return arts_route_table_lookup_typed(guid, ARTS_GUID_EDT);
}

arts_shared_ptr_t arts_route_table_lookup_epoch(arts_guid_t guid) {
  return arts_route_table_lookup_typed(guid, ARTS_GUID_EPOCH);
}

bool arts_route_table_move_item(arts_guid_t old_key, arts_guid_t new_key) {
  arts_route_item_t *new_item;
  arts_route_table_reserve_or_lookup(new_key, &new_item);
  arts_route_table_t *old_rt = arts_get_route_table(old_key);
  arts_route_item_t *old_item =
      arts_route_table_search_for_key(old_rt, old_key);
  if (old_item == NULL) {
    return false;
  }
  /* Take the install ref out of the old slot (single-flight; the cb pointer
   * carries the existing strong count unchanged). */
  arts_shared_ptr_t cb = arts_atomic_shared_exchange(&old_item->value, NULL);
  if (!cb) {
    return false;
  }
  arts_shared_ptr_t expected = NULL;
  if (atomic_compare_exchange_strong_explicit(&new_item->value, &expected, cb,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
    arts_ooo_drain(new_item);
    return true;
  }
  /* new_key already occupied (unexpected for a fresh rename target): drop the
   * moved install ref rather than leak it. */
  arts_shared_release(&cb);
  return false;
}

void arts_reset_route_table_iterator(arts_route_table_iterator_t *iter,
                                     arts_route_table_t *table) {
  iter->table = table;
  iter->index = 0;
}

arts_route_item_t *arts_route_table_iterate(arts_route_table_iterator_t *iter) {
  arts_route_table_t *current = iter->table;
  arts_route_table_t *next;
  while (current != NULL) {
    for (uint64_t i = iter->index;
         i < (uint64_t)current->size * COLLISION_RESOLVES; i++) {
      arts_guid_t slot_key =
          __atomic_load_n(&current->data[i].key, __ATOMIC_ACQUIRE);
      if (slot_key != 0) {
        iter->index = i + 1;
        iter->table = current;
        return &current->data[i];
      }
    }
    iter->index = 0;
    next = __atomic_load_n(&current->next, __ATOMIC_ACQUIRE);
    current = next;
  }
  return NULL;
}

void arts_print_item(arts_route_item_t *item) {
  if (item) {
    arts_shared_ptr_t h = arts_atomic_shared_load(&item->value);
    void *data = h ? arts_shared_get(h) : NULL;
    ARTS_INFO("GUID: %lu DATA: %p RANK: %u", item->key, data,
              arts_guid_get_rank(item->key));
    if (h) {
      arts_shared_release(&h);
    }
  }
}

uint64_t arts_clean_up_route_table(arts_route_table_t *route_table) {
  arts_route_table_iterator_t iter;
  arts_reset_route_table_iterator(&iter, route_table);

  arts_route_item_t *item = arts_route_table_iterate(&iter);
  while (item) {
    /* Detach + release the install ref; the cb's per-kind deleter performs
     * the single teardown once the last ref drops.  No per-type special
     * casing — the deleter is the sole owner. */
    arts_shared_ptr_t old = arts_atomic_shared_exchange(&item->value, NULL);
    if (old) {
      arts_shared_release(&old);
    }
    arts_ooo_free_all(item);
    item = arts_route_table_iterate(&iter);
  }
  return 0;
}

void arts_delete_route_table(arts_route_table_t *route_table) {
  if (!route_table) {
    return;
  }
  arts_delete_route_table(route_table->next);
  for (uint64_t i = 0; i < (uint64_t)route_table->size * COLLISION_RESOLVES;
       i++) {
    arts_ooo_free_all(&route_table->data[i]);
  }
  arts_free(route_table->data);
  arts_free(route_table);
}

void arts_clean_up_dbs() {
  uint64_t free_size = 0;
  for (unsigned int i = 0; i < arts_node_info.total_thread_count; i++) {
    free_size += arts_clean_up_route_table(arts_node_info.route_table[i]);
  }
  for (int s = 0; s < ARTS_REMOTE_ROUTE_SHARDS; s++) {
    free_size +=
        arts_clean_up_route_table(arts_node_info.remote_route_table[s]);
  }
  ARTS_INFO("Cleaned %lu bytes", free_size);
}

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

/* EDT satisfy handlers live in edt.c (arts_handler_edt_satisfy_slot[_ptr]) —
 * pure cores that write the acquired EDT's dep slot. */

/* Event satisfy / add-dependence handlers live in event.c
 * (arts_handler_event_satisfy_slot / arts_handler_event_add_dependence) —
 * pure cores that operate on the acquired event. */

/*
 * arts_ooo_resolve_db_dep — Used by the OoO replay path
 * (ooo_h_db_acquire) when a local DB referenced by an EDT dependency
 * arrives in the route table after the EDT was registered.  Fills the
 * EDT's dep slot with the freshly-installed DB pointer and drops one
 * depc_needed.  For ARTS_DB types the RC acquire path replaces this; for
 * non-RC pinned types the OoO replay covers the local-create-after-consumer
 * race.
 */
void arts_ooo_resolve_db_dep(struct arts_edt_s *edt, unsigned int slot,
                             struct arts_db_s *db_res) {
  arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
  if (db_res == NULL) {
    /* DB was destroyed between the OoO defer and this drain (DELETE_ITEM
     * race).  Treat this slot as a NULL dependency — the data is gone — and
     * advance the resume index past the dead dep. */
    ARTS_WARN("arts_ooo_resolve_db_dep: db_res is NULL for EDT[Guid:%lu] "
              "slot=%u (DB destroyed during OO resolution)",
              edt->guid, slot);
    depv[slot].guid = NULL_GUID;
    depv[slot].ptr = NULL;
    edt->resume_k++;
  }
  /* The DB now exists locally (db_res != NULL) — re-enter the strict
   * sequential acquire walk, which re-attempts this frontier dep through the
   * coherence acquire (proper writer_count / buffer-ref handling) and then
   * continues, scheduling the EDT once all deps are held.  On the destroy
   * branch above, resume_k already advanced past the dead dep. */
  arts_db_acquire_all(edt);
}

static void ooo_h_handle_ready_edt(void *item, void *vargs) {
  (void)item;
  struct arts_ooo_args_handle_ready_s *a = vargs;
  arts_handle_ready_edt(a->edt);
}

static void ooo_h_db_acquire(void *item, void *vargs) {
  struct arts_ooo_args_db_acquire_s *a = vargs;
  arts_ooo_resolve_db_dep(a->edt, a->slot, (struct arts_db_s *)item);
}

/* Epoch handlers (inc_*, request, send) live in epoch.c as pure cores on the
 * acquired epoch — no re-issue wrappers here. */

static void ooo_h_db_ownership_request(void *item, void *vargs) {
  (void)item;
  struct arts_ooo_args_db_ownership_request_s *a = vargs;
  struct arts_remote_ownership_request_packet_s p;
  p.header.rank = a->requester;
  p.db_guid = a->db_guid;
  arts_handler_db_ownership_request(&p);
}

static void ooo_h_db_snapshot_request(void *item, void *vargs) {
  (void)item;
  struct arts_ooo_args_db_snapshot_request_s *a = vargs;
  struct arts_remote_snapshot_request_packet_s p;
  p.header.rank = a->requester;
  p.db_guid = a->db_guid;
  p.edt_guid = a->edt_guid;
  p.slot = a->slot;
  arts_handler_db_snapshot_request(&p);
}

static void ooo_h_db_destroy(void *item, void *vargs) {
  (void)item;
  struct arts_ooo_args_db_destroy_s *a = vargs;
  struct arts_remote_destroy_packet_s p;
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

/* Event destroy replay: the event is now installed (drain runs post-install),
 * so perform the actual mark_delete.  item is pinned by dispatch_or_defer. */
static void ooo_h_event_destroy(void *item, void *vargs) {
  (void)item;
  struct arts_ooo_args_event_destroy_s *a = vargs;
  arts_route_table_mark_delete(a->guid);
}

/* EDT destroy replay: re-issue the home wire handler; the EDT is installed
 * (drain post-install) so its lookup HITs and runs the destroy body inline. */
static void ooo_h_edt_destroy(void *item, void *vargs) {
  (void)item;
  struct arts_ooo_args_edt_destroy_s *a = vargs;
  struct arts_remote_guid_only_packet_s p;
  p.guid = a->guid;
  arts_handler_edt_destroy(&p);
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
    [OOO_EVENT_DESTROY] = ooo_h_event_destroy,
    [OOO_EDT_DESTROY] = ooo_h_edt_destroy,
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
