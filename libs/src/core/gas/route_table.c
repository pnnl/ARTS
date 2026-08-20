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
#include "arts/gas/guid.h"
#include "arts/ooo.h"           /* arts_ooo_drain / arts_ooo_redrive_all */
#include "arts/runtime_state.h" /* arts_node_info */
#include "arts/system/print.h"
#include "arts/utils/lockfree_lifo.h"
#include "arts/utils/malloc.h"
#include "arts/utils/shared.h"

#define HASH64(x, y) ((uint64_t)(x) * (y))

static inline uint64_t get_route_table_key(uint64_t x, unsigned int shift) {
  uint64_t hash = 14695981039346656037U;
  switch (shift) {
  case 10:
    hash *= 1021;
    /* fall through */
  case 11:
    hash *= 2039;
    /* fall through */
  case 12:
    hash *= 4093;
    /* fall through */
  case 13:
    hash *= 8191;
    /* fall through */
  case 14:
    hash *= 16381;
    /* fall through */
  case 15:
    hash *= 32749;
    /* fall through */
  case 16:
    hash *= 65521;
    /* fall through */
  case 17:
    hash *= 131071;
    /* fall through */
  case 18:
    hash *= 262139;
    /* fall through */
  case 19:
    hash *= 524287;
    /* fall through */
  case 20:
    hash *= 1048573;
    /* fall through */
  case 21:
    hash *= 2097143;
    /* fall through */
  case 22:
    hash *= 4194301;
    /* fall through */
  case 31:
    hash *= 2147483647;
    /* fall through */
  case 32:
    hash *= 4294967291;
    /* fall through */
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
  if (ARTS_GUID_GET_TYPE(guid) == ARTS_GUID_DB && arts_db_seq_budget) {
    /* DB keys are [szhint | seq]; the creating rank is arithmetic on seq
     * (slice width = arts_db_seq_budget), so the local/remote decision is a
     * pure function of the key — identical on every thread, install and
     * lookup alike.  Self-created keys (reserved-range members included:
     * both reserve paths claim from this rank's own slice) spread
     * chunk-granular over the local per-thread tables; foreign creators and
     * the startup top region sit outside the slice and take the shared
     * remote shards, exactly as they did under the flat-key partition.
     * While arts_db_seq_budget is still 0 (the pre-parallel window) every
     * DB key falls through to the legacy path, whose keys_per_thread gate
     * is also still 0 — same remote-shard answer, so the decision for any
     * key is stable across the flip. */
    uint64_t seq = key & ARTS_GUID_DB_SEQ_MASK;
    if (seq / arts_db_seq_budget == arts_global_rank_id) {
      return arts_node_info
          .route_table[(seq >> ARTS_GUID_DB_CHUNK_BITS) % num_tables];
    }
    return arts_node_info
        .remote_route_table[key & (ARTS_REMOTE_ROUTE_SHARDS - 1)];
  }
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
 * sync/event.c).  route_table does NOT reference the per-type
 * deleter symbols by name — that would create a backward cross-object-library
 * link dependency (arts_gas → arts_memory/arts_compute) that breaks the CUDA
 * lib's separate-object-library structure.  Registration decouples it: each
 * deleter pointer is published into this table from the deleter's own TU. */
typedef void (*arts_deleter_fn_t)(void *);
static arts_deleter_fn_t g_deleter_by_kind[ARTS_GUID_LAST];

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
  route_table->data = (arts_route_item_t *)arts_calloc_aligned(
      (size_t)COLLISION_RESOLVES * route_table_size, sizeof(arts_route_item_t),
      ARTS_CACHE_LINE_SIZE);
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
/* One slot per key used to follow from monotone occupancy: a bucket only ever
 * filled, so the first free slot in the probe order was a stable choice.  Slots
 * are returned now, so a thread stalled in a later segment can claim a second
 * slot for a key another thread just claimed a freed slot for.  Uniqueness
 * therefore rests on the re-search below, not on the probe order: the loser's
 * slot is an orphan holding the key with no value, invisible to lookups, and
 * its parked payloads are re-driven when it is torn down. */
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

int arts_route_table_lookup_rank(arts_guid_t key) {
  return (int)arts_guid_get_rank(key);
}

/* Unbracketed on purpose: this takes a slot the CALLER located, in a mirror
 * table (the GPU per-device tables) that no destroy returns.  Its slots are
 * bound to their key for the table's life, so there is no identity to lose.  Do
 * not point it at the global table, whose slots are reclaimed. */
arts_shared_ptr_t arts_route_item_acquire(arts_route_item_t *item) {
  return item ? arts_atomic_shared_load(&item->value) : NULL;
}

bool arts_route_item_install_data(arts_route_item_t *item, void *obj,
                                  void (*deleter)(void *)) {
  arts_shared_ptr_t cur = arts_atomic_shared_load(&item->value);
  if (cur) {
    arts_shared_release(&cur);
    return false;
  }
  arts_shared_ptr_t cb = arts_shared_make(obj, deleter);
  if (arts_atomic_shared_compare_exchange(&item->value, NULL, cb)) {
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
  arts_shared_set_tag(cb, (uint64_t)key);
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
  arts_shared_set_tag(cb, (uint64_t)key);
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
  arts_shared_set_tag(cb, (uint64_t)key);
  if (arts_atomic_shared_compare_exchange(&item->value, NULL, cb)) {
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
bool arts_route_table_set_destroyed(arts_guid_t key) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return false;
  }
  /* A slot holding no value is a RESERVATION — somebody claimed the key and
   * parked deferred payloads on it waiting for a create that has not landed.
   * It is not destroyable. */
  arts_shared_ptr_t peek = arts_atomic_shared_load(&item->value);
  if (peek == NULL) {
    return false;
  }
  arts_shared_release(&peek);
  /* One CAS, carrying both halves.  It can only succeed while the slot still
   * names this GUID, so the identity is proven by the transition rather than
   * checked beside it; and only one caller can win it, so the teardown is
   * single-flight without anything being held.  A loser is not the destroyer
   * and says so — no retry, because there is no state to wait for. */
  arts_guid_t expect = key;
  if (!__atomic_compare_exchange_n(&item->key, &expect,
                                   ARTS_ROUTE_KEY_RETIRING, false,
                                   __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    return false;
  }
  arts_shared_ptr_t old = arts_atomic_shared_exchange(&item->value, NULL);
  bool detached = (old != NULL);
  if (old) {
    arts_shared_release(&old);
  }
  /* Return the slot, THEN look for stragglers.  A deferring thread pushes its
   * node and then, behind a full fence, re-reads this key; a teardown publishes
   * the return and then, behind a full fence, re-reads the chain.  Each stores
   * its word before loading the other's, so they cannot both miss — and
   * whatever the teardown finds it re-drives onto whichever slot owns that GUID
   * now. */
  __atomic_store_n(&item->key, (arts_guid_t)0, __ATOMIC_RELEASE);
  atomic_thread_fence(memory_order_seq_cst);
  arts_ooo_redrive_all(item);
  return detached;
}


/* Acquire a slot's object, verified by the OBJECT's identity.
 *
 * search_for_key proves the slot held `key` BEFORE the value load; a slot is
 * returned on destroy and re-claimed by another GUID, so that alone no longer
 * proves the value we loaded is the one we asked for — the object could belong
 * to a different GUID, and because DB, EVENT and EDT keys share these tables it
 * could be a different KIND, which a caller would then use through the wrong
 * struct.  Re-reading the slot's key word is not a proof either: the word is
 * not monotone (claimed, retired, returned, re-claimed), and a late message
 * for a destroyed GUID legally re-reserves that GUID into a freed slot — so
 * the word can return to a previously observed key while the value belongs to
 * an identity that held the slot in between.  The pinned cb's own tag cannot:
 * it is stamped with the key the object was published under and the ref taken
 * by the load keeps that cb from being recycled, so tag == key is a property
 * of the object in hand, not of a word that may have changed twice since. */
static inline arts_shared_ptr_t
route_item_acquire_checked(arts_route_item_t *item, arts_guid_t key) {
  arts_shared_ptr_t h = arts_atomic_shared_load(&item->value);
  if (h != NULL && arts_shared_tag(h) != (uint64_t)key) {
    arts_shared_release(&h);
    return NULL;
  }
  return h;
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
  return route_item_acquire_checked(item, guid);
}

arts_shared_ptr_t arts_route_table_lookup(arts_guid_t key) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return NULL;
  }
  return route_item_acquire_checked(item, key);
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

bool arts_route_table_move_item(arts_guid_t old_key, arts_guid_t new_key) {
  arts_route_item_t *new_item;
  arts_route_table_reserve_or_lookup(new_key, &new_item);
  arts_route_table_t *old_rt = arts_get_route_table(old_key);
  arts_route_item_t *old_item =
      arts_route_table_search_for_key(old_rt, old_key);
  if (old_item == NULL) {
    return false;
  }
  /* Take the install ref out of the old slot under the same claim a destroy
   * uses, and re-check the identity beneath it.  A bare exchange here would
   * detach whatever the slot holds — and since slots are returned and
   * re-claimed, that can be a live stranger's object, which this would then
   * re-key under new_key. */
  arts_guid_t expect_old = old_key;
  if (!__atomic_compare_exchange_n(&old_item->key, &expect_old,
                                   ARTS_ROUTE_KEY_RETIRING, false,
                                   __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    return false;
  }
  arts_shared_ptr_t cb = arts_atomic_shared_exchange(&old_item->value, NULL);
  /* Same return protocol as set_destroyed: publish the return, fence, then
   * re-drive whatever raced onto the old slot's chain — a node parked against
   * the fence pairing must be moved to wherever its own GUID lives now, not
   * stranded on a slot another identity is about to claim. */
  __atomic_store_n(&old_item->key, (arts_guid_t)0, __ATOMIC_RELEASE);
  atomic_thread_fence(memory_order_seq_cst);
  arts_ooo_redrive_all(old_item);
  if (!cb) {
    return false;
  }
  /* Re-stamp while the cb is in no slot: readers verify a pinned value by its
   * tag, so a moved object must carry the key it is about to answer for.  A
   * stale handle from the old slot racing this read sees one of the two keys
   * — old fails its lookup's compare, new is simply the rename completed. */
  arts_shared_set_tag(cb, (uint64_t)new_key);
  if (arts_atomic_shared_compare_exchange(&new_item->value, NULL, cb)) {
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
