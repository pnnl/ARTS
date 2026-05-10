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
#include "arts/gas/out_of_order.h"
#include "arts/gas/out_of_order_list.h"
#include "arts/memory/db.h"
#include "arts/runtime_state.h"
#include "arts/sync/event.h"
#include "arts/sync/shared.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

#define INIT_INVALIDATE_SIZE 128

uint64_t urand64() {
  uint64_t hi = lrand48();
  uint64_t md = lrand48();
  uint64_t lo = lrand48();
  uint64_t res = (hi << 42) + (md << 21) + lo;
  return res;
}

#define HASH64(x, y) ((uint64_t)(x) * (y))

static inline uint64_t get_route_table_key(uint64_t x, unsigned int shift) {
  uint64_t hash = 14695981039346656037U;
  switch (shift) {
  /*case 5:
      hash *= 31;
  case 6:
      hash *= 61;
  case 7:
      hash *= 127;
  case 8:
      hash *= 251;
  case 9:
      hash *= 509;*/
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
extern uint64_t max_guid;
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
  /* Vyukov MPSC OO list cannot be zero-initialized (head/tail must point
   * at the embedded stub).  Initialize every slot up front; slot reuse
   * across the table's lifetime is fine because the list is fully drained
   * by destroy/cleanup paths before any new push could land. */
  uint64_t total_slots = (uint64_t)COLLISION_RESOLVES * route_table_size;
  for (uint64_t i = 0; i < total_slots; i++) {
    arts_oo_list_init(&route_table->data[i].ooList);
  }
  return route_table;
}

/* Slot is empty when key == 0 (ARTS GUIDs never have key value 0).  This is
 * the new model: no lock bitfield, no reserved/available state -- only "key
 * claimed or not".  Once claimed, slot is permanent for that key. */
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
    arts_reader_lock(&current->readerLock, &current->writerLock);
    next = current->next;
    arts_reader_unlock(&current->readerLock);
    current = next;
  }
  return NULL;
}

/* Linearly scan for an empty slot (key == 0) and atomically claim it for
 * `key` via CAS.  Concurrent reservers race here directly (no per-GUID
 * lock) -- the key-CAS is the single point of serialization.
 *
 * Task 4q (lock-free reserve_or_lookup) preserves the "exactly one slot
 * per GUID" invariant by also accepting a slot whose key already equals
 * `key`.  Without this, two concurrent reservers scanning the same
 * collision range could each CAS a different empty slot to `key` (T1
 * claims slot K, T2's CAS at K fails so T2 moves to K+1 and CAS'd that
 * slot to `key`).  The added "expected == key" early-return turns T2's
 * failed CAS at slot K into a successful "found" — both threads return
 * the same slot K, no orphan slot leak. */
arts_route_item_t *
arts_route_table_search_for_empty(arts_route_table_t *route_table,
                                  arts_guid_t key, bool mark_used) {
  (void)mark_used; /* legacy: caller used to request "available + 1 ref" */
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
      /* CAS failed: `expected` now holds the slot's actual key.  If
       * another reserver just claimed THIS slot for OUR key, treat it
       * as a found-and-shared slot (lock-free agreement on canonical
       * slot — see header comment). */
      if (expected == key) {
        return &current->data[key_val];
      }
      key_val++;
    }

    arts_reader_lock(&current->readerLock, &current->writerLock);
    next = current->next;
    arts_reader_unlock(&current->readerLock);

    if (!next) {
      if (arts_writer_try_lock(&current->readerLock, &current->writerLock)) {
        next = current->next =
            current->newFunc(2 * current->size, current->shift + 1);
        arts_writer_unlock(&current->writerLock);
      } else {
        arts_reader_lock(&current->readerLock, &current->writerLock);
        next = current->next;
        arts_reader_unlock(&current->readerLock);
      }
    }
    current = next;
  }
  ARTS_ERROR("Route table search failed: impossible state (table=%p)",
             (void *)route_table);
}

/* Reserve a slot for `key` (or look it up if already present).  Strictly
 * lock-free: no per-GUID spinlock.  On return, *out points to the
 * canonical slot for `key` (the earliest slot whose key has been CAS'd
 * to `key` in scan order).  Data may be NULL.
 *
 * Algorithm (Task 4q — option (a) "insert-then-canonical-search"):
 *   1. search_for_key(g): cheap path; canonical slot may already exist.
 *   2. search_for_empty(g): atomic CAS slot.key 0->g.  Note that two
 *      reservers may BOTH succeed -- they may CAS different slots in
 *      the same scan range (T1 claims slot K, T2's CAS at K fails, T2
 *      moves to K+1 which is also empty and claims that).  This is OK
 *      because both slots have key=g and search_for_key always returns
 *      the FIRST match in deterministic scan order.  The "loser" slot
 *      becomes an orphan -- harmless because no caller will observe it
 *      (search_for_key skips past it once it finds the earlier match).
 *   3. Always re-run search_for_key after step 2 so every caller
 *      returns the same canonical slot pointer.  The slot we just
 *      CAS'd may not be the canonical one if a concurrent reserver
 *      won at an earlier index.
 *
 * Note: Each call's step-2 CAS may consume one slot in the worst case
 * (orphan).  Since the table never shrinks and grows on demand, this
 * is a bounded space cost per concurrent installer.  Real-world
 * concurrent installs on the same GUID are rare (the EDT graph nearly
 * always has a single creator per GUID); the orphans only occur on
 * actual races, not on the common path. */
void arts_route_table_reserve_or_lookup(arts_guid_t key,
                                        arts_route_item_t **out) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  /* Step 1: cheap existing-slot check. */
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item != NULL) {
    *out = item;
    return;
  }
  /* Step 2: claim a free slot via key-CAS (atomic; may trigger grow). */
  arts_route_item_t *claimed =
      arts_route_table_search_for_empty(route_table, key, /*mark_used*/ false);
  /* Step 3: ALWAYS re-search to find the canonical (earliest) slot for
   * `key` so every concurrent reserver agrees on the same slot pointer.
   * If our claim in step 2 was the earliest, we'll see it; otherwise
   * we'll see a winner's earlier slot and our claim becomes an orphan
   * (untouched, never observed by future search_for_key). */
  item = arts_route_table_search_for_key(route_table, key);
  if (item != NULL) {
    *out = item;
    return;
  }
  /* Defensive: search_for_key still NULL after a successful step-2
   * claim is impossible unless the table grew between our claim and
   * the re-search and the new sub-table was added but our step-2
   * walked an old chain.  Recurse to walk the full chain again. */
  (void)claimed;
  arts_route_table_reserve_or_lookup(key, out);
}

void *arts_route_table_lookup_data(arts_guid_t key) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return NULL;
  }
  return atomic_load_explicit(&item->data, memory_order_acquire);
}

void *arts_route_table_lookup_item(arts_guid_t key) {
  return arts_route_table_lookup_data(key);
}

int arts_route_table_lookup_rank(arts_guid_t key) {
  /* rank is now derivable directly from the GUID -- no need to consult the
   * route table. */
  return (int)arts_guid_get_rank(key);
}

int arts_route_table_set_rank(arts_guid_t key, int rank) {
  /* Rank is fixed by GUID encoding; legacy callers that "moved" entries are
   * being phased out in Task 1a.3.  No-op. */
  (void)key;
  (void)rank;
  return -1;
}

/* Helper: install the per-item lock (count = 1 install-existence ref) when
 * an installer wins the data CAS.  Preserves the slot's monotonic gen
 * counter (Task 4e — bumped by free_item on each free cycle).  Caller MUST
 * have just won the data slot for this `item`. */
static inline void arts_route_item_install_lock(arts_route_item_t *item) {
  uint64_t prev = atomic_load_explicit(&item->lock, memory_order_acquire);
  uint32_t cur_gen = ARTS_ROUTE_LOCK_GET_GEN(prev);
  uint64_t fresh = ARTS_ROUTE_LOCK_PACK(0, cur_gen, 1);
  atomic_store_explicit(&item->lock, fresh, memory_order_release);
}

/* Install `data` into the slot for `key` and fire any pending OoO entries.
 * No-op if data is already non-NULL (idempotent). */
void *arts_route_table_add_item(void *data, arts_guid_t key, unsigned int rank,
                                bool used) {
  (void)rank; /* extractable from key */
  (void)used; /* ref_count removed */
  arts_route_item_t *item;
  arts_route_table_reserve_or_lookup(key, &item);
  /* Check whether the data CAS will succeed (slot empty) before clobbering
   * the lock.  Idempotent install path: if data was already non-NULL we
   * leave the existing lock alone. */
  void *prev_data = atomic_load_explicit(&item->data, memory_order_acquire);
  if (prev_data == NULL) {
    arts_route_item_install_lock(item);
    atomic_store_explicit(&item->data, data, memory_order_release);
    /* Fire pending OoO entries -- installer fires after data store. */
    arts_route_table_fire_oo(key, arts_out_of_order_handler);
  }
  return item;
}

/* CAS-install `data` (NULL -> data); returns true only if this caller won.
 * On win, install lock (count=1, preserves gen) and fire pending OoO. */
bool arts_route_table_add_item_race(void *data, arts_guid_t key,
                                    unsigned int rank, bool used) {
  (void)rank;
  (void)used;
  arts_route_item_t *item;
  arts_route_table_reserve_or_lookup(key, &item);
  void *expected = NULL;
  bool installed = atomic_compare_exchange_strong_explicit(
      &item->data, &expected, data, memory_order_release, memory_order_acquire);
  if (installed) {
    arts_route_item_install_lock(item);
    arts_route_table_fire_oo(key, arts_out_of_order_handler);
  }
  return installed;
}

/* Legacy wrapper retained for the few internal call sites that pre-date the
 * `add_item_race` simplification.  used_res / used_avail / to_add_on_creation
 * referred to ref_count behavior that no longer exists. */
arts_route_item_t *internal_route_table_add_item_race(
    bool *added_item, arts_route_table_t *route_table, void *data,
    arts_guid_t key, unsigned int rank, bool used_res, bool used_avail,
    unsigned int to_add_on_creation) {
  (void)route_table;
  (void)rank;
  (void)used_res;
  (void)used_avail;
  (void)to_add_on_creation;
  arts_route_item_t *item;
  arts_route_table_reserve_or_lookup(key, &item);
  void *expected = NULL;
  bool installed = atomic_compare_exchange_strong_explicit(
      &item->data, &expected, data, memory_order_release, memory_order_acquire);
  if (added_item) {
    *added_item = installed;
  }
  if (installed) {
    arts_route_item_install_lock(item);
    arts_route_table_fire_oo(key, arts_out_of_order_handler);
  }
  return item;
}

arts_route_item_t *
internal_route_table_add_deleted_item_race(arts_route_table_t *route_table,
                                           void *data, arts_guid_t key,
                                           unsigned int rank) {
  /* Legacy path used to mark a slot DELETED on creation so subsequent
   * lookups would skip it.  In the new model there is no DELETE bit -- the
   * caller must consult the cache_s/RC state machine instead.  Provide
   * a permissive install so existing callers still link. */
  (void)route_table;
  (void)rank;
  arts_route_item_t *item;
  arts_route_table_reserve_or_lookup(key, &item);
  void *prev_data = atomic_load_explicit(&item->data, memory_order_acquire);
  if (prev_data == NULL) {
    arts_route_item_install_lock(item);
    atomic_store_explicit(&item->data, data, memory_order_release);
  }
  return item;
}

/* OoO-integrated push.  Returns ENQUEUED if data was NULL (push deferred)
 * or AVAILABLE_NOW if data is/became non-NULL during the push. */
oo_add_result_t arts_route_table_add_oo_ex(arts_guid_t key, void *payload) {
  arts_route_item_t *item;
  arts_route_table_reserve_or_lookup(key, &item);

  /* Step 0: destroyed-state check.  Once arts_route_table_set_destroyed
   * has flipped DELETE on the lock, any new OoO add must NOT enqueue --
   * the RC destroy path bypasses the route_table teardown so there is
   * no installer to fire and drain the payload.  Returning AVAILABLE_NOW
   * lets the caller (arts_out_of_order_handle_db_request) take its own
   * inline NULL-callback path, propagating the destroyed-DB semantic
   * (arts_db_request_callback(NULL) wakes the EDT with depv[slot].ptr=NULL,
   * depc_needed--).  Without this, an arts_add_dependence racing
   * against arts_db_destroy would park forever in the OoO list with
   * no future drainer. */
  uint64_t lock = atomic_load_explicit(&item->lock, memory_order_acquire);
  if (ARTS_ROUTE_LOCK_HAS_DELETE(lock)) {
    return OO_RESULT_AVAILABLE_NOW;
  }

  /* Step 1: pre-push data check (fast path). */
  void *cur = atomic_load_explicit(&item->data, memory_order_acquire);
  if (cur != NULL) {
    return OO_RESULT_AVAILABLE_NOW;
  }

  /* Step 2: push to OO list.  MPSC push always succeeds (no
   * DRAIN_HAPPENED race window). */
  (void)arts_oo_list_push(&item->ooList, payload);

  /* Step 3: post-push data recheck (TOCTOU rescue).
   *
   * Installer's data store + fire_oo may have happened in the window
   * between our Step-1 lookup and the push.  In that case our entry is
   * stranded -- the installer's fire already ran and drained whatever was
   * in the list at that moment, but our payload landed afterwards and
   * will sit forever unless we trigger another drain.  Calling fire_oo
   * here drains our entry plus any other late arrivals. */
  cur = atomic_load_explicit(&item->data, memory_order_acquire);
  if (cur != NULL) {
    arts_route_table_fire_oo(key, arts_out_of_order_handler);
    return OO_RESULT_FIRED_BY_DRAIN;
  }
  /* Step 4: post-push destroyed-state recheck.  The destroy may have
   * raced ahead between Step 0 and our push; if so, drain our payload
   * via the destroyed-cb so the EDT does not stall. */
  lock = atomic_load_explicit(&item->lock, memory_order_acquire);
  if (ARTS_ROUTE_LOCK_HAS_DELETE(lock)) {
    arts_oo_list_drain(&item->ooList, arts_oo_dispatch_destroyed_cb, NULL);
    return OO_RESULT_FIRED_BY_DRAIN;
  }
  return OO_RESULT_ENQUEUED;
}

/* Boolean wrapper for legacy 12 OoO callers in out_of_order.c.
 *
 * Returns true when the caller should do nothing (the deferred handler
 * has been or will be invoked elsewhere): ENQUEUED or FIRED_BY_DRAIN.
 *
 * Returns false only on AVAILABLE_NOW, where the payload was never
 * pushed and the caller must invoke the handler inline + free the
 * payload itself (legacy semantics). */
bool arts_route_table_add_oo(arts_guid_t key, void *payload, bool inc) {
  (void)inc; /* ref_count removed */
  return arts_route_table_add_oo_ex(key, payload) != OO_RESULT_AVAILABLE_NOW;
}

bool arts_route_table_add_oo_existing(arts_guid_t key, void *payload,
                                      bool inc) {
  (void)inc;
  return arts_route_table_add_oo_ex(key, payload) != OO_RESULT_AVAILABLE_NOW;
}

void arts_route_table_fire_oo(arts_guid_t key,
                              void (*callback)(void *data, void *ctx)) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return;
  }
  void *data = atomic_load_explicit(&item->data, memory_order_acquire);
  /* Installer always calls fire after storing data, so data is non-NULL. */
  arts_oo_list_drain(&item->ooList, callback, data);
}

void arts_route_table_drop_oo(arts_guid_t key) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return;
  }
  /* Wake EDT waiters before freeing payloads (DB-destroyed semantic).
   * Dispatch is type-aware; arts_oo_dispatch_destroyed_cb knows the
   * payload struct layout and lives in out_of_order.c. */
  arts_oo_list_drain(&item->ooList, arts_oo_dispatch_destroyed_cb, NULL);
}

/* Set the DELETE bit on a route_item WITHOUT decrementing the install
 * reference.  Used by the RC destroy path which manages teardown
 * directly (try_finalize_destroy → arts_db_free) and bypasses the
 * route_table's mark_delete + free_item refcount flow.
 *
 * Setting DELETE causes:
 *   - arts_route_table_acquire_item to fail (lookup_safe returns NULL).
 *   - arts_route_table_add_oo_ex to detect destroyed state and return
 *     OO_RESULT_AVAILABLE_NOW so callers fire NULL-callback inline.
 *
 * This leaves the install ref intact, so the route_item stays in the
 * table.  The cache_s and db_s are freed by try_finalize_destroy via
 * arts_db_free directly; the route_item slot is bounded (route_table is
 * fixed-size pre-allocated).  Idempotent: repeated calls observe DELETE
 * already set and bail. */
void arts_route_table_set_destroyed(arts_guid_t key) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return;
  }
  uint64_t prev = atomic_load_explicit(&item->lock, memory_order_acquire);
  for (;;) {
    if (ARTS_ROUTE_LOCK_HAS_DELETE(prev)) {
      return; /* already destroyed; idempotent. */
    }
    uint64_t next = prev | ARTS_ROUTE_LOCK_DELETE_BIT;
    if (atomic_compare_exchange_weak_explicit(&item->lock, &prev, next,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
      return;
    }
  }
}

/* ---------------------------------------------------------------------------
 * D1 lifecycle API (Task 4e + 4f).  See route_table.h for the full contract.
 * --------------------------------------------------------------------------*/

/* Type-aware feature flag: returns true for every dynamic-object type whose
 * struct embeds ARTS_SHARED_FIELD as its first member.  All
 * four event/DB/EDT/epoch types are migrated, so this function is true for
 * every dynamic-object tag the route_table actually carries.
 *
 * Kept as a function (rather than inlining into arts_route_item_free) so a
 * future regression — e.g. an object stored under a non-migrated tag — is
 * caught by the false branch instead of crashing on a wild deleter ptr. */
static inline bool object_has_shared_field(arts_guid_kind_t t) {
  switch (t) {
  case ARTS_GUID_EVENT: /* arts_event_s embeds shared. */
  case ARTS_GUID_DB:  /* arts_db_s embeds shared. */
  case ARTS_GUID_EDT:   /* arts_edt_s embeds shared. */
  case ARTS_GUID_EPOCH: /* arts_epoch_s embeds shared. */
    return true;
  default:
    return false;
  }
}

/* free_item: invoked by release_item / mark_delete once the slot's lock
 * count reaches 0 with DELETE set.  Dispatches to the per-object deleter
 * via arts_shared_t.  After the free completes, bumps the slot's gen
 * counter and clears DELETE + count.  The OoO list is preserved across
 * the free per the user's mandate — pending OoO ops fire on the next
 * install of the same GUID.
 *
 * legacy_free_by_type was deleted along with the legacy fallback
 * branch — every dynamic-object type now embeds ARTS_SHARED_FIELD, so the
 * dispatcher is unconditional.  The `if (object_has_shared_field(t))`
 * guard is kept as a safety net for any future regression that re-uses
 * the route_table for a non-migrated tag. */
static void arts_route_item_free(arts_route_item_t *item) {
  void *data =
      atomic_exchange_explicit(&item->data, (void *)NULL, memory_order_acq_rel);
  if (data) {
    arts_guid_kind_t t = arts_guid_get_kind(item->key);
    if (object_has_shared_field(t)) {
      arts_shared_t *s = (arts_shared_t *)data;
      s->deleter(data);
    }
  }
  /* Bump gen, clear DELETE + count.  Preserves ooList memory.  Use a CAS
   * loop so any unexpected concurrent acquirer (which would be a bug --
   * count was 0 when free_item entered) cannot clobber the new gen. */
  uint64_t prev = atomic_load_explicit(&item->lock, memory_order_acquire);
  for (;;) {
    uint32_t next_gen =
        (uint32_t)((ARTS_ROUTE_LOCK_GET_GEN(prev) + 1u) & 0x7FFFFFFFu);
    uint64_t fresh = ARTS_ROUTE_LOCK_PACK(0, next_gen, 0);
    if (atomic_compare_exchange_weak_explicit(&item->lock, &prev, fresh,
                                              memory_order_release,
                                              memory_order_acquire)) {
      break;
    }
  }
}

bool arts_route_table_acquire_item(arts_route_item_t *item) {
  if (item == NULL) {
    return false;
  }
  uint64_t prev = atomic_load_explicit(&item->lock, memory_order_acquire);
  for (;;) {
    if (ARTS_ROUTE_LOCK_HAS_DELETE(prev)) {
      return false;
    }
    if (ARTS_ROUTE_LOCK_GET_COUNT(prev) == 0u) {
      /* Slot has not yet been installed (or was just freed by a prior
       * cycle).  Caller must not dereference. */
      return false;
    }
    uint64_t next = prev + 1u; /* count occupies low 32 bits */
    if (atomic_compare_exchange_weak_explicit(&item->lock, &prev, next,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
      return true;
    }
  }
}

void arts_route_table_release_item(arts_route_item_t *item) {
  if (item == NULL) {
    return;
  }
  uint64_t prev = atomic_load_explicit(&item->lock, memory_order_acquire);
  for (;;) {
    uint32_t count = ARTS_ROUTE_LOCK_GET_COUNT(prev);
    if (count == 0u) {
      /* Underflow / double-release — leave the slot alone but signal the
       * bug to anyone watching with a debug print. */
      ARTS_INFO("route_table release_item underflow on key=%lu lock=%lu",
                (uint64_t)item->key, (uint64_t)prev);
      return;
    }
    uint64_t next = prev - 1u;
    if (atomic_compare_exchange_weak_explicit(&item->lock, &prev, next,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
      if (ARTS_ROUTE_LOCK_GET_COUNT(next) == 0u &&
          ARTS_ROUTE_LOCK_HAS_DELETE(next)) {
        arts_route_item_free(item);
      }
      return;
    }
  }
}

void arts_route_table_release(arts_guid_t guid) {
  arts_route_table_t *route_table = arts_get_route_table(guid);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, guid);
  arts_route_table_release_item(item);
}

bool arts_route_table_mark_delete(arts_guid_t key) {
  /* New semantics (Task 4e):
   *   1. Set DELETE bit (sticky).  If already set, no-op (do NOT decrement
   *      the install ref a second time).
   *   2. Decrement the install-existence ref injected by add_item_race.
   *   3. If count reaches 0, invoke free_item.
   *
   * Note: data is NOT cleared here.  free_item (invoked here on the
   * non-deferred path or by the last release_item on the deferred path)
   * uses atomic_exchange to take ownership of the data pointer and
   * invoke the deleter.  Clearing data here would race that exchange to
   * NULL and the deleter would never see the live pointer, leaking the
   * payload.  Concurrent lookup_safe calls are blocked by the DELETE
   * bit via acquire_item, so leaving data live until free_item runs is
   * safe. */
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return false;
  }
  /* Step 1+2: atomically set DELETE and decrement install ref (only on the
   * first transition, to keep the call idempotent). */
  uint64_t prev = atomic_load_explicit(&item->lock, memory_order_acquire);
  bool first_call = false;
  for (;;) {
    if (ARTS_ROUTE_LOCK_HAS_DELETE(prev)) {
      first_call = false;
      break;
    }
    uint32_t count = ARTS_ROUTE_LOCK_GET_COUNT(prev);
    uint64_t next;
    if (count == 0u) {
      /* Slot never installed (or already freed).  Just set DELETE so a
       * concurrent install — should one race in — sees the slot as
       * destroyed.  Nothing to free. */
      next = prev | ARTS_ROUTE_LOCK_DELETE_BIT;
    } else {
      next = (prev - 1u) | ARTS_ROUTE_LOCK_DELETE_BIT;
    }
    if (atomic_compare_exchange_weak_explicit(&item->lock, &prev, next,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
      first_call = true;
      prev = next;
      break;
    }
  }
  /* Step 3: free trigger.  Do NOT clear item->data here; let free_item's
   * atomic_exchange take ownership exactly once. */
  if (first_call && ARTS_ROUTE_LOCK_GET_COUNT(prev) == 0u) {
    arts_route_item_free(item);
  }
  return true;
}

/* Type-aware safe lookups: search_for_key + acquire_item + DELETE check. */
static inline void *arts_route_table_lookup_safe_typed(arts_guid_t guid,
                                                       arts_guid_kind_t expected) {
  if (arts_guid_get_kind(guid) != expected) {
    return NULL;
  }
  arts_route_table_t *route_table = arts_get_route_table(guid);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, guid);
  if (item == NULL) {
    return NULL;
  }
  if (!arts_route_table_acquire_item(item)) {
    return NULL;
  }
  void *data = atomic_load_explicit(&item->data, memory_order_acquire);
  if (data == NULL) {
    /* Slot's data was claimed (legacy claim_item path) before our acquire
     * landed — DELETE is not set yet, but the object is already gone.
     * Drop the ref we just took. */
    arts_route_table_release_item(item);
    return NULL;
  }
  return data;
}

struct arts_event_s *arts_route_table_lookup_event_safe(arts_guid_t guid) {
  return (struct arts_event_s *)arts_route_table_lookup_safe_typed(guid,
                                                                   ARTS_GUID_EVENT);
}

struct arts_db_s *arts_route_table_lookup_db_safe(arts_guid_t guid) {
  return (struct arts_db_s *)arts_route_table_lookup_safe_typed(guid, ARTS_GUID_DB);
}

struct arts_edt_s *arts_route_table_lookup_edt_safe(arts_guid_t guid) {
  return (struct arts_edt_s *)arts_route_table_lookup_safe_typed(guid,
                                                                 ARTS_GUID_EDT);
}

struct arts_epoch_s *arts_route_table_lookup_epoch_safe(arts_guid_t guid) {
  return (struct arts_epoch_s *)arts_route_table_lookup_safe_typed(guid,
                                                                   ARTS_GUID_EPOCH);
}

void *arts_route_table_claim_item(arts_guid_t key) {
  /* Atomic exchange: returns the previous data ptr and stores NULL.
   * Used by single-flight destroyers (event_destroy, auto-destroy on
   * fire) so only the winning thread frees the underlying object. */
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item == NULL) {
    return NULL;
  }
  return atomic_exchange_explicit(&item->data, (void *)NULL,
                                  memory_order_acq_rel);
}

bool arts_route_table_hide_item(arts_guid_t key) {
  /* Hide is the same operation as mark_delete in the new model: clear the
   * data ptr without disturbing the slot. */
  return arts_route_table_mark_delete(key);
}

arts_route_item_t *get_item_from_data(arts_guid_t key, void *data) {
  if (data) {
    arts_route_item_t *item =
        (arts_route_item_t *)((char *)data - offsetof(arts_route_item_t, data));
    if (key == item->key) {
      return item;
    }
  }
  return NULL;
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
    arts_reader_lock(&current->readerLock, &current->writerLock);
    next = current->next;
    arts_reader_unlock(&current->readerLock);
    current = next;
  }
  return NULL;
}

void arts_print_item(arts_route_item_t *item) {
  if (item) {
    void *data = atomic_load_explicit(&item->data, memory_order_acquire);
    ARTS_INFO("GUID: %lu DATA: %p RANK: %u", item->key, data,
              arts_guid_get_rank(item->key));
  }
}

void arts_route_table_debug_guid(arts_guid_t key, const char *label) {
  arts_route_table_t *route_table = arts_get_route_table(key);
  arts_route_item_t *item = arts_route_table_search_for_key(route_table, key);
  if (item) {
    void *data = atomic_load_explicit(&item->data, memory_order_acquire);
    ARTS_INFO("[RT-DBG:%s] Guid:%lu data=%p rank=%u", label, key, data,
              arts_guid_get_rank(item->key));
  } else {
    ARTS_INFO("[RT-DBG:%s] Guid:%lu NOT FOUND", label, key);
  }
}

uint64_t arts_clean_up_route_table(arts_route_table_t *route_table) {
  uint64_t free_size = 0;
  arts_route_table_iterator_t iter;
  arts_reset_route_table_iterator(&iter, route_table);

  arts_route_item_t *item = arts_route_table_iterate(&iter);
  while (item) {
    arts_guid_kind_t type = arts_guid_get_kind(item->key);
    /* use atomic_exchange to claim
     * the data ptr.  EVENT / BUFFER lifecycle paths free their structs at
     * fire/destroy time WITHOUT NULLing the route_table slot (Task 2.1
     * removed arts_route_table_remove_item).  If we re-free those here
     * we get a tcache double-free.  Skip lifecycle-owned types and let
     * the per-type owner reclaim memory; cleanup only handles types whose
     * data ptr survives until shutdown (DBs that share a route entry until shutdown). */
    void *data = atomic_exchange_explicit(&item->data, (void *)NULL,
                                          memory_order_acq_rel);
    if (type == ARTS_GUID_DB) {
      struct arts_db_s *db = (struct arts_db_s *)data;
      if (db) {
        free_size += db->header.size;
        arts_db_free(db);
      }
    }
    /* ARTS_GUID_EVENT / ARTS_GUID_EDT / ARTS_GUID_EPOCH: data is owned by the execution
     * / lifecycle path and freed there.  Cleanup just drops the
     * route_table reference (already done by atomic_exchange above) plus
     * the OoO list.  Re-freeing here would tcache-corrupt. */
    arts_oo_list_drop_all(&item->ooList);
    item = arts_route_table_iterate(&iter);
  }
  return free_size;
}

void arts_delete_route_table(arts_route_table_t *route_table) {
  if (!route_table) {
    return;
  }
  arts_delete_route_table(route_table->next);
  /* Safety sweep: drop any OO list memory that arts_clean_up_route_table
   * missed (e.g. entries claimed but never installed). */
  for (uint64_t i = 0; i < (uint64_t)route_table->size * COLLISION_RESOLVES;
       i++) {
    arts_oo_list_drop_all(&route_table->data[i].ooList);
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
