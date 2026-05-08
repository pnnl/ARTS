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
#ifndef ARTS_GAS_ROUTETABLE_H
#define ARTS_GAS_ROUTETABLE_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts.h"
#include "arts/defs.h"
#include "arts/gas/out_of_order_list.h"

struct arts_db_frontier_iterator_s;

/* Portable atomic-void-pointer typedef.  C uses _Atomic; C++/nvcc uses a
 * plain pointer accessed via __atomic_* builtins (which gcc/clang/nvcc all
 * support via the host compiler).  Wire-compatible -- both sides see the
 * same bit layout (sizeof(void *) on both compilers). */
#ifdef __cplusplus
typedef void *arts_atomic_voidp_t;
typedef uint64_t arts_atomic_u64_t;
#else
#include <stdatomic.h>
typedef _Atomic(void *) arts_atomic_voidp_t;
typedef _Atomic(uint64_t) arts_atomic_u64_t;
#endif

/* Lock-field encoding for arts_route_item_t::lock (Task 4e).
 * Layout: [DELETE:1 (bit 63) | gen:31 (bits 32..62) | count:32 (bits 0..31)].
 *
 *   - DELETE bit:  set by mark_delete; once set the item's underlying object
 *                  is being torn down.  acquire_item must fail after
 *                  observing DELETE; once count reaches 0 with DELETE set,
 *                  free_item is invoked.
 *   - gen counter: monotonically incremented after free_item completes.
 *                  Prevents ABA on slot reuse (the slot itself is permanent
 *                  but its data pointer can be reinstalled for the same GUID).
 *   - count:       reference count.  add_item_race installs lock with
 *                  count = 1 (the install-existence ref); each successful
 *                  acquire_item increments; each release_item decrements.
 *                  When count reaches 0 AND DELETE is set, free_item runs. */
#define ARTS_ROUTE_LOCK_DELETE_BIT (1ULL << 63)
#define ARTS_ROUTE_LOCK_GEN_SHIFT 32
#define ARTS_ROUTE_LOCK_GEN_MASK (0x7FFFFFFFULL << ARTS_ROUTE_LOCK_GEN_SHIFT)
#define ARTS_ROUTE_LOCK_COUNT_MASK 0xFFFFFFFFULL

#define ARTS_ROUTE_LOCK_GET_COUNT(lock)                                        \
  ((uint32_t)((lock) & ARTS_ROUTE_LOCK_COUNT_MASK))
#define ARTS_ROUTE_LOCK_GET_GEN(lock)                                          \
  ((uint32_t)(((lock) & ARTS_ROUTE_LOCK_GEN_MASK) >> ARTS_ROUTE_LOCK_GEN_SHIFT))
#define ARTS_ROUTE_LOCK_HAS_DELETE(lock)                                       \
  (((lock) & ARTS_ROUTE_LOCK_DELETE_BIT) != 0ULL)
#define ARTS_ROUTE_LOCK_PACK(del, gen, count)                                  \
  (((del) ? ARTS_ROUTE_LOCK_DELETE_BIT : 0ULL) |                               \
   (((uint64_t)(gen) << ARTS_ROUTE_LOCK_GEN_SHIFT) &                           \
    ARTS_ROUTE_LOCK_GEN_MASK) |                                                \
   ((uint64_t)(count) & ARTS_ROUTE_LOCK_COUNT_MASK))

#define COLLISION_RESOLVES 8
/* Number of independent shards for the remote_route_table.  Must be a
 * power of 2 so (key & (N-1)) is the shard selector. */
#define ARTS_REMOTE_ROUTE_SHARDS 8

struct arts_route_invalidate_s {
  int size;
  int used;
  struct arts_route_invalidate_s *next;
  unsigned int data[];
};

/* add_oo_ex return enum — caller decides branch.
 *
 * AVAILABLE_NOW: data was non-NULL on the pre-push or push-time check, so the
 *   payload was never inserted into the OO list.  Caller handles inline and
 *   frees the payload itself.
 *
 * ENQUEUED: data was NULL throughout; payload sits in the OO list and a
 *   future installer's fire_oo will dispatch it.  Caller does nothing.
 *
 * FIRED_BY_DRAIN: payload was successfully pushed, but the post-push
 *   recheck found data non-NULL (installer raced past us).  add_oo_ex
 *   then called fire_oo itself, which drained the list and invoked the
 *   handler on every entry, including ours, and freed each payload.
 *   Caller MUST NOT free the payload and MUST NOT re-issue the handler. */
typedef enum {
  OO_RESULT_ENQUEUED,
  OO_RESULT_AVAILABLE_NOW,
  OO_RESULT_FIRED_BY_DRAIN,
} oo_add_result_t;

/* Route_item: 4 fields. Slot is permanent (init-array, never freed).
 *
 * `lock` (Task 4e) packs DELETE / gen / count for ABA-safe ref counting on
 * the slot's data pointer.  See ARTS_ROUTE_LOCK_* macros above.  Existing
 * call sites use the legacy data CAS + claim_item path and ignore `lock`;
 * the new `acquire_item` / `release_item` / typed `_safe` lookups (declared
 * below) form the migration target for Phases 6/7/8. */
struct arts_route_item_s {
  arts_guid_t key;
  arts_atomic_voidp_t data;     /* NULL = pending OoO, else = AVAILABLE */
  arts_atomic_u64_t lock;       /* [DELETE:1 | gen:31 | count:32] */
  struct arts_oo_list_s ooList; /* lock-free list (preserved across free) */
} ARTS_ALIGNED_MAX;

typedef struct arts_route_item_s arts_route_item_t;

typedef struct arts_route_table_s arts_route_table_t;

typedef arts_route_table_t *(*new_route_table_t)(unsigned int route_table_size,
                                                 unsigned int shift);

// Add padding around locks...
struct arts_route_table_s {
  arts_route_item_t *data;
  unsigned int size;
  unsigned int shift;
  struct arts_route_table_s *next;
  volatile unsigned readerLock;
  volatile unsigned writerLock;
  new_route_table_t newFunc;
}; // __attribute__ ((aligned));

typedef struct {
  uint64_t index;
  arts_route_table_t *table;
} arts_route_table_iterator_t;

arts_route_table_t *arts_new_route_table(unsigned int route_table_size,
                                         unsigned int shift);

void *arts_route_table_add_item(void *item, arts_guid_t key, unsigned int rank,
                                bool used);
arts_route_item_t *internal_route_table_add_item_race(
    bool *added_item, arts_route_table_t *route_table, void *item,
    arts_guid_t key, unsigned int rank, bool used_res, bool used_avail,
    unsigned int to_add_on_creation);
bool arts_route_table_add_item_race(void *item, arts_guid_t key,
                                    unsigned int rank, bool used);
arts_route_item_t *
internal_route_table_add_deleted_item_race(arts_route_table_t *route_table,
                                           void *item, arts_guid_t key,
                                           unsigned int rank);

/* Lookup: returns data ptr directly (atomic_load_acquire). NULL means
 * not-yet-created or destroyed. */
void *arts_route_table_lookup_data(arts_guid_t key);

/* Item lookup — returns data ptr. Replaces legacy lookup_item. */
void *arts_route_table_lookup_item(arts_guid_t key);

int arts_route_table_lookup_rank(arts_guid_t key);
bool arts_route_table_hide_item(arts_guid_t key);
/* arts_route_table_mark_delete declared with the D1 lifecycle API below. */

/* Atomically claim the slot's data pointer: exchange with NULL and
 * return the previous value.  Returns NULL if no slot exists for key
 * or if the slot was already cleared.  Safe for single-flight
 * destruction races: only the thread that observes a non-NULL return
 * is responsible for freeing the underlying object. */
void *arts_route_table_claim_item(arts_guid_t key);

/* ---------------------------------------------------------------------------
 * D1 reference-counted item lifecycle API.
 *
 * The functions below operate on the route_item lock field defined above.
 * Future migrations of event/DB/EDT/epoch (Phases 6/7/8) will replace the
 * legacy data-CAS + claim_item dance with the acquire/release pattern.
 * Until then they coexist with the legacy paths — the lock field is
 * installed by add_item_race but only read by the new APIs and the
 * type-aware free_item dispatch (Task 4f).
 *
 * Pairing rule: every successful acquire_item (or _safe lookup) must be
 * matched with a release_item.  Mark_delete is idempotent and does not
 * itself drop the matching ref of any prior acquire — it only consumes
 * the install-existence ref injected by add_item_race.
 * --------------------------------------------------------------------------*/

/* Increment the item's ref count atomically.  Returns false if DELETE is
 * set on the slot (caller must NOT dereference data); on success the count
 * has been incremented and the caller holds a ref that must be released
 * via arts_route_table_release_item. */
bool arts_route_table_acquire_item(arts_route_item_t *item);

/* Decrement the item's ref count atomically.  When count drops to 0 AND
 * DELETE is set, invokes free_item on the slot. */
void arts_route_table_release_item(arts_route_item_t *item);

/* Set the DELETE bit on the slot for `key` and consume the install-existence
 * ref injected by add_item_race.  If count drops to 0, invokes free_item.
 * Idempotent: calling twice does NOT decrement twice (DELETE is sticky and
 * the second call is a no-op).  Also clears the slot's data pointer (legacy
 * semantics, preserved for compatibility with current callers). */
bool arts_route_table_mark_delete(arts_guid_t key);

/* Type-aware safe lookups.  Each performs search_for_key + acquire_item;
 * returns NULL if the slot is missing, the data pointer is NULL, or DELETE
 * is set.  Caller must call arts_route_table_release(guid) after every
 * successful lookup. */
struct arts_event_s *arts_route_table_lookup_event_safe(arts_guid_t guid);
struct arts_db_s *arts_route_table_lookup_db_safe(arts_guid_t guid);
struct arts_edt_s *arts_route_table_lookup_edt_safe(arts_guid_t guid);
struct arts_epoch_s *arts_route_table_lookup_epoch_safe(arts_guid_t guid);

/* Generic release counterpart for callers that already hold a ref via one
 * of the *_safe lookups above (or via acquire_item on a slot they located
 * themselves).  Resolves the slot from `guid` and decrements the ref. */
void arts_route_table_release(arts_guid_t guid);

arts_route_item_t *
arts_route_table_search_for_key(arts_route_table_t *route_table,
                                arts_guid_t key);
int arts_route_table_set_rank(arts_guid_t key, int rank);

/* Slot reserve or lookup — used internally by add_oo_ex. New entries are
 * initialized with data=NULL. */
void arts_route_table_reserve_or_lookup(arts_guid_t key,
                                        arts_route_item_t **out);

arts_route_item_t *get_item_from_data(arts_guid_t key, void *data);

/* OoO-integrated add — returns enum */
oo_add_result_t arts_route_table_add_oo_ex(arts_guid_t key, void *payload);

/* Compatibility wrapper for the 5 existing callers
 * (signal_edt / event_satisfy / add_dep / ready_edt / db_request). */
bool arts_route_table_add_oo(arts_guid_t key, void *payload, bool inc);

/* Compatibility wrapper for the legacy "_existing" variant
 * (route_table.c:786). */
bool arts_route_table_add_oo_existing(arts_guid_t key, void *payload, bool inc);

/* Fire the OO list — called by the installer (e.g. DB_CREATE). Idempotent. */
void arts_route_table_fire_oo(arts_guid_t key,
                              void (*callback)(void *data, void *ctx));

/* Drain OO list at destroy time, waking parked EDT-DB-request waiters
 * with NULL_DB so they observe the destroyed-DB semantic.  Replaces the
 * earlier "silent free" semantics that left waiters stranded. */
void arts_route_table_drop_oo(arts_guid_t key);

/* Mark a route_item as destroyed by setting the DELETE bit on the lock
 * field WITHOUT decrementing the install reference.  Used by the RC
 * destroy path which manages teardown directly via arts_db_free and
 * bypasses the route_table mark_delete + free_item refcount flow.
 *
 * After this call:
 *   - arts_route_table_acquire_item fails (lookup_safe returns NULL).
 *   - arts_route_table_add_oo_ex returns OO_RESULT_AVAILABLE_NOW so
 *     callers fire NULL-callback inline (destroyed-DB semantic).
 *
 * Idempotent: repeated calls are no-ops once DELETE is set.  Note the
 * route_item slot is leaked (install ref never dec'd) -- RC trades
 * a bounded route_table-slot leak for direct-free determinism. */
void arts_route_table_set_destroyed(arts_guid_t key);

void arts_reset_route_table_iterator(arts_route_table_iterator_t *iter,
                                     arts_route_table_t *table);
arts_route_item_t *arts_route_table_iterate(arts_route_table_iterator_t *iter);
void arts_print_item(arts_route_item_t *item);
void arts_route_table_debug_guid(arts_guid_t key, const char *label);

uint64_t arts_clean_up_route_table(arts_route_table_t *route_table);
void arts_delete_route_table(arts_route_table_t *route_table);
void arts_clean_up_dbs();

#ifdef __cplusplus
}
#endif

#endif
