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
#include "arts/utils/lockfree_lifo.h" /* arts_lf_stack_t (per-slot OoO chain) + arts_lf_link_t */
#include "arts/utils/shared.h" /* arts_shared_ptr_t, arts_atomic_shared_ptr_t */

/* Convention: all shared-object access is via caller-owned cb handles
 * (lookup_* / _acquire → handle; release required). No raw no-ref peeks. */

#define COLLISION_RESOLVES 8

/* Key values that name no object.  0 = FREE (claimable).  RETIRING is the key
 * a teardown parks in the slot while it detaches the value.  It is the KEY
 * word, not a lock beside it: one CAS from the GUID to this sentinel carries
 * both halves of what a teardown needs — proof the slot still names that
 * GUID, and sole ownership of the teardown — so there is nothing to hold and
 * nothing to wait on, and a loser is simply not the destroyer.  No observer
 * ever waits on it either: it is not 0, so a claim CAS skips the slot, and it
 * carries kind ARTS_GUID_RESERVED (never a valid object), so it equals no
 * real GUID and a key search stops matching at once — a request arriving
 * mid-teardown resolves to a fresh reservation instead of waiting. */
#define ARTS_ROUTE_KEY_RETIRING ((arts_guid_t)1)
/* Number of independent shards for the remote_route_table.  Must be a
 * power of 2 so (key & (N-1)) is the shard selector. */
#define ARTS_REMOTE_ROUTE_SHARDS 8

/* Route_item: GUID slot.  Permanent (init-array, never freed).
 *
 * `value` is an atomic shared-ptr cb slot (arts_shared_ptr_t).  The cb
 * carries the strong refcount + per-object deleter + object pointer, so the
 * old lock bitfield ([DELETE | gen | count]) is gone entirely:
 *   - count  → cb strong refcount,
 *   - DELETE → value == NULL (a destroyer atomic_exchanges it to NULL),
 *   - gen    → the cb pool's DWCAS tag (allocator ABA) + the load-side slot
 *              revalidation (load ABA).
 * Lookups acquire a caller-owned ref via arts_atomic_shared_load and release
 * it on the local handle.  The ooList is preserved across destroy/reinstall. */
struct arts_route_item_s {
  arts_guid_t key;
  /* value == NULL means "absent" — deliberately NOT distinguishing
   * "never created" from "destroyed".  The OoO defer path treats both
   * uniformly; create-vs-destroy semantics are resolved by handler category
   * (Cat B request/publish defers + drains on the next install or shutdown;
   * Cat C response/ack silent-drops on absent), not by a per-slot state bit.
   *
   * key == 0 means the slot is FREE.  A destroy returns the slot by zeroing
   * the key, so occupancy tracks live objects rather than every object the
   * run ever made; a claim stays the single key-CAS below because a returned
   * slot is indistinguishable from a never-used one. */
  arts_atomic_shared_ptr_t value; /* cb: event/db/edt (NULL = absent) */
  arts_lf_stack_t ooo_list; /* OoO defer chain (Treiber) */
} ARTS_ALIGNED_MAX;

typedef struct arts_route_item_s arts_route_item_t;

typedef struct arts_route_table_s arts_route_table_t;

typedef arts_route_table_t *(*new_route_table_t)(unsigned int route_table_size,
                                                 unsigned int shift);

struct arts_route_table_s {
  arts_route_item_t *data;
  unsigned int size;
  unsigned int shift;
  /* The growable segment chain.  Reads use an acquire-load and growth uses a
   * single NULL->segment CAS (see route_table.c): the chain only ever grows
   * (a published segment is never unlinked before teardown), so the link is
   * monotonic and ABA-free — no lock is needed to traverse or extend it. */
  struct arts_route_table_s *next;
  new_route_table_t newFunc;
};

typedef struct {
  uint64_t index;
  arts_route_table_t *table;
} arts_route_table_iterator_t;

arts_route_table_t *arts_new_route_table(unsigned int route_table_size,
                                         unsigned int shift);

/* ---------------------------------------------------------------------------
 * cb-based item lifecycle API.
 *
 * The slot holds an atomic shared-ptr cb.  Install wraps the object in a cb
 * (deleter chosen by GUID kind); lookups return a caller-owned cb handle the
 * caller releases when done; destroy detaches the cb and drops the install
 * ref (the object's deleter runs once the last reader also releases, so
 * destroy-during-use is a deferred free, never a use-after-free).
 * --------------------------------------------------------------------------*/

/* Register the cb deleter for a GUID kind.  Each object type calls this once
 * at startup (from a constructor in its own TU) so route_table can pick the
 * right deleter at install without referencing the per-type deleter symbols by
 * name (which would couple arts_gas to arts_memory/arts_compute and break the
 * separately-linked CUDA library). */
void arts_route_table_register_deleter(arts_guid_kind_t kind,
                                       void (*deleter)(void *));

/* Install `obj` under `key`.  Wraps it in a cb whose deleter is selected by
 * the GUID kind (registered via arts_route_table_register_deleter). Idempotent:
 * if the slot already holds a cb, the existing object is returned and `obj` is
 * NOT wrapped (caller still owns it).  Returns the object now installed. */
void *arts_route_table_install(void *obj, arts_guid_t key, unsigned int rank,
                               bool used);

/* Race install: CAS the cb into an empty slot.  Returns true only if this
 * caller won (and fired the OoO list).  On loss, `obj` is left untouched
 * (caller owns it).  On win, the slot owns the object via its cb. */
bool arts_route_table_install_if_absent(void *obj, arts_guid_t key,
                                        unsigned int rank, bool used);

/* Idempotent install with an EXPLICIT deleter, bypassing deleter-by-kind.
 * Pass NULL to install a cb that never frees the object (route_table holds it
 * for lookup only; the caller frees it manually).  Returns the object now
 * installed. */
void *arts_route_table_install_with_deleter(void *obj, arts_guid_t key,
                                            void (*deleter)(void *));

int arts_route_table_lookup_rank(arts_guid_t key);

/* Destroy: atomically detach the cb from `key`'s slot and drop the install
 * ref.  Single-flight (only the caller whose exchange observes a non-NULL cb
 * "wins"); idempotent.  The object's deleter runs once the last outstanding
 * reader ref is released.  Returns true if this call detached the cb.
 *
 * Also RETURNS the slot: the deferred payloads are drained and the key is
 * zeroed, so the slot is claimable again by the ordinary key-CAS.  Occupancy
 * therefore tracks live objects, not every object the run ever made. */
bool arts_route_table_set_destroyed(arts_guid_t key);

/* Type-aware safe lookups: return a caller-owned cb handle (strong ref held)
 * or NULL if the slot is absent / destroyed / a kind mismatch.  Use
 * arts_shared_get(h) for the object and arts_shared_release(&h) when done. */
arts_shared_ptr_t arts_route_table_lookup_event(arts_guid_t guid);
arts_shared_ptr_t arts_route_table_lookup_db(arts_guid_t guid);
arts_shared_ptr_t arts_route_table_lookup_edt(arts_guid_t guid);

/* Kind-agnostic handle lookup (no kind validation). */
arts_shared_ptr_t arts_route_table_lookup(arts_guid_t key);

/* Relocate the object from old_key's slot to new_key's slot by moving the
 * SAME cb (no re-wrap, no extra ref) — preserving the single-owner
 * invariant.  Used by DB rename / copy-to-new-type, which must keep one cb
 * owning the descriptor across the GUID change (creating a second cb would
 * make two slots each free the same object).  Detaches old_key (its slot
 * becomes absent) and CAS-installs the cb into new_key, firing new_key's OoO
 * list.  Returns true on success; false if old_key is absent or new_key was
 * already occupied (in which case the moved cb is released). */
bool arts_route_table_move_item(arts_guid_t old_key, arts_guid_t new_key);

arts_route_item_t *
arts_route_table_search_for_key(arts_route_table_t *route_table,
                                arts_guid_t key);

/* Linearly scan for an empty slot and CAS-claim it for `key` (or return the
 * already-claimed slot for `key`).  Operates on the GIVEN table, so it works
 * for non-global mirror tables.  `mark_used` is currently ignored. */
arts_route_item_t *
arts_route_table_search_for_empty(arts_route_table_t *route_table,
                                  arts_guid_t key, bool mark_used);

/* Safe C-linkage acquire of a slot's published object; returns a caller-owned
 * handle (NULL if empty); release via arts_shared_release.  The C++/.cu bridge
 * for the otherwise C-only atomic-slot API — replaces the removed raw peek. */
arts_shared_ptr_t arts_route_item_acquire(arts_route_item_t *item);

/* Publish `obj` into THIS slot's cb with an explicit deleter (idempotent CAS).
 * Unlike add_item/add_item_with_deleter, which locate the slot via the global
 * key→table map, this installs into a caller-located slot — required for
 * non-global mirror tables (e.g. the GPU per-device route tables) where the
 * slot lives in a different table than the one the global map would pick.
 * C-linkage bridge (slot CAS is otherwise C11-only) so C++/nvcc translation
 * units can publish a persistent mirror payload (pass NULL deleter for a cb
 * that never frees the object).  Returns true iff this caller won the install;
 * on loss the slot's existing cb is kept and `obj` is left untouched. */
bool arts_route_item_install_data(arts_route_item_t *item, void *obj,
                                  void (*deleter)(void *));

/* Slot reserve or lookup — returns the permanent route_item for `key`,
 * creating it (value == NULL) if absent.  The OoO engine (arts/ooo.h) uses
 * this to reach a slot's ooo_list. */
void arts_route_table_reserve_or_lookup(arts_guid_t key,
                                        arts_route_item_t **out);

void arts_reset_route_table_iterator(arts_route_table_iterator_t *iter,
                                     arts_route_table_t *table);
arts_route_item_t *arts_route_table_iterate(arts_route_table_iterator_t *iter);
void arts_print_item(arts_route_item_t *item);

uint64_t arts_clean_up_route_table(arts_route_table_t *route_table);
void arts_delete_route_table(arts_route_table_t *route_table);
void arts_clean_up_dbs();

#ifdef __cplusplus
}
#endif

#endif
