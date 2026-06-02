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
#include "arts/runtime_types.h" /* struct arts_edt_s / arts_db_s (OoO args) */
#include "arts/utils/lockfree_lifo.h" /* arts_lf_stack_t (per-slot OoO chain) + arts_lf_link_t */
#include "arts/utils/shared.h" /* arts_shared_ptr_t, arts_atomic_shared_ptr_t */

#define COLLISION_RESOLVES 8
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
   * (Cat B request/writeback defers + drains on the next install or shutdown;
   * Cat C response/ack silent-drops on absent), not by a per-slot state bit. */
  arts_atomic_shared_ptr_t value; /* cb: event/db/edt/epoch (NULL = absent) */
  arts_lf_stack_t
      ooo_list; /* OoO defer chain (Treiber; preserved across free) */
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
 * For objects that don't map to a GUID kind's deleter (e.g. the epoch pool
 * wrapper, which is freed by its owner) — pass NULL to install a cb that
 * never frees the object (route_table holds it for lookup only; the caller
 * frees it manually).  Returns the object now installed. */
void *arts_route_table_install_with_deleter(void *obj, arts_guid_t key,
                                            void (*deleter)(void *));

/* Unsafe peek: raw object pointer WITHOUT holding a ref (load + get + drop).
 * Valid only when the caller has an external liveness guarantee for `key`
 * (e.g. it is the home/owner and no concurrent destroy is possible).  Same
 * unsafety contract as the historical lookup_data. */
void *arts_route_table_lookup_data(arts_guid_t key);

int arts_route_table_lookup_rank(arts_guid_t key);

/* Destroy: atomically detach the cb from `key`'s slot and drop the install
 * ref.  Single-flight (only the caller whose exchange observes a non-NULL cb
 * "wins"); idempotent.  The object's deleter runs once the last outstanding
 * reader ref is released.  Returns true if this call detached the cb. */
bool arts_route_table_mark_delete(arts_guid_t key);

/* Type-aware safe lookups: return a caller-owned cb handle (strong ref held)
 * or NULL if the slot is absent / destroyed / a kind mismatch.  Use
 * arts_shared_get(h) for the object and arts_shared_release(&h) when done. */
arts_shared_ptr_t arts_route_table_lookup_event(arts_guid_t guid);
arts_shared_ptr_t arts_route_table_lookup_db(arts_guid_t guid);
arts_shared_ptr_t arts_route_table_lookup_edt(arts_guid_t guid);
arts_shared_ptr_t arts_route_table_lookup_epoch(arts_guid_t guid);

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

/* Unsafe peek of a slot's published object WITHOUT holding a ref (load + get +
 * drop).  C-linkage bridge for the slot API (which is otherwise C11-only) so
 * C++/nvcc translation units can read a slot whose object has an external
 * liveness guarantee (e.g. a persistent, never-freed mirror payload). */
void *arts_route_item_peek_data(arts_route_item_t *item);

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
 * creating it (value == NULL) if absent.  The OoO engine (below) uses this to
 * reach a slot's ooo_list. */
void arts_route_table_reserve_or_lookup(arts_guid_t key,
                                        arts_route_item_t **out);

void arts_reset_route_table_iterator(arts_route_table_iterator_t *iter,
                                     arts_route_table_t *table);
arts_route_item_t *arts_route_table_iterate(arts_route_table_iterator_t *iter);
void arts_print_item(arts_route_item_t *item);

uint64_t arts_clean_up_route_table(arts_route_table_t *route_table);
void arts_delete_route_table(arts_route_table_t *route_table);
void arts_clean_up_dbs();

/* ===========================================================================
 * Out-of-order (OoO) deferred-op engine.
 *
 * The OoO engine owns no data structure of its own: it operates on the
 * arts_route_item_s.ooo_list Treiber stack declared above.  A slot accumulates
 * deferred operations that arrived before their target object was installed;
 * the create handler drains them once the object is published.  Declarations
 * below live in the same subsystem as the GUID directory they replay against.
 * ===========================================================================*/

/* OoO replay kind.  Identifies which g_ooo_table[] handler replays a deferred
 * operation once its target object is installed in the route table. */
enum arts_ooo_kind {
  OOO_EDT_SATISFY_SLOT,
  OOO_EVENT_SATISFY_SLOT,
  OOO_EVENT_ADD_DEPENDENCE,
  OOO_HANDLE_READY_EDT,
  OOO_DB_ACQUIRE,
  OOO_EDT_SATISFY_SLOT_PTR,
  OOO_EPOCH_REQUEST,
  OOO_EPOCH_SEND,
  OOO_EPOCH_INC_ACTIVE,
  OOO_EPOCH_INC_FINISHED,
  OOO_EPOCH_INC_QUEUE,
  /* Coherence-protocol replay kinds: re-issue the wire-message handler once
   * the home-side db_s/cache is installed (DB_CREATE arrives after a
   * race-arrived OWNERSHIP_REQUEST / SNAPSHOT_REQUEST / DESTROY / WRITEBACK).
   */
  OOO_DB_OWNERSHIP_REQUEST,
  OOO_DB_SNAPSHOT_REQUEST,
  OOO_DB_DESTROY,
  OOO_DB_WRITEBACK,
  /* Lifecycle destroy replay kinds (before-create wire reorder): a DESTROY that
   * reaches home ahead of the object's CREATE defers here and replays once the
   * create handler installs+drains.  (There is deliberately no ownership-
   * invalidate replay kind: INVALIDATE is a sharer-side message addressed to
   * the current DB holder, which always has a cache; a missing cache means the
   * DB was destroyed, where the sentinel withdrawal is moot and dropping — not
   * deferring — is correct.) */
  OOO_EVENT_DESTROY,
  OOO_EDT_DESTROY,
  OOO_KIND_COUNT /* sentinel — g_ooo_table size */
};
typedef enum arts_ooo_kind ooo_kind_t;

/* Unified OoO payload.  The link is the FIRST member so a node address equals
 * its link address (Treiber stack contract).  A variable-size args blob trails
 * the header (heap-allocated as sizeof(payload) + args_size); each kind casts
 * the blob back to its own args struct.  This single type replaces the former
 * per-kind node structs and the separate oo_node wrapper. */
struct arts_ooo_payload_s {
  arts_lf_link_t link; /* MUST be first */
  ooo_kind_t kind;
  uint32_t args_size;
  /* args blob follows here */
};

static inline void *arts_ooo_payload_args(struct arts_ooo_payload_s *p) {
  return (void *)(p + 1);
}

/* ===== per-kind args =====================================================
 * The deferring entry copies one of these into the payload blob; the
 * g_ooo_table[kind] handler casts the blob back and replays the operation
 * against the now-installed target (re-issuing the entry, so the install
 * race / fire-and-linger logic stays in one place). */

struct arts_ooo_args_edt_satisfy_s {
  arts_guid_t edt_guid; /* re-signal target (may differ from the deferred-on
                           slot, e.g. GPU CDAG defers on the wrapper) */
  arts_guid_t data_guid;
  uint32_t slot;
  arts_db_access_mode_t mode;
};

/* DB_MODE_PTR delivery: inline payload trails this header in the blob
 * (args_size == sizeof(this) + size). */
struct arts_ooo_args_edt_satisfy_ptr_s {
  arts_guid_t edt_guid;
  arts_guid_t data_guid;
  uint32_t slot;
  uint32_t size;
};

struct arts_ooo_args_event_satisfy_s {
  arts_guid_t event_guid;
  arts_guid_t data_guid;
  uint32_t slot;
};

struct arts_ooo_args_event_add_dep_s {
  arts_guid_t source;
  arts_guid_t destination;
  uint32_t slot;
  arts_db_access_mode_t mode;
};

struct arts_ooo_args_handle_ready_s {
  struct arts_edt_s *edt;
};

struct arts_ooo_args_db_acquire_s {
  struct arts_edt_s *edt;
  arts_guid_t db_guid;
  uint32_t slot;
};

struct arts_ooo_args_epoch_s {
  arts_guid_t epoch_guid;
};

/* EPOCH_REQUEST replay (reply side): a non-home rank replies to an epoch
 * query by sending its local counts to dest.  Deferred when the local epoch
 * broadcast-install has not yet arrived. */
struct arts_ooo_args_epoch_request_s {
  arts_guid_t epoch_guid;
  unsigned int source;
  unsigned int dest;
};

/* EPOCH_SEND replay (reduce side): the home rank reduces received counts into
 * the epoch's global tally.  Deferred when the home epoch is not yet installed
 * (theoretical; home always owns its epoch). */
struct arts_ooo_args_epoch_send_s {
  arts_guid_t epoch_guid;
  unsigned int active;
  unsigned int finish;
};

/* Coherence replay args — re-issue the wire handler once the home db_s/cache
 * is installed.  First-class fields are reconstructed into a stack packet by
 * the handler. */
struct arts_ooo_args_db_ownership_request_s {
  unsigned int requester;
  arts_guid_t db_guid;
};

struct arts_ooo_args_db_snapshot_request_s {
  unsigned int requester;
  arts_guid_t db_guid;
  arts_guid_t edt_guid;
  uint32_t slot;
};

struct arts_ooo_args_db_destroy_s {
  unsigned int requester;
  arts_guid_t db_guid;
};

/* DB writeback: inline write-back payload trails this header. */
struct arts_ooo_args_db_writeback_s {
  unsigned int releaser;
  arts_guid_t db_guid;
  uint64_t version;
  uint64_t cv; /* releaser's sem_t address, echoed in the ACK */
  uint16_t flag;
  uint64_t data_size;
};

/* Event / EDT destroy replay (before-create reorder): the guid is enough to
 * re-issue the destroy once the object installs. */
struct arts_ooo_args_event_destroy_s {
  arts_guid_t guid;
};
struct arts_ooo_args_edt_destroy_s {
  arts_guid_t guid;
};

/* g_ooo_table handler: operate on an already-acquired, valid item with the
 * decoded args.  The handler performs NO route-table lookup / NULL-check /
 * acquire / push — dispatch_or_defer guarantees `item` is live and ref-pinned
 * for the duration of the call. */
typedef void (*arts_ooo_handler_fn)(void *item, void *args);

/* Universal non-create entry — wire RX dispatcher, API drivers, and the drain
 * walk all enter here.
 *
 *   payload == NULL : fresh entry (wire RX / API).  On miss a payload is
 *                     allocated (args copied) and pushed.
 *   payload != NULL : drain re-entry.  On miss the SAME payload is re-pushed
 *                     (no alloc/free) to await a future install.
 *
 * Per-call acquire: the slot value is (re)loaded on every call so that a
 * destroy that NULLed the slot earlier in the same drain walk is observed and
 * the node re-defers (labeled-reuse: a later create re-installs and re-drains).
 */
void arts_ooo_dispatch_or_defer(struct arts_route_item_s *slot,
                                struct arts_ooo_payload_s *payload,
                                ooo_kind_t kind, const void *args,
                                uint32_t args_size);

/* Convenience fresh entry: reserve/lookup the slot for `guid`, then
 * dispatch_or_defer with payload == NULL. */
void arts_ooo_dispatch_or_defer_guid(arts_guid_t guid, ooo_kind_t kind,
                                     const void *args, uint32_t args_size);

/* Unconditional defer (force-push) keyed on `guid` — used by the GPU
 * CDAG-invalidation path which must hold a signal until the wrapper EDT's
 * outstanding invalidations drain, regardless of the destination's install
 * state. */
void arts_ooo_push_guid(arts_guid_t guid, ooo_kind_t kind, const void *args,
                        uint32_t args_size);

/* Create handler's last step: replay accumulated payloads against the
 * just-installed item.  Detaches one snapshot of the slot's OoO chain
 * (reverse_drain → FIFO) and runs each through dispatch_or_defer.  Lock-free:
 * concurrent drains detach disjoint snapshots. */
void arts_ooo_drain(struct arts_route_item_s *slot);

/* Convenience: look up the slot for `guid` and drain it.  Used by create
 * handlers that hold only the GUID (e.g. coherence DB_CREATE). */
void arts_ooo_drain_guid(arts_guid_t guid);

/* Free every payload still queued on a slot's chain without dispatching —
 * route-table teardown only (the chain is otherwise preserved across
 * destroy/reinstall). */
void arts_ooo_free_all(struct arts_route_item_s *slot);

/* OoO replay continuation for a local DB→EDT dependency that resolved after
 * the EDT was registered: fills the EDT's dep slot with the installed DB and
 * drops one depc_needed, dispatching the EDT when it reaches zero. */
void arts_ooo_resolve_db_dep(struct arts_edt_s *edt, unsigned int slot,
                             struct arts_db_s *db_res);

#ifdef __cplusplus
}
#endif

#endif
