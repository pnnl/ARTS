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
#ifndef ARTS_OOO_H
#define ARTS_OOO_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/defs.h"
#include "arts/gas/route_table.h" /* struct arts_route_item_s (ooo_list slot) */
#include "arts/runtime_types.h"   /* struct arts_edt_s / arts_db_s (OoO args) */
#include "arts/utils/lockfree_lifo.h" /* arts_lf_link_t (payload link) */

/* ===========================================================================
 * Out-of-order (OoO) deferred-op engine.
 *
 * The OoO engine owns no data structure of its own: it operates on the
 * arts_route_item_s.ooo_list Treiber stack declared in route_table.h.  A slot
 * accumulates deferred operations that arrived before their target object was
 * installed; the create handler drains them once the object is published.  The
 * engine depends one-way on the route-table slot API (it reaches a slot's
 * ooo_list via arts_route_table_reserve_or_lookup), so it lives in its own
 * module while the GUID directory it replays against stays in route_table.
 * ===========================================================================*/

/* OoO replay kind — dispatch tag for OoO-deferrable (Cat B) handlers only.
 * Identifies which g_ooo_table[] handler replays a deferred operation once its
 * target object is installed in the route table.  Install-trigger (Cat A),
 * silent-drop (Cat C) and state-less (Cat E) handlers reach their handler only
 * through the wire dispatcher's MSG_* mapping and have NO kind here.
 *
 * Naming invariant: OOO_<NAME> == the handler arts_handler_<name> with the
 * arts_handler_ prefix stripped and upper-cased (kind ↔ handler 1:1), e.g.
 * arts_handler_event_add_dependence → OOO_EVENT_ADD_DEPENDENCE.
 *
 * The model-specific DB-coherence kinds are preprocessor-selected: MRNEW builds
 * define ARTS_PROTOCOL_MRNEW plus exactly one of ARTS_TIMING_{EAGER,LAZY};
 * MRMW builds define ARTS_PROTOCOL_MRMW alone.  Each build's enum
 * (and the mirroring g_ooo_table) carries only that model's OOO_DB_* kinds.
 * OOO_KIND_COUNT is therefore per-model — sound because every TU in one build
 * sees the same model define. */
enum arts_ooo_kind {
  /* ===== Model-agnostic — always active (all Cat B) ===== */
  OOO_EVENT_SATISFY_SLOT, /* → arts_handler_event_satisfy_slot */
  OOO_EDT_SATISFY_SLOT, /* → arts_handler_edt_satisfy_slot (mode-discriminated;
                           DB_MODE_PTR carries an inline payload trailing the
                           args struct) */
  OOO_EVENT_ADD_DEPENDENCE, /* → arts_handler_event_add_dependence */
  /* Lifecycle destroy replay kinds (before-create wire reorder): a DESTROY that
   * reaches home ahead of the object's CREATE defers here and replays once the
   * create handler installs+drains. */
  OOO_EDT_DESTROY,   /* → arts_handler_edt_destroy */
  OOO_EVENT_DESTROY, /* → arts_handler_event_destroy */
  OOO_DB_DESTROY, /* → arts_handler_db_destroy — no dep/lazy_install gate, so a
                     create/destroy reorder can land DESTROY before home CREATE
                   */

/* ===== Model-specific DB coherence — exactly one model active =====
 * Re-issue the wire handler once the home db_s/cache is installed: a
 * remote-created DB's lazy_install cache can fire a request/writeback before
 * that DB's home CREATE arrives, so the message reaches home with db_s not yet
 * installed ⇒ OoO push, replayed on the CREATE handler's drain. */
#if defined(ARTS_TIMING_EAGER)
  OOO_DB_ACQUIRE, /* → arts_db_acquire_replay_dep (re-attempts the one deferred
                     local dep; pushed by arts_db_acquire_all's per-dep 3-way)
                   */
  OOO_DB_SNAPSHOT_REQUEST,  /* → arts_handler_db_snapshot_request @ home */
  OOO_DB_OWNERSHIP_REQUEST, /* → arts_handler_db_ownership_request @ home */
  /* NO OOO_DB_OWNERSHIP_INVALIDATE — the eager protocol no longer defers
   * INVALIDATE: rw_holder is flipped only at the post-install CONFIRM (same as
   * lazy), so the target is provably installed when INVALIDATE arrives and the
   * dispatcher/self-send call the body directly. */
  OOO_DB_WRITEBACK, /* → arts_handler_db_writeback @ home */
#elif defined(ARTS_TIMING_LAZY)
  OOO_DB_ACQUIRE, /* → arts_db_acquire_replay_dep (re-attempts the one deferred
                     local dep; pushed by arts_db_acquire_all's per-dep 3-way)
                   */
  OOO_DB_SNAPSHOT_REQUEST,  /* → arts_handler_db_snapshot_request @ home */
  OOO_DB_OWNERSHIP_REQUEST, /* → arts_handler_db_ownership_request @ home */
/* NO OOO_DB_OWNERSHIP_INVALIDATE — the lazy protocol never defers INVALIDATE
 * (home publishes the target rw_holder only after that rank's cache install,
 * so the target is provably installed; the dispatcher/self-send call the body
 * directly).
 * NO OOO_DB_WRITEBACK — the lazy protocol has no synchronous writeback (the
 * dispatcher fatals on the WRITEBACK wire message). */
#elif defined(ARTS_PROTOCOL_MRMW)
  OOO_DB_ACQUIRE, /* → arts_db_acquire_replay_dep (re-attempts the one deferred
                     local dep; pushed by arts_db_acquire_all's per-dep 3-way)
                   */
  OOO_DB_SNAPSHOT_REQUEST, /* → arts_handler_db_snapshot_request @ home */
  OOO_DB_WRITEBACK,        /* → arts_handler_db_writeback @ home (no ownership
                              transfer) */
#else
#error                                                                         \
    "exactly one of ARTS_TIMING_{EAGER,LAZY} or ARTS_PROTOCOL_MRMW must be defined"
#endif

  OOO_KIND_COUNT /* sentinel — g_ooo_table size (per-model) */
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

/* EDT satisfy args (OOO_EDT_SATISFY_SLOT) — mode-discriminated.  For
 * mode == DB_MODE_PTR the inline payload (size bytes) trails this struct in the
 * blob (args_size == sizeof(this) + size); for all other modes size == 0 and no
 * payload trails (a GUID/value reference only).  The handler locates the
 * trailing payload exactly as the satisfy core already branches on mode, so one
 * kind covers both the reference and the inline-payload delivery. */
struct arts_ooo_args_edt_satisfy_s {
  arts_guid_t
      edt_guid; /* re-signal target (may differ from the deferred-on
                   slot, e.g. GPU LC invalidation defers on the wrapper) */
  arts_guid_t data_guid;
  uint32_t slot;
  arts_db_access_mode_t mode;
  uint32_t size; /* DB_MODE_PTR inline-payload byte count (0 otherwise) */
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

struct arts_ooo_args_db_acquire_s {
  struct arts_edt_s *edt;
  arts_guid_t db_guid;
  uint32_t slot;
};

/* Coherence replay args — re-issue the wire handler once the home db_s/cache
 * is installed.  First-class fields are reconstructed into a stack packet by
 * the handler.  The home FIFO records only the requester rank (MRNEW/MRSW
 * order ownership rank-by-rank). */
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
  uint64_t data_size;
};

/* DB ownership invalidate: carries the fields the wire
 * arts_msg_ownership_invalidate_packet_s delivers (db_guid + new_owner_rank).
 * The eager protocol ignores new_owner_rank; the lazy protocol uses it as the
 * TRANSFER_OWNERSHIP target. */
struct arts_ooo_args_db_ownership_invalidate_s {
  arts_guid_t db_guid;
  unsigned int new_owner_rank;
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
typedef void (*arts_ooo_handler_fn_t)(void *item, void *args);

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
 * GPU-LC-invalidation path which must hold a signal until the wrapper EDT's
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

#ifdef __cplusplus
}
#endif

#endif
