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
#ifndef ARTS_GAS_OUT_OF_ORDER_H
#define ARTS_GAS_OUT_OF_ORDER_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/runtime_types.h"
#include "arts/utils/lockfree_lifo.h" /* arts_lf_link_t */

struct arts_route_item_s;

/* OoO replay kind.  Identifies which g_ooo_table[] handler replays a deferred
 * operation once its target object is installed in the route table. */
enum arts_out_of_order_type {
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
  OOO_KIND_COUNT /* sentinel — g_ooo_table size */
};
typedef enum arts_out_of_order_type ooo_kind_t;

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
  bool inc;
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
  uint64_t waiter_addr;
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

#ifdef __cplusplus
}
#endif

#endif
