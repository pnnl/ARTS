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
#include "arts/gas/out_of_order.h"

#include "arts/compute/edt.h"
#include "arts/gas/route_table.h"
#include "arts/memory/db.h"
#include "arts/remote/handler.h"
#include "arts/runtime_state.h"
#include "arts/runtime_types.h"
#include "arts/sync/epoch.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/malloc.h"

#ifdef ARTS_COHERENCE_INTEGRATED
/* Future work will define ARTS_COHERENCE_INTEGRATED, include the
 * coherence_handlers.c sources in the build, and add the missing
 * arts_remote_lock_req_packet_s / arts_remote_get_data_packet_s /
 * arts_remote_destroy_req_packet_s / arts_remote_writeback_packet_s
 * declarations to arts/transport/protocol.h.  Until then the OO_COH_*
 * dispatch arms below are guarded out so the runtime build stays
 * clean.  See spec §4.8 and the OoO design notes. */
#include "arts/memory/coherence_handlers.h"
#endif

#include <string.h>

struct oo_signal_edt_s {
  enum arts_out_of_order_type type;
  arts_guid_t edt_packet;
  arts_guid_t data_guid;
  uint32_t slot;
  arts_db_access_mode_t mode;
};

struct oo_db_request_satisfy_s {
  enum arts_out_of_order_type type;
  struct arts_edt_s *edt;
  uint32_t slot;
  bool inc;
};

struct oo_add_dependence_s {
  enum arts_out_of_order_type type;
  arts_guid_t source;
  arts_guid_t destination;
  uint32_t slot;
  arts_guid_t data;
  arts_db_access_mode_t mode;
};

struct oo_event_satisfy_slot_s {
  enum arts_out_of_order_type type;
  arts_guid_t event_guid;
  arts_guid_t data_guid;
  uint32_t slot;
};

struct oo_handle_ready_edt_s {
  enum arts_out_of_order_type type;
  struct arts_edt_s *edt;
};

struct oo_get_from_db_s {
  enum arts_out_of_order_type type;
  arts_guid_t edt_guid;
  arts_guid_t db_guid;
  unsigned int slot;
  unsigned int offset;
  unsigned int size;
};

struct oo_signal_edt_ptr_s {
  enum arts_out_of_order_type type;
  arts_guid_t edt_guid;
  arts_guid_t db_guid;
  void *ptr;
  unsigned int size;
  unsigned int slot;
};

struct oo_put_in_db_s {
  enum arts_out_of_order_type type;
  void *ptr;
  arts_guid_t edt_guid;
  arts_guid_t db_guid;
  arts_guid_t epoch_guid;
  unsigned int slot;
  unsigned int offset;
  unsigned int size;
};

struct oo_epoch_s {
  enum arts_out_of_order_type type;
  arts_guid_t guid;
};

struct oo_epoch_send_s {
  enum arts_out_of_order_type type;
  arts_guid_t guid;
  unsigned int source;
  unsigned int dest;
};

struct oo_generic_s {
  enum arts_out_of_order_type type;
};

/*
 * arts_out_of_order_handler — Replay a deferred operation.
 *
 * When an operation arrives before its target object exists in the route
 * table (e.g., signal to an EDT that hasn't been created yet), it is
 * queued as an OO entry.  Once the target is inserted, this handler
 * replays each queued operation.
 *
 * The switch dispatches by OO type to the appropriate runtime function.
 */
inline void arts_out_of_order_handler(void *handle_me, void *memory_ptr) {
  struct oo_generic_s *type_ptr = (struct oo_generic_s *)handle_me;
  ARTS_DEBUG("OO handler: dispatching type=%d", type_ptr->type);
  switch (type_ptr->type) {
  case OO_SIGNAL_EDT: {
    struct oo_signal_edt_s *edt = (struct oo_signal_edt_s *)handle_me;
    internal_signal_edt(edt->edt_packet, edt->slot, edt->data_guid, edt->mode,
                        NULL, 0);
    break;
  }
  case OO_EVENT_SATISFY_SLOT: {
    struct oo_event_satisfy_slot_s *event =
        (struct oo_event_satisfy_slot_s *)handle_me;
    arts_event_satisfy_slot(event->event_guid, event->data_guid, event->slot);
    break;
  }
  case OO_ADD_DEPENDENCE: {
    struct oo_add_dependence_s *dep = (struct oo_add_dependence_s *)handle_me;
    arts_add_dependence(dep->source, dep->destination, dep->slot, dep->mode);
    break;
  }
  case OO_HANDLE_READY_EDT: {
    struct oo_handle_ready_edt_s *ready_edt =
        (struct oo_handle_ready_edt_s *)handle_me;
    arts_handle_ready_edt(ready_edt->edt);
    break;
  }
  case OO_DB_REQUEST_SATISFY: {
    struct oo_db_request_satisfy_s *req =
        (struct oo_db_request_satisfy_s *)handle_me;
    ARTS_DEBUG("FILL %lu %u %p", req->edt, req->slot, memory_ptr);
    arts_db_request_callback(req->edt, req->slot,
                             (struct arts_db_s *)memory_ptr);
    break;
  }
  case OO_GET_FROM_DB: {
    struct oo_get_from_db_s *req = (struct oo_get_from_db_s *)handle_me;
    arts_db_get(
        req->edt_guid, req->db_guid, req->slot, req->offset, req->size,
        &(arts_db_op_hint_t){.rank = arts_global_rank_id, .epoch = NULL_GUID});
    break;
  }
  case OO_SIGNAL_EDT_PTR: {
    struct oo_signal_edt_ptr_s *req = (struct oo_signal_edt_ptr_s *)handle_me;
    internal_signal_edt(req->edt_guid, req->slot, NULL_GUID, DB_MODE_PTR,
                        req->ptr, req->size);
    arts_free(req->ptr);
    break;
  }
  case OO_PUT_IN_DB: {
    struct oo_put_in_db_s *req = (struct oo_put_in_db_s *)handle_me;
    internal_put_in_db(req->ptr, req->edt_guid, req->db_guid, req->slot,
                       req->offset, req->size, req->epoch_guid,
                       arts_global_rank_id);
    arts_free(req->ptr);
    break;
  }
  case OO_EPOCH_ACTIVE: {
    //            ARTS_INFO("ooActveFire");
    struct oo_epoch_s *req = (struct oo_epoch_s *)handle_me;
    increment_active_epoch(req->guid);
    break;
  }
  case OO_EPOCH_FINISH: {
    //            ARTS_INFO("ooFinishFire");
    struct oo_epoch_s *req = (struct oo_epoch_s *)handle_me;
    increment_finished_epoch(req->guid);
    break;
  }
  case OO_EPOCH_SEND: {
    //            ARTS_INFO("ooEpochSendFire");
    struct oo_epoch_send_s *req = (struct oo_epoch_send_s *)handle_me;
    send_epoch(req->guid, req->source, req->dest);
    break;
  }
  case OO_EPOCH_INC_QUEUE: {
    struct oo_epoch_s *req = (struct oo_epoch_s *)handle_me;
    increment_queue_epoch(req->guid);
    break;
  }
#ifdef ARTS_COHERENCE_INTEGRATED
  case OO_COH_LOCK_REQ: {
    /* Re-issue handler — cache is now installed (DB_CREATE arrived
     * after the original race-arrived LOCK_REQ was deferred). */
    struct oo_coh_lock_req_s *req = (struct oo_coh_lock_req_s *)handle_me;
    struct arts_remote_lock_req_packet_s p;
    p.header.rank = req->requester;
    p.db_guid = req->db_guid;
    arts_coh_handle_lock_req(&p);
    break;
  }
  case OO_COH_GET_DATA: {
    struct oo_coh_get_data_s *req = (struct oo_coh_get_data_s *)handle_me;
    struct arts_remote_get_data_packet_s p;
    p.header.rank = req->requester;
    p.db_guid = req->db_guid;
    p.waiter_addr = req->waiter_addr;
    arts_coh_handle_get_data(&p);
    break;
  }
  case OO_COH_DESTROY_REQ: {
    struct oo_coh_destroy_req_s *req = (struct oo_coh_destroy_req_s *)handle_me;
    struct arts_remote_destroy_req_packet_s p;
    p.header.rank = req->requester;
    p.db_guid = req->db_guid;
    arts_coh_handle_destroy_req(&p);
    break;
  }
  case OO_COH_WRITEBACK: {
    struct oo_coh_writeback_s *req = (struct oo_coh_writeback_s *)handle_me;
    struct arts_remote_writeback_packet_s p;
    p.header.rank = req->releaser;
    p.db_guid = req->db_guid;
    p.version = req->version;
    p.seq = req->seq;
    p.flag = req->flag;
    arts_coh_handle_writeback(&p, req->data, req->data_size);
    break;
  }
#else
  case OO_COH_LOCK_REQ:
  case OO_COH_GET_DATA:
  case OO_COH_DESTROY_REQ:
  case OO_COH_WRITEBACK:
    /* Coherence handlers not yet wired into the build.
     * Producers gated by the same ifdef in coherence_handlers.c, so we
     * should never observe these tags here.  Fall through to the error
     * branch if they ever arrive. */
    ARTS_INFO("OO Handler: OO_COH_* tag observed without coherence build");
    break;
#endif
  default:
    ARTS_INFO("OO Handler Error");
  }
  arts_free(handle_me);
}

/*
 * arts_oo_dispatch_destroyed_cb — drain callback used by
 * arts_route_table_drop_oo when a DB is being destroyed.  Wakes parked
 * EDT waiters with NULL_DB so the destroyed-DB semantic propagates
 * (depv[slot].guid = NULL_GUID, ptr = NULL, depc_needed--), then frees
 * the payload.
 *
 * Without this wake, EDTs that called arts_out_of_order_handle_db_request
 * before the destroy sit in the OoO list forever (their DB will never be
 * installed because destroy happened first).  Other OoO types reference
 * the destroyed DB indirectly; for those, the payload is freed but no
 * waiter wake is needed (their consumer is itself blocked on the same
 * DB and reaches the same destroyed state via its own dispatch).
 */
void arts_oo_dispatch_destroyed_cb(void *data, void *ctx) {
  (void)ctx;
  oo_type_t *type = (oo_type_t *)data;
  switch (*type) {
  case OO_DB_REQUEST_SATISFY: {
    struct oo_db_request_satisfy_s *req =
        (struct oo_db_request_satisfy_s *)data;
    arts_db_request_callback(req->edt, req->slot, NULL);
    break;
  }
  default:
    /* Drop other types silently -- their consumers are app bugs (use
     * after destroy) and waking them with NULL would deliver wrong data. */
    break;
  }
  arts_free(data);
}

/*
 * arts_out_of_order_signal_edt — Queue an EDT signal for deferred delivery.
 *
 * If the target EDT's GUID is still in RESERVED state in the route table,
 * the signal is stored in the OO list.  If the item is already AVAILABLE
 * (race: created between our check and now), the signal is delivered
 * immediately and the OO entry is freed.
 */
void arts_out_of_order_signal_edt(arts_guid_t wait_on, arts_guid_t edt_packet,
                                  arts_guid_t data_guid, uint32_t slot,
                                  arts_db_access_mode_t mode, bool force) {
  struct oo_signal_edt_s *edt =
      (struct oo_signal_edt_s *)arts_malloc(sizeof(struct oo_signal_edt_s));
  edt->type = OO_SIGNAL_EDT;
  edt->edt_packet = edt_packet;
  edt->data_guid = data_guid;
  edt->slot = slot;
  edt->mode = mode;
  if (force) {
    arts_route_table_add_oo_existing(wait_on, edt, false);
  } else {
    bool res = arts_route_table_add_oo(wait_on, edt, false);
    if (!res) {
      internal_signal_edt(edt_packet, slot, data_guid, mode, NULL, 0);
      arts_free(edt);
    }
  }
}

void arts_out_of_order_event_satisfy_slot(arts_guid_t wait_on,
                                          arts_guid_t event_guid,
                                          arts_guid_t data_guid, uint32_t slot,
                                          bool force) {
  struct oo_event_satisfy_slot_s *event =
      (struct oo_event_satisfy_slot_s *)arts_malloc(
          sizeof(struct oo_event_satisfy_slot_s));
  event->type = OO_EVENT_SATISFY_SLOT;
  event->event_guid = event_guid;
  event->data_guid = data_guid;
  event->slot = slot;
  bool res;
  if (force) {
    arts_route_table_add_oo_existing(wait_on, event, false);
  } else {
    bool res = arts_route_table_add_oo(wait_on, event, false);
    if (!res) {
      arts_event_satisfy_slot(event_guid, data_guid, slot);
      arts_free(event);
    }
  }
}

void arts_out_of_order_add_dependence(arts_guid_t source,
                                      arts_guid_t destination, uint32_t slot,
                                      arts_db_access_mode_t mode,
                                      arts_guid_t wait_on) {
  struct oo_add_dependence_s *dep = (struct oo_add_dependence_s *)arts_malloc(
      sizeof(struct oo_add_dependence_s));
  dep->type = OO_ADD_DEPENDENCE;
  dep->source = source;
  dep->destination = destination;
  dep->slot = slot;
  dep->mode = mode;
  bool res = arts_route_table_add_oo(wait_on, dep, false);
  if (!res) {
    arts_add_dependence(source, destination, slot, mode);
    arts_free(dep);
  }
}

void arts_out_of_order_handle_ready_edt(arts_guid_t trigger_guid,
                                        struct arts_edt_s *edt) {
  struct oo_handle_ready_edt_s *ready_edt =
      (struct oo_handle_ready_edt_s *)arts_malloc(
          sizeof(struct oo_handle_ready_edt_s));
  ready_edt->type = OO_HANDLE_READY_EDT;
  ready_edt->edt = edt;
  bool res = arts_route_table_add_oo(trigger_guid, ready_edt, false);
  if (!res) {
    arts_handle_ready_edt(edt);
    arts_free(ready_edt);
  }
}

/*
 * arts_out_of_order_handle_db_request — Queue a DB acquisition for deferred
 *   resolution when the DB does not yet exist in the route table.
 *
 * If the DB becomes available before the OO entry is added (race), the
 * callback fires immediately.
 */
void arts_out_of_order_handle_db_request(arts_guid_t db_guid,
                                         struct arts_edt_s *edt,
                                         unsigned int slot, bool inc) {
  ARTS_DEBUG("OO db_request: DB[Guid:%lu] -> EDT[Guid:%lu] slot=%u inc=%d",
             db_guid, edt->current_edt, slot, inc);
  struct oo_db_request_satisfy_s *req =
      (struct oo_db_request_satisfy_s *)arts_malloc(
          sizeof(struct oo_db_request_satisfy_s));
  req->type = OO_DB_REQUEST_SATISFY;
  req->edt = edt;
  req->slot = slot;
  bool res = arts_route_table_add_oo(db_guid, req, inc);
  if (!res) {
    ARTS_DEBUG(
        "OO db_request: DB[Guid:%lu] already available — immediate callback",
        db_guid);
    struct arts_db_s *db = arts_route_table_lookup_db_safe(db_guid);
    arts_db_request_callback(req->edt, req->slot, db);
    if (db) {
      arts_route_table_release(db_guid);
    }
    arts_free(req);
  }
}

/* arts_out_of_order_handle_db_request_with_oo_list removed: legacy
 * arts_out_of_order_list_s + arts_route_table_get_oo_list path is gone,
 * no callers, replaced by arts_route_table_add_oo_ex. */

void arts_out_of_order_get_from_db(arts_guid_t edt_guid, arts_guid_t db_guid,
                                   unsigned int slot, unsigned int offset,
                                   unsigned int size) {
  struct oo_get_from_db_s *req =
      (struct oo_get_from_db_s *)arts_malloc(sizeof(struct oo_get_from_db_s));
  req->type = OO_GET_FROM_DB;
  req->edt_guid = edt_guid;
  req->db_guid = db_guid;
  req->slot = slot;
  req->offset = offset;
  req->size = size;
  bool res = arts_route_table_add_oo(db_guid, req, false);
  if (!res) {
    arts_db_get(
        req->edt_guid, req->db_guid, req->slot, req->offset, req->size,
        &(arts_db_op_hint_t){.rank = arts_global_rank_id, .epoch = NULL_GUID});
    arts_free(req);
  }
}

void arts_out_of_order_signal_edt_with_ptr(arts_guid_t edt_guid,
                                           arts_guid_t db_guid, void *ptr,
                                           unsigned int size,
                                           unsigned int slot) {
  struct oo_signal_edt_ptr_s *req = (struct oo_signal_edt_ptr_s *)arts_malloc(
      sizeof(struct oo_signal_edt_ptr_s));
  req->type = OO_SIGNAL_EDT_PTR;
  req->edt_guid = edt_guid;
  req->db_guid = db_guid;
  req->size = size;
  req->slot = slot;
  if (size > 0) {
    req->ptr = arts_malloc(size);
    memcpy(req->ptr, ptr, size);
  } else {
    req->ptr = ptr;
  }
  bool res = arts_route_table_add_oo(edt_guid, req, false);
  if (!res) {
    internal_signal_edt(req->edt_guid, req->slot, NULL_GUID, DB_MODE_PTR,
                        req->ptr, req->size);
    arts_free(req->ptr);
    arts_free(req);
  }
}

void arts_out_of_order_put_in_db(void *ptr, arts_guid_t edt_guid,
                                 arts_guid_t db_guid, unsigned int slot,
                                 unsigned int offset, unsigned int size,
                                 arts_guid_t epoch_guid) {
  struct oo_put_in_db_s *req =
      (struct oo_put_in_db_s *)arts_malloc(sizeof(struct oo_put_in_db_s));
  req->type = OO_PUT_IN_DB;
  req->ptr = ptr;
  req->edt_guid = edt_guid;
  req->db_guid = db_guid;
  req->slot = slot;
  req->offset = offset;
  req->size = size;
  req->epoch_guid = epoch_guid;
  bool res = arts_route_table_add_oo(db_guid, req, false);
  if (!res) {
    internal_put_in_db(req->ptr, req->edt_guid, req->db_guid, req->slot,
                       req->offset, req->size, req->epoch_guid,
                       arts_global_rank_id);
    arts_free(req->ptr);
    arts_free(req);
  }
}

void arts_out_of_order_inc_active_epoch(arts_guid_t epoch_guid) {
  struct oo_epoch_s *req =
      (struct oo_epoch_s *)arts_malloc(sizeof(struct oo_epoch_s));
  req->type = OO_EPOCH_ACTIVE;
  req->guid = epoch_guid;
  bool res = arts_route_table_add_oo(epoch_guid, req, false);
  if (!res) {
    increment_active_epoch(epoch_guid);
    arts_free(req);
  }
}

void arts_out_of_order_inc_finished_epoch(arts_guid_t epoch_guid) {
  struct oo_epoch_s *req =
      (struct oo_epoch_s *)arts_malloc(sizeof(struct oo_epoch_s));
  req->type = OO_EPOCH_FINISH;
  req->guid = epoch_guid;
  bool res = arts_route_table_add_oo(epoch_guid, req, false);
  if (!res) {
    increment_finished_epoch(epoch_guid);
    arts_free(req);
  }
}

void arts_out_of_order_send_epoch(arts_guid_t epoch_guid, unsigned int source,
                                  unsigned int dest) {
  struct oo_epoch_send_s *req =
      (struct oo_epoch_send_s *)arts_malloc(sizeof(struct oo_epoch_send_s));
  req->type = OO_EPOCH_SEND;
  req->source = source;
  req->dest = dest;
  bool res = arts_route_table_add_oo(epoch_guid, req, false);
  if (!res) {
    send_epoch(epoch_guid, source, dest);
    arts_free(req);
  }
}

void arts_out_of_order_inc_queue_epoch(arts_guid_t epoch_guid) {
  struct oo_epoch_s *req =
      (struct oo_epoch_s *)arts_malloc(sizeof(struct oo_epoch_s));
  req->type = OO_EPOCH_INC_QUEUE;
  req->guid = epoch_guid;
  bool res = arts_route_table_add_oo(epoch_guid, req, false);
  if (!res) {
    increment_queue_epoch(epoch_guid);
    arts_free(req);
  }
}
