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
#include "arts/runtime/sync/event_functions.h"

#include "arts.h"
#include "arts/gas/guid.h"
#include "arts/gas/out_of_order.h"
#include "arts/gas/route_table.h"
#include "arts/introspection/metrics.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/compute/edt_functions.h"
#include "arts/runtime/network/remote_functions.h"
#include "arts/system/arts_print.h"
#include "arts/system/debug.h"
#include "arts/utils/atomics.h"
#include "arts/utils/link_list.h"

#include <assert.h>
#include <time.h>

extern __thread struct arts_edt *current_edt;

bool arts_event_create_internal(arts_guid_t *guid, unsigned int route,
                             unsigned int dependent_count,
                             unsigned int latch_count, bool destroy_on_fire,
                             arts_guid_t event_data) {
  unsigned int event_size =
      sizeof(struct arts_event) + (sizeof(struct arts_dependent) * dependent_count);
  void *event_packet = ARTS_CALLOC_WITH_TYPE(1, event_size, ARTS_METRIC_EVENT_MEMORY_SIZE);

  if (event_size) {
    struct arts_event *event = (struct arts_event *)event_packet;
    event->header.type = ARTS_EVENT;
    event->header.size = event_size;
    event->dependent_count = 0;
    event->dependent.size = dependent_count;
    event->latch_count = latch_count;
    event->destroy_on_fire = (destroy_on_fire) ? dependent_count : -1;
    event->data = event_data;

    if (route == arts_global_rank_id) {
      if (*guid) {
        /* For labeled GUIDs, use race-safe addition since multiple threads/ranks
         * may try to create the same labeled event concurrently. This matches
         * the behavior in arts_remote_handle_event_move which also uses
         * arts_route_table_add_item_race for consistency. */
        if (arts_route_table_add_item_race(event_packet, *guid, arts_global_rank_id, false)) {
          arts_route_table_fire_oo(*guid, arts_out_of_order_handler);
        } else {
          /* Event already exists - free the allocated memory */
          arts_free(event_packet);
          return false;
        }
      } else {
        *guid = arts_guid_create_for_rank(route, ARTS_EVENT);
        arts_route_table_add_item(event_packet, *guid, arts_global_rank_id, false);
      }
    } else {
      arts_remote_memory_move(route, *guid, event_packet, event_size,
                           ARTS_REMOTE_EVENT_MOVE_MSG, arts_free);
}

    return true;
  }
  return false;
}

arts_guid_t arts_event_create(unsigned int route, unsigned int latch_count) {
  EVENT_CREATE_COUNTER_START();
  if (route == -1) {
    route = arts_global_rank_id;
}
  arts_guid_t guid = NULL_GUID;
  arts_event_create_internal(&guid, route, INITIAL_DEPENDENT_SIZE, latch_count,
                          false, NULL_GUID);
  EVENT_CREATE_COUNTER_STOP();
  return guid;
}

arts_guid_t arts_event_create_with_guid(arts_guid_t guid, unsigned int latch_count) {
  EVENT_CREATE_COUNTER_START();
  unsigned int route = arts_guid_get_rank(guid);
  bool ret = arts_event_create_internal(&guid, route, INITIAL_DEPENDENT_SIZE,
                                     latch_count, false, NULL_GUID);
  EVENT_CREATE_COUNTER_STOP();
  return (ret) ? guid : NULL_GUID;
}

void arts_event_free(struct arts_event *event) {
  struct arts_dependent_list *trail;
  struct arts_dependent_list *current = event->dependent.next;
  while (current) {
    trail = current;
    current = current->next;
    arts_free(trail);
  }
  arts_free(event);
}

void arts_event_destroy(arts_guid_t guid) {
  struct arts_event *event = (struct arts_event *)arts_route_table_lookup_item(guid);
  if (event != NULL) {
    arts_route_table_remove_item(guid);
    arts_event_free(event);
  }
}

void arts_event_satisfy_slot(arts_guid_t event_guid, arts_guid_t data_guid,
                          uint32_t slot) {
  SIGNAL_EVENT_COUNTER_START();
  if (current_edt && current_edt->invalidateCount > 0) {
    arts_out_of_order_event_satisfy_slot(current_edt->current_edt, event_guid, data_guid,
                                   slot, true);
    return;
  }
  ARTS_INFO("Signal Event:%u, Data:%u at %u", event_guid, data_guid, slot);
  struct arts_event *event =
      (struct arts_event *)arts_route_table_lookup_item(event_guid);
  if (!event) {
    unsigned int rank = arts_guid_get_rank(event_guid);
    if (rank != arts_global_rank_id) {
      arts_remote_event_satisfy_slot(event_guid, data_guid, slot);
    } else {
      arts_out_of_order_event_satisfy_slot(event_guid, event_guid, data_guid, slot,
                                     false);
    }
  } else {
    if (event->fired) {
      ARTS_INFO("ARTS_EVENT_LATCH_T already fired guid: %lu data: %lu slot: %u",
                event_guid, data_guid, slot);
      arts_debug_generate_seg_fault();
    }

    unsigned int res = 0U;
    if (slot == ARTS_EVENT_LATCH_INCR_SLOT) {
      res = arts_atomic_add(&event->latch_count, 1U);
    } else if (slot == ARTS_EVENT_LATCH_DECR_SLOT) {
      if (data_guid != NULL_GUID) {
        event->data = data_guid;
}
      res = arts_atomic_sub(&event->latch_count, 1U);
    } else {
      ARTS_INFO("Bad latch slot %u", slot);
      arts_debug_generate_seg_fault();
    }

    /// When the latch count reaches 0, fire the event
    if (!res) {
      /// If the event is already fired, we should not fire it again
      if (arts_atomic_swap_bool(&event->fired, true)) {
        ARTS_PRINTF("ARTS_EVENT_LATCH_T already fired guid: %lu data: %lu slot: %u",
               event_guid, data_guid, slot);
        arts_debug_generate_seg_fault();
      }
      /// If the event is not fired, we need to fire it
      else {
        struct arts_dependent_list *dependent_list = &event->dependent;
        struct arts_dependent *dependent = event->dependent.dependents;
        int i;
        int j;
        /// Capture current state
        unsigned int last_known = arts_atomic_fetch_add(&event->dependent_count, 0U);
        event->pos = last_known + 1;
        i = 0;
        int total_size = 0;
        /// Process all dependents up to last_known
        while (i < last_known) {
          j = i - total_size;
          while (i < last_known && j < dependent_list->size) {
            while (!dependent[j].doneWriting) {
              ;
}
            if (dependent[j].type == ARTS_EDT) {
              arts_signal_edt(dependent[j].addr, dependent[j].slot, event->data);
            } else if (dependent[j].type == ARTS_EVENT) {
              SIGNAL_EVENT_COUNTER_STOP();
              arts_event_satisfy_slot(dependent[j].addr, event->data,
                                   dependent[j].slot);
              SIGNAL_EVENT_COUNTER_START();
            } else if (dependent[j].type == ARTS_CALLBACK) {
              arts_edt_dep_t arg;
              arg.guid = event->data;
              arg.ptr = arts_route_table_lookup_item(event->data);
              arg.mode = ARTS_NULL;
              dependent[j].callback_t(arg);
            }
            j++;
            i++;
          }
          total_size += dependent_list->size;
          while (i < last_known && dependent_list->next == NULL) {
            ;
}
          dependent_list = dependent_list->next;
          dependent = dependent_list->dependents;
        }
        if (!event->destroy_on_fire) {
          arts_event_free(event);
          arts_route_table_remove_item(event_guid);
        }
      }
    }
  }
  ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_EVENT_SIGNAL_THROUGHPUT, ARTS_METRIC_THREAD, 1);
  SIGNAL_EVENT_COUNTER_STOP();
}

struct arts_dependent *arts_dependent_get(struct arts_dependent_list *head,
                                       int position) {
  struct arts_dependent_list *list = head;
  volatile struct arts_dependent_list *temp;

  while (1) {
    /// If the position is greater than the size of the list, we need to
    /// allocate a new list
    if (position >= list->size) {
      if (position - list->size == 0) {
        if (list->next == NULL) {
          temp = (volatile struct arts_dependent_list *)arts_calloc(
              1, sizeof(struct arts_dependent_list) +
                     (sizeof(struct arts_dependent) * list->size * 2));
          temp->size = list->size * 2;
          list->next = (struct arts_dependent_list *)temp;
        }
      }

      // EXPONENTIONAL BACK OFF THIS
      while (list->next == NULL) {
      }

      position -= list->size;
      list = list->next;
    } else {
      break;
}
  }
  return list->dependents + position;
}

void arts_add_dependence(arts_guid_t source, arts_guid_t destination,
                       uint32_t slot) {
  ARTS_INFO("Add Dependence from %u to %u at %u", source, destination, slot);
  arts_type_t mode = arts_guid_get_type(destination);
  struct arts_header *source_header =
      (struct arts_header *)arts_route_table_lookup_item(source);
  if (source_header == NULL) {
    unsigned int rank = arts_guid_get_rank(source);
    if (rank != arts_global_rank_id) {
      arts_remote_add_dependence(source, destination, slot, rank);
    } else {
      arts_out_of_order_add_dependence(source, destination, slot, mode, source);
    }
    return;
  }

  struct arts_event *event = (struct arts_event *)source_header;
  if (mode == ARTS_EDT) {
    struct arts_dependent_list *dependent_list = &event->dependent;
    struct arts_dependent *dependent;
    unsigned int position = arts_atomic_fetch_add(&event->dependent_count, 1U);
    dependent = arts_dependent_get(dependent_list, position);
    dependent->type = ARTS_EDT;
    dependent->addr = destination;
    dependent->slot = slot;
    COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
    dependent->doneWriting = true;

    unsigned int destroy_event = (event->destroy_on_fire != -1)
                                    ? arts_atomic_sub(&event->destroy_on_fire, 1U)
                                    : 1;
    if (event->fired) {
      while (event->pos == 0) {
        ;
}
      if (position >= event->pos - 1) {
        arts_signal_edt(destination, slot, event->data);
        if (!destroy_event) {
          arts_event_free(event);
          arts_route_table_remove_item(source);
        }
      }
    }
  } else if (mode == ARTS_EVENT) {
    struct arts_dependent_list *dependent_list = &event->dependent;
    struct arts_dependent *dependent;
    unsigned int position = arts_atomic_fetch_add(&event->dependent_count, 1U);
    dependent = arts_dependent_get(dependent_list, position);
    dependent->type = ARTS_EVENT;
    dependent->addr = destination;
    dependent->slot = slot;
    COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
    dependent->doneWriting = true;

    unsigned int destroy_event = (event->destroy_on_fire != -1)
                                    ? arts_atomic_sub(&event->destroy_on_fire, 1U)
                                    : 1;
    if (event->fired) {
      while (event->pos == 0) {
        ;
}
      if (event->pos - 1 <= position) {
        arts_event_satisfy_slot(destination, event->data, slot);
        if (!destroy_event) {
          arts_event_free(event);
          arts_route_table_remove_item(source);
        }
      }
    }
  }
  }

void arts_add_local_event_callback(arts_guid_t source, event_callback_t callback_t) {
  struct arts_event *event =
      (struct arts_event *)arts_route_table_lookup_item(source);
  if (event && arts_guid_get_type(source) == ARTS_EVENT) {
    struct arts_dependent_list *dependent_list = &event->dependent;
    struct arts_dependent *dependent;
    unsigned int position = arts_atomic_fetch_add(&event->dependent_count, 1U);
    dependent = arts_dependent_get(dependent_list, position);
    dependent->type = ARTS_CALLBACK;
    dependent->callback_t = callback_t;
    dependent->addr = NULL_GUID;
    dependent->slot = 0;
    COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
    dependent->doneWriting = true;

    unsigned int destroy_event = (event->destroy_on_fire != -1)
                                    ? arts_atomic_sub(&event->destroy_on_fire, 1U)
                                    : 1;
    if (event->fired) {
      while (event->pos == 0) {
        ;
}
      if (event->pos - 1 <= position) {
        arts_edt_dep_t arg;
        arg.guid = event->data;
        arg.ptr = arts_route_table_lookup_item(event->data);
        arg.mode = ARTS_NULL;
        callback_t(arg);
        if (!destroy_event) {
          arts_event_free(event);
          arts_route_table_remove_item(source);
        }
      }
    }
  }
}

bool arts_is_event_fired(arts_guid_t event) {
  bool fired = false;
  struct arts_event *actual_event =
      (struct arts_event *)arts_route_table_lookup_item(event);
  if (actual_event) {
    fired = actual_event->fired;
}
  return fired;
}

/// Persistent events
struct arts_persistent_event_version *
arts_push_persistent_event_version(struct arts_persistent_event *event);

struct arts_link_list *arts_get_event_versions(struct arts_persistent_event *event) {
  if (event->versions != NULL) {
    return event->versions;
}
  event->versions = arts_link_list_group_new(1);
  struct arts_persistent_event_version *version =
      arts_push_persistent_event_version(event);
  version->dependent.next = NULL;

  assert(version != NULL);
  return event->versions;
}

struct arts_persistent_event_version *
arts_push_persistent_event_version(struct arts_persistent_event *event) {
  struct arts_link_list *versions = arts_get_event_versions(event);
  struct arts_persistent_event_version *next =
      (struct arts_persistent_event_version *)arts_link_list_new_item(
          (sizeof(struct arts_persistent_event_version) +
           (sizeof(struct arts_dependent) * INITIAL_DEPENDENT_SIZE)));
  next->latch_count = 0;
  next->dependent_count = 0;
  next->dependent.size = INITIAL_DEPENDENT_SIZE;
  struct arts_persistent_event_version *last = NULL;
  if (versions && versions->tailPtr) {
    last =
        (struct arts_persistent_event_version *)arts_link_list_get_tail_data(versions);
}

  if (last) {
    next->version = last->version + 1;
  } else {
    next->version = 0;
}
  arts_link_list_push_back(versions, next);
  return next;
}

struct arts_persistent_event_version *
arts_get_front_persistent_event_version(struct arts_persistent_event *event) {
  struct arts_persistent_event_version *v =
      (struct arts_persistent_event_version *)arts_link_list_get_front_data(
          arts_get_event_versions(event));
  return v;
}

struct arts_persistent_event_version *
arts_get_last_persistent_event_version(struct arts_persistent_event *event) {
  struct arts_persistent_event_version *v =
      (struct arts_persistent_event_version *)arts_link_list_get_tail_data(
          arts_get_event_versions(event));
  return v;
}

bool arts_persistent_event_create_internal(arts_guid_t *guid, unsigned int route,
                                       arts_guid_t event_data) {
  if (event_data == NULL_GUID) {
    ARTS_INFO("Event data is NULL_GUID for persistent event");
    arts_debug_generate_seg_fault();
  }
  const unsigned int event_size = sizeof(struct arts_persistent_event);
  void *event_packet =
      ARTS_CALLOC_WITH_TYPE(1, event_size, ARTS_METRIC_PERSISTENT_EVENT_MEMORY_SIZE);

  if (event_size) {
    struct arts_persistent_event *event =
        (struct arts_persistent_event *)event_packet;
    event->header.type = ARTS_PERSISTENT_EVENT;
    event->header.size = event_size;
    event->versions = NULL;
    event->data = event_data;
    event->lock = 0;

    if (route == arts_global_rank_id) {
      if (*guid) {
        arts_route_table_add_item(event_packet, *guid, arts_global_rank_id, false);
        arts_route_table_fire_oo(*guid, arts_out_of_order_handler);
      } else {
        *guid = arts_guid_create_for_rank(route, ARTS_PERSISTENT_EVENT);
        arts_route_table_add_item(event_packet, *guid, arts_global_rank_id, false);
      }
    } else {
      arts_remote_memory_move(route, *guid, event_packet, event_size,
                           ARTS_REMOTE_PERSISTENT_EVENT_MOVE_MSG, arts_free);
    }
    return true;
  }
  ARTS_INFO("Failed to create persistent event");
  return false;
}

bool arts_persistent_event_free_version(struct arts_persistent_event *event) {
  struct arts_link_list *versions = event->versions;
  assert(versions != NULL);
  bool last = true;
  arts_lock(&versions->lock);

  if (versions->headPtr != versions->tailPtr) {
    last = false;
}

  /// Get the top version
  struct arts_persistent_event_version *version =
      (struct arts_persistent_event_version *)(versions->headPtr + 1);
  assert(version != NULL);

  /// Free dependencies for this version
  struct arts_dependent_list *trail;
  struct arts_dependent_list *current = version->dependent.next;
  while (current) {
    trail = current;
    current = current->next;
    arts_free(trail);
  }
  version->dependent.next = NULL;

  /// Free the version
  if (last) {
    version->latch_count = 0;
    version->dependent_count = 0;
  } else {
    versions->headPtr = versions->headPtr->next;
    struct arts_link_list_item *item = ((struct arts_link_list_item *)version) - 1;
    arts_free(item);
  }

  arts_unlock(&versions->lock);
  return last;
}

arts_guid_t arts_persistent_event_create(unsigned int route,
                                     unsigned int latch_count,
                                     arts_guid_t data_guid) {
  PERSISTENT_EVENT_CREATE_COUNTER_START();
  if (route == -1) {
    route = arts_global_rank_id;
}
  arts_guid_t guid = NULL_GUID;
  arts_persistent_event_create_internal(&guid, route, data_guid);
  PERSISTENT_EVENT_CREATE_COUNTER_STOP();
  return guid;
}

void arts_persistent_event_destroy(arts_guid_t guid) {
  struct arts_persistent_event *event =
      (struct arts_persistent_event *)arts_route_table_lookup_item(guid);
  if (event != NULL) {
    arts_lock(&event->lock);
    arts_route_table_remove_item(guid);
    while (!arts_persistent_event_free_version(event)) {
      ;
}
    arts_unlock(&event->lock);
    arts_free(event);
  }
}

void arts_persistent_event_satisfy(arts_guid_t event_guid, uint32_t action,
                                bool lock) {
  SIGNAL_PERSISTENT_EVENT_COUNTER_START();
  if (current_edt && current_edt->invalidateCount > 0) {
    arts_out_of_order_persistent_event_satisfy_slot(current_edt->current_edt, event_guid,
                                             action, true);
    return;
  }
  struct arts_persistent_event *event =
      (struct arts_persistent_event *)arts_route_table_lookup_item(event_guid);
  if (!event) {
    unsigned int rank = arts_guid_get_rank(event_guid);
    if (rank != arts_global_rank_id) {
      arts_remote_persistent_event_satisfy_slot(event_guid, action, lock);
    } else {
      arts_out_of_order_persistent_event_satisfy_slot(event_guid, event_guid, action,
                                               false);
    }
  } else {
    if (lock) {
      arts_lock(&event->lock);
}
    if (event->data == NULL_GUID) {
      ARTS_DEBUG("Data: NULL_GUID, avoiding signaling");
      arts_debug_generate_seg_fault();
    }
    unsigned int res = -1;
    struct arts_persistent_event_version *version =
        arts_get_front_persistent_event_version(event);
    assert(version != NULL);
    if (action == ARTS_EVENT_LATCH_INCR_SLOT) {
      res = arts_atomic_fetch_add(&version->latch_count, 0U);
      if (res == 1) {
        ARTS_DEBUG(
            "Latch count is 1 for Event [Guid:%lu], creating new version",
            event_guid);
        version = arts_push_persistent_event_version(event);
        ARTS_DEBUG("Created Event [Guid:%lu, Version: %u]", version->version,
                   event_guid);
      }
      res = arts_atomic_add(&version->latch_count, 1U);
      ARTS_DEBUG("Increment Event [Guid:%lu, Latch Count: %d]", event_guid,
                 res);
    } else if (action == ARTS_EVENT_LATCH_DECR_SLOT) {
      res = arts_atomic_fetch_add(&version->latch_count, 0U);
      if (res == (unsigned int)-1) {
        ARTS_DEBUG(
            "Latch count is -1 for Event [Guid:%lu], creating new version",
            event_guid);
        version = arts_push_persistent_event_version(event);
        ARTS_DEBUG("Created version %u for Event [Guid:%lu, Version: %u]",
                   version->version, event_guid);
      }
      res = arts_atomic_sub(&version->latch_count, 1U);
      ARTS_DEBUG("Decrement Event [Guid:%lu, Latch Count: %d] ", event_guid,
                 res);
    } else if (action == ARTS_EVENT_UPDATE) {
      res = arts_atomic_fetch_add(&version->latch_count, 0U);
      ARTS_DEBUG("Update Event [Guid:%lu, Latch Count: %d] ", event_guid, res);
    } else {
      ARTS_DEBUG("Bad latch slot %u", action);
      arts_debug_generate_seg_fault();
    }

    if (res == 0) {
      assert(version != NULL);
      struct arts_dependent_list *dependent_list = &version->dependent;
      struct arts_dependent *dependent = version->dependent.dependents;
      int i;
      int j;
      unsigned int last_known = arts_atomic_fetch_add(&version->dependent_count, 0U);
      i = 0;
      int total_size = 0;
      while (i < last_known) {
        j = i - total_size;
        while (i < last_known && j < dependent_list->size) {
          while (!dependent[j].doneWriting) {
            ;
}
          if (dependent[j].type == ARTS_EDT) {
            if (event->data != NULL_GUID) {
              if (dependent[j].byte_offset != 0 || dependent[j].size != 0) {
                /// Byte-slice dependency: lookup DB and compute pointer
                struct arts_db *db =
                    (struct arts_db *)arts_route_table_lookup_item(event->data);
                if (db) {
                  void *db_data = (void *)(db + 1);
                  void *slice_ptr =
                      (void *)(((char *)db_data) + dependent[j].byte_offset);
                  arts_signal_edt_ptr_with_guid(dependent[j].addr, dependent[j].slot,
                                           event->data, slice_ptr,
                                           (unsigned int)dependent[j].size);
                } else {
                  ARTS_DEBUG(
                      "ESD: DB not found for byte-slice dep event->data=%lu",
                      event->data);
                }
              } else if (dependent[j].acquire_mode != ARTS_NULL) {
                arts_type_t mode = arts_guid_get_type(event->data);
                internal_signal_edt_with_mode(
                    dependent[j].addr, dependent[j].slot, event->data, mode,
                    dependent[j].acquire_mode);
              } else {
                arts_signal_edt(dependent[j].addr, dependent[j].slot,
                              event->data);
              }
            } else {
              ARTS_DEBUG("Event data is NULL_GUID for event %u", event_guid);
            }
          } else if (dependent[j].type == ARTS_EVENT) {
            SIGNAL_PERSISTENT_EVENT_COUNTER_STOP();
            arts_persistent_event_satisfy(dependent[j].addr, dependent[j].slot,
                                       true);
            SIGNAL_PERSISTENT_EVENT_COUNTER_START();
          } else if (dependent[j].type == ARTS_CALLBACK) {
            arts_edt_dep_t arg;
            arg.guid = event->data;
            arg.ptr = arts_route_table_lookup_item(event->data);
            arg.mode = ARTS_NULL;
            dependent[j].callback_t(arg);
          }
          j++;
          i++;
        }
        total_size += dependent_list->size;
        while (i < last_known && dependent_list->next == NULL) {
          ;
}
        dependent_list = dependent_list->next;
        dependent = dependent_list->dependents;
      }

      /// Free dependencies for this version
      arts_persistent_event_free_version(event);
    }
    if (lock) {
      arts_unlock(&event->lock);
}
  }
  ARTS_METRICS_TRIGGER_EVENT(ARTS_METRIC_PERSISTENT_EVENT_SIGNAL_THROUGHPUT, ARTS_METRIC_THREAD, 1);
  SIGNAL_PERSISTENT_EVENT_COUNTER_STOP();
}

void arts_persistent_event_increment_latch(arts_guid_t event_guid) {
  arts_persistent_event_satisfy(event_guid, ARTS_EVENT_LATCH_INCR_SLOT, true);
}

void arts_persistent_event_decrement_latch(arts_guid_t event_guid) {
  arts_persistent_event_satisfy(event_guid, ARTS_EVENT_LATCH_DECR_SLOT, true);
}

void arts_add_dependence_to_persistent_event(arts_guid_t event_source,
                                        arts_guid_t edt_dest, uint32_t edt_slot) {
  /// Check that the event_source is a persistent event
  if (arts_guid_get_type(event_source) != ARTS_PERSISTENT_EVENT) {
    ARTS_DEBUG("Event source %lu is not a persistent event", event_source);
    arts_debug_generate_seg_fault();
    return;
  }
  arts_type_t mode = arts_guid_get_type(edt_dest);
  struct arts_header *source_header =
      (struct arts_header *)arts_route_table_lookup_item(event_source);
  if (source_header == NULL) {
    unsigned int rank = arts_guid_get_rank(event_source);
    if (rank != arts_global_rank_id) {
      arts_remote_add_dependence_to_persistent_event(event_source, edt_dest, edt_slot,
                                               rank);
    } else {
      arts_out_of_order_add_dependence_to_persistent_event(event_source, edt_dest,
                                                   edt_slot, mode, event_source);
    }
    return;
  }

  ARTS_DEBUG("Add Dep from Persistent Event [Guid:%lu] to EDT[Guid:"
             "%lu, Slot:%u]",
             event_source, edt_dest, edt_slot);
  struct arts_persistent_event *event =
      (struct arts_persistent_event *)source_header;
  arts_lock(&event->lock);
  struct arts_persistent_event_version *version =
      arts_get_last_persistent_event_version(event);
  assert(version != NULL);
  bool needs_update = false;
  if (mode == ARTS_EDT) {
    struct arts_dependent_list *dependent_list = &version->dependent;
    unsigned int position = arts_atomic_fetch_add(&version->dependent_count, 1U);
    struct arts_dependent *dependent = arts_dependent_get(dependent_list, position);
    assert(dependent != NULL);
    dependent->type = ARTS_EDT;
    dependent->addr = edt_dest;
    dependent->slot = edt_slot;
    dependent->acquire_mode = ARTS_NULL;
    COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
    dependent->doneWriting = true;

    unsigned int res = arts_atomic_fetch_add(&version->latch_count, 0U);
    if (res == 0) {
      needs_update = true;
}
  } else if (mode == ARTS_EVENT) {
    struct arts_dependent_list *dependent_list = &version->dependent;
    unsigned int position = arts_atomic_fetch_add(&version->dependent_count, 1U);
    struct arts_dependent *dependent = arts_dependent_get(dependent_list, position);
    dependent->type = ARTS_EVENT;
    dependent->addr = edt_dest;
    dependent->slot = edt_slot;
    dependent->acquire_mode = ARTS_NULL;
    COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
    dependent->doneWriting = true;

    if (arts_atomic_fetch_add(&version->latch_count, 0U) == 0) {
      needs_update = true;
}
  }
  arts_unlock(&event->lock);
  if (needs_update) {
    arts_persistent_event_satisfy(event_source, ARTS_EVENT_UPDATE, true);
}
  }

void arts_add_dependence_to_persistent_event_with_mode(arts_guid_t event_source,
                                                arts_guid_t edt_dest,
                                                uint32_t edt_slot,
                                                arts_type_t acquire_mode) {
  arts_add_dependence_to_persistent_event_with_mode_and_diff(event_source, edt_dest,
                                                    edt_slot, acquire_mode);
}

void arts_add_dependence_to_persistent_event_with_mode_and_diff(arts_guid_t event_source,
                                                       arts_guid_t edt_dest,
                                                       uint32_t edt_slot,
                                                       arts_type_t acquire_mode) {
  /// Check that the event_source is a persistent event
  if (arts_guid_get_type(event_source) != ARTS_PERSISTENT_EVENT) {
    ARTS_DEBUG("Event source %lu is not a persistent event", event_source);
    arts_debug_generate_seg_fault();
    return;
  }
  arts_type_t mode = arts_guid_get_type(edt_dest);
  struct arts_header *source_header =
      (struct arts_header *)arts_route_table_lookup_item(event_source);
  if (source_header == NULL) {
    unsigned int rank = arts_guid_get_rank(event_source);
    if (rank != arts_global_rank_id) {
      // TODO: Extend remote protocol to pass acquire_mode
      arts_remote_add_dependence_to_persistent_event_with_hints(
          event_source, edt_dest, edt_slot, rank, acquire_mode);
    } else {
      // TODO: Extend out-of-order handling to pass acquire_mode
      // For now, fallback to standard out-of-order add dependence
      arts_out_of_order_add_dependence_to_persistent_event(event_source, edt_dest,
                                                   edt_slot, mode, event_source);
    }
    return;
  }

  ARTS_DEBUG("Add Dep from Persistent Event [Guid:%lu] to EDT[Guid:"
             "%lu, Slot:%u, AcquireMode:%s]",
             event_source, edt_dest, edt_slot, GET_TYPE_NAME(acquire_mode));
  struct arts_persistent_event *event =
      (struct arts_persistent_event *)source_header;
  arts_lock(&event->lock);
  struct arts_persistent_event_version *version =
      arts_get_last_persistent_event_version(event);
  assert(version != NULL);
  bool needs_update = false;
  if (mode == ARTS_EDT) {
    struct arts_dependent_list *dependent_list = &version->dependent;
    unsigned int position = arts_atomic_fetch_add(&version->dependent_count, 1U);
    struct arts_dependent *dependent = arts_dependent_get(dependent_list, position);
    assert(dependent != NULL);
    dependent->type = ARTS_EDT;
    dependent->addr = edt_dest;
    dependent->slot = edt_slot;
    dependent->acquire_mode = acquire_mode;
    dependent->byte_offset = 0;
    dependent->size = 0;
    COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
    dependent->doneWriting = true;

    unsigned int res = arts_atomic_fetch_add(&version->latch_count, 0U);
    if (res == 0) {
      needs_update = true;
}
  } else if (mode == ARTS_EVENT) {
    struct arts_dependent_list *dependent_list = &version->dependent;
    unsigned int position = arts_atomic_fetch_add(&version->dependent_count, 1U);
    struct arts_dependent *dependent = arts_dependent_get(dependent_list, position);
    dependent->type = ARTS_EVENT;
    dependent->addr = edt_dest;
    dependent->slot = edt_slot;
    dependent->acquire_mode = acquire_mode;
    dependent->byte_offset = 0;
    dependent->size = 0;
    COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
    dependent->doneWriting = true;

    if (arts_atomic_fetch_add(&version->latch_count, 0U) == 0) {
      needs_update = true;
}
  }
  arts_unlock(&event->lock);
  if (needs_update) {
    arts_persistent_event_satisfy(event_source, ARTS_EVENT_UPDATE, true);
}
  }

void arts_add_dependence_to_persistent_event_with_byte_offset(
    arts_guid_t event_source, arts_guid_t edt_dest, uint32_t edt_slot,
    arts_type_t acquire_mode, uint64_t byte_offset, uint64_t size) {
  /// Check that the event_source is a persistent event
  if (arts_guid_get_type(event_source) != ARTS_PERSISTENT_EVENT) {
    ARTS_DEBUG("Event source %lu is not a persistent event", event_source);
    arts_debug_generate_seg_fault();
    return;
  }
  arts_type_t mode = arts_guid_get_type(edt_dest);
  struct arts_header *source_header =
      (struct arts_header *)arts_route_table_lookup_item(event_source);
  if (source_header == NULL) {
    unsigned int rank = arts_guid_get_rank(event_source);
    if (rank != arts_global_rank_id) {
      // ESD: Now passes byte_offset/size to remote persistent event
      arts_remote_add_dependence_to_persistent_event_with_byte_offset(
          event_source, edt_dest, edt_slot, rank, acquire_mode, byte_offset, size);
    } else {
      // Local out-of-order: byte offset is not critical for OO handling
      arts_out_of_order_add_dependence_to_persistent_event(event_source, edt_dest,
                                                   edt_slot, mode, event_source);
    }
    return;
  }

  ARTS_DEBUG("Add Dep from Persistent Event [Guid:%lu] to EDT[Guid:"
             "%lu, Slot:%u, AcquireMode:%s, ByteOffset:%lu, Size:%lu]",
             event_source, edt_dest, edt_slot, GET_TYPE_NAME(acquire_mode),
             byte_offset, size);
  struct arts_persistent_event *event =
      (struct arts_persistent_event *)source_header;
  arts_lock(&event->lock);
  struct arts_persistent_event_version *version =
      arts_get_last_persistent_event_version(event);
  assert(version != NULL);
  bool needs_update = false;
  if (mode == ARTS_EDT) {
    struct arts_dependent_list *dependent_list = &version->dependent;
    unsigned int position = arts_atomic_fetch_add(&version->dependent_count, 1U);
    struct arts_dependent *dependent = arts_dependent_get(dependent_list, position);
    assert(dependent != NULL);
    dependent->type = ARTS_EDT;
    dependent->addr = edt_dest;
    dependent->slot = edt_slot;
    dependent->acquire_mode = acquire_mode;
    dependent->byte_offset = byte_offset;
    dependent->size = size;
    COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
    dependent->doneWriting = true;

    unsigned int res = arts_atomic_fetch_add(&version->latch_count, 0U);
    if (res == 0) {
      needs_update = true;
}
  } else if (mode == ARTS_EVENT) {
    struct arts_dependent_list *dependent_list = &version->dependent;
    unsigned int position = arts_atomic_fetch_add(&version->dependent_count, 1U);
    struct arts_dependent *dependent = arts_dependent_get(dependent_list, position);
    dependent->type = ARTS_EVENT;
    dependent->addr = edt_dest;
    dependent->slot = edt_slot;
    dependent->acquire_mode = acquire_mode;
    dependent->byte_offset = byte_offset;
    dependent->size = size;
    COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
    dependent->doneWriting = true;

    if (arts_atomic_fetch_add(&version->latch_count, 0U) == 0) {
      needs_update = true;
}
  }
  arts_unlock(&event->lock);
  if (needs_update) {
    arts_persistent_event_satisfy(event_source, ARTS_EVENT_UPDATE, true);
}
  }
