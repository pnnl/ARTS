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
#include "arts/sync/event.h"

#include "arts.h"
#include "arts/compute/edt.h"
#include "arts/gas/guid.h"
#include "arts/gas/out_of_order.h"
#include "arts/gas/route_table.h"
#include "arts/memory/db.h"
#include "arts/remote/handler.h"
#include "arts/system/debug.h"
#include "arts/system/print.h"
#include "arts/system/threads.h"
#include "arts/utils/atomics.h"
#include "arts/utils/link_list.h"
#include "arts/utils/malloc.h"

#include <assert.h>
#include <time.h>

extern ARTS_THREAD_LOCAL struct arts_edt_s *current_edt;

bool arts_event_create_internal(arts_guid_t *guid, unsigned int route,
                                unsigned int dependent_count,
                                unsigned int latch_count,
                                arts_event_types_t event_type,
                                arts_guid_t event_data) {
  unsigned int event_size = sizeof(struct arts_event_s) +
                            (sizeof(struct arts_dependent_s) * dependent_count);
  void *event_packet = arts_calloc(1, event_size);

  if (event_size) {
    struct arts_event_s *event = (struct arts_event_s *)event_packet;
    event->header.type = ARTS_EVENT;
    event->header.size = event_size;
    event->dependent_count = 0;
    event->dependent.size = dependent_count;
    event->latch_count = latch_count;
    event->type = event_type;
    event->lock = 0;
    event->versions = NULL;
    event->data = event_data;

    if (route == arts_global_rank_id) {
      if (*guid) {
        /* For labeled GUIDs, use race-safe addition since multiple
         * threads/ranks may try to create the same labeled event concurrently.
         * This matches the behavior in arts_remote_handle_event_move which also
         * uses arts_route_table_add_item_race for consistency. */
        if (arts_route_table_add_item_race(event_packet, *guid,
                                           arts_global_rank_id, false)) {
          arts_route_table_fire_oo(*guid, arts_out_of_order_handler);
        } else {
          /* Event already exists - free the allocated memory */
          arts_free(event_packet);
          return false;
        }
      } else {
        *guid = arts_guid_create_for_rank(route, ARTS_EVENT);
        arts_route_table_add_item(event_packet, *guid, arts_global_rank_id,
                                  false);
      }
    } else {
      arts_remote_memory_move(route, *guid, event_packet, event_size,
                              ARTS_REMOTE_EVENT_MOVE_MSG, arts_free);
    }

    return true;
  }
  return false;
}

arts_guid_t arts_event_create(unsigned int route, arts_event_types_t type,
                              unsigned int latch_count, arts_guid_t data_guid) {
  TIME_EVENT_CREATE_START();
  INCREMENT_NUM_EVENT_CREATE_BY(1);
  if (route == ARTS_HINT_CURRENT_NODE) {
    route = arts_global_rank_id;
  }
  arts_guid_t guid = NULL_GUID;
  switch (type) {
  case ARTS_EVENT_ONCE:
  case ARTS_EVENT_STICKY:
  case ARTS_EVENT_IDEM:
    arts_event_create_internal(&guid, route, INITIAL_DEPENDENT_SIZE, 1, type,
                               NULL_GUID);
    break;
  case ARTS_EVENT_CHANNEL:
    arts_event_create_channel_internal(&guid, route, latch_count, data_guid);
    break;
  default: /* LATCH, COUNTED */
    arts_event_create_internal(&guid, route, INITIAL_DEPENDENT_SIZE,
                               latch_count, type, NULL_GUID);
    break;
  }
  TIME_EVENT_CREATE_STOP();
  return guid;
}

arts_guid_t arts_event_create_with_guid(arts_guid_t guid,
                                        arts_event_types_t type,
                                        unsigned int latch_count,
                                        arts_guid_t data_guid) {
  TIME_EVENT_CREATE_START();
  INCREMENT_NUM_EVENT_CREATE_BY(1);
  unsigned int route = arts_guid_get_rank(guid);
  bool ret = false;
  switch (type) {
  case ARTS_EVENT_ONCE:
  case ARTS_EVENT_STICKY:
  case ARTS_EVENT_IDEM:
    ret = arts_event_create_internal(&guid, route, INITIAL_DEPENDENT_SIZE, 1,
                                     type, NULL_GUID);
    break;
  case ARTS_EVENT_CHANNEL:
    ret = arts_event_create_channel_internal(&guid, route, latch_count,
                                             data_guid);
    break;
  default: /* LATCH, COUNTED */
    ret = arts_event_create_internal(&guid, route, INITIAL_DEPENDENT_SIZE,
                                     latch_count, type, NULL_GUID);
    break;
  }
  TIME_EVENT_CREATE_STOP();
  return (ret) ? guid : NULL_GUID;
}

/* ── Forward declarations ──────────────────────────────────────────── */

static struct arts_event_version_s *
channel_push_version(struct arts_event_s *event);

struct arts_dependent_s *arts_dependent_get(struct arts_dependent_list_s *head,
                                            int position);

/* ── CHANNEL version helpers ─────────────────────────────────────────── */

static struct arts_link_list_s *
channel_get_versions(struct arts_event_s *event) {
  if (event->versions != NULL) {
    return event->versions;
  }
  event->versions = arts_link_list_group_new(1);
  struct arts_event_version_s *version = channel_push_version(event);
  version->dependent.next = NULL;
  assert(version != NULL);
  return event->versions;
}

static struct arts_event_version_s *
channel_push_version(struct arts_event_s *event) {
  struct arts_link_list_s *versions = channel_get_versions(event);
  struct arts_event_version_s *next =
      (struct arts_event_version_s *)arts_link_list_new_item(
          (sizeof(struct arts_event_version_s) +
           (sizeof(struct arts_dependent_s) * INITIAL_DEPENDENT_SIZE)));
  next->latch_count = 0;
  next->dependent_count = 0;
  next->data =
      event->data; /* Inherit event-level data (DB-coupled channels). */
  next->dependent.size = INITIAL_DEPENDENT_SIZE;
  struct arts_event_version_s *last = NULL;
  if (versions && versions->tailPtr) {
    last =
        (struct arts_event_version_s *)arts_link_list_get_tail_data(versions);
  }
  if (last) {
    next->version = last->version + 1;
  } else {
    next->version = 0;
  }
  arts_link_list_push_back(versions, next);
  return next;
}

static struct arts_event_version_s *
channel_get_front_version(struct arts_event_s *event) {
  return (struct arts_event_version_s *)arts_link_list_get_front_data(
      channel_get_versions(event));
}

static struct arts_event_version_s *
channel_get_last_version(struct arts_event_s *event) {
  return (struct arts_event_version_s *)arts_link_list_get_tail_data(
      channel_get_versions(event));
}

static bool channel_free_version(struct arts_event_s *event) {
  struct arts_link_list_s *versions = event->versions;
  if (versions == NULL) {
    return true;
  }
  bool last = true;
  arts_lock(&versions->lock);

  if (versions->headPtr != versions->tailPtr) {
    last = false;
  }

  struct arts_event_version_s *version =
      (struct arts_event_version_s *)(versions->headPtr + 1);
  assert(version != NULL);

  struct arts_dependent_list_s *trail;
  struct arts_dependent_list_s *current = version->dependent.next;
  while (current) {
    trail = current;
    current = current->next;
    arts_free(trail);
  }
  version->dependent.next = NULL;

  if (last) {
    version->latch_count = 0;
    version->dependent_count = 0;
    version->data = event->data; /* Re-inherit for next generation. */
  } else {
    versions->headPtr = versions->headPtr->next;
    struct arts_link_list_item_s *item =
        ((struct arts_link_list_item_s *)version) - 1;
    arts_free(item);
  }

  arts_unlock(&versions->lock);
  return last;
}

static void channel_free_all_versions(struct arts_event_s *event) {
  if (event->versions) {
    struct arts_link_list_item_s *item = event->versions->headPtr;
    while (item) {
      struct arts_link_list_item_s *next = item->next;
      struct arts_event_version_s *version =
          (struct arts_event_version_s *)(item + 1);
      struct arts_dependent_list_s *trail;
      struct arts_dependent_list_s *current = version->dependent.next;
      while (current) {
        trail = current;
        current = current->next;
        arts_free(trail);
      }
      arts_free(item);
      item = next;
    }
    arts_free(event->versions);
    event->versions = NULL;
  }
}

/* ── CHANNEL fire loop ──────────────────────────────────────────────── */

static void channel_fire_dependents(struct arts_event_s *event,
                                    struct arts_event_version_s *version,
                                    arts_guid_t event_guid) {
  (void)event_guid;
  arts_guid_t data = version->data;
  struct arts_dependent_list_s *dependent_list = &version->dependent;
  struct arts_dependent_s *dependent = version->dependent.dependents;
  unsigned int last_known =
      arts_atomic_fetch_add(&version->dependent_count, 0U);
  int i = 0;
  int total_size = 0;
  while (i < (int)last_known) {
    int j = i - total_size;
    while (i < (int)last_known && j < (int)dependent_list->size) {
      while (!dependent[j].done_writing) {
        ;
      }
      if (dependent[j].type == ARTS_EDT) {
        if (data != NULL_GUID) {
          if (dependent[j].byte_offset != 0 || dependent[j].size != 0) {
            struct arts_db_s *db =
                (struct arts_db_s *)arts_route_table_lookup_db(data, NULL,
                                                               false);
            if (db) {
              void *db_data = (void *)(db + 1);
              void *slice_ptr =
                  (void *)(((char *)db_data) + dependent[j].byte_offset);
              arts_signal_edt_ptr_with_guid(dependent[j].addr,
                                            dependent[j].slot, data, slice_ptr,
                                            (unsigned int)dependent[j].size);
              arts_route_table_return_db(data, false);
            }
          } else {
            arts_signal_edt(dependent[j].addr, dependent[j].slot, data,
                            DB_MODE_NULL);
          }
        }
      } else if (dependent[j].type == ARTS_EVENT) {
        arts_event_satisfy_slot(dependent[j].addr, data, dependent[j].slot);
      } else if (dependent[j].type == ARTS_CALLBACK) {
        arts_edt_dep_t arg;
        arg.guid = data;
        arg.ptr = arts_route_table_lookup_db(data, NULL, false);
        dependent[j].callback_t(arg);
        if (arg.ptr) {
          arts_route_table_return_db(data, false);
        }
      }
      j++;
      i++;
    }
    total_size += (int)dependent_list->size;
    if (i >= (int)last_known) {
      break;
    }
    while (dependent_list->next == NULL) {
      ;
    }
    dependent_list = dependent_list->next;
    dependent = dependent_list->dependents;
  }

  channel_free_version(event);
}

/* ── CHANNEL satisfy ────────────────────────────────────────────────── */

static void channel_satisfy_slot(struct arts_event_s *event,
                                 arts_guid_t event_guid, uint32_t slot,
                                 arts_guid_t data_guid) {
  arts_lock(&event->lock);

  unsigned int res = (unsigned int)-1;
  struct arts_event_version_s *version = channel_get_front_version(event);
  assert(version != NULL);

  if (slot == ARTS_EVENT_LATCH_INCR_SLOT) {
    res = arts_atomic_fetch_add(&version->latch_count, 0U);
    if (res == 1) {
      version = channel_push_version(event);
    }
    res = arts_atomic_add(&version->latch_count, 1U);
  } else if (slot == ARTS_EVENT_LATCH_DECR_SLOT) {
    res = arts_atomic_fetch_add(&version->latch_count, 0U);
    if (res == (unsigned int)-1) {
      version = channel_push_version(event);
    }
    /* Store per-generation data on the TARGET version (after push check). */
    if (data_guid != NULL_GUID) {
      version->data = data_guid;
    }
    res = arts_atomic_sub(&version->latch_count, 1U);
  } else if (slot == ARTS_EVENT_UPDATE) {
    res = arts_atomic_fetch_add(&version->latch_count, 0U);
  } else {
    ARTS_ERROR("Channel event invalid slot %u (guid=%lu)", slot, event_guid);
  }

  if (res == 0) {
    channel_fire_dependents(event, version, event_guid);
  }

  arts_unlock(&event->lock);
}

/* ── CHANNEL creation ───────────────────────────────────────────────── */

bool arts_event_create_channel_internal(arts_guid_t *guid, unsigned int route,
                                        unsigned int latch_count,
                                        arts_guid_t data_guid) {
  bool ret =
      arts_event_create_internal(guid, route, INITIAL_DEPENDENT_SIZE,
                                 latch_count, ARTS_EVENT_CHANNEL, data_guid);
  return ret;
}

/* ── CHANNEL add dependence with mode ───────────────────────────────── */

void arts_event_add_dependence_with_mode(arts_guid_t event_source,
                                         arts_guid_t edt_dest,
                                         uint32_t edt_slot,
                                         arts_db_access_mode_t mode) {
  arts_type_t dest_type = arts_guid_get_type(edt_dest);
  struct arts_event_s *event =
      (struct arts_event_s *)arts_route_table_lookup_item(event_source);
  if (event == NULL) {
    unsigned int rank = arts_guid_get_rank(event_source);
    if (rank != arts_global_rank_id) {
      arts_remote_channel_add_dependence_with_mode(event_source, edt_dest,
                                                   edt_slot, rank, mode);
    } else {
      arts_out_of_order_add_dependence(event_source, edt_dest, edt_slot,
                                       DB_MODE_NULL, event_source);
    }
    return;
  }

  arts_lock(&event->lock);
  struct arts_event_version_s *version = channel_get_front_version(event);
  assert(version != NULL);

  if (event->latch_count > 0) {
    /* Satisfy-channel: single-consumer per generation (OCR channel model).
     * Each addDep represents a new generation that fires on one satisfy.
     * If the front version already has a dependent, push a new version
     * so the next satisfy fires exactly one consumer (FIFO order). */
    if (version->dependent_count > 0) {
      version = channel_push_version(event);
    }
    arts_atomic_add(&version->latch_count, 1U);
  }

  struct arts_dependent_list_s *dependent_list = &version->dependent;
  unsigned int position = arts_atomic_fetch_add(&version->dependent_count, 1U);
  struct arts_dependent_s *dependent =
      arts_dependent_get(dependent_list, (int)position);
  assert(dependent != NULL);
  dependent->type = (dest_type == ARTS_EVENT) ? ARTS_EVENT : ARTS_EDT;
  dependent->addr = edt_dest;
  dependent->slot = edt_slot;
  dependent->mode = mode;
  dependent->byte_offset = 0;
  dependent->size = 0;
  COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
  dependent->done_writing = true;

  /* Fire under the lock to prevent DECR from interleaving between the
     latch check and the fire.  The deferred UPDATE approach had a race:
     a DECR could slip in after unlock and before UPDATE re-locked. */
  if (arts_atomic_fetch_add(&version->latch_count, 0U) == 0) {
    channel_fire_dependents(event, version, event_source);
  }

  arts_unlock(&event->lock);
}

void arts_event_add_dependence_with_byte_offset(
    arts_guid_t event_source, arts_guid_t edt_dest, uint32_t edt_slot,
    arts_db_access_mode_t mode, uint64_t byte_offset, uint64_t len) {
  arts_type_t dest_type = arts_guid_get_type(edt_dest);
  struct arts_event_s *event =
      (struct arts_event_s *)arts_route_table_lookup_item(event_source);
  if (event == NULL) {
    unsigned int rank = arts_guid_get_rank(event_source);
    if (rank != arts_global_rank_id) {
      arts_remote_channel_add_dependence_with_byte_offset(
          event_source, edt_dest, edt_slot, rank, mode, byte_offset, len);
    } else {
      arts_out_of_order_add_dependence(event_source, edt_dest, edt_slot,
                                       DB_MODE_NULL, event_source);
    }
    return;
  }

  arts_lock(&event->lock);
  struct arts_event_version_s *version = channel_get_front_version(event);
  assert(version != NULL);

  if (event->latch_count > 0) {
    /* Single-consumer per generation — same as
     * arts_event_add_dependence_with_mode. */
    if (version->dependent_count > 0) {
      version = channel_push_version(event);
    }
    arts_atomic_add(&version->latch_count, 1U);
  }

  struct arts_dependent_list_s *dependent_list = &version->dependent;
  unsigned int position = arts_atomic_fetch_add(&version->dependent_count, 1U);
  struct arts_dependent_s *dependent =
      arts_dependent_get(dependent_list, (int)position);
  assert(dependent != NULL);
  dependent->type = (dest_type == ARTS_EVENT) ? ARTS_EVENT : ARTS_EDT;
  dependent->addr = edt_dest;
  dependent->slot = edt_slot;
  dependent->mode = mode;
  dependent->byte_offset = byte_offset;
  dependent->size = len;
  COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
  dependent->done_writing = true;

  /* Fire under the lock — same race fix as arts_event_add_dependence_with_mode.
   */
  if (arts_atomic_fetch_add(&version->latch_count, 0U) == 0) {
    channel_fire_dependents(event, version, event_source);
  }

  arts_unlock(&event->lock);
}

/* ── Event free / destroy ───────────────────────────────────────────── */

void arts_event_free(struct arts_event_s *event) {
  if (event->type == ARTS_EVENT_CHANNEL) {
    channel_free_all_versions(event);
  }
  struct arts_dependent_list_s *trail;
  struct arts_dependent_list_s *current = event->dependent.next;
  while (current) {
    trail = current;
    current = current->next;
    arts_free(trail);
  }
  arts_free(event);
}

void arts_event_destroy(arts_guid_t guid) {
  struct arts_event_s *event =
      (struct arts_event_s *)arts_route_table_lookup_item(guid);
  if (event != NULL) {
    arts_route_table_remove_item(guid);
    arts_event_free(event);
  }
}

void arts_event_satisfy_slot(arts_guid_t event_guid, arts_guid_t data_guid,
                             uint32_t slot) {
  TIME_EVENT_SIGNAL_START();
  INCREMENT_NUM_EVENT_SIGNAL_BY(1);
  if (current_edt && current_edt->invalidate_count > 0) {
    arts_out_of_order_event_satisfy_slot(current_edt->current_edt, event_guid,
                                         data_guid, slot, true);
    return;
  }
  ARTS_INFO("Signal Event:%u, Data:%u at %u", event_guid, data_guid, slot);
  struct arts_event_s *event =
      (struct arts_event_s *)arts_route_table_lookup_item(event_guid);
  if (!event) {
    unsigned int rank = arts_guid_get_rank(event_guid);
    if (rank != arts_global_rank_id) {
      arts_remote_event_satisfy_slot(event_guid, data_guid, slot);
    } else {
      arts_out_of_order_event_satisfy_slot(event_guid, event_guid, data_guid,
                                           slot, false);
    }
  } else {
    // CHANNEL events use their own lock-protected version-based path
    if (event->type == ARTS_EVENT_CHANNEL) {
      channel_satisfy_slot(event, event_guid, slot, data_guid);
      goto done;
    }

    // Re-satisfy guard: type-aware handling for already-fired events
    if (event->fired) {
      if (event->type == ARTS_EVENT_IDEM) {
        goto done;
      } else if (event->type == ARTS_EVENT_STICKY) {
        ARTS_WARN("Sticky event %lu: re-satisfy rejected", event_guid);
        goto done;
      } else {
        ARTS_ERROR("Event latch already fired (guid=%lu, data=%lu, slot=%u)",
                   event_guid, data_guid, slot);
      }
    }

    unsigned int res = 0U;
    if (slot == ARTS_EVENT_LATCH_INCR_SLOT) {
      if (event->type == ARTS_EVENT_ONCE || event->type == ARTS_EVENT_COUNTED) {
        ARTS_ERROR("INCR_SLOT rejected for %s event (guid=%lu)",
                   event->type == ARTS_EVENT_ONCE ? "ONCE" : "COUNTED",
                   event_guid);
      }
      res = arts_atomic_add(&event->latch_count, 1U);
    } else if (slot == ARTS_EVENT_LATCH_DECR_SLOT) {
      if (data_guid != NULL_GUID) {
        event->data = data_guid;
      }
      res = arts_atomic_sub(&event->latch_count, 1U);
    } else {
      ARTS_ERROR("Event latch invalid slot %u", slot);
    }

    /// When the latch count reaches 0, fire the event
    if (!res) {
      /// If the event is already fired, we should not fire it again
      if (arts_atomic_swap_bool(&event->fired, true)) {
        ARTS_ERROR("Event latch already fired (guid=%lu, data=%lu, slot=%u)",
                   event_guid, data_guid, slot);
      }
      /// If the event is not fired, we need to fire it
      else {
        struct arts_dependent_list_s *dependent_list = &event->dependent;
        struct arts_dependent_s *dependent = event->dependent.dependents;
        int i;
        int j;
        /// Capture current state
        unsigned int last_known =
            arts_atomic_fetch_add(&event->dependent_count, 0U);
        event->pos = last_known + 1;
        i = 0;
        int total_size = 0;
        /// Process all dependents up to last_known
        while (i < last_known) {
          j = i - total_size;
          while (i < last_known && j < dependent_list->size) {
            while (!dependent[j].done_writing) {
              ;
            }
            if (dependent[j].type == ARTS_EDT) {
              arts_signal_edt(dependent[j].addr, dependent[j].slot, event->data,
                              DB_MODE_NULL);
            } else if (dependent[j].type == ARTS_EVENT) {
              TIME_EVENT_SIGNAL_STOP();
              arts_event_satisfy_slot(dependent[j].addr, event->data,
                                      dependent[j].slot);
              TIME_EVENT_SIGNAL_START();
            } else if (dependent[j].type == ARTS_CALLBACK) {
              arts_edt_dep_t arg;
              arg.guid = event->data;
              arg.ptr = arts_route_table_lookup_db(event->data, NULL, false);
              dependent[j].callback_t(arg);
              if (arg.ptr) {
                arts_route_table_return_db(event->data, false);
              }
            }
            j++;
            i++;
          }
          total_size += (int)dependent_list->size;
          if (i >= last_known) {
            break;
          }
          while (dependent_list->next == NULL) {
            ;
          }
          dependent_list = dependent_list->next;
          dependent = dependent_list->dependents;
        }
        // Auto-destroy for LATCH/ONCE/COUNTED; STICKY/IDEM persist
        if (event->type == ARTS_EVENT_LATCH || event->type == ARTS_EVENT_ONCE ||
            event->type == ARTS_EVENT_COUNTED) {
          arts_route_table_remove_item(event_guid);
          arts_event_free(event);
        }
      }
    }
  }
done:
  TIME_EVENT_SIGNAL_STOP();
}

struct arts_dependent_s *arts_dependent_get(struct arts_dependent_list_s *head,
                                            int position) {
  struct arts_dependent_list_s *list = head;
  volatile struct arts_dependent_list_s *temp;

  while (1) {
    /// If the position is greater than the size of the list, we need to
    /// allocate a new list
    if (position >= list->size) {
      if (position - list->size == 0) {
        if (list->next == NULL) {
          temp = (volatile struct arts_dependent_list_s *)arts_calloc(
              1, sizeof(struct arts_dependent_list_s) +
                     (sizeof(struct arts_dependent_s) * list->size * 2));
          if (temp == NULL) {
            ARTS_ERROR("Event dependent list allocation failed");
          }
          temp->size = list->size * 2;
          list->next = (struct arts_dependent_list_s *)temp;
        }
      }

      // EXPONENTIONAL BACK OFF THIS
      while (list->next == NULL) {
      }

      position -= (int)list->size;
      list = list->next;
    } else {
      break;
    }
  }
  return list->dependents + position;
}

/*
 * arts_add_dependence — Unified dependence registration (two-message pattern).
 *
 * Accepts Event OR DB as source, EDT or Event as destination.
 *
 * Two-message pattern for EDT destinations:
 *   Step 1 (mode-set):  Write access_mode to depv[slot].mode on the EDT.
 *   Step 2 (waiter-reg): Register as a dependent on the source event.
 *                         The fire loop delivers data with DB_MODE_NULL,
 *                         so internal_signal_edt preserves the mode
 *                         already written in step 1.
 *
 * For DB sources: look up db->event_guid, register on that channel event,
 * and increment the latch for DB_MODE_EW.
 *
 * For NULL source (NULL_GUID): signal the EDT slot immediately with NULL
 * data and the given access_mode.
 */
void arts_add_dependence(arts_guid_t source, arts_guid_t destination,
                         uint32_t slot, arts_db_access_mode_t access_mode) {
  ARTS_INFO("Add Dependence from %lu to %lu at %u mode=%u", source, destination,
            slot, access_mode);

  /* NULL source → signal immediately (slot satisfied with no data). */
  if (source == NULL_GUID) {
    arts_type_t dest_type = arts_guid_get_type(destination);
    if (dest_type == ARTS_EDT) {
      arts_signal_edt(destination, slot, NULL_GUID, access_mode);
    } else if (dest_type == ARTS_EVENT) {
      arts_event_satisfy_slot(destination, NULL_GUID, slot);
    }
    return;
  }

  arts_type_t source_type = arts_guid_get_type(source);

  /* DB source → translate to channel event + EW latch increment. */
  if (source_type == ARTS_DB) {
    struct arts_db_s *db_res =
        (struct arts_db_s *)arts_route_table_lookup_db(source, NULL, false);
    if (db_res != NULL) {
      /* Step 1: set mode on EDT (if dest is EDT). */
      arts_type_t dest_type = arts_guid_get_type(destination);
      if (dest_type == ARTS_EDT) {
        arts_set_dep_mode(destination, slot, access_mode);
      }
      /* Step 2: register waiter on channel event (mode-less). */
      arts_event_add_dependence_with_mode(db_res->event_guid, destination, slot,
                                          DB_MODE_NULL);
      /* Step 3: EW latch increment. */
      if (access_mode == DB_MODE_EW) {
        arts_db_increment_latch(source);
      }
      arts_route_table_return_db(source, false);
    } else {
      /* DB not local — forward to DB's owner node. */
      arts_remote_db_add_dependence_with_hints(source, destination, slot,
                                               access_mode);
    }
    return;
  }

  /* Event source (ARTS_EVENT). */
  arts_type_t dest_type = arts_guid_get_type(destination);

  /* Step 1: set mode on EDT dep slot. */
  if (dest_type == ARTS_EDT) {
    arts_set_dep_mode(destination, slot, access_mode);
  }

  /* Step 2: register waiter on event. */
  struct arts_header_s *source_header =
      (struct arts_header_s *)arts_route_table_lookup_item(source);
  if (source_header == NULL) {
    unsigned int rank = arts_guid_get_rank(source);
    if (rank != arts_global_rank_id) {
      arts_remote_add_dependence(source, destination, slot, rank, access_mode);
    } else {
      arts_out_of_order_add_dependence(source, destination, slot, access_mode,
                                       source);
    }
    return;
  }

  struct arts_event_s *event = (struct arts_event_s *)source_header;

  // CHANNEL events use lock-protected version-based dependence
  if (event->type == ARTS_EVENT_CHANNEL) {
    arts_event_add_dependence_with_mode(source, destination, slot,
                                        DB_MODE_NULL);
    return;
  }

  if (dest_type == ARTS_EDT) {
    struct arts_dependent_list_s *dependent_list = &event->dependent;
    struct arts_dependent_s *dependent;
    unsigned int position = arts_atomic_fetch_add(&event->dependent_count, 1U);
    dependent = arts_dependent_get(dependent_list, (int)position);
    dependent->type = ARTS_EDT;
    dependent->addr = destination;
    dependent->slot = slot;
    dependent->mode = DB_MODE_NULL;
    COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
    dependent->done_writing = true;

    if (event->fired) {
      while (event->pos == 0) {
        ;
      }
      if (position >= event->pos - 1) {
        /* Self-signal: data only, mode already set on EDT. */
        arts_signal_edt(destination, slot, event->data, DB_MODE_NULL);
      }
    }
  } else if (dest_type == ARTS_EVENT) {
    struct arts_dependent_list_s *dependent_list = &event->dependent;
    struct arts_dependent_s *dependent;
    unsigned int position = arts_atomic_fetch_add(&event->dependent_count, 1U);
    dependent = arts_dependent_get(dependent_list, (int)position);
    dependent->type = ARTS_EVENT;
    dependent->addr = destination;
    dependent->slot = slot;
    dependent->mode = DB_MODE_NULL;
    COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
    dependent->done_writing = true;

    if (event->fired) {
      while (event->pos == 0) {
        ;
      }
      if (event->pos - 1 <= position) {
        arts_event_satisfy_slot(destination, event->data, slot);
      }
    }
  }
}

void arts_add_dependence_at(arts_guid_t source, arts_guid_t destination,
                            uint32_t slot, arts_db_access_mode_t access_mode,
                            uint64_t byte_offset, uint64_t len) {
  /* Delegates to the standard path when no byte-offset is needed. */
  if (byte_offset == 0 && len == 0) {
    arts_add_dependence(source, destination, slot, access_mode);
    return;
  }

  /* Step 1: set mode on EDT dep slot. */
  arts_type_t dest_type = arts_guid_get_type(destination);
  if (dest_type == ARTS_EDT) {
    arts_set_dep_mode(destination, slot, access_mode);
  }

  arts_type_t source_type = arts_guid_get_type(source);

  if (source_type == ARTS_DB) {
    /* DB source — look up channel event and register byte-offset waiter. */
    struct arts_db_s *db_res =
        (struct arts_db_s *)arts_route_table_lookup_db(source, NULL, false);
    if (db_res != NULL) {
      arts_event_add_dependence_with_byte_offset(
          db_res->event_guid, destination, slot, DB_MODE_NULL, byte_offset,
          len);
      arts_route_table_return_db(source, false);
    } else {
      arts_remote_db_add_dependence_with_byte_offset(
          source, destination, slot, access_mode, byte_offset, len);
    }
    if (access_mode == DB_MODE_EW) {
      arts_db_increment_latch(source);
    }
  } else {
    /* Event source — register byte-offset waiter directly. */
    arts_event_add_dependence_with_byte_offset(source, destination, slot,
                                               DB_MODE_NULL, byte_offset, len);
  }
}

void arts_add_local_event_callback(arts_guid_t source,
                                   event_callback_t callback_t) {
  struct arts_event_s *event =
      (struct arts_event_s *)arts_route_table_lookup_item(source);
  if (event && arts_guid_get_type(source) == ARTS_EVENT) {
    // CHANNEL events: register callback on latest version (lock-protected)
    if (event->type == ARTS_EVENT_CHANNEL) {
      arts_lock(&event->lock);
      struct arts_event_version_s *version = channel_get_last_version(event);
      assert(version != NULL);
      struct arts_dependent_list_s *dep_list = &version->dependent;
      unsigned int pos = arts_atomic_fetch_add(&version->dependent_count, 1U);
      struct arts_dependent_s *dep = arts_dependent_get(dep_list, (int)pos);
      assert(dep != NULL);
      dep->type = ARTS_CALLBACK;
      dep->callback_t = callback_t;
      dep->addr = NULL_GUID;
      dep->slot = 0;
      dep->mode = DB_MODE_NULL;
      dep->byte_offset = 0;
      dep->size = 0;
      COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
      dep->done_writing = true;
      /* Fire under the lock — same race fix as add_dependence_with_mode. */
      if (arts_atomic_fetch_add(&version->latch_count, 0U) == 0) {
        channel_fire_dependents(event, version, source);
      }
      arts_unlock(&event->lock);
      return;
    }

    struct arts_dependent_list_s *dependent_list = &event->dependent;
    struct arts_dependent_s *dependent;
    unsigned int position = arts_atomic_fetch_add(&event->dependent_count, 1U);
    dependent = arts_dependent_get(dependent_list, (int)position);
    dependent->type = ARTS_CALLBACK;
    dependent->callback_t = callback_t;
    dependent->addr = NULL_GUID;
    dependent->slot = 0;
    COMPILER_DO_NOT_REORDER_WRITES_BETWEEN_THIS_POINT();
    dependent->done_writing = true;

    if (event->fired) {
      // LATCH/ONCE/COUNTED: event may already be freed — UB per OCR semantics
      if (event->type == ARTS_EVENT_LATCH || event->type == ARTS_EVENT_ONCE ||
          event->type == ARTS_EVENT_COUNTED) {
        return;
      }
      // STICKY/IDEM: event persists — self-signal for out-of-range callbacks
      while (event->pos == 0) {
        ;
      }
      if (event->pos - 1 <= position) {
        arts_edt_dep_t arg;
        arg.guid = event->data;
        arg.ptr = arts_route_table_lookup_db(event->data, NULL, false);
        callback_t(arg);
        if (arg.ptr) {
          arts_route_table_return_db(event->data, false);
        }
      }
    }
  }
}

bool arts_is_event_fired(arts_guid_t event) {
  bool fired = false;
  struct arts_event_s *actual_event =
      (struct arts_event_s *)arts_route_table_lookup_item(event);
  if (actual_event) {
    fired = actual_event->fired;
  }
  return fired;
}
