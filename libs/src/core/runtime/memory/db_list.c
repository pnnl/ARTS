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
#include "arts/runtime/memory/db_list.h"
#include "arts/gas/route_table.h"
#include "arts/utils/malloc.h"

#include "arts/runtime/compute/edt_functions.h"
#include "arts/runtime/globals.h"
#include "arts/runtime/network/remote_functions.h"
#include "arts/runtime/runtime.h"
#include "arts/system/print.h"
#include "arts/utils/atomics.h"

#define WRITE_SET 0x80000000

void frontier_lock(volatile unsigned int *lock) {
  unsigned int local;
  unsigned int temp;
  while (1) {
    local = *lock;
    if ((local & 1U) == 0) {
      temp = arts_atomic_cswap(lock, local, local | 1U);
      if (temp == local) {
        return;
      }
    }
  }
}

void frontier_unlock(volatile unsigned int *lock) {
  arts_atomic_fetch_and(lock, WRITE_SET);
}

bool frontier_add_read_lock(volatile unsigned int *lock) {
  unsigned int local;
  unsigned int temp;
  while (1) {
    local = *lock;
    // Reject if a writer owns this frontier (or frontier is sealed)
    if ((local & WRITE_SET) != 0) {
      return false;
    }
    if ((local & 1U) == 0) {
      temp = arts_atomic_cswap(lock, local, local | 1U);
      if (temp == local) {
        return true;
      }
    }
  }
}

// Returns true if there is no write in the frontier, false if there is
bool frontier_add_write_lock(volatile unsigned int *lock) {
  unsigned int local;
  unsigned int temp;
  while (1) {
    local = *lock;
    // Reject if another writer already owns this frontier (or sealed)
    if ((local & WRITE_SET) != 0) {
      return false;
    }
    // Wait for lock to be free
    if ((local & 1U) == 0) {
      temp = arts_atomic_cswap(lock, local, local | WRITE_SET | 1U);
      if (temp == local) {
        return true;
      }
    }
  }
}

struct arts_db_element_s *arts_new_db_element() {
  struct arts_db_element_s *ret = (struct arts_db_element_s *)arts_calloc(
      1, sizeof(struct arts_db_element_s));
  if (!ret) {
    ARTS_ERROR("DB element allocation failed");
  }
  return ret;
}

struct arts_db_frontier_s *arts_new_db_frontier() {
  struct arts_db_frontier_s *ret = (struct arts_db_frontier_s *)arts_calloc(
      1, sizeof(struct arts_db_frontier_s));
  if (!ret) {
    ARTS_ERROR("DB frontier allocation failed");
  }
  return ret;
}

// This should be done before being released into the wild
struct arts_db_list_s *arts_new_db_list() {
  struct arts_db_list_s *ret =
      (struct arts_db_list_s *)arts_calloc(1, sizeof(struct arts_db_list_s));
  if (!ret) {
    ARTS_ERROR("DB list allocation failed");
  }
  ret->head = ret->tail = arts_new_db_frontier();
  return ret;
}

void arts_delete_db_element(struct arts_db_element_s *head) {
  struct arts_db_element_s *trail;
  struct arts_db_element_s *current = head;
  while (current) {
    trail = current;
    current = current->next;
    arts_free(trail);
  }
}

void arts_delete_local_delayed_edt(struct arts_local_delayed_edt_s *head) {
  struct arts_local_delayed_edt_s *trail;
  struct arts_local_delayed_edt_s *current = head;
  while (current) {
    trail = current;
    current = current->next;
    arts_free(trail);
  }
}

void arts_delete_db_frontier(struct arts_db_frontier_s *frontier) {
  if (frontier->list.next) {
    arts_delete_db_element(frontier->list.next);
  }
  if (frontier->localDelayed.next) {
    arts_delete_local_delayed_edt(frontier->localDelayed.next);
  }
  arts_free(frontier);
}

void arts_delete_db_list(struct arts_db_list_s *db_list) {
  if (!db_list) {
    return;
  }
  struct arts_db_frontier_s *frontier = db_list->head;
  while (frontier) {
    struct arts_db_frontier_s *next = frontier->next;
    if (frontier->list.next) {
      arts_delete_db_element(frontier->list.next);
    }
    if (frontier->localDelayed.next) {
      arts_delete_local_delayed_edt(frontier->localDelayed.next);
    }
    arts_free(frontier);
    frontier = next;
  }
  arts_free(db_list);
}

bool arts_push_db_to_element(struct arts_db_element_s *head,
                             unsigned int position, unsigned int data) {
  unsigned int j = 0;
  for (struct arts_db_element_s *current = head; current;
       current = current->next) {
    for (unsigned int i = 0; i < DBSPERELEMENT; i++) {
      if (j < position) {
        if (current->array[i] == data) {
          return false;
        }
        j++;
      } else {
        current->array[i] = data;
        return true;
      }
    }
    if (!current->next) {
      current->next = arts_new_db_element();
    }
  }
  // Need to mark unreachable
  return false;
}

void arts_push_delayed_edt(struct arts_local_delayed_edt_s *head,
                           unsigned int position, struct arts_edt_s *edt,
                           unsigned int slot, arts_db_access_mode_t mode) {
  if (!head) {
    return;
  }
  unsigned int num_elements = position / DBSPERELEMENT;
  unsigned int element_pos = position % DBSPERELEMENT;
  struct arts_local_delayed_edt_s *current = head;
  for (unsigned int i = 0; i < num_elements; i++) {
    if (!current->next) {
      current->next = (struct arts_local_delayed_edt_s *)arts_calloc(
          1, sizeof(struct arts_local_delayed_edt_s));
      if (!current->next) {
        ARTS_ERROR("DB local delayed EDT allocation failed");
      }
    }
    current = current->next;
  }
  current->edt[element_pos] = edt;
  current->slot[element_pos] = slot;
  current->mode[element_pos] = mode;
}

bool arts_push_db_to_frontier(struct arts_db_frontier_s *frontier,
                              unsigned int data, bool write, bool local,
                              bool bypass, struct arts_edt_s *edt,
                              arts_guid_t edt_guid, unsigned int slot,
                              arts_db_access_mode_t mode, bool *unique) {
  if (bypass) {
    frontier_lock(&frontier->lock);
  } else if (write && !frontier_add_write_lock(&frontier->lock)) {
    return false;
  } else if (!write && !frontier_add_read_lock(&frontier->lock)) {
    return false;
  }

  bool inserted =
      arts_push_db_to_element(&frontier->list, frontier->position, data);
  if (inserted) {
    frontier->position++;
  }
  *unique = inserted;

  if (inserted && (write && !local)) {
    frontier->exNode = data;
    frontier->exEdtGuid = edt_guid;
    frontier->exEdt = edt;
    frontier->exSlot = slot;
    frontier->exMode = mode;
  } else if (inserted && local) {
    arts_push_delayed_edt(&frontier->localDelayed, frontier->localPosition++,
                          edt, slot, mode);
  }

  frontier_unlock(&frontier->lock);
  return true;
}

// Returns if the push is to the head frontier
/* A read after write from the same node would send duplicate copies of DB.
 * To fix this, if the node is remote, we only return true if the adding the
 * rank to the frontier is unique.  If the db is local then we return if the DB
 * is added to the first frontier reguardless of if there are duplicates.
 */
/*
 * arts_push_db_to_list — Register a rank/EDT in the DB's frontier list.
 *
 * Tries each frontier from head to tail until one accepts the push (i.e.
 * the frontier's lock allows the requested access mode).  The first
 * frontier attempted is always db_list->head (the "current" frontier).
 *
 * on_head (out, optional): set to true if the push landed on the head
 *   frontier, false if a later frontier was used.  Callers use this to
 *   decide whether acquire_dbs should decrement depc_needed directly
 *   (head) or defer to frontier signaling (non-head).
 *
 * Returns true if the rank was inserted uniquely.
 */
bool arts_push_db_to_list(struct arts_db_list_s *db_list, unsigned int data,
                          bool write, bool local, bool bypass,
                          struct arts_edt_s *edt, arts_guid_t edt_guid,
                          unsigned int slot, arts_db_access_mode_t mode,
                          bool *on_head) {
  if (!db_list->head) {
    if (arts_writer_try_lock(&db_list->reader, &db_list->writer)) {
      db_list->head = db_list->tail = arts_new_db_frontier();
      arts_writer_unlock(&db_list->writer);
    }
  }
  arts_reader_lock(&db_list->reader, &db_list->writer);
  bool inserted = false;
  bool unique = true;
  bool is_head = true;
  for (struct arts_db_frontier_s *frontier = db_list->head; frontier;
       frontier = frontier->next) {
    if (arts_push_db_to_frontier(frontier, data, write, local, bypass, edt,
                                 edt_guid, slot, mode, &unique)) {
      inserted = true;
      break;
    }
    is_head = false;
    if (!frontier->next) {
      struct arts_db_frontier_s *new_frontier = arts_new_db_frontier();
      if (arts_atomic_cswap_ptr((volatile void **)&frontier->next, NULL,
                                new_frontier)) {
        arts_delete_db_frontier(new_frontier);
        while (!frontier->next) {
          ;
        }
      }
    }
  }
  if (on_head) {
    *on_head = inserted && is_head;
  }
  arts_reader_unlock(&db_list->reader);
  return inserted && unique;
}

unsigned int arts_current_frontier_size(struct arts_db_list_s *db_list) {
  unsigned int size = 0U;
  arts_reader_lock(&db_list->reader, &db_list->writer);
  if (db_list->head) {
    frontier_lock(&db_list->head->lock);
    size = db_list->head->position;
    frontier_unlock(&db_list->head->lock);
  }
  arts_reader_unlock(&db_list->reader);
  return size;
}

bool arts_db_frontier_iter_init(struct arts_db_frontier_iterator_s *iter,
                                struct arts_db_frontier_s *frontier) {
  if (!frontier || !frontier->position) {
    return false;
  }
  *iter = (struct arts_db_frontier_iterator_s){
      .frontier = frontier,
      .currentElement = &frontier->list,
  };
  return true;
}

unsigned int arts_db_frontier_iter_size(
    struct arts_db_frontier_iterator_s *iter) {
  return iter->frontier->position;
}

bool arts_db_frontier_iter_next(struct arts_db_frontier_iterator_s *iter,
                                unsigned int *next) {
  if (iter->currentIndex < iter->frontier->position) {
    *next = iter->currentElement->array[iter->currentIndex++ % DBSPERELEMENT];
    if (!(iter->currentIndex % DBSPERELEMENT)) {
      iter->currentElement = iter->currentElement->next;
    }
    return true;
  }
  return false;
}

bool arts_db_frontier_iter_has_next(struct arts_db_frontier_iterator_s *iter) {
  return (iter->currentIndex < iter->frontier->position);
}

bool arts_close_frontier(struct arts_db_list_s *db_list,
                         struct arts_db_frontier_iterator_s *iter) {
  bool valid = false;
  arts_reader_lock(&db_list->reader, &db_list->writer);
  struct arts_db_frontier_s *frontier = db_list->head;
  if (frontier) {
    frontier_lock(&frontier->lock);

    arts_atomic_fetch_or(&frontier->lock, WRITE_SET | 1U);
    valid = arts_db_frontier_iter_init(iter, frontier);

    frontier_unlock(&frontier->lock);
  }
  arts_reader_unlock(&db_list->reader);
  return valid;
}

void arts_signal_frontier_remote(struct arts_db_frontier_s *frontier,
                                 struct arts_db_s *db, unsigned int get_from) {
  frontier_lock(&frontier->lock);

  if (frontier->exEdt || frontier->exEdtGuid != NULL_GUID) {
    arts_guid_t edt_guid = frontier->exEdtGuid;
    if (edt_guid == NULL_GUID && frontier->exEdt) {
      edt_guid = frontier->exEdt->current_edt;
    }
    if (frontier->exNode == get_from) {
      arts_remote_send_already_local((int)get_from, db->guid, edt_guid,
                                     frontier->exSlot, frontier->exMode);
    } else if (frontier->exNode != arts_global_rank_id) {
      arts_remote_db_forward_full((int)frontier->exNode, (int)get_from,
                                  db->guid, edt_guid, (int)frontier->exSlot,
                                  frontier->exMode);
    } else {
      arts_remote_db_full_request(db->guid, (int)get_from, edt_guid,
                                  (int)frontier->exSlot, frontier->exMode);
    }
  }

  struct arts_db_frontier_iterator_s iter;
  if (arts_db_frontier_iter_init(&iter, frontier)) {
    unsigned int node;
    while (arts_db_frontier_iter_next(&iter, &node)) {
      if (node != arts_global_rank_id &&
          !((frontier->exEdt || frontier->exEdtGuid != NULL_GUID) &&
            node == frontier->exNode)) {
        arts_remote_db_forward((int)node, (int)get_from, db->guid,
                               DB_MODE_RO);  // Don't care about mode
      }
    }
  }

  if (frontier->localPosition) {
    struct arts_local_delayed_edt_s *current = &frontier->localDelayed;
    for (unsigned int i = 0; i < frontier->localPosition; i++) {
      unsigned int pos = i % DBSPERELEMENT;
      struct arts_edt_s *edt = current->edt[pos];
      unsigned int slot = current->slot[pos];
      arts_remote_db_request(db->guid, (int)get_from, edt, (int)slot,
                             current->mode[pos], true);
      if (pos + 1 == DBSPERELEMENT) {
        current = current->next;
      }
    }
  }

  if (arts_push_db_to_element(&frontier->list, frontier->position, get_from)) {
    frontier->position++;
  }
  frontier_unlock(&frontier->lock);
}

void arts_signal_frontier_local(struct arts_db_frontier_s *frontier,
                                struct arts_db_s *db) {
  frontier_lock(&frontier->lock);

  if (frontier->exEdt || frontier->exEdtGuid != NULL_GUID) {
    arts_guid_t edt_guid = frontier->exEdtGuid;
    struct arts_edt_s *edt = frontier->exEdt;
    if (edt_guid == NULL_GUID && edt) {
      edt_guid = edt->current_edt;
    }
    if (!edt && edt_guid != NULL_GUID) {
      edt = (struct arts_edt_s *)arts_route_table_lookup_item(edt_guid);
    }
    if (frontier->exNode == arts_global_rank_id) {
      if (edt) {
        // TODO(gpu): GPU EDTs need GPU memory, not this CPU pointer.
        arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
        depv[frontier->exSlot].ptr = db + 1;
        if (arts_atomic_sub(&edt->depc_needed, 1U) == 0) {
          arts_handle_remote_stolen_edt(edt);
        }
      } else {
        ARTS_INFO("Local frontier missing EDT[Guid:%lu] on rank %u", edt_guid,
                  arts_global_rank_id);
      }
    } else {
      arts_remote_db_full_send_now((int)frontier->exNode, db, edt_guid,
                                   frontier->exSlot, frontier->exMode);
    }
  }

  struct arts_db_frontier_iterator_s iter;
  if (arts_db_frontier_iter_init(&iter, frontier)) {
    unsigned int node;
    while (arts_db_frontier_iter_next(&iter, &node)) {
      if (node != arts_global_rank_id &&
          !((frontier->exEdt || frontier->exEdtGuid != NULL_GUID) &&
            node == frontier->exNode)) {
        arts_remote_db_send_now((int)node, db);
        ARTS_INFO("Progress Local sending to %u", node);
      }
    }
  }

  if (frontier->localPosition) {
    struct arts_local_delayed_edt_s *current = &frontier->localDelayed;
    for (unsigned int i = 0; i < frontier->localPosition; i++) {
      unsigned int pos = i % DBSPERELEMENT;
      struct arts_edt_s *edt = current->edt[pos];
      // TODO(gpu): GPU EDTs need GPU memory, not this CPU pointer.
      arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
      depv[current->slot[pos]].ptr = db + 1;

      if (arts_atomic_sub(&edt->depc_needed, 1U) == 0) {
        arts_handle_remote_stolen_edt(edt);
      }

      if (pos + 1 == DBSPERELEMENT) {
        current = current->next;
      }
    }
  }
  frontier_unlock(&frontier->lock);
}

void arts_progress_frontier(struct arts_db_s *db, unsigned int rank) {
  struct arts_db_list_s *db_list = (struct arts_db_list_s *)db->db_list;
  arts_writer_lock(&db_list->reader, &db_list->writer);
  struct arts_db_frontier_s *tail = db_list->head;
  if (db_list->head) {
    db_list->head = db_list->head->next;
    if (db_list->head) {
      if (rank == arts_global_rank_id) {
        arts_signal_frontier_local(db_list->head, db);
      } else {
        arts_signal_frontier_remote(db_list->head, db, rank);
      }
    }
  }
  arts_writer_unlock(&db_list->writer);
  // This should be safe since the writer lock ensures all readers are done
  if (tail) {
    arts_delete_db_frontier(tail);
  }
}

bool arts_progress_and_get_frontier(struct arts_db_list_s *db_list,
                                    struct arts_db_frontier_iterator_s *iter) {
  arts_writer_lock(&db_list->reader, &db_list->writer);
  struct arts_db_frontier_s *tail = db_list->head;
  db_list->head = db_list->head->next;
  arts_writer_unlock(&db_list->writer);
  // This should be safe since the writer lock ensures all readers are done
  return arts_db_frontier_iter_init(iter, tail);
}
