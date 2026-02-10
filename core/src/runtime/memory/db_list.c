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

#include "arts/runtime/globals.h"
#include "arts/runtime/runtime.h"
#include "arts/runtime/compute/edt_functions.h"
#include "arts/runtime/network/remote_functions.h"
#include "arts/system/arts_print.h"
#include "arts/system/debug.h"
#include "arts/utils/atomics.h"

#define WRITE_SET 0x80000000
#define EXCLUSIVE_SET 0x40000000

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
  unsigned int mask = WRITE_SET | EXCLUSIVE_SET;
  arts_atomic_fetch_and(lock, mask);
}

bool frontier_add_read_lock(volatile unsigned int *lock) {
  unsigned int local;
  unsigned int temp;
  while (1) {
    local = *lock;
    // Make sure exclusive not set first
    if ((local & EXCLUSIVE_SET) != 0) {
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
  // ARTS_DEBUG("Wlocking frontier: >>>>>>> %p", lock);
  unsigned int local;
  unsigned int temp;
  while (1) {
    local = *lock;
    // Make sure exclusive not set first
    if ((local & EXCLUSIVE_SET) != 0) {
      return false;
}
    // Make sure write not set first
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

bool frontier_add_exclusive_lock(volatile unsigned int *lock) {
  // ARTS_DEBUG("Elocking frontier: >>>>>>> %p", lock);
  unsigned int local;
  unsigned int temp;
  while (1) {
    local = *lock;
    // Make sure exclusive not set first
    if ((local & EXCLUSIVE_SET) != 0) {
      return false;
}
    // Make sure write not set first
    if ((local & WRITE_SET) != 0) {
      return false;
}
    // We reserved the write, now wait for lock to be free
    if ((local & 1U) == 0) {
      temp = arts_atomic_cswap(lock, local, local | EXCLUSIVE_SET | WRITE_SET | 1U);
      if (temp == local) {
        return true;
}
    }
  }
}

struct arts_db_element *arts_new_db_element() {
  struct arts_db_element *ret =
      (struct arts_db_element *)arts_calloc(1, sizeof(struct arts_db_element));
  if (!ret) {
    arts_debug_generate_seg_fault();
  }
  return ret;
}

struct arts_db_frontier *arts_new_db_frontier() {
  struct arts_db_frontier *ret =
      (struct arts_db_frontier *)arts_calloc(1, sizeof(struct arts_db_frontier));
  if (!ret) {
    arts_debug_generate_seg_fault();
  }
  return ret;
}

// This should be done before being released into the wild
struct arts_db_list *arts_new_db_list() {
  struct arts_db_list *ret =
      (struct arts_db_list *)arts_calloc(1, sizeof(struct arts_db_list));
  if (!ret) {
    arts_debug_generate_seg_fault();
  }
  ret->head = ret->tail = arts_new_db_frontier();
  return ret;
}

void arts_delete_db_element(struct arts_db_element *head) {
  struct arts_db_element *trail;
  struct arts_db_element *current = head;
  while (current) {
    trail = current;
    current = current->next;
    arts_free(trail);
  }
}

void arts_delete_local_delayed_edt(struct arts_local_delayed_edt *head) {
  struct arts_local_delayed_edt *trail;
  struct arts_local_delayed_edt *current = head;
  while (current) {
    trail = current;
    current = current->next;
    arts_free(trail);
  }
}

void arts_delete_db_frontier(struct arts_db_frontier *frontier) {
  if (frontier->list.next) {
    arts_delete_db_element(frontier->list.next);
}
  if (frontier->localDelayed.next) {
    arts_delete_local_delayed_edt(frontier->localDelayed.next);
}
  arts_free(frontier);
}

bool arts_push_db_to_element(struct arts_db_element *head, unsigned int position,
                         unsigned int data) {
  unsigned int j = 0;
  for (struct arts_db_element *current = head; current; current = current->next) {
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

void arts_push_delayed_edt(struct arts_local_delayed_edt *head, unsigned int position,
                        struct arts_edt *edt, unsigned int slot,
                        arts_type_t mode) {
  unsigned int num_elements = position / DBSPERELEMENT;
  unsigned int element_pos = position % DBSPERELEMENT;
  struct arts_local_delayed_edt *current = head;
  for (unsigned int i = 0; i < num_elements; i++) {
    if (!current->next) {
      current->next = (struct arts_local_delayed_edt *)arts_calloc(
          1, sizeof(struct arts_local_delayed_edt));
      if (!current->next) {
        arts_debug_generate_seg_fault();
      }
    }
    current = current->next;
  }
  current->edt[element_pos] = edt;
  current->slot[element_pos] = slot;
  current->mode[element_pos] = mode;
}

bool arts_push_db_to_frontier(struct arts_db_frontier *frontier, unsigned int data,
                          bool write, bool exclusive, bool local, bool bypass,
                          struct arts_edt *edt, arts_guid_t edt_guid,
                          unsigned int slot, arts_type_t mode, bool *unique) {
  if (bypass) {
    frontier_lock(&frontier->lock);
  } else if (exclusive && !frontier_add_exclusive_lock(&frontier->lock)) {
    return false;
  } else if (write && !frontier_add_write_lock(&frontier->lock)) {
    return false;
  } else if (!exclusive && !write && !frontier_add_read_lock(&frontier->lock)) {
    return false;
  }

  bool inserted =
      arts_push_db_to_element(&frontier->list, frontier->position, data);
  if (inserted) {
    frontier->position++;
}
  *unique = inserted;

  if (inserted && (exclusive || (write && !local))) {
    frontier->exNode = data;
    frontier->exEdtGuid = edt_guid;
    frontier->exEdt = edt;
    frontier->exSlot = slot;
    frontier->exMode = mode;
  } else if (inserted && local) {
    arts_push_delayed_edt(&frontier->localDelayed, frontier->localPosition++, edt,
                       slot, mode);
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
bool arts_push_db_to_list(struct arts_db_list *db_list, unsigned int data, bool write,
                      bool exclusive, bool local, bool bypass,
                      struct arts_edt *edt, arts_guid_t edt_guid,
                      unsigned int slot, arts_type_t mode) {
  if (!db_list->head) {
    if (arts_writer_try_lock(&db_list->reader, &db_list->writer)) {
      db_list->head = db_list->tail = arts_new_db_frontier();
      arts_writer_unlock(&db_list->writer);
    }
  }
  arts_reader_lock(&db_list->reader, &db_list->writer);
  bool inserted = false;
  bool unique = true;
  for (struct arts_db_frontier *frontier = db_list->head; frontier;
       frontier = frontier->next) {
    if (arts_push_db_to_frontier(frontier, data, write, exclusive, local, bypass,
                             edt, edt_guid, slot, mode, &unique)) {
      inserted = true;
      break;
    }
    if (!frontier->next) {
      struct arts_db_frontier *new_frontier = arts_new_db_frontier();
      if (arts_atomic_cswap_ptr((volatile void **)&frontier->next, NULL,
                             new_frontier)) {
        arts_delete_db_frontier(new_frontier);
        while (!frontier->next) {
          ;
}
      }
    }
  }
  arts_reader_unlock(&db_list->reader);
  return inserted && unique;
}

unsigned int arts_current_frontier_size(struct arts_db_list *db_list) {
  unsigned int size = 0U;
  arts_reader_lock(&db_list->reader, &db_list->writer);
  if (db_list->head) {
    frontier_lock(&db_list->head->lock);
    size = db_list->head->position;
    frontier_unlock(&db_list->head->lock);
  }
  arts_reader_unlock(&db_list->head->lock);
  return size;
}

struct arts_db_frontier_iterator *
arts_db_frontier_iter_create(struct arts_db_frontier *frontier) {
  struct arts_db_frontier_iterator *iter = NULL;
  if (frontier && frontier->position) {
    iter = (struct arts_db_frontier_iterator *)arts_calloc(
        1, sizeof(struct arts_db_frontier_iterator));
    if (!iter) {
      arts_debug_generate_seg_fault();
    }
    iter->frontier = frontier;
    iter->currentElement = &frontier->list;
  }
  // Need to mark unreachable
  return NULL;
}

unsigned int arts_db_frontier_iter_size(struct arts_db_frontier_iterator *iter) {
  return iter->frontier->position;
}

bool arts_db_frontier_iter_next(struct arts_db_frontier_iterator *iter,
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

bool arts_db_frontier_iter_has_next(struct arts_db_frontier_iterator *iter) {
  return (iter->currentIndex < iter->frontier->position);
}

void arts_db_frontier_iter_delete(struct arts_db_frontier_iterator *iter) {
  arts_free(iter->frontier);
  arts_free(iter);
}

struct arts_db_frontier_iterator *arts_close_frontier(struct arts_db_list *db_list) {
  struct arts_db_frontier_iterator *iter = NULL;
  arts_reader_lock(&db_list->reader, &db_list->writer);
  struct arts_db_frontier *frontier = db_list->head;
  if (frontier) {
    frontier_lock(&frontier->lock);

    arts_atomic_fetch_or(&frontier->lock, EXCLUSIVE_SET | WRITE_SET | 1U);
    iter = arts_db_frontier_iter_create(frontier);

    frontier_unlock(&frontier->lock);
  }
  arts_reader_unlock(&db_list->reader);
  return iter;
}

void arts_signal_frontier_remote(struct arts_db_frontier *frontier,
                              struct arts_db *db, unsigned int get_from) {
  frontier_lock(&frontier->lock);

  if (frontier->exEdt || frontier->exEdtGuid != NULL_GUID) {
    arts_guid_t edt_guid = frontier->exEdtGuid;
    if (edt_guid == NULL_GUID && frontier->exEdt) {
      edt_guid = frontier->exEdt->current_edt;
}
    if (frontier->exNode == get_from) {
      arts_remote_send_already_local(get_from, db->guid, edt_guid, frontier->exSlot,
                                 frontier->exMode);
    } else if (frontier->exNode != arts_global_rank_id) {
      arts_remote_db_forward_full(frontier->exNode, get_from, db->guid, edt_guid,
                              frontier->exSlot, frontier->exMode);
    } else {
      arts_remote_db_full_request(db->guid, get_from, edt_guid, frontier->exSlot,
                              frontier->exMode);
}
  }

  struct arts_db_frontier_iterator *iter = arts_db_frontier_iter_create(frontier);
  if (iter) {
    unsigned int node;
    while (arts_db_frontier_iter_next(iter, &node)) {
      if (node != arts_global_rank_id &&
          !((frontier->exEdt || frontier->exEdtGuid != NULL_GUID) &&
            node == frontier->exNode)) {
        arts_remote_db_forward(node, get_from, db->guid,
                            ARTS_DB_READ); // Don't care about mode
      }
    }
  }

  if (frontier->localPosition) {
    struct arts_local_delayed_edt *current = &frontier->localDelayed;
    for (unsigned int i = 0; i < frontier->localPosition; i++) {
      unsigned int pos = i % DBSPERELEMENT;
      struct arts_edt *edt = current->edt[pos];
      unsigned int slot = current->slot[pos];
      // send through aggregation
      arts_remote_db_request(db->guid, get_from, edt, slot, ARTS_DB_READ, true,
                          ARTS_NULL);
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

void arts_signal_frontier_local(struct arts_db_frontier *frontier,
                             struct arts_db *db) {
  frontier_lock(&frontier->lock);

  if (frontier->exEdt || frontier->exEdtGuid != NULL_GUID) {
    arts_guid_t edt_guid = frontier->exEdtGuid;
    struct arts_edt *edt = frontier->exEdt;
    if (edt_guid == NULL_GUID && edt) {
      edt_guid = edt->current_edt;
}
    if (!edt && edt_guid != NULL_GUID) {
      edt = (struct arts_edt *)arts_route_table_lookup_item(edt_guid);
}
    if (frontier->exNode == arts_global_rank_id) {
      if (edt) {
        arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
        depv[frontier->exSlot].ptr = db + 1;
        if (arts_atomic_sub(&edt->depcNeeded, 1U) == 0) {
          arts_handle_remote_stolen_edt(edt);
}
      } else {
        ARTS_INFO("Local frontier missing EDT[Guid:%lu] on rank %u", edt_guid,
                  arts_global_rank_id);
      }
    } else {
      arts_remote_db_full_send_now(frontier->exNode, db, edt_guid, frontier->exSlot,
                              frontier->exMode);
    }
  }

  struct arts_db_frontier_iterator *iter = arts_db_frontier_iter_create(frontier);
  if (iter) {
    unsigned int node;
    while (arts_db_frontier_iter_next(iter, &node)) {
      if (node != arts_global_rank_id &&
          !((frontier->exEdt || frontier->exEdtGuid != NULL_GUID) &&
            node == frontier->exNode)) {
        arts_remote_db_send_now(node, db);
        ARTS_INFO("Progress Local sending to %u", node);
      }
    }
  }

  if (frontier->localPosition) {
    struct arts_local_delayed_edt *current = &frontier->localDelayed;
    for (unsigned int i = 0; i < frontier->localPosition; i++) {
      unsigned int pos = i % DBSPERELEMENT;
      struct arts_edt *edt = current->edt[pos];
      // This is prob wrong now with GPUs
      arts_edt_dep_t *depv = (arts_edt_dep_t *)arts_get_depv(edt);
      depv[current->slot[pos]].ptr = db + 1;

      if (arts_atomic_sub(&edt->depcNeeded, 1U) == 0) {
        arts_handle_remote_stolen_edt(edt);
      }

      if (pos + 1 == DBSPERELEMENT) {
        current = current->next;
}
    }
  }
  frontier_unlock(&frontier->lock);
}

void arts_progress_frontier(struct arts_db *db, unsigned int rank) {
  struct arts_db_list *db_list = (struct arts_db_list *)db->db_list;
  arts_writer_lock(&db_list->reader, &db_list->writer);
  struct arts_db_frontier *tail = db_list->head;
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

struct arts_db_frontier_iterator *
arts_progress_and_get_frontier(struct arts_db_list *db_list) {
  arts_writer_lock(&db_list->reader, &db_list->writer);
  struct arts_db_frontier *tail = db_list->head;
  db_list->head = db_list->head->next;
  arts_writer_unlock(&db_list->writer);
  // This should be safe since the writer lock ensures all readers are done
  return arts_db_frontier_iter_create(tail);
}
