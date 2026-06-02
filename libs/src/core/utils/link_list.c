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
#include "arts/utils/link_list.h"

#include <stdatomic.h>

#include "arts.h"
#include "arts/utils/malloc.h"

/* Lock-free Vyukov MPSC.  Mirrors arts/utils/mpsc.h exactly; kept inline here
 * so the transport's per-(rank,port) outbound queues stay a self-contained
 * utility (multi-producer push_back, single-consumer pop_front). */

static inline void push_node(struct arts_link_list_s *list,
                             struct arts_link_list_item_s *n) {
  atomic_store_explicit(&n->next, NULL, memory_order_relaxed);
  struct arts_link_list_item_s *prev =
      atomic_exchange_explicit(&list->head, n, memory_order_acq_rel);
  atomic_store_explicit(&prev->next, n, memory_order_release);
}

void arts_link_list_new(struct arts_link_list_s *list) {
  atomic_store_explicit(&list->stub.next, NULL, memory_order_relaxed);
  atomic_store_explicit(&list->head, &list->stub, memory_order_relaxed);
  list->tail = &list->stub;
}

struct arts_link_list_s *arts_link_list_group_new(unsigned int list_size) {
  struct arts_link_list_s *link_list = (struct arts_link_list_s *)arts_calloc(
      list_size, sizeof(struct arts_link_list_s));
  for (unsigned int i = 0; i < list_size; i++) {
    arts_link_list_new(&link_list[i]);
  }
  return link_list;
}

/* Drain (single-consumer / quiescent) + free every heap node, then free the
 * list array itself.  The embedded stub is never heap-freed. */
void arts_link_list_delete(void *link_list) {
  struct arts_link_list_s *list = (struct arts_link_list_s *)link_list;
  void *data;
  while ((data = arts_link_list_pop_front(list, NULL)) != NULL) {
    arts_link_list_delete_item(data);
  }
  arts_free(link_list);
}

void *arts_link_list_new_item(unsigned int size) {
  struct arts_link_list_item_s *new_item =
      (struct arts_link_list_item_s *)arts_calloc(
          1, sizeof(struct arts_link_list_item_s) + size);
  atomic_store_explicit(&new_item->next, NULL, memory_order_relaxed);
  if (size) {
    return (void *)(new_item + 1);
  }
  return NULL;
}

void arts_link_list_delete_item(void *to_delete) {
  struct arts_link_list_item_s *item =
      ((struct arts_link_list_item_s *)to_delete) - 1;
  arts_free(item);
}

inline struct arts_link_list_s *
arts_link_list_get(struct arts_link_list_s *link_list, unsigned int position) {
  return (struct arts_link_list_s *)(link_list + position);
}

/* Conservative single-consumer emptiness: true only once the consumer has
 * caught up to the producer head.  Returns false while a producer has appended
 * (or is mid-link), so a drain loop keeps polling. */
uint8_t arts_link_list_is_empty(struct arts_link_list_s *link_list) {
  struct arts_link_list_item_s *head =
      atomic_load_explicit(&link_list->head, memory_order_acquire);
  return (head == link_list->tail) ? 1u : 0u;
}

/* Lock-free; any number of producers concurrently.  `item` is the data ptr
 * returned by new_item; the node header precedes it. */
void arts_link_list_push_back(struct arts_link_list_s *list, void *item) {
  push_node(list, ((struct arts_link_list_item_s *)item) - 1);
}

/* Single-consumer pop; returns the data ptr or NULL when empty / transiently
 * inconsistent (a producer is mid-link — caller re-polls).  `free_pos` is a
 * legacy out-param, always set NULL (the caller frees via delete_item). */
void *arts_link_list_pop_front(struct arts_link_list_s *list, void **free_pos) {
  if (free_pos) {
    *free_pos = NULL;
  }
  struct arts_link_list_item_s *tail = list->tail;
  struct arts_link_list_item_s *next =
      atomic_load_explicit(&tail->next, memory_order_acquire);
  if (tail == &list->stub) {
    if (!next) {
      return NULL; /* empty */
    }
    list->tail = next;
    tail = next;
    next = atomic_load_explicit(&tail->next, memory_order_acquire);
  }
  if (next) {
    list->tail = next;
    return (void *)(tail + 1);
  }
  struct arts_link_list_item_s *head =
      atomic_load_explicit(&list->head, memory_order_acquire);
  if (tail != head) {
    return NULL; /* producer mid-link — retry later */
  }
  /* Re-thread the stub so the single remaining node becomes poppable. */
  push_node(list, &list->stub);
  next = atomic_load_explicit(&tail->next, memory_order_acquire);
  if (next) {
    list->tail = next;
    return (void *)(tail + 1);
  }
  return NULL;
}
