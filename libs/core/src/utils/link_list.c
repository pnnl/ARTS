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

#include "arts.h"
#include "arts/utils/malloc.h"
#include "arts/utils/atomics.h"

void arts_link_list_new(struct arts_link_list_s *list) {
  list->headPtr = list->tailPtr = NULL;
}

void arts_link_list_delete(void *link_list) {
  struct arts_link_list_s *list = (struct arts_link_list_s *)link_list;
  struct arts_link_list_item_s *last;
  while (list->headPtr != NULL) {
    last = list->headPtr;
    list->headPtr = list->headPtr->next;
    arts_free(last);
  }
  arts_free(link_list);
}

struct arts_link_list_s *arts_link_list_group_new(unsigned int list_size) {
  struct arts_link_list_s *link_list =
      (struct arts_link_list_s *)arts_calloc(list_size, sizeof(struct arts_link_list_s));
  for (int i = 0; i < list_size; i++) {
    arts_link_list_new(&link_list[i]);
  }
  return link_list;
}

void *arts_link_list_new_item(unsigned int size) {
  struct arts_link_list_item_s *new_item = (struct arts_link_list_item_s *)arts_calloc(
      1, sizeof(struct arts_link_list_item_s) + size);
  new_item->next = NULL;
  if (size) {
    return (void *)(new_item + 1);
  }
  return NULL;
}

void arts_link_list_delete_item(void *to_delete) {
  struct arts_link_list_item_s *item = ((struct arts_link_list_item_s *)to_delete) - 1;
  arts_free(item);
}

inline struct arts_link_list_s *arts_link_list_get(struct arts_link_list_s *link_list,
                                            unsigned int position) {
  return (struct arts_link_list_s *)(link_list + position);
}

inline unsigned arts_link_list_get_size(struct arts_link_list_s *link_list) {
  unsigned size = 0;
  struct arts_link_list_item_s *head = NULL;
  arts_lock(&link_list->lock);
  head = link_list->headPtr;
  while (head != NULL) {
    size++;
    head = head->next;
  }
  arts_unlock(&link_list->lock);
  return size;
}

inline uint8_t arts_link_list_is_empty(struct arts_link_list_s *link_list) {
  struct arts_link_list_item_s *head = NULL;
  arts_lock(&link_list->lock);
  head = link_list->headPtr;
  arts_unlock(&link_list->lock);
  return (head == NULL);
}

void *arts_link_list_get_front_data(struct arts_link_list_s *link_list) {
  void *data = NULL;
  arts_lock(&link_list->lock);
  data = link_list->headPtr + 1;
  arts_unlock(&link_list->lock);
  return data;
}

void *arts_link_list_get_tail_data(struct arts_link_list_s *link_list) {
  void *data = NULL;
  arts_lock(&link_list->lock);
  data = link_list->tailPtr + 1;
  arts_unlock(&link_list->lock);
  return data;
}

void arts_link_list_push_back(struct arts_link_list_s *list, void *item) {
  struct arts_link_list_item_s *new_item = (struct arts_link_list_item_s *)item;
  new_item -= 1;
  arts_lock(&list->lock);
  if (list->headPtr == NULL) {
    list->headPtr = list->tailPtr = new_item;
  } else {
    list->tailPtr->next = new_item;
    list->tailPtr = new_item;
  }
  arts_unlock(&list->lock);
}

void *arts_link_list_pop_front(struct arts_link_list_s *list, void **free_pos) {
  void *data = NULL;
  if (free_pos) {
    *free_pos = NULL;
}
  arts_lock(&list->lock);
  if (list->headPtr) {
    data = (void *)(list->headPtr + 1);
    list->headPtr = list->headPtr->next;
  }
  arts_unlock(&list->lock);
  return data;
}
