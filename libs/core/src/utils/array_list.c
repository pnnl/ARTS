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
#include "arts/utils/array_list.h"

#include <string.h>

#include "arts.h"
#include "arts/utils/malloc.h"

arts_array_list_element_t *arts_new_array_list_element(uint64_t start,
                                                       size_t element_size,
                                                       size_t array_length) {
  arts_array_list_element_t *ret = (arts_array_list_element_t *)arts_malloc(
      sizeof(arts_array_list_element_t) + (element_size * array_length));
  ret->start = start;
  ret->next = NULL;
  ret->array = (void *)(1 + ret);
  return ret;
}

arts_array_list_t *arts_new_array_list(size_t element_size,
                                       size_t array_length) {
  arts_array_list_t *ret =
      (arts_array_list_t *)arts_malloc(sizeof(arts_array_list_t));
  ret->element_size = element_size;
  ret->array_length = array_length;
  ret->head = ret->current =
      arts_new_array_list_element(0, element_size, array_length);
  ret->index = 0;
  ret->lastRequest = 0;
  ret->lastRequestPtr = ret->head->array;
  return ret;
}

void arts_delete_array_list(arts_array_list_t *a_list) {
  arts_array_list_element_t *trail;
  arts_array_list_element_t *current = a_list->head;
  while (current) {
    trail = current;
    current = current->next;
    arts_free(trail);
  }
  arts_free(a_list);
}

uint64_t arts_push_to_array_list(arts_array_list_t *a_list, void *element) {
  uint64_t index = a_list->index;
  if (!(a_list->index % a_list->array_length) && a_list->index) {
    if (!a_list->current->next) {
      a_list->current->next = arts_new_array_list_element(
          a_list->current->start + a_list->array_length, a_list->element_size,
          a_list->array_length);
    }
    a_list->current = a_list->current->next;
  }
  uint64_t offset = a_list->index - a_list->current->start;
  void *ptr = (void *)((char *)a_list->current->array +
                       (offset * a_list->element_size));
  memcpy(ptr, element, a_list->element_size);
  a_list->index++;
  return index;
}

void *arts_next_free_from_array_list(arts_array_list_t *a_list) {
  uint64_t index = a_list->index;
  if (!(a_list->index % a_list->array_length) && a_list->index) {
    if (!a_list->current->next) {
      a_list->current->next = arts_new_array_list_element(
          a_list->current->start + a_list->array_length, a_list->element_size,
          a_list->array_length);
    }
    a_list->current = a_list->current->next;
  }
  uint64_t offset = a_list->index - a_list->current->start;
  void *ptr = (void *)((char *)a_list->current->array +
                       (offset * a_list->element_size));
  a_list->index++;
  return ptr;
}

void arts_reset_array_list(arts_array_list_t *a_list) {
  a_list->current = a_list->head;
  a_list->index = 0;
  a_list->lastRequest = 0;
  a_list->lastRequestPtr = a_list->head->array;
}

uint64_t arts_length_array_list(arts_array_list_t *a_list) {
  return a_list->index;
}

void *arts_get_from_array_list(arts_array_list_t *a_list, uint64_t index) {
  if (a_list) {
    // Fastest Path
    if (index == a_list->lastRequest) {
      return a_list->lastRequestPtr;
    }

    if (index < a_list->index) {
      a_list->lastRequest = index;

      // Faster Path
      if (a_list->index < a_list->array_length) {
        a_list->lastRequestPtr = (void *)((char *)a_list->head->array +
                                          (index * a_list->element_size));
        return a_list->lastRequestPtr;
      }

      // Slow Path
      arts_array_list_element_t *node = a_list->head;
      while (node && index >= node->start + a_list->array_length) {
        node = node->next;
      }
      if (node) {
        uint64_t offset = index - node->start;
        a_list->lastRequestPtr =
            (void *)((char *)node->array + (offset * a_list->element_size));
        return a_list->lastRequestPtr;
      }
    }
  }
  return NULL;
}

arts_array_list_iterator_t *
arts_new_array_list_iterator(arts_array_list_t *a_list) {
  arts_array_list_iterator_t *iter = (arts_array_list_iterator_t *)arts_malloc(
      sizeof(arts_array_list_iterator_t));
  iter->index = 0;
  iter->last = a_list->index;
  iter->element_size = a_list->element_size;
  iter->array_length = a_list->array_length;
  iter->current = a_list->head;
  return iter;
}

void *arts_array_list_next(arts_array_list_iterator_t *iter) {
  void *ret = NULL;
  if (iter) {
    if (iter->index < iter->last) {
      if (!(iter->index % iter->array_length) && iter->index) {
        iter->current = iter->current->next;
      }
      if (iter->current) {
        ret = (void *)((char *)iter->current->array +
                       ((iter->index - iter->current->start) *
                        iter->element_size));
        iter->index++;
      }
    }
  }
  return ret;
}

bool arts_array_list_has_next(arts_array_list_iterator_t *iter) {
  return (iter->index < iter->last);
}

void arts_delete_array_list_iterator(arts_array_list_iterator_t *iter) {
  arts_free(iter);
}
