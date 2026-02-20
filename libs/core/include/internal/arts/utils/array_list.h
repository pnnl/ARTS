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
#ifndef ARTS_UTILS_ARRAYLIST_H
#define ARTS_UTILS_ARRAYLIST_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct arts_array_list_element_s arts_array_list_element_t;

struct arts_array_list_element_s {
  uint64_t start;
  arts_array_list_element_t *next;
  void *array;
};

typedef struct {
  size_t element_size;
  size_t array_length;
  arts_array_list_element_t *head;
  arts_array_list_element_t *current;
  uint64_t index;
  uint64_t lastRequest;
  void *lastRequestPtr;
} arts_array_list_t;

typedef struct {
  uint64_t index;
  uint64_t last;
  size_t element_size;
  size_t array_length;
  arts_array_list_element_t *current;
} arts_array_list_iterator_t;

arts_array_list_element_t *arts_new_array_list_element(uint64_t start,
                                                       size_t element_size,
                                                       size_t array_length);
arts_array_list_t *arts_new_array_list(size_t element_size,
                                       size_t array_length);
void arts_delete_array_list(arts_array_list_t *a_list);
uint64_t arts_push_to_array_list(arts_array_list_t *a_list, void *element);
void *arts_next_free_from_array_list(arts_array_list_t *a_list);
void arts_reset_array_list(arts_array_list_t *a_list);
uint64_t arts_length_array_list(arts_array_list_t *a_list);
void *arts_get_from_array_list(arts_array_list_t *a_list, uint64_t index);
arts_array_list_iterator_t *
arts_new_array_list_iterator(arts_array_list_t *a_list);
void *arts_array_list_next(arts_array_list_iterator_t *iter);
bool arts_array_list_has_next(arts_array_list_iterator_t *iter);
void arts_delete_array_list_iterator(arts_array_list_iterator_t *iter);

#ifdef __cplusplus
}
#endif

#endif /* ARTS_UTILS_ARRAYLIST_H */
