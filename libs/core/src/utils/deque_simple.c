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

// Copyright (c) 2013 Amanieu d'Antras
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#include "arts/utils/deque.h"

#include <string.h>

#include "arts.h"
#include "arts/arts_defs.h"
#include "arts/utils/atomics.h"
#include "arts/utils/malloc.h"

/*
 * Chase-Lev work-stealing deque (simplified variant).
 *
 * Lock-free concurrent deque with LIFO push/pop from the owning thread
 * (front) and FIFO steal from other threads (back).  Each worker thread
 * has its own deque; other threads steal from the back.
 *
 * Fields are cache-line padded (64-byte aligned) to prevent false sharing
 * between the owning thread (modifies bottom) and stealers (read/CAS top).
 */
struct circular_array_s {
  struct circular_array_s *next;
  unsigned int size;
  void **segment;
} __attribute__((aligned(64)));

struct arts_deque_s {
  volatile uint64_t top; /* Modified by stealers via CAS */
  char pad1[56];
  volatile uint64_t bottom; /* Modified only by the owning thread */
  char pad2[56];
  struct circular_array_s *volatile activeArray;
  char pad3[56];
  struct circular_array_s
      *head; /* Head of circular array chain (for cleanup) */
  volatile unsigned int push;
  volatile unsigned int pop;
  volatile unsigned int steal;
} __attribute__((aligned(64)));

static inline struct circular_array_s *new_circular_array(unsigned int size) {
  struct circular_array_s *array = (struct circular_array_s *)arts_calloc_align(
      1, sizeof(struct circular_array_s) + (sizeof(void *) * size), 64);
  //    memset(array,0,sizeof(struct circular_array_s) + sizeof(void*) * size);
  array->size = size;
  array->segment = (void **)(array + 1);
  array->next = NULL;
  return array;
}

bool arts_deque_empty(struct arts_deque_s *deque) {
  (void)deque;
  // don't really know what this is for
  // return (deque->bottom == deque->top);
  return false;
}

void arts_deque_clear(struct arts_deque_s *deque) {
  deque->top = deque->bottom;
}

unsigned int arts_deque_size(struct arts_deque_s *deque) {
  return deque->bottom - deque->top;
}

static inline void *get_circular_array(struct circular_array_s *array,
                                       uint64_t i) {
  return array->segment[i % array->size];
}

ARTS_THREAD_LOCAL void *steal_array[STEALSIZE];

static inline void get_multiple_circular_array(struct circular_array_s *array,
                                               uint64_t i) {
  if ((i % array->size) + STEALSIZE < array->size) {
    memcpy(steal_array, &array->segment[i % array->size],
           sizeof(void *) * STEALSIZE);
  } else {
    for (unsigned int j = 0; j < STEALSIZE; j++) {
      steal_array[j] = array->segment[(i + j) % array->size];
    }
  }
}

static inline void put_circular_array(struct circular_array_s *array,
                                      uint64_t i, void *object) {
  array->segment[i % array->size] = object;
}

static inline struct circular_array_s *
grow_circular_array(struct circular_array_s *array, uint64_t b, uint64_t t) {
  struct circular_array_s *a = new_circular_array(array->size * 2);
  array->next = a;
  uint64_t i;
  for (i = t; i < b; i++) {
    put_circular_array(a, i, get_circular_array(array, i));
  }
  return a;
}

static inline void arts_deque_new_init(struct arts_deque_s *deque,
                                       unsigned int size) {
  deque->top = 1;
  deque->bottom = 1;
  deque->activeArray = new_circular_array(size);
  deque->head = deque->activeArray;
  deque->push = 0;
  deque->pop = 0;
  deque->steal = 0;
}

struct arts_deque_s *arts_deque_new(unsigned int size) {
  struct arts_deque_s *deque = (struct arts_deque_s *)arts_calloc_align(
      1, sizeof(struct arts_deque_s), 64);
  arts_deque_new_init(deque, size);
  return deque;
}

void arts_deque_delete(struct arts_deque_s *deque) {
  struct circular_array_s *trail;
  struct circular_array_s *current = deque->head;
  while (current) {
    trail = current;
    current = current->next;
    arts_free(trail);
  }
  arts_free(deque);
}

bool arts_deque_push_front(struct arts_deque_s *deque, void *item,
                           unsigned int priority) {
  (void)priority;
  struct circular_array_s *a = deque->activeArray;
  uint64_t b = deque->bottom;
  uint64_t t = deque->top;
  if (b >= a->size - 1 + t) {
    a = grow_circular_array(a, b, t);
    deque->activeArray = a;
  }
  put_circular_array(a, b, item);
  HW_MEMORY_FENCE();
  deque->bottom = b + 1;
  return true;
}

void *arts_deque_pop_front(struct arts_deque_s *deque) {
  uint64_t b = --deque->bottom;
  HW_MEMORY_FENCE();
  uint64_t t = deque->top;
  if (t > b) {
    deque->bottom = t;
    return NULL;
  }
  void *o = get_circular_array(deque->activeArray, b);
  // Success
  if (b > t) {
    return o;
  }
  if (arts_atomic_cswap_u64(&deque->top, t, t + 1) != t) {
    o = NULL;
  }
  deque->bottom = t + 1;
  return o;
}

void *arts_deque_pop_back(struct arts_deque_s *deque) {
  uint64_t t = deque->top;
  HW_MEMORY_FENCE();
  uint64_t b = deque->bottom;
  if (t < b) {
    void *o = get_circular_array(deque->activeArray, t);
    uint64_t temp = arts_atomic_cswap_u64(&deque->top, t, t + 1);
    if (temp == t) {
      return o;
    }
  }
  return NULL;
}

struct arts_deque_s *arts_deque_list_new(unsigned int list_size,
                                         unsigned int deque_size) {
  struct arts_deque_s *deque_list = (struct arts_deque_s *)arts_calloc_align(
      list_size, sizeof(struct arts_deque_s), 64);
  unsigned int i = 0;
  for (i = 0; i < list_size; i++) {
    arts_deque_new_init(&deque_list[i], deque_size);
  }

  return deque_list;
}

struct arts_deque_s *arts_deque_list_get_deque(struct arts_deque_s *deque_list,
                                               unsigned int position) {
  return deque_list + position;
}

void arts_deque_list_delete(void *deque_list) {
  //    arts_deque * ptr = (arts_deque*) deque_list;
  //    for (i = 0; i < list_size; i++)
  //        arts_deque_delete( deque_list+i  , deque_size);
  //    arts_free(deque_list);
}
