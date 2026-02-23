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

/*
 * deque.c — Runtime-selectable deque dispatch layer.
 *
 * Provides the generic arts_deque_*() API that all call sites use.
 * At startup, arts_deque_select() picks an implementation (simple or
 * priority) by setting a vtable pointer.  All subsequent calls dispatch
 * through this pointer with zero branching overhead.
 */
#include "arts/utils/deque.h"

#include "arts/system/print.h"

/*--- Static vtable instances -----------------------------------------------*/

static const struct arts_deque_ops_s simple_ops = {
    .list_new = arts_deque_simple_list_new,
    .list_get_deque = arts_deque_simple_list_get_deque,
    .list_delete = arts_deque_simple_list_delete,
    .new_deque = arts_deque_simple_new,
    .delete_deque = arts_deque_simple_delete,
    .push_front = arts_deque_simple_push_front,
    .pop_front = arts_deque_simple_pop_front,
    .pop_back = arts_deque_simple_pop_back,
    .empty = arts_deque_simple_empty,
    .clear = arts_deque_simple_clear,
    .size = arts_deque_simple_size,
};

static const struct arts_deque_ops_s priority_ops = {
    .list_new = arts_deque_priority_list_new,
    .list_get_deque = arts_deque_priority_list_get_deque,
    .list_delete = arts_deque_priority_list_delete,
    .new_deque = arts_deque_priority_new,
    .delete_deque = arts_deque_priority_delete,
    .push_front = arts_deque_priority_push_front,
    .pop_front = arts_deque_priority_pop_front,
    .pop_back = arts_deque_priority_pop_back,
    .empty = arts_deque_priority_empty,
    .clear = arts_deque_priority_clear,
    .size = arts_deque_priority_size,
};

static const struct arts_deque_ops_s *const deque_impls[] = {
    &simple_ops,
    &priority_ops,
};

#define NUM_DEQUE_IMPLS (sizeof(deque_impls) / sizeof(deque_impls[0]))

/* Active implementation — defaults to simple (index 0). */
static const struct arts_deque_ops_s *ops = &simple_ops;

/*--- Selection -------------------------------------------------------------*/

void arts_deque_select(unsigned int type) {
  if (type >= NUM_DEQUE_IMPLS) {
    ARTS_WARN("Invalid deque_type=%u, falling back to simple (0)", type);
    type = 0;
  }
  ops = deque_impls[type];
}

/*--- Generic wrappers (dispatch through vtable) ----------------------------*/

struct arts_deque_s *arts_deque_list_new(unsigned int list_size,
                                         unsigned int deque_size) {
  return ops->list_new(list_size, deque_size);
}

struct arts_deque_s *arts_deque_list_get_deque(struct arts_deque_s *deque_list,
                                               unsigned int position) {
  return ops->list_get_deque(deque_list, position);
}

void arts_deque_list_delete(void *deque_list) { ops->list_delete(deque_list); }

struct arts_deque_s *arts_deque_new(unsigned int size) {
  return ops->new_deque(size);
}

void arts_deque_delete(struct arts_deque_s *deque) { ops->delete_deque(deque); }

bool arts_deque_push_front(struct arts_deque_s *deque, void *item,
                           unsigned int priority) {
  return ops->push_front(deque, item, priority);
}

void *arts_deque_pop_front(struct arts_deque_s *deque) {
  return ops->pop_front(deque);
}

void *arts_deque_pop_back(struct arts_deque_s *deque) {
  return ops->pop_back(deque);
}

bool arts_deque_empty(struct arts_deque_s *deque) { return ops->empty(deque); }

void arts_deque_clear(struct arts_deque_s *deque) { ops->clear(deque); }

unsigned int arts_deque_size(struct arts_deque_s *deque) {
  return ops->size(deque);
}
