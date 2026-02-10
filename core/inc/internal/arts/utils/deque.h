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
#ifndef ARTS_UTILS_DEQUE_H
#define ARTS_UTILS_DEQUE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#define STEALSIZE 1024

struct arts_deque;
struct arts_deque *arts_deque_list_new(unsigned int list_size,
                                   unsigned int deque_size);
struct arts_deque *arts_deque_list_get_deque(struct arts_deque *deque_list,
                                        unsigned int position);
void arts_deque_list_delete(void *deque_list);
struct arts_deque *arts_deque_new(unsigned int size);
void arts_deque_delete(struct arts_deque *deque);
bool arts_deque_push_front(struct arts_deque *deque, void *item,
                        unsigned int priority);
void *arts_deque_pop_front(struct arts_deque *deque);
void *arts_deque_pop_back(struct arts_deque *deque);

bool arts_deque_empty(struct arts_deque *deque);
void arts_deque_clear(struct arts_deque *deque);
unsigned int arts_deque_size(struct arts_deque *deque);

#ifdef __cplusplus
}
#endif
#endif
