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
#ifndef ARTS_UTILS_LINKLIST_H
#define ARTS_UTILS_LINKLIST_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* Lock-free Vyukov MPSC queue (multi-producer push_back, single-consumer
 * pop_front).  The transport outbound queues use one per (rank,port) slot —
 * workers append; a dedicated sender thread (or, at shutdown, the flush path
 * once senders have stopped) drains.  pop_front returns NULL while a producer
 * is mid-link (claimed the tail but not yet linked it); the drain loops
 * re-poll, so no message is lost.  The C++/nvcc layout-mirror drops _Atomic
 * (the helpers are C-only). */
struct arts_link_list_s;
struct arts_link_list_item_s {
#ifdef __cplusplus
  struct arts_link_list_item_s *next;
#else
  _Atomic(struct arts_link_list_item_s *) next;
#endif
};

struct arts_link_list_s {
#ifdef __cplusplus
  struct arts_link_list_item_s *head;
#else
  _Atomic(struct arts_link_list_item_s *) head; /* producers xchg-append */
#endif
  struct arts_link_list_item_s *tail; /* single consumer reads/advances */
  struct arts_link_list_item_s stub;  /* embedded sentinel */
};

void arts_link_list_new(struct arts_link_list_s *list);
struct arts_link_list_s *arts_link_list_group_new(unsigned int list_size);
struct arts_link_list_s *arts_link_list_get(struct arts_link_list_s *link_list,
                                            unsigned int position);
uint8_t arts_link_list_is_empty(struct arts_link_list_s *link_list);
void arts_link_list_delete(void *link_list);
void arts_link_list_push_back(struct arts_link_list_s *list, void *item);
void *arts_link_list_pop_front(struct arts_link_list_s *list, void **free_pos);
void arts_link_list_delete_item(void *to_delete);
void *arts_link_list_new_item(unsigned int size);

#ifdef __cplusplus
}
#endif
#endif
