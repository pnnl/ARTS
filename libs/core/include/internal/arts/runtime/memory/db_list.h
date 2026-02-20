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
#ifndef ARTS_RUNTIME_MEMORY_DBLIST_H
#define ARTS_RUNTIME_MEMORY_DBLIST_H
#ifdef __cplusplus
extern "C" {
#endif

#include "arts/runtime/rt.h"

#define DBSPERELEMENT 8

struct arts_db_element_s {
  struct arts_db_element_s *next;
  unsigned int array[DBSPERELEMENT];
};

struct arts_local_delayed_edt_s {
  struct arts_local_delayed_edt_s *next;
  struct arts_edt_s *edt[DBSPERELEMENT];
  unsigned int slot[DBSPERELEMENT];
  arts_type_t mode[DBSPERELEMENT];
};

struct arts_db_frontier_s {
  struct arts_db_element_s list;
  unsigned int position;
  struct arts_db_frontier_s *next;
  volatile unsigned int lock;

  /*
   * Remote writer slot — at most one remote writer per frontier.
   * Set when a remote node requests WRITE access (write && !local).
   * Signaled by arts_signal_frontier_local/remote when frontier progresses.
   */
  unsigned int exNode;
  arts_guid_t exEdtGuid;
  struct arts_edt_s *exEdt;
  unsigned int exSlot;
  arts_type_t exMode;

  /*
   * This is dumb, but we need somewhere to store requests
   * that are from the guid owner but cannot be satisfied
   * because of the memory model
   */
  unsigned int localPosition;
  struct arts_local_delayed_edt_s localDelayed;
};

struct arts_db_list_s {
  struct arts_db_frontier_s *head;
  struct arts_db_frontier_s *tail;
  volatile unsigned int reader;
  volatile unsigned int writer;
};

struct arts_db_frontier_iterator_s {
  struct arts_db_frontier_s *frontier;
  unsigned int currentIndex;
  struct arts_db_element_s *currentElement;
};

struct arts_db_list_s *arts_new_db_list();
unsigned int arts_current_frontier_size(struct arts_db_list_s *db_list);
struct arts_db_frontier_iterator_s *
arts_db_frontier_iter_create(struct arts_db_frontier_s *frontier);
unsigned int
arts_db_frontier_iter_size(struct arts_db_frontier_iterator_s *iter);
bool arts_db_frontier_iter_next(struct arts_db_frontier_iterator_s *iter,
                                unsigned int *next);
bool arts_db_frontier_iter_has_next(struct arts_db_frontier_iterator_s *iter);
void arts_db_frontier_iter_delete(struct arts_db_frontier_iterator_s *iter);
void arts_progress_frontier(struct arts_db_s *db, unsigned int rank);
struct arts_db_frontier_iterator_s *
arts_progress_and_get_frontier(struct arts_db_list_s *db_list);
bool arts_push_db_to_list(struct arts_db_list_s *db_list, unsigned int data,
                          bool write, bool local, bool bypass,
                          struct arts_edt_s *edt, arts_guid_t edt_guid,
                          unsigned int slot, arts_type_t mode, bool *on_head);
struct arts_db_frontier_iterator_s *
arts_close_frontier(struct arts_db_list_s *db_list);
#ifdef __cplusplus
}
#endif

#endif /* ARTSDBLIST_H */
