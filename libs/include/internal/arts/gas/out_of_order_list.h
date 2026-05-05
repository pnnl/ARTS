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
#ifndef ARTS_GAS_OUT_OF_ORDER_LIST_H
#define ARTS_GAS_OUT_OF_ORDER_LIST_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

/* Vyukov MPSC queue (intrusive node, embedded permanent stub).
 *
 * Producers: lock-free atomic_xchg on tail + store_release on prev->next.
 * Consumer:  exactly one — single thread pops via plain head advance.
 *
 * Replaces the previous Treiber-stack-with-reverse design.  Pure FIFO,
 * no CAS loop on producer side, no reverse step on the consumer side.
 *
 * Init contract: arts_oo_list_s contains an embedded sentinel `stub`.
 * The struct is NOT safe to use after zero-initialization — callers
 * MUST invoke arts_oo_list_init() before push/drain/drop_all.  Sites:
 *   - route_table.c::arts_route_table_search_for_empty (slot first claim)
 *   - tests that allocate arts_oo_list_s on the stack/static storage
 */

#ifdef __cplusplus
struct arts_oo_node_s {
  struct arts_oo_node_s *next;
  void *data;
};
struct arts_oo_list_s {
  struct arts_oo_node_s *head;
  struct arts_oo_node_s *tail;
  struct arts_oo_node_s stub;
};
#else
#include <stdatomic.h>
struct arts_oo_node_s {
  _Atomic(struct arts_oo_node_s *) next;
  void *data;
};
struct arts_oo_list_s {
  _Atomic(struct arts_oo_node_s *) head;
  _Atomic(struct arts_oo_node_s *) tail;
  struct arts_oo_node_s stub;
};
#endif

/* push always succeeds in MPSC (no DRAIN_HAPPENED race window).  The
 * enum is retained as a single OK value so existing call sites keep
 * compiling without churn; it can be folded into a void return in a
 * follow-up cleanup. */
typedef enum {
  OO_PUSH_OK,
} oo_push_result_t;

/* MUST be called once on every arts_oo_list_s before first use. */
void arts_oo_list_init(struct arts_oo_list_s *list);

oo_push_result_t arts_oo_list_push(struct arts_oo_list_s *list, void *data);

void arts_oo_list_drain(struct arts_oo_list_s *list,
                        void (*callback)(void *data, void *ctx), void *ctx);

void arts_oo_list_drop_all(struct arts_oo_list_s *list);

#ifdef __cplusplus
}
#endif

#endif
