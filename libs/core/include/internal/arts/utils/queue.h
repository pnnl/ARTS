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

// Original copyright
// Copyright (c) 2013, Adam Morrison and Yehuda Afek.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in
//    the documentation and/or other materials provided with the
//    distribution.
//  * Neither the name of the Tel Aviv University nor the names of the
//    author of this software may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ARTS_UTILS_QUEUE_H
#define ARTS_UTILS_QUEUE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define OBJECT uint64_t
#define RING_POW (6)
#define RING_SIZE (1ull << RING_POW)
#define ALIGNMENT 8

typedef struct ring_node_s {
  volatile uint64_t val;
  volatile uint64_t idx;
  uint64_t pad[14];
} ring_node_t __attribute__((aligned(128)));

typedef struct ring_queue_s {
  volatile int64_t head __attribute__((aligned(128)));
  volatile int64_t tail __attribute__((aligned(128)));
  struct ring_queue_s *next __attribute__((aligned(128)));
  ring_node_t array[RING_SIZE];
} ring_queue_t __attribute__((aligned(128)));

typedef struct arts_queue_s {
  ring_queue_t *head;
  ring_queue_t *tail;
} arts_queue_t __attribute__((aligned(128)));

arts_queue_t *arts_new_queue();
void enqueue(OBJECT arg, arts_queue_t *queue);
OBJECT dequeue(arts_queue_t *queue);

int close_crq(ring_queue_t *rq, uint64_t t, int tries);
uint64_t node_index(uint64_t i) __attribute__((pure));
void fix_state(ring_queue_t *rq);
uint64_t set_unsafe(uint64_t i) __attribute__((pure));
int is_empty(uint64_t v) __attribute__((pure));
uint64_t node_unsafe(uint64_t i) __attribute__((pure));
int crq_is_closed(uint64_t t) __attribute__((pure));
void init_ring(ring_queue_t *r);
uint64_t tail_index(uint64_t t) __attribute__((pure));

#ifdef __cplusplus
}
#endif
#endif /* ARTS_UTILS_QUEUE_H */
