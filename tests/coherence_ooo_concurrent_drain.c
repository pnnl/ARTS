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

/* B.7: Concurrent OO push + drain race stress
 *
 * N threads concurrently push to ooList while one thread drains.
 * Vyukov MPSC: push always succeeds (no DRAIN_HAPPENED), drain runs
 * single-consumer.  Verifies push count == drain count, no leaks.
 *
 * Spec section 6 B.7
 */
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#include "arts.h"
#include "arts/gas/out_of_order_list.h"

#define N_THREADS 32
#define N_ENTRIES_PER_THREAD 1000
#define N_ITERATIONS 100

static struct arts_oo_list_s g_list;
static _Atomic(int) g_pushed = 0;
static _Atomic(int) g_drained = 0;

static void drain_callback(void *data, void *ctx) {
  (void)ctx;
  int *val = (int *)data;
  free(val);
  atomic_fetch_add(&g_drained, 1);
}

static void *push_thread(void *arg) {
  (void)arg;
  for (int i = 0; i < N_ENTRIES_PER_THREAD; i++) {
    int *payload = (int *)malloc(sizeof(int));
    *payload = i;
    (void)arts_oo_list_push(&g_list, payload);
    atomic_fetch_add(&g_pushed, 1);
  }
  return NULL;
}

static void *drain_thread(void *arg) {
  (void)arg;
  for (int i = 0; i < 10; i++) {
    arts_oo_list_drain(&g_list, drain_callback, NULL);
  }
  return NULL;
}

int main(void) {
  for (int iter = 0; iter < N_ITERATIONS; iter++) {
    atomic_store(&g_pushed, 0);
    atomic_store(&g_drained, 0);
    arts_oo_list_init(&g_list);

    pthread_t pushers[N_THREADS];
    pthread_t drainer;
    for (int i = 0; i < N_THREADS; i++) {
      pthread_create(&pushers[i], NULL, push_thread, NULL);
    }
    pthread_create(&drainer, NULL, drain_thread, NULL);

    for (int i = 0; i < N_THREADS; i++) {
      pthread_join(pushers[i], NULL);
    }
    pthread_join(drainer, NULL);

    arts_oo_list_drain(&g_list, drain_callback, NULL);

    int pushed = atomic_load(&g_pushed);
    int drained = atomic_load(&g_drained);
    int total = N_THREADS * N_ENTRIES_PER_THREAD;

    if (pushed != total) {
      fprintf(stderr, "iter %d: pushed=%d != total=%d\n", iter, pushed, total);
      return 1;
    }
    if (drained != pushed) {
      fprintf(stderr, "iter %d: drained=%d != pushed=%d (LEAK or LOST)\n", iter,
              pushed, drained);
      return 1;
    }
  }
  printf("PASS: %d iterations, no leak/lost entries\n", N_ITERATIONS);
  return 0;
}
